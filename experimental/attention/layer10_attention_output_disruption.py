#!/usr/bin/env python3
"""
Layer 10 Attention Output Disruption - Causal Test

Based on the breakthrough that attention OUTPUT (not weights) is causal,
this test disrupts the attention output to CAUSE the bug in correct format.

Hypothesis: If we make Layer 10's attention output in the correct format
look more like the buggy format's output, it should cause the bug.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

class Layer10AttentionDisruptor:
    """Disrupt attention output to cause the bug in correct format"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print("="*70)
        print("LAYER 10 ATTENTION OUTPUT DISRUPTION TEST")
        print("="*70)
        print(f"\nLoading model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Storage for attention outputs
        self.stored_outputs = {}
        self.disruption_active = False
        self.disruption_mode = None
        self.disruption_strength = 0.0
        
    def extract_attention_output_hook(self, storage_key: str):
        """Hook to extract attention output"""
        def hook(module, input, output):
            # output[0] is the attention output
            if isinstance(output, tuple) and len(output) > 0:
                self.stored_outputs[storage_key] = output[0].clone().detach()
        return hook
    
    def disrupt_begin_attention_hook(self, strength: float = 0.5):
        """
        Hook to disrupt attention output in a way that reduces BEGIN influence.
        This simulates what happens in the buggy Q&A format.
        """
        def hook(module, input, output):
            if not self.disruption_active:
                return output
            
            if isinstance(output, tuple) and len(output) > 0:
                attn_output = output[0].clone()
                original_dtype = attn_output.dtype
                
                # Method 1: Reduce contribution from early positions (BEGIN area)
                # This simulates reduced BEGIN anchoring
                batch_size, seq_len, hidden_dim = attn_output.shape
                
                # Create disruption that reduces early token influence
                if self.disruption_mode == "reduce_begin":
                    # Reduce the magnitude of features that would come from BEGIN attention
                    # Positions 0-2 are typically BEGIN-related
                    for pos in range(min(3, seq_len)):
                        reduction = strength * attn_output[:, pos:pos+1, :].mean(dim=1, keepdim=True)
                        attn_output = attn_output - reduction.to(original_dtype)
                
                elif self.disruption_mode == "scramble_begin":
                    # Add noise specifically to disrupt BEGIN-related processing
                    noise = torch.randn_like(attn_output).to(original_dtype) * strength * 0.1
                    # Apply more noise to positions that would process BEGIN info
                    position_weights = torch.exp(-torch.arange(seq_len, device=self.device, dtype=torch.float32) / 5)
                    position_weights = position_weights.view(1, -1, 1).to(original_dtype)
                    attn_output = attn_output + (noise * position_weights)
                
                elif self.disruption_mode == "shift_pattern":
                    # Shift the attention output pattern to simulate Q&A format
                    # This is like adding phantom "Q:" tokens
                    if seq_len > 2:
                        shifted = torch.cat([
                            attn_output[:, -2:, :],  # Move last tokens to front
                            attn_output[:, :-2, :]   # Shift everything else
                        ], dim=1)
                        attn_output = ((1 - strength) * attn_output + strength * shifted).to(original_dtype)
                
                elif self.disruption_mode == "inject_buggy":
                    # Directly blend with buggy format output if available
                    if "buggy_output" in self.stored_outputs:
                        buggy = self.stored_outputs["buggy_output"].to(original_dtype)
                        # Align dimensions if needed
                        min_len = min(attn_output.shape[1], buggy.shape[1])
                        attn_output[:, :min_len, :] = ((1 - strength) * attn_output[:, :min_len, :] + 
                                                       strength * buggy[:, :min_len, :]).to(original_dtype)
                
                # Ensure output maintains original dtype
                attn_output = attn_output.to(original_dtype)
                
                # Return modified output
                return (attn_output,) + output[1:]
            
            return output
        
        return hook
    
    def extract_baseline_outputs(self):
        """Extract attention outputs from both correct and buggy formats for reference"""
        print("\nExtracting baseline attention outputs...")
        
        # Correct format
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        inputs = self.tokenizer(correct_prompt, return_tensors="pt").to(self.device)
        
        # Add extraction hook
        handle = self.model.model.layers[10].self_attn.register_forward_hook(
            self.extract_attention_output_hook("correct_output")
        )
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        handle.remove()
        print(f"✓ Extracted correct format output: {self.stored_outputs['correct_output'].shape}")
        
        # Buggy format
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        inputs = self.tokenizer(buggy_prompt, return_tensors="pt").to(self.device)
        
        handle = self.model.model.layers[10].self_attn.register_forward_hook(
            self.extract_attention_output_hook("buggy_output")
        )
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        handle.remove()
        print(f"✓ Extracted buggy format output: {self.stored_outputs['buggy_output'].shape}")
        
        # Analyze difference
        correct_out = self.stored_outputs["correct_output"]
        buggy_out = self.stored_outputs["buggy_output"]
        
        # Compare magnitudes and patterns
        correct_norm = correct_out.norm(dim=-1).mean().item()
        buggy_norm = buggy_out.norm(dim=-1).mean().item()
        
        print(f"\nOutput statistics:")
        print(f"  Correct format norm: {correct_norm:.4f}")
        print(f"  Buggy format norm: {buggy_norm:.4f}")
        print(f"  Ratio: {buggy_norm/correct_norm:.4f}")
    
    def test_with_disruption(self, prompt: str, mode: str, strength: float) -> Dict:
        """Test generation with disrupted attention output"""
        self.disruption_mode = mode
        self.disruption_strength = strength
        self.disruption_active = True
        
        # Add disruption hook
        handle = self.model.model.layers[10].self_attn.register_forward_hook(
            self.disrupt_begin_attention_hook(strength)
        )
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        # Clean up
        handle.remove()
        self.disruption_active = False
        
        # Analyze response
        response_lower = response.lower()
        is_correct = "9.8 is bigger" in response_lower or "9.8 is larger" in response_lower
        shows_bug = "9.11 is bigger" in response_lower or "9.11 is larger" in response_lower
        is_gibberish = not any(c.isalnum() for c in response[:20]) or response.count("!") > 10
        
        return {
            "response": response,
            "is_correct": is_correct,
            "shows_bug": shows_bug,
            "is_gibberish": is_gibberish
        }
    
    def run_causal_test(self):
        """Run the complete causal test"""
        print("\n" + "="*70)
        print("CAUSAL TEST: DISRUPTING ATTENTION OUTPUT")
        print("="*70)
        
        # First extract baseline outputs
        self.extract_baseline_outputs()
        
        # Test prompt (correct format)
        test_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        
        # Test different disruption modes and strengths
        modes = ["reduce_begin", "scramble_begin", "shift_pattern", "inject_buggy"]
        strengths = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        results = {}
        
        for mode in modes:
            print(f"\n{'='*50}")
            print(f"Testing disruption mode: {mode}")
            print(f"{'='*50}")
            
            mode_results = []
            
            for strength in strengths:
                if strength == 0:
                    print(f"\n--- Baseline (no disruption) ---")
                    # Test without disruption
                    inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = generated[len(test_prompt):].strip()
                    
                    response_lower = response.lower()
                    result = {
                        "response": response,
                        "is_correct": "9.8 is bigger" in response_lower,
                        "shows_bug": "9.11 is bigger" in response_lower,
                        "is_gibberish": False,
                        "strength": 0
                    }
                else:
                    print(f"\n--- Disruption strength: {strength:.1%} ---")
                    result = self.test_with_disruption(test_prompt, mode, strength)
                    result["strength"] = strength
                
                mode_results.append(result)
                
                # Display result
                if result["is_gibberish"]:
                    status = "💥 GIBBERISH"
                elif result["shows_bug"]:
                    status = "🐛 BUG INDUCED!"
                elif result["is_correct"]:
                    status = "✅ Still correct"
                else:
                    status = "❓ Unclear"
                
                print(f"Result: {status}")
                print(f"Response: {result['response'][:80]}...")
            
            results[mode] = mode_results
        
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze which disruption successfully causes the bug"""
        analysis = {
            "successful_modes": [],
            "critical_strengths": {},
            "causal_evidence": False
        }
        
        for mode, mode_results in results.items():
            # Check if this mode successfully induced the bug
            baseline_correct = mode_results[0]["is_correct"]  # strength=0
            
            for result in mode_results[1:]:  # Skip baseline
                if baseline_correct and result["shows_bug"]:
                    # Found a successful bug induction!
                    if mode not in analysis["successful_modes"]:
                        analysis["successful_modes"].append(mode)
                        analysis["critical_strengths"][mode] = result["strength"]
                    analysis["causal_evidence"] = True
                    break
        
        return analysis
    
    def visualize_results(self, results: Dict, analysis: Dict):
        """Visualize the causal test results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        modes = list(results.keys())
        
        for idx, mode in enumerate(modes):
            ax = axes[idx // 2, idx % 2]
            
            mode_results = results[mode]
            strengths = [r["strength"] for r in mode_results]
            correct = [1 if r["is_correct"] else 0 for r in mode_results]
            bug = [1 if r["shows_bug"] else 0 for r in mode_results]
            gibberish = [1 if r["is_gibberish"] else 0 for r in mode_results]
            
            ax.plot(strengths, correct, 'g-o', label='Correct', linewidth=2)
            ax.plot(strengths, bug, 'r-s', label='Bug', linewidth=2)
            ax.plot(strengths, gibberish, 'k--^', label='Gibberish', linewidth=2)
            
            ax.set_xlabel('Disruption Strength')
            ax.set_ylabel('Occurrence')
            ax.set_title(f'Mode: {mode}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            
            # Mark critical strength if found
            if mode in analysis["critical_strengths"]:
                critical = analysis["critical_strengths"][mode]
                ax.axvline(x=critical, color='orange', linestyle=':', 
                          label=f'Critical: {critical:.1%}')
        
        plt.suptitle('Layer 10 Attention Output Disruption: Causal Test Results', fontsize=14)
        plt.tight_layout()
        plt.savefig('layer10_attention_disruption_results.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to layer10_attention_disruption_results.png")
    
    def generate_report(self, results: Dict, analysis: Dict) -> str:
        """Generate detailed report"""
        lines = []
        lines.append("="*70)
        lines.append("LAYER 10 ATTENTION OUTPUT DISRUPTION - CAUSAL REPORT")
        lines.append("="*70)
        
        lines.append("\n## HYPOTHESIS")
        lines.append("Disrupting Layer 10's attention output to reduce BEGIN influence")
        lines.append("should CAUSE the decimal comparison bug in the correct format.")
        
        lines.append("\n## RESULTS")
        
        if analysis["causal_evidence"]:
            lines.append("\n✅ CAUSAL RELATIONSHIP CONFIRMED!")
            lines.append(f"Successful disruption modes: {', '.join(analysis['successful_modes'])}")
            
            for mode in analysis["successful_modes"]:
                critical = analysis["critical_strengths"][mode]
                lines.append(f"  • {mode}: Bug induced at {critical:.0%} disruption")
        else:
            lines.append("\n❌ No clear causal relationship found")
            lines.append("Disruption did not successfully induce the bug")
        
        lines.append("\n## DETAILED RESULTS BY MODE")
        
        for mode, mode_results in results.items():
            lines.append(f"\n### {mode}")
            for r in mode_results:
                strength = r["strength"]
                if r["is_gibberish"]:
                    status = "GIBBERISH"
                elif r["shows_bug"]:
                    status = "BUG"
                elif r["is_correct"]:
                    status = "CORRECT"
                else:
                    status = "UNCLEAR"
                lines.append(f"  {strength:3.0%}: {status}")
        
        lines.append("\n## IMPLICATIONS")
        
        if analysis["causal_evidence"]:
            lines.append("1. Layer 10 attention output IS causally responsible for the bug")
            lines.append("2. Reducing BEGIN influence in attention output causes the bug")
            lines.append("3. The attention output difference is the mechanism")
            lines.append("4. This validates the attention output patching results")
        else:
            lines.append("1. Simple disruption may not be sufficient")
            lines.append("2. The mechanism may be more complex than BEGIN reduction")
            lines.append("3. Further investigation needed")
        
        return "\n".join(lines)


def main():
    """Run the causal test"""
    print("\n" + "="*70)
    print("STARTING LAYER 10 ATTENTION OUTPUT CAUSAL TEST")
    print("="*70)
    
    # Initialize disruptor
    disruptor = Layer10AttentionDisruptor()
    
    # Run causal test
    results = disruptor.run_causal_test()
    
    # Analyze results
    analysis = disruptor.analyze_results(results)
    
    # Generate report
    report = disruptor.generate_report(results, analysis)
    print("\n" + report)
    
    # Visualize
    disruptor.visualize_results(results, analysis)
    
    # Save results
    with open('layer10_attention_disruption_results.json', 'w') as f:
        # Convert to serializable format
        save_results = {}
        for mode, mode_results in results.items():
            save_results[mode] = []
            for r in mode_results:
                save_results[mode].append({
                    'strength': r['strength'],
                    'is_correct': r['is_correct'],
                    'shows_bug': r['shows_bug'],
                    'is_gibberish': r['is_gibberish'],
                    'response_preview': r['response'][:100]
                })
        
        json.dump({
            'results': save_results,
            'analysis': analysis
        }, f, indent=2)
    
    print("\n✓ Results saved to layer10_attention_disruption_results.json")
    
    with open('layer10_attention_disruption_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Report saved to layer10_attention_disruption_report.txt")
    
    if analysis["causal_evidence"]:
        print("\n" + "="*70)
        print("🎯 SUCCESS: CAUSAL RELATIONSHIP CONFIRMED!")
        print("Disrupting Layer 10 attention output CAUSES the bug!")
        print("="*70)


if __name__ == "__main__":
    main()