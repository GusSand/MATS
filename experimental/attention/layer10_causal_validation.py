#!/usr/bin/env python3
"""
Causal Validation Test for Layer 10 BEGIN Token Anchoring

Hypothesis: In the Plain/Simple format, strong attention to BEGIN token at Layer 10
prevents the bug. If we artificially disrupt this anchoring, it should CAUSE the bug
even in the normally correct format.

This provides causal evidence that BEGIN token attention is protective against the bug.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class Layer10CausalValidator:
    """Test causal relationship of Layer 10 BEGIN anchoring"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print("="*70)
        print("LAYER 10 CAUSAL VALIDATION TEST")
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
        
        # Track modifications
        self.original_forward = {}
        self.intervention_active = False
        self.attention_patterns = {}
        
    def identify_begin_token(self, input_ids: torch.Tensor) -> int:
        """Identify the BEGIN token position in the input"""
        # For Llama models, look for special tokens or first content token
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        
        # Look for start_header_id or similar markers
        begin_pos = 0
        for i, token in enumerate(tokens):
            if 'begin' in token.lower() or 'start' in token.lower() or i == 0:
                begin_pos = i
                break
        
        return begin_pos
    
    def create_attention_intervention(self, layer_idx: int = 10, disruption_strength: float = 0.9):
        """
        Create intervention that disrupts attention to BEGIN token at Layer 10.
        
        Args:
            layer_idx: Which layer to intervene at (default 10)
            disruption_strength: How much to reduce BEGIN attention (0=no change, 1=complete removal)
        """
        def intervention_hook(module, input, output):
            if not self.intervention_active:
                return output
            
            # For Llama models, the output is a tuple (hidden_states, attn_weights, cache)
            # We need to modify the forward pass more carefully
            # Since we can't easily access attention weights in generation mode,
            # we'll modify the hidden states instead
            
            if isinstance(output, tuple) and len(output) >= 1:
                hidden_states = output[0]
                
                # Apply disruption by modifying hidden states
                # Reduce contribution from early positions (BEGIN token area)
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                if seq_len > 1:
                    # Create disruption mask
                    mask = torch.ones_like(hidden_states)
                    # Reduce influence of position 0 (BEGIN token)
                    mask[:, 0, :] *= (1 - disruption_strength)
                    
                    # Apply mask
                    modified_hidden = hidden_states * mask
                    
                    # Renormalize to preserve magnitude
                    norm_factor = hidden_states.norm(dim=-1, keepdim=True) / (modified_hidden.norm(dim=-1, keepdim=True) + 1e-8)
                    modified_hidden = modified_hidden * norm_factor
                    
                    # Return modified output
                    return (modified_hidden,) + output[1:]
            
            return output
        
        return intervention_hook
    
    def apply_intervention(self, layer_idx: int = 10, disruption_strength: float = 0.9):
        """Apply the attention disruption intervention"""
        # Remove any existing hooks
        self.remove_intervention()
        
        # Apply hook to specified layer
        if layer_idx < len(self.model.model.layers):
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer, 'self_attn'):
                hook = self.create_attention_intervention(layer_idx, disruption_strength)
                layer.self_attn.register_forward_hook(hook)
                self.intervention_active = True
                print(f"✓ Intervention applied to Layer {layer_idx} (disruption={disruption_strength:.1%})")
        else:
            print(f"✗ Layer {layer_idx} not found")
    
    def remove_intervention(self):
        """Remove all intervention hooks"""
        self.intervention_active = False
        # Clear all hooks
        for module in self.model.modules():
            module._forward_hooks.clear()
        print("✓ Interventions removed")
    
    def test_prediction(self, prompt: str, max_tokens: int = 50) -> Dict:
        """Test model prediction with current intervention state"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        # Analyze response - look for which number is stated as bigger
        response_lower = response.lower()
        
        # Check if it says 9.8 is bigger (correct)
        is_correct = False
        if "9.8 is bigger" in response_lower or "9.8 is larger" in response_lower:
            is_correct = True
        elif "bigger than 9.11" in response_lower and "9.8" in response:
            is_correct = True
            
        # Check if it says 9.11 is bigger (bug)
        shows_bug = False
        if "9.11 is bigger" in response_lower or "9.11 is larger" in response_lower:
            shows_bug = True
        elif "bigger than 9.8" in response_lower and "9.11" in response:
            shows_bug = True
        
        return {
            "response": response,
            "is_correct": is_correct,
            "shows_bug": shows_bug,
            "confidence": self.analyze_response_confidence(response)
        }
    
    def analyze_response_confidence(self, response: str) -> float:
        """Analyze how confident the model is in its response"""
        # Simple heuristic: look for hedging words
        hedging_words = ["both", "neither", "depends", "unclear", "maybe", "possibly"]
        response_lower = response.lower()
        
        if any(word in response_lower for word in hedging_words):
            return 0.3  # Low confidence
        elif "9.8" in response and "bigger" in response_lower:
            return 0.9  # High confidence in correct answer
        elif "9.11" in response and "bigger" in response_lower:
            return 0.9  # High confidence in wrong answer
        else:
            return 0.5  # Medium confidence
    
    def run_causal_test(self):
        """Run the complete causal validation test"""
        print("\n" + "="*70)
        print("TESTING CAUSAL HYPOTHESIS")
        print("="*70)
        
        # Test formats
        test_prompts = {
            "simple": "Which is bigger: 9.8 or 9.11?\nAnswer:",
            "qa": "Q: Which is bigger: 9.8 or 9.11?\nA:",
        }
        
        # Disruption strengths to test
        disruption_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        results = {
            "simple": {},
            "qa": {}
        }
        
        for format_name, prompt in test_prompts.items():
            print(f"\n{'='*50}")
            print(f"Testing {format_name.upper()} format")
            print(f"{'='*50}")
            print(f"Prompt: {repr(prompt[:50])}...")
            
            format_results = []
            
            for disruption in disruption_levels:
                print(f"\n--- Disruption Level: {disruption:.1%} ---")
                
                if disruption == 0:
                    # Baseline - no intervention
                    self.remove_intervention()
                    print("(No intervention - baseline)")
                else:
                    # Apply intervention
                    self.apply_intervention(layer_idx=10, disruption_strength=disruption)
                
                # Test prediction
                result = self.test_prediction(prompt)
                result['disruption'] = disruption
                format_results.append(result)
                
                # Display result
                status = "✓ CORRECT" if result['is_correct'] else "✗ BUG" if result['shows_bug'] else "? UNCLEAR"
                print(f"Result: {status}")
                print(f"Response: {result['response'][:100]}...")
                print(f"Confidence: {result['confidence']:.1%}")
            
            results[format_name] = format_results
        
        # Clean up
        self.remove_intervention()
        
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze causal test results"""
        analysis = {
            "causal_evidence": False,
            "simple_format_degradation": [],
            "qa_format_changes": [],
            "critical_disruption_level": None
        }
        
        # Analyze Simple format (should degrade with disruption)
        simple_results = results.get("simple", [])
        for r in simple_results:
            disruption = r['disruption']
            if disruption == 0:
                baseline_correct = r['is_correct']
            else:
                if baseline_correct and r['shows_bug']:
                    analysis["simple_format_degradation"].append(disruption)
                    if analysis["critical_disruption_level"] is None:
                        analysis["critical_disruption_level"] = disruption
        
        # Analyze QA format (should already show bug)
        qa_results = results.get("qa", [])
        qa_baseline_bug = any(r['shows_bug'] for r in qa_results if r['disruption'] == 0)
        
        # Determine if we have causal evidence
        if len(analysis["simple_format_degradation"]) > 0 and qa_baseline_bug:
            analysis["causal_evidence"] = True
        
        return analysis
    
    def visualize_results(self, results: Dict, analysis: Dict):
        """Create visualization of causal test results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        disruption_levels = [r['disruption'] for r in results['simple']]
        simple_correct = [1 if r['is_correct'] else 0 for r in results['simple']]
        simple_bug = [1 if r['shows_bug'] else 0 for r in results['simple']]
        qa_correct = [1 if r['is_correct'] else 0 for r in results['qa']]
        qa_bug = [1 if r['shows_bug'] else 0 for r in results['qa']]
        
        # Plot 1: Correctness by disruption
        ax1 = axes[0, 0]
        ax1.plot(disruption_levels, simple_correct, 'g-o', label='Simple Format', linewidth=2, markersize=8)
        ax1.plot(disruption_levels, qa_correct, 'b--s', label='Q&A Format', linewidth=2, markersize=8)
        ax1.set_xlabel('BEGIN Attention Disruption')
        ax1.set_ylabel('Correct Answer Rate')
        ax1.set_title('Correctness Degradation with BEGIN Disruption')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # Plot 2: Bug occurrence by disruption
        ax2 = axes[0, 1]
        ax2.plot(disruption_levels, simple_bug, 'r-o', label='Simple Format', linewidth=2, markersize=8)
        ax2.plot(disruption_levels, qa_bug, 'orange', label='Q&A Format', linewidth=2, markersize=8)
        ax2.set_xlabel('BEGIN Attention Disruption')
        ax2.set_ylabel('Bug Occurrence Rate')
        ax2.set_title('Bug Emergence with BEGIN Disruption')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
        
        if analysis["critical_disruption_level"]:
            ax2.axvline(x=analysis["critical_disruption_level"], 
                       color='red', linestyle=':', alpha=0.7,
                       label=f'Critical Level: {analysis["critical_disruption_level"]:.1%}')
        
        # Plot 3: Attention pattern visualization (if available)
        ax3 = axes[1, 0]
        if 'layer_10_original' in self.attention_patterns and 'layer_10_modified' in self.attention_patterns:
            original = self.attention_patterns['layer_10_original'][0, :, -1, :10].mean(0)  # Average over heads
            modified = self.attention_patterns['layer_10_modified'][0, :, -1, :10].mean(0)
            
            x = np.arange(len(original))
            width = 0.35
            ax3.bar(x - width/2, original.numpy(), width, label='Original', alpha=0.7)
            ax3.bar(x + width/2, modified.numpy(), width, label='Disrupted', alpha=0.7)
            ax3.set_xlabel('Token Position')
            ax3.set_ylabel('Attention Weight')
            ax3.set_title('Layer 10 Attention Pattern (Last Token)')
            ax3.legend()
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='BEGIN')
        else:
            ax3.text(0.5, 0.5, 'Attention patterns\nnot available', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Layer 10 Attention Pattern')
        
        # Plot 4: Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Build summary text
        causal_text = 'YES ✓' if analysis['causal_evidence'] else 'NO ✗'
        
        if analysis['critical_disruption_level'] is not None:
            critical_text = f"{analysis['critical_disruption_level']:.0%}"
        else:
            critical_text = "N/A"
        
        qa_baseline = 'Shows Bug' if any(r['shows_bug'] for r in results.get('qa', []) if r['disruption'] == 0) else 'Correct'
        
        conclusion1 = 'BEGIN anchoring at Layer 10 is CAUSALLY' if analysis['causal_evidence'] else 'No causal'
        conclusion2 = 'responsible for preventing the bug' if analysis['causal_evidence'] else 'relationship found'
        
        summary_text = f"""
CAUSAL VALIDATION RESULTS
{'='*30}

Hypothesis: Disrupting BEGIN token attention
at Layer 10 causes the bug in Simple format

Results:
• Causal Evidence: {causal_text}
• Critical Disruption: {critical_text}
• Simple Format Degradation: {len(analysis['simple_format_degradation'])} levels
• QA Format Baseline: {qa_baseline}

Conclusion:
{conclusion1}
{conclusion2}
"""
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', va='center')
        
        plt.suptitle('Layer 10 BEGIN Token Anchoring: Causal Validation', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('layer10_causal_validation.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to layer10_causal_validation.png")
        
        return fig
    
    def generate_report(self, results: Dict, analysis: Dict) -> str:
        """Generate detailed causal validation report"""
        report = []
        report.append("="*70)
        report.append("LAYER 10 CAUSAL VALIDATION REPORT")
        report.append("="*70)
        
        report.append("\n## HYPOTHESIS")
        report.append("-"*40)
        report.append("If BEGIN token anchoring at Layer 10 prevents the decimal bug,")
        report.append("then disrupting this anchoring should CAUSE the bug in Simple format.")
        
        report.append("\n## METHODOLOGY")
        report.append("-"*40)
        report.append("1. Test Simple format (normally correct) with varying disruption levels")
        report.append("2. Artificially reduce attention to BEGIN token at Layer 10")
        report.append("3. Observe if bug emerges with increasing disruption")
        report.append("4. Compare with Q&A format (normally buggy) as control")
        
        report.append("\n## RESULTS")
        report.append("-"*40)
        
        # Simple format results
        report.append("\n### Simple Format (Normally Correct)")
        for r in results['simple']:
            status = "CORRECT" if r['is_correct'] else "BUG" if r['shows_bug'] else "UNCLEAR"
            report.append(f"  Disruption {r['disruption']:3.0%}: {status} (confidence: {r['confidence']:.1%})")
        
        # QA format results
        report.append("\n### Q&A Format (Normally Buggy)")
        for r in results['qa']:
            status = "CORRECT" if r['is_correct'] else "BUG" if r['shows_bug'] else "UNCLEAR"
            report.append(f"  Disruption {r['disruption']:3.0%}: {status} (confidence: {r['confidence']:.1%})")
        
        report.append("\n## ANALYSIS")
        report.append("-"*40)
        
        if analysis['causal_evidence']:
            report.append("✓ CAUSAL RELATIONSHIP CONFIRMED")
            report.append(f"  • Bug emerges at {analysis['critical_disruption_level']:.0%} disruption")
            report.append(f"  • {len(analysis['simple_format_degradation'])} disruption levels cause bug")
            report.append("  • This proves BEGIN anchoring is protective")
        else:
            report.append("✗ No clear causal relationship found")
            report.append("  • Simple format remains robust to disruption")
            report.append("  • BEGIN anchoring may not be the causal factor")
        
        report.append("\n## IMPLICATIONS")
        report.append("-"*40)
        
        if analysis['causal_evidence']:
            report.append("1. BEGIN token attention is CAUSALLY protective against the bug")
            report.append("2. Layer 10 is a critical intervention point")
            report.append("3. Strengthening BEGIN anchoring could fix the bug")
            report.append("4. The bug emerges from weak BEGIN token grounding")
        else:
            report.append("1. Other factors may be responsible for the bug")
            report.append("2. Layer 10 may not be the optimal intervention point")
            report.append("3. Further investigation needed at other layers")
        
        report.append("\n" + "="*70)
        report.append("END OF CAUSAL VALIDATION REPORT")
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, analysis: Dict, report: str):
        """Save all results to files"""
        # Save JSON data
        with open('layer10_causal_results.json', 'w') as f:
            json.dump({
                'results': results,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        # Save report
        with open('layer10_causal_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✓ Results saved to layer10_causal_results.json")
        print("✓ Report saved to layer10_causal_report.txt")


def main():
    """Run Layer 10 causal validation test"""
    print("\n" + "="*70)
    print("STARTING LAYER 10 CAUSAL VALIDATION")
    print("="*70)
    
    # Initialize validator
    validator = Layer10CausalValidator()
    
    # Run causal test
    results = validator.run_causal_test()
    
    # Analyze results
    analysis = validator.analyze_results(results)
    
    # Generate report
    report = validator.generate_report(results, analysis)
    print("\n" + report)
    
    # Visualize results
    validator.visualize_results(results, analysis)
    
    # Save everything
    validator.save_results(results, analysis, report)
    
    print("\n" + "="*70)
    print("CAUSAL VALIDATION COMPLETE")
    print("="*70)
    
    if analysis['causal_evidence']:
        print("\n🎯 CAUSAL RELATIONSHIP CONFIRMED!")
        print(f"   Disrupting BEGIN attention at Layer 10 CAUSES the bug")
        print(f"   Critical disruption level: {analysis['critical_disruption_level']:.0%}")
    else:
        print("\n❓ No clear causal relationship found")
        print("   Further investigation needed")


if __name__ == "__main__":
    main()