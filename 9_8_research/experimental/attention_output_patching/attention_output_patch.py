"""
Correct Implementation of Attention Output Patching
Based on BREAKTHROUGH_FINDINGS.md - patches self_attn module output
This should achieve 100% success rate by replacing attention output entirely
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class AttentionOutputPatcher:
    def __init__(self):
        print("Loading model...")
        torch.cuda.empty_cache()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print("Model loaded!")
        
        # Storage for attention outputs
        self.stored_outputs = {}
        self.patch_source = None
        
    def extract_attention_output(self, prompt, label="default"):
        """Extract and store attention output from Layer 10's self_attn module"""
        
        def extract_hook(module, inputs, outputs):
            # Store the COMPLETE attention output
            if isinstance(outputs, tuple):
                # outputs[0] is the attention output
                self.stored_outputs[label] = outputs[0].clone().detach()
            else:
                self.stored_outputs[label] = outputs.clone().detach()
            return outputs
        
        # Register hook on self_attn module
        handle = self.model.model.layers[10].self_attn.register_forward_hook(extract_hook)
        
        # Run forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hook
        handle.remove()
        
        print(f"✓ Extracted {label} attention output: {self.stored_outputs[label].shape}")
        return self.stored_outputs[label]
        
    def create_patching_hook(self, source_label):
        """Create hook that COMPLETELY REPLACES attention output"""
        
        def patch_hook(module, inputs, outputs):
            if source_label not in self.stored_outputs:
                return outputs
            
            stored = self.stored_outputs[source_label]
            
            # Get current output shape
            if isinstance(outputs, tuple):
                current_output = outputs[0]
            else:
                current_output = outputs
            
            batch_size, seq_len, hidden_dim = current_output.shape
            stored_seq_len = stored.shape[1]
            
            # COMPLETE REPLACEMENT (not modification)
            # Create new output tensor
            patched_output = torch.zeros_like(current_output)
            
            # Copy the stored output completely
            min_seq_len = min(seq_len, stored_seq_len)
            patched_output[:, :min_seq_len, :] = stored[:, :min_seq_len, :]
            
            # If current sequence is longer, use the last stored position for padding
            if seq_len > stored_seq_len:
                patched_output[:, stored_seq_len:, :] = stored[:, -1:, :].expand(-1, seq_len - stored_seq_len, -1)
            
            print(f"    ✓ Patched: Replaced attention output completely ({min_seq_len}/{seq_len} positions)")
            
            # Return in same format as original
            if isinstance(outputs, tuple):
                return (patched_output,) + outputs[1:]
            return patched_output
        
        return patch_hook
    
    def generate_with_patching(self, prompt, source_label=None):
        """Generate with optional attention output patching"""
        
        # Set up patching if requested
        handle = None
        if source_label and source_label in self.stored_outputs:
            print(f"  Applying patch from '{source_label}' format...")
            patch_hook = self.create_patching_hook(source_label)
            handle = self.model.model.layers[10].self_attn.register_forward_hook(patch_hook)
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        # Remove hook
        if handle:
            handle.remove()
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_response[len(prompt):].strip()
        
        torch.cuda.empty_cache()
        
        # Check result
        generated_lower = generated.lower()
        
        # More comprehensive checking
        says_9_8_bigger = False
        says_9_11_bigger = False
        
        # Check for "9.8 is bigger/larger/greater"
        if "9.8" in generated:
            if any(phrase in generated_lower for phrase in [
                "9.8 is bigger", "9.8 is larger", "9.8 is greater",
                "9.8 is the bigger", "9.8 is the larger"
            ]):
                says_9_8_bigger = True
        
        # Check for "9.11 is bigger/larger/greater"
        if "9.11" in generated:
            if any(phrase in generated_lower for phrase in [
                "9.11 is bigger", "9.11 is larger", "9.11 is greater",
                "9.11 is the bigger", "9.11 is the larger"
            ]):
                says_9_11_bigger = True
        
        # Determine which comes first if both present
        if says_9_8_bigger and says_9_11_bigger:
            idx_9_8 = generated_lower.find("9.8 is")
            idx_9_11 = generated_lower.find("9.11 is")
            if idx_9_8 >= 0 and idx_9_11 >= 0:
                if idx_9_8 < idx_9_11:
                    says_9_11_bigger = False
                else:
                    says_9_8_bigger = False
        
        return {
            'response': generated[:100],
            'is_correct': says_9_8_bigger and not says_9_11_bigger,
            'shows_bug': says_9_11_bigger and not says_9_8_bigger,
            'says_9_8': says_9_8_bigger,
            'says_9_11': says_9_11_bigger
        }
    
    def run_complete_experiment(self, n_trials=5):
        """Run the complete patching experiment with multiple trials"""
        print("\n" + "="*70)
        print("ATTENTION OUTPUT PATCHING EXPERIMENT")
        print("="*70)
        print("Based on BREAKTHROUGH_FINDINGS.md")
        print("Patching self_attn module output from correct → buggy format")
        
        # Define prompts
        correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
        buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        
        results = {
            'baseline_buggy': [],
            'patched_buggy': [],
            'baseline_correct': [],
            'reverse_patch': []
        }
        
        # Step 1: Extract correct attention output (do once)
        print("\n1. EXTRACTING ATTENTION OUTPUT FROM CORRECT FORMAT")
        print("-" * 50)
        self.extract_attention_output(correct_prompt, "correct_format")
        
        # Step 2: Extract buggy attention output (do once)
        print("\n2. EXTRACTING ATTENTION OUTPUT FROM BUGGY FORMAT")
        print("-" * 50)
        self.extract_attention_output(buggy_prompt, "buggy_format")
        
        # Step 3: Run trials
        print("\n3. RUNNING EXPERIMENTS")
        print("-" * 50)
        
        for trial in range(n_trials):
            print(f"\n  Trial {trial + 1}/{n_trials}:")
            
            # Test 1: Baseline correct format
            print("    Testing correct format (baseline)...")
            result = self.generate_with_patching(correct_prompt, source_label=None)
            results['baseline_correct'].append(result)
            
            # Test 2: Baseline buggy format
            print("    Testing buggy format (baseline)...")
            result = self.generate_with_patching(buggy_prompt, source_label=None)
            results['baseline_buggy'].append(result)
            
            # Test 3: Buggy format WITH correct attention (THE KEY TEST)
            print("    Testing buggy format WITH correct attention...")
            result = self.generate_with_patching(buggy_prompt, source_label="correct_format")
            results['patched_buggy'].append(result)
            
            # Test 4: Correct format WITH buggy attention (reverse test)
            print("    Testing correct format WITH buggy attention...")
            result = self.generate_with_patching(correct_prompt, source_label="buggy_format")
            results['reverse_patch'].append(result)
        
        return results
    
    def analyze_results(self, results):
        """Analyze and report results"""
        print("\n" + "="*70)
        print("RESULTS ANALYSIS")
        print("="*70)
        
        # Calculate success rates
        stats = {}
        for condition, trials in results.items():
            correct_count = sum(1 for t in trials if t['is_correct'])
            bug_count = sum(1 for t in trials if t['shows_bug'])
            stats[condition] = {
                'correct_rate': correct_count / len(trials) * 100,
                'bug_rate': bug_count / len(trials) * 100,
                'n_trials': len(trials)
            }
        
        # Print results table
        print("\n| Condition | Correct Rate | Bug Rate | N |")
        print("|-----------|-------------|----------|---|")
        print(f"| Baseline Correct | {stats['baseline_correct']['correct_rate']:.0f}% | {stats['baseline_correct']['bug_rate']:.0f}% | {stats['baseline_correct']['n_trials']} |")
        print(f"| Baseline Buggy | {stats['baseline_buggy']['correct_rate']:.0f}% | {stats['baseline_buggy']['bug_rate']:.0f}% | {stats['baseline_buggy']['n_trials']} |")
        print(f"| **Buggy + Correct Attn** | **{stats['patched_buggy']['correct_rate']:.0f}%** | **{stats['patched_buggy']['bug_rate']:.0f}%** | {stats['patched_buggy']['n_trials']} |")
        print(f"| Correct + Buggy Attn | {stats['reverse_patch']['correct_rate']:.0f}% | {stats['reverse_patch']['bug_rate']:.0f}% | {stats['reverse_patch']['n_trials']} |")
        
        # Sample responses
        print("\n" + "-"*70)
        print("SAMPLE RESPONSES")
        print("-"*70)
        
        for condition, trials in results.items():
            if trials:
                print(f"\n{condition}:")
                print(f"  {trials[0]['response'][:80]}...")
        
        # Key findings
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        # Check if patching worked
        if stats['patched_buggy']['correct_rate'] > 80:
            print("✅ SUCCESS! Patching attention output FIXES the bug!")
            print(f"   Bug rate: {stats['baseline_buggy']['bug_rate']:.0f}% → {stats['patched_buggy']['bug_rate']:.0f}%")
            print(f"   Correct rate: {stats['baseline_buggy']['correct_rate']:.0f}% → {stats['patched_buggy']['correct_rate']:.0f}%")
            print("\n   This confirms Layer 10 attention output is CAUSAL!")
        else:
            print("❌ Patching did not achieve high success rate")
        
        # Check reverse patching
        if stats['reverse_patch']['bug_rate'] > 50:
            print("\n✅ Reverse patching INDUCES the bug in correct format!")
            print(f"   Bug rate: {stats['baseline_correct']['bug_rate']:.0f}% → {stats['reverse_patch']['bug_rate']:.0f}%")
            print("   This confirms bidirectional causality!")
        
        return stats

def main():
    print("="*70)
    print("ATTENTION OUTPUT PATCHING - CAUSAL VALIDATION")
    print("="*70)
    
    # Initialize
    patcher = AttentionOutputPatcher()
    
    # Run experiment
    results = patcher.run_complete_experiment(n_trials=5)
    
    # Analyze
    stats = patcher.analyze_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    with open(f'/home/paperspace/dev/MATS9/attention_output_patching/results_{timestamp}.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for condition, trials in results.items():
            serializable_results[condition] = trials
        
        json.dump({
            'results': serializable_results,
            'stats': stats,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n✅ Results saved to results_{timestamp}.json")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if stats['patched_buggy']['correct_rate'] >= 100:
        print("🎉 100% SUCCESS ACHIEVED!")
        print("Layer 10 attention output is definitively CAUSAL for the decimal bug.")
    elif stats['patched_buggy']['correct_rate'] >= 80:
        print("✅ HIGH SUCCESS RATE!")
        print("Strong evidence for causal relationship.")
    else:
        print("❌ Causal relationship not established")
        print("May need to investigate implementation or other factors.")
    
    return results, stats

if __name__ == "__main__":
    results, stats = main()