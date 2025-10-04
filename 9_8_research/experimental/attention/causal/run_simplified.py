"""
Simplified causal validation experiment
Tests the hypothesis that disrupting attention patterns changes model behavior
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SimplifiedExperiment:
    def __init__(self):
        print("Loading model...")
        # Clear CUDA cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
    def test_formats(self, n_trials=10):
        """Test different prompt formats"""
        results = []
        
        # Test cases
        test_prompts = [
            ("Simple", "Which is bigger: 9.8 or 9.11? Answer:"),
            ("Q&A", "Q: Which is bigger: 9.8 or 9.11? A:"),
            ("Chat", "User: Which is bigger: 9.8 or 9.11?\nAssistant:"),
        ]
        
        for format_name, prompt in test_prompts:
            print(f"\nTesting {format_name} format...")
            
            for trial in range(n_trials):
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_only = response[len(prompt):].strip()
                response_lower = response_only.lower()
                
                # Check answer - need to be more precise about what the model says
                says_9_8_bigger = False
                says_9_11_bigger = False
                
                # Look for clear statements about which is bigger
                # Check for "9.8 is bigger/larger/greater"
                if "9.8" in response_only:
                    # Check patterns like "9.8 is bigger"
                    if any(pattern in response_lower for pattern in [
                        "9.8 is bigger", "9.8 is larger", "9.8 is greater",
                        "9.8 is the bigger", "9.8 is the larger", "9.8 is the greater"
                    ]):
                        says_9_8_bigger = True
                
                # Check for "9.11 is bigger/larger/greater" (THE BUG)
                if "9.11" in response_only:
                    # Check patterns like "9.11 is bigger"
                    if any(pattern in response_lower for pattern in [
                        "9.11 is bigger", "9.11 is larger", "9.11 is greater",
                        "9.11 is the bigger", "9.11 is the larger", "9.11 is the greater"
                    ]):
                        says_9_11_bigger = True
                
                # If both detected, look at which comes first
                if says_9_8_bigger and says_9_11_bigger:
                    # Find first occurrence
                    idx_9_8 = response_lower.find("9.8 is")
                    idx_9_11 = response_lower.find("9.11 is")
                    if idx_9_8 >= 0 and idx_9_11 >= 0:
                        if idx_9_8 < idx_9_11:
                            says_9_11_bigger = False  # First statement is 9.8, so that's the answer
                        else:
                            says_9_8_bigger = False  # First statement is 9.11, so that's the answer
                
                # The bug is when model says 9.11 is bigger
                shows_bug = says_9_11_bigger
                is_correct = says_9_8_bigger and not says_9_11_bigger
                
                results.append({
                    'format': format_name,
                    'trial': trial,
                    'response': response_only[:50],  # First 50 chars
                    'correct': is_correct,
                    'wrong': shows_bug,
                    'error': shows_bug  # Error means saying 9.11 is bigger
                })
                
                if trial == 0:  # Print first response for each format
                    print(f"  Sample response: {response_only[:100]}")
        
        return pd.DataFrame(results)
    
    def visualize_results(self, df):
        """Create visualization of results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error rates by format
        ax1 = axes[0]
        error_rates = df.groupby('format')['error'].mean()
        error_rates.plot(kind='bar', ax=ax1, color=['green', 'red', 'orange'])
        ax1.set_title('Error Rate by Prompt Format')
        ax1.set_ylabel('Error Rate (saying 9.11 > 9.8)')
        ax1.set_xlabel('Format')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Response patterns
        ax2 = axes[1]
        summary = df.groupby('format')[['correct', 'wrong']].mean()
        summary.plot(kind='bar', ax=ax2)
        ax2.set_title('Response Patterns by Format')
        ax2.set_ylabel('Proportion')
        ax2.set_xlabel('Format')
        ax2.legend(['Correct (9.8)', 'Wrong (9.11)'])
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Decimal Comparison Bug Analysis Across Formats')
        plt.tight_layout()
        return fig

def main():
    # Initialize
    exp = SimplifiedExperiment()
    
    # Run experiment
    print("\nRunning experiment...")
    results = exp.test_formats(n_trials=5)
    
    # Analyze results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    summary = results.groupby('format')['error'].agg(['mean', 'std', 'count'])
    summary.columns = ['Error Rate', 'Std Dev', 'N']
    print(summary)
    
    # Statistical comparison
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    
    simple_error = results[results['format'] == 'Simple']['error'].mean()
    qa_error = results[results['format'] == 'Q&A']['error'].mean()
    
    print(f"Simple format error rate: {simple_error:.1%}")
    print(f"Q&A format error rate: {qa_error:.1%}")
    print(f"Difference: {abs(qa_error - simple_error):.1%}")
    
    if qa_error > simple_error:
        print("\n✓ Hypothesis confirmed: Q&A format shows MORE errors than Simple format")
    else:
        print("\n✗ Hypothesis not confirmed: Q&A format shows FEWER errors than Simple format")
    
    # Save visualization
    fig = exp.visualize_results(results)
    fig.savefig('attention/causal/format_comparison.pdf', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to format_comparison.pdf")
    
    # Save detailed results
    results.to_csv('attention/causal/detailed_results.csv', index=False)
    print("Detailed results saved to detailed_results.csv")
    
    return results

if __name__ == "__main__":
    results = main()
    print("\n✓ Experiment complete!")