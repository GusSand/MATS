"""
Comprehensive experiment to demonstrate format-dependent decimal comparison bug
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DecimalBugExperiment:
    def __init__(self):
        print("Loading model...")
        if torch.cuda.is_available():
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
        print("Model loaded successfully!")
        
    def generate_response(self, prompt):
        """Generate a single response"""
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
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_response[len(prompt):].strip()
        
        # Clear cache periodically
        torch.cuda.empty_cache()
        
        return generated
    
    def check_response(self, response):
        """Check if response shows the bug"""
        response_lower = response.lower()
        
        # Check for clear statements
        says_9_8_bigger = False
        says_9_11_bigger = False
        
        # Look for "9.8 is bigger/larger/greater"
        if "9.8" in response:
            if any(pattern in response_lower for pattern in [
                "9.8 is bigger", "9.8 is larger", "9.8 is greater",
                "9.8 is the bigger", "9.8 is the larger"
            ]):
                says_9_8_bigger = True
        
        # Look for "9.11 is bigger/larger/greater" (the bug)
        if "9.11" in response:
            if any(pattern in response_lower for pattern in [
                "9.11 is bigger", "9.11 is larger", "9.11 is greater", 
                "9.11 is the bigger", "9.11 is the larger"
            ]):
                says_9_11_bigger = True
        
        # If both detected, check which comes first
        if says_9_8_bigger and says_9_11_bigger:
            idx_9_8 = response_lower.find("9.8 is")
            idx_9_11 = response_lower.find("9.11 is")
            if idx_9_8 >= 0 and idx_9_11 >= 0:
                if idx_9_8 < idx_9_11:
                    says_9_11_bigger = False
                else:
                    says_9_8_bigger = False
        
        return {
            'is_correct': says_9_8_bigger and not says_9_11_bigger,
            'shows_bug': says_9_11_bigger and not says_9_8_bigger,
            'says_9_8': says_9_8_bigger,
            'says_9_11': says_9_11_bigger
        }
    
    def run_experiment(self, n_trials=10):
        """Run the full experiment"""
        test_formats = [
            ("Simple", "Which is bigger: 9.8 or 9.11?\nAnswer:"),
            ("Q&A", "Q: Which is bigger: 9.8 or 9.11?\nA:"),
            ("Chat", "User: Which is bigger: 9.8 or 9.11?\nAssistant:"),
            ("Direct", "9.8 or 9.11? The bigger number is"),
        ]
        
        results = []
        
        for format_name, prompt in test_formats:
            print(f"\nTesting {format_name} format ({n_trials} trials)...")
            
            for trial in range(n_trials):
                response = self.generate_response(prompt)
                check_result = self.check_response(response)
                
                results.append({
                    'format': format_name,
                    'trial': trial,
                    'prompt': prompt[:50],
                    'response': response[:100],
                    'is_correct': check_result['is_correct'],
                    'shows_bug': check_result['shows_bug'],
                    'says_9_8': check_result['says_9_8'],
                    'says_9_11': check_result['says_9_11']
                })
                
                # Print first response as example
                if trial == 0:
                    symbol = "✅" if check_result['is_correct'] else "❌" if check_result['shows_bug'] else "❓"
                    print(f"  {symbol} Sample: {response[:60]}...")
        
        return pd.DataFrame(results)
    
    def analyze_results(self, df):
        """Analyze and print results"""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        # Calculate statistics by format
        summary = df.groupby('format').agg({
            'is_correct': 'mean',
            'shows_bug': 'mean',
            'says_9_8': 'mean',
            'says_9_11': 'mean'
        }).round(3)
        
        summary.columns = ['Correct Rate', 'Bug Rate', 'Says 9.8', 'Says 9.11']
        print(summary)
        
        # Key finding
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        for format_name in df['format'].unique():
            format_data = df[df['format'] == format_name]
            bug_rate = format_data['shows_bug'].mean()
            correct_rate = format_data['is_correct'].mean()
            
            print(f"\n{format_name} format:")
            print(f"  Correct rate: {correct_rate:.1%}")
            print(f"  Bug rate: {bug_rate:.1%}")
            
            if bug_rate > 0.5:
                print(f"  ❌ HIGH BUG RATE - Format triggers the bug!")
            elif correct_rate > 0.8:
                print(f"  ✅ LOW BUG RATE - Format avoids the bug!")
        
        return summary
    
    def visualize_results(self, df):
        """Create visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bug rates by format
        ax1 = axes[0]
        bug_rates = df.groupby('format')['shows_bug'].mean()
        colors = ['red' if rate > 0.5 else 'green' for rate in bug_rates]
        bug_rates.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Bug Rate by Prompt Format', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Bug Rate (Says 9.11 > 9.8)')
        ax1.set_xlabel('Format')
        ax1.set_ylim([0, 1])
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Correct vs Bug comparison
        ax2 = axes[1]
        summary_data = df.groupby('format')[['is_correct', 'shows_bug']].mean()
        summary_data.plot(kind='bar', ax=ax2, color=['green', 'red'])
        ax2.set_title('Response Patterns by Format', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Proportion')
        ax2.set_xlabel('Format')
        ax2.legend(['Correct (9.8 > 9.11)', 'Bug (9.11 > 9.8)'])
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Format-Dependent Decimal Comparison Bug in Llama-3.1-8B', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

def main():
    # Initialize experiment
    exp = DecimalBugExperiment()
    
    # Run with fewer trials to avoid GPU issues
    print("\nRunning experiment...")
    results_df = exp.run_experiment(n_trials=5)
    
    # Analyze
    summary = exp.analyze_results(results_df)
    
    # Visualize
    fig = exp.visualize_results(results_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'results_{timestamp}.csv', index=False)
    fig.savefig(f'bug_analysis_{timestamp}.pdf', dpi=300, bbox_inches='tight')
    
    # Save summary
    with open(f'summary_{timestamp}.json', 'w') as f:
        json.dump({
            'summary': summary.to_dict(),
            'total_trials': len(results_df),
            'formats_tested': results_df['format'].unique().tolist()
        }, f, indent=2)
    
    print(f"\n✅ Results saved with timestamp: {timestamp}")
    print("Files created:")
    print(f"  - results_{timestamp}.csv")
    print(f"  - bug_analysis_{timestamp}.pdf")
    print(f"  - summary_{timestamp}.json")
    
    return results_df

if __name__ == "__main__":
    results = main()
    print("\n✅ Experiment complete!")