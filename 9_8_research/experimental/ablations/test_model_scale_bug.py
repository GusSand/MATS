#!/usr/bin/env python
"""
Test multiple models of different scales for the decimal comparison bug
Models to test:
- pythia-160m
- pythia-410m 
- pythia-1b
- gemma-2-2b
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def test_model_for_bug(model_name, model_path, num_runs=10):
    """Test a specific model for the decimal comparison bug"""
    print(f"\n🤖 Testing {model_name}")
    print("-"*60)
    
    try:
        # Load model and tokenizer
        print(f"Loading {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if "gemma" in model_path else torch.float32,
            device_map="auto"
        )
        model.eval()
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def generate(prompt, max_new_tokens=30, temperature=0.2):
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=temperature > 0,
                )
            
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full[len(prompt):]
        
        # Test formats based on model type
        if "pythia" in model_path:
            # Pythia uses GPT-NeoX tokenizer, similar to GPT-2
            test_prompts = [
                "Q: Which is bigger: 9.8 or 9.11?\nA:",
                "Question: Which number is larger, 9.8 or 9.11?\nAnswer:",
            ]
        elif "gemma" in model_path:
            # Gemma uses specific chat format
            test_prompts = [
                "Which is bigger: 9.8 or 9.11?\nAnswer:",
                "<start_of_turn>user\nWhich is bigger: 9.8 or 9.11?<end_of_turn>\n<start_of_turn>model\n",
            ]
        elif "llama-3" in model_path.lower():
            # Llama-3 uses specific chat format
            test_prompts = [
                "Which is bigger: 9.8 or 9.11?\nAnswer:",
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            ]
        else:
            # Default formats
            test_prompts = [
                "Q: Which is bigger: 9.8 or 9.11?\nA:",
                "Which is bigger: 9.8 or 9.11?\nThe answer is",
            ]
        
        best_results = {"correct": 0, "incorrect": 0, "unclear": 0}
        best_prompt = None
        best_examples = []
        
        # Try each prompt format
        for prompt in test_prompts:
            correct = 0
            incorrect = 0
            unclear = 0
            examples = []
            
            print(f"\nTesting prompt: {repr(prompt[:50])}...")
            
            for i in range(num_runs):
                response = generate(prompt, temperature=0.2)
                response_lower = response.lower()
                
                # Analyze response
                if "9.8" in response and any(w in response_lower for w in ["bigger", "larger", "greater", "9.8", "correct"]):
                    correct += 1
                    symbol = "✓"
                elif "9.11" in response and any(w in response_lower for w in ["bigger", "larger", "greater", "9.11"]):
                    incorrect += 1
                    symbol = "✗"
                else:
                    unclear += 1
                    symbol = "?"
                
                if i < 3:
                    examples.append((symbol, response[:50]))
            
            # Print examples for this prompt
            for symbol, example in examples:
                print(f"  {symbol} {example}...")
            
            print(f"  Results: {correct}/10 correct, {incorrect}/10 incorrect, {unclear}/10 unclear")
            
            # Track best performing prompt
            if correct + incorrect > best_results["correct"] + best_results["incorrect"]:
                best_results = {"correct": correct, "incorrect": incorrect, "unclear": unclear}
                best_prompt = prompt
                best_examples = examples
        
        # Summary for this model
        print(f"\n📊 Best results for {model_name}:")
        print(f"  Prompt: {repr(best_prompt[:50])}...")
        print(f"  Correct (9.8):   {best_results['correct']}/{num_runs} = {best_results['correct']*10}%")
        print(f"  Incorrect (9.11): {best_results['incorrect']}/{num_runs} = {best_results['incorrect']*10}%")
        print(f"  Unclear:          {best_results['unclear']}/{num_runs} = {best_results['unclear']*10}%")
        
        bug_severity = "No bug" if best_results['incorrect'] == 0 else \
                      f"Bug present ({best_results['incorrect']*10}% error rate)"
        
        # Clean up memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_results['incorrect'] > 0, best_results['incorrect'] * 10, bug_severity
        
    except Exception as e:
        print(f"  Error testing {model_name}: {e}")
        return None, None, f"Error: {str(e)}"

def main():
    print("🔬 Testing Model Scale vs Decimal Bug")
    print("="*60)
    print("Testing if the decimal comparison bug emerges with model scale")
    print("Bug: Model incorrectly says '9.11 is bigger than 9.8'")
    
    # Models to test (name, huggingface path)
    models_to_test = [
        ("Pythia-160M", "EleutherAI/pythia-160m"),
        ("Pythia-410M", "EleutherAI/pythia-410m"),
        ("Pythia-1B", "EleutherAI/pythia-1b"),
        ("Gemma-2-2B", "google/gemma-2-2b"),
        ("Gemma-2-9B", "google/gemma-2-9b"),
        ("Llama-3-8B", "meta-llama/Meta-Llama-3-8B"),
    ]
    
    results = []
    
    for model_name, model_path in models_to_test:
        has_bug, error_rate, severity = test_model_for_bug(model_name, model_path)
        results.append({
            "model": model_name,
            "has_bug": has_bug,
            "error_rate": error_rate,
            "severity": severity
        })
    
    # Final summary
    print("\n\n" + "="*60)
    print("📊 SUMMARY: Model Scale vs Decimal Bug")
    print("="*60)
    print(f"{'Model':<15} {'Bug Present':<15} {'Error Rate':<12} {'Severity'}")
    print("-"*60)
    
    for result in results:
        bug_symbol = "❌" if result["has_bug"] else "✅" if result["has_bug"] is not None else "⚠️"
        error_str = f"{result['error_rate']}%" if result["error_rate"] is not None else "N/A"
        print(f"{result['model']:<15} {bug_symbol:<15} {error_str:<12} {result['severity']}")
    
    print("\n💡 Key Insights:")
    print("• GPT-2: No bug (from previous tests)")
    
    # Check if bug emerges with scale
    pythia_results = [r for r in results if "Pythia" in r["model"] and r["has_bug"] is not None]
    if pythia_results:
        if any(r["has_bug"] for r in pythia_results):
            print("• Pythia: Bug may emerge with scale")
        else:
            print("• Pythia: No clear bug across scales")
    
    if any(r["model"] == "Gemma-2-2B" and r["has_bug"] for r in results):
        print("• Gemma-2-2B: Shows the bug")
    
    print("• Llama-3.1-8B: Strong bug (from previous tests)")
    print("\nConclusion: Testing whether bug emerges with model scale...")

if __name__ == "__main__":
    main()