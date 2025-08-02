#!/usr/bin/env python
"""
Test Gemma models now that we have access
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def test_gemma_model(model_name, model_path, num_runs=10):
    """Test a Gemma model for the decimal comparison bug"""
    print(f"\n🤖 Testing {model_name}")
    print("-"*60)
    
    try:
        # Load model and tokenizer
        print(f"Loading {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
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
        
        # Test different Gemma formats
        test_prompts = [
            # Standard format
            "Q: Which is bigger: 9.8 or 9.11?\nA:",
            "Which is bigger: 9.8 or 9.11?\nAnswer:",
            
            # Gemma chat format
            "<start_of_turn>user\nWhich is bigger: 9.8 or 9.11?<end_of_turn>\n<start_of_turn>model\n",
            
            # Try with instruction format if it's an IT model
            "Instruction: Answer the following question.\nQuestion: Which is bigger: 9.8 or 9.11?\nAnswer:",
        ]
        
        # For models with chat templates
        if hasattr(tokenizer, 'apply_chat_template') and "-it" in model_path.lower():
            try:
                messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                test_prompts.insert(0, chat_prompt)
                print(f"Using chat template: {repr(chat_prompt[:80])}...")
            except:
                pass
        
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
        print(f"  Prompt: {repr(best_prompt[:50] if best_prompt else 'N/A')}...")
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
        return None, None, f"Error: {str(e)[:100]}..."

def main():
    print("🔬 Testing Gemma Models for Decimal Bug")
    print("="*60)
    print("Testing all Gemma models with HuggingFace access")
    
    # All Gemma models to test
    models_to_test = [
        # Gemma 1 series
        ("Gemma-2B", "google/gemma-2b"),
        ("Gemma-2B-IT", "google/gemma-2b-it"),
        ("Gemma-7B", "google/gemma-7b"),
        ("Gemma-7B-IT", "google/gemma-7b-it"),
        
        # Gemma 2 series
        ("Gemma-2-2B", "google/gemma-2-2b"),
        ("Gemma-2-2B-IT", "google/gemma-2-2b-it"),
        ("Gemma-2-9B", "google/gemma-2-9b"),
        ("Gemma-2-9B-IT", "google/gemma-2-9b-it"),
    ]
    
    results = []
    
    for model_name, model_path in models_to_test:
        has_bug, error_rate, severity = test_gemma_model(model_name, model_path)
        results.append({
            "model": model_name,
            "has_bug": has_bug,
            "error_rate": error_rate,
            "severity": severity
        })
    
    # Final summary
    print("\n\n" + "="*60)
    print("📊 GEMMA MODELS - Decimal Bug Test Summary")
    print("="*60)
    print(f"{'Model':<20} {'Bug Present':<15} {'Error Rate':<12} {'Severity'}")
    print("-"*60)
    
    for result in results:
        bug_symbol = "❌" if result["has_bug"] else "✅" if result["has_bug"] is not None else "⚠️"
        error_str = f"{result['error_rate']}%" if result["error_rate"] is not None else "N/A"
        print(f"{result['model']:<20} {bug_symbol:<15} {error_str:<12} {result['severity']}")
    
    print("\n💡 Key Insights:")
    
    # Check patterns
    base_models = [r for r in results if "-IT" not in r["model"]]
    it_models = [r for r in results if "-IT" in r["model"]]
    
    base_with_bug = sum(1 for r in base_models if r["has_bug"])
    it_with_bug = sum(1 for r in it_models if r["has_bug"])
    
    print(f"• Base models with bug: {base_with_bug}/{len(base_models)}")
    print(f"• Instruction-tuned models with bug: {it_with_bug}/{len(it_models)}")
    
    if it_with_bug > base_with_bug:
        print("• Pattern: Instruction tuning may increase bug likelihood")

if __name__ == "__main__":
    main()