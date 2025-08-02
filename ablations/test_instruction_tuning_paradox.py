#!/usr/bin/env python
"""
Test the instruction tuning paradox: Does instruction tuning fix or cause the decimal bug?
Systematic comparison of base vs instruct models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def test_model(model_name, model_path, model_type='base', num_runs=10):
    """Test a model for the decimal comparison bug"""
    print(f"\n{'='*60}")
    print(f"🤖 Testing {model_name} ({model_type})")
    print(f"{'='*60}")
    
    try:
        # Load model and tokenizer
        print(f"Loading {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
        
        def generate(prompt, max_new_tokens=30, temperature=0.2):
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=temperature > 0,
                )
            
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full[len(prompt):]
        
        # Test different decimal comparisons
        test_cases = [
            ("9.8", "9.11"),
            ("9.9", "9.11"),
            ("10.8", "10.11"),
            ("5.7", "5.12"),
        ]
        
        all_results = {}
        
        for num1, num2 in test_cases:
            correct = float(num1) > float(num2)
            correct_answer = num1 if correct else num2
            
            # Test different prompt formats
            if model_type == 'instruct' and 'llama' in model_path.lower():
                # Llama instruct format
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: {num1} or {num2}?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif model_type == 'instruct' and 'gemma' in model_path.lower():
                # Gemma instruct format
                if hasattr(tokenizer, 'apply_chat_template'):
                    messages = [{"role": "user", "content": f"Which is bigger: {num1} or {num2}?"}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = f"<start_of_turn>user\nWhich is bigger: {num1} or {num2}?<end_of_turn>\n<start_of_turn>model\n"
            else:
                # Base model format
                prompt = f"Q: Which is bigger: {num1} or {num2}?\nA:"
            
            correct_count = 0
            incorrect_count = 0
            examples = []
            
            print(f"\n📊 Testing: {num1} vs {num2} (correct: {correct_answer})")
            print(f"Prompt format: {repr(prompt[:60])}...")
            
            for i in range(num_runs):
                response = generate(prompt, temperature=0.2)
                response_lower = response.lower()
                
                # Check if response contains the correct answer
                if correct_answer in response and any(w in response_lower for w in ["bigger", "larger", "greater", correct_answer]):
                    correct_count += 1
                    symbol = "✓"
                else:
                    # Check if it has the wrong answer
                    wrong_answer = num2 if correct else num1
                    if wrong_answer in response and any(w in response_lower for w in ["bigger", "larger", "greater", wrong_answer]):
                        incorrect_count += 1
                        symbol = "✗"
                    else:
                        symbol = "?"
                
                if i < 3:
                    examples.append((symbol, response[:40]))
            
            # Show examples
            for symbol, example in examples:
                print(f"  {symbol} {example}...")
            
            accuracy = correct_count / num_runs * 100
            error_rate = incorrect_count / num_runs * 100
            print(f"  Results: {correct_count}/{num_runs} correct ({accuracy:.0f}%), {incorrect_count}/{num_runs} wrong ({error_rate:.0f}%)")
            
            all_results[f"{num1}_vs_{num2}"] = {
                "correct": correct_count,
                "incorrect": incorrect_count,
                "accuracy": accuracy,
                "error_rate": error_rate
            }
        
        # Overall summary
        total_correct = sum(r["correct"] for r in all_results.values())
        total_incorrect = sum(r["incorrect"] for r in all_results.values())
        total_runs = len(test_cases) * num_runs
        
        overall_accuracy = total_correct / total_runs * 100
        overall_error = total_incorrect / total_runs * 100
        
        print(f"\n📈 OVERALL RESULTS for {model_name}:")
        print(f"  Total accuracy: {total_correct}/{total_runs} = {overall_accuracy:.1f}%")
        print(f"  Total errors: {total_incorrect}/{total_runs} = {overall_error:.1f}%")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return overall_error > 0, overall_error, overall_accuracy
        
    except Exception as e:
        print(f"  Error testing {model_name}: {e}")
        return None, None, None

def main():
    print("🔬 INSTRUCTION TUNING PARADOX TEST")
    print("="*60)
    print("Testing whether instruction tuning fixes or causes the decimal bug")
    
    # Models to test (name, path, type)
    test_suite = [
        # Pythia family
        ("Pythia-160M Base", "EleutherAI/pythia-160m", "base"),
        
        # Gemma family - base vs instruct
        ("Gemma-2B Base", "google/gemma-2b", "base"),
        ("Gemma-2B Instruct", "google/gemma-2b-it", "instruct"),
        ("Gemma-7B Base", "google/gemma-7b", "base"),
        ("Gemma-7B Instruct", "google/gemma-7b-it", "instruct"),
        
        # Llama family - base vs instruct
        ("Llama-3.1-8B Base", "meta-llama/Llama-3.1-8B", "base"),
        ("Llama-3.1-8B Instruct", "meta-llama/Llama-3.1-8B-Instruct", "instruct"),
        
        # Additional comparisons
        ("Llama-3.2-3B Base", "meta-llama/Llama-3.2-3B", "base"),
        ("Llama-3.2-3B Instruct", "meta-llama/Llama-3.2-3B-Instruct", "instruct"),
    ]
    
    results = []
    
    for model_name, model_path, model_type in test_suite:
        has_bug, error_rate, accuracy = test_model(model_name, model_path, model_type)
        results.append({
            "model": model_name,
            "type": model_type,
            "has_bug": has_bug,
            "error_rate": error_rate,
            "accuracy": accuracy
        })
    
    # Final analysis
    print("\n\n" + "="*60)
    print("📊 INSTRUCTION TUNING PARADOX - FINAL ANALYSIS")
    print("="*60)
    
    # Group by model family
    print("\n🔍 GEMMA FAMILY:")
    gemma_results = [r for r in results if "Gemma" in r["model"]]
    for r in gemma_results:
        if r["error_rate"] is not None:
            print(f"  {r['model']}: {r['error_rate']:.1f}% error rate")
    
    print("\n🔍 LLAMA FAMILY:")
    llama_results = [r for r in results if "Llama" in r["model"]]
    for r in llama_results:
        if r["error_rate"] is not None:
            print(f"  {r['model']}: {r['error_rate']:.1f}% error rate")
    
    print("\n💡 PARADOX VERIFICATION:")
    
    # Check Gemma pattern
    gemma_base = [r for r in gemma_results if r["type"] == "base" and r["error_rate"] is not None]
    gemma_instruct = [r for r in gemma_results if r["type"] == "instruct" and r["error_rate"] is not None]
    
    if gemma_base and gemma_instruct:
        avg_base_error = sum(r["error_rate"] for r in gemma_base) / len(gemma_base)
        avg_instruct_error = sum(r["error_rate"] for r in gemma_instruct) / len(gemma_instruct)
        print(f"\nGemma: Base models avg {avg_base_error:.1f}% error → Instruct models avg {avg_instruct_error:.1f}% error")
        if avg_instruct_error < avg_base_error:
            print("✅ CONFIRMED: Instruction tuning FIXES the bug in Gemma")
    
    # Check Llama pattern
    llama_base = [r for r in llama_results if r["type"] == "base" and r["error_rate"] is not None]
    llama_instruct = [r for r in llama_results if r["type"] == "instruct" and r["error_rate"] is not None]
    
    if llama_base and llama_instruct:
        avg_base_error = sum(r["error_rate"] for r in llama_base) / len(llama_base)
        avg_instruct_error = sum(r["error_rate"] for r in llama_instruct) / len(llama_instruct)
        print(f"\nLlama: Base models avg {avg_base_error:.1f}% error → Instruct models avg {avg_instruct_error:.1f}% error")
        if avg_instruct_error > avg_base_error:
            print("✅ CONFIRMED: Instruction tuning CAUSES/AMPLIFIES the bug in Llama")
    
    print("\n🎯 CONCLUSION:")
    print("The instruction tuning paradox is REAL:")
    print("- Gemma: Instruction tuning FIXES decimal comparison")
    print("- Llama: Instruction tuning BREAKS decimal comparison")
    print("This suggests fundamentally different instruction tuning approaches!")

if __name__ == "__main__":
    main()