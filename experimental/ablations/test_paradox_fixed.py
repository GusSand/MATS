#!/usr/bin/env python
"""
Fixed test for instruction tuning paradox - focusing on key comparisons
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def test_model_simple(model_name, model_path, model_type='base', num_runs=20):
    """Simplified test focusing on 9.8 vs 9.11"""
    print(f"\n{'='*60}")
    print(f"🤖 Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print(f"Loading {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
        
        def generate(prompt, max_new_tokens=50, temperature=0.2):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
            generated = full[len(prompt):]
            return generated
        
        # Use appropriate prompt format
        if model_type == 'instruct':
            if 'llama' in model_path.lower():
                # Try simple format for Llama instruct
                prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
            elif 'gemma' in model_path.lower() and hasattr(tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        else:
            # Base model
            prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        
        print(f"Using prompt: {repr(prompt[:60])}...")
        
        correct = 0
        incorrect = 0
        examples = []
        
        for i in range(num_runs):
            response = generate(prompt, temperature=0.2)
            
            # Simple check
            if "9.8" in response and any(w in response.lower() for w in ["bigger", "larger", "greater", "9.8"]):
                correct += 1
                symbol = "✓"
            elif "9.11" in response and any(w in response.lower() for w in ["bigger", "larger", "greater", "9.11"]):
                incorrect += 1
                symbol = "✗"
            else:
                symbol = "?"
            
            if i < 5:
                examples.append((symbol, response[:50]))
        
        # Show examples
        print("\nExample responses:")
        for symbol, example in examples:
            print(f"  {symbol} {example}...")
        
        accuracy = correct / num_runs * 100
        error_rate = incorrect / num_runs * 100
        
        print(f"\n📊 Results: {correct}/{num_runs} correct ({accuracy:.0f}%), {incorrect}/{num_runs} wrong ({error_rate:.0f}%)")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return error_rate
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    print("🔬 INSTRUCTION TUNING PARADOX - SIMPLIFIED TEST")
    print("Testing: Which is bigger: 9.8 or 9.11?")
    print("Correct answer: 9.8")
    
    # Key comparisons
    test_pairs = [
        # Gemma family
        ("Gemma-2B Base", "google/gemma-2b", "base"),
        ("Gemma-2B Instruct", "google/gemma-2b-it", "instruct"),
        ("Gemma-7B Base", "google/gemma-7b", "base"),
        ("Gemma-7B Instruct", "google/gemma-7b-it", "instruct"),
        
        # Test with our known working Llama
        ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "instruct"),
    ]
    
    results = {}
    
    for model_name, model_path, model_type in test_pairs:
        error_rate = test_model_simple(model_name, model_path, model_type)
        results[model_name] = error_rate
    
    # Analysis
    print("\n\n" + "="*60)
    print("📊 FINAL ANALYSIS - INSTRUCTION TUNING PARADOX")
    print("="*60)
    
    print("\n🔍 GEMMA FAMILY:")
    print(f"  Gemma-2B: Base {results.get('Gemma-2B Base', 'N/A')}% → Instruct {results.get('Gemma-2B Instruct', 'N/A')}%")
    print(f"  Gemma-7B: Base {results.get('Gemma-7B Base', 'N/A')}% → Instruct {results.get('Gemma-7B Instruct', 'N/A')}%")
    
    print("\n🔍 LLAMA FAMILY:")
    print(f"  Llama-3.1-8B-Instruct: {results.get('Llama-3.1-8B-Instruct', 'N/A')}% error")
    
    # Check patterns
    gemma_2b_base = results.get('Gemma-2B Base', 100)
    gemma_2b_inst = results.get('Gemma-2B Instruct', 0)
    gemma_7b_base = results.get('Gemma-7B Base', 100)
    gemma_7b_inst = results.get('Gemma-7B Instruct', 0)
    
    print("\n💡 PARADOX VERIFICATION:")
    if gemma_2b_base > gemma_2b_inst:
        print("✅ Gemma-2B: Instruction tuning REDUCES errors")
    elif gemma_2b_base < gemma_2b_inst:
        print("❌ Gemma-2B: Instruction tuning INCREASES errors")
        
    if gemma_7b_base > gemma_7b_inst:
        print("✅ Gemma-7B: Instruction tuning REDUCES errors")
    elif gemma_7b_base < gemma_7b_inst:
        print("❌ Gemma-7B: Instruction tuning INCREASES errors")
    
    llama_inst = results.get('Llama-3.1-8B-Instruct', 0)
    if llama_inst > 50:
        print("✅ Llama-3.1-8B-Instruct: Shows strong decimal bug")

if __name__ == "__main__":
    main()