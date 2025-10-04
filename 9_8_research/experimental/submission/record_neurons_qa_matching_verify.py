#!/usr/bin/env python
"""
Record neurons using the EXACT same generation settings as verify_llama_bug.py
This should reproduce the higher rate of correct answers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Recording Neurons with verify_llama_bug.py Settings")
print("="*60)

# Load model
model_path = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Q&A format prompt - EXACT same as verify_llama_bug.py
prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

print(f"\nPrompt: {repr(prompt)}")
print("Using temperature=0.2 (same as verify_llama_bug.py)")

# Track results
results = {"correct": 0, "incorrect": 0, "unclear": 0}
correct_neurons_saved = False

# Run 20 times to get statistics
for attempt in range(1, 21):
    # Storage for activations
    layer_activations = defaultdict(list)
    
    def capture_activations_hook(module, input, output, layer_idx):
        """Hook to capture MLP activations"""
        if hasattr(output, 'detach'):
            activation = output.detach().cpu().float()
            layer_activations[layer_idx].append(activation)
    
    # Register hooks
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        hook = layer.mlp.act_fn.register_forward_hook(
            lambda module, input, output, idx=idx: capture_activations_hook(module, input, output, idx)
        )
        hooks.append(hook)
    
    # Generate - matching verify_llama_bug.py EXACTLY
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # temperature > 0
        )
    
    # Get full response (matching verify_llama_bug.py)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part
    generated = full[len(prompt):]
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Check answer using same logic as verify_llama_bug.py
    generated_lower = generated.lower()
    
    says_9_8_bigger = (
        ("9.8" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.11" in generated_lower) or
        ("larger than 9.11" in generated_lower) or
        ("greater than 9.11" in generated_lower)
    )
    
    says_9_11_bigger = (
        ("9.11" in generated and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.8" in generated_lower) or
        ("larger than 9.8" in generated_lower) or
        ("greater than 9.8" in generated_lower)
    )
    
    # Check for first clear statement if both detected
    if says_9_8_bigger and says_9_11_bigger:
        words = generated_lower.split()
        for i, word in enumerate(words):
            if word in ["9.8", "9.11"] and i + 1 < len(words):
                if words[i + 1] in ["is", "are"] and i + 2 < len(words):
                    if words[i + 2] in ["bigger", "larger", "greater"]:
                        if word == "9.8":
                            says_9_8_bigger = True
                            says_9_11_bigger = False
                        else:
                            says_9_8_bigger = False
                            says_9_11_bigger = True
                        break
    
    # Categorize result
    if says_9_8_bigger and not says_9_11_bigger:
        results["correct"] += 1
        symbol = "✅"
        
        # Save neurons for first correct answer
        if not correct_neurons_saved:
            correct_neurons_saved = True
            print(f"\n✅ SAVING NEURONS FOR CORRECT ANSWER!")
            
            # Get tokens for analysis
            generated_ids = outputs[0][len(inputs['input_ids'][0]):]
            prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
            
            # Find important positions
            important_positions = []
            for i, token in enumerate(generated_tokens):
                if i + 2 < len(generated_tokens) and generated_tokens[i:i+3] == ['9', '.', '8']:
                    important_positions.extend([len(prompt_tokens) + i, len(prompt_tokens) + i + 1, len(prompt_tokens) + i + 2])
                elif i + 2 < len(generated_tokens) and generated_tokens[i:i+3] == ['9', '.', '11']:
                    important_positions.extend([len(prompt_tokens) + i, len(prompt_tokens) + i + 1, len(prompt_tokens) + i + 2])
                elif any(word in token.lower() for word in ['greater', 'bigger', 'larger', 'than', 'is']):
                    important_positions.append(len(prompt_tokens) + i)
            
            # Save to file
            with open("neurons_qa_CORRECT_temp02.txt", "w") as f:
                f.write("🔍 Neurons for CORRECT Answer with temp=0.2 (matching verify_llama_bug.py)\n")
                f.write("="*60 + "\n")
                f.write(f"Generated text: {generated}\n")
                f.write(f"Important positions: {important_positions}\n\n")
                
                # Analyze neurons
                all_top_neurons = []
                
                for layer_idx in sorted(layer_activations.keys()):
                    activations = layer_activations[layer_idx]
                    layer_act = torch.cat(activations, dim=1)
                    
                    for pos in important_positions:
                        if pos < layer_act.shape[1]:
                            pos_activations = layer_act[0, pos, :]
                            top_values, top_indices = torch.topk(pos_activations, k=min(10, pos_activations.shape[0]))
                            
                            for rank, (neuron_idx, value) in enumerate(zip(top_indices, top_values)):
                                all_top_neurons.append({
                                    'layer': layer_idx,
                                    'neuron': neuron_idx.item(),
                                    'activation': value.item(),
                                    'position': pos,
                                    'token': generated_tokens[pos - len(prompt_tokens)] if pos >= len(prompt_tokens) else prompt_tokens[pos]
                                })
                
                # Sort and get top 10
                all_top_neurons.sort(key=lambda x: x['activation'], reverse=True)
                top_10_neurons = all_top_neurons[:10]
                
                f.write("\nTop 10 neurons by activation strength:\n")
                f.write(f"{'Rank':<6} {'Layer':<8} {'Neuron':<10} {'Activation':<12} {'Token':<15} {'Position'}\n")
                f.write("-" * 70 + "\n")
                
                for rank, neuron_info in enumerate(top_10_neurons, 1):
                    f.write(f"{rank:<6} {neuron_info['layer']:<8} {neuron_info['neuron']:<10} "
                            f"{neuron_info['activation']:<12.4f} {neuron_info['token']:<15} {neuron_info['position']}\n")
            
            print(f"Neurons saved to: neurons_qa_CORRECT_temp02.txt")
            
    elif says_9_11_bigger and not says_9_8_bigger:
        results["incorrect"] += 1
        symbol = "❌"
    else:
        results["unclear"] += 1
        symbol = "❓"
    
    print(f"Attempt {attempt:2d}: {symbol} {generated[:50].strip()}...")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY:")
print("="*60)
print(f"Correct:   {results['correct']}/20 ({results['correct']*5}%)")
print(f"Incorrect: {results['incorrect']}/20 ({results['incorrect']*5}%)")
print(f"Unclear:   {results['unclear']}/20 ({results['unclear']*5}%)")

if results['correct'] > 0:
    print(f"\n✅ Success rate matches verify_llama_bug.py better!")
else:
    print(f"\n⚠️ No correct answers found - may need more attempts")