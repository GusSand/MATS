#!/usr/bin/env python
"""
Run Q&A format multiple times and record neurons when it gives the CORRECT answer
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔍 Finding Correct Answer Neurons in Q&A Format")
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

# Q&A format prompt
prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

print(f"\nPrompt: {repr(prompt)}")

# Try multiple times to find correct answer
max_attempts = 50
correct_found = False
attempt = 0

while attempt < max_attempts and not correct_found:
    attempt += 1
    
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
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate with temperature for variation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,  # Higher temperature for more variation
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Get generated text
    generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Check if answer is correct
    generated_lower = generated_text.lower()
    
    # Look for clear indicators of which number is considered bigger
    says_9_8_bigger = (
        ("9.8" in generated_text and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.11" in generated_lower) or
        ("larger than 9.11" in generated_lower) or
        ("greater than 9.11" in generated_lower)
    )
    
    says_9_11_bigger = (
        ("9.11" in generated_text and any(w in generated_lower for w in ["bigger", "larger", "greater"])) or
        ("bigger than 9.8" in generated_lower) or
        ("larger than 9.8" in generated_lower) or
        ("greater than 9.8" in generated_lower)
    )
    
    # More sophisticated check for first clear statement
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
    
    print(f"\nAttempt {attempt}: {generated_text[:60]}...")
    
    if says_9_8_bigger and not says_9_11_bigger:
        print("✅ CORRECT ANSWER FOUND!")
        correct_found = True
        
        # Now analyze the neurons for this correct answer
        prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        
        print(f"\nGenerated tokens: {generated_tokens[:20]}...")
        
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
        with open("neurons_qa_format_CORRECT_answer.txt", "w") as f:
            f.write("🔍 Neurons for CORRECT Answer (9.8 is bigger)\n")
            f.write("="*60 + "\n")
            f.write(f"Generated text: {generated_text}\n")
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
            
            # Top neurons per layer
            f.write("\n" + "="*60 + "\n")
            f.write("TOP NEURONS PER LAYER (Top 5 layers):\n")
            f.write("="*60 + "\n")
            
            layer_grouped = defaultdict(list)
            for neuron in all_top_neurons:
                layer_grouped[neuron['layer']].append(neuron)
            
            active_layers = sorted(layer_grouped.keys(), 
                                  key=lambda l: max(n['activation'] for n in layer_grouped[l]), 
                                  reverse=True)[:5]
            
            for layer_idx in active_layers:
                f.write(f"\nLayer {layer_idx}:\n")
                layer_neurons = sorted(layer_grouped[layer_idx], key=lambda x: x['activation'], reverse=True)[:5]
                for neuron_info in layer_neurons:
                    f.write(f"  Neuron {neuron_info['neuron']:<8} Activation: {neuron_info['activation']:<10.4f} "
                            f"Token: {neuron_info['token']:<15} Pos: {neuron_info['position']}\n")
        
        print(f"✅ Correct answer neurons saved to: neurons_qa_format_CORRECT_answer.txt")
        
    elif says_9_11_bigger and not says_9_8_bigger:
        print("❌ Incorrect answer (9.11 is bigger)")
    else:
        print("❓ Unclear answer")

if not correct_found:
    print(f"\n⚠️ Could not find correct answer in {max_attempts} attempts!")
    print("The Q&A format may be consistently producing the wrong answer with these settings.")