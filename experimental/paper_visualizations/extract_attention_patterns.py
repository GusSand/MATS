#!/usr/bin/env python3
"""
Extract actual attention patterns from Llama model for visualization
This script captures real attention weights from different prompt formats
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class AttentionExtractor:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="eager"  # Needed to access attention weights
        )
        self.model.eval()
        
        self.attention_weights = {}
        self.hooks = []
        
    def save_attention_hook(self, layer_idx):
        """Hook to save attention weights"""
        def hook_fn(module, input, output):
            # output is (hidden_states, attention_weights) for attention modules
            if len(output) > 1 and output[1] is not None:
                # Average over heads and batch dimension
                attn = output[1].detach().cpu().numpy()
                # Shape: [batch, num_heads, seq_len, seq_len]
                attn_mean = attn.mean(axis=(0, 1))  # Average over batch and heads
                self.attention_weights[f"layer_{layer_idx}"] = attn_mean
            return output
        return hook_fn
    
    def register_hooks(self, layers=[8, 9, 10, 11, 12]):
        """Register hooks on specified layers"""
        for layer_idx in layers:
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        self.save_attention_hook(layer_idx)
                    )
                    self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}
    
    def extract_attention_patterns(self, prompt, max_length=50):
        """Extract attention patterns for a given prompt"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Register hooks
        self.register_hooks()
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention from all layers if available
        if hasattr(outputs, 'attentions') and outputs.attentions:
            all_attention = []
            for layer_idx, attn in enumerate(outputs.attentions):
                if layer_idx in [8, 9, 10, 11, 12]:
                    # Average over heads
                    attn_avg = attn[0].mean(dim=0).cpu().numpy()
                    all_attention.append({
                        'layer': layer_idx,
                        'attention': attn_avg.tolist(),
                        'shape': attn_avg.shape
                    })
        
        # Clear hooks
        self.clear_hooks()
        
        return {
            'prompt': prompt,
            'tokens': tokens,
            'attention_patterns': all_attention if 'all_attention' in locals() else self.attention_weights
        }
    
    def compare_formats(self):
        """Compare attention patterns across different formats"""
        test_cases = [
            ("simple", "Which is bigger: 9.8 or 9.11? Answer:"),
            ("qa", "Q: Which is bigger: 9.8 or 9.11? A:"),
            ("chat", "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        ]
        
        results = {}
        
        for format_name, prompt in test_cases:
            print(f"\nExtracting attention for {format_name} format...")
            pattern_data = self.extract_attention_patterns(prompt)
            results[format_name] = pattern_data
            
            # Calculate statistics
            if pattern_data['attention_patterns']:
                for layer_data in pattern_data['attention_patterns']:
                    if isinstance(layer_data, dict) and 'attention' in layer_data:
                        attn = np.array(layer_data['attention'])
                        
                        # Find where decimal tokens are
                        tokens = pattern_data['tokens']
                        decimal_indices = []
                        for i, token in enumerate(tokens):
                            if '9' in token or '8' in token or '11' in token or '.' in token:
                                decimal_indices.append(i)
                        
                        if decimal_indices:
                            # Calculate attention to decimal tokens
                            decimal_attention = attn[:, decimal_indices].mean()
                            layer_data['decimal_attention_score'] = float(decimal_attention)
                            
                        # Calculate entropy
                        attn_flat = attn.flatten()
                        attn_flat = attn_flat[attn_flat > 0]  # Remove zeros
                        if len(attn_flat) > 0:
                            entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
                            layer_data['entropy'] = float(entropy)
        
        return results

def main():
    # Initialize extractor
    extractor = AttentionExtractor()
    
    # Extract patterns
    print("Extracting attention patterns from model...")
    results = extractor.compare_formats()
    
    # Save results
    with open('attention_patterns_data.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=2)
    
    print("\nAttention patterns saved to attention_patterns_data.json")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("ATTENTION PATTERN SUMMARY")
    print("="*50)
    
    for format_name in results:
        print(f"\n{format_name.upper()} FORMAT:")
        if 'attention_patterns' in results[format_name]:
            for layer_data in results[format_name]['attention_patterns']:
                if isinstance(layer_data, dict):
                    layer = layer_data.get('layer', '?')
                    decimal_score = layer_data.get('decimal_attention_score', 0)
                    entropy = layer_data.get('entropy', 0)
                    print(f"  Layer {layer}: Decimal attention={decimal_score:.3f}, Entropy={entropy:.3f}")
    
    return results

if __name__ == "__main__":
    results = main()