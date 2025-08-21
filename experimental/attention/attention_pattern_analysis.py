#!/usr/bin/env python3
"""
Analyze attention patterns to understand the discrepancy
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_attention_patterns():
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Test prompts
    correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"
    buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
    
    print("\nTokenizing prompts...")
    correct_tokens = tokenizer(correct_prompt, return_tensors="pt").to(device)
    buggy_tokens = tokenizer(buggy_prompt, return_tensors="pt").to(device)
    
    print(f"Correct format: {correct_tokens.input_ids.shape[1]} tokens")
    print(f"Buggy format: {buggy_tokens.input_ids.shape[1]} tokens")
    
    # Decode tokens to see what they are
    print("\nCorrect format tokens:")
    for i, token_id in enumerate(correct_tokens.input_ids[0]):
        token = tokenizer.decode([token_id])
        print(f"  {i:2d}: '{token}'")
    
    print("\nBuggy format tokens:")
    for i, token_id in enumerate(buggy_tokens.input_ids[0]):
        token = tokenizer.decode([token_id])
        print(f"  {i:2d}: '{token}'")
    
    # Find where the key numbers appear
    print("\nKey token positions:")
    correct_text = tokenizer.decode(correct_tokens.input_ids[0])
    buggy_text = tokenizer.decode(buggy_tokens.input_ids[0])
    
    # Get attention patterns with output_attentions
    print("\nExtracting attention patterns...")
    with torch.no_grad():
        correct_out = model(**correct_tokens, output_attentions=True)
        buggy_out = model(**buggy_tokens, output_attentions=True)
    
    # Analyze Layer 10 attention
    layer_10_correct = correct_out.attentions[10][0]  # Shape: (heads, seq, seq)
    layer_10_buggy = buggy_out.attentions[10][0]
    
    print(f"\nLayer 10 attention shapes:")
    print(f"  Correct: {layer_10_correct.shape}")
    print(f"  Buggy: {layer_10_buggy.shape}")
    
    # Average over heads and look at last token attention
    correct_last_attn = layer_10_correct.mean(0)[-1, :].cpu().numpy()
    buggy_last_attn = layer_10_buggy.mean(0)[-1, :].cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot attention from last token
    ax1 = axes[0, 0]
    ax1.bar(range(len(correct_last_attn)), correct_last_attn)
    ax1.set_title('Correct Format: Last Token Attention (Layer 10)')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Attention Weight')
    
    ax2 = axes[0, 1]
    ax2.bar(range(len(buggy_last_attn)), buggy_last_attn)
    ax2.set_title('Buggy Format: Last Token Attention (Layer 10)')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Attention Weight')
    
    # Plot full attention matrices
    ax3 = axes[1, 0]
    sns.heatmap(layer_10_correct.mean(0).cpu().numpy(), ax=ax3, cmap='Blues')
    ax3.set_title('Correct Format: Full Attention Matrix (Layer 10)')
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Query Position')
    
    ax4 = axes[1, 1]
    sns.heatmap(layer_10_buggy.mean(0).cpu().numpy(), ax=ax4, cmap='Reds')
    ax4.set_title('Buggy Format: Full Attention Matrix (Layer 10)')
    ax4.set_xlabel('Key Position')
    ax4.set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig('attention_pattern_comparison.png', dpi=150)
    print("\n✓ Saved visualization to attention_pattern_comparison.png")
    
    # Analyze attention to BEGIN token
    print("\nAttention to position 0 (BEGIN):")
    print(f"  Correct format: {correct_last_attn[0]:.4f}")
    print(f"  Buggy format: {buggy_last_attn[0]:.4f}")
    
    return {
        'correct_tokens': correct_tokens.input_ids.shape[1],
        'buggy_tokens': buggy_tokens.input_ids.shape[1],
        'begin_attn_correct': float(correct_last_attn[0]),
        'begin_attn_buggy': float(buggy_last_attn[0])
    }

if __name__ == "__main__":
    results = analyze_attention_patterns()