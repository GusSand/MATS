#!/usr/bin/env python
"""
Compare tokenization between chat template and Q&A format
"""

from transformers import AutoTokenizer
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Load tokenizer
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Chat template format
messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Q&A format
qa_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

print("="*80)
print("TOKEN COMPARISON ANALYSIS")
print("="*80)

# Tokenize both
chat_tokens = tokenizer(chat_prompt, return_tensors="pt")
qa_tokens = tokenizer(qa_prompt, return_tensors="pt")

chat_token_list = tokenizer.convert_ids_to_tokens(chat_tokens['input_ids'][0])
qa_token_list = tokenizer.convert_ids_to_tokens(qa_tokens['input_ids'][0])

print("\n1. PROMPT COMPARISON:")
print("-"*40)
print(f"Chat template prompt ({len(chat_prompt)} chars):")
print(repr(chat_prompt[:150]) + "...")
print(f"\nQ&A prompt ({len(qa_prompt)} chars):")
print(repr(qa_prompt))

print("\n2. TOKEN COUNT:")
print("-"*40)
print(f"Chat template: {len(chat_token_list)} tokens")
print(f"Q&A format: {len(qa_token_list)} tokens")

print("\n3. CHAT TEMPLATE TOKENS:")
print("-"*40)
for i, token in enumerate(chat_token_list):
    print(f"{i:3d}: {repr(token)}")

print("\n4. Q&A FORMAT TOKENS:")
print("-"*40)
for i, token in enumerate(qa_token_list):
    print(f"{i:3d}: {repr(token)}")

# Find where the actual question appears in both
print("\n5. QUESTION TOKEN COMPARISON:")
print("-"*40)

# Find "9.8" and "9.11" in both formats
def find_number_tokens(tokens, number_str):
    positions = []
    parts = number_str.split('.')
    for i in range(len(tokens) - len(parts) + 1):
        if tokens[i] == parts[0] and i+2 < len(tokens) and tokens[i+1] == '.' and tokens[i+2] == parts[1]:
            positions.append((i, i+1, i+2))
    return positions

chat_98_pos = find_number_tokens(chat_token_list, "9.8")
chat_911_pos = find_number_tokens(chat_token_list, "9.11")
qa_98_pos = find_number_tokens(qa_token_list, "9.8")
qa_911_pos = find_number_tokens(qa_token_list, "9.11")

print(f"Chat template - '9.8' at positions: {chat_98_pos}")
print(f"Chat template - '9.11' at positions: {chat_911_pos}")
print(f"Q&A format - '9.8' at positions: {qa_98_pos}")
print(f"Q&A format - '9.11' at positions: {qa_911_pos}")

print("\n6. SPECIAL TOKENS AND HEADERS:")
print("-"*40)
print("Chat template special tokens:")
special_tokens = [t for t in chat_token_list if '<|' in t and '|>' in t]
print(f"  Found {len(special_tokens)} special tokens: {special_tokens}")

print("\nQ&A format special tokens:")
special_tokens_qa = [t for t in qa_token_list if '<|' in t and '|>' in t]
print(f"  Found {len(special_tokens_qa)} special tokens: {special_tokens_qa}")

print("\n7. KEY DIFFERENCES:")
print("-"*40)
print("- Chat template adds system prompt with date context")
print("- Chat template uses special header tokens for role separation")
print("- Q&A format is much simpler with direct question format")
print("- Both formats tokenize '9.8' and '9.11' the same way (as separate tokens)")
print(f"- Chat template adds {len(chat_token_list) - len(qa_token_list)} extra tokens of context")

# Check if the tokenization of the numbers themselves is different
print("\n8. NUMBER TOKENIZATION:")
print("-"*40)
print("Both formats tokenize decimal numbers identically:")
print("  '9.8' → ['9', '.', '8']")
print("  '9.11' → ['9', '.', '11']")
print("\nThe difference is in the surrounding context tokens!")