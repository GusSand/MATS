#!/usr/bin/env python3
"""
Logit lens analysis with deterministic generation (temperature=0.0)
Comparing formats that give CORRECT vs WRONG answers
"""

import torch
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check GPU
if torch.cuda.is_available():
    logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    logger.warning("⚠️ No GPU available - this will be slow!")

# Load model
logger.info("Loading model...")
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
logger.info("✓ Model loaded successfully")

def generate_and_verify(prompt, expected_result):
    """Generate response and verify it matches expected result"""
    # Load a separate model for generation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if not hasattr(generate_and_verify, 'gen_model'):
        logger.info("Loading generation model...")
        generate_and_verify.gen_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        generate_and_verify.gen_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    inputs = generate_and_verify.gen_tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = generate_and_verify.gen_model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,  # DETERMINISTIC
            do_sample=False,  # NO SAMPLING
            pad_token_id=generate_and_verify.gen_tokenizer.eos_token_id
        )
    
    full_response = generate_and_verify.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_response[len(prompt):].strip()
    
    # Check result
    if "9.8" in generated and "bigger" in generated.lower() and "than 9.11" in generated:
        actual = "CORRECT"
    elif "9.11" in generated and "bigger" in generated.lower():
        actual = "WRONG"
    else:
        actual = "UNCLEAR"
    
    logger.info(f"  Expected: {expected_result}, Got: {actual}")
    logger.info(f"  Response: '{generated[:60]}...'")
    
    return actual == expected_result, generated

def logit_lens_analysis(model, prompt, format_name="", expected_result=""):
    """
    Run logit lens analysis showing what token the model predicts at each layer
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"LOGIT LENS ANALYSIS - {format_name}")
    logger.info(f"Expected: {expected_result}")
    logger.info(f"Prompt: {repr(prompt[:80])}...")
    
    # First verify the generation
    matches, generated = generate_and_verify(prompt, expected_result)
    if not matches:
        logger.warning(f"⚠️ Result doesn't match expected! Continuing anyway...")
    
    # Now do the logit lens analysis
    with model.trace(prompt) as tracer:
        # Collect hidden states at each layer
        hidden_states = []
        
        for layer_idx in range(model.config.num_hidden_layers):
            if layer_idx == 0:
                hidden = model.model.layers[layer_idx].input[0][0]
            else:
                hidden = model.model.layers[layer_idx].output[0]
            
            hidden_states.append(hidden.save())
        
        # Also get final hidden state
        final_hidden = model.model.norm.output.save()
        hidden_states.append(final_hidden)
    
    # Decode each hidden state to see what token it predicts
    layer_predictions = []
    layer_probs = []
    
    # Get token IDs for "9.8" and "9.11"
    # These are multi-token: [24, 13, 23] for "9.8" and [24, 13, 806] for "9.11"
    # We'll look at token 24 which is "9"
    token_9 = 24
    token_8 = 23
    token_11 = 806
    
    with torch.no_grad():
        for layer_idx, hidden_state in enumerate(hidden_states):
            # Apply final layernorm and unembedding to get logits
            if layer_idx < len(hidden_states) - 1:
                normalized = model.model.norm(hidden_state.value)
            else:
                normalized = hidden_state.value
            
            # Get logits using the language model head
            logits = model.lm_head(normalized)
            
            # Handle different tensor dimensions
            if len(logits.shape) == 1:
                answer_logits = logits
            elif len(logits.shape) == 2:
                answer_logits = logits[-1, :]
            elif len(logits.shape) == 3:
                answer_logits = logits[0, -1, :]
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            # Get probabilities
            probs = torch.softmax(answer_logits, dim=-1)
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probs, 5)
            top_tokens = [model.tokenizer.decode(idx.item()) for idx in top5_indices]
            
            # Track specific tokens
            prob_9 = probs[token_9].item() if token_9 < len(probs) else 0
            prob_8 = probs[token_8].item() if token_8 < len(probs) else 0
            prob_11 = probs[token_11].item() if token_11 < len(probs) else 0
            
            layer_predictions.append({
                'layer': layer_idx,
                'top_token': top_tokens[0],
                'top5': list(zip(top_tokens, top5_probs.cpu().numpy())),
                'prob_9': prob_9,
                'prob_8': prob_8,
                'prob_11': prob_11
            })
            
            layer_probs.append({
                'layer': layer_idx,
                'prob_9': prob_9,
                'prob_8': prob_8,
                'prob_11': prob_11,
                'format': format_name
            })
    
    return layer_predictions, layer_probs, generated

# Define prompts
logger.info("\n" + "="*60)
logger.info("STARTING DETERMINISTIC LOGIT LENS ANALYSIS")
logger.info("="*60)

# Format that gives WRONG answer (9.11 is bigger)
wrong_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

# Format that gives CORRECT answer (9.8 is bigger)
correct_prompt = "Which is bigger: 9.8 or 9.11?\nAnswer:"

# Run analyses
wrong_predictions, wrong_probs, wrong_response = logit_lens_analysis(
    model, wrong_prompt, "Q&A Format (WRONG)", "WRONG"
)

correct_predictions, correct_probs, correct_response = logit_lens_analysis(
    model, correct_prompt, "Direct Question (CORRECT)", "CORRECT"
)

# Visualization
def plot_logit_lens_comparison(wrong_probs, correct_probs):
    """Create visualization comparing wrong vs correct formats"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layers = [p['layer'] for p in wrong_probs]
    
    # Plot 1: Token "9" probability
    ax = axes[0, 0]
    ax.plot(layers, [p['prob_9'] for p in wrong_probs], 
            label='Wrong Format', color='red', marker='o', alpha=0.7)
    ax.plot(layers, [p['prob_9'] for p in correct_probs], 
            label='Correct Format', color='green', marker='s', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('P(token "9")')
    ax.set_title('Probability of Token "9" Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Token "8" vs "11" comparison
    ax = axes[0, 1]
    wrong_diff = [p['prob_11'] - p['prob_8'] for p in wrong_probs]
    correct_diff = [p['prob_11'] - p['prob_8'] for p in correct_probs]
    
    ax.plot(layers, wrong_diff, label='Wrong Format', color='red', marker='o', alpha=0.7)
    ax.plot(layers, correct_diff, label='Correct Format', color='green', marker='s', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('P(11) - P(8)')
    ax.set_title('Token Preference: Positive = Prefers "11", Negative = Prefers "8"')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Evolution of top tokens (Wrong format)
    ax = axes[1, 0]
    key_layers = [0, 7, 14, 20, 25, 31]
    y_pos = np.arange(len(key_layers))
    
    for i, layer in enumerate(key_layers):
        pred = wrong_predictions[layer]
        ax.text(0.1, i, f"L{layer}: '{pred['top_token'][:15]}'", fontsize=10)
    
    ax.set_ylim(-0.5, len(key_layers)-0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Layer {l}" for l in key_layers])
    ax.set_title('Top Token Evolution - Wrong Format')
    ax.axis('off')
    
    # Plot 4: Evolution of top tokens (Correct format)
    ax = axes[1, 1]
    for i, layer in enumerate(key_layers):
        pred = correct_predictions[layer]
        ax.text(0.1, i, f"L{layer}: '{pred['top_token'][:15]}'", fontsize=10)
    
    ax.set_ylim(-0.5, len(key_layers)-0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Layer {l}" for l in key_layers])
    ax.set_title('Top Token Evolution - Correct Format')
    ax.axis('off')
    
    plt.suptitle('Logit Lens: Comparing Wrong vs Correct Formats (Temperature=0.0)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'logit_lens_deterministic_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Plot saved to {filename}")
    plt.show()

# Create visualization
plot_logit_lens_comparison(wrong_probs, correct_probs)

# Print key findings
logger.info("\n" + "="*60)
logger.info("KEY FINDINGS")
logger.info("="*60)

logger.info("\nWRONG Format (Q&A) - Top predictions by layer:")
for i in [0, 7, 14, 20, 25, 31]:
    pred = wrong_predictions[i]
    logger.info(f"  Layer {pred['layer']:2d}: '{pred['top_token']:15s}' (P(9)={pred['prob_9']:.4f}, P(8)={pred['prob_8']:.4f}, P(11)={pred['prob_11']:.4f})")

logger.info("\nCORRECT Format (Direct) - Top predictions by layer:")
for i in [0, 7, 14, 20, 25, 31]:
    pred = correct_predictions[i]
    logger.info(f"  Layer {pred['layer']:2d}: '{pred['top_token']:15s}' (P(9)={pred['prob_9']:.4f}, P(8)={pred['prob_8']:.4f}, P(11)={pred['prob_11']:.4f})")

# Find divergence point
logger.info("\n" + "="*60)
logger.info("DIVERGENCE ANALYSIS")
logger.info("="*60)

for i in range(len(wrong_predictions)):
    wrong_pref = wrong_probs[i]['prob_11'] - wrong_probs[i]['prob_8']
    correct_pref = correct_probs[i]['prob_11'] - correct_probs[i]['prob_8']
    
    if abs(wrong_pref - correct_pref) > 0.01:  # Significant difference
        logger.info(f"Layer {i}: Divergence detected!")
        logger.info(f"  Wrong format: P(11)-P(8) = {wrong_pref:.4f}")
        logger.info(f"  Correct format: P(11)-P(8) = {correct_pref:.4f}")
        if i <= 15:
            logger.info(f"  → Early divergence at layer {i}")
            break

logger.info("\n✅ Analysis complete!")
logger.info(f"Wrong format response: '{wrong_response[:60]}...'")
logger.info(f"Correct format response: '{correct_response[:60]}...'")