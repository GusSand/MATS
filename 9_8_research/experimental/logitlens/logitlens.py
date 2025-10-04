import torch
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set back to INFO for cleaner output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logitlens_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# GPU availability check
def check_gpu_status():
    """Check and report GPU availability and usage"""
    logger.info("=" * 50)
    logger.info("GPU STATUS CHECK")
    logger.info("=" * 50)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✓ CUDA is available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}")
            logger.info(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"    Current Memory Used: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            logger.info(f"    Peak Memory Used: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")
        
        current_device = torch.cuda.current_device()
        logger.info(f"  Current active device: GPU {current_device}")
        return True
    else:
        logger.warning("✗ CUDA is NOT available - running on CPU")
        logger.warning("  This will be SIGNIFICANTLY slower!")
        return False

# Check GPU status at start
gpu_available = check_gpu_status()

# Load model with GPU tracking
logger.info("\n" + "=" * 50)
logger.info("LOADING MODEL")
logger.info("=" * 50)

start_time = time.time()
logger.info("Loading meta-llama/Llama-3.1-8B-Instruct...")

# Device map configuration for multi-GPU
if gpu_available and torch.cuda.device_count() > 1:
    logger.info(f"Multiple GPUs detected ({torch.cuda.device_count()}). Using automatic device mapping.")
    device_map = "auto"  # This will automatically distribute layers across GPUs
else:
    device_map = "auto"

try:
    model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map=device_map)
    load_time = time.time() - start_time
    logger.info(f"✓ Model loaded successfully in {load_time:.2f} seconds")
    
    # Check where model layers are placed
    if hasattr(model.model, 'hf_device_map'):
        logger.info("\nModel device distribution:")
        device_counts = {}
        for name, device in model.model.hf_device_map.items():
            device_str = str(device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1
        for device, count in device_counts.items():
            logger.info(f"  {count} layers on {device}")
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    raise

def logit_lens_analysis(model, prompt, format_name=""):
    """
    Run logit lens analysis showing what token the model predicts at each layer
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"RUNNING LOGIT LENS ANALYSIS - {format_name.upper()} FORMAT")
    logger.info(f"{'='*50}")
    logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    
    analysis_start = time.time()
    # Token positions we care about (where model should say 9.8 or 9.11)
    # You'll need to adjust based on your exact tokenization
    
    logger.info(f"Starting trace with {model.config.num_hidden_layers} layers...")
    trace_start = time.time()
    
    with model.trace(prompt) as tracer:
        # Collect hidden states at each layer
        hidden_states = []
        
        logger.debug("Collecting hidden states from each layer...")
        for layer_idx in range(model.config.num_hidden_layers):
            # Get residual stream after each layer
            if layer_idx == 0:
                hidden = model.model.layers[layer_idx].input[0][0]
            else:
                hidden = model.model.layers[layer_idx].output[0]
            
            hidden_states.append(hidden.save())
        
        # Also get final hidden state
        final_hidden = model.model.norm.output.save()
        hidden_states.append(final_hidden)
    
    trace_time = time.time() - trace_start
    logger.info(f"✓ Trace completed in {trace_time:.2f} seconds")
    logger.info(f"  Collected {len(hidden_states)} hidden states")
    
    # Now decode each hidden state to see what token it predicts
    layer_predictions = []
    layer_probs = []
    
    logger.info("Decoding predictions at each layer...")
    decode_start = time.time()
    
    # Check memory before decoding
    if gpu_available:
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            logger.debug(f"  GPU {i} memory before decoding: {mem_used:.2f} GB")
    
    with torch.no_grad():
        for layer_idx, hidden_state in enumerate(hidden_states):
            if layer_idx % 8 == 0:  # Log progress every 8 layers
                logger.debug(f"  Processing layer {layer_idx}/{len(hidden_states)-1}...")
            # Apply final layernorm and unembedding to get logits
            if layer_idx < len(hidden_states) - 1:
                # Need to apply final LN for intermediate layers
                normalized = model.model.norm(hidden_state.value)
            else:
                normalized = hidden_state.value
            
            # Get logits using the language model head
            logits = model.lm_head(normalized)
            
            # Log device placement and shape
            if layer_idx == 0:
                logger.debug(f"  Logits shape: {logits.shape}")
                logger.debug(f"  Logits computed on device: {logits.device}")
            
            # Handle different tensor dimensions
            if len(logits.shape) == 1:
                # 1D tensor - already the last position
                answer_logits = logits
            elif len(logits.shape) == 2:
                # 2D tensor [seq_len, vocab_size] - take last position
                answer_logits = logits[-1, :]
            elif len(logits.shape) == 3:
                # 3D tensor [batch, seq_len, vocab_size] - take first batch, last position
                answer_logits = logits[0, -1, :]
            else:
                logger.error(f"Unexpected logits shape: {logits.shape}")
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            # Get top 5 predictions at this layer
            probs = torch.softmax(answer_logits, dim=-1)
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            # Decode the top predictions
            top_tokens = [model.tokenizer.decode(idx.item()) for idx in top5_indices]
            
            layer_predictions.append({
                'layer': layer_idx,
                'top_prediction': top_tokens[0],
                'top5': list(zip(top_tokens, top5_probs.cpu().numpy())),
                'logits_98': answer_logits[model.tokenizer.encode("9.8")[0]].item(),
                'logits_911': answer_logits[model.tokenizer.encode("9.11")[0]].item(),
            })
            
            # Track probability of "9.8" vs "9.11" tokens
            # Get the actual token IDs - need to check different variations
            tokens_98_variations = [
                model.tokenizer.encode("9.8", add_special_tokens=False),
                model.tokenizer.encode(" 9.8", add_special_tokens=False),
                model.tokenizer.encode("9.8", add_special_tokens=False)
            ]
            tokens_911_variations = [
                model.tokenizer.encode("9.11", add_special_tokens=False),
                model.tokenizer.encode(" 9.11", add_special_tokens=False),
                model.tokenizer.encode("9.11", add_special_tokens=False)
            ]
            
            # Log what tokens we're tracking
            if layer_idx == 0:
                logger.info(f"Token variations for 9.8: {tokens_98_variations}")
                logger.info(f"Token variations for 9.11: {tokens_911_variations}")
            
            # For simplicity, use the first token of each
            token_98 = tokens_98_variations[0][0] if tokens_98_variations[0] else 0
            token_911 = tokens_911_variations[0][0] if tokens_911_variations[0] else 0
            
            prob_98 = probs[token_98].item() if token_98 < len(probs) else 0
            prob_911 = probs[token_911].item() if token_911 < len(probs) else 0
            
            layer_probs.append({
                'layer': layer_idx,
                'prob_98': prob_98,
                'prob_911': prob_911,
                'format': format_name
            })
    
    decode_time = time.time() - decode_start
    logger.info(f"✓ Decoding completed in {decode_time:.2f} seconds")
    
    total_time = time.time() - analysis_start
    logger.info(f"✓ Total analysis time: {total_time:.2f} seconds")
    
    # Memory check after analysis
    if gpu_available:
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            peak_mem = torch.cuda.max_memory_allocated(i) / 1024**3
            logger.info(f"  GPU {i} - Current memory: {mem_used:.2f} GB, Peak: {peak_mem:.2f} GB")
    
    return layer_predictions, layer_probs

# Multi-GPU performance assessment
def assess_multi_gpu_potential():
    logger.info("\n" + "=" * 50)
    logger.info("MULTI-GPU ASSESSMENT")
    logger.info("=" * 50)
    
    if not gpu_available:
        logger.warning("No GPU available - multi-GPU not applicable")
        return
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count == 1:
        logger.info("Single GPU system detected.")
        logger.info("Multi-GPU benefits:")
        logger.info("  • Would allow larger batch sizes")
        logger.info("  • Could distribute model layers across GPUs (model parallelism)")
        logger.info("  • Would reduce memory pressure per GPU")
    else:
        logger.info(f"Multi-GPU system with {gpu_count} GPUs detected!")
        logger.info("Current benefits being utilized:")
        logger.info("  ✓ Model layers distributed across GPUs (via device_map='auto')")
        logger.info("  ✓ Reduced memory pressure per GPU")
        logger.info("  ✓ Potential for larger batch processing")
        logger.info("\nFor this logit lens analysis:")
        logger.info("  • Single sequence processing doesn't benefit much from data parallelism")
        logger.info("  • Model parallelism (current setup) is optimal for large models")
        logger.info("  • Consider batch processing multiple prompts for better GPU utilization")

assess_multi_gpu_potential()

# Run on both formats
logger.info("\n" + "=" * 50)
logger.info("STARTING MAIN ANALYSIS")
logger.info("=" * 50)

# Use the tokenizer's chat template for proper formatting
messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
chat_prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

simple_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

logger.info(f"\nChat prompt format: {repr(chat_prompt[:100])}...")
logger.info(f"Simple prompt format: {repr(simple_prompt)}")

# Analyze both
main_start = time.time()
chat_predictions, chat_probs = logit_lens_analysis(model, chat_prompt, "chat")
simple_predictions, simple_probs = logit_lens_analysis(model, simple_prompt, "simple")
logger.info(f"\n✓ Both analyses completed in {time.time() - main_start:.2f} seconds total")

# Visualization
def plot_logit_lens_results(chat_probs, simple_probs):
    """
    Create a visualization showing how model's preference evolves through layers
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    layers = [p['layer'] for p in chat_probs]
    
    # Plot 1: Probability evolution
    ax1.plot(layers, [p['prob_98'] for p in chat_probs], 
             label='Chat: P(9.8)', color='red', linestyle='--', marker='o')
    ax1.plot(layers, [p['prob_911'] for p in chat_probs], 
             label='Chat: P(9.11)', color='red', linestyle='-', marker='o')
    
    ax1.plot(layers, [p['prob_98'] for p in simple_probs], 
             label='Simple: P(9.8)', color='blue', linestyle='--', marker='s')
    ax1.plot(layers, [p['prob_911'] for p in simple_probs], 
             label='Simple: P(9.11)', color='blue', linestyle='-', marker='s')
    
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Probability')
    ax1.set_title('Logit Lens: Probability of 9.8 vs 9.11 Across Layers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Logit difference
    chat_diff = [p['logits_911'] - p['logits_98'] for p in chat_predictions]
    simple_diff = [p['logits_911'] - p['logits_98'] for p in simple_predictions]
    
    ax2.plot(layers, chat_diff, label='Chat Format', color='red', marker='o')
    ax2.plot(layers, simple_diff, label='Simple Format', color='blue', marker='s')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.fill_between([7, 15], -10, 10, alpha=0.2, color='yellow', 
                     label='Hijacker Circuit Layers')
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Logit Difference (9.11 - 9.8)')
    ax2.set_title('Where the Bug Emerges: Logit Difference Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logit_lens_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# Create the plot
logger.info("\n" + "=" * 50)
logger.info("GENERATING VISUALIZATION")
logger.info("=" * 50)
plot_start = time.time()
plot_logit_lens_results(chat_probs, simple_probs)
logger.info(f"✓ Plot saved to logit_lens_analysis.png in {time.time() - plot_start:.2f} seconds")

# Print key transitions
logger.info("\n" + "=" * 50)
logger.info("KEY FINDINGS")
logger.info("=" * 50)
print("\n=== KEY FINDINGS ===")
print("\nChat Format - Top prediction by layer:")
for i in [0, 5, 10, 15, 20, 25, 31]:
    pred = chat_predictions[i]
    print(f"Layer {pred['layer']:2d}: '{pred['top_prediction']}' "
          f"(9.8 logit: {pred['logits_98']:.2f}, 9.11 logit: {pred['logits_911']:.2f})")

print("\nSimple Format - Top prediction by layer:")
for i in [0, 5, 10, 15, 20, 25, 31]:
    pred = simple_predictions[i]
    print(f"Layer {pred['layer']:2d}: '{pred['top_prediction']}' "
          f"(9.8 logit: {pred['logits_98']:.2f}, 9.11 logit: {pred['logits_911']:.2f})")

# Find where divergence happens
def find_divergence_layer(chat_preds, simple_preds):
    """Find the layer where chat and simple formats diverge"""
    for i, (c, s) in enumerate(zip(chat_preds, simple_preds)):
        diff_chat = c['logits_911'] - c['logits_98']
        diff_simple = s['logits_911'] - s['logits_98']
        
        if diff_chat > 0 and diff_simple < 0:  # Chat prefers 9.11, Simple prefers 9.8
            print(f"\n🎯 DIVERGENCE at Layer {i}!")
            print(f"   Chat: 9.11 leads by {diff_chat:.2f} logits")
            print(f"   Simple: 9.8 leads by {-diff_simple:.2f} logits")
            return i
    return -1

divergence_layer = find_divergence_layer(chat_predictions, simple_predictions)

# Final summary
logger.info("\n" + "=" * 50)
logger.info("EXECUTION SUMMARY")
logger.info("=" * 50)
logger.info(f"✓ Script completed successfully")
logger.info(f"✓ GPU used: {gpu_available}")
if gpu_available:
    logger.info(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        final_mem = torch.cuda.memory_allocated(i) / 1024**3
        peak_mem = torch.cuda.max_memory_allocated(i) / 1024**3
        logger.info(f"  GPU {i} - Final memory: {final_mem:.2f} GB, Peak: {peak_mem:.2f} GB")
logger.info(f"✓ Log file saved to: logitlens_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")