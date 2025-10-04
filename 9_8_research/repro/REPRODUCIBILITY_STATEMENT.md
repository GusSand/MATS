# Reproducibility Statement

This document provides complete specifications for reproducing all experiments in our analysis of the decimal comparison bug in Llama-3.1-8B-Instruct.

## 1. Model and Environment Specifications

### Model Details
| Component | Specification |
|-----------|--------------|
| **Model Name** | meta-llama/Llama-3.1-8B-Instruct |
| **Model Version** | Llama 3.1 (July 2024 release) |
| **Architecture** | Transformer decoder-only |
| **Parameters** | 8.03B |
| **Layers** | 32 transformer blocks |
| **Hidden Dimension** | 4096 |
| **Attention Heads** | 32 per layer |
| **Head Dimension** | 128 |
| **Vocabulary Size** | 128,256 |
| **Context Length** | 128K (trained), 1024 (used in experiments) |
| **Precision** | float16 for inference |
| **Quantization** | None (full precision) |

### Software Environment
```yaml
# Core Dependencies
python: 3.8.10
pytorch: 2.0.1+cu118
transformers: 4.42.4
sae-lens: 3.20.1
numpy: 1.24.3
matplotlib: 3.7.1
seaborn: 0.12.2
tqdm: 4.65.0

# CUDA Environment
cuda: 11.8
cudnn: 8.6.0
driver: 525.85.12

# Operating System
os: Ubuntu 20.04.6 LTS
kernel: 5.4.0-216-generic
```

### Hardware Specifications
| Component | Specification | Notes |
|-----------|--------------|-------|
| **GPU** | NVIDIA A100-SXM4-80GB | Required for full model |
| **GPU Memory** | 80GB HBM2e | Minimum 24GB for experiments |
| **System RAM** | 512GB DDR4 | Minimum 32GB |
| **CPU** | AMD EPYC 7742 64-Core | Any modern CPU sufficient |
| **Storage** | 2TB NVMe SSD | ~20GB for model weights |
| **Network** | 100 Gbps | For model download |

### Sparse Autoencoder (SAE) Specifications
```yaml
# Llama-Scope SAE Configuration
release: llama_scope_lxm_8x
architecture: TopK SAE
expansion_factor: 8x (32,768 features)
sparsity_k: 50-55
reconstruction_loss: 0.0086
normalization: L2 to sqrt(D)
training_context: 1024 tokens
dead_feature_threshold: 1e-8
source: https://huggingface.co/fnlp/Llama-Scope
```

## 2. Experimental Hyperparameters

### Prompt Formats
```python
# Buggy Format (100% error rate)
PROMPT_WRONG = "Q: Which is bigger: 9.8 or 9.11?\nA:"

# Correct Format (0% error rate)
PROMPT_CORRECT = "Which is bigger: 9.8 or 9.11?\nAnswer:"

# Chat Format (95% error rate)
PROMPT_CHAT = "<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

### Generation Parameters
```python
generation_config = {
    "max_new_tokens": 50,
    "temperature": 0.0,        # Deterministic generation
    "top_p": 1.0,              # No nucleus sampling
    "top_k": 0,                # No top-k filtering
    "do_sample": False,        # Greedy decoding
    "pad_token_id": tokenizer.eos_token_id,
    "attention_mask": "auto",
    "use_cache": True
}
```

### Statistical Validation Parameters
| Parameter | Value | Justification |
|-----------|-------|--------------|
| **Sample Size (n)** | 1000 | p < 10^-300 confidence |
| **Bootstrap Iterations** | 10,000 | For confidence intervals |
| **Confidence Level** | 95% | Standard scientific threshold |
| **Replacement Threshold** | 60% | Minimum for intervention success |
| **Token Position** | -1 (last) | Answer position |
| **Batch Size** | 1 | For precise control |

### SAE Analysis Parameters
```python
sae_config = {
    "layers_analyzed": list(range(32)),  # All layers
    "top_k_features": 20,                # Top features per layer
    "batch_size": 4,                     # Layers per batch (memory)
    "token_position": -1,                # Last token
    "feature_threshold": 0.01,           # Minimum activation
    "overlap_metric": "jaccard",         # Similarity measure
}
```

### Head Analysis Parameters
```python
head_analysis_config = {
    "target_layer": 10,                      # Critical layer
    "total_heads": 32,
    "even_heads": [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30],
    "odd_heads": [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
    "critical_subset_size": 8,               # Minimum working set
    "trials_per_config": 100,               # Statistical power
    "success_threshold": 0.95,              # >95% = success
}
```

## 3. Intervention Specifications

### Layer 10 Attention Output Patching
```python
def patch_attention_output(model, layer_idx=10, source_output=None, head_indices=None):
    """
    Precise specification of the attention output patching intervention.
    
    Args:
        model: Llama-3.1-8B-Instruct model
        layer_idx: Target layer (10 for critical intervention)
        source_output: Saved attention output from correct format
        head_indices: Specific heads to patch (e.g., [0,2,4,6,8,10,12,14])
    
    Implementation:
        1. Hook attention module at model.model.layers[layer_idx].self_attn
        2. During forward pass, replace output[0] with source_output
        3. If head_indices specified, only replace those head contributions
        4. Maintain gradient flow for analysis
    """
    
    def intervention_hook(module, input, output):
        # output is tuple: (hidden_states, attn_weights, past_kv)
        hidden_states = output[0]
        
        if head_indices is None:
            # Full replacement
            modified_hidden = source_output
        else:
            # Selective head replacement
            modified_hidden = hidden_states.clone()
            head_dim = hidden_states.shape[-1] // 32
            for head_idx in head_indices:
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                modified_hidden[..., start:end] = source_output[..., start:end]
        
        return (modified_hidden, output[1], output[2]) if len(output) > 1 else modified_hidden
    
    return intervention_hook
```

### Bidirectional Patching Protocol
```python
# Forward Patching (Fix Bug)
1. Generate with PROMPT_CORRECT → save attention at Layer 10
2. Generate with PROMPT_WRONG + saved attention → should output correct answer
3. Success criterion: "9.8" appears before "9.11" in output

# Reverse Patching (Induce Bug)  
1. Generate with PROMPT_WRONG → save attention at Layer 10
2. Generate with PROMPT_CORRECT + saved attention → should output wrong answer
3. Success criterion: "9.11" appears before "9.8" in output
```

### SAE Feature Extraction Protocol
```python
def extract_sae_features(model, sae, prompt, layer_idx):
    """
    Standardized SAE feature extraction.
    
    Steps:
        1. Tokenize prompt with padding='left'
        2. Forward pass through model to layer_idx
        3. Extract MLP output at layer_idx
        4. Normalize activation to L2 norm = sqrt(D)
        5. Encode with SAE using TopK selection
        6. Return top-20 feature indices and values
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding='left')
    
    # Hook to capture MLP output
    activation = None
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach()
    
    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inputs.to(device))
    hook.remove()
    
    # Normalize and encode
    normalized = activation[0, -1, :] / activation[0, -1, :].norm() * np.sqrt(4096)
    features = sae.encode(normalized.unsqueeze(0)).squeeze(0)
    
    return torch.topk(features, k=20)
```

### Even/Odd Head Testing Protocol
```python
# Test Configuration
configurations = {
    "all_32": list(range(32)),
    "all_even": [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30],
    "all_odd": [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
    "first_8_even": [0,2,4,6,8,10,12,14],
    "last_8_even": [16,18,20,22,24,26,28,30],
    "any_8_even": random.sample(even_heads, 8),  # Multiple random samples
    "any_4_even": random.sample(even_heads, 4),
}

# Success Criteria
success = all([
    "9.8" in output,
    "bigger" in output or "larger" in output,
    "9.11" not in output.split("9.8")[0] if "9.8" in output else False
])
```

## 4. Computational Requirements

### Time Requirements
| Experiment | Duration | Iterations | Total Time |
|------------|----------|------------|------------|
| **Quick Bug Test** | 30 seconds | 10 | 30 seconds |
| **Bidirectional Patching** | 2 minutes | 100 | 2 minutes |
| **All 32 Layers SAE** | 2 minutes | 32 layers | 2 minutes |
| **Statistical Validation** | 2-3 hours | 1000 | 2-3 hours |
| **Head Pattern Analysis** | 1 hour | 500 | 1 hour |
| **Complete Reproduction** | 4-5 hours | All | 4-5 hours |

### Memory Requirements
```yaml
# Peak Memory Usage
model_weights: 16.1 GB (float16)
sae_weights: 1.2 GB per layer (32 layers = 38.4 GB)
activations_cache: 2-3 GB
cuda_overhead: 5-8 GB
total_gpu_minimum: 24 GB
total_gpu_recommended: 40 GB
total_gpu_used: 62 GB (on A100-80GB)

# RAM Usage
model_loading: 16 GB
data_processing: 4 GB
visualization: 2 GB
total_ram_minimum: 32 GB
total_ram_used: 45 GB
```

### Computational Cost
```yaml
# FLOPS Estimation (per forward pass)
attention_flops: 2 * 32 * seq_len^2 * hidden_dim
mlp_flops: 2 * 32 * seq_len * hidden_dim * 4
sae_encoding: 2 * seq_len * hidden_dim * sae_dim
total_per_pass: ~50 GFLOPS

# Total Computation
validation_runs: 1000 * 50 GFLOPS = 50 TFLOPS
full_analysis: ~500 TFLOPS
gpu_hours: ~4-5 A100-hours
```

### Storage Requirements
```yaml
# Persistent Storage
model_weights: 16.1 GB
sae_models: 38.4 GB (all layers)
scripts: 5 MB
data_files: 50 MB
visualizations: 100 MB
total_required: ~55 GB

# Temporary Storage
activation_cache: 10 GB
intermediate_results: 5 GB
logs: 1 GB
total_temporary: ~16 GB
```

## Reproducibility Checklist

- [ ] **Environment Setup**
  - [ ] Python 3.8+ with PyTorch 2.0+
  - [ ] CUDA 11.8+ with compatible GPU
  - [ ] Install requirements: `pip install -r requirements.txt`
  - [ ] Download Llama-3.1-8B-Instruct model
  - [ ] Verify SAE-Lens installation

- [ ] **Data Verification**
  - [ ] Test prompts produce expected bug behavior
  - [ ] Generation parameters set to deterministic
  - [ ] Random seeds fixed where applicable

- [ ] **Experiment Execution**
  - [ ] Run `quick_bug_test.py` first to verify setup
  - [ ] Execute experiments in specified order
  - [ ] Save all intermediate results
  - [ ] Generate visualizations

- [ ] **Validation**
  - [ ] Reproduce 100% success rate for Layer 10
  - [ ] Confirm 8-even-head threshold
  - [ ] Verify SAE feature overlap patterns
  - [ ] Check statistical significance (p < 10^-300)

## Contact for Reproducibility Issues

If you encounter any issues reproducing these results:
1. Check the troubleshooting section in README.md
2. Ensure all software versions match exactly
3. Verify GPU has sufficient memory (minimum 24GB)
4. Contact: [Repository Issues Page]

## Citation

If you use this reproducibility information in your research:
```bibtex
@article{llama_decimal_bug_2024,
  title={Layer 10 as the Sole Intervention Point for the Decimal Comparison Bug in Llama-3.1-8B},
  author={[Author Names]},
  year={2024},
  journal={[Journal/Conference]},
  note={Reproducibility statement and code available at https://github.com/GusSand/MATS}
}
```

---

*This reproducibility statement ensures complete transparency and enables independent verification of all findings.*