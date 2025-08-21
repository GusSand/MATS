#!/usr/bin/env python
"""
Manually load SAE from HuggingFace and run analysis
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import json
from safetensors.torch import load_file
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("🔬 Manual SAE Loading and Analysis")
print("=" * 70)

# Download SAE files manually
print("\nDownloading SAE files from HuggingFace...")
repo_id = "EleutherAI/sae-llama-3.1-8b-32x"
layer = 23

# Download config
cfg_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"layers.{layer}.mlp/cfg.json"
)
print(f"Config downloaded: {cfg_path}")

# Load config
with open(cfg_path, 'r') as f:
    sae_cfg = json.load(f)
print(f"SAE config: d_in={sae_cfg['d_in']}, d_sae={sae_cfg['d_sae']}")

# Download weights
weights_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"layers.{layer}.mlp/sae.safetensors"
)
print(f"Weights downloaded: {weights_path}")

# Load weights
sae_state_dict = load_file(weights_path)
print(f"Loaded weights: {list(sae_state_dict.keys())}")

# Build SAE manually
class SimpleSAE(torch.nn.Module):
    def __init__(self, cfg, state_dict):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg['d_in']
        self.d_sae = cfg['d_sae']
        
        # Load weights
        self.W_enc = torch.nn.Parameter(state_dict['W_enc'])
        self.b_enc = torch.nn.Parameter(state_dict['b_enc'])
        self.W_dec = torch.nn.Parameter(state_dict['W_dec'])
        self.b_dec = torch.nn.Parameter(state_dict['b_dec'])
        
        print(f"Encoder shape: {self.W_enc.shape}")
        print(f"Decoder shape: {self.W_dec.shape}")
    
    def encode(self, x):
        # x: [batch, d_in]
        # Standard SAE encoding: ReLU(x @ W_enc + b_enc)
        pre_acts = torch.matmul(x, self.W_enc) + self.b_enc
        return torch.relu(pre_acts)
    
    def decode(self, f):
        # f: [batch, d_sae]
        # Standard SAE decoding: f @ W_dec + b_dec
        return torch.matmul(f, self.W_dec) + self.b_dec

# Create SAE
device = "cuda" if torch.cuda.is_available() else "cpu"
sae = SimpleSAE(sae_cfg, sae_state_dict).to(device)
sae.eval()

print(f"\n✓ SAE built successfully!")
print(f"  Input dim: {sae.d_in}")
print(f"  SAE features: {sae.d_sae}")
print(f"  Expansion: {sae.d_sae / sae.d_in:.1f}x")

# Load model
print(f"\nLoading Llama model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model.eval()

# Define prompts
PROMPT_BAD = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
PROMPT_GOOD = "Which is bigger: 9.8 or 9.11?\nAnswer:"

def analyze_prompt(prompt, label):
    """Analyze SAE features for a prompt"""
    # Prepare prompt
    if isinstance(prompt, list):
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prompt
    
    print(f"\n{label}: '{prompt_text[:40]}...'")
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Capture MLP output
    mlp_out = None
    def hook_fn(module, input, output):
        nonlocal mlp_out
        mlp_out = output.detach()
    
    hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    hook.remove()
    
    # Get last token activations
    last_token = mlp_out[0, -1, :].unsqueeze(0).to(device)
    
    # Encode with SAE
    with torch.no_grad():
        features = sae.encode(last_token)
    
    features = features.squeeze(0)
    
    # Get top features
    nonzero = features > 0.01
    if nonzero.sum() > 0:
        active_features = features[nonzero]
        active_indices = torch.where(nonzero)[0]
        
        sorted_acts, sorted_idx = torch.sort(active_features, descending=True)
        
        top_features = []
        for i in range(min(20, len(sorted_acts))):
            feat_idx = active_indices[sorted_idx[i]].item()
            activation = sorted_acts[i].item()
            top_features.append((feat_idx, activation))
        
        print(f"Active features: {nonzero.sum().item()}")
        print(f"Top 10 features:")
        for i, (idx, act) in enumerate(top_features[:10]):
            print(f"  {i+1:2d}. Feature {idx:5d}: {act:.4f}")
        
        return top_features
    else:
        print("No active features!")
        return []

# Analyze both prompts
print("\n" + "="*70)
print("COMPARATIVE ANALYSIS")
print("="*70)

bad_features = analyze_prompt(PROMPT_BAD, "Bad state (chat)")
good_features = analyze_prompt(PROMPT_GOOD, "Good state (simple)")

# Compare
bad_set = {f[0] for f in bad_features}
good_set = {f[0] for f in good_features}
shared = bad_set.intersection(good_set)

print(f"\n" + "="*70)
print("ENTANGLEMENT ANALYSIS")
print(f"="*70)
print(f"Shared features: {len(shared)} out of {len(bad_set)} and {len(good_set)}")

if shared:
    print(f"\nShared feature indices: {sorted(list(shared))[:15]}")
    
    # Check amplification
    bad_dict = dict(bad_features)
    good_dict = dict(good_features)
    
    print("\nAmplification analysis:")
    amplifications = []
    for feat in shared:
        if feat in bad_dict and feat in good_dict:
            ratio = bad_dict[feat] / good_dict[feat] if good_dict[feat] > 0 else 0
            amplifications.append((feat, bad_dict[feat], good_dict[feat], ratio))
    
    amplifications.sort(key=lambda x: x[3], reverse=True)
    
    print("Top amplified features:")
    for feat, bad_act, good_act, ratio in amplifications[:10]:
        print(f"  Feature {feat:5d}: {ratio:5.2f}x (bad: {bad_act:.3f}, good: {good_act:.3f})")

print("\n✅ Analysis complete!")