#!/home/paperspace/dev/MATS/9_8_research/experiments/sae_env/bin/python
"""
SAE Analysis for Security Decision

Use Sparse Autoencoders to decompose the security decision into interpretable features.
Goal: Validate distributed hypothesis - are there specific "security features" or
is the signal distributed across many features?

Uses pretrained Llama-Scope SAEs if available, otherwise trains small SAEs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Prompts
SECURE_PROMPT = '''// WARNING: Use snprintf to prevent buffer overflows
int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''

NEUTRAL_PROMPT = '''int list_item_to_string(node *head, char *str, size_t str_size) {
    if (head == NULL) return EXIT_FAILURE;
    '''


def check_sae_availability():
    """Check if pretrained Llama-Scope SAEs are available."""
    try:
        from sae_lens import SAE
        print("sae_lens is available")

        # Try loading a test SAE
        test_layer = 16
        sae_id = f"l{test_layer}r_8x"

        print(f"Attempting to load residual SAE for layer {test_layer}...")
        sae = SAE.from_pretrained(
            release="llama_scope_lxr_8x",
            sae_id=sae_id,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"SAE loaded successfully!")
        if hasattr(sae, 'cfg'):
            d_in = getattr(sae.cfg, 'd_in', 'unknown')
            d_sae = getattr(sae.cfg, 'd_sae', 'unknown')
            print(f"  Input dim: {d_in}, SAE features: {d_sae}")

        return True, SAE

    except ImportError:
        print("sae_lens not installed")
        return False, None
    except Exception as e:
        print(f"Failed to load SAE: {e}")
        return False, None


class SAESecurityAnalyzer:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        # Token IDs
        self.snprintf_token = self.tokenizer.encode(" snprintf", add_special_tokens=False)[0]
        self.sprintf_token = self.tokenizer.encode(" sprintf", add_special_tokens=False)[0]

        print(f"Model loaded: {self.n_layers} layers, hidden_size={self.hidden_size}")

        # Storage
        self.residual_stream = {}
        self.hooks = []

        # SAEs
        self.saes = {}

    def load_pretrained_saes(self, layers: list):
        """Load pretrained Llama-Scope SAEs for specified layers."""
        from sae_lens import SAE

        loaded = 0
        for layer in layers:
            sae_id = f"l{layer}r_8x"
            try:
                sae = SAE.from_pretrained(
                    release="llama_scope_lxr_8x",
                    sae_id=sae_id,
                    device=str(self.device)
                )
                self.saes[layer] = sae
                loaded += 1
                print(f"  Layer {layer}: loaded")
            except Exception as e:
                print(f"  Layer {layer}: failed - {str(e)[:50]}")

        print(f"Loaded {loaded}/{len(layers)} SAEs")
        return loaded > 0

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def collect_residual_stream(self, prompt: str, layers: list) -> dict:
        """Collect residual stream at specified layers (last token position)."""
        self.residual_stream = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Store last token activation
                self.residual_stream[layer_idx] = h[:, -1, :].detach().clone()
                return output
            return hook_fn

        # Register hooks only at specified layers
        self.clear_hooks()
        for layer_idx in layers:
            if layer_idx < self.n_layers:
                layer = self.model.model.layers[layer_idx]
                hook = layer.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)

        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get final probabilities
        final_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(final_logits, dim=-1)

        result = {
            'activations': {k: v.cpu() for k, v in self.residual_stream.items()},
            'snprintf_prob': probs[self.snprintf_token].item(),
            'sprintf_prob': probs[self.sprintf_token].item(),
        }

        self.clear_hooks()
        return result

    def encode_with_sae(self, activation: torch.Tensor, layer: int) -> dict:
        """
        Encode activation through SAE and return feature activations.
        """
        if layer not in self.saes:
            return None

        sae = self.saes[layer]

        # Move activation to same device as SAE
        activation = activation.to(next(sae.parameters()).device)

        # Encode through SAE
        with torch.no_grad():
            # SAE encode returns sparse feature activations
            feature_acts = sae.encode(activation)

        return {
            'feature_activations': feature_acts.cpu().numpy(),
            'num_active': (feature_acts > 0).sum().item(),
            'max_activation': feature_acts.max().item(),
            'mean_activation': feature_acts[feature_acts > 0].mean().item() if (feature_acts > 0).any() else 0,
        }

    def analyze_security_features(self, layers: list = None):
        """
        Analyze SAE features that differ between secure and neutral contexts.
        """
        if layers is None:
            layers = list(self.saes.keys())

        print("\nCollecting activations for secure prompt...")
        secure_acts = self.collect_residual_stream(SECURE_PROMPT, layers)

        print("Collecting activations for neutral prompt...")
        neutral_acts = self.collect_residual_stream(NEUTRAL_PROMPT, layers)

        print(f"\nBaseline probabilities:")
        print(f"  Secure P(snprintf):  {secure_acts['snprintf_prob']*100:.2f}%")
        print(f"  Neutral P(snprintf): {neutral_acts['snprintf_prob']*100:.2f}%")

        results = {
            'baseline': {
                'secure_snprintf': secure_acts['snprintf_prob'],
                'neutral_snprintf': neutral_acts['snprintf_prob'],
            },
            'layer_analysis': {}
        }

        print("\n" + "="*70)
        print("SAE FEATURE ANALYSIS BY LAYER")
        print("="*70)

        for layer in sorted(layers):
            if layer not in self.saes:
                continue

            secure_act = secure_acts['activations'].get(layer)
            neutral_act = neutral_acts['activations'].get(layer)

            if secure_act is None or neutral_act is None:
                continue

            # Encode through SAE
            secure_features = self.encode_with_sae(secure_act, layer)
            neutral_features = self.encode_with_sae(neutral_act, layer)

            if secure_features is None or neutral_features is None:
                continue

            # Compare feature activations
            secure_feats = secure_features['feature_activations'].flatten()
            neutral_feats = neutral_features['feature_activations'].flatten()

            # Find features that differ significantly
            diff = secure_feats - neutral_feats

            # Top features more active in secure context
            top_secure_idx = np.argsort(diff)[-10:][::-1]

            # Top features more active in neutral context
            top_neutral_idx = np.argsort(diff)[:10]

            # Features only active in secure
            secure_only = np.where((secure_feats > 0) & (neutral_feats == 0))[0]

            # Features only active in neutral
            neutral_only = np.where((neutral_feats > 0) & (secure_feats == 0))[0]

            print(f"\n--- Layer {layer} ---")
            print(f"  Secure: {secure_features['num_active']} active features")
            print(f"  Neutral: {neutral_features['num_active']} active features")
            print(f"  Secure-only features: {len(secure_only)}")
            print(f"  Neutral-only features: {len(neutral_only)}")

            print(f"\n  Top features MORE active in secure context:")
            for idx in top_secure_idx[:5]:
                print(f"    Feature {idx}: secure={secure_feats[idx]:.4f}, neutral={neutral_feats[idx]:.4f}, diff={diff[idx]:+.4f}")

            print(f"\n  Top features MORE active in neutral context:")
            for idx in top_neutral_idx[:5]:
                print(f"    Feature {idx}: secure={secure_feats[idx]:.4f}, neutral={neutral_feats[idx]:.4f}, diff={diff[idx]:+.4f}")

            results['layer_analysis'][layer] = {
                'secure_num_active': secure_features['num_active'],
                'neutral_num_active': neutral_features['num_active'],
                'secure_only_count': len(secure_only),
                'neutral_only_count': len(neutral_only),
                'top_secure_features': [
                    {'idx': int(idx), 'secure': float(secure_feats[idx]),
                     'neutral': float(neutral_feats[idx]), 'diff': float(diff[idx])}
                    for idx in top_secure_idx[:10]
                ],
                'top_neutral_features': [
                    {'idx': int(idx), 'secure': float(secure_feats[idx]),
                     'neutral': float(neutral_feats[idx]), 'diff': float(diff[idx])}
                    for idx in top_neutral_idx[:10]
                ],
                'secure_only_features': [int(x) for x in secure_only[:20]],
                'neutral_only_features': [int(x) for x in neutral_only[:20]],
            }

        return results

    def find_security_features(self, results: dict) -> dict:
        """
        Identify candidate "security features" across layers.
        """
        print("\n" + "="*70)
        print("SECURITY FEATURE CANDIDATES")
        print("="*70)

        # Aggregate features that consistently differ
        all_secure_features = []
        all_neutral_features = []

        for layer, data in results['layer_analysis'].items():
            for feat in data['top_secure_features'][:5]:
                if feat['diff'] > 0.1:  # Significant difference
                    all_secure_features.append({
                        'layer': layer,
                        'feature_idx': feat['idx'],
                        'diff': feat['diff']
                    })

            for feat in data['top_neutral_features'][:5]:
                if feat['diff'] < -0.1:
                    all_neutral_features.append({
                        'layer': layer,
                        'feature_idx': feat['idx'],
                        'diff': feat['diff']
                    })

        # Sort by magnitude
        all_secure_features.sort(key=lambda x: x['diff'], reverse=True)
        all_neutral_features.sort(key=lambda x: x['diff'])

        print("\nTop security-promoting features (more active in secure context):")
        for feat in all_secure_features[:10]:
            print(f"  L{feat['layer']} Feature {feat['feature_idx']}: diff={feat['diff']:+.4f}")

        print("\nTop security-suppressing features (more active in neutral context):")
        for feat in all_neutral_features[:10]:
            print(f"  L{feat['layer']} Feature {feat['feature_idx']}: diff={feat['diff']:+.4f}")

        return {
            'security_promoting': all_secure_features[:20],
            'security_suppressing': all_neutral_features[:20]
        }


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("="*70)
    print("SAE SECURITY ANALYSIS")
    print("="*70)

    # Check SAE availability
    print("\nChecking SAE availability...")
    sae_available, SAE_class = check_sae_availability()

    if not sae_available:
        print("\n❌ Pretrained SAEs not available. Would need to train custom SAEs.")
        print("This requires significant compute time and data preparation.")
        return None

    # Initialize analyzer
    print("\n" + "="*70)
    analyzer = SAESecurityAnalyzer()

    # Load SAEs for layers 16-31 (where we know security signal is processed)
    target_layers = list(range(16, 32))
    print(f"\nLoading SAEs for layers {target_layers[0]}-{target_layers[-1]}...")

    if not analyzer.load_pretrained_saes(target_layers):
        print("Failed to load SAEs")
        return None

    # Run analysis
    results = analyzer.analyze_security_features(target_layers)

    # Find security features
    security_features = analyzer.find_security_features(results)
    results['security_features'] = security_features

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: DISTRIBUTED vs SPARSE")
    print("="*70)

    total_secure_only = sum(data['secure_only_count'] for data in results['layer_analysis'].values())
    total_neutral_only = sum(data['neutral_only_count'] for data in results['layer_analysis'].values())

    print(f"\nTotal secure-only features across all layers: {total_secure_only}")
    print(f"Total neutral-only features across all layers: {total_neutral_only}")

    # Check if distributed
    layers_with_diff = len([l for l, d in results['layer_analysis'].items()
                           if d['secure_only_count'] > 0 or d['neutral_only_count'] > 0])
    print(f"Layers with differential features: {layers_with_diff}/{len(target_layers)}")

    if layers_with_diff > len(target_layers) * 0.7:
        print("\n→ FINDING: Security signal is DISTRIBUTED across most layers")
    else:
        print("\n→ FINDING: Security signal may be more LOCALIZED")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(results_dir / f"sae_security_analysis_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_dir}/sae_security_analysis_{timestamp}.json")

    return results


if __name__ == "__main__":
    results = main()
