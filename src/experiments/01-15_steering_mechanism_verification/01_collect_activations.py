#!/usr/bin/env python3
"""
Steering Mechanism Verification - Activation Collection

Collects activations for three conditions:
A: Vulnerable prompts, alpha=0.0 (baseline)
B: Vulnerable prompts, alpha=3.5 (steered)
C: Secure prompts, alpha=0.0 (natural reference)

Verifies that steering moves activations in the predicted direction.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import re

# Add parent experiments to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import (
    MODEL_NAME, HIDDEN_SIZE, N_SAMPLES, BATCH_SIZE,
    ALPHA_BASELINE, ALPHA_STEERED, STEERING_LAYER,
    LAYERS_TO_EXTRACT, GENERATION_CONFIG,
    DATA_DIR, DATASET_PATH, ACTIVATION_CACHE,
    STRICT_PATTERNS, EXPANDED_SECURE_ADDITIONS, BOUNDS_CHECK_PATTERNS,
    C_CODE_INDICATORS, REFUSAL_PATTERNS,
)


# =============================================================================
# SCORING (simplified from baseline_behavior/scoring.py)
# =============================================================================

def score_completion(output: str, vuln_type: str) -> str:
    """
    Score a completion as 'secure', 'insecure', or 'other'.
    Uses EXPANDED scoring mode.
    """
    patterns = STRICT_PATTERNS.get(vuln_type, {})

    secure_pattern = patterns.get('secure', '')
    insecure_pattern = patterns.get('insecure', '')

    has_strict_secure = bool(re.search(secure_pattern, output)) if secure_pattern else False
    has_strict_insecure = bool(re.search(insecure_pattern, output)) if insecure_pattern else False

    # Check expanded additions (e.g., asprintf)
    has_expanded_addition = False
    additions = EXPANDED_SECURE_ADDITIONS.get(vuln_type, [])
    for pattern in additions:
        if re.search(pattern, output):
            has_expanded_addition = True
            break

    # Check bounds-check heuristic
    has_bounds_check = False
    for pattern in BOUNDS_CHECK_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            has_bounds_check = True
            break

    # Expanded scoring logic
    has_expanded_secure = has_strict_secure or has_expanded_addition or has_bounds_check

    if has_expanded_secure and not has_strict_insecure:
        return 'secure'
    elif has_strict_insecure and not has_bounds_check:
        return 'insecure'
    else:
        return 'other'


def detect_refusal(output: str) -> bool:
    """Check if output is a refusal."""
    # Check for C code indicators
    has_code = False
    for pattern in C_CODE_INDICATORS:
        if re.search(pattern, output):
            has_code = True
            break

    # Check for refusal language
    has_refusal = False
    output_lower = output.lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, output_lower):
            has_refusal = True
            break

    # Refusal = no code AND has refusal language
    return (not has_code) and has_refusal


# =============================================================================
# ACTIVATION COLLECTOR
# =============================================================================

class ActivationCollector:
    """Collects activations during generation with optional steering."""

    def __init__(self, model, tokenizer, steering_vector=None, steering_layer=31):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vector = steering_vector
        self.steering_layer = steering_layer
        self.collected_activations = {}
        self.hooks = []

    def _make_collection_hook(self, layer_idx):
        """Create a hook that saves activations for a specific layer."""
        def hook(module, input, output):
            # output is (hidden_states, ...) for transformer layers
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Save last token position (the "decision point")
            # Shape: (batch, seq_len, hidden_dim) -> (hidden_dim,)
            self.collected_activations[layer_idx] = hidden_states[0, -1, :].detach().cpu().numpy()
        return hook

    def _make_steering_hook(self, alpha):
        """Create a hook that applies steering at the steering layer."""
        steering_tensor = torch.tensor(
            self.steering_vector,
            device=self.model.device,
            dtype=self.model.dtype
        )

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            # Apply steering: add alpha * steering_vector to last token
            if alpha > 0:
                hidden_states = hidden_states.clone()
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + alpha * steering_tensor

            if rest:
                return (hidden_states,) + rest
            return hidden_states
        return hook

    def register_hooks(self, layers_to_extract, alpha=0.0):
        """Register forward hooks for activation collection and steering."""
        self.collected_activations = {}
        self.hooks = []

        for layer_idx in layers_to_extract:
            # Get the layer module
            layer = self.model.model.layers[layer_idx]

            # Steering hook (must run first at steering layer to modify activations)
            if layer_idx == self.steering_layer and alpha > 0:
                steering_hook = layer.register_forward_hook(self._make_steering_hook(alpha))
                self.hooks.append(steering_hook)

            # Collection hook (runs after layer forward, after steering)
            hook = layer.register_forward_hook(self._make_collection_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_with_activations(self, prompt, alpha=0.0, vuln_type="sprintf"):
        """Generate text and collect activations at the last prompt token."""
        self.register_hooks(LAYERS_TO_EXTRACT, alpha=alpha)

        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # First, do a forward pass on just the prompt to collect activations
            with torch.no_grad():
                _ = self.model(**inputs)

            # Now collected_activations has the activations at the last prompt token
            activations = dict(self.collected_activations)

            # Remove hooks before generation to avoid collecting at every step
            self.remove_hooks()

            # Re-register only steering hook for generation (no collection)
            if alpha > 0 and self.steering_vector is not None:
                layer = self.model.model.layers[self.steering_layer]
                steering_hook = layer.register_forward_hook(self._make_steering_hook(alpha))
                self.hooks.append(steering_hook)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    temperature=GENERATION_CONFIG["temperature"],
                    top_p=GENERATION_CONFIG["top_p"],
                    do_sample=GENERATION_CONFIG["do_sample"],
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Score the output
            classification = score_completion(generated_text, vuln_type)
            is_refusal = detect_refusal(generated_text)
            if is_refusal:
                classification = 'other'

        finally:
            self.remove_hooks()

        return generated_text, activations, classification


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset():
    """Load the CWE-787 expanded dataset."""
    dataset = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def load_cached_activations():
    """Load cached activations to compute steering direction."""
    data = np.load(ACTIVATION_CACHE)
    X = data['X_layer_31']  # (210, 4096)
    y = data['y_layer_31']  # (210,) - 0=vulnerable, 1=secure
    return X, y


def compute_steering_direction(X, y):
    """
    Compute mean-difference steering direction.
    Direction = mean(secure) - mean(vulnerable)
    """
    secure_mean = X[y == 1].mean(axis=0)
    vulnerable_mean = X[y == 0].mean(axis=0)
    direction = secure_mean - vulnerable_mean
    return direction.astype(np.float32)


def select_samples(dataset, n_samples, seed=42):
    """Randomly select n_samples pairs from the dataset."""
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    return indices.tolist()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Steering Mechanism Verification - Activation Collection")
    print("=" * 60)

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded on: {model.device}")

    # Load cached activations and compute steering direction
    print("\nComputing steering direction from cached activations...")
    X_cached, y_cached = load_cached_activations()
    steering_vector = compute_steering_direction(X_cached, y_cached)
    print(f"  Steering vector shape: {steering_vector.shape}")
    print(f"  Steering vector norm: {np.linalg.norm(steering_vector):.4f}")

    # Save steering direction for reference
    np.save(DATA_DIR / "steering_direction.npy", steering_vector)

    # Initialize collector
    collector = ActivationCollector(model, tokenizer, steering_vector, STEERING_LAYER)

    # Load dataset
    print(f"\nLoading dataset from: {DATASET_PATH}")
    dataset = load_dataset()
    print(f"  Total pairs: {len(dataset)}")

    # Select samples
    indices = select_samples(dataset, N_SAMPLES)
    print(f"  Selected {len(indices)} samples for each condition")

    # Results storage
    results = {
        "condition_A": [],  # vulnerable, alpha=0
        "condition_B": [],  # vulnerable, alpha=3.5
        "condition_C": [],  # secure, alpha=0
        "prompt_indices": indices,
        "config": {
            "n_samples": len(indices),
            "alpha_baseline": ALPHA_BASELINE,
            "alpha_steered": ALPHA_STEERED,
            "steering_layer": STEERING_LAYER,
            "layers_extracted": LAYERS_TO_EXTRACT,
            "model": MODEL_NAME,
            "generation_config": GENERATION_CONFIG,
        }
    }

    # Process in batches
    def process_condition(condition_name, prompts, alpha, vuln_types):
        print(f"\n[{condition_name}] Processing {len(prompts)} prompts, alpha={alpha}")
        condition_results = []

        for i, (prompt, vuln_type) in enumerate(tqdm(zip(prompts, vuln_types), total=len(prompts))):
            output, activations, classification = collector.generate_with_activations(
                prompt, alpha=alpha, vuln_type=vuln_type
            )

            condition_results.append({
                "prompt_idx": indices[i],
                "output": output,
                "classification": classification,
                "vuln_type": vuln_type,
                "activations": {str(k): v.tolist() for k, v in activations.items()}
            })

            # Clear GPU cache periodically
            if (i + 1) % BATCH_SIZE == 0:
                torch.cuda.empty_cache()

        return condition_results

    # Prepare prompts
    selected_pairs = [dataset[i] for i in indices]
    vulnerable_prompts = [p["vulnerable"] for p in selected_pairs]
    secure_prompts = [p["secure"] for p in selected_pairs]
    vuln_types = [p["vulnerability_type"] for p in selected_pairs]

    # Condition A: Vulnerable prompts, alpha=0.0
    results["condition_A"] = process_condition("Condition A", vulnerable_prompts, ALPHA_BASELINE, vuln_types)

    # Condition B: Vulnerable prompts, alpha=3.5
    results["condition_B"] = process_condition("Condition B", vulnerable_prompts, ALPHA_STEERED, vuln_types)

    # Condition C: Secure prompts, alpha=0.0
    results["condition_C"] = process_condition("Condition C", secure_prompts, ALPHA_BASELINE, vuln_types)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    for condition in ["condition_A", "condition_B", "condition_C"]:
        classifications = [r["classification"] for r in results[condition]]
        secure_rate = sum(1 for c in classifications if c == "secure") / len(classifications)
        insecure_rate = sum(1 for c in classifications if c == "insecure") / len(classifications)
        other_rate = sum(1 for c in classifications if c == "other") / len(classifications)
        print(f"{condition}: secure={secure_rate:.1%}, insecure={insecure_rate:.1%}, other={other_rate:.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON (with activations as lists)
    json_path = DATA_DIR / f"activations_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save activations as numpy for faster loading
    np_data = {}
    for condition in ["condition_A", "condition_B", "condition_C"]:
        for layer in LAYERS_TO_EXTRACT:
            key = f"{condition}_L{layer}"
            acts = [np.array(r["activations"][str(layer)]) for r in results[condition]]
            np_data[key] = np.stack(acts)

    np_path = DATA_DIR / f"activations_{timestamp}.npz"
    np.savez(np_path, **np_data)
    print(f"Numpy activations saved to: {np_path}")

    # Save a summary file
    summary = {
        "timestamp": timestamp,
        "n_samples": len(indices),
        "layers_extracted": LAYERS_TO_EXTRACT,
        "steering_layer": STEERING_LAYER,
        "alpha_steered": ALPHA_STEERED,
        "steering_vector_norm": float(np.linalg.norm(steering_vector)),
        "classification_rates": {}
    }
    for condition in ["condition_A", "condition_B", "condition_C"]:
        classifications = [r["classification"] for r in results[condition]]
        summary["classification_rates"][condition] = {
            "secure": sum(1 for c in classifications if c == "secure") / len(classifications),
            "insecure": sum(1 for c in classifications if c == "insecure") / len(classifications),
            "other": sum(1 for c in classifications if c == "other") / len(classifications),
        }

    with open(DATA_DIR / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Activation collection complete!")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
