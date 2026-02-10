#!/usr/bin/env python3
"""
Experiment 10, Step 6: End-to-End Steered Generation Pipeline

Full pipeline: probe → route → steer → score on neutral Python prompts.

For each of 21 neutral Python prompts × 10 seeds:
  1. Probe classifies which CWE type the prompt belongs to
  2. Route to correct steering vector with best alpha
  3. Generate steered completion
  4. Score with the routed CWE's scorer

Reports overall secure rate and per-CWE breakdown.
Compare with baseline (unsteered) to measure improvement.

Reuses: ModelLoader, SteeringGenerator from shared/; scorers from datasets/
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering")
DATASETS_DIR = EXPERIMENT_DIR / "datasets"
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))
sys.path.insert(0, str(DATASETS_DIR))

from model_loader import ModelLoader
from steering_generator import SteeringGenerator

# Import scorers
from cwe89.scoring import score_cwe89
from cwe78.scoring import score_cwe78
from cwe79.scoring import score_cwe79

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31
SEEDS = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
TOP_P = 0.9

SCORERS = {
    "CWE-89": score_cwe89,
    "CWE-78": score_cwe78,
    "CWE-79": score_cwe79,
}

NEUTRAL_FILE = DATASETS_DIR / "neutral_eval" / "data" / "neutral_python_prompts.jsonl"

CWE_NAMES = {0: "CWE-89", 1: "CWE-78", 2: "CWE-79"}
CWE_LABELS = {"CWE-89": 0, "CWE-78": 1, "CWE-79": 2}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, code_prefix):
    """Format as Python code completion."""
    user_message = ("Complete the following Python function. Only write the "
                    "function body, no explanation.\n\n" + code_prefix)
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


@torch.no_grad()
def get_l31_activation(model, tokenizer, prompt, layer=31):
    """Get Layer 31 last-token activation for a single prompt."""
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured["act"] = h[:, -1, :].detach().cpu()

    target_layer = model.model.layers[layer]
    hook = target_layer.register_forward_hook(hook_fn)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    _ = model(**inputs)

    hook.remove()
    return captured["act"].numpy().astype(np.float32)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 10, Step 6: E2E Steered Generation Pipeline (Python)")
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Seeds: {len(SEEDS)}")
    print("=" * 70)

    # ─── Load probe ──────────────────────────────────────────────────────
    print("\nLoading probe weights...")
    probe_files = sorted(DATA_DIR.glob("cwe_probe_weights_*.npz"))
    if not probe_files:
        print("ERROR: No probe weights found. Run Step 5 first.")
        return
    probe_data = np.load(probe_files[-1])

    probe = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    probe.coef_ = probe_data["coef"]
    probe.intercept_ = probe_data["intercept"]
    probe.classes_ = probe_data["classes"]
    print(f"  Probe loaded: {probe.coef_.shape[0]} classes")

    # ─── Load steering directions and best alphas ────────────────────────
    print("\nLoading steering vectors and alphas...")

    # Load LOBO results for best alphas
    lobo_files = sorted(RESULTS_DIR.glob("lobo_results_*.json"))
    if lobo_files:
        with open(lobo_files[-1]) as f:
            lobo_data = json.load(f)
        best_alphas = {
            cwe: lobo_data["cwe_results"][cwe]["best_alpha"]
            for cwe in lobo_data["cwe_results"]
        }
    else:
        best_alphas = {"CWE-89": 3.0, "CWE-78": 3.0, "CWE-79": 3.0}
        print("  WARNING: No LOBO results, using default alpha=3.0")

    directions = {}
    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        cwe_num = cwe.split("-")[1]
        npy_files = sorted(DATA_DIR.glob(f"direction_cwe{cwe_num}_L{LAYER}_*.npy"))
        if not npy_files:
            print(f"  ERROR: No direction for {cwe}. Run Step 2 first.")
            return
        directions[cwe] = np.load(npy_files[-1])
        print(f"  {cwe}: norm={np.linalg.norm(directions[cwe]):.4f}, α={best_alphas[cwe]}")

    # ─── Load model ──────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    generator = SteeringGenerator(loader)

    # ─── Load neutral prompts ────────────────────────────────────────────
    neutral_prompts = load_jsonl(NEUTRAL_FILE)
    print(f"\nLoaded {len(neutral_prompts)} neutral prompts")

    # ─── Generate (baseline + steered) ───────────────────────────────────
    print(f"\n{'='*70}")
    print("GENERATING: baseline + steered")
    print(f"{'='*70}")

    baseline_results = []
    steered_results = []

    for item in tqdm(neutral_prompts, desc="Neutral prompts"):
        true_cwe = item["cwe"]
        formatted = format_chat_prompt(tokenizer, item["prompt"])

        # Probe classification
        act = get_l31_activation(model, tokenizer, formatted, LAYER)
        pred_label = probe.predict(act)[0]
        pred_cwe = CWE_NAMES[pred_label]
        pred_proba = probe.predict_proba(act)[0]
        confidence = pred_proba[pred_label]

        direction = directions[pred_cwe]
        alpha = best_alphas[pred_cwe]
        scorer = SCORERS.get(true_cwe, SCORERS.get(pred_cwe))

        for seed in SEEDS:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Baseline (no steering)
            baseline_output = generator.generate_baseline(
                prompt=formatted,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_NEW_TOKENS,
            )
            baseline_label = scorer(baseline_output)
            baseline_results.append({
                "id": item["id"],
                "cwe": true_cwe,
                "seed": seed,
                "label": baseline_label,
                "output": baseline_output[:500],
            })

            # Steered
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            steered_output = generator.generate_with_steering(
                prompt=formatted,
                direction=direction,
                layer=LAYER,
                alpha=alpha,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_NEW_TOKENS,
            )
            steered_label = scorer(steered_output)
            steered_results.append({
                "id": item["id"],
                "cwe": true_cwe,
                "pred_cwe": pred_cwe,
                "confidence": float(confidence),
                "alpha": alpha,
                "seed": seed,
                "label": steered_label,
                "output": steered_output[:500],
            })

    loader.unload()

    # ─── Results ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("E2E RESULTS")
    print(f"{'='*70}")

    # Per-CWE breakdown
    print(f"\n{'Mode':<12} {'CWE':<10} {'Secure':>8} {'Insecure':>10} {'Other':>8} {'n':>6} {'Sec%':>8}")
    print("-" * 64)

    for mode, results in [("Baseline", baseline_results), ("Steered", steered_results)]:
        for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
            cr = [r for r in results if r["cwe"] == cwe]
            n = len(cr)
            ns = sum(1 for r in cr if r["label"] == "secure")
            ni = sum(1 for r in cr if r["label"] == "insecure")
            no = sum(1 for r in cr if r["label"] == "other")
            print(f"{mode:<12} {cwe:<10} {ns:>8} {ni:>10} {no:>8} {n:>6} "
                  f"{ns/n*100 if n else 0:>7.1f}%")

        n_total = len(results)
        n_sec = sum(1 for r in results if r["label"] == "secure")
        n_ins = sum(1 for r in results if r["label"] == "insecure")
        n_oth = sum(1 for r in results if r["label"] == "other")
        print(f"{mode:<12} {'Overall':<10} {n_sec:>8} {n_ins:>10} {n_oth:>8} {n_total:>6} "
              f"{n_sec/n_total*100 if n_total else 0:>7.1f}%")
        print()

    # Improvement
    baseline_sec = sum(1 for r in baseline_results if r["label"] == "secure")
    steered_sec = sum(1 for r in steered_results if r["label"] == "secure")
    n = len(baseline_results)
    improvement = (steered_sec - baseline_sec) / n * 100
    print(f"Baseline secure rate: {baseline_sec/n*100:.1f}%")
    print(f"Steered secure rate:  {steered_sec/n*100:.1f}%")
    print(f"Improvement: {improvement:+.1f}pp")

    # Routing accuracy
    correct_routing = sum(1 for r in steered_results
                         if r["cwe"] == r["pred_cwe"] and r["seed"] == SEEDS[0])
    total_routing = sum(1 for r in steered_results if r["seed"] == SEEDS[0])
    print(f"\nRouting accuracy: {correct_routing}/{total_routing} "
          f"({correct_routing/total_routing*100:.1f}%)")

    # ─── Save ────────────────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "layer": LAYER,
        "n_seeds": len(SEEDS),
        "best_alphas": best_alphas,
        "n_neutral": len(neutral_prompts),
        "baseline_secure_rate": baseline_sec / n,
        "steered_secure_rate": steered_sec / n,
        "improvement_pp": improvement,
        "routing_accuracy": correct_routing / total_routing if total_routing > 0 else 0,
    }

    results_path = RESULTS_DIR / f"e2e_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Full results
    full_path = RESULTS_DIR / f"e2e_full_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump({
            "baseline_results": baseline_results,
            "steered_results": steered_results,
        }, f, indent=2)
    print(f"Full results saved: {full_path}")

    print("\nE2E pipeline complete.")


if __name__ == "__main__":
    main()
