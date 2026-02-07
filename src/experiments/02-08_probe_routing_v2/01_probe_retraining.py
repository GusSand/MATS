#!/usr/bin/env python3
"""
Experiment 8.5 — Part A: Probe Retraining with Neutral/Mixed Data

Retrains CWE-type classifiers using:
1. Neutral-only data (21 samples, LOO cross-validation)
2. Prefix-augmented neutral data (105 samples, leave-one-base-out CV)
3. Mixed data (adversarial train → neutral test)
4. Mixed+augmented (adversarial + augmented neutral train → neutral test)

Also trains binary format-string probes for the 2-tier architecture.
Tests across layers [0, 8, 16, 24, 31] to find where CWE-type info is encoded.

Reuses: ModelLoader from shared/, activation collection pattern from Phase 4.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering")
sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))

from model_loader import ModelLoader

DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
PROMPTS_PATH = EXPERIMENT_DIR / "neutral_eval" / "data" / "neutral_eval_prompts.jsonl"
VECTORS_DIR = EXPERIMENT_DIR / "cross_cwe_analysis" / "data"

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PROBE_LAYERS = [0, 8, 16, 24, 31]

ADVERSARIAL_DATASETS = {
    "CWE-787": Path("/home/paperspace/MATS/src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl"),
    "CWE-119": EXPERIMENT_DIR / "datasets" / "cwe119" / "data" / "cwe119_expanded_20260207_024627.jsonl",
    "CWE-134": EXPERIMENT_DIR / "datasets" / "cwe134" / "data" / "cwe134_expanded_20260207_024627.jsonl",
}

CWE_TO_IDX = {"CWE-787": 0, "CWE-119": 1, "CWE-134": 2}
IDX_TO_CWE = {0: "CWE-787", 1: "CWE-119", 2: "CWE-134"}

# Instruction variants for prefix augmentation
INSTRUCTION_VARIANTS = [
    "Complete the following C function. Only write the function body, no explanation.",  # Original
    "Write the implementation for this C function. Only provide the code.",
    "Implement the C function below. Return only the function body.",
    "Fill in the body of this C function. No explanation needed.",
    "Complete this C code. Provide only the implementation.",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, code_prefix, instruction=None):
    """Format code prefix into chat prompt with given instruction."""
    if instruction is None:
        instruction = INSTRUCTION_VARIANTS[0]
    user_message = f"{instruction}\n\n{code_prefix}"
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def collect_activations(model, tokenizer, device, prompts, layers):
    """
    Collect last-token activations at specified layers.
    Reused from Phase 4 (neutral_probe_routing.py).
    """
    activations = {layer: [] for layer in layers}
    hooks = []
    current_acts = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            current_acts[layer_idx] = h[:, -1, :].detach().cpu().numpy().astype(np.float32)
        return hook_fn

    for layer_idx in layers:
        hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    for prompt in tqdm(prompts, desc="Collecting activations"):
        current_acts.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        for layer_idx in layers:
            activations[layer_idx].append(current_acts[layer_idx].squeeze(0))

    for hook in hooks:
        hook.remove()

    return {layer: np.stack(acts) for layer, acts in activations.items()}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 8.5 — Part A: Probe Retraining")
    print(f"Model: {MODEL_NAME}")
    print(f"Probe layers: {PROBE_LAYERS}")
    print(f"Instruction variants: {len(INSTRUCTION_VARIANTS)}")
    print("=" * 70)

    # ─── Load neutral prompts ────────────────────────────────────────────
    neutral_prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                neutral_prompts.append(json.loads(line))
    print(f"Loaded {len(neutral_prompts)} neutral prompts")
    for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
        n = sum(1 for p in neutral_prompts if p["cwe"] == cwe)
        print(f"  {cwe}: {n}")

    # ─── Load adversarial datasets ───────────────────────────────────────
    adv_data = {}
    for cwe, path in ADVERSARIAL_DATASETS.items():
        with open(path) as f:
            adv_data[cwe] = [json.loads(line) for line in f if line.strip()]
        print(f"  Adversarial {cwe}: {len(adv_data[cwe])} pairs")

    # ─── Load model ──────────────────────────────────────────────────────
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    # ─── Format prompts ──────────────────────────────────────────────────

    # Augmented neutral: 21 base × 5 variants = 105
    print("\nFormatting augmented neutral prompts (21 × 5 = 105)...")
    aug_formatted = []
    aug_labels = []
    aug_groups = []       # base prompt index (for LOBO CV)
    aug_variant_idx = []  # 0 = original instruction
    aug_ids = []

    for base_idx, p in enumerate(neutral_prompts):
        for var_idx, instruction in enumerate(INSTRUCTION_VARIANTS):
            aug_formatted.append(format_chat_prompt(tokenizer, p["prompt"], instruction))
            aug_labels.append(CWE_TO_IDX[p["cwe"]])
            aug_groups.append(base_idx)
            aug_variant_idx.append(var_idx)
            aug_ids.append(p["id"])

    aug_labels = np.array(aug_labels)
    aug_groups = np.array(aug_groups)
    aug_variant_idx = np.array(aug_variant_idx)

    # Adversarial: 105 × 3 CWEs = 315 (vulnerable variants)
    print("Formatting adversarial prompts (315)...")
    adv_formatted = []
    adv_labels = []
    for cwe in ["CWE-787", "CWE-119", "CWE-134"]:
        for pair in adv_data[cwe]:
            adv_formatted.append(format_chat_prompt(tokenizer, pair["vulnerable"]))
            adv_labels.append(CWE_TO_IDX[cwe])
    adv_labels = np.array(adv_labels)

    total_prompts = len(aug_formatted) + len(adv_formatted)
    print(f"\nTotal forward passes: {total_prompts} "
          f"({len(aug_formatted)} augmented + {len(adv_formatted)} adversarial)")

    # ─── Collect activations ─────────────────────────────────────────────
    all_formatted = aug_formatted + adv_formatted
    all_activations = collect_activations(model, tokenizer, device, all_formatted, PROBE_LAYERS)

    # Split back
    n_aug = len(aug_formatted)
    aug_acts = {layer: all_activations[layer][:n_aug] for layer in PROBE_LAYERS}
    adv_acts = {layer: all_activations[layer][n_aug:] for layer in PROBE_LAYERS}

    # Extract original-variant-only (variant_idx == 0) = the 21 neutral prompts
    original_mask = aug_variant_idx == 0
    neutral_acts = {layer: aug_acts[layer][original_mask] for layer in PROBE_LAYERS}
    neutral_labels = aug_labels[original_mask]
    neutral_ids = [neutral_prompts[i]["id"] for i in range(len(neutral_prompts))]

    loader.unload()

    # ─── Save activations ────────────────────────────────────────────────
    print("\nSaving activations...")
    for layer in PROBE_LAYERS:
        np.save(DATA_DIR / f"neutral_original_L{layer}.npy", neutral_acts[layer])
        np.save(DATA_DIR / f"neutral_augmented_L{layer}.npy", aug_acts[layer])
        np.save(DATA_DIR / f"adversarial_L{layer}.npy", adv_acts[layer])

    labels_meta = {
        "neutral_original_labels": neutral_labels.tolist(),
        "neutral_original_ids": neutral_ids,
        "augmented_labels": aug_labels.tolist(),
        "augmented_groups": aug_groups.tolist(),
        "augmented_variant_idx": aug_variant_idx.tolist(),
        "augmented_ids": aug_ids,
        "adversarial_labels": adv_labels.tolist(),
        "instruction_variants": INSTRUCTION_VARIANTS,
    }
    with open(DATA_DIR / "labels_metadata.json", "w") as f:
        json.dump(labels_meta, f, indent=2)
    print("  Saved activation arrays and labels metadata.")

    # ═══════════════════════════════════════════════════════════════════════
    # PROBING
    # ═══════════════════════════════════════════════════════════════════════

    results = {
        "timestamp": timestamp,
        "experiment": "Experiment 8.5 Part A: Probe Retraining",
        "model": MODEL_NAME,
        "layers": PROBE_LAYERS,
        "n_neutral_original": int(original_mask.sum()),
        "n_neutral_augmented": len(aug_labels),
        "n_adversarial": len(adv_labels),
        "n_instruction_variants": len(INSTRUCTION_VARIANTS),
        "per_layer": {},
    }

    for layer in PROBE_LAYERS:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print(f"{'='*70}")

        X_neutral = neutral_acts[layer]    # [21, 4096]
        y_neutral = neutral_labels         # [21]
        X_aug = aug_acts[layer]            # [105, 4096]
        y_aug = aug_labels                 # [105]
        X_adv = adv_acts[layer]            # [315, 4096]
        y_adv = adv_labels                 # [315]

        layer_result = {"three_way": {}, "binary": {}}

        # ── 3-Way: Method 1 — Neutral-only LOO ──────────────────────────
        print(f"\n  3-Way Method 1: Neutral-only LOO (n={len(y_neutral)})")
        scaler1 = StandardScaler()
        X1 = scaler1.fit_transform(X_neutral)
        clf1 = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                  multi_class='multinomial')
        y_pred1 = cross_val_predict(clf1, X1, y_neutral, cv=LeaveOneOut())
        acc1 = accuracy_score(y_neutral, y_pred1)
        cm1 = confusion_matrix(y_neutral, y_pred1, labels=[0, 1, 2])

        per_cwe1 = {}
        for idx, name in IDX_TO_CWE.items():
            mask = y_neutral == idx
            if mask.sum() > 0:
                per_cwe1[name] = float((y_pred1[mask] == y_neutral[mask]).mean())

        print(f"    Overall: {acc1*100:.1f}%")
        for name, a in per_cwe1.items():
            print(f"    {name}: {a*100:.1f}%")
        print(f"    CM: {cm1.tolist()}")

        layer_result["three_way"]["neutral_loo"] = {
            "accuracy": float(acc1),
            "per_cwe": per_cwe1,
            "confusion_matrix": cm1.tolist(),
            "predictions": y_pred1.tolist(),
        }

        # ── 3-Way: Method 2 — Augmented LOBO ────────────────────────────
        print(f"\n  3-Way Method 2: Augmented LOBO (n={len(y_aug)}, 21 groups)")
        scaler2 = StandardScaler()
        X2 = scaler2.fit_transform(X_aug)
        clf2 = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                  multi_class='multinomial')
        logo = LeaveOneGroupOut()
        y_pred2 = cross_val_predict(clf2, X2, y_aug, cv=logo, groups=aug_groups)
        acc2_all = accuracy_score(y_aug, y_pred2)

        # Accuracy on original variants only (these are what deployment sees)
        orig_mask = aug_variant_idx == 0
        acc2_orig = accuracy_score(y_aug[orig_mask], y_pred2[orig_mask])
        cm2 = confusion_matrix(y_aug[orig_mask], y_pred2[orig_mask], labels=[0, 1, 2])

        per_cwe2 = {}
        for idx, name in IDX_TO_CWE.items():
            mask = (y_aug == idx) & orig_mask
            if mask.sum() > 0:
                per_cwe2[name] = float((y_pred2[mask] == y_aug[mask]).mean())

        print(f"    All variants: {acc2_all*100:.1f}%")
        print(f"    Original only: {acc2_orig*100:.1f}%")
        for name, a in per_cwe2.items():
            print(f"    {name} (orig): {a*100:.1f}%")

        layer_result["three_way"]["augmented_lobo"] = {
            "accuracy_all_variants": float(acc2_all),
            "accuracy_original_only": float(acc2_orig),
            "per_cwe_original": per_cwe2,
            "confusion_matrix_original": cm2.tolist(),
        }

        # ── 3-Way: Method 3 — Mixed (adv train → neutral test) ──────────
        print(f"\n  3-Way Method 3: Mixed (adv train → neutral test)")
        scaler3 = StandardScaler()
        X3_train = scaler3.fit_transform(X_adv)
        X3_test = scaler3.transform(X_neutral)
        clf3 = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                  multi_class='multinomial')
        clf3.fit(X3_train, y_adv)
        y_pred3 = clf3.predict(X3_test)
        probas3 = clf3.predict_proba(X3_test)
        acc3 = accuracy_score(y_neutral, y_pred3)
        cm3 = confusion_matrix(y_neutral, y_pred3, labels=[0, 1, 2])

        per_cwe3 = {}
        misrouted3 = []
        for idx, name in IDX_TO_CWE.items():
            mask = y_neutral == idx
            if mask.sum() > 0:
                per_cwe3[name] = float((y_pred3[mask] == y_neutral[mask]).mean())

        for i in range(len(y_neutral)):
            if y_pred3[i] != y_neutral[i]:
                misrouted3.append({
                    "id": neutral_ids[i],
                    "true": IDX_TO_CWE[y_neutral[i]],
                    "predicted": IDX_TO_CWE[y_pred3[i]],
                    "probabilities": {IDX_TO_CWE[j]: float(probas3[i][j]) for j in range(3)},
                })

        print(f"    Overall: {acc3*100:.1f}%")
        for name, a in per_cwe3.items():
            print(f"    {name}: {a*100:.1f}%")
        if misrouted3:
            print(f"    Misrouted ({len(misrouted3)}):")
            for m in misrouted3:
                print(f"      {m['id']}: {m['true']} → {m['predicted']} "
                      f"(P={m['probabilities']})")

        layer_result["three_way"]["mixed_adv_train"] = {
            "accuracy": float(acc3),
            "per_cwe": per_cwe3,
            "confusion_matrix": cm3.tolist(),
            "misrouted": misrouted3,
        }

        # ── 3-Way: Method 4 — Mixed+Augmented ───────────────────────────
        # WARNING: This has data leakage — augmented set includes original
        # neutral prompts (variant_idx=0) which ARE the test set.
        # Results will be artificially inflated. Reported for completeness.
        print(f"\n  3-Way Method 4: Mixed+Augmented (adv+aug → neutral test) [LEAKY]")
        X_combined = np.vstack([X_adv, X_aug])
        y_combined = np.concatenate([y_adv, y_aug])
        scaler4 = StandardScaler()
        X4_train = scaler4.fit_transform(X_combined)
        X4_test = scaler4.transform(X_neutral)
        clf4 = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                  multi_class='multinomial')
        clf4.fit(X4_train, y_combined)
        y_pred4 = clf4.predict(X4_test)
        acc4 = accuracy_score(y_neutral, y_pred4)
        cm4 = confusion_matrix(y_neutral, y_pred4, labels=[0, 1, 2])

        per_cwe4 = {}
        for idx, name in IDX_TO_CWE.items():
            mask = y_neutral == idx
            if mask.sum() > 0:
                per_cwe4[name] = float((y_pred4[mask] == y_neutral[mask]).mean())

        print(f"    Overall: {acc4*100:.1f}%")
        for name, a in per_cwe4.items():
            print(f"    {name}: {a*100:.1f}%")

        layer_result["three_way"]["mixed_augmented"] = {
            "accuracy": float(acc4),
            "per_cwe": per_cwe4,
            "confusion_matrix": cm4.tolist(),
        }

        # ── Binary Probe: Format-String vs Buffer ────────────────────────
        print(f"\n  Binary: Format-string (CWE-134) vs Buffer (CWE-787+119)")
        y_bin_neutral = (y_neutral == 2).astype(int)  # 1=CWE-134, 0=buffer
        y_bin_adv = (y_adv == 2).astype(int)
        y_bin_aug = (y_aug == 2).astype(int)
        buf_mask = y_bin_neutral == 0
        fmt_mask = y_bin_neutral == 1

        # Binary A: Neutral-only LOO
        scaler_ba = StandardScaler()
        Xba = scaler_ba.fit_transform(X_neutral)
        clf_ba = LogisticRegression(C=1.0, max_iter=1000)
        y_pred_ba = cross_val_predict(clf_ba, Xba, y_bin_neutral, cv=LeaveOneOut())
        acc_ba = accuracy_score(y_bin_neutral, y_pred_ba)
        buf_acc_ba = float((y_pred_ba[buf_mask] == y_bin_neutral[buf_mask]).mean())
        fmt_acc_ba = float((y_pred_ba[fmt_mask] == y_bin_neutral[fmt_mask]).mean())
        print(f"    LOO: {acc_ba*100:.1f}% (buffer={buf_acc_ba*100:.1f}%, format={fmt_acc_ba*100:.1f}%)")

        # Binary B: Adv-trained → neutral test
        scaler_bb = StandardScaler()
        Xbb_train = scaler_bb.fit_transform(X_adv)
        Xbb_test = scaler_bb.transform(X_neutral)
        clf_bb = LogisticRegression(C=1.0, max_iter=1000)
        clf_bb.fit(Xbb_train, y_bin_adv)
        y_pred_bb = clf_bb.predict(Xbb_test)
        acc_bb = accuracy_score(y_bin_neutral, y_pred_bb)
        buf_acc_bb = float((y_pred_bb[buf_mask] == y_bin_neutral[buf_mask]).mean())
        fmt_acc_bb = float((y_pred_bb[fmt_mask] == y_bin_neutral[fmt_mask]).mean())
        print(f"    Adv-trained: {acc_bb*100:.1f}% (buffer={buf_acc_bb*100:.1f}%, format={fmt_acc_bb*100:.1f}%)")

        # Binary C: Mixed (adv + augmented) → neutral test
        X_mix_bin = np.vstack([X_adv, X_aug])
        y_mix_bin = np.concatenate([y_bin_adv, y_bin_aug])
        scaler_bc = StandardScaler()
        Xbc_train = scaler_bc.fit_transform(X_mix_bin)
        Xbc_test = scaler_bc.transform(X_neutral)
        clf_bc = LogisticRegression(C=1.0, max_iter=1000)
        clf_bc.fit(Xbc_train, y_mix_bin)
        y_pred_bc = clf_bc.predict(Xbc_test)
        acc_bc = accuracy_score(y_bin_neutral, y_pred_bc)
        buf_acc_bc = float((y_pred_bc[buf_mask] == y_bin_neutral[buf_mask]).mean())
        fmt_acc_bc = float((y_pred_bc[fmt_mask] == y_bin_neutral[fmt_mask]).mean())
        print(f"    Mixed: {acc_bc*100:.1f}% (buffer={buf_acc_bc*100:.1f}%, format={fmt_acc_bc*100:.1f}%)")

        layer_result["binary"] = {
            "neutral_loo": {
                "accuracy": float(acc_ba),
                "buffer_accuracy": buf_acc_ba,
                "format_accuracy": fmt_acc_ba,
                "predictions": y_pred_ba.tolist(),
            },
            "adv_trained": {
                "accuracy": float(acc_bb),
                "buffer_accuracy": buf_acc_bb,
                "format_accuracy": fmt_acc_bb,
                "predictions": y_pred_bb.tolist(),
            },
            "mixed_trained": {
                "accuracy": float(acc_bc),
                "buffer_accuracy": buf_acc_bc,
                "format_accuracy": fmt_acc_bc,
                "predictions": y_pred_bc.tolist(),
            },
        }

        results["per_layer"][f"layer_{layer}"] = layer_result

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLES
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*90}")
    print("TABLE 1: 3-WAY ROUTING ACCURACY")
    print(f"{'='*90}")
    print(f"{'Method':<45} {'Overall':>8} {'CWE-787':>8} {'CWE-119':>8} {'CWE-134':>8}")
    print("-" * 85)

    # Baseline from Exp 8 Phase 4
    print(f"{'Exp8 adv-trained L31 (baseline)':<45} {'66.7%':>8} {'85.7%':>8} {'14.3%':>8} {'100.0%':>8}")

    for layer in PROBE_LAYERS:
        lr = results["per_layer"][f"layer_{layer}"]["three_way"]
        for key, label in [
            ("neutral_loo", f"Neutral LOO L{layer}"),
            ("augmented_lobo", f"Augmented LOBO L{layer}"),
            ("mixed_adv_train", f"Mixed (adv→neut) L{layer}"),
            ("mixed_augmented", f"Mixed+Aug L{layer}"),
        ]:
            r = lr[key]
            acc = r.get("accuracy", r.get("accuracy_original_only", 0))
            pc = r.get("per_cwe", r.get("per_cwe_original", {}))
            print(f"{label:<45} {acc*100:>7.1f}% "
                  f"{pc.get('CWE-787', 0)*100:>7.1f}% "
                  f"{pc.get('CWE-119', 0)*100:>7.1f}% "
                  f"{pc.get('CWE-134', 0)*100:>7.1f}%")

    print(f"\n{'='*80}")
    print("TABLE 2: BINARY PROBE ACCURACY (Format-String vs Buffer)")
    print(f"{'='*80}")
    print(f"{'Method':<45} {'Overall':>8} {'Buffer':>8} {'Format':>8}")
    print("-" * 75)

    for layer in PROBE_LAYERS:
        br = results["per_layer"][f"layer_{layer}"]["binary"]
        for key, label in [
            ("neutral_loo", f"Binary LOO L{layer}"),
            ("adv_trained", f"Binary adv-trained L{layer}"),
            ("mixed_trained", f"Binary mixed L{layer}"),
        ]:
            r = br[key]
            buf = r.get("buffer_accuracy", 0)
            fmt = r.get("format_accuracy", 0)
            print(f"{label:<45} {r['accuracy']*100:>7.1f}% "
                  f"{buf*100:>7.1f}% {fmt*100:>7.1f}%")

    # ─── Save binary probe weights for deployment (Part C) ───────────────
    # NOTE: Use adv-trained probe (no data leakage). Mixed training includes
    # neutral test data in training set, so its 100% accuracy is spurious.
    print("\nTraining final binary probe for deployment (L31, adv-trained)...")
    X_final = adv_acts[31]
    y_final = (adv_labels == 2).astype(int)
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)
    clf_final = LogisticRegression(C=1.0, max_iter=1000)
    clf_final.fit(X_final_scaled, y_final)

    np.save(DATA_DIR / "binary_probe_weights.npy", clf_final.coef_)
    np.save(DATA_DIR / "binary_probe_bias.npy", clf_final.intercept_)
    np.save(DATA_DIR / "binary_probe_scaler_mean.npy", scaler_final.mean_)
    np.save(DATA_DIR / "binary_probe_scaler_scale.npy", scaler_final.scale_)
    print("  Binary probe weights saved (adv-trained, no leakage) for Part C.")

    # ─── Save results ────────────────────────────────────────────────────
    results_path = RESULTS_DIR / f"3way_probe_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print("\n" + "=" * 70)
    print("Part A complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
