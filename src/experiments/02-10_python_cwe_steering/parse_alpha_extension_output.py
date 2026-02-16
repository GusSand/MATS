#!/usr/bin/env python3
"""
Parse the output log from 03b_lobo_alpha_extension.py (which was terminated mid-run)
and save structured JSON results for CWE-89 (complete) and CWE-78 (partial: 6/7 folds,
fold 6 missing α=15.0, fold 7 never run).

Produces:
  - results/alpha_extension_partial_89_78.json  (full fold data)
  - results/alpha_extension_results_partial.json (aggregated summary + merged with prior)
"""
import json
import re
from pathlib import Path
from datetime import datetime

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_LOG = Path("/tmp/claude-1000/-home-paperspace-MATS/tasks/b2c581f.output")
PRIOR_RESULTS_FILE = RESULTS_DIR / "lobo_full_20260210_080939.json"

NEW_ALPHAS = [6.0, 7.0, 8.0, 10.0, 12.0, 15.0]


def parse_output_log(log_path):
    """Parse the structured output log and extract fold-level results."""
    with open(log_path) as f:
        lines = f.readlines()

    all_cwe_results = {}
    current_cwe = None
    current_fold = None
    fold_results = []

    for line in lines:
        line = line.rstrip()

        # Detect CWE section start
        m = re.match(r'^LOBO ALPHA EXTENSION: (CWE-\d+)', line)
        if m:
            # Save previous CWE if any
            if current_cwe and fold_results:
                all_cwe_results[current_cwe] = fold_results
            current_cwe = m.group(1)
            fold_results = []
            current_fold = None
            continue

        # Detect fold start
        m = re.match(r'^\s+--- Fold (\d+)/(\d+): hold out base_id=(\S+) ---', line)
        if m:
            current_fold = {
                "fold_id": m.group(3),
                "fold_num": int(m.group(1)),
                "total_folds": int(m.group(2)),
                "n_test": None,
                "direction_norm": None,
                "alpha_results": {},
            }
            fold_results.append(current_fold)
            continue

        # Detect AGGREGATE section (stop parsing fold data)
        if re.match(r'^\s+.*ALPHA EXTENSION AGGREGATE', line):
            current_fold = None
            continue

        # Detect "NEW BEST" or "Prior" lines (not fold data)
        if re.match(r'^\s+(NEW BEST|Prior)', line):
            current_fold = None
            continue

        if current_fold is None:
            continue

        # Parse Train line
        m = re.match(r'^\s+Train: (\d+) \((\d+) insecure \+ (\d+) secure\)', line)
        if m:
            # We don't store train count in fold data, but could if needed
            continue

        # Parse Test line
        m = re.match(r'^\s+Test: (\d+) prompts', line)
        if m:
            current_fold["n_test"] = int(m.group(1))
            continue

        # Parse Direction norm
        m = re.match(r'^\s+Direction norm: ([\d.]+)', line)
        if m:
            current_fold["direction_norm"] = float(m.group(1))
            continue

        # Parse alpha result line: α=6.0: secure=4/150 (2.7%), insecure=145/150 (96.7%), other=1/150 (0.7%)
        m = re.match(
            r'^\s+α=([\d.]+): secure=(\d+)/(\d+) \(([\d.]+)%\), '
            r'insecure=(\d+)/(\d+) \(([\d.]+)%\), '
            r'other=(\d+)/(\d+) \(([\d.]+)%\)',
            line
        )
        if m:
            alpha = float(m.group(1))
            n_secure = int(m.group(2))
            n_total = int(m.group(3))
            n_insecure = int(m.group(5))
            n_other = int(m.group(8))

            alpha_key = str(alpha)
            current_fold["alpha_results"][alpha_key] = {
                "n": n_total,
                "n_secure": n_secure,
                "n_insecure": n_insecure,
                "n_other": n_other,
                "secure_rate": n_secure / n_total if n_total > 0 else 0,
                "insecure_rate": n_insecure / n_total if n_total > 0 else 0,
                "other_rate": n_other / n_total if n_total > 0 else 0,
            }
            continue

    # Save the last CWE
    if current_cwe and fold_results:
        all_cwe_results[current_cwe] = fold_results

    return all_cwe_results


def aggregate_folds(fold_results, alphas=None):
    """Aggregate results across folds for each alpha. Only include alphas present in ALL folds."""
    if alphas is None:
        # Find alphas present in all folds
        alpha_sets = [set(f["alpha_results"].keys()) for f in fold_results]
        common_alphas = set.intersection(*alpha_sets) if alpha_sets else set()
        alphas = sorted([float(a) for a in common_alphas])

    agg = {}
    for alpha in alphas:
        alpha_key = str(alpha)
        total_n = 0
        total_sec = 0
        total_ins = 0
        total_oth = 0
        folds_with_alpha = 0

        for f in fold_results:
            if alpha_key in f["alpha_results"]:
                r = f["alpha_results"][alpha_key]
                total_n += r["n"]
                total_sec += r["n_secure"]
                total_ins += r["n_insecure"]
                total_oth += r["n_other"]
                folds_with_alpha += 1

        if total_n > 0:
            agg[alpha_key] = {
                "n": total_n,
                "n_secure": total_sec,
                "n_insecure": total_ins,
                "n_other": total_oth,
                "secure_rate": total_sec / total_n,
                "insecure_rate": total_ins / total_n,
                "other_rate": total_oth / total_n,
                "n_folds": folds_with_alpha,
            }

    return agg


def main():
    timestamp = "partial"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Parsing LOBO Alpha Extension output log")
    print(f"Log file: {OUTPUT_LOG}")
    print("=" * 70)

    # ─── Parse the output log ─────────────────────────────────────────────
    parsed = parse_output_log(OUTPUT_LOG)

    for cwe, folds in parsed.items():
        print(f"\n{cwe}: {len(folds)} folds parsed")
        for f in folds:
            alphas_parsed = sorted(f["alpha_results"].keys())
            print(f"  Fold {f['fold_num']}/{f['total_folds']} (base_id={f['fold_id']}): "
                  f"{len(alphas_parsed)} alphas ({', '.join(alphas_parsed)})")

    # ─── Load prior results ───────────────────────────────────────────────
    print(f"\nLoading prior results from {PRIOR_RESULTS_FILE.name}...")
    with open(PRIOR_RESULTS_FILE) as f:
        prior_results = json.load(f)

    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        if cwe in prior_results:
            print(f"  {cwe}: best α={prior_results[cwe]['best_alpha']}, "
                  f"secure={prior_results[cwe]['best_rate']*100:.1f}%")

    # ─── Build fold-level output matching 03b format ──────────────────────
    # Structure matches what 03b_lobo_alpha_extension.py would have saved
    all_cwe_results = {}

    for cwe, folds in parsed.items():
        # Clean up fold data to match expected format
        clean_folds = []
        for f in folds:
            clean_folds.append({
                "fold_id": f["fold_id"],
                "n_test": f["n_test"],
                "direction_norm": f["direction_norm"],
                "alpha_results": f["alpha_results"],
            })

        # Aggregate across all available folds
        # For CWE-89: all 7 folds, all 6 alphas
        # For CWE-78: 6 folds (1-5 have all 6 alphas, fold 6 has 5 alphas)
        agg = aggregate_folds(folds)

        # Find best new alpha
        best_alpha_new = max(
            [float(a) for a in agg.keys()],
            key=lambda a: agg[str(a)]["secure_rate"]
        )
        best_rate_new = agg[str(best_alpha_new)]["secure_rate"]

        all_cwe_results[cwe] = {
            "fold_results": clean_folds,
            "aggregated": agg,
            "best_alpha_new": best_alpha_new,
            "best_rate_new": best_rate_new,
        }

        print(f"\n{cwe} AGGREGATE (from parsed output):")
        for alpha_key in sorted(agg.keys(), key=float):
            a = agg[alpha_key]
            folds_note = f" ({a['n_folds']} folds)" if a.get("n_folds") else ""
            print(f"  α={alpha_key}: secure={a['n_secure']}/{a['n']} "
                  f"({a['secure_rate']*100:.1f}%), "
                  f"insecure={a['n_insecure']}/{a['n']} "
                  f"({a['insecure_rate']*100:.1f}%), "
                  f"other={a['n_other']}/{a['n']} "
                  f"({a['other_rate']*100:.1f}%){folds_note}")

        # Compare with prior
        if cwe in prior_results:
            prior_best = prior_results[cwe]["best_rate"]
            prior_alpha = prior_results[cwe]["best_alpha"]
            if best_rate_new > prior_best:
                print(f"  NEW BEST: α={best_alpha_new} ({best_rate_new*100:.1f}%) "
                      f"beats prior α={prior_alpha} ({prior_best*100:.1f}%)")
            else:
                print(f"  Prior α={prior_alpha} ({prior_best*100:.1f}%) still best. "
                      f"New best: α={best_alpha_new} ({best_rate_new*100:.1f}%)")

    # ─── Save full fold data ──────────────────────────────────────────────
    full_path = RESULTS_DIR / f"alpha_extension_{timestamp}_89_78.json"
    with open(full_path, "w") as f:
        json.dump(all_cwe_results, f, indent=2, default=str)
    print(f"\nFull fold data saved: {full_path}")

    # ─── Save aggregated summary + merged with prior ──────────────────────
    # Build merged alpha curves (prior α=0..5 + new α=6..15)
    merged = {}
    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        merged_agg = {}

        # Prior alphas (0..5)
        if cwe in prior_results:
            for alpha_key, data in prior_results[cwe]["aggregated"].items():
                # Ensure other_rate is present
                if "other_rate" not in data:
                    data["other_rate"] = data.get("n_other", 0) / data["n"] if data["n"] > 0 else 0
                merged_agg[alpha_key] = data

        # New alphas from parsed output
        if cwe in all_cwe_results:
            for alpha_key, data in all_cwe_results[cwe]["aggregated"].items():
                merged_agg[alpha_key] = data

        if not merged_agg:
            continue

        # Find overall best
        all_alphas_for_cwe = sorted([float(k) for k in merged_agg.keys()])
        best_alpha = max(all_alphas_for_cwe,
                         key=lambda a: merged_agg[str(a)]["secure_rate"])

        merged[cwe] = {
            "aggregated": merged_agg,
            "all_alphas": all_alphas_for_cwe,
            "best_alpha": best_alpha,
            "best_rate": merged_agg[str(best_alpha)]["secure_rate"],
            "best_other_rate": merged_agg[str(best_alpha)].get("other_rate", 0),
        }

    output_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "03b_lobo_alpha_extension (parsed from output log)",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "layer": 31,
        "new_alpha_grid": NEW_ALPHAS,
        "prior_alpha_grid": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "n_seeds": 10,
        "seeds": [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555],
        "generation_config": {
            "temperature": 0.6,
            "top_p": 0.9,
            "max_new_tokens": 512,
            "do_sample": True,
        },
        "notes": {
            "CWE-89": "Complete: 7/7 folds, all 6 new alphas",
            "CWE-78": "Partial: 6/7 folds (missing fold 7: process_grep). "
                       "Fold 6 (ping_host) has 5/6 alphas (missing α=15.0). "
                       "Process was terminated before completion.",
            "CWE-79": "Not started (process terminated during CWE-78)",
        },
        "cwe_results": {
            cwe: {
                "aggregated": r["aggregated"],
                "best_alpha_new": r["best_alpha_new"],
                "best_rate_new": r["best_rate_new"],
            }
            for cwe, r in all_cwe_results.items()
        },
        "merged_alpha_curves": merged,
    }

    results_path = RESULTS_DIR / f"alpha_extension_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Aggregated results saved: {results_path}")

    # ─── Print combined table ─────────────────────────────────────────────
    all_alphas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] + NEW_ALPHAS
    print(f"\n{'='*100}")
    print("COMBINED ALPHA CURVE (prior + extension)")
    print(f"{'='*100}")

    header = f"{'CWE':<8}"
    for a in all_alphas:
        header += f" {'α='+str(int(a)):>7}"
    header += f" {'Best α':>8} {'Best%':>7}"
    print(header)
    print("-" * len(header))

    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        if cwe not in merged:
            print(f"{cwe:<8} (no data)")
            continue
        row = f"{cwe:<8}"
        for a in all_alphas:
            alpha_key = str(float(a))
            if alpha_key in merged[cwe]["aggregated"]:
                rate = merged[cwe]["aggregated"][alpha_key]["secure_rate"]
                row += f" {rate*100:>6.1f}%"
            else:
                row += f" {'N/A':>7}"
        row += f" {merged[cwe]['best_alpha']:>8}"
        row += f" {merged[cwe]['best_rate']*100:>6.1f}%"
        print(row)

    # Coherence table
    print(f"\n{'='*100}")
    print("COHERENCE: 'other' rate by alpha")
    print(f"{'='*100}")
    header = f"{'CWE':<8}"
    for a in all_alphas:
        header += f" {'α='+str(int(a)):>7}"
    print(header)
    print("-" * len(header))

    for cwe in ["CWE-89", "CWE-78", "CWE-79"]:
        if cwe not in merged:
            print(f"{cwe:<8} (no data)")
            continue
        row = f"{cwe:<8}"
        for a in all_alphas:
            alpha_key = str(float(a))
            if alpha_key in merged[cwe]["aggregated"]:
                d = merged[cwe]["aggregated"][alpha_key]
                n_other = d.get("n_other", 0)
                n_total = d["n"]
                rate = n_other / n_total if n_total > 0 else 0
                row += f" {rate*100:>6.1f}%"
            else:
                row += f" {'N/A':>7}"
        print(row)

    print(f"\nDone. Files saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
