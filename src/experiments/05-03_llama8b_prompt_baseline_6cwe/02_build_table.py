#!/usr/bin/env python3
"""Build comparison tables from prompt-baseline summary JSONs.

Emits:
  - Per-CWE x per-condition strict secure rate table (markdown).
  - Side-by-side table comparing prompting baselines vs steering Table 2.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "config"))
import phrasings  # noqa: E402
import datasets as ds  # noqa: E402

# Llama-8B steering Table 2 (strict secure rate at best alpha) — copy from
# docs/research_journal.md SVEN comparison table (committed 947b8d0).
STEERING_TABLE_2 = {
    "CWE-787": {"rate": 0.733, "alpha": 4.0, "note": "from cross-model summary"},
    "CWE-119": {"rate": 0.200, "alpha": "?",  "note": ""},
    "CWE-134": {"rate": 0.749, "alpha": "?",  "note": ""},
    "CWE-89":  {"rate": 0.785, "alpha": 12,   "note": "from extended sweep Exp 10b"},
    "CWE-78":  {"rate": 0.220, "alpha": 5.0,  "note": ""},
    "CWE-79":  {"rate": 0.305, "alpha": 5.0,  "note": ""},
}


def fmt_pct(x):
    return f"{x*100:.1f}%"


def build_main_table(summary: dict) -> str:
    """7 conditions x 6 CWEs strict-secure table."""
    cwes = list(summary["per_cwe_per_condition"].keys())
    conds = phrasings.CONDITION_IDS

    lines = []
    header = "| Condition | " + " | ".join(cwes) + " |"
    sep = "|" + "|".join(["---"] * (len(cwes) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for cid in conds:
        row = [cid]
        for cwe in cwes:
            cell = summary["per_cwe_per_condition"][cwe].get(cid)
            row.append(fmt_pct(cell["strict_secure_rate"]) if cell else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_vs_steering_table(adv_summary: dict) -> str:
    """Side-by-side: best prompting condition vs steering."""
    cwes = list(adv_summary["per_cwe_per_condition"].keys())
    lines = [
        "| CWE | Baseline (no instr) | Best prompt-eng | Steering (Table 2) | Δ (prompt − steering) |",
        "|---|---|---|---|---|",
    ]
    for cwe in cwes:
        per_cond = adv_summary["per_cwe_per_condition"][cwe]
        baseline = per_cond.get("baseline", {}).get("strict_secure_rate", 0)
        # Best non-baseline, non-neutral-control condition
        ranked = []
        for cid, m in per_cond.items():
            if cid in ("baseline", "sys_neutral", "usr_neutral"):
                continue
            ranked.append((cid, m["strict_secure_rate"]))
        ranked.sort(key=lambda x: -x[1])
        best_cid, best_rate = ranked[0] if ranked else (None, 0)

        steer = STEERING_TABLE_2[cwe]["rate"]
        delta = best_rate - steer
        lines.append(
            f"| {cwe} | {fmt_pct(baseline)} | {fmt_pct(best_rate)} ({best_cid}) "
            f"| {fmt_pct(steer)} | {delta*100:+.1f}pp |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to *_summary_*.json")
    ap.add_argument("--neutral-summary", default=None, help="Optional neutral summary for combined output")
    ap.add_argument("--secure-summary", default=None, help="Optional S7 secure-variant summary")
    args = ap.parse_args()

    with open(args.summary) as f:
        adv = json.load(f)

    print("# Prompt-engineering baseline — Llama-3.1-8B-Instruct\n")
    print(f"Variant: **adversarial** (n={adv['metadata']['n_seeds']} seeds, "
          f"T={adv['metadata']['temperature']}, top_p={adv['metadata']['top_p']})\n")
    print("## Strict secure rate per CWE × condition\n")
    print(build_main_table(adv))
    print()
    print("## Best prompting condition vs steering Table 2\n")
    print(build_vs_steering_table(adv))
    print()

    if args.neutral_summary:
        with open(args.neutral_summary) as f:
            neu = json.load(f)
        print("## Neutral deployment set (secondary)\n")
        print(build_main_table(neu))
        print()

    if args.secure_summary:
        with open(args.secure_summary) as f:
            sec = json.load(f)
        print("## Secure-variant prompts (S7 sanity check)\n")
        print(build_main_table(sec))
        print()


if __name__ == "__main__":
    main()
