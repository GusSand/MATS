TITLE: SVEN Baseline Comparison on CodeGen-2B-multi
DATE: 2026-05-02
PARTICIPANTS: Gustavo, Claude
SUMMARY: Set up and ran a head-to-head comparison between SVEN (He & Vechev 2023) prefix steering and our mean-difference activation steering on CodeGen-2B-multi using our 105x6 CWE benchmark. SVEN outperforms our method on 5/6 CWEs on this model. Our method works better on Llama-8B.

INITIAL PROMPT: Task: SVEN baseline on our benchmark. Goal: Produce a head-to-head comparison between SVEN (He & Vechev 2023, eth-sri/sven) and our steering method on our existing 105x6 CWE benchmark, so we can replace the one-sentence punt in §2 of the paper with a real number.

KEY DECISIONS:
- Confirmed SVEN uses CodeGen-multi (not mono), making both C and Python CWEs viable
- Plan (a) as primary: same-model comparison on CodeGen-2B-multi
- Dropped plan (b) (native-model comparison) — not worth GPU hours
- Adapted C prompts mechanically from instruction format to code-completion stubs
- Used SVEN's default generation params (temp=0.4, top_p=0.95) for fairness
- Used SVEN's Python 3.10 venv with torch 1.13.1 for checkpoint compatibility

FILES CHANGED:
- baselines/sven/ — Cloned SVEN repo + venv
- baselines/sven/adapt_c_prompts.py — C prompt adapter
- baselines/sven/run_on_our_benchmark.py — SVEN inference wrapper
- baselines/sven/run_baseline_codegen.py — Unsteered CodeGen baseline
- baselines/sven/run_steering_codegen.py — Our steering on CodeGen with LOBO
- baselines/sven/smoke_test.py — SVEN verification script
- baselines/sven/adapted_prompts/ — 6 adapted prompt JSONL files
- baselines/sven/results/ — All result JSON files
- outputs/sven_comparison.md — Comparison report
- docs/research_journal.md — Added SVEN comparison entry
- docs/experiments/05-02_codegen2b_sven_baseline_comparison.md — Detailed experiment report
- docs/DATA_INVENTORY.md — Added SVEN comparison data entries
