TITLE: Experiment 22 — Knowledge-Execution Gap
DATE: 2026-03-02
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran Experiment 22 testing whether LLMs that generate insecure code can correctly explain security principles. Tested 3 models (Llama-8B, Mistral-7B, Mistral-24B) across 6 CWEs. All models achieved 100% knowledge accuracy (after manual review) while generating insecure code 43-100% of the time. Llama-70B failed to load due to OOM with transformers 5.0.

INITIAL PROMPT: Run Experiment 22: a two-condition test comparing each model's security KNOWLEDGE vs its code generation BEHAVIOR. Code generation baselines exist from prior experiments — only need to run the knowledge queries.

KEY DECISIONS:
- Assumed Developer role
- Ran all 4 models in sequence: Llama-8B, Mistral-7B, Mistral-24B, Llama-70B
- Llama-70B OOM'd repeatedly (transformers 5.0 materializes bf16 before quantizing) — skipped as optional
- Identified CWE-89 knowledge scores as scorer false negatives (keyword mismatch, not actual failures)
- Identified Mistral-7B SC-134 and SC-79 as scorer false negatives
- Did NOT modify the scoring script — flagged for manual review per instructions

FILES CHANGED:
- src/experiments/exp22_knowledge_gap/exp22_knowledge_gap.py — New experiment script
- src/experiments/exp22_knowledge_gap/results/ — Results for 3 models
- docs/experiments/03-02_multi_model_knowledge_execution_gap.md — Detailed experiment report
- docs/research_journal.md — Updated with Experiment 22 entry
- docs/conversations/2026-03/2026-03-02-1900-exp22-knowledge-execution-gap.md — This conversation
