TITLE: Experiment 22b — Format-Reliability Gap v2 (Code Review Design)
DATE: 2026-03-03
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran redesigned Experiment 22b after Exp 22's 100% accuracy was flagged by the Iron Law. New design uses non-leading code review prompts ("Are there any issues?") with 84 prompts (10 insecure + 4 secure distractors × 6 CWEs). Ran 3 models with keyword scorer (GPT-4o unavailable). Results show real variance: CWE-134 genuinely hard (0-10%), CWE-89 universally recognized (100%), but TN rates reveal response bias in Llama-8B and Mistral-24B.

INITIAL PROMPT: Here are the instructions for the new experiment [pasted research journal, manuscript tracking, and Claude Code instructions for Exp 22b]. The python scripts are in the experiments directory. After you are done rename the scripts with our date and document, commit and push per CLAUDE.md.

KEY DECISIONS:
- Assumed Developer role (continuation from Exp 22 session)
- Scripts were already created in claude.ai and placed in experiments directory
- Fixed Qwen model ID: Qwen2-14B → Qwen2.5-14B to match cached model
- Used keyword scorer as fallback when OPENAI_API_KEY was not available in shell
- Ran all 3 models: Llama-8B (~15min), Mistral-7B (~15min), Mistral-24B (~2h at 85s/prompt)
- Renamed directory from exp22b/ to 03-03_exp22b_format_reliability_gap/
- Did NOT manually override any scores (lesson from Exp 22)

FILES CHANGED:
- src/experiments/03-03_exp22b_format_reliability_gap/exp22b_run.py — Experiment runner (moved from experiments root, fixed Qwen ID)
- src/experiments/03-03_exp22b_format_reliability_gap/exp22b_prompts.py — 84 code review prompts (moved from experiments root)
- src/experiments/03-03_exp22b_format_reliability_gap/results/ — Results for all 3 models
- docs/experiments/03-03_multi_model_format_reliability_gap_v2.md — Detailed experiment report
- docs/research_journal.md — Updated with Experiment 22b entry
- docs/conversations/2026-03/2026-03-03-0000-exp22b-format-reliability-gap-v2.md — This conversation
