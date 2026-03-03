TITLE: Experiment 22b — Format-Reliability Gap v2 (Code Review Design)
DATE: 2026-03-03
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran redesigned Experiment 22b after Exp 22's 100% accuracy was flagged by the Iron Law. New design uses non-leading code review prompts ("Are there any issues?") with 84 prompts (10 insecure + 4 secure distractors × 6 CWEs). Ran 3 models first with keyword scorer (GPT-4o unavailable), then re-scored all with GPT-4o judge, then re-scored CWE-119 with relaxed judge prompt after finding original GPT-4o was too strict on suggests_fix and secure distractors had real strncpy limitations.

INITIAL PROMPT: Here are the instructions for the new experiment [pasted research journal, manuscript tracking, and Claude Code instructions for Exp 22b]. The python scripts are in the experiments directory. After you are done rename the scripts with our date and document, commit and push per CLAUDE.md.

KEY DECISIONS:
- Assumed Developer role (continuation from Exp 22 session)
- Scripts were already created in claude.ai and placed in experiments directory
- Fixed Qwen model ID: Qwen2-14B → Qwen2.5-14B to match cached model
- Used keyword scorer as fallback when OPENAI_API_KEY was not available in shell
- Ran all 3 models: Llama-8B (~15min), Mistral-7B (~15min), Mistral-24B (~2h at 85s/prompt)
- Renamed directory from exp22b/ to 03-03_exp22b_format_reliability_gap/
- Did NOT manually override any scores (lesson from Exp 22)
- Set up .env file with OPENAI_API_KEY, re-scored all models with GPT-4o
- Investigated CWE-119 failures: GPT-4o too strict on "suggests_fix" + strncpy distractors not unambiguous
- Ran CWE-119 re-scoring with user-provided relaxed judge script (exp22b_rescore_119.py)

SCORING HISTORY:
1. Keyword scorer (initial, OPENAI_API_KEY unavailable)
2. GPT-4o judge (re-scored all CWEs after API key set up)
3. GPT-4o relaxed CWE-119 judge (re-scored only CWE-119 with broader mitigation acceptance)

FINAL RESULTS (GPT-4o + CWE-119 relaxed):
Review Accuracy:
| CWE | Llama-8B | Mistral-7B | Mistral-24B |
|-----|----------|------------|-------------|
| CWE-787 | 90% | 70% | 90% |
| CWE-119 | 90%* | 100%* | 100%* |
| CWE-134 | 20% | 10% | 20% |
| CWE-89 | 100% | 100% | 100% |
| CWE-78 | 60% | 50% | 100% |
| CWE-79 | 50% | 0% | 80% |

*CWE-119 re-scored with relaxed judge (original GPT-4o: 10%, 20%, 30%)

FILES CHANGED:
- src/experiments/03-03_exp22b_format_reliability_gap/exp22b_run.py — Experiment runner (moved from experiments root, fixed Qwen ID)
- src/experiments/03-03_exp22b_format_reliability_gap/exp22b_prompts.py — 84 code review prompts (moved from experiments root)
- src/experiments/03-03_exp22b_format_reliability_gap/exp22b_rescore_119.py — CWE-119 re-scoring with relaxed GPT-4o judge
- src/experiments/03-03_exp22b_format_reliability_gap/results/ — Results for all 3 models (keyword, GPT-4o, and rescored)
- docs/experiments/03-03_multi_model_format_reliability_gap_v2.md — Detailed experiment report (updated with GPT-4o + rescored results)
- docs/research_journal.md — Updated with final GPT-4o scored results
- docs/conversations/2026-03/2026-03-03-0000-exp22b-format-reliability-gap-v2.md — This conversation
- /home/paperspace/MATS/.env — OPENAI_API_KEY (not committed, in .gitignore)
