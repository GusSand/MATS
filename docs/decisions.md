# Decisions

Append-only. One line per settled choice: `YYYY-MM-DD · choice → link`.
2026-11-22 · R13 reproduction gate waived (942/1,000 vs 950; one-directional drift on marginal subsets, hardware differs from March) → docs/experiments/11-21_llama8b_decimal_true_head_patching.md
2026-11-22 · Rerun Pythia-160M and Gemma-2B head claims with the R13 protocol; Mistral head claims dropped unless code appears (recommendation, Gus said "rerun") → docs/reviews/11-22_cross_model_head_scripts_audit.md
2026-11-22 · Legacy experiment files stay grandfathered; any legacy file whose numbers v2 cites is re-reviewed before those numbers go into the draft → docs/inbox_archive.md
2026-11-22 · R14 Pythia reproduction gate upheld: drop Pythia and Gemma head-level claims (incl. Pythia late emergence) from v2; keep all four models in v2 with a row each in the cross-model table (Pythia: failed reproduction; Gemma: behavioral premise absent) → docs/experiments/11-22_pythia_gemma_true_head_patching.md
2026-11-23 · Don't withdraw Even Heads yet: run a three-stage search for what repairs the error under correct patching (R17 map → heads where the map points → Mistral layer 7) → docs/experiments/11-23_llama8b_localization_map.md
