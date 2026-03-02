# Manuscript Notes & Tracking — NeurIPS 2026 Submission

**Working Title**: Surgical Repair of Insecure Code Generation in LLMs: From Mechanistic Diagnosis to Deployment-Ready Intervention
**Target**: NeurIPS 2026 (deadline TBD, first draft March 13, 2026)
**ArXiv**: By April 21, 2026 (before defense April 24)
**Last updated**: March 2, 2026

---

## 1. Key Claims to Support with Evidence

| # | Claim | Supporting Experiment | Status |
|---|-------|----------------------|--------|
| 1 | LLMs generate insecure code due to attention competition, not knowledge gaps | Exp 22b (code review vs generation gap) + activation patching, logit lens, attention head analysis | ✅ Mechanistic done; 🔲 Exp 22b behavioral evidence RUNNING |
| 2 | Security representations are distributed, not localized (hierarchical convergence L0→L31) | Linear probing across layers, information emergence analysis | ✅ Done |
| 3 | Steering vectors reduce insecure code by 74% (94.3% → 24.8%) | LOBO cross-validation, CodeQL scoring | ✅ Done |
| 4 | Surgical repair of 8 attention heads achieves 100% accuracy | Targeted head patching (L25 even/odd heads) | ✅ Done |
| 5 | Cross-CWE transfer fails — vectors are vulnerability-specific | Cross-CWE steering experiments, 6×6 transfer matrix | ✅ Done |
| 6 | Probe-gated routing achieves 95.2% accuracy for CWE classification | Exp 8.5 binary probe at L16 | ✅ Done |
| 7 | 2-tier deployment achieves 88.6% overall secure rate | Exp 8.5 E2E pipeline | ✅ Done |
| 8 | Overhead is <3.1% (102% was measurement artifact from token count) | Exp 9 — overhead resolved | ✅ Done |
| 9 | Results generalize across Python CWEs (CWE-89, 78, 79) | Exp 10, 10b Python expansion | ✅ Done |
| 10 | Results validated across 5 models / 3 architecture families | Exps 11-27 cross-model validation | ✅ Done (Llama 8B/70B, Mistral 7B/24B, Qwen 14B) |

---

## 2. Model Framing Decision (March 2, 2026)

**5 model checkpoints across 3 architecture families:**

| Family | Checkpoints | Params |
|--------|------------|--------|
| Llama 3.1 | 8B-Instruct, 70B-Instruct | 8B, 70B |
| Mistral | 7B-Instruct-v0.3, Small-24B-Instruct | 7B, 24B |
| Qwen | 2.5-14B-Instruct | 14B |

**Paper framing**: "We validate across five model checkpoints spanning three architecture families (Llama-3.1-8B and 70B, Mistral-7B and Small-24B, Qwen-14B), testing both cross-architecture generalization and within-family scaling from 7B to 70B parameters."

Within-family comparisons (Llama 8B→70B, Mistral 7B→24B) test **scale** within an architecture. Cross-family comparisons (Llama vs Mistral vs Qwen) test **architecture**. Both are features, not limitations.

---

## 3. Experiment 22: Format-Reliability Gap (Behavioral Evidence)

### Exp 22a (COMPLETED — ceiling effect, use for appendix only)
- **Design**: Direct knowledge queries ("What is the difference between sprintf and snprintf?")
- **Result**: 100% knowledge accuracy across all models and CWEs (after manual review)
- **Problem**: Leading prompts — named both vulnerability and fix. Tests comprehension, not knowledge retrieval.
- **Self-critique**: 18/18 pass (models correctly identify insecure code when shown it)
- **Use in paper**: One sentence — "When directly queried about security principles, models achieve 100% accuracy (Appendix X)"
- **CWE-89 keyword scorer false negatives**: Models used "directly inserted"/"not properly sanitized" instead of "concatenation"/"f-string". Manual override to 100%. This exposed scoring fragility.

### Exp 22b (RUNNING — primary behavioral evidence for Section 3.1)
- **Design**: Non-leading code review. Show insecure code, ask "Review this function. Are there any issues?" No mention of security, vulnerability type, or fix.
- **Scoring**: GPT-4o as judge with structured rubric (identifies_issue / names_vulnerability / suggests_fix → JSON)
- **Distractors**: 4 secure code snippets per CWE to measure true negative rate (prevents gaming by always saying "insecure")
- **Prompts**: 84 total (10 insecure + 4 secure × 6 CWEs)
- **Models**: Llama-8B, Mistral-7B (minimum); Mistral-24B if time allows
- **Expected**: Review accuracy 80-100% with true negative rate >70%, vs code generation baseline 0-57%
- **Gap**: Review Accuracy − Code Generation Security Rate = the Format-Reliability Gap, quantified
- **Scripts**: `exp22b_prompts.py` (84 prompts) + `exp22b_run.py` (runner with GPT-4o judge)
- **Output**: Gap Table → Table 1 or Figure in Section 3.1

### Key framing for paper:
The gap table shows models can RECOGNIZE vulnerabilities in code review but FAIL TO AVOID them during generation. The write-up paragraph (Section 3.1):

> "To establish that insecure code generation reflects an execution failure rather than a knowledge deficit, we presented each model with short code snippets containing security vulnerabilities and asked simply 'Are there any issues?' — with no mention of security, the vulnerability class, or the expected fix. Models correctly identified the vulnerability and suggested the secure alternative in X% of cases. To control for response bias, we also presented equivalent secure code; models correctly recognized these as safe Y% of the time. Despite this demonstrated ability to detect and diagnose vulnerabilities on inspection, the same models produced secure code in only Z% of generation trials. This [X − Z]-percentage-point gap between recognition and generation — which we term the Format-Reliability Gap — holds across all six CWE types."

---

## 4. Figures & Tables Needed

### Figures
- [ ] **Fig 1**: Overview diagram — "Outside-In, Inside-Out" research framework
- [ ] **Fig 2**: Attention head functional clustering (L25 even=decimal, odd=format tokens)
- [ ] **Fig 3**: Information emergence across layers (L0 early encoding → L31 sudden emergence)
- [ ] **Fig 4**: Steering vector effect — before/after secure code rate across CWEs
- [ ] **Fig 5**: Probe-then-steer architecture diagram
- [ ] **Fig 6**: Latency comparison (controlled, equal token counts)
- [ ] **Fig 7**: Cross-CWE transfer matrix (6×6 heatmap, 3.8x diagonal dominance)
- [ ] **Fig 8** (if space): SAE decomposition of security-relevant features

### Tables
- [ ] **Table 0/1**: Format-Reliability Gap (from Exp 22b) — Review Accuracy vs Code Generation per CWE
- [ ] **Table 2**: Steering vector results per CWE with LOBO CV
- [ ] **Table 3**: Probe routing accuracy by layer and training method (from Exp 8.5 Part A)
- [ ] **Table 4**: 2-tier deployment — strategy comparison (from Exp 8.5 Part B)
- [ ] **Table 5**: E2E pipeline results (from Exp 8.5 Part C + Exp 9 timing)
- [ ] **Table 6**: Latency benchmark (Exp 9, controlled comparison, all methods <3.1%)
- [ ] **Table 7**: Python CWE expansion results (Exp 10, 10b)
- [ ] **Table 8**: Cross-model validation — 5 models × CWE-787 + CWE-89

---

## 5. Paper Structure & Draft Status

| Section | Status | Notes |
|---------|--------|-------|
| **Abstract** | ✅ DRAFTED | ~200 words. In working draft v0.1. |
| **1. Introduction** | ✅ DRAFTED | 3-act structure: problem (automation bias), insight (Format-Reliability Gap), solution (surgical steering). Contribution 1 paragraph needs compression — move L25 detail to Section 4. In working draft v0.1. |
| **2. Background & Related Work** | 🔲 WRITE LAST | Positioning: Emergent Misalignment (Betley 2025), RESTA (2024), Even Heads (2025), Goodfire/Rakuten (Nguyen 2025), Lost at C (Sandoval 2023) |
| **3. Format-Reliability Gap** | 🔲 WRITE NEXT | Needs Exp 22b results. Behavioral evidence → Latent Interference Hypothesis. Consider softer framing than "hypothesis" for main text. |
| **4. Mechanistic Analysis** | 🔲 TODO | Probes, logit lens, hierarchical convergence, L25 heads, patching, SAE. Most data-rich section. |
| **5. Surgical Intervention** | 🔲 TODO | LOBO results, per-CWE tables, transfer matrix |
| **6. Deployment Architecture** | 🔲 TODO | Probe routing, E2E pipeline, overhead benchmarks |
| **7. Cross-Language & Cross-Model** | 🔲 TODO | 5 models, 6 CWEs, mechanistic replication |
| **8. Discussion** | 🔲 TODO | Limitations, emergent misalignment connection, future work |
| **9. Conclusion** | 🔲 TODO | |

**Writing order**: Introduction ✅ → Section 3 (NEXT) → Section 5 → Section 4 → Section 6 → Section 7 → Section 8 → Section 2 (LAST)

**Working draft**: `/mnt/user-data/outputs/manuscript_working_draft_v01.md` (contains Abstract + Introduction)

---

## 6. Running List of Points to Include

### From Exp 8.5 Discussion (Feb 9, 2026)

1. **Performance bottleneck is architectural, not computational**: The 102% overhead comes from PyTorch hook dispatch breaking CUDA graphs, not from the steering arithmetic. Important general lesson for deploying activation interventions.

2. **Probe-then-steer as the solution**: Cite Goodfire/Rakuten (Nguyen et al., 2025) for sidecar/decoupled architecture pattern. Separate classification from generation.

3. **Distribution shift in probes**: Adversarial-trained probes fail on neutral prompts (66.7% → fixed to 95.2% with neutral retraining). Same phenomenon Goodfire found — SAE probes generalize better across distribution shifts.

4. **Data leakage bug (Iron Law compliance)**: Method 4 (Mixed+Augmented) showed spurious 100% accuracy. Caught and flagged. Brief mention as evidence of rigor.

5. **Binary probe sufficient**: 2-tier binary routing only costs 5.7pp vs perfect routing. L16 binary probe achieves 100% on neutral LOO.

6. **CWE-119 is the weak point**: Gets 64.3% with buffer vector (non-native). Bimodal: gets→fgets works 82-91%, strcpy→strncpy fails. Discuss in limitations.

### From Mechanistic Analysis

7. **Format-Reliability Gap**: Core theoretical contribution. Models can explain security principles but fail to execute under format pressure. Now supported by Exp 22b behavioral evidence.

8. **Latent Interference Hypothesis**: Format requirements create interference suppressing security representations. Consider softer framing — evidence is strong but causal ablation of format tokens not done.

9. **Hierarchical convergence**: Security encoded at L0 (100% probe accuracy) but suppressed until L31 emergence (logit lens: 0.0001%→37%). Architecturally distinct from known circuits.

10. **Even/odd head specialization in L25**: Even=decimal logic, odd=format tokens. Novel finding.

### From Cross-Model Validation (Exps 11-27)

11. **Mistral-7B**: Distributed emergence L21-28 (vs Llama's sharp L31). Tokenizer forces multi-token planning ("snprintf"→["sn","printf"]). Superior steering effectiveness with minimal correctness penalty.

12. **Llama-70B**: Late emergence L75-79. Peak lower (52.4%) than 7B models (~73%) — may be quantization artifact (4-bit) or stronger internal resistance.

13. **Mistral-Small-24B**: CWE-787 and CWE-119 LOBO completed. Logit lens done. Results pending full analysis.

14. **Qwen-14B**: CWE-787 (+50.5pp), CWE-89 (+15.6pp), CWE-119 completed.

15. **Functional correctness (Exp 25-27)**: Neutral prompt penalty is -4.8pp (Llama) / -9.5pp (Mistral), much smaller than adversarial penalty (-36pp / -12pp). Correctness penalties concentrated on adversarial prompts.

### Positioning & Related Work

16. **Emergent Misalignment (Betley et al., 2025)**: They observe phenomenon (fine-tuning on insecure code → broad misalignment), we explain mechanism and provide fix. Complementary.

17. **RESTA (2024)**: Task arithmetic for safety. Broader but less surgical (0.8% vs full model).

18. **Even Heads (2025)**: Attention head analysis methodology. Our L25 finding extends to security.

19. **Goodfire/Rakuten (Nguyen et al., 2025)**: Probe-based routing, SAE generalization. Cite for sidecar architecture pattern.

20. **Lost at C (Sandoval et al., 2023)**: Our own prior work. Established behavioral problem (83% acceptance of vulnerable code, 36% bugs from LLM suggestions).

### Reviewer Anticipation

21. **"Why not just fine-tune?"**: Surgical, interpretable, reversible. Fine-tuning risks catastrophic forgetting.

22. **"Only 3 CWE types"**: Now 6 CWEs across 2 languages (C: 787, 119, 134; Python: 89, 78, 79).

23. **"Single model"**: Now 5 models across 3 architecture families.

24. **"102% overhead"**: Resolved — measurement artifact from token count differences. All methods <3.1%.

25. **"How does this compare to CodeQL?"**: Preventive (generation-time) vs detective (post-hoc). Complementary. We use CodeQL as ground truth.

26. **"The knowledge test is trivial"**: Addressed by Exp 22b redesign — non-leading prompts with secure distractors and GPT-4o judge.

---

## 7. Key Results Summary (for quick reference)

### Cross-Model CWE-787 Steering
| Model | Baseline | Best Steered | Improvement |
|-------|----------|-------------|-------------|
| Llama-8B | 6.7% | 73.3% (α=4.0) | +66.6pp |
| Mistral-7B | 3.8% | 74.3% | +70.5pp |
| Qwen-14B | 3.8% | 54.3% | +50.5pp |
| Llama-70B | 1.9% | 52.4% | +50.5pp |

### Cross-Model CWE-89 Steering
| Model | Baseline | Best Steered | Improvement |
|-------|----------|-------------|-------------|
| Llama-8B | 57% | 78.5% (α=12.0) | +21.5pp |
| Mistral-7B | 42.9% | 63.5% | +20.6pp |
| Qwen-14B | 38.4% | 54% | +15.6pp |
| Llama-70B | 52.1% | 60.6% | +8.6pp |

### Python CWE Expansion (Llama-8B)
| CWE | Baseline | Best Steered | Improvement |
|-----|----------|-------------|-------------|
| CWE-89 | 57% | 78.5% (α=12.0) | +21.5pp |
| CWE-78 | 14.3% | 22% (α=5.0) | +7.7pp |
| CWE-79 | 0.2% | 30.5% (α=5.0) | +30.3pp |

### Transfer Matrix (6×6)
Diagonal avg 49.9%, off-diagonal 13.1% (3.8x ratio). Confirms vulnerability-specific vectors.

### E2E Pipeline
88.6% overall secure rate, <3.1% overhead. Probe at L16: 95.2% routing accuracy.

### Functional Correctness (Neutral Prompts)
Llama-8B: -4.8pp penalty. Mistral-7B: -9.5pp penalty. Much smaller than adversarial.

---

## 8. Timeline & Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| Feb 10-14 | Exp 9 (probe-then-steer), benchmarks | ✅ Done |
| Feb 15-21 | Python CWE expansion (CWE-89, 78, 79) | ✅ Done |
| Feb 22-28 | Cross-model validation (Mistral-7B, Llama-70B, Qwen-14B, Mistral-24B) | ✅ Done |
| Mar 1-2 | Exp 22 behavioral evidence; paper scaffolding; Intro + Abstract drafted | ✅ Done |
| Mar 3-6 | **Exp 22b results**; Section 3 draft; Section 5 draft | 🔲 Current |
| Mar 7-13 | **First draft complete** — all sections | 🔲 Target |
| Mar 14-28 | Revisions, figure polish, related work pass | |
| Apr 1-14 | Final revisions, advisor review | |
| Apr 21 | **ArXiv submission** | |
| Apr 24 | **Dissertation defense** | |

---

## 9. Open Questions

- [RESOLVED] "Latent Interference Hypothesis" framing → use softer language in main text, stronger in discussion
- [RESOLVED] Model count → "5 model checkpoints across 3 architecture families"
- [RESOLVED] 102% overhead → measurement artifact, all methods <3.1%
- [RESOLVED] Knowledge gap citation → no existing citation, run our own experiment (Exp 22b)
- [OPEN] "0.8% of parameters" claim needs verification for final architecture
- [OPEN] Goodfire citation: check if formal publication venue or cite as technical report
- [OPEN] SAE decomposition: include in main paper or appendix?
- [OPEN] How much mechanistic detail fits in 9 NeurIPS pages?
- [OPEN] Two steering-resistant Python folds (admin_delete, user_profile_update) — discuss in limitations
- [OPEN] 70B results: peak lower than 7B models — quantization artifact or genuine? Frame carefully in Section 7
