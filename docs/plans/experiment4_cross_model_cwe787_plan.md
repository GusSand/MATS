# Experiment 4: Cross-Model CWE-787 Steering Validation

## Goal
Test whether Mistral-7B-Instruct-v0.3 and Llama-3.1-70B-Instruct (8-bit) show the same CWE-787 secure code improvement as Llama-3.1-8B-Instruct (0% → 52.4% secure with LOBO steering).

---

## Phase 0: Environment Setup (~5 min with pre-built image)

User creates a Paperspace image with NVIDIA drivers + CUDA + PyTorch pre-installed. Then:

```bash
pip install transformers accelerate bitsandbytes numpy scipy scikit-learn tqdm matplotlib seaborn huggingface_hub
huggingface-cli login  # Both Llama and Mistral v0.3 are gated models
```

Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

---

## Phase 1: Shared Infrastructure (New Files)

Create `src/experiments/02-05_cross_model_cwe787_steering/` with:

### Directory Structure
```
02-05_cross_model_cwe787_steering/
├── shared/
│   ├── __init__.py
│   ├── model_loader.py           # Unified model loading (fp16 + 8-bit quantization)
│   ├── activation_collector.py   # Model-agnostic activation collection at all layers
│   ├── layer_sweep.py            # Linear probe sweep to find optimal steering layer
│   └── steering_generator.py     # Model-agnostic steering + generation
├── experiment_4a_mistral7b/
│   ├── experiment_config.py      # Mistral-specific config
│   ├── 01_baseline_behavior.py   # Unsteered baseline rates
│   ├── 02_collect_activations.py # Activations at all 32 layers
│   ├── 03_layer_sweep.py         # Probe sweep → find optimal layer
│   ├── 04_pilot_lobo.py          # 2-fold pilot to validate steering works
│   ├── 05_full_lobo.py           # Full 7-fold LOBO (if pilot positive)
│   └── data/
├── experiment_4b_llama70b/
│   ├── experiment_config.py      # 70B-specific config (8-bit, 80 layers, 8192 dim)
│   ├── 01_baseline_behavior.py
│   ├── 02_collect_activations.py
│   ├── 03_layer_sweep.py
│   ├── 04_pilot_lobo.py
│   ├── 05_full_lobo.py
│   └── data/
└── 06_cross_model_analysis.py    # Comparison figures + tables
```

### Key Shared Components

**`shared/model_loader.py`** — Unified loading for Llama/Mistral, fp16 or 8-bit:
- `ModelLoader(model_name, quantization=None|"8bit")`
- Exposes: `.model`, `.tokenizer`, `.n_layers`, `.hidden_size`, `.get_layer(idx)`
- Both Mistral and Llama use `model.model.layers[N]` as hook target

**`shared/activation_collector.py`** — Collects last-token activations:
- Adapted from `01-08_llama8b_sr_scg_separation/utils/activation_collector.py`
- `collect_dataset(dataset, layers)` → runs all 210 prompts, saves NPZ
- Hook pattern: `output[0][:, -1, :].detach().cpu()` (same as existing)

**`shared/layer_sweep.py`** — Probe-based optimal layer selection:
- Trains logistic regression at each layer, reports 5-fold CV accuracy
- Also computes mean-diff direction norm and cluster separation
- `select_top_layers(results, n=5)` → top-5 candidates for steering tests

**`shared/steering_generator.py`** — Steering + generation:
- Adapted from `01-12_llama8b_cwe787_lobo_steering/run_experiment.py:45-114`
- `generate_with_steering(prompt, direction, layer, alpha, ...)` → steered text
- 8-bit note: activations are float16 even in quantized models, so steering hooks work identically

### Reused Existing Code (Unchanged)
- **Dataset**: `01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl` (105 pairs)
- **Scoring**: `01-12_llama8b_cwe787_baseline_behavior/scoring.py` (via sys.path import)
- **Refusal detection**: `01-12_llama8b_cwe787_baseline_behavior/refusal_detection.py`
- **LOBO split logic**: Adapted from `01-12_llama8b_cwe787_lobo_steering/lobo_splits.py` (parameterized layer instead of hardcoded L31)
- **Scoring patterns**: Copied into each experiment_config.py (existing codebase convention)

---

## Phase 2: Experiment 4A — Mistral-7B-Instruct-v0.3

**Model**: `mistralai/Mistral-7B-Instruct-v0.3` (32 layers, 4096 hidden dim, ~14GB fp16)

### Step 2.1: Baseline Behavior (~15 min)
- Generate 1 completion per 105 vulnerable prompts (no steering)
- Score with STRICT + EXPANDED + refusal detection
- **Decision gate**: If baseline secure rate > 50%, steering has limited room → report as finding

### Step 2.2: Collect Activations (~105 min)
- All 32 layers × 210 prompts × 4096 dim
- Output: `activations_TIMESTAMP.npz` (~50-80 MB) + `metadata_TIMESTAMP.json`

### Step 2.3: Layer Sweep (<1 min)
- Linear probes at all 32 layers
- Select top-5 layers by CV accuracy
- Output: `layer_sweep_results.json` + probe accuracy plot

### Step 2.4: Pilot LOBO (~42 min)
- Top-1 layer from sweep
- 2 LOBO folds × α∈{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0} × ~15 test prompts
- **Decision gate**: If best α achieves >10% secure improvement → proceed to full LOBO

### Step 2.5: Full LOBO (~2 hr, only if pilot positive)
- 7 folds × same α grid × ~15 test prompts each
- Same output format as existing LOBO results for direct comparison

---

## Phase 3: Experiment 4B — Llama-3.1-70B-Instruct (8-bit)

**Model**: `meta-llama/Meta-Llama-3.1-70B-Instruct` (80 layers, 8192 hidden dim, ~70GB 8-bit)

### Step 3.1: Baseline Behavior (~36 min)
- Same as Mistral but slower generation (~25 tok/s vs ~60 tok/s)
- **Critical check**: 70B may already be substantially more secure at baseline

### Step 3.2: Collect Activations (~14 min)
- All 80 layers × 210 prompts × 8192 dim
- Output: ~200-300 MB compressed NPZ
- Memory: activations moved to CPU immediately via hooks, fits in 88GB RAM

### Step 3.3: Layer Sweep (<2 min)
- Probes at all 80 layers
- Expected optimal: layers 65-79 (same ~80-97% relative depth as Llama 8B's L31/32)

### Step 3.4: Pilot LOBO (~90 min)
- 2 folds × α∈{0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0}
- Wider α grid — optimal α may differ due to larger hidden dim / different direction norms

### Step 3.5: Full LOBO (~5.25 hr, only if pilot positive)
- 7 folds × same α grid

---

## Phase 4: Cross-Model Analysis

File: `06_cross_model_analysis.py`

### Outputs
- **Table 1**: Model comparison (baseline %, best steered %, optimal layer, optimal α, direction norm)
- **Figure 1**: Probe accuracy vs relative layer depth (all 3 models overlaid)
- **Figure 2**: STRICT secure % vs α (all 3 models at their optimal layers)
- **Figure 3**: Per-fold secure rates heatmap (models × folds × optimal α)

---

## Execution Order & Decision Gates

```
[0] Environment setup (pre-built image + pip install)        ~5 min
[1] Create shared/ infrastructure + smoke tests              ~30 min code writing
[2] 4A: Mistral baseline                                     ~15 min
    └─ GATE: Is baseline insecure rate > 50%?
[3] 4A: Mistral activation collection                        ~105 min
[4] 4A: Mistral layer sweep                                  <1 min
[5] 4A: Mistral pilot (2 folds)                              ~42 min
    └─ GATE: Does steering improve >10pp?
[6] 4A: Mistral full LOBO                                    ~2 hr
[7] Unload Mistral, load Llama 70B                           ~5 min
[8] 4B: Llama 70B baseline                                   ~36 min
    └─ GATE: Same check
[9] 4B: Llama 70B activation collection                      ~14 min
[10] 4B: Llama 70B layer sweep                               <2 min
[11] 4B: Llama 70B pilot (2 folds)                           ~90 min
    └─ GATE: Same check
[12] 4B: Llama 70B full LOBO                                 ~5.25 hr
[13] Cross-model analysis + figures                          ~15 min
```

**Total estimated**: ~10-12 hours (if both models go to full LOBO)

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mistral doesn't follow raw task prompts well | Garbage output | Baseline step will reveal; wrap in `[INST]...[/INST]` if needed |
| 70B already secure at baseline | No room for improvement | Report as finding — "larger models don't need steering" |
| Alpha scale differs for 70B | Miss optimal α | Extended grid up to 10.0; also compute α/direction_norm |
| 70B OOM during generation + hooks | Crash | Hooks add only 16KB per token (negligible); monitor with nvidia-smi |
| Direction norms incomparable across models | Misleading comparison | Report both raw and normalized (α × norm) metrics |

---

## Verification Approach

1. **Smoke tests** before each major step (model generates C code, hooks fire, scoring works)
2. **Data integrity**: Assert activation shapes, no NaN/Inf, correct label counts (105+105)
3. **Quantization check**: Verify hook outputs are float16/32 even with 8-bit weights
4. **Reproducibility**: `torch.manual_seed(42)`, save configs + git hash in all output JSON
5. **Comparison validity**: Same dataset, same scoring, same LOBO splits, same generation config (temp=0.6, top_p=0.9, max_tokens=512)
