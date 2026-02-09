#!/usr/bin/env python3
"""
Experiment 9b: Probe-Then-Steer Timing Benchmark

Compares 5 approaches:
1. Baseline (no steering, no hooks)
2. Hook-based steering (Exp 8.5 approach — register_forward_hook during generate)
3. Probe-then-steer Option A (monkey-patch forward)
4. Probe-then-steer Option B (torch.compile)
5. Probe-then-steer Option D (weight bias — zero Python wrapper)

Protocol: 50 iterations, max_new_tokens=64, torch.cuda.synchronize() before timing.
5 warmup iterations for torch.compile. Uses first neutral prompt for benchmarking.

Also reports probe routing accuracy vs Exp 8.5 ground truth.
"""
import sys
import time
import json
import functools
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering")
PROBE_DATA_DIR = Path("/home/paperspace/MATS/src/experiments/02-08_probe_routing_v2/data")
VECTORS_DIR = EXPERIMENT_DIR / "cross_cwe_analysis" / "data"
PROMPTS_PATH = EXPERIMENT_DIR / "neutral_eval" / "data" / "neutral_eval_prompts.jsonl"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(EXPERIMENT_DIR / "shared"))
sys.path.insert(0, str(SCRIPT_DIR))

from model_loader import ModelLoader
from probe_router import ProbeRouter
from steered_generator import SteeredGenerator

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31
BENCHMARK_ITERATIONS = 50
MAX_NEW_TOKENS_BENCH = 64  # Matches Exp 8.5 protocol

ALPHAS = {
    "buffer": 4.0,
    "format_string": 1.0,
}

VECTOR_FILES = {
    "buffer": "direction_cwe787_L31_20260206_031901.npy",
    "format_string": "direction_cwe134_L31_20260206_031901.npy",
}


# ─── Chat prompt formatting (matches Exp 8.5) ───────────────────────────────

def format_chat_prompt(tokenizer, code_prefix):
    user_message = ("Complete the following C function. Only write the "
                    "function body, no explanation.\n\n" + code_prefix)
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─── Hook-based steering (Exp 8.5 baseline for comparison) ──────────────────

def make_steering_hook(alpha_val, vec):
    """Exp 8.5-style hook — fires on every generated token."""
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[:, -1, :] = h[:, -1, :] + alpha_val * vec
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 9b: Probe-Then-Steer Timing Benchmark")
    print(f"Model: {MODEL_NAME}")
    print(f"Iterations: {BENCHMARK_ITERATIONS}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS_BENCH}")
    print("=" * 70)

    # ─── Load model ──────────────────────────────────────────────────────
    print("\nLoading model...")
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    # ─── Load steering vectors ───────────────────────────────────────────
    vectors_np = {}
    for key, fname in VECTOR_FILES.items():
        vec = np.load(VECTORS_DIR / fname).astype(np.float32)
        vectors_np[key] = vec
        print(f"  Loaded {key} vector: norm={np.linalg.norm(vec):.4f}")

    # ─── Load neutral prompts ────────────────────────────────────────────
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} neutral prompts")

    # ─── Initialize components ───────────────────────────────────────────
    router = ProbeRouter(model, tokenizer, str(PROBE_DATA_DIR), probe_layer=LAYER)
    generator = SteeredGenerator(model, tokenizer, vectors_np, ALPHAS,
                                 steering_layer=LAYER)

    # ─── Probe routing accuracy check ────────────────────────────────────
    print(f"\n{'='*70}")
    print("ROUTING ACCURACY CHECK")
    print(f"{'='*70}")

    correct = 0
    total = len(prompts)
    for p in prompts:
        formatted = format_chat_prompt(tokenizer, p["prompt"])
        cwe_type, conf = router.classify(formatted)
        true_type = "format_string" if p["cwe"] == "CWE-134" else "buffer"
        is_correct = cwe_type == true_type
        if is_correct:
            correct += 1
        status = "OK" if is_correct else "MISS"
        print(f"  {p['id']} [{p['cwe']}] → {cwe_type} (conf={conf:.3f}) [{status}]")

    route_accuracy = correct / total
    print(f"\nRouting accuracy: {correct}/{total} ({route_accuracy*100:.1f}%)")

    # ─── Prepare benchmark prompt ────────────────────────────────────────
    bench_prompt = format_chat_prompt(tokenizer, prompts[0]["prompt"])
    bench_inputs = tokenizer(bench_prompt, return_tensors="pt").to(device)
    bench_cwe = "buffer"  # First prompt is CWE-787

    vec_tensor = torch.tensor(vectors_np["buffer"], dtype=torch.float16).to(device)

    # ─── Token count diagnostic ──────────────────────────────────────────
    print(f"\nToken count diagnostic (no min_new_tokens):")
    with torch.no_grad():
        out_base = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id)
    n_base = out_base.shape[1] - bench_inputs.input_ids.shape[1]

    hook_diag = model.model.layers[LAYER].register_forward_hook(
        make_steering_hook(ALPHAS["buffer"], vec_tensor))
    with torch.no_grad():
        out_steered = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
    hook_diag.remove()
    n_steered = out_steered.shape[1] - bench_inputs.input_ids.shape[1]
    print(f"  Baseline tokens: {n_base}/{MAX_NEW_TOKENS_BENCH}")
    print(f"  Steered tokens:  {n_steered}/{MAX_NEW_TOKENS_BENCH}")
    if n_base != n_steered:
        print(f"  ⚠ Token count mismatch! Using min_new_tokens={MAX_NEW_TOKENS_BENCH} to control.")

    # ─── Warmup ──────────────────────────────────────────────────────────
    print(f"\nWarming up (5 iterations)...")
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(**bench_inputs, max_new_tokens=1,
                               pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()

    results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 1: Baseline (no steering, no hooks)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[1/7] Baseline generation ({BENCHMARK_ITERATIONS} iters)...")
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                               do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    results["baseline"] = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "all_ms": [float(t) for t in times],
    }
    print(f"  mean={results['baseline']['mean_ms']:.1f}ms, "
          f"std={results['baseline']['std_ms']:.1f}ms, "
          f"p95={results['baseline']['p95_ms']:.1f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 2: Hook-based steering (Exp 8.5 approach)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[2/7] Hook-based steering ({BENCHMARK_ITERATIONS} iters)...")
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        hook = model.model.layers[LAYER].register_forward_hook(
            make_steering_hook(ALPHAS["buffer"], vec_tensor)
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                               do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        hook.remove()

    results["hook_based"] = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "all_ms": [float(t) for t in times],
    }
    print(f"  mean={results['hook_based']['mean_ms']:.1f}ms, "
          f"std={results['hook_based']['std_ms']:.1f}ms, "
          f"p95={results['hook_based']['p95_ms']:.1f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 3: Probe-then-steer Option A (monkey-patch)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[3/7] Probe-then-steer Option A: monkey-patch ({BENCHMARK_ITERATIONS} iters)...")
    times_probe = []
    times_gen = []
    times_total = []
    for _ in range(BENCHMARK_ITERATIONS):
        # Phase 1: Probe
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cwe_type, _ = router.classify(bench_prompt)
        torch.cuda.synchronize()
        t_probe = (time.perf_counter() - t0) * 1000

        # Phase 2: Steered generation (monkey-patch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with generator._apply_steering_monkeypatch(bench_cwe):
            with torch.no_grad():
                _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                                   do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        t_gen = (time.perf_counter() - t1) * 1000

        times_probe.append(t_probe)
        times_gen.append(t_gen)
        times_total.append(t_probe + t_gen)

    results["monkeypatch"] = {
        "probe_mean_ms": float(np.mean(times_probe)),
        "gen_mean_ms": float(np.mean(times_gen)),
        "mean_ms": float(np.mean(times_total)),
        "std_ms": float(np.std(times_total)),
        "p95_ms": float(np.percentile(times_total, 95)),
        "all_ms": [float(t) for t in times_total],
    }
    print(f"  probe={results['monkeypatch']['probe_mean_ms']:.1f}ms, "
          f"gen={results['monkeypatch']['gen_mean_ms']:.1f}ms, "
          f"total={results['monkeypatch']['mean_ms']:.1f}ms "
          f"(std={results['monkeypatch']['std_ms']:.1f}ms)")

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 4: Probe-then-steer Option B (torch.compile)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[4/7] Probe-then-steer Option B: torch.compile ({BENCHMARK_ITERATIONS} iters)...")
    print("  Warming up torch.compile (5 iterations)...")

    # Warmup for compilation
    compile_ok = True
    try:
        for _ in range(5):
            with generator._apply_steering_compiled(bench_cwe):
                with torch.no_grad():
                    _ = model.generate(**bench_inputs, max_new_tokens=1,
                                       do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
    except Exception as e:
        print(f"  torch.compile failed during warmup: {e}")
        print("  Falling back — Option B results will mirror Option A")
        compile_ok = False

    times_probe = []
    times_gen = []
    times_total = []
    for _ in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cwe_type, _ = router.classify(bench_prompt)
        torch.cuda.synchronize()
        t_probe = (time.perf_counter() - t0) * 1000

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ctx = generator._apply_steering_compiled if compile_ok else generator._apply_steering_monkeypatch
        with ctx(bench_cwe):
            with torch.no_grad():
                _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                                   do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        t_gen = (time.perf_counter() - t1) * 1000

        times_probe.append(t_probe)
        times_gen.append(t_gen)
        times_total.append(t_probe + t_gen)

    results["compiled"] = {
        "compile_ok": compile_ok,
        "probe_mean_ms": float(np.mean(times_probe)),
        "gen_mean_ms": float(np.mean(times_gen)),
        "mean_ms": float(np.mean(times_total)),
        "std_ms": float(np.std(times_total)),
        "p95_ms": float(np.percentile(times_total, 95)),
        "all_ms": [float(t) for t in times_total],
    }
    print(f"  compile_ok={compile_ok}, "
          f"probe={results['compiled']['probe_mean_ms']:.1f}ms, "
          f"gen={results['compiled']['gen_mean_ms']:.1f}ms, "
          f"total={results['compiled']['mean_ms']:.1f}ms "
          f"(std={results['compiled']['std_ms']:.1f}ms)")

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 5: Probe-then-steer Option D (weight bias — zero Python wrapper)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[5/7] Probe-then-steer Option D: weight_bias ({BENCHMARK_ITERATIONS} iters)...")
    times_probe = []
    times_gen = []
    times_total = []
    for _ in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cwe_type, _ = router.classify(bench_prompt)
        torch.cuda.synchronize()
        t_probe = (time.perf_counter() - t0) * 1000

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with generator._apply_steering_weight_bias(bench_cwe):
            with torch.no_grad():
                _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                                   do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        t_gen = (time.perf_counter() - t1) * 1000

        times_probe.append(t_probe)
        times_gen.append(t_gen)
        times_total.append(t_probe + t_gen)

    results["weight_bias"] = {
        "probe_mean_ms": float(np.mean(times_probe)),
        "gen_mean_ms": float(np.mean(times_gen)),
        "mean_ms": float(np.mean(times_total)),
        "std_ms": float(np.std(times_total)),
        "p95_ms": float(np.percentile(times_total, 95)),
        "all_ms": [float(t) for t in times_total],
    }
    print(f"  probe={results['weight_bias']['probe_mean_ms']:.1f}ms, "
          f"gen={results['weight_bias']['gen_mean_ms']:.1f}ms, "
          f"total={results['weight_bias']['mean_ms']:.1f}ms "
          f"(std={results['weight_bias']['std_ms']:.1f}ms)")

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 6: Persistent weight_bias (set once, no per-iteration teardown)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[6/8] Persistent weight_bias ({BENCHMARK_ITERATIONS} iters)...")
    print("  (bias set once before loop, NOT torn down between iterations)")

    # Set up bias once
    layer_obj = model.model.layers[LAYER]
    down_proj = layer_obj.mlp.down_proj
    had_bias = down_proj.bias is not None
    original_bias = down_proj.bias.data.clone() if had_bias else None
    steering_bias = ALPHAS["buffer"] * vec_tensor
    if had_bias:
        down_proj.bias.data.add_(steering_bias)
    else:
        down_proj.bias = torch.nn.Parameter(steering_bias, requires_grad=False)

    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                               do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    # Clean up
    if had_bias:
        down_proj.bias.data = original_bias
    else:
        down_proj.bias = None

    results["persistent_weight_bias"] = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "all_ms": [float(t) for t in times],
    }
    print(f"  mean={results['persistent_weight_bias']['mean_ms']:.1f}ms, "
          f"std={results['persistent_weight_bias']['std_ms']:.1f}ms, "
          f"p95={results['persistent_weight_bias']['p95_ms']:.1f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 7: Persistent monkeypatch (patched once, no teardown)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[7/8] Persistent monkeypatch ({BENCHMARK_ITERATIONS} iters)...")
    print("  (forward patched once before loop, NOT restored between iterations)")

    layer_obj = model.model.layers[LAYER]
    original_forward = layer_obj.forward
    alpha_val = ALPHAS["buffer"]

    @functools.wraps(original_forward)
    def persistent_steered_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        h = output[0]
        if h.dim() == 3:
            h[:, -1, :] += alpha_val * vec_tensor
        else:
            h.add_(alpha_val * vec_tensor)
        return output

    layer_obj.forward = persistent_steered_forward

    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                               do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    layer_obj.forward = original_forward

    results["persistent_monkeypatch"] = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "all_ms": [float(t) for t in times],
    }
    print(f"  mean={results['persistent_monkeypatch']['mean_ms']:.1f}ms, "
          f"std={results['persistent_monkeypatch']['std_ms']:.1f}ms, "
          f"p95={results['persistent_monkeypatch']['p95_ms']:.1f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK 8: Full pipeline with hook (Exp 8.5 protocol — probe + hook gen)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[8/8] Full hook-based pipeline ({BENCHMARK_ITERATIONS} iters)...")
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Probe pass
        cwe_type, _ = router.classify(bench_prompt)

        # Hook-based generation
        sel_vec = torch.tensor(vectors_np[cwe_type], dtype=torch.float16).to(device)
        hook = model.model.layers[LAYER].register_forward_hook(
            make_steering_hook(ALPHAS[cwe_type], sel_vec)
        )
        with torch.no_grad():
            _ = model.generate(**bench_inputs, max_new_tokens=MAX_NEW_TOKENS_BENCH,
                               do_sample=False, min_new_tokens=MAX_NEW_TOKENS_BENCH,
                               pad_token_id=tokenizer.pad_token_id)
        hook.remove()

        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    results["full_hook_pipeline"] = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "all_ms": [float(t) for t in times],
    }
    print(f"  mean={results['full_hook_pipeline']['mean_ms']:.1f}ms, "
          f"std={results['full_hook_pipeline']['std_ms']:.1f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ═══════════════════════════════════════════════════════════════════════

    baseline_ms = results["baseline"]["mean_ms"]

    print(f"\n{'='*70}")
    print("TIMING RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'Mean (ms)':>10} {'Std':>8} {'P95':>10} {'Overhead':>10}")
    print("-" * 70)

    for label, key in [
        ("Baseline (no steering)", "baseline"),
        ("Hook-based (Exp 8.5)", "hook_based"),
        ("Full hook pipeline", "full_hook_pipeline"),
        ("Probe-then-steer A (monkeypatch)", "monkeypatch"),
        ("Probe-then-steer B (compiled)", "compiled"),
        ("Probe-then-steer D (weight_bias)", "weight_bias"),
        ("Persistent weight_bias", "persistent_weight_bias"),
        ("Persistent monkeypatch", "persistent_monkeypatch"),
    ]:
        r = results[key]
        overhead = (r["mean_ms"] / baseline_ms - 1) * 100
        print(f"{label:<35} {r['mean_ms']:>9.1f} {r['std_ms']:>7.1f} "
              f"{r['p95_ms']:>9.1f} {overhead:>+9.1f}%")

    print(f"\nRouting accuracy: {correct}/{total} ({route_accuracy*100:.1f}%)")

    # Check success criteria
    candidates = ["monkeypatch", "compiled", "weight_bias",
                   "persistent_weight_bias", "persistent_monkeypatch"]
    best_method = min(candidates, key=lambda k: results[k]["mean_ms"])
    best_overhead = (results[best_method]["mean_ms"] / baseline_ms - 1) * 100

    print(f"\nBest method: {best_method} ({best_overhead:+.1f}% overhead)")
    print(f"  Target: <10% overhead → {'PASS' if best_overhead < 10 else 'FAIL'}")
    print(f"  Routing accuracy target: >=95.2% → {'PASS' if route_accuracy >= 0.952 else 'FAIL'}")

    # ─── Save results ────────────────────────────────────────────────────
    output = {
        "timestamp": timestamp,
        "experiment": "Experiment 9b: Probe-Then-Steer Benchmark",
        "model": MODEL_NAME,
        "layer": LAYER,
        "benchmark_iterations": BENCHMARK_ITERATIONS,
        "max_new_tokens": MAX_NEW_TOKENS_BENCH,
        "routing_accuracy": float(route_accuracy),
        "routing_correct": correct,
        "routing_total": total,
        "timing": {k: {kk: vv for kk, vv in v.items() if kk != "all_ms"}
                   for k, v in results.items()},
        "baseline_ms": float(baseline_ms),
        "best_method": best_method,
        "best_overhead_pct": float(best_overhead),
    }

    results_path = RESULTS_DIR / f"benchmark_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save full timing data
    full_path = RESULTS_DIR / f"benchmark_full_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full timing data: {full_path}")

    loader.unload()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
