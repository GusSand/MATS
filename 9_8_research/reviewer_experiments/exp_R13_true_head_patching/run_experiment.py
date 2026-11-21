#!/usr/bin/env python3
"""
Experiment R13: true per-head patching at Layer 10 (Llama-3.1-8B-Instruct)
==========================================================================
Experiment file: docs/experiments/11-21_llama8b_decimal_true_head_patching.md

Fixes two bugs in the head-subset experiments behind "Even Heads" (v1) and the
v2 draft, all of which forward-hooked `layers[10].self_attn`:

  Bug A (hook point). `self_attn` output is post-o_proj (post-W_O). Slicing it
    as [32, 128] swaps residual-coordinate blocks, not heads. True per-head
    patching acts on the INPUT of `o_proj` (forward_pre_hook).
  Bug B (decode steps). The hook stayed on through generate(); on each decode
    step (seq_len 1) it wrote the clean prompt's position 0 (BOS) into the
    patched blocks.

Cells (hook x scope), all in one process on one GPU, sequential:
  post_wo/all_steps    old protocol; must reproduce R8 and R1 subset-for-subset
  pre_wo/prefill       true heads, prompt pass only, left-aligned (paper protocol)
  pre_wo/prefill_right true heads, prompt pass only, right-aligned (answer positions patched)
  post_wo/prefill      attribution: bug B removed, bug A kept
  pre_wo/all_steps     attribution: bug A removed, bug B kept
  post_wo/decode_only  bug B alone: all_steps with the buggy run's OWN activations

Greedy decoding is deterministic: one run per configuration.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("R13_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
LAYER_IDX = 10
CORRECT_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
BUGGY_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
MAX_NEW = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
R8_JSON = os.path.join(HERE, "..", "exp_R8_threshold_analysis", "results_20260314_091723.json")
R1_JSON = os.path.join(HERE, "..", "exp_R1_arbitrary_heads", "results_20260314_023132.json")

CELLS = [("post_wo", "all_steps"), ("pre_wo", "prefill"), ("pre_wo", "prefill_right"),
         ("post_wo", "prefill"), ("pre_wo", "all_steps"), ("post_wo", "decode_only")]

CORRECT_PATTERNS = ["9.8 is bigger", "9.8 is larger", "9.8 is greater",
                    "9.8 is the bigger", "9.8 is the larger"]
BUG_PATTERNS = ["9.11 is bigger", "9.11 is larger", "9.11 is greater",
                "9.11 is the bigger", "9.11 is the larger"]


def is_correct(text):
    """Outcome 'correct'. Identical to the R8/R1 classifier."""
    lo = text.lower()
    correct = any(p in lo for p in CORRECT_PATTERNS)
    bug = any(p in lo for p in BUG_PATTERNS)
    if not correct and not bug:
        lines = text.strip().split("\n")
        if lines and "9.8" in lines[0] and "9.11" not in lines[0]:
            correct = True
    return correct and not bug


def is_bug(text):
    """Outcome 'bug': a bug pattern matched and the text is not correct."""
    return any(p in text.lower() for p in BUG_PATTERNS) and not is_correct(text)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def provenance():
    import transformers
    info = {"hostname": socket.gethostname(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "git_sha": os.environ["R13_GIT_SHA"], "script_sha256": sha256(__file__),
            "model": MODEL_NAME, "layer": LAYER_IDX}
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        info["gpu"] = "no nvidia-smi"
    return info


class Patcher:
    def __init__(self):
        print(f"Loading {MODEL_NAME} on {DEVICE} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None)
        self.model.eval()
        cfg = self.model.config
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        attn = self.model.model.layers[LAYER_IDX].self_attn
        self.attn, self.o_proj = attn, attn.o_proj
        assert self.o_proj.in_features == self.n_heads * self.head_dim
        assert self.o_proj.bias is None
        self.last_shapes = []

    def capture(self, prompt, hook):
        store = {}
        if hook == "post_wo":
            def fn(mod, inp, out):
                store["x"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            handle = self.attn.register_forward_hook(fn)
        else:
            def fn(mod, args):
                store["x"] = args[0].detach().clone()
            handle = self.o_proj.register_forward_pre_hook(fn)
        try:
            ids = self.tok(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**ids)
        finally:
            handle.remove()
        return store["x"]

    def _swap(self, h, saved, heads, align):
        bs, sl, d = h.shape
        hr = h.view(bs, sl, self.n_heads, self.head_dim).clone()
        sr = saved.view(saved.shape[0], -1, self.n_heads, self.head_dim).to(hr.dtype)
        ts = sr.shape[1]
        ml = min(sl, ts)
        idx = torch.tensor(heads, device=h.device, dtype=torch.long)
        if align == "right":
            # suffix-aligned, BOS (position 0) left untouched on both sides:
            # clean 1..ts-1 -> buggy sl-(ts-1)..sl-1  (17/19 tokens: clean 1..16 -> buggy 3..18)
            n = min(sl, ts) - 1
            if n > 0:
                hr[:, sl - n:, idx, :] = sr[:, ts - n:, idx, :]
        else:
            hr[:, :ml, idx, :] = sr[:, :ml, idx, :]
        return hr.view(bs, sl, d)

    def generate(self, prompt, saved=None, heads=None, hook=None, scope=None):
        handle = None
        self.last_shapes = []
        if saved is not None and heads:
            prefill_only = scope in ("prefill", "prefill_right")
            align = "right" if scope == "prefill_right" else "left"

            def active(h):
                self.last_shapes.append(int(h.shape[1]))
                return not prefill_only or len(self.last_shapes) == 1

            if hook == "post_wo":
                def fn(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if not active(h):
                        return None
                    new = self._swap(h, saved, heads, align)
                    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
                handle = self.attn.register_forward_hook(fn)
            else:
                def fn(mod, args):
                    if not active(args[0]):
                        return None
                    return (self._swap(args[0], saved, heads, align),) + tuple(args[1:])
                handle = self.o_proj.register_forward_pre_hook(fn)
        try:
            ids = self.tok(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def load_refs():
    for p in (R8_JSON, R1_JSON):
        if not os.path.exists(p):
            sys.exit(f"FATAL: reference results missing: {p}")
    return json.load(open(R8_JSON)), json.load(open(R1_JSON))


def build_configs(n_heads, r1):
    """R8 subsets regenerated with R8's RNG call order; R1 subsets read from R1's JSON."""
    even = list(range(0, n_heads, 2))
    odd = list(range(1, n_heads, 2))
    rng = np.random.RandomState(42)
    cfgs = [("all_heads", list(range(n_heads))), ("even_16", even), ("odd_16", odd)]
    cfgs += [(f"single_{h}", [h]) for h in range(n_heads)]
    cfgs += [(f"even7_{i:03d}", sorted(rng.choice(even, 7, replace=False).tolist()))
             for i in range(500)]
    cfgs += [(f"even8_{i:03d}", sorted(rng.choice(even, 8, replace=False).tolist()))
             for i in range(500)]
    base7 = sorted(rng.choice(even, 7, replace=False).tolist())
    for n_odd in [0, 1, 2, 3, 5, 9, 16]:
        extra = [] if n_odd == 0 else sorted(rng.choice(odd, min(n_odd, 16), replace=False).tolist())
        cfgs.append((f"trans7plus{n_odd}odd", sorted(base7 + extra)))
    cfgs += [(f"rand8_{i:03d}", x["heads"]) for i, x in enumerate(r1["section1_random_mixed_8"])]
    for p in r1["section4_matched_controls"]:
        cfgs.append((f"pair{p['pair_idx']:02d}_even", p["even"]["heads"]))
        cfgs.append((f"pair{p['pair_idx']:02d}_odd", p["odd"]["heads"]))
    cfgs += [(f"oddonly_k{k}", v["heads"]) for k, v in r1["section3_odd_only"].items()]
    names = [n for n, _ in cfgs]
    assert len(names) == len(set(names))
    return cfgs


def reference_outcomes(r8, r1):
    ref = {f"even7_{i:03d}": r for i, r in enumerate(r8["seven_head_results"])}
    ref.update({f"even8_{i:03d}": r for i, r in enumerate(r8["eight_head_results"])})
    ref.update({f"rand8_{i:03d}": r for i, r in enumerate(r1["section1_random_mixed_8"])})
    for p in r1["section4_matched_controls"]:
        ref[f"pair{p['pair_idx']:02d}_even"] = p["even"]
        ref[f"pair{p['pair_idx']:02d}_odd"] = p["odd"]
    ref.update({f"oddonly_k{k}": v for k, v in r1["section3_odd_only"].items()})
    return {k: {"heads": v["heads"], "correct": v["success_rate"] >= 0.5} for k, v in ref.items()}


def reproduction(configs, ref):
    out = {}
    for group in ["even7_", "even8_", "rand8_", "pair", "oddonly_"]:
        keys = [k for k in ref if k.startswith(group)]
        matched = [k for k in keys if k in configs and configs[k]["heads"] == ref[k]["heads"]]
        agree = sum(configs[k]["correct"] == ref[k]["correct"] for k in matched)
        out[group] = {"n_ref": len(keys), "heads_matched": len(matched), "outcome_agree": agree}
    return out


def summarize(done):
    def rate(prefix):
        v = [c["correct"] for k, c in done.items() if k.startswith(prefix)]
        return [sum(v), len(v)]
    s = {k: rate(k) for k in ["all_heads", "even_16", "odd_16", "single_", "even7_", "even8_",
                              "rand8_", "oddonly_"]}
    pairs = sorted({k[:6] for k in done if k.startswith("pair")})
    s["pairs_even_correct_odd_not"] = [sum(done[p + "_even"]["correct"] and not done[p + "_odd"]["correct"]
                                           for p in pairs), len(pairs)]
    s["pairs_even_correct"] = [sum(done[p + "_even"]["correct"] for p in pairs), len(pairs)]
    s["pairs_odd_correct"] = [sum(done[p + "_odd"]["correct"] for p in pairs), len(pairs)]
    s["transition"] = {k: done[k]["correct"] for k in done if k.startswith("trans")}
    return s


REPRO_THRESHOLDS = {"R8_even7_plus_even8": 950, "R1_rand8": 190, "R1_pairs": 57}


def reproduction_gate(rep):
    vals = {"R8_even7_plus_even8": rep["even7_"]["outcome_agree"] + rep["even8_"]["outcome_agree"],
            "R1_rand8": rep["rand8_"]["outcome_agree"], "R1_pairs": rep["pair"]["outcome_agree"]}
    full = (rep["even7_"]["heads_matched"] + rep["even8_"]["heads_matched"] == 1000
            and rep["rand8_"]["heads_matched"] == 200 and rep["pair"]["heads_matched"] == 60)
    ok = full and all(vals[k] >= t for k, t in REPRO_THRESHOLDS.items())
    return {"values": vals, "thresholds": REPRO_THRESHOLDS, "all_heads_matched": full,
            "reproduction_gate_ok": ok, "oddonly_reported_not_gated": rep["oddonly_"]}


def head_isolation_check(p):
    """Patch one head into the o_proj input; only that head's slice may change."""
    x = p.capture(BUGGY_PROMPT, "pre_wo")
    y = p.capture(CORRECT_PROMPT, "pre_wo")
    out = p._swap(x, y, [5], "left").view(x.shape[0], x.shape[1], p.n_heads, p.head_dim)
    diff = (out - x.view_as(out)).abs().sum(dim=(0, 1, 3))
    changed = [int(i) for i in torch.nonzero(diff).flatten()]
    return {"changed_heads": changed, "ok": changed == [5]}


def run_cell(p, hook, scope, cfgs, out_dir, limit=None):
    tag = f"{hook}__{scope}"
    path = os.path.join(out_dir, f"results_{tag}.json")
    if os.path.exists(path):
        sys.exit(f"FATAL: {path} exists; run directories are single-use")
    gen_scope = "all_steps" if scope == "decode_only" else scope
    buggy_saved = p.capture(BUGGY_PROMPT, hook)
    saved = buggy_saved if scope == "decode_only" else p.capture(CORRECT_PROMPT, hook)
    unpatched = p.generate(BUGGY_PROMPT)
    allh = list(range(p.n_heads))

    ident = p.generate(BUGGY_PROMPT, buggy_saved, allh, hook, gen_scope)
    det = [p.generate(BUGGY_PROMPT, saved, allh, hook, gen_scope) for _ in range(3)]
    shapes = list(p.last_shapes)
    t_buggy = p.tok(BUGGY_PROMPT, return_tensors="pt")["input_ids"].shape[1]
    gates = {"identity_ok": ident == unpatched, "identity_output": ident,
             "identity_correct": is_correct(ident),
             "determinism_ok": len(set(det)) == 1, "determinism_outputs": det,
             "hook_call_shapes": shapes,
             "hook_shapes_ok": shapes[0] == t_buggy and all(s == 1 for s in shapes[1:])}
    print(f"[{tag}] gates: identity={gates['identity_ok']} determinism={gates['determinism_ok']} "
          f"shapes={gates['hook_shapes_ok']}", flush=True)
    if gen_scope != "all_steps" and not (gates["identity_ok"] and gates["determinism_ok"]
                                         and gates["hook_shapes_ok"]):
        json.dump({"tag": tag, "gates": gates, "aborted": True}, open(path, "w"), indent=1)
        sys.exit(f"FATAL: [{tag}] prefill-cell gate failed; stopping (pre-registered)")

    done, t0 = {}, time.time()
    todo = cfgs[:limit] if limit else cfgs
    for i, (name, heads) in enumerate(todo):
        text = p.generate(BUGGY_PROMPT, saved, heads, hook, gen_scope)
        done[name] = {"heads": heads, "correct": is_correct(text), "bug": is_bug(text), "output": text}
        if (i + 1) % 200 == 0:
            print(f"[{tag}] {i+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    res = {"tag": tag, "hook": hook, "scope": scope,
           "unit": "attention head" if hook == "pre_wo" else "128-dim block of post-W_O output",
           "gates": gates, "summary": summarize(done) if not limit else None,
           "configs": done, "seconds": time.time() - t0}
    json.dump(res, open(path, "w"), indent=1)
    print(f"[{tag}] summary: {json.dumps({k: v for k, v in (res['summary'] or {}).items() if k != 'transition'})}",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="new run directory (must not exist)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N configs")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU mechanics test on random weights: phenomenon gate recorded, not enforced")
    ap.add_argument("--cells", default=None, help="comma list hook/scope; default all six")
    a = ap.parse_args()
    if not os.environ.get("R13_GIT_SHA"):
        sys.exit("FATAL: set R13_GIT_SHA to the committed SHA of this script")
    os.makedirs(a.out, exist_ok=False)

    r8, r1 = load_refs()
    p = Patcher()
    base = {"clean": p.generate(CORRECT_PROMPT), "buggy": p.generate(BUGGY_PROMPT)}
    gate = is_correct(base["clean"]) and is_bug(base["buggy"])
    run = {"provenance": provenance(), "started": datetime.now().isoformat(),
           "command": " ".join(sys.argv), "baseline": base, "phenomenon_gate_ok": gate}

    def save(r):
        json.dump(r, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    save(run)
    print(f"phenomenon gate: {gate}  {base}", flush=True)
    if not gate and not (a.limit or a.smoke):
        sys.exit("FATAL: phenomenon gate failed; nothing downstream is interpretable")

    iso = head_isolation_check(p)
    bos = {h: float((p.capture(CORRECT_PROMPT, h)[:, 0] - p.capture(BUGGY_PROMPT, h)[:, 0])
                    .abs().max()) for h in ("pre_wo", "post_wo")}
    run["head_isolation"], run["bos_max_abs_diff_clean_vs_buggy"] = iso, bos
    print(f"head isolation: {iso}  BOS diff: {bos}", flush=True)
    if not iso["ok"]:
        save(run)
        sys.exit("FATAL: head isolation check failed")

    cfgs = build_configs(p.n_heads, r1)
    ref = reference_outcomes(r8, r1)
    cells = CELLS if not a.cells else [tuple(c.split("/")) for c in a.cells.split(",")]
    results = {}
    for hook, scope in cells:
        results[(hook, scope)] = run_cell(p, hook, scope, cfgs, a.out, a.limit)
        if (hook, scope) == ("post_wo", "all_steps"):
            rep_ = reproduction(results[(hook, scope)]["configs"], ref)
            run["reproduction_vs_R8_R1"] = rep_
            run["reproduction_gate"] = reproduction_gate(rep_)
            print(f"reproduction gate: {run['reproduction_gate']}", flush=True)
        run["cells_done"] = [f"{h}/{s}" for h, s in results]
        save(run)
    a_pre = results.get(("pre_wo", "prefill"), {}).get("configs", {}).get("all_heads")
    a_post = results.get(("post_wo", "prefill"), {}).get("configs", {}).get("all_heads")
    if a_pre and a_post:
        first = lambda t: p.tok(t, add_special_tokens=False)["input_ids"][:1]
        run["all_heads_equivalence"] = {"same_text": a_pre["output"] == a_post["output"],
                                        "same_first_token": first(a_pre["output"]) == first(a_post["output"]),
                                        "same_class": a_pre["correct"] == a_post["correct"]}
    run["finished"] = datetime.now().isoformat()
    save(run)


if __name__ == "__main__":
    main()
