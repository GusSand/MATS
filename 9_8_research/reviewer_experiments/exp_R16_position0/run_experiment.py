#!/usr/bin/env python3
"""
Experiment R16: does position 0 explain R13's L40S vs A100 gap?
===============================================================
Experiment file: docs/experiments/11-22_llama8b_position0_gpu_gap.md

Imports R13's script unchanged (sha256 checked). The only change is the donor tensor's position 0:
  copy   R13 as run (gate: must reproduce R13's outputs on this GPU)
  keep0  donor[:, 0] := the buggy run's own position 0 (A100)
  inject donor[:, 0] := L40S buggy position 0 + A100 (clean - buggy) position-0 difference (L40S)
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
R13_DIR = os.path.join(HERE, "..", "exp_R13_true_head_patching")
R13_PATH = os.path.join(R13_DIR, "run_experiment.py")
R13_SHA256 = "1ce5b0cc2e2cdc74c21de37de5340cd00f5b14e27ae13cf6893d2e09f280348d"
REF = {"a100": os.path.join(R13_DIR, "results", "r13_24084222"),
       "l40s": os.path.join(R13_DIR, "results", "r13_24014365")}
CELLS = ["prefill", "all_steps"]   # hook is post_wo in both, as in R13


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_cell(gpu, scope):
    with open(os.path.join(REF[gpu], f"results_post_wo__{scope}.json")) as f:
        return json.load(f)["configs"]


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def decide(variant_cfgs, own_copy_cfgs, gpu_from, gpu_to_cfgs, D):
    """r = fraction of D whose class moves to the other GPU's class; N_new = changes outside D."""
    moved = sum(variant_cfgs[k]["correct"] == gpu_to_cfgs[k]["correct"] for k in D)
    r = moved / len(D) if D else None
    n_new = sum(variant_cfgs[k]["correct"] != own_copy_cfgs[k]["correct"] for k in variant_cfgs if k not in D)
    if r is None:
        verdict = "no gap to explain (D empty)"
    elif r >= 0.8 and n_new <= 4:
        verdict = "explains the gap"
    elif r <= 0.2:
        verdict = "does not explain the gap"
    else:
        verdict = f"explains part of the gap (r = {r:.2f})"
    r8 = lambda c: sum(c[k]["correct"] for k in c if k.startswith(("even7_", "even8_")))
    return {"D_size": len(D), "moved": moved, "r": r, "N_new": n_new, "verdict": verdict,
            "r8_correct_variant": r8(variant_cfgs), "r8_correct_own_copy": r8(own_copy_cfgs),
            "r8_correct_other_gpu": r8(gpu_to_cfgs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, choices=["a100", "l40s"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", default=None, help="l40s: position0.pt from the A100 job")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N configs, gates not enforced")
    a = ap.parse_args()
    if not os.environ.get("R16_GIT_SHA"):
        sys.exit("FATAL: set R16_GIT_SHA")
    if sha256(R13_PATH) != R13_SHA256 and not a.limit:
        sys.exit("FATAL: R13 script changed")
    if a.gpu == "l40s" and not (a.delta and os.path.exists(a.delta)):
        sys.exit("FATAL: --delta (A100 position0.pt) required for the L40S job")
    os.makedirs(a.out, exist_ok=False)
    os.environ["R13_GIT_SHA"] = os.environ["R16_GIT_SHA"]
    spec = importlib.util.spec_from_file_location("r13", R13_PATH)
    r13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r13)
    import torch

    p = r13.Patcher()
    _, r1 = r13.load_refs()
    cfgs = r13.build_configs(p.n_heads, r1)
    if a.limit:
        cfgs = cfgs[:a.limit]
    prov = r13.provenance()
    prov.update({"r16_script_sha256": sha256(__file__), "r13_script_sha256": sha256(R13_PATH),
                 "r16_git_sha": os.environ["R16_GIT_SHA"], "gpu_role": a.gpu})
    run = {"provenance": prov, "started": datetime.now().isoformat(), "command": " ".join(sys.argv)}
    path = os.path.join(a.out, "run.json")

    clean = p.capture(r13.CORRECT_PROMPT, "post_wo")
    buggy = p.capture(r13.BUGGY_PROMPT, "post_wo")
    d0 = (clean[:, 0].float() - buggy[:, 0].float())
    run["pos0_max_abs_diff"] = float(d0.abs().max())
    run["pos0_n_nonzero"] = int((d0 != 0).sum())
    print(f"[{a.gpu}] position-0 clean-buggy max|diff| = {run['pos0_max_abs_diff']}  "
          f"nonzero entries = {run['pos0_n_nonzero']}", flush=True)
    variants = {"copy": clean}
    if a.gpu == "a100":
        torch.save({"clean0": clean[:, 0].cpu(), "buggy0": buggy[:, 0].cpu(), "delta_fp32": d0.cpu()},
                   os.path.join(a.out, "position0.pt"))
        k0 = clean.clone()
        k0[:, 0] = buggy[:, 0]
        variants["keep0"] = k0
    else:
        ref = torch.load(a.delta)
        delta = ref["delta_fp32"].to(clean.device)
        run["injected_delta_max_abs"] = float(delta.abs().max())
        inj = clean.clone()
        inj[:, 0] = (buggy[:, 0].float() + delta).to(clean.dtype)
        run["inject_pos0_max_abs_change_vs_buggy0"] = float((inj[:, 0].float() - buggy[:, 0].float()).abs().max())
        variants["inject"] = inj
    dump(run, path)

    other = "l40s" if a.gpu == "a100" else "a100"
    run["cells"] = {}
    for scope in CELLS:
        ref_own, ref_other = load_cell(a.gpu, scope), load_cell(other, scope)
        out = {}
        for vname, donor in variants.items():
            t0, done = time.time(), {}
            for name, heads in cfgs:
                text = p.generate(r13.BUGGY_PROMPT, donor, heads, "post_wo", scope)
                done[name] = {"heads": heads, "correct": r13.is_correct(text), "bug": r13.is_bug(text),
                              "output": text}
            out[vname] = done
            dump({"scope": scope, "variant": vname, "configs": done},
                 os.path.join(a.out, f"results_post_wo__{scope}__{vname}.json"))
            print(f"[{a.gpu}/{scope}/{vname}] {len(done)} configs, {time.time()-t0:.0f}s", flush=True)
        copy = out["copy"]
        gate = {"n": len(copy), "text_equal_to_r13": sum(copy[k]["output"] == ref_own[k]["output"] for k in copy),
                "class_equal_to_r13": sum(copy[k]["correct"] == ref_own[k]["correct"] for k in copy)}
        gate["ok"] = gate["text_equal_to_r13"] == gate["n"]
        D = [k for k in ref_own if ref_own[k]["correct"] != ref_other[k]["correct"] and k in copy]
        cell = {"gate_copy_reproduces_r13": gate}
        for vname in variants:
            if vname != "copy":
                cell[vname] = decide(out[vname], copy, a.gpu, ref_other, D)
        cell["interpreted"] = gate["ok"]
        run["cells"][scope] = cell
        print(f"[{a.gpu}/{scope}] gate {gate}  " + "  ".join(
            f"{v}: {cell[v]}" for v in variants if v != "copy"), flush=True)
        dump(run, path)
    run["finished"] = datetime.now().isoformat()
    dump(run, path)


if __name__ == "__main__":
    main()
