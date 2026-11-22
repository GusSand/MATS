#!/usr/bin/env python3
"""
Experiment R16: does the layer-10 position-0 write explain R13's L40S vs A100 gap?
=================================================================================
Experiment file: docs/experiments/11-22_llama8b_position0_gpu_gap.md
Design reviews: docs/reviews/11-22_llama8b_position0_gpu_gap_design*.md (rounds 1-4)

Imports R13's script unchanged (sha256 checked); R13's hook is untouched. Variants change only the donor:
  A100  copy (gate) | keep0: donor[:,0] := buggy0 | placebo: clean0_A with keep0's steps (-d0) applied at
        other coordinates of the same blocks, each landing exactly
  L40S  copy (gate) | inject: donor[:,0] := buggy0_L + d0_A | placebo: clean0_L with inject's steps (+d0_A)
        at other coordinates of the same blocks |
        donorA100: the A100's whole clean donor | a100clean0: donor[:,0] := clean0_A (only if buggy0_L != buggy0_A)
Exit code 2 when a gate fails or the premise is absent, so `afterok` blocks the L40S job.
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
R13_NODE = {"a100": "ga004", "l40s": "gl044"}
CELLS = ["prefill", "all_steps"]   # hook is post_wo in both, as in R13
N_BLOCKS, BLOCK = 32, 128


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def ulp_step(x, up):
    """Move each entry of x one ULP up (or down) via its integer bit pattern (fp16 on GPU, fp32 in smoke)."""
    import torch
    it = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32}[x.dtype]
    neg_min = {torch.int16: -32767, torch.int32: -2147483647}[it]   # sign bit + 1: smallest negative subnormal
    b = x.contiguous().view(it).clone()
    pos, neg, zero = x > 0, x < 0, x == 0
    if up:
        b[pos] += 1
        b[neg] -= 1
        b[zero] = 1
    else:
        b[pos] -= 1
        b[neg] += 1
        b[zero] = neg_min
    return b.view(x.dtype)


UNPLACED_TOL = 0.01   # placebo valid if unplaced steps carry <= 1% of total |step| (review round 4)


def make_placebo(base0, step):
    """base0 (this GPU's copy donor position 0) with the test variant's per-coordinate steps (-d0 for keep0 on
    A100, +dA for inject on L40S) applied at OTHER coordinates of the same blocks. A target qualifies only if
    the step lands exactly and nonzero there: fp16(x[c2] + s) - x[c2] == s. Sources without a qualifying free
    target are recorded as unplaced; the placebo is valid if the unplaced steps carry at most UNPLACED_TOL
    of the total |step| (review rounds 3-4)."""
    import torch
    flat_b, flat_s = base0.reshape(-1).clone(), step.reshape(-1)
    nz = [int(i) for i in torch.nonzero(flat_s).flatten()]
    taken, moves, unplaced = set(nz), [], []
    for c in nz:
        blk, off = divmod(c, BLOCK)
        s_ = flat_s[c]
        chosen = None
        for j in range(BLOCK):
            c2 = blk * BLOCK + (off + BLOCK // 2 + j) % BLOCK
            if c2 in taken:
                continue
            new = (flat_b[c2].float() + s_).to(flat_b.dtype)
            got = new.float() - flat_b[c2].float()
            if got != 0 and got == s_:
                chosen = (c2, new)
                break
        if chosen is None:
            unplaced.append(c)
            continue
        c2, new = chosen
        taken.add(c2)
        moves.append({"src": c, "dst": c2, "step": float(s_)})
        flat_b[c2] = new
    tot = float(flat_s.abs().sum())
    frac = float(sum(abs(float(flat_s[c])) for c in unplaced)) / tot if tot else 1.0
    return flat_b.view_as(base0), {"moves": moves, "unplaced": unplaced, "unplaced_step_fraction": frac,
                                   "valid": bool(moves) and frac <= UNPLACED_TOL}


def blocks_nonzero(d0):
    return sorted({int(i) // BLOCK for i in (d0.reshape(-1) != 0).nonzero().flatten()})


MIN_EXPOSED = 10


def decide(var, copy, other_ref, D, exposed, placebo_moved=None, label=True, placebo_valid=True):
    """Pre-registered rule on exposed D (configs whose blocks include a block where the difference is nonzero).
    placebo_moved: the placebo's `moved` count on the same exposed D. label=False: descriptive (r on all D)."""
    De = [k for k in D if k in exposed] if label else list(D)
    moved = sum(var[k]["correct"] == other_ref[k]["correct"] for k in De)
    n = len(De)
    r = moved / n if n else None
    outside = [k for k in var if k not in D and var[k]["correct"] != copy[k]["correct"]]
    up = sum(var[k]["correct"] and not copy[k]["correct"] for k in outside)
    unexposed_same = sum(var[k]["output"] == copy[k]["output"] for k in var if k not in exposed)
    n_unexposed = sum(1 for k in var if k not in exposed)
    n_new = len(outside)
    out = {"D": len(D), "D_exposed": len([k for k in D if k in exposed]), "n_scored": n, "moved": moved, "r": r,
           "moved_over_all_D": [moved, len(D)], "N_new": n_new, "outside_to_correct": up,
           "outside_to_not_correct": n_new - up, "unexposed_text_identical": [unexposed_same, n_unexposed],
           "text_equal_other_gpu_r13": sum(var[k]["output"] == other_ref[k]["output"] for k in var),
           "placebo_moved": placebo_moved}
    r8 = lambda c: sum(c[k]["correct"] for k in c if k.startswith(("even7_", "even8_")))
    out["r8_correct"] = {"variant": r8(var), "own_copy": r8(copy), "other_gpu": r8(other_ref)}
    if not label:
        out["verdict"] = "descriptive"
        return out
    pm = placebo_moved if placebo_moved is not None else 0
    if n == 0:
        v = "refuted by construction: no D config is exposed to the position-0 difference"
    elif n < MIN_EXPOSED:
        v = f"descriptive only: {n} exposed D configs (< {MIN_EXPOSED})"
    elif 5 * moved >= 4 * n:                                   # r >= 0.8, integer arithmetic
        if n_new > 4:
            v = "perturbs broadly"
        elif not placebo_valid:
            v = "not distinguishable from fragility (no valid placebo)"
        elif 2 * (moved - pm) < n:                             # r - r_placebo < 0.5
            v = "not distinguishable from fragility"
        elif out["D"] - n == 0 or 5 * moved >= 4 * len(D):
            v = "explains the gap"
        else:
            v = f"explains the exposed part (moved/|D| = {moved}/{len(D)})"
    elif 5 * moved <= n:                                       # r <= 0.2
        v = "perturbs, does not explain" if n_new > 4 else "does not explain the gap"
    elif n_new > 4:
        v = "perturbs broadly"
    elif not placebo_valid:
        v = "not distinguishable from fragility (no valid placebo)"
    elif 2 * pm >= moved:                                      # placebo moves at least half as many
        v = "not distinguishable from fragility"
    elif 2 * (moved - pm) < n:
        v = f"partial, below the placebo margin (r = {moved}/{n}, r_pl = {pm}/{n})"
    else:
        v = f"explains part of the gap (r = {moved}/{n})"
    out["verdict"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, choices=["a100", "l40s"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", default=None, help="l40s: position0.pt from the A100 job")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N configs; gates recorded, not enforced")
    ap.add_argument("--synthetic-delta", action="store_true", help="smoke only: fake a one-ULP d0 on CPU")
    a = ap.parse_args()
    if not os.environ.get("R16_GIT_SHA"):
        sys.exit("FATAL: set R16_GIT_SHA")
    if a.synthetic_delta and not a.limit:
        sys.exit("--synthetic-delta is smoke-only")
    if sha256(R13_PATH) != R13_SHA256 and not a.limit:
        sys.exit("FATAL: R13 script changed")
    a100_run = None
    if a.gpu == "l40s":
        if not (a.delta and os.path.exists(a.delta)):
            sys.exit("FATAL: --delta (A100 position0.pt) required for the L40S job")
        a100_run = load_json(os.path.join(os.path.dirname(a.delta), "run.json"))
        if not a.limit and not (a100_run.get("all_gates_ok") and a100_run.get("pos0_n_nonzero", 0) > 0):
            sys.exit("FATAL: A100 job's gates failed or its position-0 difference is zero")
    os.makedirs(a.out, exist_ok=False)
    os.environ["R13_GIT_SHA"] = os.environ["R16_GIT_SHA"]
    spec = importlib.util.spec_from_file_location("r13", R13_PATH)
    r13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r13)
    import torch

    p = r13.Patcher()
    global N_BLOCKS, BLOCK
    N_BLOCKS, BLOCK = p.n_heads, p.model.config.hidden_size // p.n_heads   # post_wo block = hidden / n_heads
    assert N_BLOCKS * BLOCK == p.model.config.hidden_size
    if not a.limit:
        assert (N_BLOCKS, BLOCK) == (32, 128) and p.n_heads * p.head_dim == 4096
    r8, r1 = r13.load_refs()
    cfgs = r13.build_configs(p.n_heads, r1)
    if a.limit:
        cfgs = cfgs[:a.limit]
    prov = r13.provenance()
    prov.update({"r16_script_sha256": sha256(__file__), "r13_script_sha256": sha256(R13_PATH),
                 "r16_git_sha": os.environ["R16_GIT_SHA"], "gpu_role": a.gpu,
                 "r13_ref_sha256": {g: {s: sha256(os.path.join(REF[g], f"results_post_wo__{s}.json"))
                                        for s in CELLS} for g in REF},
                 "delta_file_sha256": sha256(a.delta) if a.delta else None})
    r13_prov = load_json(os.path.join(REF[a.gpu], "run.json"))["provenance"]
    run = {"provenance": prov, "started": datetime.now().isoformat(), "command": " ".join(sys.argv),
           "same_node_as_r13": prov["hostname"].startswith(R13_NODE[a.gpu]),
           "same_gpu_string_as_r13": prov.get("gpu", "").split(",")[0] == r13_prov.get("gpu", "").split(",")[0]}
    path = os.path.join(a.out, "run.json")
    gates_ok = []

    clean = p.capture(r13.CORRECT_PROMPT, "post_wo")
    buggy = p.capture(r13.BUGGY_PROMPT, "post_wo")
    # capture vs live generate prefill, bitwise (review fix 2)
    live = {}
    for name, prompt in (("buggy", r13.BUGGY_PROMPT), ("clean", r13.CORRECT_PROMPT)):
        store = []
        h = p.attn.register_forward_hook(lambda m, i, o: store.append((o[0] if isinstance(o, tuple) else o)
                                                                       .detach().clone()) if not store else None)
        try:
            p.generate(prompt)
        finally:
            h.remove()
        ref = buggy if name == "buggy" else clean
        live[name] = {"shape": list(store[0].shape), "pos0_equal": bool(torch.equal(store[0][:, 0], ref[:, 0])),
                      "all_equal": bool(torch.equal(store[0], ref))}
    run["capture_vs_live"] = live
    gates_ok.append(live["buggy"]["pos0_equal"])
    d0 = clean[:, 0].float() - buggy[:, 0].float()
    if a.synthetic_delta:      # smoke: pretend the GPU produced a one-ULP difference in block 1
        c0 = clean.clone()
        c0[:, 0, BLOCK + 5] = ulp_step(buggy[:, 0, BLOCK + 5:BLOCK + 6], True)[:, 0]
        clean = c0
        d0 = clean[:, 0].float() - buggy[:, 0].float()
    run["pos0_max_abs_diff"] = float(d0.abs().max())
    run["pos0_n_nonzero"] = int((d0 != 0).sum())
    run["pos0_nonzero_blocks"] = blocks_nonzero(d0)
    print(f"[{a.gpu}] capture vs live: {live}  d0 max {run['pos0_max_abs_diff']} nonzero "
          f"{run['pos0_n_nonzero']} blocks {run['pos0_nonzero_blocks']}", flush=True)

    variants = {"copy": clean}
    if a.gpu == "a100":
        if run["pos0_n_nonzero"] == 0:
            run["premise"] = "absent on this node: position-0 difference is zero"
            torch.save({"clean0": clean[:, 0].cpu(), "buggy0": buggy[:, 0].cpu()}, os.path.join(a.out, "position0.pt"))
            dump(run, path)
            sys.exit(2)
        torch.save({"clean0": clean[:, 0].cpu(), "buggy0": buggy[:, 0].cpu(), "delta_fp32": d0.cpu(),
                    "clean_full": clean.cpu()}, os.path.join(a.out, "position0.pt"))
        exp_blocks = run["pos0_nonzero_blocks"]
        k0 = clean.clone()
        k0[:, 0] = buggy[:, 0]
        variants["keep0"] = k0
        dump(run, path)
        pl0, plinfo = make_placebo(clean[:, 0], -d0)         # mirrors keep0 (removes d0) at other coordinates
    else:
        A = torch.load(a.delta)
        dA = A["delta_fp32"].to(clean.device)
        exp_blocks = blocks_nonzero(dA)
        run["reverse_premise_l40s_d0_zero"] = run["pos0_n_nonzero"] == 0
        inj = clean.clone()
        inj[:, 0] = (buggy[:, 0].float() + dA).to(clean.dtype)
        diff = inj[:, 0].float() - buggy[:, 0].float()
        run["inject_check"] = {"exact_delta": bool(torch.equal(diff, dA)),
                               "n_changed": int((inj[:, 0] != buggy[:, 0]).sum()),
                               "n_delta_nonzero": int((dA != 0).sum()),
                               "buggy0_equal_a100": bool(torch.equal(buggy[:, 0].cpu(), A["buggy0"])),
                               "clean0_equal_a100": bool(torch.equal(clean[:, 0].cpu(), A["clean0"]))}
        variants["inject"] = inj
        gates_ok.append(run["reverse_premise_l40s_d0_zero"])
        gates_ok.append(run["inject_check"]["exact_delta"])
        variants["donorA100"] = A["clean_full"].to(clean.device)
        if not run["inject_check"]["buggy0_equal_a100"]:
            c0 = clean.clone()
            c0[:, 0] = A["clean0"].to(clean.device)
            variants["a100clean0"] = c0
        dump(run, path)
        pl0, plinfo = make_placebo(clean[:, 0], dA)          # mirrors inject (adds dA) at other coordinates
        torch.save({"clean0": clean[:, 0].cpu(), "buggy0": buggy[:, 0].cpu()}, os.path.join(a.out, "position0_l40s.pt"))
        print(f"[l40s] inject check {run['inject_check']}", flush=True)
    pl = clean.clone()
    pl[:, 0] = pl0
    variants["placebo"] = pl
    run["placebo"] = plinfo
    run["exposure_blocks"] = exp_blocks
    exposed = {name for name, heads in cfgs if set(heads) & set(exp_blocks)}
    run["n_exposed_configs"] = len(exposed)
    dump(run, path)

    other = "l40s" if a.gpu == "a100" else "a100"
    run["cells"] = {}
    for scope in CELLS:
        ref_own = load_json(os.path.join(REF[a.gpu], f"results_post_wo__{scope}.json"))["configs"]
        ref_other = load_json(os.path.join(REF[other], f"results_post_wo__{scope}.json"))["configs"]
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
        gate = {"n": len(copy), "text_equal_to_r13": sum(copy[k]["output"] == ref_own[k]["output"] for k in copy)}
        gate["ok"] = gate["text_equal_to_r13"] == gate["n"]
        gates_ok.append(gate["ok"])
        D = [k for k in ref_own if k in copy and ref_own[k]["correct"] != ref_other[k]["correct"]]
        Dx = [k for k in copy if copy[k]["correct"] != ref_other[k]["correct"]]   # exploratory fallback
        pre_gates = live["buggy"]["pos0_equal"] and (a.gpu == "a100" or (run["reverse_premise_l40s_d0_zero"]
                                                                         and run["inject_check"]["exact_delta"]))
        cell = {"gate_copy_reproduces_r13": gate, "placebo_valid": plinfo["valid"]}
        for Dset, suffix in ((D, ""), (Dx, "_exploratory_Dprime")):
            if suffix and gate["ok"]:
                continue
            pr = decide(out["placebo"], copy, ref_other, Dset, exposed, label=False)
            pr["moved_exposed"] = sum(out["placebo"][k]["correct"] == ref_other[k]["correct"]
                                      for k in Dset if k in exposed)
            cell["placebo" + suffix] = pr
            for vname in variants:
                if vname in ("copy", "placebo"):
                    continue
                descriptive = vname in ("donorA100", "a100clean0")
                cell[vname + suffix] = decide(out[vname], copy, ref_other, Dset, exposed,
                                              placebo_moved=None if descriptive else pr["moved_exposed"],
                                              label=not descriptive, placebo_valid=plinfo["valid"])
        if scope == "all_steps":
            refs = r13.reference_outcomes(r8, r1)
            cell["r8_r1_reproduction"] = {v: r13.reproduction_gate(r13.reproduction(out[v], refs))["values"]
                                          for v in variants}
        for vname in ("keep0", "inject", "placebo"):   # these write only into exposed blocks
            if vname in cell:
                u = cell[vname]["unexposed_text_identical"]
                cell[vname]["gate_unexposed_identical"] = u[0] == u[1]
                gates_ok.append(u[0] == u[1])
        cell["interpreted"] = (gate["ok"] and pre_gates and all(cell[v].get("gate_unexposed_identical", True)
                                                                for v in ("keep0", "inject", "placebo") if v in cell))
        run["cells"][scope] = cell
        print(f"[{a.gpu}/{scope}] gate {gate}", flush=True)
        for vname in variants:
            if vname != "copy":
                print(f"[{a.gpu}/{scope}/{vname}] {cell[vname]}", flush=True)
        dump(run, path)
    run["all_gates_ok"] = all(gates_ok)
    run["finished"] = datetime.now().isoformat()
    dump(run, path)
    if not run["all_gates_ok"] and not a.limit:
        sys.exit(2)


if __name__ == "__main__":
    main()
