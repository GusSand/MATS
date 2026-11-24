#!/usr/bin/env python3
"""
R21 analysis: pre-registered decision rules on the two GPUs' margins.
Usage: python -I analyze.py <a100_run_dir> <l40s_run_dir> [--out verdicts.json]
Experiment file: docs/experiments/11-24_llama8b_gpu_margin_offset.md

D (per cell) = configs whose R13 class differs between GPUs, read from R13's own result files, not from R21.
M = configs with a decision step on both GPUs. Delta_k = m_A100(k) - m_L40S(k).
p_pos = share of nonzero Delta that is positive. c = median of Delta over M.
Primary margin: m (raw fp16 logits, cast to fp32). It is replaced by the fp32 lens margin at the final hidden
state ("lens_final") when c == 0 or more than 20% of Delta are exactly 0 (design review finding 2).
Both margins are always reported.
"""

import argparse
import hashlib
import json
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R13 = os.path.join(HERE, "..", "exp_R13_true_head_patching", "results")
REF = {"a100": os.path.join(R13, "r13_24084222"), "l40s": os.path.join(R13, "r13_24014365")}
CELLS = ["prefill", "all_steps"]
N_CONFIGS = 1307
R13_SHA256 = "1ce5b0cc2e2cdc74c21de37de5340cd00f5b14e27ae13cf6893d2e09f280348d"
N_PERM = 1000
ZERO_SHARE_MAX = 0.2


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(d, scope):
    with open(os.path.join(d, f"results_post_wo__{scope}.json")) as f:
        return json.load(f)["configs"]


def decide(p_pos, recall, precision, n_nonzero=1):
    """Rules in the order pre-registered; first match wins."""
    if n_nonzero == 0:
        return "no measurable difference"
    if p_pos >= 0.9 and recall >= 0.8 and precision >= 0.5:
        return "shared offset explains the one-way flips"
    if p_pos >= 0.9:
        return "offset present, does not alone predict which configs flip"
    if p_pos <= 0.1:
        return "reverse offset: contradicts the direction of D"
    if 0.3 <= p_pos <= 0.7:
        return "no shared offset: differences are two-sided"
    return f"partial offset (p_pos = {p_pos:.3f})"


def offset_pred(mL, M, c):
    if c >= 0:
        return [k for k in M if mL[k] < 0 < mL[k] + c]
    return [k for k in M if mL[k] + c < 0 < mL[k]]


def sym_rev(mL, M, delta):
    import bisect
    pool = sorted([v for v in delta.values()] + [-v for v in delta.values()])
    n = len(pool)
    return sum(bisect.bisect_left(pool, -mL[k]) / n for k in M if mL[k] > 0)


def score(mA, mL, M, D):
    delta = {k: mA[k] - mL[k] for k in M}
    nz = [k for k in M if delta[k] != 0]
    p_pos = sum(delta[k] > 0 for k in nz) / len(nz) if nz else 0.0
    c = st.median(delta.values())
    Dp = offset_pred(mL, M, c)
    hit = len(set(Dp) & set(D))
    recall = hit / len(D) if D else 0.0
    precision = hit / len(Dp) if Dp else 0.0
    q = st.quantiles(delta.values(), n=4)
    rec_q = {}
    for name, cq in zip(("Q1", "median", "Q3"), (q[0], c, q[2])):
        dp = offset_pred(mL, M, cq)
        h = len(set(dp) & set(D))
        rec_q[name] = {"c": cq, "D_pred": len(dp), "recall": h / len(D) if D else None,
                       "precision": h / len(dp) if dp else None}
    # H0 density check with a window that does not depend on c (finding 1)
    w = st.median(abs(v) for v in delta.values())
    # permutation control (finding 3b): offsets shuffled across configs
    Dset = set(D)
    rng = random.Random(0)
    vals = [delta[k] for k in M]
    fwd_hit, rev = [], []
    for _ in range(N_PERM):
        rng.shuffle(vals)
        fh = r = 0
        for k, d in zip(M, vals):
            if mL[k] < 0 < mL[k] + d and k in Dset:
                fh += 1
            if mL[k] + d < 0 < mL[k]:
                r += 1
        fwd_hit.append(fh)
        rev.append(r)
    out = {"p_pos": p_pos, "n_pos": sum(delta[k] > 0 for k in M), "n_neg": sum(delta[k] < 0 for k in M),
           "n_zero": len(M) - len(nz), "c_median_delta": c, "delta_quartiles": q,
           "delta_min_max": [min(delta.values()), max(delta.values())],
           "D_pred": len(Dp), "D_hit": hit, "recall": recall, "precision": precision, "recall_at_quartiles": rec_q,
           "D_delta": {k: delta.get(k) for k in D},
           "D_sign_ok": sum(1 for k in D if k in delta and mA[k] > 0 > mL[k]),
           "D_with_tie": sum(1 for k in D if k in delta and (mA[k] == 0 or mL[k] == 0)),
           "density_window_w": w,
           "n_mL_in_(-w,0)": sum(-w < mL[k] < 0 for k in M), "n_mL_in_(0,w)": sum(0 < mL[k] < w for k in M),
           # expected reverse flips under H0 (round-2 finding 1): Delta symmetrized (pool Delta and -Delta);
           # for each L40S-correct config, the share of pooled values that would push it below zero
           "expected_reverse_flips_symmetric_H0": sym_rev(mL, M, delta),
           "observed_reverse_flips": sum(mA[k] < 0 < mL[k] for k in M),
           "permutation": {"n_perm": N_PERM, "mean_D_recovered": st.mean(fwd_hit),
                           "mean_reverse_flips": st.mean(rev), "D": len(D)},
           "verdict": decide(p_pos, recall, precision, len(nz)) if D else "descriptive only: D empty"}
    return out, delta


def analyze_cell(A, L, rA, rL):
    assert len(A) == N_CONFIGS and len(L) == N_CONFIGS, (len(A), len(L))
    D = sorted(k for k in rA if rA[k]["correct"] != rL[k]["correct"])
    M = sorted(k for k in A if "m" in A[k] and "m" in L[k])
    out = {"n_configs": len(A), "D": len(D), "M": len(M), "D_in_M": sum(k in set(M) for k in D)}
    if not M:
        out["verdict"] = "descriptive only: M empty"
        return out
    prim, delta = score({k: A[k]["m"] for k in M}, {k: L[k]["m"] for k in M}, M, D)
    lensf, _ = score({k: A[k]["lens"][-1] for k in M}, {k: L[k]["lens"][-1] for k in M}, M, D)
    use_lens = prim["c_median_delta"] == 0 or prim["n_zero"] > ZERO_SHARE_MAX * len(M)
    out["primary_margin"] = "lens_final" if use_lens else "m"
    out["m"], out["lens_final"] = prim, lensf
    out["verdict"] = (lensf if use_lens else prim)["verdict"]
    c = (lensf if use_lens else prim)["c_median_delta"]
    # descriptive: logit-lens profile of Delta
    nl = len(A[M[0]]["lens"])
    prof = [st.median(A[k]["lens"][l] - L[k]["lens"][l] for k in M) for l in range(nl)]
    out["lens_median_delta_by_layer"] = prof
    out["first_layer_abs_ge_half_c"] = next((l for l in range(nl) if c and abs(prof[l]) >= 0.5 * abs(c)), None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a100")
    ap.add_argument("l40s")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = {g: json.load(open(os.path.join(d, "run.json"))) for g, d in (("a100", a.a100), ("l40s", a.l40s))}
    for g, r in runs.items():   # reject smoke, partial or mismatched runs (finding 8)
        assert "--limit" not in r["command"], f"{g}: smoke run"
        assert r["provenance"]["gpu_role"] == g, f"{g}: wrong role"
        assert r.get("finished"), f"{g}: unfinished run"
    for key in ("r21_script_sha256", "r21_git_sha", "token_ids"):
        assert runs["a100"]["provenance"][key] == runs["l40s"]["provenance"][key], key
    for g, r in runs.items():
        assert r["provenance"]["r13_script_sha256"] == R13_SHA256, f"{g}: R13 script changed"
    res = {"analyze_sha256": sha256(__file__),
           "runs": {g: {"dir": d, "job": runs[g]["provenance"].get("slurm_job"), "gpu": runs[g]["provenance"].get("gpu"),
                        "hostname": runs[g]["provenance"].get("hostname"),
                        "same_node_as_r13": runs[g].get("same_node_as_r13"),
                        "same_gpu_name_as_r13": runs[g].get("same_gpu_string_as_r13")} for g, d in (("a100", a.a100), ("l40s", a.l40s))},
           "gates_ok": {g: r.get("all_gates_ok") for g, r in runs.items()},
           "unpatched_m": {g: r["unpatched"].get("m") for g, r in runs.items()}}
    um = res["unpatched_m"]
    res["unpatched_delta"] = um["a100"] - um["l40s"] if None not in um.values() else None
    ul = {g: (r["unpatched"].get("lens") or [None])[-1] for g, r in runs.items()}
    res["unpatched_delta_lens_final"] = ul["a100"] - ul["l40s"] if None not in ul.values() else None
    res["cells"] = {}
    for s in CELLS:
        A, L = load(a.a100, s), load(a.l40s, s)
        rA, rL = load(REF["a100"], s), load(REF["l40s"], s)
        res["cells"][s] = analyze_cell(A, L, rA, rL)
    res["interpreted"] = all(res["gates_ok"].values())
    if not res["interpreted"]:
        for c in res["cells"].values():
            c["verdict_raw"], c["verdict"] = c["verdict"], "not interpreted: gates failed"
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
