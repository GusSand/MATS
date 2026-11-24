#!/usr/bin/env python3
"""
Experiment R20: can a steering vector at the end of the question fix the decimal-comparison error on held-out
questions without breaking correct answers?
=============================================================================================================
Experiment file: docs/experiments/11-24_llama8b_steering_fix.md

R19 (VERIFIED-WITH-FLAGS): in big-first format-dependent questions the Q&A flip is carried by the residual stream at
the last fractional token (LFT, the token before "?") at layers 9-14. R20 turns that into an intervention: add a fixed
vector (no per-question donor run) at one prompt position and layer, and measure fixes and breaks by generation.

Questions are R19's 9,600 (3 wordings x X 1..20 x Y 2..9 x Z 10..19 x 2 orders). Split by X (seed 20; X=9 forced into
test): train 8 X, dev 4 X (wordings W1, W2 only), test 8 X (all three wordings; W3 is unseen in train and dev).

Directions (train only, per layer and site):
  fmt  mean over all train questions of resid_simple - resid_QA (label-free format direction)
  fd   the same mean over train format-dependent questions only (simple correct, Q&A bug)
  cb   in Q&A, mean resid of correct minus mean resid of bug answers, within each order, averaged over orders
Sites: LFT, and the final prompt token ("A:" colon). Steering adds alpha * v at that one position, prefill only.

Dev: teacher-forced proxy grid (direction x site x layer x alpha), then generation for the top 3 proxy configs per
direction; each direction's config is chosen by dev generation net (fixes - breaks). Primary direction = highest dev
net. Test: generation for each direction's config plus 3 random same-norm directions each, and the primary applied to
simple-format prompts. Pre-registered decision rule in decide(). Exit code 2 on any gate failure or exception.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R19_PATH = os.path.join(HERE, "..", "exp_R19_last_fraction", "run_experiment.py")
R19_SHA = "a4effdba53010f882d4c00ce243fc8618105acc9a7bece3047f9b3979a03b20e"
R19_RUN = os.path.join(HERE, "..", "exp_R19_last_fraction", "results", "r19_24227404", "run.json")

SPLIT_SEED = 20
N_TEST_X, N_DEV_X = 8, 4                    # train gets the remaining 8
TRAIN_DEV_WORDINGS = ["W1", "W2"]
GRID_LAYERS = [4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]
SITES = ["lft", "colon"]
DIRS = ["fmt", "fd", "cb"]
TOPK = 3
N_RANDOM = 3
BATCH = 64
MAX_NEW = 20
SEED = 20
SIGN_P = 0.01
MAX_BREAK = 0.02
BOTH_FIX = 0.3
FD_FIX = 0.5
QMARK, DOT = 5380, 13


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_r19():
    got = sha256(R19_PATH)
    if got != R19_SHA:
        print(f"R19 sha256 {got} != {R19_SHA}", flush=True)
        sys.exit(2)
    os.environ.setdefault("R19_MODEL", os.environ.get("R20_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    spec = importlib.util.spec_from_file_location("r19", R19_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_xs():
    xs = [x for x in range(1, 21) if x != 9]
    random.Random(SPLIT_SEED).shuffle(xs)
    test = sorted([9] + xs[:N_TEST_X - 1])
    dev = sorted(xs[N_TEST_X - 1:N_TEST_X - 1 + N_DEV_X])
    train = sorted(xs[N_TEST_X - 1 + N_DEV_X:])
    return train, dev, test


def binom_two_sided(k, n):
    """Exact two-sided sign test p (p = 0.5)."""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] * (1 + 1e-9)))


def transitions(base, steered):
    """base/steered: lists of classes. fix = not correct -> correct; break = correct -> not correct."""
    fx = sum(b != "correct" and s == "correct" for b, s in zip(base, steered))
    br = sum(b == "correct" and s != "correct" for b, s in zip(base, steered))
    return fx, br


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / dd
    return [c - h, c + h]


def decide(t):
    """t: {"fixes", "breaks", "n_base_correct", "random_nets": [..], "null_net" (or None), "both_fix_rate",
    "fd_fix_rate", "prompt_swap_net"}. First match; then the "(<= prompt swap)" qualifier."""
    f, b = t["fixes"], t["breaks"]
    net = f - b
    p = binom_two_sided(f, f + b)
    br = b / t["n_base_correct"] if t["n_base_correct"] else 0.0
    nulls = list(t["random_nets"]) + ([t["null_net"]] if t.get("null_net") is not None else [])
    out = {"net": net, "sign_p": p, "break_rate": br, "break_ci95": wilson(b, t["n_base_correct"]),
           "control_max_net": max(nulls)}
    if net <= 0 or p >= SIGN_P:
        lab = "no reliable fix"
    elif net <= max(nulls):
        lab = "no better than controls"
    elif br > MAX_BREAK:
        lab = "net-positive but damaging"
    elif t["both_fix_rate"] >= BOTH_FIX:
        lab = "fixes errors beyond the format flip"
    elif t["fd_fix_rate"] >= FD_FIX:
        lab = "fixes the format flip only"
    else:
        lab = "small net improvement"
    if lab not in ("no reliable fix", "no better than controls") and net <= t["prompt_swap_net"]:
        lab += " (<= prompt swap)"
    out["label"] = lab
    return out


def decide_selftest():
    base = {"n_base_correct": 1000, "random_nets": [0, 2, -3], "null_net": None, "both_fix_rate": 0.0,
            "fd_fix_rate": 0.0, "prompt_swap_net": 0}
    assert decide({**base, "fixes": 5, "breaks": 6})["label"] == "no reliable fix"
    assert decide({**base, "fixes": 12, "breaks": 6})["label"] == "no reliable fix"          # p >= 0.01
    assert decide({**base, "fixes": 40, "breaks": 5, "random_nets": [50]})["label"] == "no better than controls"
    assert decide({**base, "fixes": 40, "breaks": 5, "null_net": 35})["label"] == "no better than controls"
    assert decide({**base, "fixes": 200, "breaks": 30})["label"] == "net-positive but damaging"
    assert decide({**base, "fixes": 40, "breaks": 5, "prompt_swap_net": 35})["label"] == \
        "small net improvement (<= prompt swap)"
    assert decide({**base, "fixes": 5, "breaks": 6, "prompt_swap_net": 35})["label"] == "no reliable fix"
    assert decide({**base, "fixes": 200, "breaks": 20, "both_fix_rate": 0.3})["label"] == \
        "fixes errors beyond the format flip"
    assert decide({**base, "fixes": 200, "breaks": 20, "fd_fix_rate": 0.5})["label"] == "fixes the format flip only"
    assert decide({**base, "fixes": 40, "breaks": 5})["label"] == "small net improvement"
    assert abs(binom_two_sided(0, 10) - 2 / 1024) < 1e-12 and binom_two_sided(5, 10) == 1.0
    assert transitions(["correct", "bug", "neither", "correct"], ["bug", "correct", "correct", "correct"]) == (2, 1)
    print("decide selftest: OK", flush=True)


# ---------- batched forward / generation with an additive steer at one position per row ------------------------
class Steer:
    """Adds vecs[b] to the resid input of decoder layer L at unpadded prompt position pos[b] (prefill only)."""

    def __init__(self, M, L, pos, vecs):
        self.M, self.L, self.pos, self.vecs, self.calls = M, L, pos, vecs, []

    def attach(self, pads, T):
        idx = torch.tensor([p + s for p, s in zip(pads, self.pos)], device=self.M.model.device)
        rows = torch.arange(len(pads), device=self.M.model.device)
        v = self.vecs.to(self.M.model.device).float()

        def fn(h):
            self.calls.append(int(h.shape[1]))
            if len(self.calls) > 1:
                assert h.shape[1] == 1, f"steer: decode call with length {h.shape[1]}"
                return None
            assert h.shape[1] == T, f"steer: prefill length {h.shape[1]} != {T}"
            h = h.clone()
            h[rows, idx] = (h[rows, idx].float() + v).to(h.dtype)
            return h
        self.handle = self.M._register("resid", self.L, fn)

    def detach(self):
        self.handle.remove()


def run_batch(M, id_lists, steer=None, logits_only=False):
    """Left-padded batch (pad = eos). Returns last-position logits [B, V] (fp32) or (texts, new ids)."""
    tok = M.tok
    dev = M.model.device
    T = max(len(x) for x in id_lists)
    pads = [T - len(x) for x in id_lists]
    ids = torch.tensor([[tok.eos_token_id] * p + x for p, x in zip(pads, id_lists)], device=dev)
    mask = torch.tensor([[0] * p + [1] * len(x) for p, x in zip(pads, id_lists)], device=dev)
    if steer is not None:
        for b, (p, x) in enumerate(zip(pads, id_lists)):
            assert ids[b, p + steer.pos[b]].item() == x[steer.pos[b]]
        steer.attach(pads, T)
    try:
        with torch.no_grad():
            if logits_only:
                pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
                return M.model(input_ids=ids, attention_mask=mask, position_ids=pos_ids).logits[:, -1].float()
            out = M.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
    finally:
        if steer is not None:
            steer.detach()
    if steer is not None and not logits_only:
        assert len(steer.calls) >= 2, f"steer calls {steer.calls[:5]}"
    new = [row[T:].tolist() for row in out]
    return [tok.decode(n, skip_special_tokens=True) for n in new], new


def batched(M, qs, key, steer_fn=None, logits_only=False, prefix=False):
    """Run questions qs (dicts) in batches of BATCH in list order. key: 's' (simple) or 'q' (Q&A).
    steer_fn(batch) -> Steer or None. prefix: append the teacher-forced ' X.' tokens (logits mode)."""
    outs = []
    for i in range(0, len(qs), BATCH):
        b = qs[i:i + BATCH]
        ids = [q[key + "_ids"] + (q["pre"] if prefix else []) for q in b]
        st = steer_fn(b) if steer_fn else None
        r = run_batch(M, ids, st, logits_only)
        outs.append(r if logits_only else r[0])
    if logits_only:
        return torch.cat(outs) if outs else torch.empty(0)
    return [t for o in outs for t in o]


# ---------- questions -------------------------------------------------------------------------------------------
def build_questions(tok, r19):
    qs = []
    for w in r19.WORDINGS:
        for X in r19.XS:
            for Y in r19.YS:
                for Z in r19.ZS:
                    for o in ("big_first", "small_first"):
                        big, small = f"{X}.{Y}", f"{X}.{Z}"
                        n1, n2 = (big, small) if o == "big_first" else (small, big)
                        st, qt = r19.texts(w, n1, n2)
                        q = {"wording": w, "X": X, "Y": Y, "Z": Z, "order": o, "big": big, "small": small,
                             "s_text": st, "q_text": qt, "s_ids": tok(st)["input_ids"], "q_ids": tok(qt)["input_ids"],
                             "pre": tok(f" {X}.", add_special_tokens=False)["input_ids"],
                             "yi": tok(str(Y), add_special_tokens=False)["input_ids"],
                             "zi": tok(str(Z), add_special_tokens=False)["input_ids"]}
                        for f in ("s", "q"):
                            ids = q[f + "_ids"]
                            dots = [i for i in range(len(ids)) if ids[i] == DOT]
                            ok = len(dots) == 2 and dots[1] + 2 < len(ids) and ids[dots[1] + 2] == QMARK
                            q[f + "_lft"] = dots[1] + 1 if ok else None
                            q[f + "_colon"] = len(ids) - 1
                        qs.append(q)
    return qs


def site_pos(q, f, site):
    return q[f + "_" + site]


def body(a, run, path, r17, r18, r19):
    M = r17.Model()
    tok = M.tok
    gates = run["gates"]
    nL = M.n_layers
    layers = [L for L in GRID_LAYERS if L < nL] if not a.smoke else list(range(1, nL))
    alphas = ALPHAS if not a.smoke else [1.0, 4.0]
    gen = torch.Generator().manual_seed(SEED)

    def stop_if_failed():
        if not a.smoke and not all(v["ok"] for v in gates.values()):
            run["all_gates_ok"] = False
            run["finished"] = datetime.now().isoformat()
            dump(run, path)
            sys.exit(2)

    qs = build_questions(tok, r19)
    bad = [q for q in qs if q["s_lft"] is None or q["q_lft"] is None or len(q["yi"]) != 1 or len(q["zi"]) != 1
           or q["q_ids"][q["q_lft"]] != q["s_ids"][q["s_lft"]]]
    gates["question_tokenization"] = {"n_bad": len(bad), "ok": not bad}
    for q in qs:
        q["yi"], q["zi"] = q["yi"][0], q["zi"][0]
    train_x, dev_x, test_x = split_xs()
    run["split"] = {"train_X": train_x, "dev_X": dev_x, "test_X": test_x}
    if a.smoke:
        qs = [q for q in qs if q["X"] in (1, 2, 9) and q["Y"] in (2, 8) and q["Z"] in (10, 11)]
        train_x, dev_x, test_x = [1], [2], [9]

    # ---- gate: steering mechanics on the canonical batch ----
    canon = [q for q in qs if q["wording"] == "W1" and q["X"] == 9 and q["Y"] == 8 and q["Z"] == 11
             and q["order"] == "big_first"][0]
    probe = [canon] + [q for q in qs if q is not canon][:7]
    d = M.model.config.hidden_size
    lg0 = batched(M, probe, "q", logits_only=True, prefix=True)
    lgz = batched(M, probe, "q", lambda b: Steer(M, layers[0], [q["q_lft"] for q in b], torch.zeros(len(b), d)),
                  logits_only=True, prefix=True)
    gates["zero_steer_identity"] = {"ok": bool(torch.equal(lg0, lgz))}
    lgn = batched(M, probe, "q", lambda b: Steer(M, layers[0], [q["q_lft"] for q in b], 50 * torch.ones(len(b), d)),
                  logits_only=True, prefix=True)
    gates["active_steer"] = {"max_abs_diff": float((lgn - lg0).abs().max()), "ok": bool((lgn - lg0).abs().max() > 0)}
    g0 = batched(M, probe, "q")
    gz = batched(M, probe, "q", lambda b: Steer(M, layers[0], [q["q_colon"] for q in b], torch.zeros(len(b), d)))
    gates["zero_steer_generation_identity"] = {"ok": g0 == gz}
    single = run_batch(M, [canon["q_ids"] + canon["pre"]], logits_only=True)[0]
    gates["batch_vs_single_logits"] = {"max_abs_diff": float((single - lg0[0]).abs().max()), "ok": True}  # report-only
    dump(run, path)
    print("gates:", {k: v["ok"] for k, v in gates.items()}, flush=True)
    stop_if_failed()

    # ---- baseline: batched generation for every question, both formats (same order and batches as R19) ----
    t0 = time.time()
    ts = batched(M, qs, "s")
    tq = batched(M, qs, "q")
    for q, x, y in zip(qs, ts, tq):
        q["s_out"], q["q_out"] = x, y
        q["s_class"] = r18.classify_pair(x, q["big"], q["small"])
        q["q_class"] = r18.classify_pair(y, q["big"], q["small"])
    run["baseline_seconds"] = time.time() - t0
    if not a.smoke and os.path.exists(R19_RUN):
        ref = {(s["wording"], s["X"], s["Y"], s["Z"], s["order"]): s for s in json.load(open(R19_RUN))["screen"]}
        agr = [(ref[k]["simple_class"] == q["s_class"]) + (ref[k]["qa_class"] == q["q_class"]) for q in qs
               for k in [(q["wording"], q["X"], q["Y"], q["Z"], q["order"])]]
        frac = sum(agr) / (2 * len(qs))
        gates["baseline_matches_r19"] = {"agreement": frac, "ok": frac >= 0.99}
    elif not a.smoke:
        gates["baseline_matches_r19"] = {"ok": False, "why": "R19 run.json missing"}
    if a.smoke:                     # random-weight model: assign seeded labels so every code path runs
        rng = random.Random(SEED)
        for q in qs:
            q["s_class"], q["q_class"] = rng.choice(["correct", "bug"]), rng.choice(["correct", "bug"])
    dump(run, path)
    stop_if_failed()

    def part(xs, wordings):
        return [q for q in qs if q["X"] in xs and q["wording"] in wordings]
    train = part(train_x, TRAIN_DEV_WORDINGS)
    dev = part(dev_x, TRAIN_DEV_WORDINGS)
    test = part(test_x, list(r19.WORDINGS))

    def cat(q):
        s, c = q["s_class"], q["q_class"]
        if s == "correct" and c == "bug":
            return "format_dependent"
        if s == "bug" and c == "bug":
            return "both_bug"
        if s == "bug" and c == "correct":
            return "reverse_dependent"
        if s == "correct" and c == "correct":
            return "both_correct"
        return "other"
    for q in qs:
        q["cat"] = cat(q)
    run["split_counts"] = {nm: {c: sum(q["cat"] == c for q in p) for c in
                                ("format_dependent", "both_bug", "reverse_dependent", "both_correct", "other")}
                           for nm, p in (("train", train), ("dev", dev), ("test", test))}
    print("split:", run["split"], run["split_counts"], flush=True)

    # ---- subset baselines: the same batch composition as the steered runs (fix/break reference) ----
    def gen_cls(p, f, steer_fn=None):
        tx = batched(M, p, f, steer_fn)
        return [r18.classify_pair(t, q["big"], q["small"]) for t, q in zip(tx, p)], tx
    dev_base, _ = gen_cls(dev, "q")
    tbase, tbase_txt = gen_cls(test, "q")
    sbase, sbase_txt = gen_cls(test, "s")
    if a.smoke:
        dev_base = [q["q_class"] for q in dev]
        tbase, sbase = [q["q_class"] for q in test], [q["s_class"] for q in test]
    run["subset_baseline_disagreement"] = {
        "dev_q": sum(x != q["q_class"] for x, q in zip(dev_base, dev)),
        "test_q": sum(x != q["q_class"] for x, q in zip(tbase, test)),
        "test_s": sum(x != q["s_class"] for x, q in zip(sbase, test))}
    pf, pb = transitions(tbase, sbase)
    run["prompt_swap_comparator"] = {"fixes": pf, "breaks": pb, "net": pf - pb,
                                     "note": "Q&A -> simple format, no steering (subset-batched baselines)"}
    print("subset baselines:", run["subset_baseline_disagreement"], run["prompt_swap_comparator"], flush=True)

    # ---- matched label nulls (seeded): cb with labels shuffled within order; fd with a random subset of train ----
    nrng = random.Random(SEED + 1)
    for o in ("big_first", "small_first"):
        grp = [q for q in train if q["order"] == o and q["q_class"] in ("correct", "bug")]
        labs = [q["q_class"] for q in grp]
        nrng.shuffle(labs)
        for q, lab in zip(grp, labs):
            q["null_cb"] = lab
    n_fd = sum(q["cat"] == "format_dependent" for q in train)
    for i in nrng.sample(range(len(train)), n_fd):
        train[i]["null_fd"] = True

    # ---- directions from train: accumulate sums per (layer, site) ----
    acc = {}

    def add(key, L, site, v):
        s = acc.setdefault((key, L, site), [torch.zeros(d, dtype=torch.float64), 0])
        s[0] += v.double().cpu()
        s[1] += 1

    for f in ("s", "q"):
        for i in range(0, len(train), BATCH):
            b = train[i:i + BATCH]
            store = {}
            hs = []
            for L in layers:
                def fn(h, L=L):
                    store[L] = h.detach().clone()
                    return None
                hs.append(M._register("resid", L, fn))
            try:
                run_batch(M, [q[f + "_ids"] for q in b], logits_only=True)
            finally:
                for h in hs:
                    h.remove()
            T = max(len(q[f + "_ids"]) for q in b)
            for j, q in enumerate(b):
                pad = T - len(q[f + "_ids"])
                for L in layers:
                    for site in SITES:
                        v = store[L][j, pad + site_pos(q, f, site)].float()
                        add(f"all_{f}", L, site, v)
                        add(f"norm_{f}", L, site, v.norm().reshape(1).expand(d))
                        if q["cat"] == "format_dependent":
                            add(f"fd_{f}", L, site, v)
                        if q.get("null_fd"):
                            add(f"nfd_{f}", L, site, v)
                        if f == "q" and q["q_class"] in ("correct", "bug"):
                            add(f"{q['q_class']}_{q['order']}", L, site, v)
                            add(f"n{q['null_cb']}_{q['order']}", L, site, v)

    def mean(key, L, site):
        s = acc.get((key, L, site))
        return None if s is None or s[1] == 0 else (s[0] / s[1]).float()

    vec = {}
    for L in layers:
        for site in SITES:
            vec[("fmt", L, site)] = mean("all_s", L, site) - mean("all_q", L, site)
            fs, fq = mean("fd_s", L, site), mean("fd_q", L, site)
            vec[("fd", L, site)] = None if fs is None else fs - fq
            per = []
            for o in ("big_first", "small_first"):
                c, bg = mean(f"correct_{o}", L, site), mean(f"bug_{o}", L, site)
                if c is not None and bg is not None:
                    per.append(c - bg)
            vec[("cb", L, site)] = None if len(per) < 2 else (per[0] + per[1]) / 2
            fs, fq = mean("nfd_s", L, site), mean("nfd_q", L, site)
            vec[("fd_null", L, site)] = None if fs is None else fs - fq
            per = []
            for o in ("big_first", "small_first"):
                c, bg = mean(f"ncorrect_{o}", L, site), mean(f"nbug_{o}", L, site)
                if c is not None and bg is not None:
                    per.append(c - bg)
            vec[("cb_null", L, site)] = None if len(per) < 2 else (per[0] + per[1]) / 2
    run["direction_norms"] = {f"{k[0]}|L{k[1]}|{k[2]}": (None if v is None else float(v.norm())) for k, v in vec.items()}
    run["resid_norm_q"] = {f"L{L}|{site}": {"norm_of_mean": float(mean("all_q", L, site).norm()),
                                            "mean_token_norm": float(mean("norm_q", L, site)[0])}
                           for L in layers for site in SITES}
    run["direction_counts"] = {k: acc[(k, layers[0], "lft")][1] for k in
                               ("all_s", "all_q", "fd_s", "correct_big_first", "bug_big_first", "correct_small_first",
                                "bug_small_first") if (k, layers[0], "lft") in acc}
    dump(run, path)

    def steer_for(direction, L, site, alpha, f="q", v_override=None):
        v = v_override if v_override is not None else vec[(direction, L, site)]
        return lambda b: Steer(M, L, [site_pos(q, f, site) for q in b], (alpha * v).unsqueeze(0).repeat(len(b), 1))

    # ---- dev proxy grid (teacher-forced Δ sign on Q&A dev questions) ----
    t0 = time.time()

    def deltas(lg, b):
        return [float(lg[j, q["yi"]] - lg[j, q["zi"]]) for j, q in enumerate(b)]
    base_d = deltas(batched(M, dev, "q", logits_only=True, prefix=True), dev)
    proxy = []
    for direction in DIRS:
        for site in SITES:
            for L in layers:
                if vec[(direction, L, site)] is None:
                    continue
                for al in alphas:
                    dd = deltas(batched(M, dev, "q", steer_for(direction, L, site, al), logits_only=True, prefix=True),
                                dev)
                    fx = sum(b0 <= 0 < x for b0, x in zip(base_d, dd))
                    br = sum(x <= 0 < b0 for b0, x in zip(base_d, dd))
                    proxy.append({"dir": direction, "site": site, "layer": L, "alpha": al, "fix": fx, "break": br,
                                  "net": fx - br})
    run["dev_proxy"] = proxy
    run["dev_proxy_seconds"] = time.time() - t0
    dump(run, path)

    # ---- dev generation for the top TOPK proxy configs per direction (ties keep the better proxy rank) ----
    chosen = {}
    run["dev_gen"] = []
    for direction in DIRS:
        cands = sorted([p for p in proxy if p["dir"] == direction],
                       key=lambda p: (-p["net"], p["alpha"], p["layer"], SITES.index(p["site"])))[:TOPK]
        best = None
        for p in cands:
            texts = batched(M, dev, "q", steer_for(direction, p["layer"], p["site"], p["alpha"]))
            cls = [r18.classify_pair(t, q["big"], q["small"]) for t, q in zip(texts, dev)]
            fx, br = transitions(dev_base, cls)
            rec = {**{k: p[k] for k in ("dir", "site", "layer", "alpha")}, "proxy_net": p["net"], "fix": fx,
                   "break": br, "net": fx - br}
            run["dev_gen"].append(rec)
            if best is None or rec["net"] > best["net"]:
                best = rec
        if best is not None:
            chosen[direction] = best
    run["chosen"] = chosen
    primary = max(DIRS, key=lambda k: (chosen[k]["net"] if k in chosen else -1e9, -DIRS.index(k)))
    run["primary_direction"] = primary
    dump(run, path)
    print("chosen:", json.dumps(chosen), "primary:", primary, flush=True)

    # ---- test ----
    run["test"] = {}
    CATS = ("format_dependent", "both_bug", "reverse_dependent", "both_correct", "other")
    disc = [i for i, q in enumerate(test) if q["cat"] in ("format_dependent", "reverse_dependent")]

    def changed_texts(cls, tx):
        return {i: tx[i] for i in range(len(test)) if cls[i] != tbase[i]}
    for direction, c in chosen.items():
        L, site, al = c["layer"], c["site"], c["alpha"]
        v = vec[(direction, L, site)]
        cls, texts = gen_cls(test, "q", steer_for(direction, L, site, al))
        fx, br = transitions(tbase, cls)
        controls = {}
        for k in range(N_RANDOM):
            u = torch.randn(d, generator=gen)
            u = u / u.norm() * v.norm()
            rc, rt = gen_cls(test, "q", steer_for(direction, L, site, al, v_override=u))
            rf, rb = transitions(tbase, rc)
            controls[f"random{k}"] = {"fixes": rf, "breaks": rb, "net": rf - rb, "changed_texts": changed_texts(rc, rt)}
        nv = vec.get((direction + "_null", L, site))
        if nv is not None:
            rc, rt = gen_cls(test, "q", steer_for(direction, L, site, al, v_override=nv / nv.norm() * v.norm()))
            rf, rb = transitions(tbase, rc)
            controls["label_null"] = {"fixes": rf, "breaks": rb, "net": rf - rb, "changed_texts": changed_texts(rc, rt),
                                      "cos_v_null": float(torch.nn.functional.cosine_similarity(v, nv, dim=0))}
        if direction == "fd" and vec.get(("fmt", L, site)) is not None:
            controls["cos_fd_fmt"] = float(torch.nn.functional.cosine_similarity(v, vec[("fmt", L, site)], dim=0))
        rc, rt = gen_cls(test, "q", steer_for(direction, L, site, al, v_override=-v))
        rf, rb = transitions(tbase, rc)
        controls["minus_v"] = {"fixes": rf, "breaks": rb, "net": rf - rb, "report_only": True,
                               "changed_texts": changed_texts(rc, rt)}
        rnets = [controls[f"random{k}"]["net"] for k in range(N_RANDOM)]
        strata = {}
        for k in CATS:
            idx = [i for i, q in enumerate(test) if q["cat"] == k]
            sf, sb = transitions([tbase[i] for i in idx], [cls[i] for i in idx])
            strata[k] = {"n": len(idx), "fixes": sf, "breaks": sb,
                         "n_subset_base_differs": sum(tbase[i] != test[i]["q_class"] for i in idx)}
            for o in ("big_first", "small_first"):
                io = [i for i in idx if test[i]["order"] == o]
                sf, sb = transitions([tbase[i] for i in io], [cls[i] for i in io])
                strata[k][o] = {"n": len(io), "fixes": sf, "breaks": sb}
        by = {}
        for key, vals in (("order", ("big_first", "small_first")), ("wording", tuple(r19.WORDINGS))):
            for val in vals:
                idx = [i for i, q in enumerate(test) if q[key] == val]
                sf, sb = transitions([tbase[i] for i in idx], [cls[i] for i in idx])
                by[f"{key}={val}"] = {"n": len(idx), "fixes": sf, "breaks": sb,
                                      "n_base_correct": sum(tbase[i] == "correct" for i in idx)}
        n_bc = sum(b == "correct" for b in tbase)
        nfd, nbb = strata["format_dependent"]["n"], strata["both_bug"]["n"]
        t = {"fixes": fx, "breaks": br, "n_base_correct": n_bc, "random_nets": rnets,
             "null_net": controls["label_null"]["net"] if "label_null" in controls else None,
             "prompt_swap_net": run["prompt_swap_comparator"]["net"],
             "fd_fix_rate": strata["format_dependent"]["fixes"] / nfd if nfd else 0.0,
             "both_fix_rate": strata["both_bug"]["fixes"] / nbb if nbb else 0.0}
        res = {"config": c, "n": len(test), "n_base_correct": n_bc, "base_correct": n_bc,
               "steered_correct": sum(x == "correct" for x in cls), "controls": controls, "strata": strata, "by": by,
               "fd_fix_rate": t["fd_fix_rate"], "both_fix_rate": t["both_fix_rate"],
               "format_swap_index": {"n_discordant": len(disc),
                                     "steered_agrees_simple": sum(cls[i] == sbase[i] for i in disc),
                                     "unsteered_agrees_simple": sum(tbase[i] == sbase[i] for i in disc)},
               "decision": decide(t), "classes": cls, "texts": texts}
        run["test"][direction] = res
        dump(run, path)
        print(f"test {direction}:", json.dumps({k: res[k] for k in ("config", "decision", "strata")}), flush=True)

    # primary applied to simple-format prompts; prompt-swap comparator; canonical
    c = chosen[primary]
    sc, st = gen_cls(test, "s", steer_for(primary, c["layer"], c["site"], c["alpha"], f="s"))
    sf, sb = transitions(sbase, sc)
    run["primary_on_simple"] = {"fixes": sf, "breaks": sb, "n_base_correct": sum(x == "correct" for x in sbase),
                                "changed_texts": {i: st[i] for i in range(len(test)) if sc[i] != sbase[i]}}
    cs = run_batch(M, [canon["q_ids"]], Steer(M, c["layer"], [site_pos(canon, "q", c["site"])],
                                               (c["alpha"] * vec[(primary, c["layer"], c["site"])]).unsqueeze(0)))[0][0]
    run["canonical"] = {"in_test": canon["X"] in test_x, "base_class": canon["q_class"], "steered_text": cs,
                        "steered_class": r18.classify_pair(cs, canon["big"], canon["small"])}
    # report-only: batched vs single-row steered generation on up to 32 changed rows + 32 seeded random test rows
    pcls = run["test"][primary]["classes"]
    rs = random.Random(SEED)
    chg = [i for i in range(len(test)) if pcls[i] != tbase[i]]
    samp = rs.sample(chg, min(32, len(chg))) + rs.sample(range(len(test)), min(32, len(test)))
    agree = 0
    for i in samp:
        q = test[i]
        txt = run_batch(M, [q["q_ids"]], Steer(M, c["layer"], [site_pos(q, "q", c["site"])],
                                                (c["alpha"] * vec[(primary, c["layer"], c["site"])]).unsqueeze(0)))[0][0]
        agree += r18.classify_pair(txt, q["big"], q["small"]) == pcls[i]
    run["steered_batch_vs_single"] = {"agree": agree, "n": len(samp)}
    lg1 = batched(M, probe, "q", logits_only=True, prefix=True)
    gates["hooks_removed"] = {"ok": bool(torch.equal(lg0, lg1))}
    run["base_test"] = [{**{k: q[k] for k in ("wording", "X", "Y", "Z", "order", "s_class", "q_class", "cat")},
                         "q_base_subset": tbase[i], "s_base_subset": sbase[i], "q_out_subset": tbase_txt[i],
                         "s_out_subset": sbase_txt[i]} for i, q in enumerate(test)]
    run["baseline_texts_all"] = [{**{k: q[k] for k in ("wording", "X", "Y", "Z", "order")}, "s": q["s_out"],
                                  "q": q["q_out"]} for q in qs]
    run["all_gates_ok"] = all(v["ok"] for v in gates.values())
    run["finished"] = datetime.now().isoformat()
    dump(run, path)
    print("primary:", primary, json.dumps(run["test"][primary]["decision"]), flush=True)
    print("canonical:", json.dumps(run["canonical"]), flush=True)
    print("all gates ok:", run["all_gates_ok"], flush=True)
    if not a.smoke and not run["all_gates_ok"]:
        sys.exit(2)


def provenance(model_name, r19_sha):
    import socket
    import subprocess
    import transformers
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                                      text=True).strip()
    except Exception:
        pass
    return {"hostname": socket.gethostname(), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": gpu, "slurm_job": os.environ.get("SLURM_JOB_ID"), "git_sha": os.environ.get("R20_GIT_SHA"),
            "script_sha256": sha256(os.path.abspath(__file__)), "r19_sha256": r19_sha, "model": model_name}


def main():
    try:
        _main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny model on CPU: few questions, seeded fake labels")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    decide_selftest()
    if a.selftest:
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "run.json")
    r19 = load_r19()
    r18, _ = r19.load_r18()
    r18.classify_selftest()
    r17 = r18.load_r17()
    run = {"provenance": provenance(r17.MODEL_NAME, R19_SHA), "started": datetime.now().isoformat(),
           "command": " ".join(sys.argv), "gates": {}}
    dump(run, path)
    try:
        body(a, run, path, r17, r18, r19)
    except SystemExit:
        raise
    except Exception:
        run["error"] = traceback.format_exc()
        run["all_gates_ok"] = False
        run["finished"] = datetime.now().isoformat()
        dump(run, path)
        print(run["error"], flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
