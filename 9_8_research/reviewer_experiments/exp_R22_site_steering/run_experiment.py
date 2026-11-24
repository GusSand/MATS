#!/usr/bin/env python3
"""
Experiment R22: can a fixed vector at R19's site (last fractional token, layers 9-14) repair big-first format-dependent
questions, and does R19's per-question patch carry question-specific information?
Experiment file: docs/experiments/11-24_llama8b_site_steering.md
Imports R20 (sha256 checked) for batched steering/generation and the sign test; R19/R18/R17 through R20.
Exit code 2 on any gate failure or exception.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from fractions import Fraction

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R20_PATH = os.path.join(HERE, "..", "exp_R20_steering_fix", "run_experiment.py")
R20_SHA = "effa9b292313395ab19519a0f810a021d76e53f656d6dd1e72b7ed7966fcf5f1"
R17_REF = os.path.join(HERE, "..", "exp_R17_localization_map", "results", "r17_24213609", "run.json")

WORDINGS = {"W1": "Which is bigger:", "W2": "Which is larger:", "W3": "Which number is bigger:",
            "W4": "Which is greater:", "W5": "Which number is larger:",
            "W6": "Which number is greater:", "W7": "Which of these is bigger:",
            "W8": "Which of these numbers is larger:", "W9": "Which decimal is bigger:"}
FIT_W, TEST_W = ["W1", "W2", "W3", "W4", "W5"], ["W6", "W7", "W8", "W9"]
XS, YS, ZS = range(1, 41), range(2, 10), range(10, 20)
FRAC_TRAIN, FRAC_DEV = 0.3, 0.2                 # of (X, Y, Z) triples; the rest is test
LAYERS = list(range(9, 15))
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0]
DIRS = ["fd", "fmt", "cb"]
HEAD_LAYER = 9                                  # fixed: the headline is fd at alpha 1, layer 9 (R19's modal layer)
N_RANDOM, N_NULL, N_OTHER = 20, 5, 5
CAP = {"both_bug": 300, "both_correct": 1500, "other": 100}     # test eval subset; fd and reverse are never capped
CAP_DEV = {"both_bug": 100, "both_correct": 300, "other": 50}
MIN_TEST_FD, MIN_DEV_FD, MIN_TRAIN_FD = 20, 10, 10
CEILING, FIX, MAX_BREAK = 0.5, 0.5, 0.02
SEED = 21
CATS = ("format_dependent", "both_bug", "reverse_dependent", "both_correct", "other")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dump(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def load_r20():
    got = sha256(R20_PATH)
    if got != R20_SHA:
        print(f"R20 sha256 {got} != {R20_SHA}", flush=True)
        sys.exit(2)
    os.environ.setdefault("R20_MODEL", os.environ.get("R22_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    spec = importlib.util.spec_from_file_location("r20", R20_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def texts(w, n1, n2):
    return f"{WORDINGS[w]} {n1} or {n2}?\nAnswer:", f"Q: {WORDINGS[w]} {n1} or {n2}?\nA:"


def decide(t):
    """First match. own_rand: fix rates of random vectors with each question's own delta norm."""
    if (t["n_test_fd"] < MIN_TEST_FD or t["n_train_fd"] < MIN_TRAIN_FD
            or (t["n_dev_fd"] is not None and t["n_dev_fd"] < MIN_DEV_FD)):      # None: headline, uses no dev data
        return "descriptive only (too few format-dependent questions)"
    if t["own_fix"] < CEILING:
        return "site does not replicate on fresh questions"
    if t["own_fix"] - sum(t["own_rand"]) / len(t["own_rand"]) < 0.3:
        return "own patch not distinguishable from random pushes"
    if t["fix"] < FIX:
        return "vector does not repair the site"
    if t["fix"] <= max(t["rand_fix"]):
        return "no better than random"
    if t["null_fix"] and t["fix"] <= max(t["null_fix"]):
        return "no better than a generic format vector"
    if (t["net"] > max([0] + t["rand_net"] + t["null_net"]) and t["net"] + t["small_net"] > 0 and t["break_rate"] <= MAX_BREAK
            and t["small_break_rate"] <= MAX_BREAK):
        return "usable fix on big-first questions"
    return "repairs the site but not usable"


def decide_specific(n_fd, own, other, own_rand):
    """Question specificity: own patch vs same-wording patches of other test format-dependent questions."""
    m, r = sum(other) / len(other), sum(own_rand) / len(own_rand)
    if n_fd < MIN_TEST_FD:
        return "not testable (too few questions)"
    if own < CEILING or own - r < 0.3:
        return "not testable (own patch does not beat random pushes)"
    if own - m >= 0.3:
        return "question-specific"
    if own - m <= 0.1 and m > max(own_rand):
        return "shared across questions"
    return "mixed"


def selftest():
    b = dict(n_test_fd=30, n_dev_fd=15, n_train_fd=20, own_fix=.9, own_rand=[.1, .2], fix=.8, net=40, break_rate=.02,
             small_break_rate=.02, rand_fix=[.1, .7], rand_net=[3], null_fix=[.2], null_net=[5],
             small_net=-10)
    assert decide(b).startswith("usable")
    for k, v in (("n_dev_fd", 9), ("n_test_fd", 19), ("n_train_fd", 9)):
        assert decide({**b, k: v}).startswith("descriptive"), k
    assert decide({**b, "own_fix": .4}).startswith("site does not")
    assert decide({**b, "own_rand": [.7, .9]}) == "own patch not distinguishable from random pushes"
    assert decide({**b, "n_dev_fd": None}).startswith("usable") and decide({**b, "n_dev_fd": None, "n_test_fd": 19}).startswith("desc")
    assert decide({**b, "fix": .49}).startswith("vector does not")
    assert decide({**b, "rand_fix": [.8]}) == "no better than random"
    assert decide({**b, "null_fix": [.8]}) == "no better than a generic format vector"
    for k, v in (("break_rate", .021), ("small_break_rate", .021), ("net", -1), ("small_net", -40), ("rand_net", [50]), ("null_net", [40])):
        assert decide({**b, k: v}) == "repairs the site but not usable", k
    assert decide({**b, "null_fix": [], "null_net": []}).startswith("usable")
    assert decide_specific(30, .9, [.2, .4], [.1]) == "question-specific"
    assert decide_specific(30, .9, [.85], [.1]) == "shared across questions"
    assert decide_specific(30, .9, [.7], [.1]) == "mixed"
    assert decide_specific(30, .9, [.2], [.8]).startswith("not testable (own")
    assert decide_specific(19, .9, [.2], [.1]).startswith("not testable (too")
    print("selftest: OK", flush=True)


def build(tok, r20):
    qs, bad = [], 0
    for w in WORDINGS:
        for X in XS:
            for Y in YS:
                for Z in ZS:
                    for o in ("big_first", "small_first"):
                        big, small = f"{X}.{Y}", f"{X}.{Z}"
                        n1, n2 = (big, small) if o == "big_first" else (small, big)
                        st, qt = texts(w, n1, n2)
                        q = {"wording": w, "X": X, "Y": Y, "Z": Z, "order": o, "big": big, "small": small,
                             "s_ids": tok(st)["input_ids"], "q_ids": tok(qt)["input_ids"],
                             "pre": tok(f" {X}.", add_special_tokens=False)["input_ids"]}
                        yi, zi = (tok(str(v), add_special_tokens=False)["input_ids"] for v in (Y, Z))
                        ok = len(yi) == 1 and len(zi) == 1 and Fraction(big) > Fraction(small)
                        q["yi"], q["zi"] = yi[0], zi[0]
                        for f in ("s", "q"):
                            ids = q[f + "_ids"]
                            dots = [i for i, t in enumerate(ids) if t == r20.DOT]
                            ok &= len(dots) == 2 and dots[1] + 2 < len(ids) and ids[dots[1] + 2] == r20.QMARK
                            q[f + "_lft"] = dots[1] + 1 if len(dots) == 2 else 0
                        d, r = q["s_ids"], q["q_ids"]
                        ok &= len(r) == len(d) + 2 and [i for i in range(1, len(d)) if d[i] != r[i + 2]] == [1, len(d) - 2]
                        ok &= q["q_lft"] == q["s_lft"] + 2
                        bad += not ok
                        qs.append(q)
    return qs, bad


def category(s, c):
    if "neither" in (s, c):
        return "other"
    return {("correct", "bug"): "format_dependent", ("bug", "bug"): "both_bug",
            ("bug", "correct"): "reverse_dependent", ("correct", "correct"): "both_correct"}[(s, c)]


def subset(pop, caps, seed):
    """All format-dependent and reverse questions, a seeded sample of the rest; weight = N_category / n_category."""
    out = []
    for c in CATS:
        pool = [q for q in pop if q["cat"] == c]
        take = pool if c not in caps or len(pool) <= caps[c] else random.Random(seed).sample(pool, caps[c])
        out += [(q, len(pool) / len(take)) for q in take]
    return [q for q, _ in out], [w for _, w in out]


def summarise(r20, qs, w, base, cls):
    fix = [b != "correct" and c == "correct" for b, c in zip(base, cls)]
    brk = [b == "correct" and c != "correct" for b, c in zip(base, cls)]
    fd = [i for i, q in enumerate(qs) if q["cat"] == "format_dependent"]
    nc = sum(wi for wi, b in zip(w, base) if b == "correct")
    by = {c: {"n": sum(q["cat"] == c for q in qs), "fix": sum(f for f, q in zip(fix, qs) if q["cat"] == c),
              "break": sum(f for f, q in zip(brk, qs) if q["cat"] == c)} for c in CATS}
    return {"fd_n": len(fd), "fd_correct": sum(cls[i] == "correct" for i in fd),
            "fd_fix_rate": sum(cls[i] == "correct" for i in fd) / len(fd) if fd else 0.0,
            "fd_ci95": r20.wilson(sum(cls[i] == "correct" for i in fd), len(fd)), "by_cat": by,
            "fd_base_correct": sum(base[i] == "correct" for i in fd),
            "net_weighted": sum(wi * (f - b) for wi, f, b in zip(w, fix, brk)),
            "break_rate_weighted": sum(wi * b for wi, b in zip(w, brk)) / nc if nc else 0.0}


def body(a, run, path, r17, r18, r20):
    M = r17.Model()
    tok, gates = M.tok, run["gates"]
    dim, nL = M.model.config.hidden_size, M.n_layers
    layers = LAYERS if not a.smoke else list(range(1, nL))
    alphas = ALPHAS if not a.smoke else [1.0, 2.0]
    n_random, n_null, n_other = (N_RANDOM, N_NULL, N_OTHER) if not a.smoke else (2, 2, 2)

    def save():
        dump(run, path)

    def stop_if_failed():
        if not a.smoke and not all(g["ok"] for g in gates.values()):
            run["all_gates_ok"], run["finished"] = False, datetime.now().isoformat()
            save()
            sys.exit(2)

    qs, bad = build(tok, r20)
    gates["tokenization_alignment"] = {"n_bad": bad, "n": len(qs), "ok": bad == 0}

    # ---- split of (X, Y, Z) triples (seed 21); wordings W6-W9 only in test ----
    tr = [(X, Y, Z) for X in XS for Y in YS for Z in ZS]
    random.Random(SEED).shuffle(tr)
    n1, n2 = int(FRAC_TRAIN * len(tr)), int((FRAC_TRAIN + FRAC_DEV) * len(tr))
    part = {t: ("train" if i < n1 else "dev" if i < n2 else "test") for i, t in enumerate(tr)}
    for q in qs:
        p = part[(q["X"], q["Y"], q["Z"])]
        q["part"] = p if (q["wording"] in TEST_W) == (p == "test") else None
    qs = [q for q in qs if q["part"]]
    if a.smoke:
        keep = {p: [t for t in tr if part[t] == p][:8] for p in ("train", "dev", "test")}
        qs = [q for q in qs if (q["X"], q["Y"], q["Z"]) in keep[q["part"]]]

    def capture(sub, f):
        """resid at the LFT for each question: [n, len(layers), dim] (cpu fp32)."""
        out = torch.zeros(len(sub), len(layers), dim)
        for i in range(0, len(sub), r20.BATCH):
            b = sub[i:i + r20.BATCH]
            T = max(len(q[f + "_ids"]) for q in b)
            store, hs = {}, []
            for L in layers:
                def fn(h, L=L):
                    store[L] = h.detach().clone()
                    return None
                hs.append(M._register("resid", L, fn))
            try:
                r20.run_batch(M, [q[f + "_ids"] for q in b], logits_only=True)
            finally:
                for h in hs:
                    h.remove()
            for j, q in enumerate(b):
                pos = T - len(q[f + "_ids"]) + q[f + "_lft"]
                for k, L in enumerate(layers):
                    out[i + j, k] = store[L][j, pos].float().cpu()
        return out

    def steer(L, vecs):
        """vecs: [dim] for all questions or [n, dim] per question, aligned with the question list."""
        def make(qlist):
            idx = {id(q): i for i, q in enumerate(qlist)}

            def fn(b):
                v = vecs.unsqueeze(0).repeat(len(b), 1) if vecs.dim() == 1 else vecs[[idx[id(q)] for q in b]]
                return r20.Steer(M, L, [q["q_lft"] for q in b], v)
            return fn
        return make

    def gen(qlist, f="q", st=None):
        tx = r20.batched(M, qlist, f, st(qlist) if st else None)
        return [r18.classify_pair(t, q["big"], q["small"]) for t, q in zip(tx, qlist)], tx

    # ---- gate: own-delta addition on the canonical question reproduces R17's p13 replacement patch ----
    can_q = tok(texts("W1", "9.8", "9.11")[1])["input_ids"]
    can = {"s_ids": tok(texts("W1", "9.8", "9.11")[0])["input_ids"], "q_ids": can_q, "s_lft": 13, "q_lft": 15,
           "pre": tok(" 9.", add_special_tokens=False)["input_ids"], "yi": tok("8", add_special_tokens=False)["input_ids"][0],
           "zi": tok("11", add_special_tokens=False)["input_ids"][0], "big": "9.8", "small": "9.11"}
    lg0 = r20.batched(M, [can], "q", logits_only=True, prefix=True)
    d_of = lambda lg: float(lg[0, can["yi"]] - lg[0, can["zi"]])
    dc, db = d_of(r20.batched(M, [can], "s", logits_only=True, prefix=True)), d_of(lg0)
    own = capture([can], "s") - capture([can], "q")
    rs = {}
    for k, L in enumerate(layers):
        rs[L] = (d_of(r20.batched(M, [can], "q", steer(L, own[:, k])([can]), logits_only=True, prefix=True)) - db) / (dc - db)
    zero = r20.batched(M, [can], "q", steer(layers[0], torch.zeros(dim))([can]), logits_only=True, prefix=True)
    gates["zero_steer_identity"] = {"ok": bool(torch.equal(zero, lg0))}
    ok = False
    if not a.smoke:
        ref = {c["layer"]: c["r"] for c in json.load(open(R17_REF))["cells"] if c["comp"] == "resid" and c["set"] == "p13"}
        ok = all(abs(rs[L] - ref[L]) <= 0.05 for L in layers) and abs(dc - 1.171875) <= 0.04 and abs(db + 0.90625) <= 0.04
    gates["r17_reproduction"] = {"r": rs, "d_clean": dc, "d_buggy": db, "ok": ok}
    save()
    print("gates:", {k: g["ok"] for k, g in gates.items()}, flush=True)
    stop_if_failed()

    # ---- gate: mixed-length, left-padded batch. Own delta added -> residual equals the simple-format one at each LFT ----
    mix = [q for w_ in WORDINGS for q in [x for x in qs if x["wording"] == w_][:4]][:r20.BATCH]
    md = capture(mix, "s")
    mq = capture(mix, "q")
    L0, k0 = layers[0], 0
    T = max(len(q["q_ids"]) for q in mix)
    grabbed = {}

    def grab(h):
        grabbed["h"] = h.detach().clone()
        return None
    st = steer(L0, md[:, k0] - mq[:, k0])(mix)(mix)
    orig_attach = st.attach

    def attach(pads, T_):
        orig_attach(pads, T_)
        grabbed["hook"] = M._register("resid", L0, grab)      # registered after the steer hook: sees its output
    st.attach = attach
    r20.run_batch(M, [q["q_ids"] for q in mix], st, logits_only=True)
    grabbed["hook"].remove()
    err = max(float((grabbed["h"][j, T - len(q["q_ids"]) + q["q_lft"]].float().cpu() - md[j, k0]).norm()
                    / md[j, k0].norm()) for j, q in enumerate(mix))
    z = [r20.batched(M, mix, "q", steer(L0, torch.zeros(dim))(mix) if on else None) for on in (False, True)]
    gates["mixed_batch_steer"] = {"max_rel_err": err, "n_lengths": len({len(q["q_ids"]) for q in mix}),
                                  "zero_generation_identical": z[0] == z[1],
                                  "ok": err < 0.01 and z[0] == z[1] and len({len(q["q_ids"]) for q in mix}) > 1}
    save()
    print("mixed batch:", gates["mixed_batch_steer"], flush=True)
    stop_if_failed()

    # ---- screen (batched, unsteered, both formats) ----
    t0 = time.time()
    sc, sx = gen(qs, "s")
    qc, qx = gen(qs, "q")
    for q, s_, c_ in zip(qs, sc, qc):
        q["s_class"], q["q_class"] = s_, c_
    if a.smoke:
        rng = random.Random(SEED)
        for q in qs:
            q["s_class"], q["q_class"] = rng.choice(["correct", "bug"]), rng.choice(["correct", "bug"])
    for q in qs:
        q["cat"] = category(q["s_class"], q["q_class"])
    P = {p: [q for q in qs if q["part"] == p and q["order"] == "big_first"] for p in ("train", "dev", "test")}
    small = [q for q in qs if q["part"] == "test" and q["order"] == "small_first"]
    dsmall = [q for q in qs if q["part"] == "dev" and q["order"] == "small_first"]
    run["counts"] = {p: {c: sum(q["cat"] == c for q in v) for c in CATS}
                     for p, v in {**P, "test_small_first": small, "dev_small_first": dsmall}.items()}
    run["screen_seconds"] = time.time() - t0
    dump([{**{k: q[k] for k in ("wording", "X", "Y", "Z", "order", "part", "s_class", "q_class", "cat")},
           "s": x, "q": y} for q, x, y in zip(qs, sx, qx)], os.path.join(a.out, "screen.json"))
    save()
    print("counts:", run["counts"], f"{run['screen_seconds']:.0f}s", flush=True)
    nfd = {p: run["counts"][p]["format_dependent"] for p in P}
    if min(nfd["train"], nfd["test"]) == 0:
        run["decision"] = {"primary": {"label": "descriptive only (too few format-dependent questions)"}, "n_fd": nfd}
        run["all_gates_ok"], run["finished"] = all(g["ok"] for g in gates.values()), datetime.now().isoformat()
        save()
        return

    # ---- directions (train, big-first only) ----
    train = P["train"]
    delta = capture(train, "s")
    rq = capture(train, "q")
    delta = delta - rq
    isfd = torch.tensor([q["cat"] == "format_dependent" for q in train])
    qcl = [q["q_class"] for q in train]
    ic = torch.tensor([c == "correct" for c in qcl])
    ib = torch.tensor([c == "bug" for c in qcl])
    vec = {"fd": delta[isfd].mean(0), "fmt": delta.mean(0), "cb": rq[ic].mean(0) - rq[ib].mean(0)}   # [layers, dim]
    fd_deltas = delta[isfd]
    nonfd = [i for i in range(len(train)) if not bool(isfd[i])]
    lab = [i for i, c in enumerate(qcl) if c in ("correct", "bug")]
    nulls = {"fd": [], "cb": []}
    for j in range(n_null):
        nrng = random.Random(SEED * 100 + j)
        pick = []                        # non-format-dependent, matched to the fd set on (wording, Z), with replacement
        for i in [i for i in range(len(train)) if bool(isfd[i])]:
            pool = [m for m in nonfd if train[m]["wording"] == train[i]["wording"] and train[m]["Z"] == train[i]["Z"]]
            pick.append(nrng.choice(pool or nonfd))
        nulls["fd"].append(delta[pick].mean(0))
        perm = lab[:]
        nrng.shuffle(perm)
        k = int(ic.sum())
        nulls["cb"].append(rq[perm[:k]].mean(0) - rq[perm[k:]].mean(0))
    run["norms"] = {d: [float(x.norm()) for x in vec[d]] for d in DIRS}
    cosf = torch.nn.functional.cosine_similarity
    run["cos"] = {"fd_fmt": [float(c) for c in cosf(vec["fd"], vec["fmt"], dim=-1)],
                  "fd_null": [[float(c) for c in cosf(vec["fd"], n, dim=-1)] for n in nulls["fd"]]}
    torch.save({"vec": vec, "nulls": nulls, "fd_deltas": fd_deltas}, os.path.join(a.out, "vectors.pt"))
    run["fd_delta_norm_mean"] = [float(x) for x in fd_deltas.norm(dim=-1).mean(0)]
    run["fd_delta_cos_to_mean"] = [float(torch.nn.functional.cosine_similarity(fd_deltas[:, k], vec["fd"][k][None], dim=-1).mean())
                                   for k in range(len(layers))]
    save()

    # ---- dev selection (generation; weighted net over BOTH orders; secondary configs only) ----
    dbig, dwb = subset(P["dev"], CAP_DEV, SEED)
    dsm, dws = subset(dsmall, {**CAP_DEV, "format_dependent": 100, "reverse_dependent": 100}, SEED)
    dbase, _ = gen(dbig)
    dsbase, _ = gen(dsm)
    run["dev_baseline_disagreement"] = sum(x != q["q_class"] for x, q in zip(dbase + dsbase, dbig + dsm))
    if a.smoke:
        dbase, dsbase = [q["q_class"] for q in dbig], [q["q_class"] for q in dsm]
    run["dev_grid"], chosen, dev_classes = [], {}, {}
    for d in DIRS:
        for k, L in enumerate(layers):
            for al in alphas:
                st = steer(L, al * vec[d][k])
                cls, _ = gen(dbig, st=st)
                scl, _ = gen(dsm, st=st)
                x, y = summarise(r20, dbig, dwb, dbase, cls), summarise(r20, dsm, dws, dsbase, scl)
                rec = {"dir": d, "layer": L, "alpha": al, "net": x["net_weighted"] + y["net_weighted"],
                       "fd_fix_rate": x["fd_fix_rate"], "break_rate": x["break_rate_weighted"],
                       "small_break_rate": y["break_rate_weighted"]}
                run["dev_grid"].append(rec)
                dev_classes[f"{d}|L{L}|a{al}"] = {"big": cls, "small": scl}
                if d not in chosen or rec["net"] > chosen[d]["net"]:       # ties keep the lower layer, then smaller alpha
                    chosen[d] = rec
    dump({"keys": [[q["wording"], q["X"], q["Y"], q["Z"], q["cat"]] for q in dbig],
          "keys_small": [[q["wording"], q["X"], q["Y"], q["Z"], q["cat"]] for q in dsm],
          "base": dbase, "base_small": dsbase, "classes": dev_classes}, os.path.join(a.out, "dev.json"))
    run["chosen"] = chosen
    save()
    print("dev-selected (secondary):", json.dumps(chosen), flush=True)

    # ---- test ----
    ev, w = subset(P["test"], CAP, SEED)
    sev, sw = subset(small, {**CAP, "format_dependent": 300, "reverse_dependent": 300}, SEED)
    base, base_tx = gen(ev)
    sbase, sbase_tx = gen(sev)
    run["subset_baseline_disagreement"] = {"test": sum(x != q["q_class"] for x, q in zip(base, ev)),
                                           "small": sum(x != q["q_class"] for x, q in zip(sbase, sev))}
    if a.smoke:
        base, sbase = [q["q_class"] for q in ev], [q["q_class"] for q in sev]
    key = lambda q: {k: q[k] for k in ("wording", "X", "Y", "Z", "order", "s_class", "q_class", "cat")}
    run["eval"] = [{**key(q), "weight": wi, "base": b, "base_text": t} for q, wi, b, t in zip(ev, w, base, base_tx)]
    run["eval_small"] = [{**key(q), "weight": wi, "base": b, "base_text": t} for q, wi, b, t in zip(sev, sw, sbase, sbase_tx)]
    run["cond"] = {}

    def cond(name, st):
        cls, tx = gen(ev, st=st)
        s = summarise(r20, ev, w, base, cls)
        s["classes"] = cls
        s["changed_texts"] = {i: tx[i] for i in range(len(ev)) if cls[i] != base[i]}
        run["cond"][name] = s
        save()
        return s

    # per-question patches: own donor (every layer); at the headline layer, other questions and own-norm random pushes
    head = HEAD_LAYER if HEAD_LAYER in layers else layers[0]
    kh = layers.index(head)
    ev_own = capture(ev, "s") - capture(ev, "q")
    for k, L in enumerate(layers):
        cond(f"own|L{L}", steer(L, ev_own[:, k]))
    for j in range(n_random):
        u = torch.randn(len(ev), dim, generator=torch.Generator().manual_seed(SEED * 1000 + j))
        u = u / u.norm(dim=-1, keepdim=True) * ev_own[:, kh].norm(dim=-1, keepdim=True)
        cond(f"own_norm_random{j}", steer(head, u))
    fdi = [i for i, q in enumerate(ev) if q["cat"] == "format_dependent"]
    fdq = [ev[i] for i in fdi]
    run["other"] = {"same_wording": [], "train": []}
    partner = {i: [m for m in fdi if m != i and ev[m]["wording"] == ev[i]["wording"]
                   and (ev[m]["X"], ev[m]["Y"], ev[m]["Z"]) != (ev[i]["X"], ev[i]["Y"], ev[i]["Z"])] for i in fdi}
    solo = [i for i in fdi if not partner[i]]
    run["other"]["n_without_same_wording_partner"] = len(solo)      # excluded from own, other and random in Q2
    paired = [n for n, i in enumerate(fdi) if partner[i]]
    for j in range(n_other):
        trng = random.Random(SEED * 200 + j)
        pick = []
        for i in fdi:                    # leave-one-out test donors: same wording, different triple
            pick.append(trng.choice(partner[i] or [i]))
        for nm, v in (("same_wording", ev_own[pick, kh]),
                      ("train", fd_deltas[[trng.randrange(len(fd_deltas)) for _ in fdi], kh])):
            cls, tx = gen(fdq, st=steer(head, v))
            run["other"][nm].append({"fix_rate": sum(cls[n] == "correct" for n in paired) / max(1, len(paired)), "classes": cls, "texts": tx})
    own_fix = run["cond"][f"own|L{head}"]["fd_fix_rate"]
    own_rand = [run["cond"][f"own_norm_random{j}"]["fd_fix_rate"] for j in range(n_random)]
    other = [x["fix_rate"] for x in run["other"]["same_wording"]]
    q2 = lambda nm: sum(run["cond"][nm]["classes"][fdi[n]] == "correct" for n in paired) / max(1, len(paired))
    q2_own, q2_rand = q2(f"own|L{head}"), [q2(f"own_norm_random{j}") for j in range(n_random)]
    run["specificity"] = {"layer": head, "n_paired": len(paired), "own_fix": q2_own, "own_rand": q2_rand,
                          "other_same_wording": other, "other_train": [x["fix_rate"] for x in run["other"]["train"]],
                          "label": decide_specific(len(paired), q2_own, other, q2_rand)}
    torch.save({"fd_eval_index": fdi, "own_patch_headline_layer": ev_own[fdi, kh]}, os.path.join(a.out, "own_fd.pt"))

    # prompt-swap comparator over the whole test population (screen classes)
    sw_f = sum(q["q_class"] != "correct" and q["s_class"] == "correct" for q in P["test"])
    sw_b = sum(q["q_class"] == "correct" and q["s_class"] != "correct" for q in P["test"])
    run["prompt_swap"] = {"fixes": sw_f, "breaks": sw_b, "net": sw_f - sw_b, "n": len(P["test"])}

    run["decision"] = {}
    configs = [("fd", "fd", head, 1.0)] + [(f"{d}_selected", d, chosen[d]["layer"], chosen[d]["alpha"]) for d in DIRS]
    for ci, (name, d, L, al) in enumerate(configs):
        k = layers.index(L)
        v = al * vec[d][k]
        s = cond(name, steer(L, v))
        cond(f"{name}_minus", steer(L, -v))
        rf, rn, nf, nn = [], [], [], []
        for j in range(n_random):
            u = torch.randn(dim, generator=torch.Generator().manual_seed(SEED * 10000 + 100 * ci + j))
            x = cond(f"{name}_random{j}", steer(L, u / u.norm() * v.norm()))
            rf.append(x["fd_fix_rate"])
            rn.append(x["net_weighted"])
        for j, nv in enumerate(nulls.get(d, [])):
            x = cond(f"{name}_null{j}", steer(L, nv[k] / nv[k].norm() * v.norm()))
            nf.append(x["fd_fix_rate"])
            nn.append(x["net_weighted"])
        cls, tx = gen(sev, st=steer(L, v))
        ss = summarise(r20, sev, sw, sbase, cls)
        ss["classes"] = cls
        ss["changed_texts"] = {i: tx[i] for i in range(len(sev)) if cls[i] != sbase[i]}
        run["cond"][f"{name}|small_first"] = ss
        bc, sbc = s["by_cat"]["both_correct"], ss["by_cat"]["both_correct"]
        t = {"dir": d, "layer": L, "alpha": al, "n_test_fd": nfd["test"],
             "n_dev_fd": None if ci == 0 else nfd["dev"],
             "n_train_fd": nfd["train"], "own_fix": own_fix, "own_rand": own_rand, "fix": s["fd_fix_rate"],
             "net": s["net_weighted"], "break_rate": s["break_rate_weighted"],
             "both_correct_break_ci95": r20.wilson(bc["break"], bc["n"]),
             "small_both_correct_break_ci95": r20.wilson(sbc["break"], sbc["n"]),
             "duplicate_of_headline": ci > 0 and (d, L, al) == ("fd", head, 1.0),
             "small_break_rate": ss["break_rate_weighted"], "small_net": ss["net_weighted"], "rand_fix": rf,
             "rand_net": rn, "null_fix": nf, "null_net": nn}
        run["decision"][name] = {**t, "label": decide(t) if ci == 0 else f"{name} (L{L}, alpha {al}): {decide(t)}"}
        save()
        print(name, json.dumps(run["decision"][name]), flush=True)
    lg1 = r20.batched(M, [can], "q", logits_only=True, prefix=True)
    gates["hooks_removed"] = {"ok": bool(torch.equal(lg0, lg1))}
    run["decision"]["primary"] = {"config": "fd, alpha 1, layer 9", "label": run["decision"]["fd"]["label"]}
    run["all_gates_ok"], run["finished"] = all(x["ok"] for x in gates.values()), datetime.now().isoformat()
    save()
    print("primary:", run["decision"]["primary"], "specificity:", run["specificity"]["label"], flush=True)
    print("all gates ok:", run["all_gates_ok"], flush=True)
    if not a.smoke and not run["all_gates_ok"]:
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny model on CPU, seeded fake labels, mechanics only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    run, path = {"gates": {}}, os.path.join(a.out, "run.json")
    try:
        os.makedirs(a.out, exist_ok=True)
        r20 = load_r20()
        r20.decide_selftest()
        r19 = r20.load_r19()
        r18, _ = r19.load_r18()
        r18.classify_selftest()
        r17 = r18.load_r17()
        prov = r20.provenance(r17.MODEL_NAME, r20.R19_SHA)
        prov.update({"script_sha256": sha256(os.path.abspath(__file__)), "r20_sha256": R20_SHA,
                     "git_sha": os.environ.get("R22_GIT_SHA")})
        run.update({"provenance": prov, "started": datetime.now().isoformat(), "command": " ".join(sys.argv)})
        dump(run, path)
        body(a, run, path, r17, r18, r20)
    except SystemExit:
        raise
    except Exception:
        run.update({"error": traceback.format_exc(), "all_gates_ok": False, "finished": datetime.now().isoformat()})
        dump(run, path)
        print(run["error"], flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
