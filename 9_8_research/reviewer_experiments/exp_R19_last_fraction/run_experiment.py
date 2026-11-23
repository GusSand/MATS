#!/usr/bin/env python3
"""
Experiment R19: does the format flip live at the question's last fractional token, in both number orders?
=========================================================================================================
Experiment file: docs/experiments/11-23_llama8b_last_fraction.md

Post hoc in R18 (VERIFIED-WITH-FLAGS): in big-first format-dependent questions the multi-digit fractional token
repaired 6/6; in small-first ones the last fractional token before "?" carried the decision. R19 tests, pre-registered,
"the last fractional token (LFT) carries the format flip" on a larger screen: 3 wordings x X 1..20 x Y 2..9 x
Z 10..19 x 2 orders, simple vs Q&A. Imports R18 (sha256 checked) for its classifier, generation and capture helpers,
and R17's Model through R18.

Per eligible question: resid patches (simple -> Q&A, offset 2) at the LFT and at the other fractional token (OFT),
layers 4..14 (primary band 9..14); at each site's best primary-band layer: prefill-only generation, 5 random same-norm
directions, and every other prompt position as competitors; reverse direction reported.
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

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R18_PATH = os.path.join(HERE, "..", "exp_R18_eleven_token", "run_experiment.py")
R18_SHA_PREFIX = "1c5e555b"
R17_REF = os.path.join(HERE, "..", "exp_R17_localization_map", "results", "r17_24213609", "run.json")

WORDINGS = {"W1": "Which is bigger:", "W2": "Which is larger:", "W3": "Which number is bigger:"}
XS, YS, ZS = range(1, 21), range(2, 10), range(10, 20)
LAYERS = list(range(4, 15))
BAND = list(range(9, 15))                  # primary band (R18: specific direction clean at 9-14)
OFFSET = 2
R_REPAIR, R_CONTROL, MARGIN = 0.5, 0.2, 0.3
N_CTRL = 5
MAX_PER_ORDER = 120
MIN_PER_ORDER = 10
BATCH = 64
SEED = 19
QMARK = 5380                               # "?\n"
DOT = 13


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_r18():
    got = sha256(R18_PATH)
    if not got.startswith(R18_SHA_PREFIX):
        print(f"R18 sha256 {got} does not start with {R18_SHA_PREFIX}", flush=True)
        sys.exit(2)
    os.environ.setdefault("R18_MODEL", os.environ.get("R19_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    spec = importlib.util.spec_from_file_location("r18", R18_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, got


def in_r18_grid(q):
    """Questions that generated the hypothesis (R18's grid): excluded from the decision, reported separately."""
    return q["wording"] == "W1" and 6 <= q["Y"] <= 9 and 10 <= q["Z"] <= 13


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [c - h, c + h]


def texts(wording, n1, n2):
    w = WORDINGS[wording]
    return f"{w} {n1} or {n2}?\nAnswer:", f"Q: {w} {n1} or {n2}?\nA:"


def decide(strata):
    """strata: {"big_first": [rows], "small_first": [rows]}; row has lft_repair, oft_repair (bools).
    First match: both hold; one holds (naming any descriptive stratum); no stratum with n >= 10; every testable
    stratum below 0.2 (naming descriptive strata); else partial."""
    out, holds, desc = {}, {}, []
    for k, rows in strata.items():
        n = len(rows)
        kl, ko = sum(r["lft_repair"] for r in rows), sum(r["oft_repair"] for r in rows)
        if n < MIN_PER_ORDER:
            out[k] = {"n": n, "lft_repairs": kl, "oft_repairs": ko, "label": "descriptive only"}
            desc.append(k)
            continue
        lr, orr = kl / n, ko / n
        h = lr >= 0.5 and lr - orr >= 0.3
        out[k] = {"n": n, "lft_rate": lr, "oft_rate": orr, "lft_ci95": wilson(kl, n), "oft_ci95": wilson(ko, n),
                  "holds": h, "lft_below_0.2": lr < 0.2}
        holds[k] = h
    note = "".join(f"; {k} descriptive (n < {MIN_PER_ORDER})" for k in desc)
    if len(holds) == 2 and all(holds.values()):
        lab = "last-fractional-token site holds in both orders"
    elif any(holds.values()):
        lab = "holds for " + ", ".join(k for k, v in holds.items() if v) + " only" + note
    elif not holds:
        lab = "descriptive only (too few eligible questions)"
    elif all(out[k]["lft_below_0.2"] for k in holds):
        lab = "not supported" + note
    else:
        lab = "partial" + note
    out["label"] = lab
    return out


def decide_selftest():
    mk = lambda l, o: {"lft_repair": l, "oft_repair": o}
    good = [mk(True, False)] * 8 + [mk(False, False)] * 4
    assert decide({"big_first": good, "small_first": good})["label"].startswith("last-fractional")
    assert decide({"big_first": good, "small_first": [mk(False, True)] * 12})["label"] == "holds for big_first only"
    assert decide({"big_first": [mk(False, False)] * 12, "small_first": [mk(False, True)] * 12})["label"] == "not supported"
    assert decide({"big_first": good[:5], "small_first": good[:3]})["label"].startswith("descriptive")
    assert decide({"big_first": [mk(True, True)] * 12, "small_first": good})["label"] == "holds for small_first only"
    part = [mk(True, False)] * 4 + [mk(False, False)] * 8          # 0.33: neither holds nor < 0.2
    assert decide({"big_first": part, "small_first": part})["label"] == "partial"
    assert decide({"big_first": good[:5], "small_first": good})["label"] == \
        "holds for small_first only; big_first descriptive (n < 10)"
    assert decide({"big_first": good[:5], "small_first": [mk(False, False)] * 12})["label"] == \
        "not supported; big_first descriptive (n < 10)"
    print("decide selftest: OK", flush=True)


def batch_generate(M, prompts, max_new=20):
    """Left-padded batched greedy generation (screen only; candidates are re-checked unbatched)."""
    tok = M.tok
    old_side, old_pad = tok.padding_side, tok.pad_token
    tok.padding_side, tok.pad_token = "left", tok.eos_token
    out_texts, out_ids = [], []
    try:
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(M.model.device)
            with torch.no_grad():
                out = M.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            T = enc["input_ids"].shape[1]
            for row in out:
                new = row[T:].tolist()
                out_ids.append(new)
                out_texts.append(tok.decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side, tok.pad_token = old_side, old_pad
    return out_texts, out_ids


def provenance(model_name, r18_sha):
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
            "gpu": gpu, "slurm_job": os.environ.get("SLURM_JOB_ID"), "git_sha": os.environ.get("R19_GIT_SHA"),
            "script_sha256": sha256(os.path.abspath(__file__)), "r18_sha256": r18_sha, "model": model_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny model on CPU: small screen, forced B rows")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    decide_selftest()
    if a.selftest:
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "run.json")
    r18, r18_sha = load_r18()
    r18.classify_selftest()
    r17 = r18.load_r17()
    run = {"provenance": provenance(r17.MODEL_NAME, r18_sha), "started": datetime.now().isoformat(),
           "command": " ".join(sys.argv), "gates": {}}
    dump(run, path)
    try:
        body(a, run, path, r17, r18)
    except SystemExit:
        raise
    except Exception:
        run["error"] = traceback.format_exc()
        run["all_gates_ok"] = False
        run["finished"] = datetime.now().isoformat()
        dump(run, path)
        print(run["error"], flush=True)
        sys.exit(2)


def analyse_question(M, r18, gen, q, layers, band, smoke):
    """Unbatched eligibility + site analysis for one question dict (wording, X, Y, Z, order)."""
    tok = M.tok
    X, Y, Z, order, w = q["X"], q["Y"], q["Z"], q["order"], q["wording"]
    big, small = f"{X}.{Y}", f"{X}.{Z}"
    n1, n2 = (big, small) if order == "big_first" else (small, big)
    s_txt, q_txt = texts(w, n1, n2)
    row = {k: q[k] for k in ("wording", "X", "Y", "Z", "order")}
    dp, rp = tok(s_txt)["input_ids"], tok(q_txt)["input_ids"]
    yi = tok(str(Y), add_special_tokens=False)["input_ids"]
    zi = tok(str(Z), add_special_tokens=False)["input_ids"]
    pre = tok(f" {X}.", add_special_tokens=False)["input_ids"]
    if len(yi) != 1 or len(zi) != 1:
        row["skipped"] = "multi-token fractional part"
        return row
    yi, zi = yi[0], zi[0]
    mism = [i for i in range(1, len(dp)) if i + OFFSET >= len(rp) or dp[i] != rp[i + OFFSET]]
    if len(rp) - len(dp) != OFFSET or mism != [1, len(dp) - 2]:
        row["skipped"] = f"alignment {mism}"
        return row
    dots = [i for i in range(len(dp)) if dp[i] == DOT]
    qm = dp.index(QMARK) if QMARK in dp else -1
    if len(dots) != 2 or qm < 0 or dots[1] + 1 != qm - 1:
        row["skipped"] = "position lookup"
        return row
    lft, oft = dots[1] + 1, dots[0] + 1
    row.update({"lft": lft, "oft": oft, "lft_token": dp[lft], "oft_token": dp[oft]})
    # unbatched eligibility
    gd, td = r18.gen_ids(M, dp, 20)
    gr, tr = r18.gen_ids(M, rp, 20)
    cd_, cr_ = r18.classify_pair(td, big, small), r18.classify_pair(tr, big, small)
    row.update({"simple_text": td, "qa_text": tr, "simple_class": cd_, "qa_class": cr_})
    if not smoke and not (cd_ == "correct" and cr_ == "bug" and gd[:len(pre) + 1] == pre + [yi]
                          and gr[:len(pre) + 1] == pre + [zi]):
        row["skipped"] = "unbatched eligibility"
        return row
    di, ri = dp + pre, rp + pre
    cdc, lgd = r18.capture_resid(M, di, layers)
    crc, lgr = r18.capture_resid(M, ri, layers)
    dc, db = float(lgd[yi] - lgd[zi]), float(lgr[yi] - lgr[zi])
    g = dc - db
    row.update({"d_clean": dc, "d_buggy": db, "gap": g})
    if not smoke and not (db < 0 < dc and g >= 0.5):
        row["skipped"] = "baseline deltas"
        return row
    rr = lambda lg: (float(lg[yi] - lg[zi]) - db) / g
    # every prompt position (except BOS) at every band layer: the competitor map (always computed)
    cmap = {p: {L: rr(M.logits(ri, [("resid", L, {p + OFFSET: cdc[L][p]})])) for L in band} for p in range(1, len(dp))}
    row["position_map_band"] = cmap
    for site, pos in (("lft", lft), ("oft", oft)):
        rs = {L: rr(M.logits(ri, [("resid", L, {pos + OFFSET: cdc[L][pos]})])) for L in layers}
        Lb = max(band, key=lambda L: rs[L])
        row[f"{site}_r"] = rs
        row[f"{site}_best_layer"] = Lb
        comp = [cmap[p][L] for p in cmap if p != pos for L in band]
        row[f"{site}_margin"] = rs[Lb] - max(comp)                                   # primary: any band layer
        row[f"{site}_margin_same_layer"] = rs[Lb] - max(cmap[p][Lb] for p in cmap if p != pos)
        best_other = max(((p, L) for p in cmap if p != pos for L in band), key=lambda t: cmap[t[0]][t[1]])
        row[f"{site}_best_competitor"] = {"pos": best_other[0], "layer": best_other[1], "token": dp[best_other[0]],
                                          "r": cmap[best_other[0]][best_other[1]]}
        rep = False
        if rs[Lb] >= R_REPAIR or smoke:
            _, text = r18.gen_ids(M, rp, 20, ("resid", Lb, {pos + OFFSET: cdc[Lb][pos]}))
            h = crc[Lb][pos + OFFSET].float()
            dn = (cdc[Lb][pos].float() - h).norm()
            ctl = []
            for _ in range(N_CTRL):
                u = torch.randn(h.shape, generator=gen).to(h.device)
                ctl.append(rr(M.logits(ri, [("resid", Lb, {pos + OFFSET: h + dn * u / u.norm()})])))
            m = sum(ctl) / len(ctl)
            row[f"{site}_gen"] = text
            row[f"{site}_gen_class"] = r18.classify_pair(text, big, small)
            row[f"{site}_ctrl_mean"] = m
            row[f"{site}_ctrl_sd"] = (sum((x - m) ** 2 for x in ctl) / (len(ctl) - 1)) ** 0.5
            rev = M.logits(di, [("resid", Lb, {pos: crc[Lb][pos + OFFSET]})])
            row[f"{site}_reverse_r"] = (float(rev[yi] - rev[zi]) - dc) / (db - dc)
            rep = (rs[Lb] >= R_REPAIR and row[f"{site}_gen_class"] == "correct" and m < R_CONTROL
                   and row[f"{site}_margin"] >= MARGIN)
        row[f"{site}_repair"] = rep
    return row


def body(a, run, path, r17, r18):
    M = r17.Model()
    tok = M.tok
    gates = run["gates"]
    gen = torch.Generator().manual_seed(SEED)
    nL = M.n_layers
    layers = [L for L in LAYERS if L < nL] if not a.smoke else list(range(1, nL))
    band = [L for L in BAND if L < nL] if not a.smoke else list(range(1, nL))

    def stop_if_failed():
        if not a.smoke and not all(v["ok"] for v in gates.values()):
            run["all_gates_ok"] = False
            run["finished"] = datetime.now().isoformat()
            dump(run, path)
            sys.exit(2)

    # ---- gate: canonical question reproduces R17 at the LFT ----
    lg0 = M.logits(tok(texts("W1", "9.8", "9.11")[1])["input_ids"] + tok(" 9.", add_special_tokens=False)["input_ids"])
    canon = analyse_question(M, r18, gen, {"wording": "W1", "X": 9, "Y": 8, "Z": 11, "order": "big_first"},
                             layers, band, a.smoke)
    ok = False
    if os.path.exists(R17_REF) and "lft_r" in canon:
        ref = json.load(open(R17_REF))
        ref_r = {c["layer"]: c["r"] for c in ref["cells"] if c["comp"] == "resid" and c["set"] == "p13"}
        ok = (all(abs(canon["lft_r"][L] - ref_r[L]) <= 0.03 for L in layers)
              and abs(canon["d_clean"] - 1.171875) <= 0.04 and abs(canon["d_buggy"] + 0.90625) <= 0.04
              and canon["lft"] == 13)
    gates["r17_reproduction"] = {"canon": {k: canon.get(k) for k in ("lft", "d_clean", "d_buggy", "lft_r")}, "ok": ok}
    gates["hooks_clean"] = {"ok": bool(torch.equal(M.logits(tok(texts("W1", "9.8", "9.11")[1])["input_ids"]
                                                            + tok(" 9.", add_special_tokens=False)["input_ids"]), lg0))}
    dump(run, path)
    print("gates:", {k: v["ok"] for k, v in gates.items()}, flush=True)
    stop_if_failed()

    # ---- screen (batched) ----
    qs = [{"wording": w, "X": X, "Y": Y, "Z": Z, "order": o} for w in WORDINGS for X in XS for Y in YS for Z in ZS
          for o in ("big_first", "small_first")]
    if a.smoke:
        qs = qs[:24]
    prompts_s, prompts_q = [], []
    for q in qs:
        big, small = f"{q['X']}.{q['Y']}", f"{q['X']}.{q['Z']}"
        n1, n2 = (big, small) if q["order"] == "big_first" else (small, big)
        s_txt, q_txt = texts(q["wording"], n1, n2)
        prompts_s.append(s_txt)
        prompts_q.append(q_txt)
    t0 = time.time()
    ts, _ = batch_generate(M, prompts_s)
    tq, _ = batch_generate(M, prompts_q)
    for q, a_, b_ in zip(qs, ts, tq):
        big, small = f"{q['X']}.{q['Y']}", f"{q['X']}.{q['Z']}"
        q["simple_class"], q["qa_class"] = r18.classify_pair(a_, big, small), r18.classify_pair(b_, big, small)
    screen_s = time.time() - t0
    # gate: batched vs unbatched agreement on a seeded sample of 64 questions (both formats)
    rng = random.Random(SEED)
    samp = rng.sample(range(len(qs)), min(64, len(qs)))
    agree = 0
    for i in samp:
        q = qs[i]
        big, small = f"{q['X']}.{q['Y']}", f"{q['X']}.{q['Z']}"
        n1, n2 = (big, small) if q["order"] == "big_first" else (small, big)
        s_txt, q_txt = texts(q["wording"], n1, n2)
        c1 = r18.classify_pair(r18.gen_ids(M, tok(s_txt)["input_ids"], 20)[1], big, small)
        c2 = r18.classify_pair(r18.gen_ids(M, tok(q_txt)["input_ids"], 20)[1], big, small)
        agree += (c1 == q["simple_class"]) + (c2 == q["qa_class"])
    frac = agree / (2 * len(samp))
    run["batch_agreement"] = {"agree": frac, "n": 2 * len(samp)}          # report-only (pre-registered)
    fd = [q for q in qs if q["simple_class"] == "correct" and q["qa_class"] == "bug"]
    run["screen_summary"] = {
        "n_questions": len(qs), "seconds": screen_s,
        "counts": {f: {c: sum(q[f] == c for q in qs) for c in ("correct", "bug", "neither")}
                   for f in ("simple_class", "qa_class")},
        "format_dependent": len(fd),
        "format_dependent_by_order": {o: sum(q["order"] == o for q in fd) for o in ("big_first", "small_first")},
        "format_dependent_by_wording": {w: sum(q["wording"] == w for q in fd) for w in WORDINGS},
        "reverse_dependent": sum(q["simple_class"] == "bug" and q["qa_class"] == "correct" for q in qs)}
    run["screen"] = [{k: q[k] for k in ("wording", "X", "Y", "Z", "order", "simple_class", "qa_class")} for q in qs]
    dump(run, path)
    print("screen:", run["screen_summary"], run["batch_agreement"], flush=True)
    stop_if_failed()

    # ---- site analysis on seeded samples per order ----
    if a.smoke:
        fd = qs[:4]
    rows = []
    for o in ("big_first", "small_first"):
        pool = [q for q in fd if q["order"] == o]
        random.Random(SEED).shuffle(pool)
        for q in pool[:MAX_PER_ORDER]:
            rows.append(analyse_question(M, r18, gen, q, layers, band, a.smoke))
            if len(rows) % 20 == 0:
                run["rows"] = rows
                dump(run, path)
    run["rows"] = rows
    for r in rows:
        r["in_r18_grid"] = in_r18_grid(r)
    strata = {o: [r for r in rows if r["order"] == o and "skipped" not in r and not r["in_r18_grid"]]
              for o in ("big_first", "small_first")}
    run["decision"] = decide(strata)
    ov = {o: [r for r in rows if r["order"] == o and "skipped" not in r and r["in_r18_grid"]] for o in strata}
    run["r18_grid_overlap"] = {o: {"n": len(v), "lft_repairs": sum(r["lft_repair"] for r in v),
                                   "oft_repairs": sum(r["oft_repair"] for r in v)} for o, v in ov.items()}
    run["unbatched_recheck"] = {o: {"n": sum(r["order"] == o for r in rows),
                                    "failed_eligibility": sum(r["order"] == o and r.get("skipped") == "unbatched eligibility"
                                                              for r in rows)} for o in strata}
    run["decision"]["n_skipped"] = {o: sum(r["order"] == o and "skipped" in r for r in rows) for o in strata}
    run["by_wording"] = {w: {o: {"n": len([r for r in strata[o] if r["wording"] == w]),
                                 "lft_repair": sum(r["lft_repair"] for r in strata[o] if r["wording"] == w),
                                 "oft_repair": sum(r["oft_repair"] for r in strata[o] if r["wording"] == w)}
                             for o in strata} for w in WORDINGS}
    run["canonical"] = canon
    gates["hooks_removed"] = {"ok": bool(torch.equal(M.logits(tok(texts("W1", "9.8", "9.11")[1])["input_ids"]
                                                              + tok(" 9.", add_special_tokens=False)["input_ids"]), lg0))}
    run["all_gates_ok"] = all(v["ok"] for v in gates.values())
    run["finished"] = datetime.now().isoformat()
    dump(run, path)
    print("decision:", json.dumps(run["decision"]), flush=True)
    print("by wording:", json.dumps(run["by_wording"]), flush=True)
    print("all gates ok:", run["all_gates_ok"], flush=True)
    if not a.smoke and not run["all_gates_ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
