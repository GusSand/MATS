#!/usr/bin/env python3
"""
Experiment R18: what does the "11" token carry, what puts it there, and does the site generalize?
================================================================================================
Experiment file: docs/experiments/11-23_llama8b_eleven_token.md

Follows R17 (docs/experiments/11-23_llama8b_localization_map.md): on the canonical pair, the residual stream at the
"11" token (donor position 13 / recipient 15) at layers 4-14 carries the 9.8-vs-9.11 decision. Imports R17's script
by path (sha256 checked) for its Model class (hooks, capture, teacher-forced logits, prefill-only generation).

Parts (all teacher-forced metric: delta = logit(Y) - logit(Z) after the prefix " X."; r as in R17):
  E  upstream factorial on the canonical pair: prefix {none, "Q:"} x first token {"Which", " Which"} x ending
     {"\\nAnswer:", "\\nA:"}; generation class and delta for each of the 8 prompts.
  C  content vs salience: "11"-token residuals from sentences where 9.11 is a decimal / Bible verse / date / version,
     patched into the recipient's "11" (layers 4..14); and into the donor's "11" (reverse).
  D  direction tests at the "11" token: interpolation along d = donor - recipient, 20 random same-norm directions,
     an in-manifold null (difference of two decimal-context "11" residuals, rescaled), norm ratio, logit(8) and
     logit(11) separately, cosine of d with the (verse/date/version - decimal) direction.
  A  screen on Transluce's grid: X 1..20, Y 6..9, Z 10..13, both orders; simple, Q&A, and Q&A with "Which".
  B  on format-dependent pairs from A: resid patch at the multi-digit fractional token (Z), layers 4..14, vs the
     single-digit fractional token (Y) as a comparison site; generation check and random controls at the best layer.
Exit code 2 on any gate failure or exception.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R17_PATH = os.path.join(HERE, "..", "exp_R17_localization_map", "run_experiment.py")
R17_SHA256 = "6bf85484302aafd9"          # prefix of the full sha256, checked below
R17_FULL = None                          # filled at load time for provenance
R13_PATH = os.path.join(HERE, "..", "exp_R13_true_head_patching", "run_experiment.py")

LAYERS = list(range(4, 15))              # R17's repair band
C_LAYERS = [4, 6, 8, 10, 12, 14]
OFFSET = 2
R_REPAIR, R_CONTROL = 0.5, 0.2
N_RANDOM = 20
SEED = 18
MAX_B_PAIRS = 100
MIN_B_PAIRS = 10
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TARGET_POS = 15                          # recipient "11" position; C's length-matched contexts aim here
FILLER_SENTS = ["Birds fly.", "Water is wet.", "The sky is blue.", "Cats sleep.", "It rained.", "Snow is cold today.",
                "We like long walks in the park.", "The door is open.", "Yes.", "Okay.", "Hello there."]
NEUTRAL_PREFIX = [198, 198]              # two "\n" tokens: a content-free 2-token stand-in for "Q:"
R17_BASELINE = (1.171875, -0.90625)      # R17 gates.baseline d_clean, d_buggy
R17_BASE_TOL = 0.04                      # about 2 fp16 ulps at logits of 15-25

CONTEXTS = {
    "decimal": ["The price is $9.11", "It weighs 9.11", "Let x = 9.11", "The temperature was 9.11",
                "The measured value was 9.11"],
    "verse": ["Read the Bible verse John 9.11", "As written in Genesis 9.11", "Mark 9.11", "Revelation 9.11"],
    "date": ["The attacks happened on 9.11", "We remember 9.11", "Never forget the attacks of 9.11",
             "On the morning of 9.11"],
    "version": ["Please update to version 9.11", "Python 9.11", "iOS 9.11", "Firefox 9.11"],
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_r17():
    global R17_FULL
    R17_FULL = sha256(R17_PATH)
    if not R17_FULL.startswith(R17_SHA256):
        sys.exit(f"R17 script sha256 {R17_FULL} does not start with {R17_SHA256}")
    os.environ.setdefault("R17_MODEL", os.environ.get("R18_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    os.environ.setdefault("R13_MODEL", os.environ["R17_MODEL"])
    spec = importlib.util.spec_from_file_location("r17", R17_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- prompts and pair classifier ------------------------------------------------------------------------
def question(a, b, prefix, which, ending):
    """Natural-text prompt. which in {"Which", " Which"}: " Which" without "Q:" means a leading space."""
    return f"{'Q:' if prefix else ''}{which} is bigger: {a} or {b}?\n{ending}"


def build_ids(tok, a, b, prefix, which_id, ending):
    """Token-level prompt with an explicit first-word token (BOS included). prefix: False, True ("Q:"), or a list of
    token ids used verbatim (the neutral 2-token control)."""
    rest = tok(f" is bigger: {a} or {b}?\n{ending}", add_special_tokens=False)["input_ids"]
    if isinstance(prefix, list):
        q = list(prefix)
    else:
        q = tok("Q:", add_special_tokens=False)["input_ids"] if prefix else []
    return [tok.bos_token_id] + q + [which_id] + rest


def pat(num, verb):
    return re.compile(r"(?<![\d.])" + re.escape(num) + r" is (the )?" + verb)


def classify_pair(text, big, small, with_flag=False):
    """correct / bug / neither for 'which is bigger', generalizing R13's classifier to any pair.
    Fallback (no pattern matched): a first line naming only one of the two numbers decides, in either direction."""
    lo = text.lower()
    verbs = ["bigger", "larger", "greater"]
    c = any(pat(big, v).search(lo) for v in verbs)
    b = any(pat(small, v).search(lo) for v in verbs)
    fallback = False
    if not c and not b:
        first = text.strip().split("\n")[0] if text.strip() else ""
        has_big = bool(re.search(r"(?<![\d.])" + re.escape(big) + r"(?![\d])", first))
        has_small = bool(re.search(r"(?<![\d.])" + re.escape(small) + r"(?![\d])", first))
        listing = any(re.match(r"^\s*[B-E][:.)]\s", ln) for ln in text.strip().split("\n")[1:])
        if not listing:                  # multiple-choice listings (R14's failure mode) never use the fallback
            c, b = has_big and not has_small, has_small and not has_big
            fallback = c or b
    cls = "correct" if c and not b else ("bug" if b and not c else "neither")
    return (cls, fallback) if with_flag else cls


def classify_selftest():
    assert classify_pair(" 9.8 is bigger than 9.11.\nExplanation", "9.8", "9.11") == "correct"
    assert classify_pair(" 9.11 is bigger than 9.8.\nQ: Which", "9.8", "9.11") == "bug"
    assert classify_pair(" 19.8 is bigger", "9.8", "9.11") == "neither"
    assert classify_pair(" 15.7 is larger than 15.12", "15.7", "15.12") == "correct"
    assert classify_pair(" 15.12\nB: 15.7", "15.7", "15.12") == "neither"
    assert classify_pair(" 9.8\n", "9.8", "9.11") == "correct"
    assert classify_pair(" 9.11 is the bigger number", "9.8", "9.11") == "bug"
    assert classify_pair(" 1.10 is bigger than 1.6", "1.6", "1.10") == "bug"
    assert classify_pair(" 9.11\n", "9.8", "9.11", with_flag=True) == ("bug", True)
    assert classify_pair(" 11.6 is bigger", "1.6", "1.10") == "neither"
    assert classify_pair(" 9.8\nQ: Which is bigger: 9.7 or 9.12?\nA: 9.7", "9.8", "9.11") == "correct"
    print("classify selftest: OK", flush=True)


# ---------- decision rules -----------------------------------------------------------------------------------------
def perm_p(dec_vals, other_vals):
    """Exact one-sided permutation p for mean(decimal) - mean(others) over all splits of the pooled sentences."""
    from itertools import combinations
    pool = dec_vals + other_vals
    k, obs = len(dec_vals), sum(dec_vals) / len(dec_vals) - sum(other_vals) / len(other_vals)
    tot = sum(pool)
    n_ge = n = 0
    for idx in combinations(range(len(pool)), k):
        sd = sum(pool[i] for i in idx)
        stat = sd / k - (tot - sd) / (len(pool) - k)
        n += 1
        n_ge += stat >= obs - 1e-12
    return n_ge / n


def decide_C(sent_r):
    """sent_r: {category: [per-sentence mean forward r over the C layers]} (length-matched contexts).
    First match:
      content   decimal mean >= 0.5, every other category mean <= 0.2, gap >= 0.3, permutation p < 0.05
      reading-independent  every category mean >= 0.3 and (max - min) / mean(all) < 0.25
      context-bound        every category mean < 0.2
      mixed"""
    cm = {k: sum(v) / len(v) for k, v in sent_r.items()}
    others = [cm[k] for k in ("verse", "date", "version")]
    dec = cm["decimal"]
    p = perm_p(sent_r["decimal"], [x for k in ("verse", "date", "version") for x in sent_r[k]])
    allm = sum(cm.values()) / len(cm)
    if dec >= 0.5 and max(others) <= 0.2 and dec - max(others) >= 0.3 and p < 0.05:
        lab = "content: decimal-context 11 repairs, verse/date/version 11 do not"
    elif min(cm.values()) >= 0.3 and allm > 0 and (max(cm.values()) - min(cm.values())) / allm < 0.25:
        lab = "reading-independent: every foreign 11 repairs about equally"
    elif max(cm.values()) < 0.2:
        lab = "context-bound: no foreign 11 repairs"
    else:
        lab = "mixed"
    return {"category_mean": cm, "perm_p_decimal_vs_others": p, "label": lab}


def decide_transluce(rev_cat, cos_concept, cos_base_max):
    """cos_base_max: the 95th percentile of |cos| under the label-permutation null (primary baseline)."""
    """Pre-registered Transluce reading: verse/date/version 11s induce the bug in the donor (reverse), decimal 11s
    do not; and d (donor - recipient) points away from the verse/date/version direction beyond the baseline."""
    nondec = [rev_cat[k] for k in ("verse", "date", "version")]
    rev_ok = sum(nondec) / 3 >= 0.5 and rev_cat["decimal"] <= 0.2
    cos_ok = cos_concept is not None and cos_concept < 0 and abs(cos_concept) > cos_base_max
    return {"reverse_supports": rev_ok, "cosine_supports": cos_ok,
            "label": "supports the Transluce reading" if rev_ok and cos_ok else
                     ("partly supports (reverse only)" if rev_ok else
                      ("partly supports (cosine only)" if cos_ok else "does not support the Transluce reading"))}


def decide_D(per_layer):
    """Layer specific if r_real exceeds all 20 random r, r_real - random mean >= 0.3, r_real exceeds every
    in-manifold draw and r_real - in-manifold mean >= 0.3. Label 'specific direction' if >= 6 of 11 layers."""
    ok = [p for p in per_layer if p["r_real"] > p["rand_max"] and p["r_real"] - p["rand_mean"] >= 0.3
          and p["r_real"] > p["inman_max"] and p["r_real"] - p["inman_mean"] >= 0.3]
    sec = [p for p in per_layer if p["r_real"] > p["rand_mean"] + 3 * p["rand_sd"]]
    gen_ok = [p for p in per_layer if p["rand_mean"] >= p["r_real"] - 0.2]
    if len(ok) >= 6:
        lab = "specific direction"
    elif len(gen_ok) >= 6:
        lab = "generic: random directions do about as well"
    else:
        lab = "inconclusive"
    return {"n_layers_specific": len(ok), "n_layers_generic": len(gen_ok), "n_layers_3sd_secondary": len(sec),
            "label": lab}


def decide_combined(c_label, d_label):
    if c_label.startswith("reading-independent") and d_label.startswith("generic"):
        return "salience: foreign 11s and random directions both repair"
    if d_label.startswith("specific"):
        return "a specific direction carries the difference"
    return "no combined claim"


def decide_B(rows):
    """rows: per eligible question, z_repair (bool: repairs AND beats every other position by >= 0.3), y_repair."""
    n = len(rows)
    if n < MIN_B_PAIRS:
        return {"n": n, "label": "descriptive only (fewer than 10 eligible pairs)"}
    z = sum(r["z_repair"] for r in rows) / n
    y = sum(r["y_repair"] for r in rows) / n
    if z >= 0.5 and z - y >= 0.3:
        lab = "site generalizes: the multi-digit fractional token carries the decision"
    elif z < 0.2:
        lab = "site does not generalize"
    else:
        lab = "partial"
    return {"n": n, "z_rate": z, "y_rate": y, "label": lab}


def decide_selftest():
    c = decide_C({"decimal": [0.7, 0.8, 0.6, 0.75, 0.7], "verse": [0.1, 0.0, 0.1, 0.05], "date": [0.0, 0.1, -0.1, 0.0],
                  "version": [0.2, 0.1, 0.15, 0.1]})
    assert c["label"].startswith("content") and c["perm_p_decimal_vs_others"] < 0.05
    assert decide_C({"decimal": [1.6, 1.5], "verse": [1.5, 1.45], "date": [1.4, 1.5], "version": [1.55, 1.6]}
                    )["label"].startswith("reading-independent")
    assert decide_C({"decimal": [0.1, 0.0], "verse": [0.0, 0.1], "date": [-0.1, 0.0], "version": [0.1, 0.15]}
                    )["label"].startswith("context-bound")
    assert decide_C({"decimal": [0.6, 0.6], "verse": [0.3, 0.3], "date": [0.1, 0.1], "version": [0.2, 0.2]}
                    )["label"] == "mixed"
    # content needs the permutation test too: 2 decimal of 5 pooled -> best possible p = 1/10
    assert not decide_C({"decimal": [0.7, 0.6], "verse": [0.1], "date": [0.0], "version": [0.1]}
                        )["label"].startswith("content")
    t = decide_transluce({"decimal": 0.1, "verse": 0.6, "date": 0.7, "version": 0.5}, -0.3, 0.1)
    assert t["label"] == "supports the Transluce reading"
    assert decide_transluce({"decimal": 0.1, "verse": 0.6, "date": 0.7, "version": 0.5}, 0.3, 0.1)["label"].startswith("partly")
    assert decide_transluce({"decimal": 0.5, "verse": 0.6, "date": 0.7, "version": 0.5}, -0.05, 0.1)["label"].startswith("does not")
    mk = lambda r, m, mx, s, im, imx: {"r_real": r, "rand_mean": m, "rand_max": mx, "rand_sd": s,
                                       "inman_mean": im, "inman_max": imx}
    assert decide_D([mk(0.8, 0.0, 0.6, 0.3, 0.1, 0.5)] * 6 + [mk(0.3, 0.0, 0.6, 0.3, 0.1, 0.5)] * 5)["label"] == "specific direction"
    assert decide_D([mk(0.8, 0.0, 0.9, 0.3, 0.1, 0.5)] * 11)["label"] == "inconclusive"     # a random draw beats it
    assert decide_D([mk(0.8, 0.0, 0.6, 0.3, 0.6, 0.7)] * 11)["label"] == "inconclusive"     # in-manifold too close
    assert decide_D([mk(0.7, 0.6, 0.9, 0.3, 0.6, 0.7)] * 11)["label"].startswith("generic")
    assert decide_combined("reading-independent: x", "generic: y").startswith("salience")
    assert decide_combined("mixed", "generic: y") == "no combined claim"
    assert decide_combined("mixed", "specific direction").startswith("a specific")
    rows = [{"z_repair": True, "y_repair": False}] * 7 + [{"z_repair": False, "y_repair": False}] * 3
    assert decide_B(rows)["label"].startswith("site generalizes")
    assert decide_B(rows[:5])["label"].startswith("descriptive")
    rows = [{"z_repair": True, "y_repair": True}] * 7 + [{"z_repair": False, "y_repair": False}] * 3
    assert decide_B(rows)["label"] == "partial"
    assert decide_B([{"z_repair": False, "y_repair": False}] * 12)["label"] == "site does not generalize"
    print("decide selftest: OK", flush=True)


# ---------- helpers ------------------------------------------------------------------------------------------------
def gen_ids(M, ids, max_new, patch=None):
    """Greedy generation from explicit token ids; optional prefill-only patch (comp, L, {pos: tensor})."""
    x = torch.tensor([ids], device=M.model.device)
    T = len(ids)
    calls, handle = [], None
    if patch is not None:
        comp, L, posmap = patch

        def fn(h):
            calls.append(int(h.shape[1]))
            if len(calls) > 1:
                return None
            assert h.shape[1] == T
            h = h.clone()
            for p, v in posmap.items():
                h[0, p] = v.to(h.dtype)
            return h
        handle = M._register(comp, L, fn)
    try:
        with torch.no_grad():
            out = M.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=M.tok.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    if patch is not None:
        assert len(calls) >= 2 and all(c == 1 for c in calls[1:])
    new = out[0][T:].tolist()
    return new, M.tok.decode(new, skip_special_tokens=True)


def capture_resid(M, ids, layers):
    """resid (input of decoder layer L) for the given layers, all positions; plus last logits."""
    store = {}
    hs = []
    for L in layers:
        def fn(h, L=L):
            store[L] = h[0].detach().clone()
            return None
        hs.append(M._register("resid", L, fn))
    try:
        with torch.no_grad():
            lg = M.model(torch.tensor([ids], device=M.model.device)).logits[0, -1].float()
    finally:
        for h in hs:
            h.remove()
    return store, lg


def find_frac(ids, tok_id, after_dot=True):
    """Position of tok_id that directly follows a '.' (id 13). Exactly one occurrence required."""
    hits = [i for i in range(1, len(ids)) if ids[i] == tok_id and (not after_dot or ids[i - 1] == 13)]
    assert len(hits) == 1, f"token {tok_id} after '.' found at {hits}"
    return hits[0]


def provenance(model_name):
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
            "gpu": gpu, "slurm_job": os.environ.get("SLURM_JOB_ID"), "git_sha": os.environ.get("R18_GIT_SHA"),
            "script_sha256": sha256(os.path.abspath(__file__)), "r17_sha256": R17_FULL,
            "r13_sha256": sha256(R13_PATH), "model": model_name}


# ---------- main ---------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny model on CPU: small grid, gates recorded not enforced")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    classify_selftest()
    decide_selftest()
    if a.selftest:
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "run.json")
    r17 = load_r17()
    run = {"provenance": provenance(r17.MODEL_NAME), "started": datetime.now().isoformat(),
           "command": " ".join(sys.argv), "gates": {}}
    dump(run, path)
    try:
        body(a, run, path, r17)
    except SystemExit:
        raise
    except Exception:
        run["error"] = traceback.format_exc()
        run["all_gates_ok"] = False
        run["finished"] = datetime.now().isoformat()
        dump(run, path)
        print(run["error"], flush=True)
        sys.exit(2)


def body(a, run, path, r17):
    M = r17.Model()
    tok = M.tok
    gates = run["gates"]
    gen = torch.Generator().manual_seed(SEED)
    nL = M.n_layers
    layers = [L for L in LAYERS if L < nL] if not a.smoke else list(range(1, nL))
    c_layers = [L for L in C_LAYERS if L < nL] if not a.smoke else list(range(1, nL))

    def stop_if_failed():
        if not a.smoke and not all(v["ok"] for v in gates.values()):
            run["all_gates_ok"] = False
            run["finished"] = datetime.now().isoformat()
            dump(run, path)
            sys.exit(2)

    WHICH, SWHICH = tok("Which", add_special_tokens=False)["input_ids"], tok(" Which", add_special_tokens=False)["input_ids"]
    assert len(WHICH) == 1 and len(SWHICH) == 1
    WHICH, SWHICH = WHICH[0], SWHICH[0]

    # ---------------- canonical pair: gates and R17 reproduction ----------------
    big, small = "9.8", "9.11"
    pre_ids = tok(" 9.", add_special_tokens=False)["input_ids"]
    y_id, z_id = tok("8", add_special_tokens=False)["input_ids"][0], tok("11", add_special_tokens=False)["input_ids"][0]
    d_prompt = build_ids(tok, big, small, False, WHICH, "Answer:")
    r_prompt = build_ids(tok, big, small, True, SWHICH, "A:")
    nat_d = tok(question(big, small, False, "Which", "Answer:"))["input_ids"]
    nat_r = tok(question(big, small, True, " Which", "A:"))["input_ids"]
    gates["token_build"] = {"ok": d_prompt == nat_d and r_prompt == nat_r}
    d_ids, r_ids = d_prompt + pre_ids, r_prompt + pre_ids
    zpos_d = find_frac(d_ids, z_id)
    zpos_r = zpos_d + OFFSET
    gates["positions"] = {"zpos_donor": zpos_d, "ok": r_ids[zpos_r] == z_id}

    def dlt(lg, yi=y_id, zi=z_id):
        return float(lg[yi] - lg[zi])
    cap_d, lg_d = capture_resid(M, d_ids, range(nL))
    cap_r, lg_r = capture_resid(M, r_ids, range(nL))
    d_c, d_b = dlt(lg_d), dlt(lg_r)
    gap = d_c - d_b
    rr = lambda lg: (dlt(lg) - d_b) / gap
    gates["baseline"] = {"d_clean": d_c, "d_buggy": d_b, "ok": d_b < 0 < d_c and gap >= 1.0}
    r17_ref = None
    ref_path = os.path.join(HERE, "..", "exp_R17_localization_map", "results", "r17_24213609", "run.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        r17_ref = {c["layer"]: c["r"] for c in ref["cells"] if c["comp"] == "resid" and c["set"] == "p13"}
        run["r17_versions"] = {k: ref["provenance"].get(k) for k in ("torch", "transformers", "gpu")}
    gates["r17_baseline_equal"] = {"d_clean": d_c, "d_buggy": d_b, "r17": R17_BASELINE,
                                   "ok": abs(d_c - R17_BASELINE[0]) <= R17_BASE_TOL and abs(d_b - R17_BASELINE[1]) <= R17_BASE_TOL,
                                   "bitwise_equal": d_c == R17_BASELINE[0] and d_b == R17_BASELINE[1]}
    repro = {}
    for L in layers:
        lg = M.logits(r_ids, [("resid", L, {zpos_r: cap_d[L][zpos_d]})])
        repro[L] = rr(lg)
    gates["r17_reproduction"] = {"r": repro, "ref": r17_ref,
                                 "ok": r17_ref is not None and all(abs(repro[L] - r17_ref[L]) < 0.02 for L in layers)}
    gates["hooks_clean"] = {"ok": bool(torch.equal(M.logits(r_ids), lg_r))}
    dump(run, path)
    print("gates:", {k: v["ok"] for k, v in gates.items()}, flush=True)
    stop_if_failed()

    # ---------------- E: upstream factorial, with the effect at the "11" site ----------------
    def site_r(ids):
        """Write this prompt's '11' residual (layers 4..14) into the recipient's '11'; r per layer and mean."""
        zp = find_frac(ids, z_id)
        cap, _ = capture_resid(M, ids, layers)
        rs = {L: rr(M.logits(r_ids, [("resid", L, {zpos_r: cap[L][zp]})])) for L in layers}
        return rs, sum(rs.values()) / len(rs)
    E = []
    variants = [(pf, w, e) for pf in (False, True) for w in ("Which", " Which") for e in ("Answer:", "A:")]
    variants += [(NEUTRAL_PREFIX, w, "A:") for w in ("Which", " Which")]
    for prefix, wname, ending in variants:
        which_id = WHICH if wname == "Which" else SWHICH
        ids = build_ids(tok, big, small, prefix, which_id, ending)
        new, text = gen_ids(M, ids, 20)
        lg = M.logits(ids + pre_ids)
        rs, rmean = site_r(ids + pre_ids)
        E.append({"prefix": "neutral2" if isinstance(prefix, list) else ("Q:" if prefix else "none"), "which": wname,
                  "ending": ending, "text": text, "class": classify_pair(text, big, small), "delta": dlt(lg),
                  "first4": new[:4], "argmax": int(lg.argmax()), "site_r": rs, "site_r_mean": rmean})
    run["E"] = E
    fac = [e for e in E if e["prefix"] != "neutral2"]
    def cell(key, val):
        g = [e for e in fac if e[key] == val]
        return {"delta": sum(e["delta"] for e in g) / len(g), "site_r": sum(e["site_r_mean"] for e in g) / len(g)}
    def inter(k1, v1, k2, v2, f):
        m = lambda a1, a2: [e[f] if f == "delta" else e["site_r_mean"] for e in fac if e[k1] == a1 and e[k2] == a2]
        avg = lambda xs: sum(xs) / len(xs)
        (x1, x0), (y1, y0) = v1, v2
        return (avg(m(x1, y1)) - avg(m(x1, y0))) - (avg(m(x0, y1)) - avg(m(x0, y0)))
    run["E_effects"] = {
        "main": {"Q_prefix": {f: cell("prefix", "Q:")[f] - cell("prefix", "none")[f] for f in ("delta", "site_r")},
                 "space_Which": {f: cell("which", " Which")[f] - cell("which", "Which")[f] for f in ("delta", "site_r")},
                 "ending_A": {f: cell("ending", "A:")[f] - cell("ending", "Answer:")[f] for f in ("delta", "site_r")}},
        "interactions_delta": {
            "Q x space": inter("prefix", ("Q:", "none"), "which", (" Which", "Which"), "delta"),
            "Q x ending": inter("prefix", ("Q:", "none"), "ending", ("A:", "Answer:"), "delta"),
            "space x ending": inter("which", (" Which", "Which"), "ending", ("A:", "Answer:"), "delta")}}
    ev = {(e["prefix"], e["which"], e["ending"]): e for e in E}
    run["E_effects"]["neutral_prefix_contrasts"] = {
        w: {"neutral2 - none": {f: ev[("neutral2", w, "A:")][f] - ev[("none", w, "A:")][f] for f in ("delta", "site_r_mean")},
            "Q: - neutral2": {f: ev[("Q:", w, "A:")][f] - ev[("neutral2", w, "A:")][f] for f in ("delta", "site_r_mean")}}
        for w in ("Which", " Which")}
    ecls = {(e["prefix"], e["which"], e["ending"]): e["class"] for e in E}
    gates["E_canonical_classes"] = {"ok": ecls[("none", "Which", "Answer:")] == "correct"
                                    and ecls[("Q:", " Which", "A:")] == "bug"}
    dump(run, path)
    print("E:", [(e["prefix"], e["which"], e["ending"], e["class"], round(e["delta"], 2), round(e["site_r_mean"], 2))
                 for e in E], flush=True)

    # ---------------- C: content vs reading-independence (length-matched primary, original secondary) ----------
    tail3 = tok(" 9.11", add_special_tokens=False)["input_ids"][-3:]
    from itertools import permutations
    def ctx_ids(sent, matched):
        """Original: the sentence alone. Matched: complete neutral filler sentences before it, chosen so that the
        natural tokenization puts its '11' exactly at the recipient's '11' position."""
        if not matched:
            return tok(sent)["input_ids"]
        for k in range(1, 5):
            for combo in permutations(FILLER_SENTS, k):
                ids = tok(" ".join(combo) + " " + sent)["input_ids"]
                if len(ids) - 1 == zpos_r:
                    return ids
        return tok(sent)["input_ids"]                   # no exact match: caught by the position check below
    ctx_caps = {"matched": {}, "original": {}}
    C_rows = []
    for version in ("matched", "original"):
        for cat, sents in CONTEXTS.items():
            for s in sents:
                ids = ctx_ids(s, version == "matched")
                row = {"version": version, "cat": cat, "sent": s, "text": tok.decode(ids[1:])}
                if ids[-3:] != tail3 or (version == "matched" and len(ids) - 1 != zpos_r):
                    row["skipped"] = "tokenization or position"
                    C_rows.append(row)
                    continue
                cap, _ = capture_resid(M, ids, range(nL))
                ctx_caps[version].setdefault(cat, []).append((s, cap))
                row.update({"pos11": len(ids) - 1, "fwd": {}, "rev": {}, "dlogit8": {}, "dlogit11": {}, "norm_ratio": {}})
                for L in c_layers:
                    v = cap[L][len(ids) - 1]
                    lg = M.logits(r_ids, [("resid", L, {zpos_r: v})])
                    row["fwd"][L] = rr(lg)
                    row["dlogit8"][L] = float(lg[y_id] - lg_r[y_id])
                    row["dlogit11"][L] = float(lg[z_id] - lg_r[z_id])
                    row["norm_ratio"][L] = float(v.float().norm() / cap_r[L][zpos_r].float().norm())
                    lg2 = M.logits(d_ids, [("resid", L, {zpos_d: v})])
                    row["rev"][L] = (dlt(lg2) - d_c) / (d_b - d_c)
                row["fwd_mean"] = sum(row["fwd"].values()) / len(row["fwd"])
                row["rev_mean"] = sum(row["rev"].values()) / len(row["rev"])
                C_rows.append(row)
    run["C_rows"] = C_rows
    ok_counts = {v: {cat: len(ctx_caps[v].get(cat, [])) for cat in CONTEXTS} for v in ctx_caps}
    gates["C_contexts"] = {"counts": ok_counts, "ok": all(n >= 2 for v in ok_counts.values() for n in v.values())}
    stop_if_failed()
    C_sum = {}
    for version in ("matched", "original"):
        rows = [r for r in C_rows if r["version"] == version and "fwd_mean" in r]
        sent_r = {cat: [r["fwd_mean"] for r in rows if r["cat"] == cat] for cat in CONTEXTS}
        rev = {cat: sum(r["rev_mean"] for r in rows if r["cat"] == cat) / max(1, len(sent_r[cat])) for cat in CONTEXTS}
        n = len(rows)
        if n > 2:                                      # correlation of r with the "11" position (descriptive)
            xs, ys = [r["pos11"] for r in rows], [r["fwd_mean"] for r in rows]
            mx, my = sum(xs) / n, sum(ys) / n
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
            corr = sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else None
        else:
            corr = None
        C_sum[version] = {**decide_C(sent_r), "reverse_mean_r": rev, "corr_r_vs_pos11": corr}
    run["C_summary"] = C_sum                       # "matched" is the pre-registered primary
    dump(run, path)
    print("C:", {v: C_sum[v]["label"] for v in C_sum}, C_sum["matched"]["category_mean"], flush=True)

    # ---------------- D: direction tests ----------------
    D = []
    dec = ctx_caps["matched"].get("decimal", [])
    inman_pairs = [(i, j) for i in range(len(dec)) for j in range(len(dec)) if i != j]
    nondec = [c for k in ("verse", "date", "version") for _, c in ctx_caps["matched"].get(k, [])]
    for L in layers:
        h_r = cap_r[L][zpos_r].float()
        d = cap_d[L][zpos_d].float() - h_r
        row = {"layer": L, "norm_d": float(d.norm()), "norm_h": float(h_r.norm()), "norm_ratio": float(d.norm() / h_r.norm())}
        row["alpha"] = {al: rr(M.logits(r_ids, [("resid", L, {zpos_r: h_r + al * d})])) for al in ALPHAS}
        row["r_real"] = row["alpha"][1.0]
        lg = M.logits(r_ids, [("resid", L, {zpos_r: h_r + d})])
        row["dlogit8"], row["dlogit11"] = float(lg[y_id] - lg_r[y_id]), float(lg[z_id] - lg_r[z_id])
        rs = []
        for _ in range(N_RANDOM):
            u = torch.randn(h_r.shape, generator=gen).to(h_r.device)
            rs.append(rr(M.logits(r_ids, [("resid", L, {zpos_r: h_r + d.norm() * u / u.norm()})])))
        m = sum(rs) / len(rs)
        row["rand_r"] = rs
        row["rand_mean"], row["rand_max"] = m, max(rs)
        row["rand_sd"] = (sum((x - m) ** 2 for x in rs) / (len(rs) - 1)) ** 0.5
        im, cos_base = [], []
        for i, j in inman_pairs:
            e = dec[i][1][L][-1].float() - dec[j][1][L][-1].float()
            im.append(rr(M.logits(r_ids, [("resid", L, {zpos_r: h_r + d.norm() * e / e.norm()})])))
            cos_base.append(abs(float(torch.nn.functional.cosine_similarity(d, e, dim=0))))
        row["inman_pairs"] = [(dec[i][0], dec[j][0]) for i, j in inman_pairs]
        row["inman_r"] = im
        row["inman_mean"] = sum(im) / len(im) if im else float("nan")
        row["inman_max"] = max(im) if im else float("nan")
        if dec and nondec:
            decm = sum(c[L][-1].float() for _, c in dec) / len(dec)
            cd = sum(c[L][-1].float() for c in nondec) / len(nondec) - decm
            row["cos_d_concept"] = float(torch.nn.functional.cosine_similarity(d, cd, dim=0))
            row["cos_pairdiff_abs_max"] = max(cos_base) if cos_base else None      # secondary baseline
            # primary: label-permutation null over all splits of the pooled sentences (same group sizes)
            from itertools import combinations
            pool = torch.stack([c[L][-1].float() for _, c in dec] + [c[L][-1].float() for c in nondec])
            k, tot = len(dec), pool.sum(0)
            nulls = []
            for idx in combinations(range(len(pool)), k):
                sd = pool[list(idx)].sum(0)
                v = (tot - sd) / (len(pool) - k) - sd / k
                nulls.append(abs(float(torch.nn.functional.cosine_similarity(d, v, dim=0))))
            nulls.sort()
            row["cos_perm_abs_p95"] = nulls[int(0.95 * (len(nulls) - 1))]
        D.append(row)
    run["D"] = D
    run["D_summary"] = decide_D([r for r in D if r["inman_mean"] == r["inman_mean"]])
    cc = [r["cos_d_concept"] for r in D if "cos_d_concept" in r]
    cb = [r["cos_perm_abs_p95"] for r in D if r.get("cos_perm_abs_p95") is not None]
    run["transluce"] = decide_transluce(C_sum["matched"]["reverse_mean_r"],
                                        sum(cc) / len(cc) if cc else None, sum(cb) / len(cb) if cb else 1.0)
    run["transluce"]["cos_d_concept_mean"] = sum(cc) / len(cc) if cc else None
    run["transluce"]["cos_perm_abs_p95_mean"] = sum(cb) / len(cb) if cb else None
    run["combined"] = decide_combined(C_sum["matched"]["label"], run["D_summary"]["label"])
    dump(run, path)
    print("D:", run["D_summary"], "Transluce:", run["transluce"], flush=True)

    # ---------------- A: screen ----------------
    grid = [(X, Y, Z) for X in range(1, 21) for Y in range(6, 10) for Z in range(10, 14)]
    if a.smoke:
        grid = grid[:6]
    A = []
    t0 = time.time()
    for X, Y, Z in grid:
        big, small = f"{X}.{Y}", f"{X}.{Z}"
        for order in ("big_first", "small_first"):
            n1, n2 = (big, small) if order == "big_first" else (small, big)
            row = {"X": X, "Y": Y, "Z": Z, "order": order}
            for fmt, (prefix, wid, ending) in {"simple": (False, WHICH, "Answer:"), "qa": (True, SWHICH, "A:"),
                                               "qa_Which": (True, WHICH, "A:")}.items():
                ids = build_ids(tok, n1, n2, prefix, wid, ending)
                new, text = gen_ids(M, ids, 20)
                row[fmt] = {"text": text, "class": classify_pair(text, big, small), "first": new[:5]}
            A.append(row)
    run["A"] = A
    fd = [r for r in A if r["simple"]["class"] == "correct" and r["qa"]["class"] == "bug"]
    run["A_summary"] = {
        "n_questions": len(A),
        "counts": {fmt: {k: sum(r[fmt]["class"] == k for r in A) for k in ("correct", "bug", "neither")}
                   for fmt in ("simple", "qa", "qa_Which")},
        "format_dependent": len(fd),
        "format_dependent_fixed_by_Which": sum(r["qa_Which"]["class"] == "correct" for r in fd),
        "seconds": time.time() - t0}
    dump(run, path)
    print("A:", run["A_summary"], flush=True)
    if a.smoke:                  # smoke only: exercise part B on the first questions regardless of class
        fd = A[:3]

    # ---------------- B: does the site generalize? ----------------
    import random as _random
    sample = list(fd)
    _random.Random(SEED).shuffle(sample)
    B = []
    for r in sample[:MAX_B_PAIRS]:
        X, Y, Z, order = r["X"], r["Y"], r["Z"], r["order"]
        big, small = f"{X}.{Y}", f"{X}.{Z}"
        n1, n2 = (big, small) if order == "big_first" else (small, big)
        pre = tok(f" {X}.", add_special_tokens=False)["input_ids"]
        yi, zi = tok(str(Y), add_special_tokens=False)["input_ids"], tok(str(Z), add_special_tokens=False)["input_ids"]
        row = {"X": X, "Y": Y, "Z": Z, "order": order}
        if len(yi) != 1 or len(zi) != 1:
            row["skipped"] = "multi-token fractional part"
            B.append(row)
            continue
        yi, zi = yi[0], zi[0]
        dp = build_ids(tok, n1, n2, False, WHICH, "Answer:")
        rp = build_ids(tok, n1, n2, True, SWHICH, "A:")
        if not a.smoke and (r["simple"]["first"][:len(pre) + 1] != pre + [yi]
                            or r["qa"]["first"][:len(pre) + 1] != pre + [zi]):
            row["skipped"] = "answer does not start with ' X.' then the fractional token"
            B.append(row)
            continue
        di, ri = dp + pre, rp + pre
        try:
            zd, yd = find_frac(di, zi), find_frac(di, yi)
        except AssertionError as e:
            row["skipped"] = f"position lookup: {e}"
            B.append(row)
            continue
        cd, lgd = capture_resid(M, di, layers)
        cr, lgr = capture_resid(M, ri, layers)
        dc, db = float(lgd[yi] - lgd[zi]), float(lgr[yi] - lgr[zi])
        g = dc - db
        row.update({"d_clean": dc, "d_buggy": db, "gap": g})
        if not a.smoke and not (db < 0 < dc and g >= 0.5):
            row["skipped"] = "baseline deltas"
            B.append(row)
            continue
        for site, pos in (("z", zd), ("y", yd)):
            rs = {L: (float((lambda lg: lg[yi] - lg[zi])(M.logits(ri, [("resid", L, {pos + OFFSET: cd[L][pos]})]))) - db) / g
                  for L in layers}
            Lb = max(rs, key=rs.get)
            row[f"{site}_r"] = rs
            row[f"{site}_best_layer"] = Lb
            rep = False
            if rs[Lb] >= R_REPAIR or a.smoke:
                _, text = gen_ids(M, rp, 20, ("resid", Lb, {pos + OFFSET: cd[Lb][pos]}))
                h = cr[Lb][pos + OFFSET].float()
                dn = (cd[Lb][pos].float() - h).norm()
                ctl = []
                for _ in range(5):
                    u = torch.randn(h.shape, generator=gen).to(h.device)
                    lg = M.logits(ri, [("resid", Lb, {pos + OFFSET: h + dn * u / u.norm()})])
                    ctl.append((float(lg[yi] - lg[zi]) - db) / g)
                row[f"{site}_gen"] = text
                row[f"{site}_ctrl_mean"] = sum(ctl) / len(ctl)
                row[f"{site}_ctrl_sd"] = (sum((x - row[f"{site}_ctrl_mean"]) ** 2 for x in ctl) / (len(ctl) - 1)) ** 0.5
                rep = classify_pair(text, big, small) == "correct" and row[f"{site}_ctrl_mean"] < R_CONTROL
            if site == "z":
                # every other aligned prompt position at Z's best layer (capture all positions at that layer)
                cdz, _ = capture_resid(M, di, [Lb])
                others = {}
                for dpos in range(1, len(dp)):
                    if dpos == pos:
                        continue
                    lg = M.logits(ri, [("resid", Lb, {dpos + OFFSET: cdz[Lb][dpos]})])
                    others[dpos] = (float(lg[yi] - lg[zi]) - db) / g
                row["z_other_positions_r"] = others
                row["z_margin"] = rs[Lb] - max(others.values())
                rep = rep and row["z_margin"] >= 0.3
            row[f"{site}_repair"] = rep
        B.append(row)
    run["B"] = B
    run["B_summary"] = decide_B([b for b in B if "skipped" not in b])
    run["B_summary"]["n_skipped"] = sum("skipped" in b for b in B)
    gates["hooks_removed"] = {"ok": bool(torch.equal(M.logits(r_ids), lg_r))}
    run["all_gates_ok"] = all(v["ok"] for v in gates.values())
    run["finished"] = datetime.now().isoformat()
    dump(run, path)
    print("B:", run["B_summary"], flush=True)
    print("all gates ok:", run["all_gates_ok"], flush=True)
    if not a.smoke and not run["all_gates_ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
