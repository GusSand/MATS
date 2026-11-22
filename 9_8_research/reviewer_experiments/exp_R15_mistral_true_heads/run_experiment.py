#!/usr/bin/env python3
"""
Experiment R15: test the v2 draft's Mistral-7B head claims with true per-head, prompt-only patching
==================================================================================================
Experiment file: docs/experiments/11-22_mistral7b_true_head_patching.md
Design review round 1: docs/reviews/11-22_mistral7b_true_head_patching_design.md

The claims came from a collaborator by email (no code). This run tests them; it cannot reproduce the
numbers. Patching code is R14's (imported by path; its sha256 must equal R14_EXPECTED_SHA256).
All pre-registered rules are computed here (`decide`) and written to run.json.
"""

import argparse
import importlib.util
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R14_PATH = os.path.join(HERE, "..", "exp_R14_cross_model_true_heads", "run_experiment.py")
R14_EXPECTED_SHA256 = os.environ.get("R14_EXPECTED_SHA256")   # set by run.sbatch from the committed file
_spec = importlib.util.spec_from_file_location("r14", R14_PATH)
r14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r14)

REPO = "mistralai/Mistral-7B-Instruct-v0.2"
SNAPSHOT = "63a8b081895390a26e140280378bc85ec8bce07a"
MAX_NEW = 64
FIVE = [2, 10, 11, 13, 17]
PARITY_LAYERS = [2, 17]
KS = [4, 8, 12, 16]
N_SUB = 20
H22 = 22
DRAFT_TOP5 = [22, 29, 31, 10, 21]

# R14's 18 selftest cases (copied; R14 keeps them inside selftest())
R14_CASES = [(" 9.8 is bigger than 9.11.", "correct"), (" 9.11 is bigger than 9.8.", "bug"),
             ("\n\nI think the answer is:\n\n9.11\n\nA:\n\nThe answer is:", "bug"),
             ("I think the answer is:\n\n9.8", "correct"), ("I think the answer is 9.11.\n\nA:", "bug"),
             ("\n9.11\nIs -1/2 at most -1.1", "bug"), ("The answer is:\n\nThe answer is:", "neither"),
             ("9.8\n\nQ: Which is bigger: 9.8 or 9.1", "correct"), ("Q: Which is bigger: 9.8 or 9.11?\n9.11", "neither"),
             ("**9.11** is bigger than 9.8", "bug"), ("**9.8** is bigger than **9.11**.", "correct"),
             ("Comparing 9.8 and 9.11:\n9.8 wins", "neither"), ("\n-0.8\nWhat is 7.80?", "neither"),
             ("9.80 is bigger than 9.11", "correct"), ("The answer is 9.80.", "correct"),
             (" 9.8\n\nQ: Which is bigger: 9.11 or 9.12?\nA: 9.11 is bigger", "correct"),
             ("19.8 is bigger than 9.11", "bug"), ("29.11 is bigger than 3", "neither")]
RE_LESS = re.compile(r"\bis\s+(?:less|smaller|lower)\b", re.I)


# ---------------------------------------------------------------- pair-generic classifier (R14 rules)

def make_classifier(big, small):
    """R14's classify, with 9.8 -> big and 9.11 -> small. big may carry trailing zeros (4.3 -> 4.30)."""
    nb = r"(?<![\d.])" + re.escape(big) + r"0*(?!\d)"
    ns = r"(?<![\d.])" + re.escape(small) + r"(?!\d)"
    cmp_ = r"\s+is\s+(?:the\s+)?(?:bigger|larger|greater)"
    rc, rb, r_big, r_small = re.compile(nb + cmp_), re.compile(ns + cmp_), re.compile(nb), re.compile(ns)

    def answer_line(text):
        for line in r14._clean(text).split("\n"):
            s = line.strip()
            if not s:
                continue
            a, b = bool(r_big.search(s)), bool(r_small.search(s))
            if a and b:
                return None
            if a:
                return "big"
            if b:
                return "small"
        return None

    def is_correct(text):
        lo = r14._clean(text).lower()
        c, b = bool(rc.search(lo)), bool(rb.search(lo))
        if c or b:
            return c and not b
        return answer_line(text) == "big"

    def is_bug(text):
        if is_correct(text):
            return False
        lo = r14._clean(text).lower()
        if rb.search(lo):
            return True
        if rc.search(lo):
            return False
        return answer_line(text) == "small"

    return lambda t: "correct" if is_correct(t) else ("bug" if is_bug(t) else "neither")


CLS = make_classifier("9.8", "9.11")


def wrap_chat(tok, content):
    chat = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    return chat


def formats(tok, big, small):
    q = f"Which is bigger: {big} or {small}?"
    f = {"chat": wrap_chat(tok, q), "comparative": f"Compare {big} and {small}. Which is larger?",
         "qa": f"Q: {q}\nA:", "simple": f"{q}\nAnswer:"}
    return f


def chat_wrapped_wordings(tok):
    """Descriptive: the other wordings inside the chat template (separates template from wording)."""
    return {"chat_comparative": wrap_chat(tok, "Compare 9.8 and 9.11. Which is larger?"),
            "chat_qa": wrap_chat(tok, "Q: Which is bigger: 9.8 or 9.11?\nA:"),
            "chat_simple": wrap_chat(tok, "Which is bigger: 9.8 or 9.11?\nAnswer:")}


def gen_pairs(n=100, seed=0):
    """X.D vs X.DD with the shorter one larger: DD < 10*D, DD not ending in 0. Distinct, seeded."""
    rng = np.random.RandomState(seed)
    out = set()
    while len(out) < n:
        x, d = int(rng.randint(1, 10)), int(rng.randint(2, 10))
        dd = int(rng.randint(10, 10 * d))
        if dd % 10 == 0 or ((x, d) == (9, 8) and dd == 11):
            continue
        out.add((f"{x}.{d}", f"{x}.{dd}"))
    return sorted(out)


def parity_subsets(seed=15):
    """even / odd / mixed (mixed = at least one even and one odd head) at k in KS, up to N_SUB distinct each;
    plus 'even_noH22': even-only subsets without head 22 at k = 4, 8, 12 (review fix 1)."""
    rng = np.random.RandomState(seed)
    even, odd, allh = list(range(0, 32, 2)), list(range(1, 32, 2)), list(range(32))
    even_no22 = [h for h in even if h != H22]
    out = []
    for kind, pool, ks in (("even", even, KS), ("odd", odd, KS), ("mixed", allh, KS),
                           ("even_noH22", even_no22, KS[:3])):
        for k in ks:
            seen = []
            target = 1 if (kind in ("even", "odd") and k == len(pool)) else N_SUB
            while len(seen) < target:
                s = sorted(rng.choice(pool, k, replace=False).tolist())
                if kind == "mixed" and (all(h % 2 == 0 for h in s) or all(h % 2 for h in s)):
                    continue
                if s not in seen:
                    seen.append(s)
            out += [(kind, k, s) for s in seen]
    return out


def random_layer_sets(n=10, seed=5, n_layers=32):
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        s = sorted(rng.choice(n_layers, 5, replace=False).tolist())
        if s != FIVE and s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------- multi-layer patching

class Multi:
    """True-head (o_proj input) patches at several layers; prompt pass only; right-aligned, position 0
    never patched. patches: list of (layer, heads, donor_activation)."""

    def __init__(self, p):
        self.p = p
        self.last_calls, self.last_hit_max = {}, None

    def _swap(self, h, saved, heads, end):
        bs, sl, d = h.shape
        bw = d // self.p.n_heads
        hr = h.view(bs, sl, self.p.n_heads, bw).clone()
        sr = saved.view(saved.shape[0], -1, self.p.n_heads, bw).to(hr.dtype)
        ts = sr.shape[1]
        n = min(end, ts) - 1
        idx = self.p.torch.tensor(heads, device=h.device, dtype=self.p.torch.long)
        if n > 0:
            hr[:, end - n:end, idx, :] = sr[:, ts - n:ts, idx, :]
        return hr.view(bs, sl, d)

    def _hooks(self, patches, end, calls):
        handles = []
        for L, heads, saved in patches:
            def fn(mod, args, L=L, heads=heads, saved=saved):
                calls.setdefault(L, []).append(int(args[0].shape[1]))
                if len(calls[L]) > 1:          # decode steps: leave untouched
                    return None
                return (self._swap(args[0], saved, heads, end),) + tuple(args[1:])
            handles.append(self.p.proj(L).register_forward_pre_hook(fn))
        return handles

    def generate(self, prompt, patches=()):
        p = self.p
        ids = p.encode(prompt)
        calls = {}
        hs = self._hooks(patches, ids["input_ids"].shape[1], calls)
        try:
            with p.torch.no_grad():
                out = p.model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                                       pad_token_id=p.tok.pad_token_id)
        finally:
            for h in hs:
                h.remove()
        new = out[0][ids["input_ids"].shape[1]:]
        self.last_calls = calls
        self.last_hit_max = len(new) >= MAX_NEW and int(new[-1]) != p.tok.eos_token_id
        return p.tok.decode(new, skip_special_tokens=True)

    def delta(self, prompt, big="9.8", small="9.11", patches=()):
        """log P(' big') - log P(' small') as the reply's first words, teacher-forced, same prompt patch."""
        p = self.p
        tp = p.ids(prompt)
        lp = {}
        for name, cont in (("big", " " + big), ("small", " " + small)):
            full = p.ids(prompt + cont)
            if full[:len(tp)] != tp or len(full) <= len(tp):
                return None
            x = p.torch.tensor([full], device=p.device)
            calls = {}
            hs = self._hooks(patches, len(tp), calls)
            try:
                with p.torch.no_grad():
                    logits = p.model(input_ids=x).logits[0].float()
            finally:
                for h in hs:
                    h.remove()
            logp = p.torch.log_softmax(logits, -1)
            lp[name] = float(sum(logp[i - 1, full[i]] for i in range(len(tp), len(full))))
        return lp["big"] - lp["small"]

    def capture(self, prompt, layers):
        return {L: self.p.capture(prompt, "pre_wo", L) for L in layers}


# ---------------------------------------------------------------- pre-registered rules

def decide(run, res):
    """All pre-registered rules (experiment file, 'Decision rules'). Pure function of run/res."""
    d = {}
    sc = run["screen"]
    d["S"] = (sc["chat"]["class"] == "bug" and sc["comparative"]["class"] == "bug"
              and sc["qa"]["class"] != "bug" and sc["simple"]["class"] != "bug"
              and "correct" in (sc["qa"]["class"], sc["simple"]["class"]))
    if "layer_sweep" not in res:
        d["note"] = "no pair: M1-M6 not run"
        return d
    base = sc["chat"]["delta"]
    dt = base is not None and base <= 0
    d["delta_testable"] = dt
    d["delta_baseline_chat"] = base
    pos = lambda v: v.get("delta") is not None and v["delta"] > 0
    corr = lambda v: v["class"] == "correct"
    ls = res["layer_sweep"]
    d["M1_correct_layers"] = [int(L) for L, v in ls.items() if corr(v)]
    d["M1_delta_pos_layers"] = [int(L) for L, v in ls.items() if pos(v)]
    d["M1a_no_layer_repairs_gen"] = not d["M1_correct_layers"]
    d["M1b_four_of_five_delta"] = (sum(pos(ls[str(L)]) for L in FIVE) >= 4) if dt else "untestable"
    par = res["parity"]
    for L in PARITY_LAYERS:
        rows = [x for x in par if x["layer"] == L]
        for reading, ok in (("gen", corr), ("delta", pos)):
            if reading == "delta" and not dt:
                d[f"M2_L{L}_delta"] = "untestable"
                continue
            e16 = [x for x in rows if x["kind"] == "even" and x["k"] == 16]
            odd = [x for x in rows if x["kind"] == "odd"]
            e_no = [x for x in rows if x["kind"] == "even_noH22"]
            odd_ks = [x for x in odd if x["k"] in KS[:3]]
            a = bool(e16) and ok(e16[0])
            b = not any(ok(x) for x in odd)
            c = sum(ok(x) for x in e_no) > sum(ok(x) for x in odd_ks)
            verdict = ("parity holds" if (a and b and c) else
                       "even-only repair carried by H22" if (a and b) else "not supported")
            rates = {}
            for kind in ("even", "odd", "mixed", "even_noH22"):
                for k in KS:
                    sel = [x for x in rows if x["kind"] == kind and x["k"] == k]
                    if sel:
                        rates[f"{kind}_k{k}"] = [sum(ok(x) for x in sel), len(sel)]
            withh = [x for x in rows if x["kind"] == "even" and H22 in x["heads"]]
            without = [x for x in rows if x["kind"] == "even" and H22 not in x["heads"]]
            rates["even_with_H22"] = [sum(ok(x) for x in withh), len(withh)]
            rates["even_without_H22"] = [sum(ok(x) for x in without), len(without)]
            d[f"M2_L{L}_{reading}"] = {"a_even16": a, "b_no_odd": b, "c_noH22_gt_odd": c,
                                        "verdict": verdict, "rates": rates}
    sh = res["single_heads"]
    corr_at = {L: [v["head"] for v in sh.values() if v["layer"] == L and corr(v)] for L in PARITY_LAYERS}
    d["M3_correct_single_heads"] = {str(L): corr_at[L] for L in PARITY_LAYERS}
    d["M3_gen"] = all(corr_at[L] == [H22] for L in PARITY_LAYERS)
    if dt:
        tot = {}
        for v in sh.values():
            if v.get("delta") is not None:
                tot[v["head"]] = tot.get(v["head"], 0.0) + (v["delta"] - base)
        rank = sorted(tot, key=lambda h: -tot[h])
        d["M3b_H22_largest_summed_delta"] = bool(rank) and rank[0] == H22
        d["summed_delta_top10"] = [[h, round(tot[h], 3)] for h in rank[:10]]
        d["draft_top5_ranks"] = {str(h): (rank.index(h) + 1 if h in rank else None) for h in DRAFT_TOP5}
    else:
        d["M3b_H22_largest_summed_delta"] = "untestable"
    for cfg in ("five", "all"):
        ctrl = res[f"head_{cfg}_layers"]
        d[f"M4_{cfg}_correct_heads"] = [int(h) for h, v in ctrl.items() if corr(v)]
    d["M4"] = corr(res["head_five_layers"][str(H22)]) or corr(res["head_all_layers"][str(H22)])
    d["M4_H22_special"] = d["M4"] and all(len(d[f"M4_{c}_correct_heads"]) <= 4 for c in ("five", "all"))
    d["M5"] = corr(res["five_layer_all_heads"])
    d["M5_random_5layer_correct"] = [sum(corr(v) for v in res["random_5layer"]), len(res["random_5layer"])]
    rv = res["reverse"]["five"]
    d["M6"] = d["M5"] and corr(rv)
    d["M6_reverse_class"] = rv["class"]
    if "generalization" in res:
        g = res["generalization"]
        d["M7_bug_counts"] = {k: sum(r["formats"][k]["class"] == "bug" for r in g) for k in g[0]["formats"]}
        el = [r for r in g if "five_layer_repair" in r]
        d["M7_eligible"], d["M7_repaired"] = len(el), sum(r["five_layer_repair"]["class"] == "correct" for r in el)
        d["M7_repair_transfers"] = bool(el) and d["M7_repaired"] == len(el)
    return d


# ---------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None, help="CPU smoke only")
    ap.add_argument("--limit", action="store_true", help="smoke: tiny config lists")
    ap.add_argument("--force-pair", default=None, help="smoke only: DONOR,RECIPIENT format names")
    a = ap.parse_args()
    t0 = time.time()
    r14.selftest()
    print(f"r14 selftest {time.time()-t0:.1f}s", flush=True)
    for t, want in R14_CASES:
        assert CLS(t) == want == r14.classify(t), (t, CLS(t), want)
    assert make_classifier("4.3", "4.28")("4.30 is larger than 4.28") == "correct"
    if (a.force_pair or a.repo) and not a.limit:
        sys.exit("--force-pair/--repo are smoke-only (need --limit)")
    if not os.environ.get("R15_GIT_SHA"):
        sys.exit("FATAL: set R15_GIT_SHA")
    r14_sha = r14.sha256(R14_PATH)
    if not a.limit and r14_sha != R14_EXPECTED_SHA256:
        sys.exit(f"FATAL: R14 script sha256 {r14_sha} != expected {R14_EXPECTED_SHA256}")
    os.makedirs(a.out, exist_ok=False)
    r14.MAX_NEW = MAX_NEW
    os.environ["R14_GIT_SHA"] = os.environ["R15_GIT_SHA"]   # r14.provenance reads it; never a stale value
    p = r14.Patcher(a.repo or REPO, revision="main" if a.repo else SNAPSHOT, dtype="bfloat16")
    if not a.repo and not str(p.commit).startswith(SNAPSHOT[:8]):
        sys.exit(f"FATAL: loaded snapshot {p.commit}, expected {SNAPSHOT}")
    m = Multi(p)
    nL, nH = p.n_layers, p.n_heads
    run = {"provenance": r14.provenance(p, {"r15_script_sha256": r14.sha256(__file__),
                                            "r14_script_sha256": r14_sha, "r15_git_sha": os.environ["R15_GIT_SHA"],
                                            "max_new": MAX_NEW}),
           "started": datetime.now().isoformat()}
    path, rpath = os.path.join(a.out, "run.json"), os.path.join(a.out, "results.json")
    res = {}
    save = lambda: (r14.dump(run, path), r14.dump(res, rpath))

    def rec(prompt, patches=(), cls=CLS, big="9.8", small="9.11"):
        txt = m.generate(prompt, patches)
        return {"class": cls(txt), "delta": m.delta(prompt, big, small, patches), "output": txt,
                "hit_max_new": m.last_hit_max, "says_less": bool(RE_LESS.search(txt))}

    fm = formats(p.tok, "9.8", "9.11")
    tt = p.tok.apply_chat_template([{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}],
                                   tokenize=True, add_generation_prompt=True)
    tt = list(tt["input_ids"] if hasattr(tt, "keys") else tt)   # transformers 5 returns a BatchEncoding
    run["chat_template_check"] = {"string_path_ids": p.ids(fm["chat"]), "tokenize_true_ids": [int(x) for x in tt],
                                  "same": p.ids(fm["chat"]) == [int(x) for x in tt]}
    scr = {}
    for k, pr in {**fm, **chat_wrapped_wordings(p.tok)}.items():
        scr[k] = {"prompt": pr, "n_tokens": len(p.ids(pr)), **rec(pr)}
        print(f"[screen] {k} ({scr[k]['n_tokens']} tok) {scr[k]['class']} delta={scr[k]['delta']} "
              f"hit_max={scr[k]['hit_max_new']}: {scr[k]['output']!r}", flush=True)
    run["screen"] = scr
    recip = "chat" if scr["chat"]["class"] == "bug" else None
    donor = next((d for d in ("simple", "qa") if scr[d]["class"] == "correct"), None)
    if a.force_pair:
        donor, recip = a.force_pair.split(",")
    run["pair"] = {"donor": donor, "recipient": recip,
                   "note": "qa donor is off-by-one aligned to chat" if donor == "qa" else ""}
    print(f"[pair] {donor} -> {recip}", flush=True)
    save()

    five = [L for L in FIVE if L < nL]
    if donor and recip:
        dp, rp = fm[donor], fm[recip]
        run["alignment_right"] = r14.alignment_info(p, dp, rp, "right")
        iso = r14.head_isolation(p, dp, rp, min(2, nL - 1))
        run["head_isolation"] = iso
        print(f"[gate] head isolation {iso}", flush=True)
        D = m.capture(dp, range(nL))
        own = m.capture(rp, five)
        allh = list(range(nH))
        unp = m.generate(rp)
        ident = m.generate(rp, [(L, allh, own[L]) for L in five])
        det = [m.generate(rp, [(L, allh, D[L]) for L in five]) for _ in range(3)]
        t = len(p.ids(rp))
        g = {"identity_ok": ident == unp, "determinism_ok": len(set(det)) == 1,
             "hook_shapes_ok": all(v[0] == t and all(s == 1 for s in v[1:]) for v in m.last_calls.values()),
             "identity_output": ident, "unpatched": unp, "determinism_outputs": det,
             "hook_calls": {str(L): v for L, v in m.last_calls.items()}}
        g["ok"] = iso["ok"] and g["identity_ok"] and g["determinism_ok"] and g["hook_shapes_ok"]
        run["gates"] = g
        print(f"[gate] identity={g['identity_ok']} determinism={g['determinism_ok']} shapes={g['hook_shapes_ok']}",
              flush=True)
        save()
        if not g["ok"]:
            sys.exit("FATAL: gate failed (pre-registered stop)")

        res["all_layers_all_heads"] = rec(rp, [(L, allh, D[L]) for L in range(nL)])
        print(f"[sanity] all layers all heads: {res['all_layers_all_heads']['class']}", flush=True)
        res["layer_sweep"] = {str(L): rec(rp, [(L, allh, D[L])]) for L in range(nL)}
        print(f"[M1] correct layers: {[L for L, v in res['layer_sweep'].items() if v['class'] == 'correct']}; "
              f"delta>0 layers: {[L for L, v in res['layer_sweep'].items() if (v['delta'] or 0) > 0]}", flush=True)
        save()
        t1 = time.time()
        res["single_heads"] = {}
        for L in ([2, 17] if a.limit else range(nL)):
            for h in ([21, 22] if a.limit else range(nH)):
                res["single_heads"][f"L{L}_H{h}"] = {"layer": L, "head": h, **rec(rp, [(L, [h], D[L])])}
            print(f"[M3] layer {L}, {time.time()-t1:.0f}s; correct heads: "
                  f"{[v['head'] for v in res['single_heads'].values() if v['layer'] == L and v['class'] == 'correct']}",
                  flush=True)
            save()
        subs = parity_subsets()
        if a.limit:
            subs = [s for s in subs if s[1] == 16][:3] + [s for s in subs if s[0] == "even_noH22"][:2]
        res["parity"] = []
        for L in [x for x in PARITY_LAYERS if x < nL]:
            for kind, k, s in subs:
                res["parity"].append({"layer": L, "kind": kind, "k": k, "heads": s, **rec(rp, [(L, s, D[L])])})
            save()
        res["five_layer_all_heads"] = rec(rp, [(L, allh, D[L]) for L in five])
        res["random_5layer"] = [{"layers": ls_, **rec(rp, [(L, allh, D[L]) for L in ls_])}
                                for ls_ in random_layer_sets(3 if a.limit else 10, n_layers=nL)]
        heads_ctrl = [21, 22] if a.limit else range(nH)
        res["head_five_layers"] = {str(h): rec(rp, [(L, [h], D[L]) for L in five]) for h in heads_ctrl}
        res["head_all_layers"] = {str(h): rec(rp, [(L, [h], D[L]) for L in range(nL)]) for h in heads_ctrl}
        Rb = m.capture(rp, five)
        res["reverse"] = {str(L): rec(dp, [(L, allh, Rb[L])]) for L in five}
        res["reverse"]["five"] = rec(dp, [(L, allh, Rb[L]) for L in five])
        print(f"[M4-M6] five={res['five_layer_all_heads']['class']} "
              f"random5={[x['class'] for x in res['random_5layer']]} "
              f"h22_five={res['head_five_layers'][str(H22 % nH)]['class'] if str(H22 % nH) in res['head_five_layers'] else None} "
              f"reverse_five={res['reverse']['five']['class']}", flush=True)
        save()

    # M7 generalization
    pairs = gen_pairs()[: (3 if a.limit else 100)]
    res["generalization"] = []
    for big, small in pairs:
        cls = make_classifier(big, small)
        f2 = formats(p.tok, big, small)
        row = {"big": big, "small": small, "formats": {}}
        for k, pr in f2.items():
            txt = m.generate(pr)
            row["formats"][k] = {"class": cls(txt), "output": txt, "hit_max_new": m.last_hit_max}
        if row["formats"]["chat"]["class"] == "bug" and row["formats"]["simple"]["class"] == "correct":
            Dg = m.capture(f2["simple"], five)
            txt = m.generate(f2["chat"], [(L, list(range(nH)), Dg[L]) for L in five])
            row["five_layer_repair"] = {"class": cls(txt), "output": txt}
        res["generalization"].append(row)
        save()
    run["decisions"] = decide(run, res)
    print(f"[decisions] {run['decisions']}", flush=True)
    run["finished"] = datetime.now().isoformat()
    save()


if __name__ == "__main__":
    main()
