#!/usr/bin/env python3
"""
Experiment R15: test the v2 draft's Mistral-7B head claims with true per-head, prompt-only patching
==================================================================================================
Experiment file: docs/experiments/11-22_mistral7b_true_head_patching.md

The claims came from a collaborator by email (no code). This run tests them; it cannot reproduce the
numbers. Patching code is R14's (imported by path; both script hashes are recorded).
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R14_PATH = os.path.join(HERE, "..", "exp_R14_cross_model_true_heads", "run_experiment.py")
_spec = importlib.util.spec_from_file_location("r14", R14_PATH)
r14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r14)

REPO = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW = 40
FIVE = [2, 10, 11, 13, 17]
PARITY_LAYERS = [2, 17]
KS = [4, 8, 12, 16]
N_SUB = 20
H22 = 22


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


def formats(tok, big, small):
    q = f"Which is bigger: {big} or {small}?"
    chat = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    if tok.bos_token and chat.startswith(tok.bos_token):
        chat = chat[len(tok.bos_token):]
    return {"chat": chat, "comparative": f"Compare {big} and {small}. Which is larger?",
            "qa": f"Q: {q}\nA:", "simple": f"{q}\nAnswer:"}


def gen_pairs(n=100, seed=0):
    """X.D vs X.DD with the shorter one larger: DD < 10*D, DD not ending in 0. Distinct, seeded."""
    rng = np.random.RandomState(seed)
    out = set()
    while len(out) < n:
        x, d = int(rng.randint(1, 10)), int(rng.randint(2, 10))
        dd = int(rng.randint(10, 10 * d))
        if dd % 10 == 0 or (x, d) == (9, 8) and dd == 11:
            continue
        out.add((f"{x}.{d}", f"{x}.{dd}"))
    return sorted(out)


def parity_subsets(seed=15):
    rng = np.random.RandomState(seed)
    even, odd, allh = list(range(0, 32, 2)), list(range(1, 32, 2)), list(range(32))
    out = []
    for kind, pool in (("even", even), ("odd", odd), ("mixed", allh)):
        for k in KS:
            seen = []
            target = 1 if (kind != "mixed" and k == len(pool)) else N_SUB
            while len(seen) < target:
                s = sorted(rng.choice(pool, k, replace=False).tolist())
                if s not in seen:
                    seen.append(s)
            out += [(kind, k, s) for s in seen]
    return out


# ---------------------------------------------------------------- multi-layer patching

class Multi:
    """True-head (o_proj input) patches at several layers; prompt pass only; right-aligned, position 0
    never patched. patches: list of (layer, heads, donor_activation)."""

    def __init__(self, p):
        self.p = p

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
        self.last_calls = calls
        return p.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

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


# ---------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None, help="CPU smoke only")
    ap.add_argument("--limit", action="store_true", help="smoke: tiny config lists")
    ap.add_argument("--force-pair", default=None, help="smoke only: DONOR,RECIPIENT format names")
    a = ap.parse_args()
    for t, want in [(" 9.8 is bigger than 9.11.", "correct"), ("**9.11** is bigger", "bug")]:
        assert CLS(t) == want
    r14.selftest()
    # the pair-generic classifier must equal R14's on 9.8/9.11
    for t in [" 9.8 is bigger than 9.11.", "I think the answer is:\n\n9.11", "9.80 is bigger than 9.11",
              " 9.8\n\nQ: Which is bigger: 9.11 or 9.12?\nA: 9.11 is bigger", "The answer is:\n\nThe answer is:",
              "Comparing 9.8 and 9.11:\n9.8 wins", "19.8 is bigger than 9.11", "29.11 is bigger than 3"]:
        assert CLS(t) == r14.classify(t), t
    assert make_classifier("4.3", "4.28")("4.30 is larger than 4.28") == "correct"
    if (a.force_pair or a.repo) and not a.limit:
        sys.exit("--force-pair/--repo are smoke-only (need --limit)")
    if not os.environ.get("R15_GIT_SHA"):
        sys.exit("FATAL: set R15_GIT_SHA")
    os.makedirs(a.out, exist_ok=False)
    r14.MAX_NEW = MAX_NEW
    os.environ.setdefault("R14_GIT_SHA", os.environ["R15_GIT_SHA"])   # r14.provenance reads it
    p = r14.Patcher(a.repo or REPO, dtype="bfloat16")
    m = Multi(p)
    nL, nH = p.n_layers, p.n_heads
    run = {"provenance": r14.provenance(p, {"r15_script_sha256": r14.sha256(__file__),
                                            "r14_script_sha256": r14.sha256(R14_PATH),
                                            "r15_git_sha": os.environ["R15_GIT_SHA"], "max_new": MAX_NEW}),
           "started": datetime.now().isoformat()}
    path = os.path.join(a.out, "run.json")
    save = lambda: r14.dump(run, path)

    fm = formats(p.tok, "9.8", "9.11")
    scr = {}
    for k, pr in fm.items():
        t = m.generate(pr)
        scr[k] = {"prompt": pr, "output": t, "class": CLS(t), "delta": m.delta(pr), "n_tokens": len(p.ids(pr))}
        print(f"[screen] {k} ({scr[k]['n_tokens']} tok) {scr[k]['class']} delta={scr[k]['delta']}: {t!r}", flush=True)
    run["screen"] = scr
    recip = "chat" if scr["chat"]["class"] == "bug" else None
    donor = next((d for d in ("simple", "qa") if scr[d]["class"] == "correct"), None)
    if a.force_pair:
        donor, recip = a.force_pair.split(",")
    run["pair"] = {"donor": donor, "recipient": recip}
    print(f"[pair] {donor} -> {recip}", flush=True)
    save()

    five = [L for L in FIVE if L < nL]
    if donor and recip:
        dp, rp = fm[donor], fm[recip]
        iso = r14.head_isolation(p, dp, rp, min(2, nL - 1))
        run["head_isolation"] = iso
        print(f"[gate] head isolation {iso}", flush=True)
        D = m.capture(dp, range(nL))
        own = m.capture(rp, five)
        allh = list(range(nH))
        unp = m.generate(rp)
        ident = m.generate(rp, [(L, allh, own[L]) for L in five])
        det = [m.generate(rp, [(L, allh, D[L]) for L in five]) for _ in range(3)]
        shapes = {str(L): v for L, v in m.last_calls.items()}
        t = len(p.ids(rp))
        g = {"identity_ok": ident == unp, "determinism_ok": len(set(det)) == 1,
             "hook_shapes_ok": all(v[0] == t and all(s == 1 for s in v[1:]) for v in m.last_calls.values()),
             "identity_output": ident, "unpatched": unp, "determinism_outputs": det, "hook_calls": shapes}
        g["ok"] = iso["ok"] and g["identity_ok"] and g["determinism_ok"] and g["hook_shapes_ok"]
        run["gates"] = g
        print(f"[gate] identity={g['identity_ok']} determinism={g['determinism_ok']} shapes={g['hook_shapes_ok']}",
              flush=True)
        save()
        if not g["ok"]:
            sys.exit("FATAL: gate failed (pre-registered stop)")

        def rec(patches, donor_prompt=None, recip_prompt=rp, cls=CLS):
            txt = m.generate(recip_prompt, patches)
            return {"class": cls(txt), "delta": m.delta(recip_prompt, patches=patches), "output": txt}

        res = {}
        res["all_layers_all_heads"] = rec([(L, allh, D[L]) for L in range(nL)])
        print(f"[sanity] all layers all heads: {res['all_layers_all_heads']['class']}", flush=True)
        res["layer_sweep"] = {str(L): rec([(L, allh, D[L])]) for L in range(nL)}
        print(f"[M1] correct layers: {[L for L, v in res['layer_sweep'].items() if v['class'] == 'correct']}",
              flush=True)
        save_res = lambda: r14.dump(res, os.path.join(a.out, "results.json"))
        save_res()
        layers_sh = [2, 17] if a.limit else range(nL)
        heads_sh = [21, 22] if a.limit else range(nH)
        t0 = time.time()
        res["single_heads"] = {}
        for L in layers_sh:
            for h in heads_sh:
                res["single_heads"][f"L{L}_H{h}"] = {"layer": L, "head": h, **rec([(L, [h], D[L])])}
            print(f"[M3] layer {L} done, {time.time()-t0:.0f}s; correct heads: "
                  f"{[h for h in heads_sh if res['single_heads'][f'L{L}_H{h}']['class'] == 'correct']}", flush=True)
            save_res()
        subs = parity_subsets()
        if a.limit:
            subs = [s for s in subs if s[1] == 16][:3]
        res["parity"] = []
        for L in PARITY_LAYERS:
            if L >= nL:
                continue
            for kind, k, s in subs:
                res["parity"].append({"layer": L, "kind": kind, "k": k, "heads": s, **rec([(L, s, D[L])])})
            save_res()
        res["five_layer_all_heads"] = rec([(L, allh, D[L]) for L in five])
        res["h22_five_layers"] = rec([(L, [H22 % nH], D[L]) for L in five])
        res["h22_all_layers"] = rec([(L, [H22 % nH], D[L]) for L in range(nL)])
        Rb = m.capture(rp, five)
        res["reverse"] = {str(L): rec([(L, allh, Rb[L])], recip_prompt=dp) for L in five}
        res["reverse"]["five"] = rec([(L, allh, Rb[L]) for L in five], recip_prompt=dp)
        print(f"[M4-M6] five={res['five_layer_all_heads']['class']} h22_five={res['h22_five_layers']['class']} "
              f"h22_all={res['h22_all_layers']['class']} reverse_five={res['reverse']['five']['class']}", flush=True)
        save_res()
    else:
        res = {"no_pair": True}

    # M7 generalization
    pairs = gen_pairs()[: (3 if a.limit else 100)]
    gen = []
    for big, small in pairs:
        cls = make_classifier(big, small)
        f2 = formats(p.tok, big, small)
        row = {"big": big, "small": small, "formats": {}}
        for k, pr in f2.items():
            t = m.generate(pr)
            row["formats"][k] = {"class": cls(t), "output": t}
        if row["formats"]["chat"]["class"] == "bug" and row["formats"]["simple"]["class"] == "correct":
            Dg = m.capture(f2["simple"], five)
            t = m.generate(f2["chat"], [(L, list(range(nH)), Dg[L]) for L in five])
            row["five_layer_repair"] = {"class": cls(t), "output": t}
        gen.append(row)
    res["generalization"] = gen
    errs = {k: sum(r["formats"][k]["class"] == "bug" for r in gen) for k in fm}
    elig = [r for r in gen if "five_layer_repair" in r]
    print(f"[M7] bug counts {errs} of {len(gen)}; eligible {len(elig)}, repaired "
          f"{sum(r['five_layer_repair']['class'] == 'correct' for r in elig)}", flush=True)
    r14.dump(res, os.path.join(a.out, "results.json"))
    run["finished"] = datetime.now().isoformat()
    save()


if __name__ == "__main__":
    main()
