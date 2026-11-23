#!/usr/bin/env python3
"""
Experiment R17: where does Llama-3.1-8B decide 9.8 vs 9.11? (corrected activation-patching map)
=============================================================================================
Experiment file: docs/experiments/11-23_llama8b_localization_map.md

Donor  = "Which is bigger: 9.8 or 9.11?\\nAnswer:" + " 9."   (17 + 3 tokens)
Recip. = "Q: Which is bigger: 9.8 or 9.11?\\nA:"   + " 9."   (19 + 3 tokens)
Both models' greedy answers start with the tokens " ", "9", "." (gate); the choice is made at the next token.
Metric: delta = logit("8") - logit("11") at the last teacher-forced position. r = (d_p - d_b) / (d_c - d_b).

Right alignment: donor position i -> recipient position i + 2 (i >= 1). Prompt positions: donor 1..16;
answer-prefix positions: donor 17..19. Recipient positions 1-2 ("Q", ":") have no counterpart; BOS is not patched.

Components at every layer L: resid (input of decoder layer L), attn (self_attn output = o_proj output),
mlp (mlp output). Position sets: each single prompt position, all prompt positions, each answer position,
all answer positions. Prompt cells with r >= 0.5 are confirmed by greedy generation with a prefill-only patch.
Exit code 2 on any gate failure.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
R13_PATH = os.path.join(HERE, "..", "exp_R13_true_head_patching", "run_experiment.py")
R13_SHA256 = "1ce5b0cc2e2cdc74c21de37de5340cd00f5b14e27ae13cf6893d2e09f280348d"

MODEL_NAME = os.environ.get("R17_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DONOR_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"
RECIP_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"
PREFIX_IDS = [220, 24, 13]            # " ", "9", "."  (Llama-3 tokenizer)
TOK_8, TOK_11 = 23, 806
OFFSET = 2                            # recipient position = donor position + 2
ALLOWED_MISMATCH = {1, 15}            # donor "Which"/"Answer" vs recipient " Which"/"A"
MAX_NEW = 20
R_REPAIR, R_PARTIAL = 0.5, 0.2
MIN_GAP = 1.0                         # d_c - d_b must be at least this (logits)
R_CONTROL = 0.2                       # controls must stay below this for a repair
R_POSCTRL = 0.99                      # positive control: final-norm input at the last position (by construction)
R_SOFTCTRL = 0.8                      # reported only: resid at layer n-1, all answer positions
MISMATCH_POS = {1, 15}                # donor positions whose token differs from the recipient's
SEED = 17
COMPONENTS = ["resid", "attn", "mlp"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_r13():
    got = sha256(R13_PATH)
    if got != R13_SHA256:
        sys.exit(f"R13 script sha256 {got} != expected {R13_SHA256}")
    spec = importlib.util.spec_from_file_location("r13", R13_PATH)
    mod = importlib.util.module_from_spec(spec)
    os.environ.setdefault("R13_MODEL", MODEL_NAME)
    spec.loader.exec_module(mod)
    return mod


class Model:
    def __init__(self):
        print(f"Loading {MODEL_NAME} on {DEVICE} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)

    def ids(self, prompt, prefix=True):
        ids = self.tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
        return ids + (PREFIX_IDS if prefix else [])

    def module(self, comp, L):
        return {"resid": self.layers[L], "attn": self.layers[L].self_attn, "mlp": self.layers[L].mlp}[comp]

    # ---- hooks -------------------------------------------------------------------------------------
    def _register(self, comp, L, fn_tensor):
        """fn_tensor(h) -> new h or None. resid: pre-hook on the decoder layer; attn/mlp: forward hook."""
        if comp == "final":            # input of the final norm (pre-hook, positional)
            def pre_final(m, args):
                new = fn_tensor(args[0])
                return None if new is None else (new,) + tuple(args[1:])
            return self.model.model.norm.register_forward_pre_hook(pre_final)
        mod = self.module(comp, L)
        if comp == "resid":
            def pre(m, args, kwargs):
                if args:
                    new = fn_tensor(args[0])
                    return None if new is None else ((new,) + tuple(args[1:]), kwargs)
                new = fn_tensor(kwargs["hidden_states"])
                if new is None:
                    return None
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = new
                return (args, kwargs)
            return mod.register_forward_pre_hook(pre, with_kwargs=True)

        def post(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            new = fn_tensor(h)
            if new is None:
                return None
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return mod.register_forward_hook(post)

    def capture(self, ids):
        """All components at all layers for one token sequence; returns {comp: [L] -> [T, d]} and logits.
        store["final"] is the input of the final norm (= output of the last decoder layer)."""
        store = {c: [None] * self.n_layers for c in COMPONENTS}
        def fin(m, args):
            store["final"] = args[0][0].detach().clone()
        handles = [self.model.model.norm.register_forward_pre_hook(fin)]
        for c in COMPONENTS:
            for L in range(self.n_layers):
                def fn(h, c=c, L=L):
                    store[c][L] = h[0].detach().clone()
                    return None
                handles.append(self._register(c, L, fn))
        try:
            with torch.no_grad():
                logits = self.model(torch.tensor([ids], device=DEVICE)).logits[0, -1].float()
        finally:
            for h in handles:
                h.remove()
        return store, logits

    def logits(self, ids, patches=(), probe=None):
        """patches: list of (comp, L, {recip_pos: tensor[d]}). Returns last-position logits (fp32).
        probe=(comp, L): a capture hook registered AFTER the patch hooks; its tensor is stored in self.probed."""
        handles = []
        for comp, L, posmap in patches:
            def fn(h, posmap=posmap):
                h = h.clone()
                for p, v in posmap.items():
                    h[0, p] = v.to(h.dtype)
                return h
            handles.append(self._register(comp, L, fn))
        if probe is not None:
            def grab(h):
                self.probed = h[0].detach().clone()
                return None
            handles.append(self._register(probe[0], probe[1], grab))
        try:
            with torch.no_grad():
                return self.model(torch.tensor([ids], device=DEVICE)).logits[0, -1].float()
        finally:
            for h in handles:
                h.remove()

    def generate(self, prompt, patch=None):
        """Greedy MAX_NEW tokens. patch = (comp, L, {recip_pos: tensor}) applied on the prefill call only."""
        ids = self.tok(prompt, return_tensors="pt").to(DEVICE)
        T = ids["input_ids"].shape[1]
        calls, handle = [], None
        if patch is not None:
            comp, L, posmap = patch

            def fn(h):
                calls.append(int(h.shape[1]))
                if len(calls) > 1:
                    return None
                assert h.shape[1] == T, f"prefill length {h.shape[1]} != {T}"
                h = h.clone()
                for p, v in posmap.items():
                    h[0, p] = v.to(h.dtype)
                return h
            handle = self._register(comp, L, fn)
        try:
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
        finally:
            if handle is not None:
                handle.remove()
        if patch is not None:
            assert len(calls) >= 2 and all(c == 1 for c in calls[1:]), f"hook calls {calls[:5]}"
        return out[0][T:].tolist(), self.tok.decode(out[0][T:], skip_special_tokens=True)


def delta(lg):
    return float(lg[TOK_8] - lg[TOK_11])


def position_sets(n_prompt_donor, n_total_donor):
    prompt = list(range(1, n_prompt_donor))
    answer = list(range(n_prompt_donor, n_total_donor))
    sets = [(f"p{i}", "prompt", [i]) for i in prompt] + [("prompt_all", "prompt", prompt)]
    sets += [(f"a{i - n_prompt_donor}", "answer", [i]) for i in answer] + [("answer_all", "answer", answer)]
    return sets


def near_input(comp, L, dpos):
    """attn/mlp at a mismatched-token position in layers <= 2: counted, but cannot carry 'localized' alone."""
    return comp in ("attn", "mlp") and L <= 2 and bool(set(dpos) & MISMATCH_POS)


def input_substitution(comp, L, dpos):
    """Pre-registered exclusion: patches that mostly swap a token's identity rather than localize a mechanism."""
    hits_mismatch = bool(set(dpos) & MISMATCH_POS)
    if comp == "resid":
        return L == 0 or hits_mismatch
    return L == 0 and hits_mismatch


def classify(r13, text):
    return "correct" if r13.is_correct(text) else ("bug" if r13.is_bug(text) else "neither")


def fallback_only(r13, text):
    lo = text.lower()
    return r13.is_correct(text) and not any(p in lo for p in r13.CORRECT_PATTERNS)


def decide(cells, n_layers):
    """Pre-registered outcome labels (experiment file, 'Pre-registered decision rules')."""
    counted = [c for c in cells if c["kind"] == "prompt" and not c["input_substitution"]]
    rep = [c for c in counted if c["repair"]]
    rep_nn = [c for c in rep if not c.get("near_input")]        # near-input cells carry no label, no trigger
    single = [c for c in rep_nn if c["set"] != "prompt_all"]
    out = {"repairing_prompt_cells": [(c["comp"], c["layer"], c["set"]) for c in rep]}
    if single:
        out["label"] = "localized: single-position prompt patches repair"
    elif any(c["set"] == "prompt_all" for c in rep_nn):
        out["label"] = "distributed across prompt positions: only all-position patches repair (non-near-input cells)"
    else:
        out["label"] = "no prompt-only patch repairs"
    out["near_input_repairs"] = [(c["comp"], c["layer"], c["set"]) for c in rep if c.get("near_input")]
    out["position_specific_repairs"] = [(c["comp"], c["layer"], c["set"]) for c in rep if c.get("position_specific")]
    out["input_substitution_repairs"] = [(c["comp"], c["layer"], c["set"]) for c in cells
                                         if c["kind"] == "prompt" and c["input_substitution"] and c["repair"]]
    resid_all = [c["layer"] for c in cells if c["kind"] == "prompt" and c["comp"] == "resid"
                 and c["set"] == "prompt_all" and c["repair"]]
    out["resid_prompt_all_latest_repairing_layer"] = max(resid_all, default=None)
    out["flips_without_r"] = [(c["comp"], c["layer"], c["set"]) for c in counted
                              if c.get("gen_class") == "correct" and c["r"] < R_REPAIR and c["delta"] > 0]
    out["repairs_via_classifier_fallback"] = [(c["comp"], c["layer"], c["set"]) for c in rep if c.get("fallback")]
    attn_rep = sorted({c["layer"] for c in rep_nn if c["comp"] == "attn"})
    if attn_rep:
        out["stage2_layers"] = attn_rep
        out["stage2_basis"] = "attn prompt cells that repair"
    else:
        cand = [c for c in counted if c["comp"] == "attn" and c["set"] == "prompt_all" and c["r"] >= R_PARTIAL
                and not c.get("near_input")
                and c.get("control_max_r", -1.0) < R_CONTROL]
        cand = sorted(cand, key=lambda c: -c["r"])[:3]
        out["stage2_layers"] = sorted(c["layer"] for c in cand)
        out["stage2_basis"] = "top attn prompt_all layers with r >= 0.2" if cand else "none"
    ans = [c["layer"] for c in cells if c["kind"] == "answer" and c["set"] == "a2" and c["comp"] == "resid"
           and c["r"] >= R_REPAIR]
    out["answer_dot_resid_first_layer_r_ge_0.5"] = min(ans, default=None)
    out["answer_attn_layers_r_ge_0.5"] = sorted({c["layer"] for c in cells if c["kind"] == "answer"
                                                 and c["comp"] == "attn" and c["r"] >= R_REPAIR})
    return out


def decide_selftest():
    base = dict(kind="prompt", comp="resid", layer=3, set="p3", r=0.0, delta=-1.0, repair=False,
                input_substitution=False)
    mk = lambda **k: {**base, **k}
    assert decide([mk()], 32)["label"] == "no prompt-only patch repairs"
    assert decide([mk(set="prompt_all", r=0.9, repair=True)], 32)["label"].startswith("distributed")
    d = decide([mk(comp="attn", layer=7, r=0.8, repair=True)], 32)
    assert d["label"].startswith("localized") and d["stage2_layers"] == [7]
    d = decide([mk(comp="attn", layer=5, set="prompt_all", r=0.3), mk(comp="attn", layer=9, set="prompt_all", r=0.1)], 32)
    assert d["stage2_layers"] == [5] and d["label"] == "no prompt-only patch repairs"
    d = decide([mk(comp="attn", layer=5, set="prompt_all", r=0.6, repair=False)], 32)   # r high, generation not
    assert d["label"] == "no prompt-only patch repairs" and d["stage2_layers"] == [5]
    # only an mlp cell repairs: label localized, stage 2 falls back to the attn r >= 0.2 rule
    d = decide([mk(comp="mlp", layer=4, r=0.9, repair=True), mk(comp="attn", layer=6, set="prompt_all", r=0.25)], 32)
    assert d["label"].startswith("localized") and d["stage2_layers"] == [6] and d["stage2_basis"].startswith("top")
    # input substitution excluded from the label, reported separately
    d = decide([mk(layer=0, set="p15", r=0.9, repair=True, input_substitution=True)], 32)
    assert d["label"] == "no prompt-only patch repairs" and d["input_substitution_repairs"] == [("resid", 0, "p15")]
    # flips without r, fallback flag, answer-position outputs
    d = decide([mk(comp="mlp", r=0.3, delta=0.4, gen_class="correct"),
                mk(comp="attn", r=0.7, repair=True, fallback=True),
                {**base, "kind": "answer", "set": "a2", "layer": 12, "r": 0.6},
                {**base, "kind": "answer", "set": "a2", "layer": 20, "r": 0.9},
                {**base, "kind": "answer", "comp": "attn", "set": "a1", "layer": 15, "r": 0.55}], 32)
    assert d["flips_without_r"] == [("mlp", 3, "p3")] and d["repairs_via_classifier_fallback"] == [("attn", 3, "p3")]
    assert d["answer_dot_resid_first_layer_r_ge_0.5"] == 12 and d["answer_attn_layers_r_ge_0.5"] == [15]
    assert input_substitution("resid", 0, [5]) and input_substitution("resid", 9, [15])
    assert input_substitution("resid", 9, list(range(1, 17))) and not input_substitution("resid", 9, [5])
    assert input_substitution("mlp", 0, [1]) and not input_substitution("mlp", 1, [1])
    assert not input_substitution("attn", 0, [5])
    # near-input single repair alone cannot carry 'localized'; falls through to the next rule
    d = decide([mk(comp="mlp", layer=1, set="p15", r=0.9, repair=True, near_input=True)], 32)
    assert d["label"] == "no prompt-only patch repairs" and d["near_input_repairs"] == [("mlp", 1, "p15")]
    d = decide([mk(comp="mlp", layer=1, set="p15", r=0.9, repair=True, near_input=True),
                mk(comp="attn", layer=8, set="prompt_all", r=0.9, repair=True)], 32)
    assert d["label"].startswith("distributed") and d["stage2_layers"] == [8]
    # stage-2 fallback skips a checked cell whose random-direction control is high
    d = decide([mk(comp="attn", layer=5, set="prompt_all", r=0.4, control_max_r=0.3),
                mk(comp="attn", layer=6, set="prompt_all", r=0.3, control_max_r=0.1)], 32)
    assert d["stage2_layers"] == [6]
    d = decide([mk(comp="mlp", layer=1, set="prompt_all", r=0.9, repair=True, near_input=True),
                mk(comp="attn", layer=2, set="prompt_all", r=0.9, repair=True, near_input=True)], 32)
    assert d["label"] == "no prompt-only patch repairs" and d["stage2_layers"] == [] and d["stage2_basis"] == "none"
    assert near_input("mlp", 2, [15]) and not near_input("mlp", 3, [15]) and not near_input("resid", 1, [15])
    print("decide selftest: OK", flush=True)


def provenance():
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=HERE, text=True).strip()
        except Exception:
            return None
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                                      text=True).strip()
    except Exception:
        pass
    import transformers
    return {"hostname": socket.gethostname(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": gpu, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "git_sha": os.environ.get("R17_GIT_SHA") or git("rev-parse", "HEAD"),
            "script_sha256": sha256(os.path.abspath(__file__)), "r13_sha256": R13_SHA256, "model": MODEL_NAME}


def rel_pos_err(a, b):
    """Max over non-BOS positions of ||a_t - b_t|| / ||b_t|| (fp32)."""
    a, b = a[1:].float(), b[1:].float()
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny model on CPU: gates recorded, not enforced")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-confirm", type=int, default=0, help="smoke only: confirm the first N counted prompt cells")
    a = ap.parse_args()
    if a.force_confirm and not a.smoke:
        sys.exit("--force-confirm is smoke-only")
    decide_selftest()
    if a.selftest:
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "run.json")
    run = {"provenance": provenance(), "started": datetime.now().isoformat(), "command": " ".join(sys.argv),
           "gates": {}}
    dump(run, path)
    try:
        gates_phase(a, run, path)
    except Exception:
        import traceback
        run["error"] = traceback.format_exc()
        run["all_gates_ok"] = False
        run["finished"] = datetime.now().isoformat()
        dump(run, path)
        print(run["error"], flush=True)
        sys.exit(2)


def gates_phase(a, run, path):
    r13 = load_r13()
    M = Model()
    gates = run["gates"]
    gen = torch.Generator().manual_seed(SEED)

    def stop_if_failed():
        if not a.smoke and not all(v["ok"] for v in gates.values()):
            run["finished"] = datetime.now().isoformat()
            run["all_gates_ok"] = False
            dump(run, path)
            sys.exit(2)

    d_ids, r_ids = M.ids(DONOR_PROMPT), M.ids(RECIP_PROMPT)
    dp_ids, rp_ids = M.ids(DONOR_PROMPT, prefix=False), M.ids(RECIP_PROMPT, prefix=False)
    nd_prompt = len(dp_ids)
    run["tokens"] = {"donor": d_ids, "recipient": r_ids}
    mism = [i for i in range(1, len(d_ids)) if d_ids[i] != r_ids[i + OFFSET]]
    gates["alignment"] = {"n_donor_prompt": nd_prompt, "n_recip_prompt": len(rp_ids), "mismatch": mism,
                          "ok": len(rp_ids) - nd_prompt == OFFSET and len(r_ids) - len(d_ids) == OFFSET
                          and set(mism) <= MISMATCH_POS}

    # phenomenon: classes, the three prefix tokens and the deciding fourth token
    g_d, g_r = M.generate(DONOR_PROMPT), M.generate(RECIP_PROMPT)
    ph = {"donor": g_d[1], "recipient": g_r[1], "donor_class": classify(r13, g_d[1]),
          "recip_class": classify(r13, g_r[1]), "donor_first4": g_d[0][:4], "recip_first4": g_r[0][:4]}
    ph["ok"] = (ph["donor_class"] == "correct" and ph["recip_class"] == "bug"
                and g_d[0][:4] == PREFIX_IDS + [TOK_8] and g_r[0][:4] == PREFIX_IDS + [TOK_11])
    gates["phenomenon"] = ph

    # captures and baselines
    cap_d, lg_d = M.capture(d_ids)
    cap_r, lg_r = M.capture(r_ids)
    cap_dp, _ = M.capture(dp_ids)            # donor prompt only (descriptive comparison only)
    cap_rp, _ = M.capture(rp_ids)            # recipient prompt only (for the generation identity check)
    d_c, d_b = delta(lg_d), delta(lg_r)
    gap = d_c - d_b
    top = lambda lg: [[int(i), float(v)] for v, i in zip(*lg.topk(5))]
    gates["baseline"] = {"d_clean": d_c, "d_buggy": d_b, "gap": gap, "r_flip": (-d_b / gap) if gap else None,
                         "top5_donor": top(lg_d), "top5_recip": top(lg_r),
                         "capture_equals_plain": bool(torch.equal(lg_r, M.logits(r_ids)))}
    gates["baseline"]["ok"] = (d_b < 0 < d_c and gap >= MIN_GAP and gates["baseline"]["capture_equals_plain"]
                               and int(lg_d.argmax()) == TOK_8 and int(lg_r.argmax()) == TOK_11)
    # descriptive: prompt-only vs prefixed donor capture, scaled by the residual norm at that layer and position
    capdiff = max(float(((cap_dp[c][L][1:].float() - cap_d[c][L][1:nd_prompt].float()).norm(dim=-1)
                         / cap_d["resid"][L][1:nd_prompt].float().norm(dim=-1).clamp_min(1e-6)).max())
                  for c in COMPONENTS for L in range(M.n_layers))
    run["donor_captures_rel_diff"] = capdiff
    # decomposition per non-BOS position, including the last layer against the final-norm input
    worst = 0.0
    for L in range(M.n_layers):
        nxt = cap_d["resid"][L + 1] if L + 1 < M.n_layers else cap_d["final"]
        s = cap_d["resid"][L].float() + cap_d["attn"][L].float() + cap_d["mlp"][L].float()
        worst = max(worst, rel_pos_err(s, nxt))
    bitwise = all(bool(torch.equal((cap_d["resid"][L] + cap_d["attn"][L]) + cap_d["mlp"][L],
                                   cap_d["resid"][L + 1] if L + 1 < M.n_layers else cap_d["final"]))
                  for L in range(M.n_layers))
    gates["decomposition"] = {"max_rel_err": worst, "bitwise_replay_equal": bitwise, "ok": worst < 1e-2}
    check_layers = sorted({0, M.n_layers // 3, M.n_layers - 1})
    # identity (teacher-forced): own values -> bitwise identical logits
    ident = {}
    for comp in COMPONENTS:
        for L in check_layers:
            posmap = {p: cap_r[comp][L][p] for p in range(len(r_ids))}
            ident[f"{comp}{L}"] = bool(torch.equal(M.logits(r_ids, [(comp, L, posmap)]), lg_r))
    gates["identity"] = {"cells": ident, "ok": all(ident.values())}
    # active patch: a probe registered after the patch sees the donor values at the patched positions
    active = {}
    prompt_d = list(range(1, nd_prompt))
    for comp in COMPONENTS:
        for L in check_layers:
            posmap = {dp + OFFSET: cap_d[comp][L][dp] for dp in prompt_d}
            M.logits(r_ids, [(comp, L, posmap)], probe=(comp, L))
            active[f"{comp}{L}"] = all(bool(torch.equal(M.probed[p], v.to(M.probed.dtype))) for p, v in posmap.items())
    gates["active_patch"] = {"cells": active, "ok": all(active.values())}
    # generation identity: prefill-only patch with the recipient's own prompt values -> identical tokens
    gid = {}
    for comp in COMPONENTS:
        L = M.n_layers // 3
        posmap = {p: cap_rp[comp][L][p] for p in range(1, len(rp_ids))}
        gid[f"{comp}{L}"] = M.generate(RECIP_PROMPT, (comp, L, posmap))[0] == g_r[0]
    gates["generation_identity"] = {"cells": gid, "ok": all(gid.values())}
    # positive control (by construction): final-norm input at the last position
    lg_f = M.logits(r_ids, [("final", 0, {len(r_ids) - 1: cap_d["final"][len(d_ids) - 1]})])
    r_fin = (delta(lg_f) - d_b) / gap if gap else float("nan")
    gates["positive_control"] = {"r_final_norm_last": r_fin, "ok": r_fin >= R_POSCTRL}
    # softer, reported: resid at the last layer, all answer positions
    posmap = {dp + OFFSET: cap_d["resid"][M.n_layers - 1][dp] for dp in range(nd_prompt, len(d_ids))}
    r_soft = (delta(M.logits(r_ids, [("resid", M.n_layers - 1, posmap)])) - d_b) / gap if gap else float("nan")
    run["soft_positive_control"] = {"r_resid_last_layer_answer_all": r_soft, "meets_0.8": r_soft >= R_SOFTCTRL}
    dump(run, path)
    print("gates:", {k: v["ok"] for k, v in gates.items()}, f"d_c={d_c:.3f} d_b={d_b:.3f}", flush=True)
    stop_if_failed()

    sweep(a, run, path, gates, M, r13, gen, cap_d, cap_r, d_ids, r_ids, nd_prompt, d_b, d_c, gap, lg_r)


def sweep(a, run, path, gates, M, r13, gen, cap_d, cap_r, d_ids, r_ids, nd_prompt, d_b, d_c, gap, lg_r):
    sets = position_sets(nd_prompt, len(d_ids))
    cells = run["cells"] = []
    t0 = time.time()
    for comp in COMPONENTS:
        for L in range(M.n_layers):
            for name, kind, dpos in sets:
                posmap = {dp + OFFSET: cap_d[comp][L][dp] for dp in dpos}
                d_p = delta(M.logits(r_ids, [(comp, L, posmap)]))
                cells.append({"comp": comp, "layer": L, "set": name, "kind": kind, "donor_pos": dpos,
                              "delta": d_p, "r": (d_p - d_b) / gap if gap else float("nan"),
                              "input_substitution": kind == "prompt" and input_substitution(comp, L, dpos),
                              "near_input": kind == "prompt" and near_input(comp, L, dpos),
                              "repair": False})
        print(f"[{comp}] swept {M.n_layers} layers, {time.time() - t0:.0f}s", flush=True)
    eff = {comp: any(abs(c["delta"] - d_b) > 1e-3 for c in cells if c["comp"] == comp) for comp in COMPONENTS}
    gates["patch_has_effect"] = {"cells": eff, "ok": all(eff.values())}

    # generation check + controls for prompt cells with r >= 0.5 or delta > 0
    def r_of(lg):
        return (delta(lg) - d_b) / gap
    forced = 0
    for c in cells:
        if c["kind"] != "prompt":
            continue
        force = forced < a.force_confirm and not c["input_substitution"]
        forced += force
        if not (c["r"] >= R_REPAIR or c["delta"] > 0 or force):
            continue
        comp, L, dpos = c["comp"], c["layer"], c["donor_pos"]
        posmap = {dp + OFFSET: cap_d[comp][L][dp] for dp in dpos}     # prompt positions: causal, = prompt-only
        _, text = M.generate(RECIP_PROMPT, (comp, L, posmap))
        c["gen_text"], c["gen_class"], c["fallback"] = text, classify(r13, text), fallback_only(r13, text)
        # controls (teacher-forced): misaligned donor (shift -1 / +1) and a same-norm random direction
        ctrl = {}
        if len(dpos) == 1:           # positional specificity (descriptive): donor value from the neighbour
            for sft in (-1, 1):
                pm = {dp + OFFSET: cap_d[comp][L][dp + sft] for dp in dpos if 1 <= dp + sft < nd_prompt}
                ctrl[f"shift{sft:+d}"] = r_of(M.logits(r_ids, [(comp, L, pm)])) if pm else None
            shifts = [v for v in ctrl.values() if v is not None]
            c["position_specific"] = bool(shifts) and max(shifts) < c["r"] / 2
        pm = {}
        for dp in dpos:
            h = cap_r[comp][L][dp + OFFSET].float()
            n = (cap_d[comp][L][dp].float() - h).norm()
            u = torch.randn(h.shape, generator=gen)
            pm[dp + OFFSET] = h + n * (u / u.norm()).to(h.device)
        ctrl["random_dir"] = r_of(M.logits(r_ids, [(comp, L, pm)]))
        c["controls"] = ctrl
        c["control_max_r"] = ctrl["random_dir"]          # the repair veto (random direction only)
        # reverse (necessity, descriptive): buggy values into the clean run at the mirrored positions
        pm = {dp: cap_r[comp][L][dp + OFFSET] for dp in dpos}
        d_rev = delta(M.logits(d_ids, [(comp, L, pm)]))
        c["reverse_r"] = (d_rev - d_c) / (d_b - d_c)
        c["repair"] = (c["r"] >= R_REPAIR and c["gen_class"] == "correct" and c["control_max_r"] < R_CONTROL)
    gates["hooks_removed"] = {"ok": bool(torch.equal(M.logits(r_ids), lg_r))}
    run["cells"] = cells
    run["decision"] = decide(cells, M.n_layers)
    run["all_gates_ok"] = all(v["ok"] for v in gates.values())
    run["finished"] = datetime.now().isoformat()
    dump(run, path)
    print("decision:", json.dumps(run["decision"]), flush=True)
    print("all gates ok:", run["all_gates_ok"], flush=True)
    if not a.smoke and not run["all_gates_ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
