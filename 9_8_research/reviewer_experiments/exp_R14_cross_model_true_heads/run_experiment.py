#!/usr/bin/env python3
"""
Experiment R14: true per-head patching in Pythia-160M and Gemma-2B (base, IT)
=============================================================================
Experiment file: docs/experiments/11-22_pythia_gemma_true_head_patching.md
Design review round 1: docs/reviews/11-22_pythia_gemma_true_head_patching_design.md

Same two bugs as R13 in the original cross-model scripts:
  A  forward hook on attention output = post-W_O (post-dense / post-o_proj) residual blocks
  B  hook left on through generate(): decode steps get donor position 0
Plus: the original donor prompt was itself wrong (bug_rate 1.0), so stage 1 screens prompt formats
for a (correct donor, buggy recipient) pair with a fixed mechanical rule.

  --model pythia160m | gemma2b | gemma2b_it      main experiment, one model per invocation
  --model pythia160m --dynamics --main-run DIR   training checkpoints
  --selftest                                     classifier + summary/rule assertions (CPU, no model)
Greedy decoding is deterministic: one run per configuration.
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime

MAX_NEW = 20

MODELS = {
    "pythia160m": {"repo": "EleutherAI/pythia-160m", "layer": 6, "orig": "pythia", "dtype": "float16"},
    "gemma2b": {"repo": "google/gemma-2b", "layer": 8, "orig": "gemma", "dtype": "float16"},
    "gemma2b_it": {"repo": "google/gemma-2b-it", "layer": 8, "orig": None, "dtype": "bfloat16"},
}
CHECKPOINTS = ["step1000", "step2000", "step4000", "step8000", "step16000", "step32000",
               "step64000", "step80000", "step100000", "step120000", "step143000"]
FULL_SUBSET_CHECKPOINTS = ["step120000", "step143000"]

FORMATS = [
    ("F1", "Which is bigger: 9.8 or 9.11?"),
    ("F2", "Q: Which is bigger: 9.8 or 9.11?\nA:"),
    ("F3", "Which is bigger: 9.8 or 9.11?\nAnswer:"),
    ("F4", "Compare 9.8 and 9.11. Which is larger?"),
    ("F5", "9.8 vs 9.11 - which is greater?"),
    ("F6", "Between 9.8 and 9.11, which is greater?"),
]
ORIG_DONOR, ORIG_RECIP = FORMATS[0][1], FORMATS[1][1]

# Original configurations and outcomes (rate >= 0.5 -> fixed), from
# cross_model_validation/pythia_160m_validation_20250926_174151.json and
# cross_model_validation/gemma_2b_validation_20250926_174510.json
ORIG_REFS = {
    "pythia": {
        "layer": 6,
        "configs": {"all_even": ([0, 2, 4, 6, 8, 10], 6, True), "all_odd": ([1, 3, 5, 7, 9, 11], 6, False),
                    "first_half_even": ([0, 2, 4], 6, False), "second_half_even": ([6, 8, 10], 6, True),
                    **{f"sweep_L{L}": ([0, 2, 4, 6, 8, 10], L, L == 6) for L in range(6, 12)}},
        "must": ["all_even", "all_odd"],
    },
    "gemma": {
        "layer": 8,
        "configs": {"all_even_4": ([0, 2, 4, 6], 8, True), "all_odd_4": ([1, 3, 5, 7], 8, True),
                    "even_2": ([0, 2], 8, False), "even_3": ([0, 2, 4], 8, False),
                    "mixed_4": ([0, 1, 4, 5], 8, True),
                    **{f"sweep_L{L}": ([0, 2, 4, 6], L, L in (8, 10)) for L in range(8, 18)}},
        "must": ["all_even_4", "all_odd_4"],
    },
}
# training_dynamics_analysis/results/pythia_training_dynamics_20250926_191344.json: (baseline, even, odd)
TD_REF = {"step1000": (0, 0, 0), "step2000": (0, 0, 0), "step4000": (0, 0, 0), "step8000": (0, 0, 0),
          "step16000": (0, 0, 0), "step32000": (1, 1, 1), "step64000": (1, 1, 1), "step80000": (0, 0, 0),
          "step100000": (0, 0, 0), "step120000": (0, 0, 0), "step143000": (0, 1, 0)}

# ---------------------------------------------------------------- classifiers

# phrases: R13's list (+ "the greater"), with number boundaries; 9.80, 9.800 count as 9.8
_CMP = r"\s+is\s+(?:the\s+)?(?:bigger|larger|greater)"
RE_CORRECT_PHRASE = re.compile(r"(?<![\d.])9\.80*(?!\d)" + _CMP)
RE_BUG_PHRASE = re.compile(r"(?<![\d.])9\.11(?!\d)" + _CMP)
RE98 = re.compile(r"(?<![\d.])9\.80*(?!\d)")
RE911 = re.compile(r"(?<![\d.])9\.11(?!\d)")


def _clean(text):
    """Strip Markdown, then cut at the first line starting with 'Q:' (a turn the model invents)."""
    out = []
    for line in re.sub(r"[*`_]", "", text).split("\n"):
        if line.strip().startswith("Q:"):
            break
        out.append(line)
    return "\n".join(out)


def answer_line(text):
    """First non-empty line naming exactly one of 9.8 / 9.11 -> that number. Scanning stops (None)
    at a line naming both or starting with 'Q:', so a repeated question never counts."""
    for line in _clean(text).split("\n"):
        s = line.strip()
        if not s:
            continue
        a, b = bool(RE98.search(s)), bool(RE911.search(s))
        if a and b:
            return None
        if a:
            return "9.8"
        if b:
            return "9.11"
    return None


def is_correct(text):
    """R14 'correct': a correct phrase and no bug phrase; else, with no phrase, the answer line is 9.8."""
    lo = _clean(text).lower()
    c, b = bool(RE_CORRECT_PHRASE.search(lo)), bool(RE_BUG_PHRASE.search(lo))
    if c or b:
        return c and not b
    return answer_line(text) == "9.8"


def is_bug(text):
    """R14 'bug': a bug phrase and not correct; else, with no phrase, the answer line is 9.11."""
    if is_correct(text):
        return False
    lo = _clean(text).lower()
    if RE_BUG_PHRASE.search(lo):
        return True
    if RE_CORRECT_PHRASE.search(lo):
        return False
    return answer_line(text) == "9.11"


def classify(text):
    return "correct" if is_correct(text) else ("bug" if is_bug(text) else "neither")


def orig_fixed(output):
    """check_bug_fixed from cross_model_validation/test_pythia_160m.py:145-168, verbatim."""
    output_lower = output.lower()
    correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8 is greater",
                        "9.8 is more", "9.8", "9.8 is the bigger"]
    bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11 is greater",
                    "9.11 is more", "9.11 is the bigger"]
    has_correct = any(pattern in output_lower for pattern in correct_patterns)
    has_bug = any(pattern in output_lower for pattern in bug_patterns)
    if not has_correct and not has_bug:
        first_line = output.strip().split('\n')[0] if output.strip() else ""
        if '9.8' in first_line and '9.11' not in first_line:
            has_correct = True
    return has_correct and not has_bug


def td_fixed(output):
    """check_bug_fixed from test_pythia_training_dynamics.py, verbatim."""
    output_lower = output.lower()
    correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
    bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]
    has_correct = any(pattern in output_lower for pattern in correct_patterns)
    has_bug = any(pattern in output_lower for pattern in bug_patterns)
    return has_correct and not has_bug


# ---------------------------------------------------------------- summaries and rules (pure)

def key(heads):
    return "h" + "_".join(map(str, heads))


def all_subsets(n):
    for k in range(1, n + 1):
        for c in itertools.combinations(range(n), k):
            yield list(c)


def mask(heads):
    m = 0
    for h in heads:
        m |= 1 << h
    return m


def summarize(done, n):
    """done: {key(heads): {"heads", "correct", "bug"}} over all 2^n - 1 subsets."""
    even = [h for h in range(n) if h % 2 == 0]
    odd = [h for h in range(n) if h % 2 == 1]
    get = lambda heads: done[key(heads)]["correct"]
    cls = lambda v: "correct" if v["correct"] else ("bug" if v["bug"] else "neither")
    s = {"all_heads": get(list(range(n))), "all_even": get(even), "all_odd": get(odd),
         "all_odd_class": cls(done[key(odd)]),
         "singles_correct": [h for h in range(n) if get([h])]}
    by_k = {}
    for v in done.values():
        d = by_k.setdefault(str(len(v["heads"])), {"correct": 0, "bug": 0, "neither": 0})
        d[cls(v)] += 1
    s["class_by_size"] = by_k
    # even-only vs odd-only subsets (S -> S+1 is a size-preserving bijection, so this is a count comparison)
    ev = sum(get(list(S)) for k in range(1, len(even) + 1) for S in itertools.combinations(even, k))
    od = sum(get(list(S)) for k in range(1, len(odd) + 1) for S in itertools.combinations(odd, k))
    s["even_only_correct"], s["odd_only_correct"], s["n_per_side"] = ev, od, 2 ** len(even) - 1
    # descriptive: even-majority vs odd-majority correct rates
    me, mo = [0, 0], [0, 0]
    for v in done.values():
        ne = sum(h % 2 == 0 for h in v["heads"])
        no = len(v["heads"]) - ne
        if ne != no:
            t = me if ne > no else mo
            t[0] += v["correct"]
            t[1] += 1
    s["more_even"], s["more_odd"] = me, mo
    s["majority_rate_gap"] = me[0] / me[1] - mo[0] / mo[1]
    # rule (d): balanced bipartitions. D(A,B) = |#correct subsets within A - #correct within B|
    full = (1 << n) - 1
    cm = [mask(v["heads"]) for v in done.values() if v["correct"]]
    em, contig = mask(even), mask(range(n // 2))

    def D(a):
        b = full ^ a
        return sum((m & ~a) == 0 for m in cm) - sum((m & ~b) == 0 for m in cm)

    splits = [mask(A) for A in itertools.combinations(range(n), n // 2) if 0 in A]
    Ds = [abs(D(a)) for a in splits]
    d_par_signed = D(em)
    d_par = abs(d_par_signed)
    d_con = abs(D(contig))
    rank_frac = sum(x >= d_par for x in Ds) / len(Ds)
    top = sorted(zip(Ds, splits), key=lambda t: -t[0])[:5]
    s["bipartition"] = {"n_splits": len(Ds), "D_even_minus_odd": d_par_signed, "D_contiguous": d_con,
                        "max_D": max(Ds), "frac_splits_D_ge_parity": rank_frac,
                        "top5": [[dv, [h for h in range(n) if a >> h & 1]] for dv, a in top]}
    # minimal sufficient sets
    cset = set(cm)
    minimal = []
    for m in cm:
        sub, ok = (m - 1) & m, True
        while sub:
            if sub in cset:
                ok = False
                break
            sub = (sub - 1) & m
        if ok:
            minimal.append([h for h in range(n) if m >> h & 1])
    s["minimal_sufficient_sets"] = sorted(minimal, key=lambda x: (len(x), x))
    s["necessary_heads"] = sorted(set.intersection(*map(set, minimal))) if minimal else []
    s["n_correct"], s["n_configs"] = len(cm), len(done)
    return s


def rules(s, family):
    """Pre-registered rules on one summary. family: 'pythia' or 'gemma'."""
    r = {"repair": bool(s["all_heads"])}
    b = s["bipartition"]
    r["parity_a_all_even"] = bool(s["all_even"])
    r["parity_b_all_odd_not"] = not s["all_odd"]
    r["parity_c_even_only_gt_odd_only"] = s["even_only_correct"] > s["odd_only_correct"]
    r["parity_d_split_specific"] = (b["D_even_minus_odd"] > 0 and abs(b["D_even_minus_odd"]) > b["D_contiguous"]
                                    and b["frac_splits_D_ge_parity"] <= 0.05
                                    and b["max_D"] == abs(b["D_even_minus_odd"]))   # no split scores higher
    r["parity_holds"] = all(r[k] for k in ("parity_a_all_even", "parity_b_all_odd_not",
                                           "parity_c_even_only_gt_odd_only", "parity_d_split_specific"))
    if family == "gemma":
        r["gemma_parity_independent"] = bool(s["all_even"]) and bool(s["all_odd"])
        r["pythia_style_parity_descriptive"] = parity_verdict(s, r)   # Gemma's rule is the one above
    else:
        r["verdict"] = parity_verdict(s, r)
    return r


def parity_verdict(s, r):
    """Pre-registered wording for the parity outcome. First matching branch, in this order."""
    b, nec = s["bipartition"], s["necessary_heads"]
    even_set = {h for h in range(int(max(s["class_by_size"], key=int))) if h % 2 == 0}
    if not r["repair"]:
        return "repair fails at this layer; head rules not interpreted"
    if r["parity_holds"]:
        return "parity holds" + (f", with heads {nec} necessary" if nec and set(nec) != even_set else "")
    abc = r["parity_a_all_even"] and r["parity_b_all_odd_not"] and r["parity_c_even_only_gt_odd_only"]
    if not abc:
        return "parity not supported"
    if abs(b["D_even_minus_odd"]) <= 3:
        return "inconclusive: parity evidence rests on 3 or fewer subsets"
    if nec and set(nec) < even_set:
        return f"even-only subsets repair, but the effect is carried by specific heads ({nec}), not by parity"
    top = b["top5"][0]
    return (f"even-only subsets repair, but parity is not uniquely the most specific split "
            f"(top split {top[1]} with |D| = {top[0]} vs even/odd {abs(b['D_even_minus_odd'])})")


# ---------------------------------------------------------------- model code (imports torch lazily)

def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Patcher:
    def __init__(self, repo, revision="main", dtype="float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {repo}@{revision} on {self.device} ...", flush=True)
        self.repo, self.revision = repo, revision
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dt = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=dt, device_map="auto" if self.device == "cuda" else None)
        self.model.eval()
        cfg = self.model.config
        self.commit = getattr(cfg, "_commit_hash", None)
        if not self.commit and not os.path.isdir(repo):
            from huggingface_hub import snapshot_download
            self.commit = os.path.basename(snapshot_download(repo, revision=revision, local_files_only=True))
        self.attn_impl = getattr(cfg, "_attn_implementation", None)
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        if hasattr(self.model, "gpt_neox"):
            self.layers, self.attn_name, self.proj_name = self.model.gpt_neox.layers, "attention", "dense"
        else:
            self.layers, self.attn_name, self.proj_name = self.model.model.layers, "self_attn", "o_proj"
        self.n_layers = len(self.layers)
        for L in range(self.n_layers):
            assert self.proj(L).in_features == self.n_heads * self.head_dim
        ids = self.tok("x")["input_ids"]
        self.adds_bos = self.tok.bos_token_id is not None and ids[0] == self.tok.bos_token_id
        self.last_shapes = []

    def attn(self, L):
        return getattr(self.layers[L], self.attn_name)

    def proj(self, L):
        return getattr(self.attn(L), self.proj_name)

    def encode(self, prompt):
        return self.tok(prompt, return_tensors="pt").to(self.device)

    def ids(self, prompt):
        return self.tok(prompt)["input_ids"]

    def capture(self, prompt, hook, L):
        store = {}
        if hook == "post_wo":
            def fn(mod, inp, out):
                store["x"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            handle = self.attn(L).register_forward_hook(fn)
        else:
            def fn(mod, args):
                store["x"] = args[0].detach().clone()
            handle = self.proj(L).register_forward_pre_hook(fn)
        try:
            with self.torch.no_grad():
                self.model(**self.encode(prompt))
        finally:
            handle.remove()
        return store["x"]

    @staticmethod
    def aligned_positions(ts, sl, align):
        """(donor_pos, recipient_pos) pairs that get patched. Right alignment never patches position 0
        (BOS for Gemma, the first token / attention sink for Pythia)."""
        if align == "right":
            n = min(sl, ts) - 1
            return [(ts - n + j, sl - n + j) for j in range(max(n, 0))]
        return [(i, i) for i in range(min(sl, ts))]

    def _swap(self, h, saved, heads, align):
        # block width = d / n_heads: head_dim on the o_proj input (pre_wo); on the post-W_O
        # output it is hidden_size / n_heads, the original scripts' slicing (equal for Gemma-2B)
        bs, sl, d = h.shape
        bw = d // self.n_heads
        hr = h.view(bs, sl, self.n_heads, bw).clone()
        sr = saved.view(saved.shape[0], -1, self.n_heads, bw).to(hr.dtype)
        ts = sr.shape[1]
        idx = self.torch.tensor(heads, device=h.device, dtype=self.torch.long)
        if align == "right":
            n = min(sl, ts) - 1
            if n > 0:
                hr[:, sl - n:, idx, :] = sr[:, ts - n:, idx, :]
        else:
            ml = min(sl, ts)
            hr[:, :ml, idx, :] = sr[:, :ml, idx, :]
        return hr.view(bs, sl, d)

    def generate(self, prompt, saved=None, heads=None, hook=None, scope=None, L=None, align="left"):
        handle = None
        self.last_shapes = []
        if saved is not None and heads:
            prefill_only = scope == "prefill"

            def active(h):
                self.last_shapes.append(int(h.shape[1]))
                return not prefill_only or len(self.last_shapes) == 1

            if hook == "post_wo":
                def fn(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if not active(h):
                        return None
                    new = self._swap(h, saved, heads, align)
                    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
                handle = self.attn(L).register_forward_hook(fn)
            else:
                def fn(mod, args):
                    if not active(args[0]):
                        return None
                    return (self._swap(args[0], saved, heads, align),) + tuple(args[1:])
                handle = self.proj(L).register_forward_pre_hook(fn)
        try:
            ids = self.encode(prompt)
            with self.torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def alignment_info(p, donor, recip, align):
    """Aligned token pairs and n_effective = patched pairs whose activations can differ
    (a pair at the same absolute position with identical token prefixes is a causal no-op)."""
    di, ri = p.ids(donor), p.ids(recip)
    pairs = p.aligned_positions(len(di), len(ri), align)
    toks = lambda ids_, i: p.tok.convert_ids_to_tokens([ids_[i]])[0]
    eff = [(a, b) for a, b in pairs if not (a == b and di[:a + 1] == ri[:b + 1])]
    return {"n_donor": len(di), "n_recipient": len(ri), "n_patched": len(pairs), "n_effective": len(eff),
            "pairs": [[a, b, toks(di, a), toks(ri, b)] for a, b in pairs]}


def formats_for(p, key_):
    fm = list(FORMATS)
    if key_ == "gemma2b_it":
        chat = p.tok.apply_chat_template([{"role": "user", "content": FORMATS[0][1]}],
                                         tokenize=False, add_generation_prompt=True)
        if p.tok.bos_token and chat.startswith(p.tok.bos_token):
            chat = chat[len(p.tok.bos_token):]   # tokenizer adds BOS itself
        fm.append(("F7", chat))
    return fm


def screen(p, fm):
    out = {}
    for name, prompt in fm:
        text = p.generate(prompt)
        out[name] = {"prompt": prompt, "output": text, "class": classify(text),
                     "orig_fixed": orig_fixed(text), "td_fixed": td_fixed(text),
                     "n_tokens": len(p.ids(prompt))}
    recip = "F2" if out["F2"]["class"] == "bug" else next(
        (n for n, _ in fm if out[n]["class"] == "bug"), None)
    donor = next((n for n, _ in fm if out[n]["class"] == "correct"), None)
    return out, donor, recip


def head_isolation(p, donor, recip, L):
    x = p.capture(recip, "pre_wo", L)
    y = p.capture(donor, "pre_wo", L)
    h = 1
    out = p._swap(x, y, [h], "right").view(x.shape[0], x.shape[1], p.n_heads, p.head_dim)
    diff = (out - x.view_as(out)).abs().sum(dim=(0, 1, 3))
    changed = [int(i) for i in p.torch.nonzero(diff).flatten()]
    return {"head": h, "changed_heads": changed, "ok": changed == [h]}


def prefill_gates(p, donor, recip, L, align):
    """Identity, determinism, hook shapes for a prompt-only pre_wo cell (R13 gates)."""
    allh = list(range(p.n_heads))
    unpatched = p.generate(recip)
    own = p.capture(recip, "pre_wo", L)
    ident = p.generate(recip, own, allh, "pre_wo", "prefill", L, align)
    saved = p.capture(donor, "pre_wo", L)
    det = [p.generate(recip, saved, allh, "pre_wo", "prefill", L, align) for _ in range(3)]
    shapes = list(p.last_shapes)
    t = len(p.ids(recip))
    g = {"identity_ok": ident == unpatched, "identity_output": ident, "unpatched": unpatched,
         "determinism_ok": len(set(det)) == 1, "determinism_outputs": det,
         "hook_call_shapes": shapes, "hook_shapes_ok": shapes[0] == t and all(s == 1 for s in shapes[1:])}
    g["ok"] = g["identity_ok"] and g["determinism_ok"] and g["hook_shapes_ok"]
    return g, saved


def run_subsets(p, saved, recip, L, align, tag, path, limit=None):
    subs = list(all_subsets(p.n_heads))
    if limit:
        n = p.n_heads
        subs = [list(range(n)), list(range(0, n, 2)), list(range(1, n, 2))] + [[h] for h in range(min(limit, n))]
    done, t0 = {}, time.time()
    for i, heads in enumerate(subs):
        text = p.generate(recip, saved, heads, "pre_wo", "prefill", L, align)
        done[key(heads)] = {"heads": heads, "correct": is_correct(text), "bug": is_bug(text), "output": text}
        if (i + 1) % 500 == 0:
            print(f"[{tag}] {i+1}/{len(subs)}  {time.time()-t0:.0f}s", flush=True)
            dump({"tag": tag, "partial": True, "configs": done}, path)
    return done


def orig_cell(p, refkey):
    """Old protocol: post-W_O forward hook, all decode steps, left-aligned, F1 -> F2, orig_fixed."""
    ref = ORIG_REFS[refkey]
    n, L0 = p.n_heads, ref["layer"]
    cfgs = dict(ref["configs"])
    cfgs["x_all"] = (list(range(n)), L0, None)
    cfgs.update({f"x_single_{h}": ([h], L0, None) for h in range(n)})
    saved_by_L, rows = {}, {}
    for name, (heads, L, want) in cfgs.items():
        if L not in saved_by_L:
            saved_by_L[L] = p.capture(ORIG_DONOR, "post_wo", L)
        text = p.generate(ORIG_RECIP, saved_by_L[L], heads, "post_wo", "all_steps", L, "left")
        rows[name] = {"heads": heads, "layer": L, "output": text, "orig_fixed": orig_fixed(text),
                      "class": classify(text), "ref": want}
    refs = [k for k in rows if rows[k]["ref"] is not None]
    agree = [k for k in refs if rows[k]["orig_fixed"] == rows[k]["ref"]]
    gate = {"n_ref": len(refs), "agree": len(agree), "disagree": sorted(set(refs) - set(agree)),
            "must": {k: rows[k]["orig_fixed"] == rows[k]["ref"] for k in ref["must"]}}
    gate["ok"] = len(agree) >= 0.9 * len(refs) and all(gate["must"].values())
    return rows, gate


def provenance(p, extra):
    import torch
    import transformers
    info = {"hostname": socket.gethostname(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "git_sha": os.environ["R14_GIT_SHA"], "script_sha256": sha256(__file__),
            "repo": p.repo, "revision": p.revision, "commit": p.commit, "attn_implementation": p.attn_impl,
            "n_heads": p.n_heads, "head_dim": p.head_dim, "n_layers": p.n_layers, "tokenizer_adds_bos": p.adds_bos,
            "dtype": str(next(p.model.parameters()).dtype), "command": " ".join(sys.argv), **extra}
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        info["gpu"] = "no nvidia-smi"
    return info


def subset_cell(p, run, out_dir, tag, dp, rp, L, align, family, limit, save):
    g, saved = prefill_gates(p, dp, rp, L, align)
    print(f"[{tag}] gates: identity={g['identity_ok']} determinism={g['determinism_ok']} "
          f"shapes={g['hook_shapes_ok']}", flush=True)
    if not g["ok"]:
        run[tag] = {"gates": g, "aborted": True}
        save()
        sys.exit(f"FATAL: [{tag}] gate failed (pre-registered stop)")
    path = os.path.join(out_dir, f"results_{tag}.json")
    done = run_subsets(p, saved, rp, L, align, tag, path, limit)
    dump({"tag": tag, "partial": False, "configs": done}, path)   # outputs safe before any analysis
    summ = summarize(done, p.n_heads) if not limit else None
    rl = rules(summ, family) if summ else None
    dump({"tag": tag, "layer": L, "align": align, "gates": g, "summary": summ, "rules": rl,
          "configs": done}, path)
    run[tag] = {"layer": L, "align": align, "gates": g, "summary": summ, "rules": rl}
    print(f"[{tag}] rules: {json.dumps(rl)}", flush=True)
    if summ:
        brief = {k: summ[k] for k in ("all_heads", "all_even", "all_odd", "all_odd_class", "singles_correct",
                                      "even_only_correct", "odd_only_correct", "n_correct", "bipartition")}
        print(f"[{tag}] summary: {json.dumps(brief)}", flush=True)
    save()
    return summ


def main_model(a, out_dir):
    k = a.model
    spec = MODELS[k]
    L = spec["layer"]
    family = "pythia" if k.startswith("pythia") else "gemma"
    p = Patcher(a.repo or spec["repo"], dtype=spec["dtype"])
    run = {"provenance": provenance(p, {"model_key": k}), "started": datetime.now().isoformat()}
    path = os.path.join(out_dir, "run.json")

    def save():
        dump(run, path)

    fm = formats_for(p, k)
    sc, donor, recip = screen(p, fm)
    if a.force_pair:   # smoke tests only: random weights never give a (correct, bug) pair
        donor, recip = a.force_pair.split(",")
    run["screen"], run["pair"] = sc, {"donor": donor, "recipient": recip}
    for n_, v in sc.items():
        print(f"[{k}] screen {n_} ({v['n_tokens']} tok) {v['class']}: {v['output']!r}", flush=True)
    print(f"[{k}] pair: {donor} -> {recip}", flush=True)
    save()

    if spec["orig"] and not a.skip_orig:
        rows, gate = orig_cell(p, spec["orig"])
        run["orig_cell"], run["reproduction_gate"] = rows, gate
        print(f"[{k}] reproduction gate: {json.dumps(gate)}", flush=True)
        save()

    if donor is None or recip is None:
        run["verdict_input"] = "no_phenomenon"
        run["finished"] = datetime.now().isoformat()
        save()
        print(f"[{k}] no (correct, bug) pair: corrected cells not run (pre-registered)", flush=True)
        return
    dp, rp = sc[donor]["prompt"], sc[recip]["prompt"]

    iso = head_isolation(p, dp, rp, L)
    run["head_isolation"] = iso
    print(f"[{k}] head isolation: {iso}", flush=True)
    if not iso["ok"]:
        save()
        sys.exit(f"FATAL: [{k}] head isolation failed")

    run["alignment"] = {al: alignment_info(p, dp, rp, al) for al in ("right", "left")}
    print(f"[{k}] n_effective: right={run['alignment']['right']['n_effective']} "
          f"left={run['alignment']['left']['n_effective']}", flush=True)

    allh = list(range(p.n_heads))
    t_pre = p.generate(rp, p.capture(dp, "pre_wo", L), allh, "pre_wo", "prefill", L, "right")
    t_post = p.generate(rp, p.capture(dp, "post_wo", L), allh, "post_wo", "prefill", L, "right")
    run["all_heads_equivalence"] = {"same_text": t_pre == t_post, "pre": t_pre, "post": t_post}
    print(f"[{k}] all-heads equivalence: {t_pre == t_post}  pre={t_pre!r} post={t_post!r}", flush=True)

    sweep = {}
    for LL in range(p.n_layers):
        text = p.generate(rp, p.capture(dp, "pre_wo", LL), allh, "pre_wo", "prefill", LL, "right")
        sweep[str(LL)] = {"output": text, "class": classify(text), "correct": is_correct(text)}
    run["layer_sweep_all_heads_right"] = sweep
    good = [int(x) for x, v in sweep.items() if v["correct"]]
    print(f"[{k}] layer sweep (all heads, right) correct at: {good}", flush=True)
    save()

    s_right = subset_cell(p, run, out_dir, "pre_wo__prefill_right", dp, rp, L, "right", family, a.limit, save)
    if run["alignment"]["left"]["n_effective"] == 0:
        run["pre_wo__prefill"] = {"vacuous": True, "reason": "left alignment patches only a shared prefix"}
        print(f"[{k}] pre_wo__prefill: vacuous (n_effective = 0), not run", flush=True)
    else:
        subset_cell(p, run, out_dir, "pre_wo__prefill", dp, rp, L, "left", family, a.limit, save)

    # pre-registered fallback: all heads fail at the original layer, but some layer repairs
    allh_ok = sweep[str(L)]["correct"]
    if not allh_ok and good:
        Lf = good[0]
        run["fallback_layer"] = Lf
        print(f"[{k}] fallback layer {Lf} (first layer where all heads repair)", flush=True)
        subset_cell(p, run, out_dir, f"pre_wo__prefill_right__L{Lf}", dp, rp, Lf, "right", family, a.limit, save)
    run["finished"] = datetime.now().isoformat()
    save()


def late_training_verdict(c143, c120):
    """Pre-registered (review round 2, fix 2). c143/c120: checkpoint records."""
    def ok(c):
        return c and c.get("phenomenon_present") and c.get("gates", {}).get("ok") and c.get("rules")
    if not ok(c143):
        return "untestable: no interpretable full-subset run at step143000"
    caveat = " (not interpreted: step143000 differs from the main run; INBOX)" \
        if c143.get("matches_main_run") is False else ""
    if not c143["rules"]["parity_holds"]:
        return "not supported: parity does not hold at step143000" + caveat
    if not c120 or "error" in c120:
        return "untestable: step120000 errored or did not run" + caveat
    if not c120.get("phenomenon_present"):
        return "untestable: no correct/bug pair at step120000" + caveat
    if not ok(c120):
        return "untestable: step120000 gates failed" + caveat
    if not c120["rules"]["repair"]:
        return "repair itself appears late; parity emergence not separable" + caveat
    if c120["rules"]["parity_holds"]:
        return "refuted: parity already holds at step120000" + caveat
    return "supported: parity holds at step143000, not at step120000" + caveat


def dynamics(a, out_dir):
    """Pythia checkpoints: screen; old protocol even/odd (td_fixed); true heads on the main pair.
    The two checkpoints the late-training claim rests on run first; each checkpoint is isolated."""
    import torch
    L = MODELS["pythia160m"]["layer"]
    try:
        with open(os.path.join(a.main_run, "run.json")) as f:
            main_run = json.load(f)
        pr = main_run["pair"]
        main_pair = (pr["donor"], pr["recipient"])
    except Exception as e:   # main run died early: still run screens and the old-protocol reproduction
        print(f"[dyn] main run unreadable ({e}); true-head runs skipped", flush=True)
        main_run, main_pair = {"provenance": {"commit": None}}, (None, None)
    main_right = os.path.join(a.main_run, "results_pre_wo__prefill_right.json")
    main_cfgs = {}
    if os.path.exists(main_right):
        with open(main_right) as f:
            mr = json.load(f)
        if mr.get("summary"):
            main_cfgs = mr["configs"]
    out = {"main_pair": main_pair, "main_commit": main_run["provenance"]["commit"], "checkpoints": {}}
    main_screen = {k_: v["output"] for k_, v in main_run.get("screen", {}).items()}
    path = os.path.join(out_dir, "dynamics.json")
    order = FULL_SUBSET_CHECKPOINTS[::-1] + [c for c in CHECKPOINTS if c not in FULL_SUBSET_CHECKPOINTS]
    cks = order[:2] if a.limit else order
    for ck in cks:
        r = {}
        out["checkpoints"][ck] = r   # filled in place, so a crash keeps whatever was recorded
        try:
            dynamics_checkpoint(a, out, r, ck, L, main_pair, main_cfgs, main_screen, out_dir)
        except Exception as e:  # one bad checkpoint must not take the others with it
            r["error"] = f"{type(e).__name__}: {e}"
            print(f"[dyn {ck}] ERROR {r['error']}", flush=True)
        out["late_training_verdict"] = late_training_verdict(out["checkpoints"].get("step143000"),
                                                             out["checkpoints"].get("step120000"))
        dump(out, path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[dyn] late-training verdict: {out['late_training_verdict']}", flush=True)


def dynamics_checkpoint(a, out, r, ck, L, main_pair, main_cfgs, main_screen, out_dir):
    p = Patcher(a.repo or MODELS["pythia160m"]["repo"], revision=ck, dtype="float16")
    if "provenance" not in out:
        out["provenance"] = provenance(p, {"mode": "dynamics"})
    n = p.n_heads
    even, odd = list(range(0, n, 2)), list(range(1, n, 2))
    r["matches_main_run"], r["matches_main_run_reason"] = None, \
        ("not step143000" if ck != "step143000" else "not reached")
    sc, donor, recip = screen(p, FORMATS)
    r.update({"commit": p.commit, "screen": {x: v["class"] for x, v in sc.items()},
              "screen_outputs": {x: v["output"] for x, v in sc.items()}, "own_pair": [donor, recip]})
    if ck == "step143000":   # always available, even with no phenomenon
        r["screen_matches_main_run"] = (main_screen == r["screen_outputs"]) if main_screen else None
    base = p.generate(ORIG_RECIP)
    sv = p.capture(ORIG_DONOR, "post_wo", L)
    old = {nm: p.generate(ORIG_RECIP, sv, h, "post_wo", "all_steps", L, "left")
           for nm, h in (("even", even), ("odd", odd))}
    r["old_protocol"] = {"baseline": td_fixed(base), "even": td_fixed(old["even"]),
                         "odd": td_fixed(old["odd"]), "outputs": {"baseline": base, **old}, "ref": TD_REF[ck]}
    r["old_protocol"]["agrees_with_ref"] = [int(r["old_protocol"][x]) for x in ("baseline", "even", "odd")] \
        == list(TD_REF[ck])
    is143 = ck == "step143000"
    if not (main_pair[0] and main_pair[1]):
        if is143:
            r["matches_main_run_reason"] = "main run has no pair"
    else:
        dp, rp = dict(FORMATS)[main_pair[0]], dict(FORMATS)[main_pair[1]]
        r["alignment_right"] = alignment_info(p, dp, rp, "right")
        dcls, rcls = classify(p.generate(dp)), classify(p.generate(rp))
        r["main_pair_baseline"] = {"donor": dcls, "recipient": rcls}
        r["phenomenon_present"] = dcls == "correct" and rcls == "bug"
        g, saved = prefill_gates(p, dp, rp, L, "right")
        r["gates"] = g
        if not g["ok"]:
            print(f"[dyn {ck}] GATE FAILED: checkpoint not interpreted", flush=True)
            if is143:
                r["matches_main_run_reason"] = "gates failed"
        if r["phenomenon_present"] and g["ok"]:
            cfg = {"all": list(range(n)), "even": even, "odd": odd, **{f"single_{h}": [h] for h in range(n)}}
            r["true_heads"] = {}
            for nm, h in cfg.items():
                t = p.generate(rp, saved, h, "pre_wo", "prefill", L, "right")
                r["true_heads"][nm] = {"heads": h, "correct": is_correct(t), "output": t}
            done = None
            if ck in FULL_SUBSET_CHECKPOINTS:
                sp = os.path.join(out_dir, f"subsets_{ck}.json")
                done = run_subsets(p, saved, rp, L, "right", f"dyn {ck}", sp, a.limit)
                dump({"checkpoint": ck, "partial": False, "configs": done}, sp)
                summ = summarize(done, n) if not a.limit else None
                dump({"checkpoint": ck, "summary": summ, "configs": done}, sp)
                r["summary"], r["rules"] = summ, (rules(summ, "pythia") if summ else None)
            if ck == "step143000":
                if not main_cfgs:
                    r["matches_main_run_reason"] = "main run has no complete right-aligned results"
                else:
                    cmp = done if (done and r.get("summary")) else \
                        {key(v["heads"]): v for v in r["true_heads"].values()}
                    common = [k_ for k_ in cmp if k_ in main_cfgs]
                    r["matches_main_run"] = bool(common) and all(cmp[k_]["output"] == main_cfgs[k_]["output"]
                                                                  for k_ in common)
                    r["matches_main_run_reason"] = f"compared {len(common)} configs"
                    if not r["matches_main_run"]:
                        print(f"[dyn {ck}] MISMATCH with main run: 143k verdict not interpreted (pre-registered)",
                              flush=True)
        elif not r["phenomenon_present"] and is143:
            r["matches_main_run_reason"] = "no phenomenon at this checkpoint"
    print(f"[dyn {ck}] commit {p.commit} screen {r['screen']} "
          f"old {[r['old_protocol'][x] for x in ('baseline','even','odd')]} ref {TD_REF[ck]} "
          f"phenomenon={r.get('phenomenon_present')} gates={r.get('gates', {}).get('ok')} "
          f"verdict={(r.get('rules') or {}).get('verdict')} matches_main={r['matches_main_run']} "
          f"({r['matches_main_run_reason']})", flush=True)
    del p


# ---------------------------------------------------------------- self-test

def selftest():
    cases = [(" 9.8 is bigger than 9.11.", "correct"), (" 9.11 is bigger than 9.8.", "bug"),
             ("\n\nI think the answer is:\n\n9.11\n\nA:\n\nThe answer is:", "bug"),
             ("I think the answer is:\n\n9.8", "correct"), ("I think the answer is 9.11.\n\nA:", "bug"),
             ("\n9.11\nIs -1/2 at most -1.1", "bug"), ("The answer is:\n\nThe answer is:", "neither"),
             ("9.8\n\nQ: Which is bigger: 9.8 or 9.1", "correct"), ("Q: Which is bigger: 9.8 or 9.11?\n9.11", "neither"),
             ("**9.11** is bigger than 9.8", "bug"), ("**9.8** is bigger than **9.11**.", "correct"),
             ("Comparing 9.8 and 9.11:\n9.8 wins", "neither"), ("\n-0.8\nWhat is 7.80?", "neither"),
             ("9.80 is bigger than 9.11", "correct"), ("The answer is 9.80.", "correct"),
             (" 9.8\n\nQ: Which is bigger: 9.11 or 9.12?\nA: 9.11 is bigger", "correct"),
             ("19.8 is bigger than 9.11", "bug"), ("29.11 is bigger than 3", "neither")]
    for t, want in cases:
        got = classify(t)
        assert got == want, (t, got, want)

    def synth(n, f):
        return {key(h): {"heads": h, "correct": f(h), "bug": not f(h)} for h in all_subsets(n)}

    def verdict(n, f, fam):
        s = summarize(synth(n, f), n)
        return s, rules(s, fam)

    truths = 0
    E = set(range(0, 12, 2))
    truths += 1
    s, r = verdict(12, lambda h: E <= set(h), "pythia")          # needs all six even heads
    assert r["parity_holds"], s["bipartition"]
    assert r["verdict"] == "parity holds", r["verdict"]
    truths += 1
    s, r = verdict(12, lambda h: 8 in h, "pythia")                # head 8 alone
    assert not r["parity_holds"] and r["parity_c_even_only_gt_odd_only"], s["bipartition"]
    assert r["verdict"].startswith("even-only subsets repair, but the effect is carried by specific heads"), r["verdict"]
    truths += 1
    s, r = verdict(12, lambda h: 8 in h and any(x % 2 == 0 and x != 8 for x in h), "pythia")
    assert r["verdict"] == "parity holds, with heads [8] necessary", r["verdict"]
    truths += 1
    s, r = verdict(12, lambda h: {6, 8, 10} <= set(h), "pythia")  # late even heads
    assert not r["parity_holds"], s["bipartition"]
    assert s["minimal_sufficient_sets"] == [[6, 8, 10]]
    truths += 1
    s, r = verdict(12, lambda h: sum(x % 2 == 0 for x in h) >= 2, "pythia")  # any two even heads
    assert r["parity_holds"], s["bipartition"]
    truths += 1
    s, r = verdict(12, lambda h: len(h) >= 4, "pythia")           # count only
    assert not r["parity_holds"] and r["repair"]
    truths += 1
    s, r = verdict(8, lambda h: len(h) >= 4, "gemma")
    assert r["gemma_parity_independent"] and not r["parity_holds"]
    truths += 1
    s, r = verdict(8, lambda h: False, "gemma")
    assert not r["repair"] and not r["gemma_parity_independent"]
    # seeded output noise (review round 3): flip each outcome with prob 0.01, 40 seeds
    import random

    def noisy_passes(f, p_flip=0.01, seeds=40):
        hits = 0
        for seed in range(seeds):
            rnd = random.Random(seed)
            d = {}
            for h in all_subsets(12):
                c = f(h) != (rnd.random() < p_flip)
                d[key(h)] = {"heads": h, "correct": c, "bug": not c}
            hits += rules(summarize(d, 12), "pythia")["parity_holds"]
        return hits
    late = noisy_passes(lambda h: {6, 8, 10} <= set(h))
    true_par = noisy_passes(lambda h: sum(x % 2 == 0 for x in h) >= 3)
    assert late <= 2, ("late-even-heads false 'parity holds'", late)
    assert true_par >= 36, ("true parity power", true_par)
    ck = lambda pres, rep, par: {"phenomenon_present": pres, "gates": {"ok": True},
                                 "rules": {"repair": rep, "parity_holds": par}}
    lt = late_training_verdict
    assert lt(ck(True, True, True), ck(True, True, False)).startswith("supported")
    assert lt(ck(True, True, True), ck(True, True, True)).startswith("refuted")
    assert lt(ck(True, True, True), ck(True, False, False)).startswith("repair itself appears late")
    assert lt(ck(True, True, True), {"phenomenon_present": False}).startswith("untestable")
    assert lt(ck(True, True, False), ck(True, True, False)).startswith("not supported")
    assert lt({"error": "x"}, None).startswith("untestable")
    mm = ck(True, True, True)
    mm["matches_main_run"] = False
    assert lt(mm, ck(True, True, False)).endswith("(not interpreted: step143000 differs from the main run; INBOX)")
    assert lt(ck(True, True, True), {"error": "boom"}).startswith("untestable: step120000 errored")
    g = ck(True, True, False)
    g["gates"]["ok"] = False
    assert lt(ck(True, True, True), g).startswith("untestable: step120000 gates failed")
    print("selftest OK: classifier (%d cases), rules on %d synthetic truths, noise test "
          "({6,8,10}: %d/40 false passes; >=3 even: %d/40), late-training verdict (9 cases)"
          % (len(cases), truths, late, true_par))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--out", help="new directory (must not exist)")
    ap.add_argument("--dynamics", action="store_true", help="Pythia training checkpoints")
    ap.add_argument("--main-run", default=None, help="output dir of the pythia160m main run")
    ap.add_argument("--repo", default=None, help="override repo (CPU smoke tests only)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: few configs only")
    ap.add_argument("--force-pair", default=None, help="smoke test only: DONOR,RECIPIENT format names")
    ap.add_argument("--skip-orig", action="store_true", help="smoke test only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    if not (a.model and a.out):
        sys.exit("--model and --out are required")
    if (a.force_pair or a.skip_orig or a.repo) and not a.limit:
        sys.exit("--force-pair/--skip-orig/--repo are for smoke tests (require --limit)")
    if not os.environ.get("R14_GIT_SHA"):
        sys.exit("FATAL: set R14_GIT_SHA to the committed SHA of this script")
    if a.dynamics and (a.model != "pythia160m" or not a.main_run or not os.path.isdir(a.main_run)):
        sys.exit("--dynamics needs --model pythia160m and --main-run <directory of the main run>")
    os.makedirs(a.out, exist_ok=False)
    if a.dynamics:
        dynamics(a, a.out)
    else:
        main_model(a, a.out)


if __name__ == "__main__":
    main()
