#!/usr/bin/env python3
"""
Experiment R21: is R13's one-way L40S vs A100 gap a shared logit-margin offset?
==============================================================================
Experiment file: docs/experiments/11-24_llama8b_gpu_margin_offset.md

Imports R13's script unchanged (sha256 checked) and runs its own Patcher.generate. The only change is a
wrapper on model.generate that also returns raw logits and hidden states (return_dict_in_generate,
output_logits, output_hidden_states); greedy decoding is otherwise untouched. The copy gate (texts equal
R13's on this GPU) checks that the wrapper changes nothing.

Per config it records, at the decision step (the first generated token after the text " 9."):
  m      = logit("8") - logit("11")   (raw logits, fp32 copy of the model's output)
  lens   = 33 logit-lens margins: final norm + the two lm_head rows, fp32, on hidden_states[l][:, -1]
           (l = 0..31 get the final norm; l = 32 is already normed in HF Llama)
Cells: unpatched buggy prompt (1 run), post_wo/prefill and post_wo/all_steps (1,307 configs each).
The cross-GPU analysis is analyze.py; this job only measures.
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
CELLS = ["prefill", "all_steps"]          # hook is post_wo in both, as in R13 and R16
LENS_TOL = 0.05                           # final-layer lens vs true margin (fp32 lens vs fp16 lm_head)
N_DET = 50                                # determinism control: first N prefill configs re-run in the same job


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, choices=["a100", "l40s"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N configs; gates recorded, not enforced")
    ap.add_argument("--smoke-tdec", type=int, default=None,
                    help="smoke only (needs --limit): random weights never write ' 9.', so use this step instead")
    a = ap.parse_args()
    if a.smoke_tdec is not None and not a.limit:
        sys.exit("--smoke-tdec is smoke-only")
    if not os.environ.get("R21_GIT_SHA"):
        sys.exit("FATAL: set R21_GIT_SHA")
    if sha256(R13_PATH) != R13_SHA256 and not a.limit:
        sys.exit("FATAL: R13 script changed")
    os.makedirs(a.out, exist_ok=False)
    os.environ["R13_GIT_SHA"] = os.environ["R21_GIT_SHA"]
    spec = importlib.util.spec_from_file_location("r13", R13_PATH)
    r13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r13)
    import torch

    p = r13.Patcher()
    tok, model = p.tok, p.model
    id8 = tok.encode("8", add_special_tokens=False)
    id11 = tok.encode("11", add_special_tokens=False)
    if not a.limit:
        assert len(id8) == 1 and len(id11) == 1, (id8, id11)
    id8, id11 = id8[0], id11[0]
    W = model.lm_head.weight[[id8, id11]].float()          # (2, d)
    norm = model.model.norm
    n_layers = model.config.num_hidden_layers

    stash = {}
    orig_generate = model.generate

    def wrapped(**kw):
        o = orig_generate(**kw, return_dict_in_generate=True, output_logits=True, output_hidden_states=True)
        stash["o"] = o
        return o.sequences
    model.generate = wrapped                                 # instance attribute; R13's code calls this

    plen = tok(r13.BUGGY_PROMPT, return_tensors="pt")["input_ids"].shape[1]

    def measure(text):
        o = stash.pop("o")
        gen = o.sequences[0, plen:].tolist()
        t_dec = next((t for t in range(1, len(gen)) if tok.decode(gen[:t], skip_special_tokens=True) == " 9."),
                     None)
        if t_dec is None and a.smoke_tdec is not None and a.smoke_tdec < len(gen):
            t_dec = a.smoke_tdec
        rec = {"output": text, "correct": r13.is_correct(text), "bug": r13.is_bug(text), "gen_ids": gen,
               "t_dec": t_dec}
        # top-2 gap at every step up to the decision step (or the first 4 steps)
        last = t_dec if t_dec is not None else min(3, len(o.logits) - 1)
        rec["top2_gap"] = []
        for t in range(last + 1):
            v = torch.topk(o.logits[t][0].float(), 2)
            rec["top2_gap"].append([int(v.indices[0]), int(v.indices[1]), float(v.values[0] - v.values[1])])
        if t_dec is None:
            return rec
        lg = o.logits[t_dec][0].float()
        rec["logit8"], rec["logit11"] = float(lg[id8]), float(lg[id11])
        rec["m"] = rec["logit8"] - rec["logit11"]
        rec["argmax"] = int(lg.argmax())
        rec["max_abs_logit"] = float(lg.abs().max())
        hs = o.hidden_states[t_dec]                          # tuple of n_layers + 1, each (1, 1, d)
        assert len(hs) == n_layers + 1
        lens = []
        with torch.no_grad():
            for l, h in enumerate(hs):
                x = h[:, -1]
                x = norm(x) if l < n_layers else x
                z = x.float() @ W.T
                lens.append(float(z[0, 0] - z[0, 1]))
        rec["lens"] = lens
        rec["lens_final_err"] = abs(lens[-1] - rec["m"])
        return rec

    r8, r1 = r13.load_refs()
    cfgs = r13.build_configs(p.n_heads, r1)
    if a.limit:
        cfgs = cfgs[:a.limit]
    prov = r13.provenance()
    prov.update({"r21_script_sha256": sha256(__file__), "r13_script_sha256": sha256(R13_PATH),
                 "r21_git_sha": os.environ["R21_GIT_SHA"], "gpu_role": a.gpu, "token_ids": [id8, id11]})
    r13_prov = load_json(os.path.join(REF[a.gpu], "run.json"))["provenance"]
    run = {"provenance": prov, "started": datetime.now().isoformat(), "command": " ".join(sys.argv),
           "same_node_as_r13": prov["hostname"].startswith(R13_NODE[a.gpu]),
           "same_gpu_string_as_r13": prov.get("gpu", "").split(",")[0] == r13_prov.get("gpu", "").split(",")[0]}
    path = os.path.join(a.out, "run.json")
    gates = {}

    # unpatched buggy prompt
    text = p.generate(r13.BUGGY_PROMPT)
    run["unpatched"] = measure(text)
    run["unpatched_text_equal_r13_baseline"] = text == load_json(os.path.join(REF[a.gpu], "run.json"))["baseline"]["buggy"]
    gates["unpatched_text"] = run["unpatched_text_equal_r13_baseline"]
    dump(run, path)
    print(f"[{a.gpu}] unpatched: {json.dumps({k: run['unpatched'].get(k) for k in ('output', 't_dec', 'm')})}",
          flush=True)

    clean = p.capture(r13.CORRECT_PROMPT, "post_wo")
    run["cells"] = {}
    for scope in CELLS:
        ref_own = load_json(os.path.join(REF[a.gpu], f"results_post_wo__{scope}.json"))["configs"]
        t0, done = time.time(), {}
        for i, (name, heads) in enumerate(cfgs):
            text = p.generate(r13.BUGGY_PROMPT, clean, heads, "post_wo", scope)
            done[name] = {"heads": heads, **measure(text)}
            if (i + 1) % 200 == 0:
                print(f"[{a.gpu}/{scope}] {i+1}/{len(cfgs)} {time.time()-t0:.0f}s", flush=True)
        dump({"scope": scope, "configs": done}, os.path.join(a.out, f"results_post_wo__{scope}.json"))
        n_eq = sum(done[k]["output"] == ref_own[k]["output"] for k in done)
        with_m = [k for k in done if "m" in done[k]]
        cls_ok = [k for k in with_m if done[k]["argmax"] in (id8, id11)
                  and (done[k]["argmax"] == id8) == done[k]["correct"]]
        lens_bad = [k for k in with_m if done[k]["lens_final_err"] > LENS_TOL]
        cell = {"n": len(done), "text_equal_r13": n_eq, "gate_copy": n_eq == len(done),
                "n_with_decision_step": len(with_m), "lens_final_err_max": max(
                    (done[k]["lens_final_err"] for k in with_m), default=None),
                "n_lens_over_tol": len(lens_bad), "gate_lens": not lens_bad,
                "class_matches_argmax": len(cls_ok), "gate_class": len(cls_ok) == len(done),
                "max_abs_logit": max((done[k]["max_abs_logit"] for k in with_m), default=None),
                "seconds": time.time() - t0}
        gates[f"{scope}_copy"] = cell["gate_copy"]
        gates[f"{scope}_lens"] = cell["gate_lens"]
        gates[f"{scope}_class"] = cell["gate_class"]
        run["cells"][scope] = cell
        print(f"[{a.gpu}/{scope}] {cell}", flush=True)
        dump(run, path)
    # determinism control (design review finding 3a): unpatched + first N_DET prefill configs again, same job
    det = {"unpatched": None, "configs": {}}
    text = p.generate(r13.BUGGY_PROMPT)
    r2 = measure(text)
    det["unpatched"] = (r2["output"] == run["unpatched"]["output"] and r2.get("m") == run["unpatched"].get("m")
                        and r2.get("lens") == run["unpatched"].get("lens"))
    first = load_json(os.path.join(a.out, "results_post_wo__prefill.json"))["configs"]
    for name, heads in cfgs[:N_DET]:
        r2 = measure(p.generate(r13.BUGGY_PROMPT, clean, heads, "post_wo", "prefill"))
        det["configs"][name] = (r2["output"] == first[name]["output"] and r2.get("m") == first[name].get("m")
                                and r2.get("lens") == first[name].get("lens"))
    det["n_identical"] = sum(det["configs"].values())
    det["ok"] = bool(det["unpatched"]) and det["n_identical"] == len(det["configs"])
    run["determinism"] = det
    gates["determinism"] = det["ok"]
    print(f"[{a.gpu}] determinism: {det['ok']} ({det['n_identical']}/{len(det['configs'])})", flush=True)
    run["gates"] = gates
    run["all_gates_ok"] = all(gates.values())
    run["finished"] = datetime.now().isoformat()
    dump(run, path)
    if not run["all_gates_ok"] and not a.limit:
        sys.exit(2)


if __name__ == "__main__":
    main()
