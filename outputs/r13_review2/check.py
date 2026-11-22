import sys, os, json, torch
sys.path.insert(0, "/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching")
os.environ["R13_GIT_SHA"] = "review"
import run_experiment as R
p = R.Patcher()
r8, r1 = R.load_refs()
cfgs = R.build_configs(p.n_heads, r1)
print("n_cfgs", len(cfgs))
tc = p.tok(R.CORRECT_PROMPT)["input_ids"]; tb = p.tok(R.BUGGY_PROMPT)["input_ids"]
print(len(tc), len(tb)); print([p.tok.decode([t]) for t in tc]); print([p.tok.decode([t]) for t in tb])
# BOS identity
for hook in ["post_wo", "pre_wo"]:
    a = p.capture(R.CORRECT_PROMPT, hook); b = p.capture(R.BUGGY_PROMPT, hook)
    print(hook, "BOS maxdiff", (a[:, 0] - b[:, 0]).abs().max().item())
# right alignment: record what the o_proj input looks like after the pre-hook
clean = p.capture(R.CORRECT_PROMPT, "pre_wo"); bug = p.capture(R.BUGGY_PROMPT, "pre_wo")
rec = []
h2 = p.o_proj.register_forward_pre_hook(lambda m, a: rec.append(a[0].detach().clone()))
# register recorder AFTER patch hook: hooks run in registration order, so register patch first
h2.remove()
def run(scope, heads):
    rec.clear()
    orig = p.o_proj.register_forward_pre_hook
    out = None
    # wrap: patch hook registered inside generate; add recorder via forward hook on o_proj capturing input
    hh = p.o_proj.register_forward_hook(lambda m, a, o: rec.append(a[0].detach().clone()))
    try:
        out = p.generate(R.BUGGY_PROMPT, clean, heads, "pre_wo", scope)
    finally:
        hh.remove()
    return out
H = p.head_dim
for scope in ["prefill", "prefill_right", "all_steps"]:
    run(scope, [5])
    x = rec[0].view(1, -1, 32, H); bv = bug.view(1, -1, 32, H); cv = clean.view(1, -1, 32, H)
    changed_heads = sorted(set(torch.nonzero((x - bv).abs().amax(dim=(0, 1, 3)) > 0).flatten().tolist()))
    pos = torch.nonzero((x[0, :, 5] - bv[0, :, 5]).abs().amax(-1) > 0).flatten().tolist()
    if scope == "prefill_right":
        ok = torch.equal(x[0, 2:, 5], cv[0, :, 5]); src = "clean0..16 -> buggy2..18"
    else:
        ok = torch.equal(x[0, :17, 5], cv[0, :, 5]); src = "clean0..16 -> buggy0..16"
    dec = [ (r.view(1,1,32,H)[0,0,5] - cv[0,0,5]).abs().max().item() for r in rec[1:3]]
    print(scope, "changed heads", changed_heads, "changed positions", pos, src, ok, "decode head5 - cleanBOS", dec, "n calls", len(rec))
# decode_only
rec.clear()
hh = p.o_proj.register_forward_hook(lambda m, a, o: rec.append(a[0].detach().clone()))
bpost = p.capture(R.BUGGY_PROMPT, "post_wo")
out = p.generate(R.BUGGY_PROMPT, bpost, list(range(32)), "post_wo", "all_steps")
hh.remove()
print("decode_only text == unpatched?", out == p.generate(R.BUGGY_PROMPT))
# all-heads equivalence on tiny model
cpost = p.capture(R.CORRECT_PROMPT, "post_wo")
t1 = p.generate(R.BUGGY_PROMPT, clean, list(range(32)), "pre_wo", "prefill")
t2 = p.generate(R.BUGGY_PROMPT, cpost, list(range(32)), "post_wo", "prefill")
print("all-heads equiv same text", t1 == t2)
# summarize/reproduction on a reduced config list via run_cell (limit=None)
keep = [c for c in cfgs if c[0] in ("all_heads","even_16","odd_16") or c[0].startswith(("single_0","even7_00","even8_00","trans","rand8_00","pair0","oddonly"))]
print("reduced", len(keep))
out_dir = "/Users/gus/github/MATS/outputs/r13_review2/run"; os.makedirs(out_dir, exist_ok=True)
res = R.run_cell(p, "post_wo", "all_steps", keep, out_dir)
print(json.dumps(res["summary"])[:800])
print(R.reproduction(res["configs"], R.reference_outcomes(r8, r1)))
