import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("m", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# placebo mirrors the variant: A100 copy=clean0, keep0 removes d0; placebo removes the same-size step elsewhere
torch.manual_seed(0)
buggy0 = (torch.randn(1, 4096) * 0.05).half()
clean0 = buggy0.clone(); c = 1000
clean0[0, c] = m.ulp_step(buggy0[0, c:c+1], True)[0]
d0 = clean0.float() - buggy0.float()
pl, info = m.make_placebo(clean0, -d0)
moves = info["moves"]
assert info["valid"]
diff = (pl.float() - clean0.float()).reshape(-1)
nzp = torch.nonzero(diff).flatten().tolist()
assert c not in nzp and len(nzp) == 1 and nzp[0] // 128 == c // 128, nzp
assert float(diff[nzp[0]]) == -float(d0[0, c]), moves
print("placebo: copy-relative change at", nzp, "same block, same size, opposite of d0:", moves[0])
# decide branches on synthetic dicts
R = lambda c_: {"correct": c_, "output": "x" if c_ else "y"}
keys = [f"k{i}" for i in range(40)]
other = {k: R(False) for k in keys}                         # other GPU (L40S): all bug
copy = {k: R(True) for k in keys}                           # this GPU (A100): all correct -> D = all 40
D, exposed = keys, set(keys)
var = {k: R(False) for k in keys}                           # keep0 moves all 40
assert m.decide(var, copy, other, D, exposed, placebo_moved=0)["verdict"] == "explains the gap"
assert m.decide(var, copy, other, D, exposed, placebo_moved=40)["verdict"] == "not distinguishable from fragility"
assert m.decide(var, copy, other, D, exposed, placebo_moved=20)["verdict"] == "explains the gap"   # 2*(40-20) >= 40
assert m.decide(var, copy, other, D, exposed, placebo_moved=21)["verdict"] == "not distinguishable from fragility"
part = set(keys[:12])
v = m.decide(var, copy, other, D, part, placebo_moved=0)["verdict"]
assert v.startswith("explains the exposed part"), v
assert m.decide(var, copy, other, D, set(keys[:5]), placebo_moved=0)["verdict"].startswith("descriptive only")
assert m.decide(var, copy, other, D, set(), placebo_moved=0)["verdict"].startswith("refuted by construction")
none_moved = {k: R(True) for k in keys}
assert m.decide(none_moved, copy, other, D, exposed, placebo_moved=0)["verdict"] == "does not explain the gap"
extra = dict(none_moved); extra.update({f"o{i}": R(False) for i in range(5)})
copy2 = dict(copy); copy2.update({f"o{i}": R(True) for i in range(5)}); other2 = dict(other); other2.update({f"o{i}": R(True) for i in range(5)})
assert m.decide(extra, copy2, other2, D, exposed, placebo_moved=0)["verdict"] == "perturbs, does not explain"
d = m.decide(var, copy, other, D, set(), label=False)
assert d["verdict"] == "descriptive" and d["r"] == 1.0
print("decide: all branches as pre-registered")
# dense d0: 128 nonzero entries in one block cannot all be placed -> invalid placebo, no crash
dense = clean0.clone(); dense[0, 256:384] = m.ulp_step(buggy0[0, 256:384], True)
dd = dense.float() - buggy0.float()
_, info2 = m.make_placebo(dense, -dd)
assert not info2["valid"] and info2["unplaced"], info2["unplaced"][:3]
# rule 3 and rule 5 placebo / N_new checks
assert m.decide(var, copy, other, D, exposed, placebo_moved=0, placebo_valid=False)["verdict"] == "not distinguishable from fragility (no valid placebo)"
half = {k: R(False) if i < 20 else R(True) for i, k in enumerate(keys)}
assert m.decide(half, copy, other, D, exposed, placebo_moved=0)["verdict"] == "explains part of the gap (r = 20/40)"
assert m.decide(half, copy, other, D, exposed, placebo_moved=20)["verdict"] == "not distinguishable from fragility"
half2 = dict(half); half2.update({f"o{i}": R(False) for i in range(5)})
assert m.decide(half2, copy2, other2, D, exposed, placebo_moved=0)["verdict"] == "perturbs broadly"
moved_all_new = dict(var); moved_all_new.update({f"o{i}": R(False) for i in range(5)})
assert m.decide(moved_all_new, copy2, other2, D, exposed, placebo_moved=0)["verdict"] == "perturbs broadly"
print("dense placebo, rule-3 invalid placebo, rule-5 placebo and N_new branches: OK")
