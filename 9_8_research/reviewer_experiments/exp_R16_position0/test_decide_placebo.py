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
assert m.decide(half, copy, other, D, exposed, placebo_moved=0, placebo_valid=False)["verdict"] == "not distinguishable from fragility (no valid placebo)"
k10 = keys[:10]; c10 = {k: copy[k] for k in k10}; o10 = {k: other[k] for k in k10}
v3 = {k: R(False) if i < 3 else R(True) for i, k in enumerate(k10)}
v = m.decide(v3, c10, o10, k10, set(k10), placebo_moved=0)["verdict"]
assert "fragility" not in v and v.startswith("partial, below the placebo margin"), v
assert m.decide(v3, c10, o10, k10, set(k10), placebo_moved=2)["verdict"] == "not distinguishable from fragility"
half2 = dict(half); half2.update({f"o{i}": R(False) for i in range(5)})
assert m.decide(half2, copy2, other2, D, exposed, placebo_moved=0)["verdict"] == "perturbs broadly"
moved_all_new = dict(var); moved_all_new.update({f"o{i}": R(False) for i in range(5)})
assert m.decide(moved_all_new, copy2, other2, D, exposed, placebo_moved=0)["verdict"] == "perturbs broadly"
# one tiny unplaceable source among many normal ones: still valid (round 4)
blk = torch.full((1, 4096), 0.15).half(); blk[0, 0:128] = (torch.rand(128) * 0.1 + 0.1).half()
blk[0, 7] = 2e-4
st = torch.zeros(1, 4096); st[0, 7] = 1.2e-7
for j in (11, 20, 33, 47, 60, 71, 85, 99, 110, 120):
    st[0, j] = float(m.ulp_step(blk[0, j:j+1], True)[0]) - float(blk[0, j])
_, info3 = m.make_placebo(blk, st)
assert info3["unplaced"] == [7] and info3["unplaced_step_fraction"] < 0.01 and info3["valid"], info3
print("dense placebo, tiny-unplaced tolerance, rule-3/5 placebo and N_new branches: OK")
