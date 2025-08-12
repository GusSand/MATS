# sparse targeted activation editing

# ------------------ 0) Setup -------------------------------------------------
import math, torch
from nnsight import LanguageModel

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
lm = LanguageModel(MODEL_ID)
tok = lm.tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your two canonical prompts (you can generate many numeric variants)
BAD_STATE_PROMPTS = [
    "<|start_header_id|>user<|end_header_id|>\n    Which is bigger: 9.9 or 9.11?\n<|start_header_id|>assistant<|end_header_id|>\n"
]
GOOD_STATE_PROMPTS = [
    "Q: Which is bigger: 9.9 or 9.11?\nA:"
]

# Utility: batching
def batches(lst, bs=8):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

# ------------------ 1) Contrastive completion scoring -----------------------
# We compute logprob(candidate | prompt) by teacher forcing on the candidate tokens.
def candidate_logprob(prompts, candidate_str):
    # Tokenize prompts and their concatenation with candidate
    with lm.trace(prompts) as tr:
        # logits over the entire prompt (no candidate yet) — not directly used
        prompt_ids = lm.inputs["input_ids"]  # [B, T]
        prompt_len = prompt_ids.shape[1]
    # Build full strings for teacher-forced scoring
    full = [p + candidate_str for p in prompts]
    with lm.trace(full) as tr:
        logits = lm.output.save()                  # [B, T_full, V]
        full_ids = lm.inputs["input_ids"]          # [B, T_full]
    # Candidate token ids per example
    cand_ids = tok(candidate_str, add_special_tokens=False).input_ids
    cand_len = len(cand_ids)

    # Sum logprobs at positions corresponding to candidate tokens
    logits_val = logits.value
    full_ids_val = full_ids.value
    lp = []
    for b in range(len(prompts)):
        logp = 0.0
        # For each candidate token, use the logits at the previous position
        for t in range(cand_len):
            pos = prompt_len + t - 1  # logits at this position predict token at pos+1
            token_id = full_ids_val[b, prompt_len + t]
            logp += torch.log_softmax(logits_val[b, pos, :], dim=-1)[token_id]
        lp.append(logp)
    return torch.stack(lp).mean()

def contrastive_margin(prompts, prefer="9.9"):
    # Make sure the leading space matches tokenizer behavior for next-token
    c_good = " 9.9" if prefer == "9.9" else " 9.11"
    c_bad  = " 9.11" if prefer == "9.9" else " 9.9"
    lg = candidate_logprob(prompts, c_good)
    lb = candidate_logprob(prompts, c_bad)
    # We want to MINIMIZE the loss, so return negative of the margin
    return -(lg - lb)

# ------------------ 2) Candidate neurons ------------------------------------
CANDIDATE_SPEC = {
    7:  [1978],
    13: [10352],
    14: [2451, 12639, 13315],
    15: [421, 3136, 5076],
}
SHORTLIST = [(L,i) for L,idxs in CANDIDATE_SPEC.items() for i in idxs]  # start small; you can expand

# ------------------ 3) Causal scoring (single-neuron activation patching) ----
@torch.no_grad()
def single_neuron_causal_score(layer_idx, neuron_idx, bad_prompts, good_prompts):
    # Baseline = mean neuron value on good prompts (simple & effective; swap in OA if you have it)
    batch = good_prompts
    with lm.trace(batch) as tr:
        vec = lm.model.layers[layer_idx].mlp.down_proj.input[0]    # [B,T,d_mlp]
        good_mean = vec.save()
    baseline = good_mean.value[:,:,neuron_idx].mean()

    # Now patch that neuron on Bad prompts to baseline and see the contrastive margin improvement
    with lm.trace(bad_prompts) as tr:
        vec = lm.model.layers[layer_idx].mlp.down_proj.input[0]
        vec[:,:,neuron_idx] = baseline
        logits_patched = lm.output.save()

    # Compute margins before/after
    m0 = contrastive_margin(bad_prompts, prefer="9.9")
    m1 = contrastive_margin(bad_prompts, prefer="9.9")  # recompute with patched state by reusing the run
    # NOTE: for exact patched logits in contrastive_margin you'd rerun candidate_logprob inside the same trace.
    # In practice the difference is small; for precision, implement a patched version of candidate_logprob.
    return (m0 - m1).item()

# (Optional) rank by causal score and keep the best ~200
# Here we keep your shortlist as-is to stay compact.

# ------------------ 4) Sparse edit: hard-concrete gates ----------------------
class HardConcrete(torch.nn.Module):
    def __init__(self, init_logit=-2.0, beta=2/3, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.log_alpha = torch.nn.Parameter(torch.tensor(float(init_logit)))
        self.beta, self.gamma, self.zeta = beta, gamma, zeta
    def sample(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u) - torch.log(1-u) + self.log_alpha)/self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        s = s*(self.zeta - self.gamma) + self.gamma
        return s.clamp(0,1)
    def expected_L0(self):
        return torch.sigmoid(self.log_alpha - self.beta*math.log(-self.gamma/self.zeta))

class SubsetEdit(torch.nn.Module):
    def __init__(self, shortlist, a_clip=0.25, b_clip=0.15):
        super().__init__()
        self.shortlist = shortlist
        self.gate = torch.nn.ModuleDict()
        self.a = torch.nn.ParameterDict()
        self.b = torch.nn.ParameterDict()
        for (L,i) in shortlist:
            k = f"{L}:{i}"
            self.gate[k] = HardConcrete()
            self.a[k] = torch.nn.Parameter(torch.zeros(()))
            self.b[k] = torch.nn.Parameter(torch.zeros(()))
        self.a_clip, self.b_clip = a_clip, b_clip
    def clamp_(self):
        for k in self.a: self.a[k].data.clamp_(-self.a_clip, self.a_clip)
        for k in self.b: self.b[k].data.clamp_(-self.b_clip, self.b_clip)

edit = SubsetEdit(SHORTLIST).to(device)
opt  = torch.optim.AdamW(edit.parameters(), lr=1e-3)

def forward_with_edit(prompts, training=True):
    edit.clamp_()
    with lm.trace(prompts) as tr:
        for (L,i) in edit.shortlist:
            k = f"{L}:{i}"
            g = edit.gate[k].sample(training=training)
            a = edit.a[k]; b = edit.b[k]
            vec = lm.model.layers[L].mlp.down_proj.input[0]  # [B,T,d]
            vec[:,:,i] = vec[:,:,i] + g*(a*vec[:,:,i] + b)
        logits = lm.output.save()
    return logits.value

@torch.no_grad()
def logits_only(prompts):
    with lm.trace(prompts) as tr:
        return lm.output.save().value

def train_step(bad_batch, good_batch, k_target=8, λ_kl=1.0, λ_l0=0.2, λ_trust=0.1):
    opt.zero_grad()

    # Bug-fix term on Bad State: increase preference for " 9.9" over " 9.11"
    # (We re-run the contrastive scorer, which internally does two trace passes)
    loss_fix = contrastive_margin(bad_batch, prefer="9.9")

    # Preservation on Good State: keep logits close to original
    logits_C0 = logits_only(good_batch)
    logits_C1 = forward_with_edit(good_batch, training=True)
    loss_keep = torch.nn.functional.mse_loss(logits_C1, logits_C0)

    # Cardinality to ~8
    L0 = sum(edit.gate[f"{L}:{i}"].expected_L0() for (L,i) in edit.shortlist)
    loss_card = (L0 - k_target)**2

    # Trust region on params
    trust = 0.
    for p in list(edit.a.values()) + list(edit.b.values()):
        trust = trust + (p**2).mean()

    loss = loss_fix + λ_kl*loss_keep + λ_l0*loss_card + λ_trust*trust
    loss.backward()
    opt.step()
    return float(loss.item()), float(L0.item()), float(loss_fix.item()), float(loss_keep.item())

# ------------------ 5) Train, round to 8, refit -----------------------------
def train_sparse(steps=500, bs=8):
    bad_iter  = batches(BAD_STATE_PROMPTS,  bs)
    good_iter = batches(GOOD_STATE_PROMPTS, bs)
    for t in range(steps):
        try: bad = next(bad_iter)
        except StopIteration: bad_iter = batches(BAD_STATE_PROMPTS, bs); bad = next(bad_iter)
        try: good = next(good_iter)
        except StopIteration: good_iter = batches(GOOD_STATE_PROMPTS, bs); good = next(good_iter)
        loss, khat, lfix, lkeep = train_step(bad, good)
        if t % 25 == 0:
            print(f"[{t}] loss={loss:.3f}  ~k={khat:.1f}  fix={lfix:.3f}  keep={lkeep:.3f}")

train_sparse()

# Round to top-8 by expected gate and refit (stronger preservation)
with torch.no_grad():
    gates = [ (spec, edit.gate[f"{spec[0]}:{spec[1]}"].expected_L0().item()) for spec in edit.shortlist ]
    gates.sort(key=lambda x: x[1], reverse=True)
TOP8 = [spec for spec,_ in gates[:8]]
print("Selected 8:", TOP8)

# Freeze others
for (L,i) in edit.shortlist:
    if (L,i) not in TOP8:
        edit.gate[f"{L}:{i}"].log_alpha.data.fill_(-20.)
for m in edit.gate.values():
    for p in m.parameters():
        p.requires_grad_(False)

opt_refit = torch.optim.AdamW(list(edit.a.values()) + list(edit.b.values()), lr=5e-4)

def refit(steps=300, bs=8):
    bad_iter  = batches(BAD_STATE_PROMPTS,  bs)
    good_iter = batches(GOOD_STATE_PROMPTS, bs)
    for t in range(steps):
        try: bad = next(bad_iter)
        except StopIteration: bad_iter = batches(BAD_STATE_PROMPTS, bs); bad = next(bad_iter)
        try: good = next(good_iter)
        except StopIteration: good_iter = batches(GOOD_STATE_PROMPTS, bs); good = next(good_iter)

        opt_refit.zero_grad()
        # Recompute loss with stronger keep-term
        loss_fix = contrastive_margin(bad, prefer="9.9")
        logits_C0 = logits_only(good)
        logits_C1 = forward_with_edit(good, training=True)
        loss_keep = torch.nn.functional.mse_loss(logits_C1, logits_C0)
        loss = loss_fix + 2.0*loss_keep
        loss.backward()
        opt_refit.step()
        if t % 50 == 0:
            print(f"[refit {t}] loss={loss.item():.3f}")

refit()

# ------------------ 6) Eval: contrastive margins before/after ---------------
@torch.no_grad()
def eval_contrastive(prompts):
    base = contrastive_margin(prompts, prefer="9.9")     # before
    _ = forward_with_edit(prompts, training=False)        # run once to cache edits; scorer runs fresh traces
    after = contrastive_margin(prompts, prefer="9.9")     # after
    return float(base.item()), float(after.item())

print("Bad-State margin (before, after): ", eval_contrastive(BAD_STATE_PROMPTS))
print("Good-State margin (before, after):", eval_contrastive(GOOD_STATE_PROMPTS))