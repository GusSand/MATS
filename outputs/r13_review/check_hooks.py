import os, sys, glob, importlib.util, torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
torch.manual_seed(0)
tokdir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT/snapshots/*/"))[0]
tok = AutoTokenizer.from_pretrained(tokdir)
C="Which is bigger: 9.8 or 9.11?\nAnswer:"; B="Q: Which is bigger: 9.8 or 9.11?\nA:"
for p in (C,B):
    ids=tok(p)["input_ids"]; print(len(ids), [tok.decode([i]) for i in ids])
# tiny GQA llama, 12 layers, 32 q heads, 8 kv heads, head_dim 8
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=256, intermediate_size=512, num_hidden_layers=12,
                  num_attention_heads=32, num_key_value_heads=8, head_dim=8, max_position_embeddings=128)
model = LlamaForCausalLM(cfg).eval()
spec = importlib.util.spec_from_file_location("r13", "/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py")
r13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r13)
r13.DEVICE="cpu"
p = object.__new__(r13.Patcher)
p.tok=tok; tok.pad_token=tok.eos_token; p.model=model; p.n_heads=32; p.head_dim=8
p.attn=model.model.layers[10].self_attn; p.o_proj=p.attn.o_proj
# 1. call shapes during generate
shapes=[]
h=p.o_proj.register_forward_pre_hook(lambda m,a: shapes.append(tuple(a[0].shape)))
p.generate(B); h.remove(); print("o_proj call shapes:", shapes[:3], "... n=", len(shapes))
# 2. pre_wo all heads == post_wo all coords (prefill)
sc_pre=p.capture(C,"pre_wo"); sc_post=p.capture(C,"post_wo")
a=p.generate(B, sc_pre, list(range(32)),"pre_wo","prefill"); b=p.generate(B, sc_post, list(range(32)),"post_wo","prefill")
print("all-heads pre==post (prefill):", a==b)
# 3. per-head correctness: patch head 5 at pre_wo, check o_proj input delta only in slice 5
cap={}
def grab(m,a): cap["x"]=a[0].detach().clone()
st={"n":0}
def patch(m,a):
    st["n"]+=1
    return (p._swap(a[0], sc_pre, [5]),)+tuple(a[1:])
h1=p.o_proj.register_forward_pre_hook(patch)
h2=p.o_proj.register_forward_pre_hook(grab)
ids=tok(B,return_tensors="pt")
with torch.no_grad(): model(**ids)
h1.remove(); h2.remove()
base={}
h2=p.o_proj.register_forward_pre_hook(lambda m,a: base.__setitem__("x",a[0].detach().clone()))
with torch.no_grad(): model(**ids)
h2.remove()
d=(cap["x"]-base["x"]).view(1,-1,32,8).abs().sum(dim=(0,1,3))
print("heads changed:", (d>0).nonzero().flatten().tolist())
# manual head5 output via eager computation check: compare with attention module internal
# 4. BOS activation identical clean vs buggy?
sb_pre=p.capture(B,"pre_wo"); sb_post=p.capture(B,"post_wo")
print("BOS pre_wo equal:", torch.allclose(sc_pre[:,0],sb_pre[:,0],atol=0), " post_wo:", torch.allclose(sc_post[:,0],sb_post[:,0],atol=0))
# 5. identity control in prefill and all_steps
u=p.generate(B)
for hook,ss in (("pre_wo",sb_pre),("post_wo",sb_post)):
    for scope in ("prefill","all_steps"):
        print("identity",hook,scope, p.generate(B, ss, list(range(32)), hook, scope)==u)
# 6. decode-only BOS overwrite with clean vs buggy saved give same output in all_steps?
print("all_steps clean-saved vs buggy-saved differ only via prefill; bug-B injection identical:", torch.equal(sc_post[:,:1],sb_post[:,:1]))
# 7. GQA mapping: which kv head does q head h use
print("num_key_value_groups", p.attn.num_key_value_groups)
