import sys, torch
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1])
t = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT")
C="Which is bigger: 9.8 or 9.11?\nAnswer:"; B="Q: Which is bigger: 9.8 or 9.11?\nA:"
c=t(C)["input_ids"]; b=t(B)["input_ids"]
ct=t.convert_ids_to_tokens(c); bt=t.convert_ids_to_tokens(b)
print(len(c), len(b)); print(ct); print(bt)
import run_experiment as R
class P: pass
p=P(); p.n_heads=32; p.head_dim=128
ts,sl=len(c),len(b)
saved=torch.arange(ts).float().view(1,ts,1).expand(1,ts,4096).contiguous()+1000
h=-torch.arange(sl).float().view(1,sl,1).expand(1,sl,4096).contiguous()
out=R.Patcher._swap(p,h,saved,list(range(32)),"right").view(1,sl,32,128)[0,:,0,0]
for j in range(sl):
    v=int(out[j]); src = (v-1000) if v>=1000 else None
    print(j, bt[j], "<-", (src, ct[src]) if src is not None else "unpatched")
out=R.Patcher._swap(p,h,saved,[5],"right").view(1,sl,32,128)[0]
print("only head5 changed:", sorted(set(torch.nonzero((out!=h.view(1,sl,32,128)[0]).any(-1))[:,1].tolist())))
