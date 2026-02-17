#!/usr/bin/env python3
"""
Investigation: Mistral-7B Logit Lens — Zero Emergence Bug?

Phase 1: Token ID verification (no GPU)
Phase 2: Top-K logit lens diagnostic (GPU)
Phase 3: Diagnosis
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

SHARED_DIR = Path(__file__).parent.parent / "02-05_cross_cwe_steering" / "shared"
sys.path.insert(0, str(SHARED_DIR))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CWE787_DATASET = (
    Path(__file__).parent.parent
    / "01-12_cwe787_dataset_expansion" / "data"
    / "cwe787_expanded_20260112_143316.jsonl"
)
CWE89_DATASET = (
    Path(__file__).parent.parent
    / "02-05_cross_cwe_steering" / "datasets" / "cwe89" / "data"
    / "cwe89_expanded_20260209_221808.jsonl"
)


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Token ID Verification
# ═══════════════════════════════════════════════════════════════════════════

def phase1():
    print("=" * 70)
    print("PHASE 1: Token ID Verification")
    print("=" * 70)

    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    print(f"\nLlama vocab size: {llama_tok.vocab_size}")
    print(f"Mistral vocab size: {mistral_tok.vocab_size}")

    # ── CWE-787: snprintf / sprintf ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("CWE-787: snprintf / sprintf tokenization")
    print(f"{'─'*70}")

    for label, text in [(" snprintf", " snprintf"), ("snprintf", "snprintf"),
                         (" sprintf", " sprintf"), ("sprintf", "sprintf"),
                         ("sn", "sn"), ("printf", "printf"), (" printf", " printf"),
                         ("str", "str"), (" str", " str"),
                         ("n", "n"), (" sn", " sn")]:
        llama_ids = llama_tok.encode(text, add_special_tokens=False)
        mistral_ids = mistral_tok.encode(text, add_special_tokens=False)
        llama_subtokens = llama_tok.tokenize(text)
        mistral_subtokens = mistral_tok.tokenize(text)

        print(f"\n  Text: '{text}'")
        print(f"    Llama:   ids={llama_ids}  subtokens={llama_subtokens}")
        print(f"    Mistral: ids={mistral_ids}  subtokens={mistral_subtokens}")

        # What the Exp 12 script would have used (ids[0])
        if llama_ids:
            print(f"    Llama ids[0]={llama_ids[0]} -> decodes to '{llama_tok.decode([llama_ids[0]])}'")
        if mistral_ids:
            print(f"    Mistral ids[0]={mistral_ids[0]} -> decodes to '{mistral_tok.decode([mistral_ids[0]])}'")

    # ── CWE-89: SQL tokens ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("CWE-89: SQL injection tokens")
    print(f"{'─'*70}")

    for label, text in [("?", "?"), ("execute", "execute"), (" execute", " execute"),
                         ("parameterized", "parameterized"), ("placeholder", "placeholder"),
                         ("%s", "%s"), ("cursor", "cursor"), (" cursor", " cursor")]:
        llama_ids = llama_tok.encode(text, add_special_tokens=False)
        mistral_ids = mistral_tok.encode(text, add_special_tokens=False)
        llama_subtokens = llama_tok.tokenize(text)
        mistral_subtokens = mistral_tok.tokenize(text)

        print(f"\n  Text: '{text}'")
        print(f"    Llama:   ids={llama_ids}  subtokens={llama_subtokens}")
        print(f"    Mistral: ids={mistral_ids}  subtokens={mistral_subtokens}")

    # ── Critical check: what the Exp 12 script actually used ─────────────
    print(f"\n{'─'*70}")
    print("CRITICAL: What Exp 12 script would have computed")
    print(f"{'─'*70}")

    # Reproduce find_best_secure_tokens for Mistral
    for cwe_type in ["CWE-787", "CWE-89"]:
        print(f"\n  {cwe_type}:")
        if cwe_type == "CWE-787":
            candidates = {
                "snprintf": " snprintf",
                "snprintf_no_space": "snprintf",
            }
        elif cwe_type == "CWE-89":
            candidates = {
                "question_mark": "?",
                "execute": "execute",
                "parameterized": "parameterized",
                "placeholder": "placeholder",
            }

        for label_name, text in candidates.items():
            ids = mistral_tok.encode(text, add_special_tokens=False)
            subtokens = mistral_tok.tokenize(text)
            if ids:
                decoded = mistral_tok.decode([ids[0]])
                full_decoded = mistral_tok.decode(ids)
                print(f"    {label_name}: '{text}'")
                print(f"      All IDs: {ids}")
                print(f"      Subtokens: {subtokens}")
                print(f"      ids[0]={ids[0]} -> '{decoded}'")
                print(f"      Full decode: '{full_decoded}'")
                if len(ids) > 1:
                    print(f"      *** WARNING: Multi-token! Script only tracked ids[0] = '{decoded}', "
                          f"not the full token '{full_decoded}' ***")

    # ── Also check: does Llama have single-token "snprintf"? ────────────
    print(f"\n{'─'*70}")
    print("KEY COMPARISON: Is 'snprintf' a single token?")
    print(f"{'─'*70}")
    for name, tok in [("Llama", llama_tok), ("Mistral", mistral_tok)]:
        for text in [" snprintf", "snprintf", " sprintf", "sprintf"]:
            ids = tok.encode(text, add_special_tokens=False)
            subtokens = tok.tokenize(text)
            single = "YES" if len(ids) == 1 else f"NO ({len(ids)} tokens)"
            print(f"  {name:8s} '{text:12s}': single_token={single}  ids={ids}  subtokens={subtokens}")

    return {"phase": 1, "status": "complete"}


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Top-K Logit Lens Diagnostic (GPU)
# ═══════════════════════════════════════════════════════════════════════════

def phase2():
    print(f"\n\n{'='*70}")
    print("PHASE 2: Top-K Logit Lens Diagnostic")
    print("=" * 70)

    from model_loader import ModelLoader

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    loader = ModelLoader(MODEL_NAME)
    model = loader.model
    tokenizer = loader.tokenizer
    device = loader.device

    # ── Check architecture ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Architecture Check")
    print(f"{'─'*70}")

    print(f"  model.config.tie_word_embeddings: {model.config.tie_word_embeddings}")
    print(f"  model.lm_head.weight.shape: {model.lm_head.weight.shape}")
    print(f"  model.model.embed_tokens.weight.shape: {model.model.embed_tokens.weight.shape}")

    # Check if lm_head and embed_tokens share the same memory
    lm_head_ptr = model.lm_head.weight.data_ptr()
    embed_ptr = model.model.embed_tokens.weight.data_ptr()
    tied = lm_head_ptr == embed_ptr
    print(f"  Weights share memory (actually tied): {tied}")
    if not tied:
        # Check cosine similarity between first few rows
        cos = torch.nn.functional.cosine_similarity(
            model.lm_head.weight[:10].float(),
            model.model.embed_tokens.weight[:10].float(),
            dim=1
        )
        print(f"  Cosine similarity (first 10 rows): {cos.mean().item():.6f}")

    print(f"  model.model.norm: {model.model.norm}")

    # ── Load datasets ────────────────────────────────────────────────────
    cwe787_data = load_jsonl(CWE787_DATASET)
    cwe89_data = load_jsonl(CWE89_DATASET)

    # ── Helper: format prompt ────────────────────────────────────────────
    def format_prompt(text):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── Helper: extract top-K and specific token probs ───────────────────
    def logit_lens_topk(model, tokenizer, device, formatted_prompt, layers, k=20,
                        track_tokens=None):
        """Run logit lens and return top-K tokens + tracked token probs at each layer."""
        residual_stream = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                residual_stream[layer_idx] = h[:, -1, :].detach().clone()
            return hook_fn

        for layer_idx in layers:
            hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        for hook in hooks:
            hook.remove()

        results = {}
        for layer_idx in layers:
            residual = residual_stream[layer_idx]
            normed = model.model.norm(residual)
            logits = model.lm_head(normed).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

            # Top K
            top_probs, top_ids = torch.topk(probs, k)
            top_tokens = [(tokenizer.decode([t.item()]), float(top_probs[i].item()))
                          for i, t in enumerate(top_ids)]

            # Tracked tokens
            tracked = {}
            if track_tokens:
                for label, tid in track_tokens.items():
                    tracked[label] = float(probs[tid].item())

            results[layer_idx] = {
                "top_k": top_tokens,
                "tracked": tracked,
            }

        # Also get final output
        final_logits = outputs.logits[0, -1, :]
        final_probs = torch.softmax(final_logits, dim=-1)
        top_probs, top_ids = torch.topk(final_probs, k)
        top_tokens = [(tokenizer.decode([t.item()]), float(top_probs[i].item()))
                      for i, t in enumerate(top_ids)]
        tracked = {}
        if track_tokens:
            for label, tid in track_tokens.items():
                tracked[label] = float(final_probs[tid].item())
        results["final"] = {"top_k": top_tokens, "tracked": tracked}

        return results

    # ── Build tracked token dict ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Building tracked token dictionary")
    print(f"{'─'*70}")

    # CWE-787 tokens
    cwe787_tokens = {}
    for label, text in [("snprintf_space", " snprintf"), ("snprintf", "snprintf"),
                         ("sprintf_space", " sprintf"), ("sprintf", "sprintf"),
                         ("sn", "sn"), ("printf", "printf"), ("printf_space", " printf"),
                         ("str", "str"), ("n", "n")]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cwe787_tokens[label] = ids[0]
        # Also track ALL subtokens if multi-token
        if len(ids) > 1:
            for i, tid in enumerate(ids):
                cwe787_tokens[f"{label}_part{i}"] = tid
        print(f"  {label}: '{text}' -> id={ids[0]} ('{tokenizer.decode([ids[0]])}')"
              f"{' [MULTI: ' + str(ids) + ']' if len(ids) > 1 else ''}")

    # CWE-89 tokens
    cwe89_tokens = {}
    for label, text in [("question_mark", "?"), ("execute", "execute"),
                         ("parameterized", "parameterized"), ("placeholder", "placeholder"),
                         ("percent_s", "%s"), ("cursor", "cursor"),
                         ("execute_space", " execute"), ("cursor_space", " cursor")]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cwe89_tokens[label] = ids[0]
        if len(ids) > 1:
            for i, tid in enumerate(ids):
                cwe89_tokens[f"{label}_part{i}"] = tid
        print(f"  {label}: '{text}' -> id={ids[0]} ('{tokenizer.decode([ids[0]])}')"
              f"{' [MULTI: ' + str(ids) + ']' if len(ids) > 1 else ''}")

    LAYERS_TO_CHECK = [0, 16, 31]

    # ── CWE-787: 3 secure + 3 insecure ──────────────────────────────────
    print(f"\n{'─'*70}")
    print("CWE-787: Top-20 Logit Lens Analysis")
    print(f"{'─'*70}")

    # Pick first 3 unique base_ids
    seen = set()
    cwe787_selected = []
    for item in cwe787_data:
        if item["base_id"] not in seen and len(cwe787_selected) < 3:
            cwe787_selected.append(item)
            seen.add(item["base_id"])

    cwe787_results = []
    for item in cwe787_selected:
        print(f"\n  === Prompt: {item['id']} (base: {item['base_id']}) ===")

        for prompt_type, code in [("secure", item["secure"]),
                                   ("vulnerable", item["vulnerable"])]:
            formatted = format_prompt(code)
            result = logit_lens_topk(model, tokenizer, device, formatted,
                                     LAYERS_TO_CHECK, k=20,
                                     track_tokens=cwe787_tokens)

            print(f"\n    [{prompt_type.upper()}]")
            for layer in LAYERS_TO_CHECK + ["final"]:
                layer_label = f"Layer {layer}" if isinstance(layer, int) else "Final"
                print(f"\n      {layer_label} — Top 20:")
                for i, (tok, prob) in enumerate(result[layer]["top_k"]):
                    marker = " <<<<" if tok.strip() in ["snprintf", "sprintf", "sn", "printf"] else ""
                    print(f"        {i+1:2d}. '{tok}' = {prob:.6f} ({prob*100:.3f}%){marker}")

                print(f"      Tracked tokens:")
                for label, prob in sorted(result[layer]["tracked"].items(),
                                          key=lambda x: -x[1]):
                    if prob > 1e-8:
                        print(f"        {label}: {prob:.8f} ({prob*100:.5f}%)")

            cwe787_results.append({
                "id": item["id"], "base_id": item["base_id"],
                "prompt_type": prompt_type, "result": result
            })

    # ── CWE-89: 3 secure + 3 insecure ──────────────────────────────────
    print(f"\n{'─'*70}")
    print("CWE-89: Top-20 Logit Lens Analysis")
    print(f"{'─'*70}")

    seen = set()
    cwe89_selected = []
    for item in cwe89_data:
        if item["base_id"] not in seen and len(cwe89_selected) < 3:
            cwe89_selected.append(item)
            seen.add(item["base_id"])

    cwe89_results = []
    for item in cwe89_selected:
        print(f"\n  === Prompt: {item['pair_id']} (base: {item['base_id']}) ===")

        for prompt_type, code in [("secure", item["secure_prompt"]),
                                   ("insecure", item["insecure_prompt"])]:
            formatted = format_prompt(code)
            result = logit_lens_topk(model, tokenizer, device, formatted,
                                     LAYERS_TO_CHECK, k=20,
                                     track_tokens=cwe89_tokens)

            print(f"\n    [{prompt_type.upper()}]")
            for layer in LAYERS_TO_CHECK + ["final"]:
                layer_label = f"Layer {layer}" if isinstance(layer, int) else "Final"
                print(f"\n      {layer_label} — Top 20:")
                for i, (tok, prob) in enumerate(result[layer]["top_k"]):
                    print(f"        {i+1:2d}. '{tok}' = {prob:.6f} ({prob*100:.3f}%)")

                print(f"      Tracked tokens:")
                for label, prob in sorted(result[layer]["tracked"].items(),
                                          key=lambda x: -x[1]):
                    if prob > 1e-8:
                        print(f"        {label}: {prob:.8f} ({prob*100:.5f}%)")

            cwe89_results.append({
                "id": item["pair_id"], "base_id": item["base_id"],
                "prompt_type": prompt_type, "result": result
            })

    loader.unload()

    return {
        "cwe787": cwe787_results,
        "cwe89": cwe89_results,
        "architecture": {
            "tie_word_embeddings": model.config.tie_word_embeddings,
            "actually_tied": tied,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Diagnosis
# ═══════════════════════════════════════════════════════════════════════════

def phase3_diagnosis():
    print(f"\n\n{'='*70}")
    print("PHASE 3: Diagnosis Summary")
    print("=" * 70)
    print("""
    Check the output above for:

    A. TOKEN ID BUG: Did find_best_secure_tokens() return the wrong ID?
       - If " snprintf" splits into multiple tokens, ids[0] is only a prefix
       - Compare: Llama " snprintf" tokens vs Mistral " snprintf" tokens

    B. TOKENIZATION SPLITTING BUG: Does "snprintf" split differently?
       - If Mistral tokenizes " snprintf" as [" sn", "printf"], tracking only
         the first subtoken " sn" would give near-zero probability

    C. REAL FINDING: Does Mistral distribute probability differently?
       - Check top-20: is the probability going to a different but related token?
       - Maybe Mistral uses "str" + "n" + "copy" instead of "snprintf"

    D. UNEMBEDDING MATRIX ISSUE: Is lm_head correct?
       - Check tie_word_embeddings config
       - Check if weights are actually shared in memory
    """)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Phase 1: no GPU
    phase1_results = phase1()

    # Phase 2: needs GPU
    phase2_results = phase2()

    # Phase 3: diagnosis
    phase3_diagnosis()

    print(f"\nInvestigation complete at {timestamp}")
