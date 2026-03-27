#!/usr/bin/env python3
"""
Experiment R12: GPT-2 Reconciliation with Hoang's Results
==========================================================
BLOCKING: Must resolve discrepancy before paper merge.

Gus's Experiment 10: GPT-2 shows 0% error rate
Hoang's results: GPT-2-Small shows 100% error rate

Root cause hypothesis: Gus used text-based evaluation on a base model that
gives incoherent outputs → "unclear" classified as "no bug". Hoang used
logit-based evaluation revealing the model's actual preference.

Sections:
  1. Reproduce Gus's original GPT-2 result
  2. Test Hoang's prompt on GPT-2-Small
  3. Test all GPT-2 variants
  4. Evaluate the evaluation (logit vs text)
  5. Test Hoang's 5-head circuit on Gus's prompts
  6. MLP investigation on GPT-2
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")

GPT2_MODELS = {
    "gpt2": "GPT-2-Small (124M)",
    "gpt2-medium": "GPT-2-Medium (355M)",
    "gpt2-large": "GPT-2-Large (774M)",
    "gpt2-xl": "GPT-2-XL (1.5B)",
}

PROMPT_FORMATS = {
    "hoang_compare": "Compare two numbers: {x}.11 and {x}.8, which one is bigger?",
    "gus_simple": "Which is bigger: {x}.8 or {x}.11? Answer:",
    "gus_qa": "Q: Which is bigger: {x}.8 or {x}.11?\nA:",
}

# Hoang's 5-head circuit for GPT-2-Small (12 layers, 12 heads)
HOANG_CIRCUIT_HEADS = [(2, 2), (5, 1), (6, 11), (9, 9), (10, 2)]

X_VALUES = list(range(1, 10))  # X=1..9
N_TRIALS = 10
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 50
N_PATCHING_TRIALS = 50


def save_checkpoint(data: dict):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=lambda o:
                  int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    os.replace(tmp, CHECKPOINT_PATH)


def load_checkpoint() -> Optional[dict]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class GPT2Experiment:
    """Manages GPT-2 model loading and inference."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        print(f"\nLoading {model_name} ({GPT2_MODELS.get(model_name, 'unknown')})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hooks = []
        self.saved_activations = {}
        print(f"  Loaded on {DEVICE}. Params: {sum(p.numel() for p in self.model.parameters()):,}")

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS,
                 temperature: float = TEMPERATURE) -> str:
        """Generate text completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full[len(prompt):]

    def get_logits(self, prompt: str) -> torch.Tensor:
        """Get logits for the next token after prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Last token logits
        return outputs.logits[0, -1, :]

    def get_token_probs(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """Get probabilities for specific tokens at the next position."""
        logits = self.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)
        result = {}
        for token in tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) >= 1:
                token_id = token_ids[0]
                result[token] = probs[token_id].item()
            else:
                result[token] = 0.0
        return result

    def get_logit_difference(self, prompt: str, correct_num: str, incorrect_num: str) -> Dict:
        """Compute logit difference between correct and incorrect answer tokens.

        For GPT-2, numbers like '1.8' tokenize as multiple tokens (e.g., '1', '.', '8').
        The discriminating tokens are the decimal parts: '8' vs '11' (or ' 8' vs ' 11').
        We ONLY use tokens whose first token ID is unique to one side, to avoid
        shared prefixes (e.g., '1', '.') giving logit_diff=0.
        """
        logits = self.get_logits(prompt)

        correct_decimal = correct_num.split(".")[1]   # "8"
        incorrect_decimal = incorrect_num.split(".")[1]  # "11"

        # Token candidates to check (with and without space prefix)
        candidates = {
            "correct": [correct_num, " " + correct_num, "." + correct_decimal,
                       " ." + correct_decimal, correct_decimal, " " + correct_decimal],
            "incorrect": [incorrect_num, " " + incorrect_num, "." + incorrect_decimal,
                         " ." + incorrect_decimal, incorrect_decimal, " " + incorrect_decimal],
        }

        # Resolve each candidate string to its first token ID
        resolved = {"correct": [], "incorrect": []}
        for side, token_strs in candidates.items():
            for t in token_strs:
                ids = self.tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    resolved[side].append((t, ids[0]))

        # Find token IDs that appear on BOTH sides — these are ambiguous
        correct_ids = {tid for _, tid in resolved["correct"]}
        incorrect_ids = {tid for _, tid in resolved["incorrect"]}
        shared_ids = correct_ids & incorrect_ids

        # Filter to only unique (discriminating) tokens
        best = {"correct": float('-inf'), "incorrect": float('-inf')}
        best_tokens = {"correct": "", "incorrect": ""}

        for side in ["correct", "incorrect"]:
            for t, tid in resolved[side]:
                if tid in shared_ids:
                    continue  # skip ambiguous tokens
                logit_val = logits[tid].item()
                if logit_val > best[side]:
                    best[side] = logit_val
                    best_tokens[side] = t

        return {
            "correct_logit": best["correct"],
            "incorrect_logit": best["incorrect"],
            "logit_diff": best["correct"] - best["incorrect"],
            "correct_preferred": best["correct"] > best["incorrect"],
            "correct_token_used": best_tokens["correct"],
            "incorrect_token_used": best_tokens["incorrect"],
        }

    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.saved_activations.clear()

    def cleanup(self):
        self._clear_hooks()
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache() if DEVICE == "cuda" else None


# ── Evaluation Functions ────────────────────────────────────────────

def evaluate_gus_original(output: str, correct_answer: str, incorrect_answer: str) -> str:
    """Gus's EXACT original evaluation from Experiment 10.

    This is a cascading if/elif — first checks for correct answer.
    Note: the word list includes the number itself, so if the number appears
    at all, the condition is trivially true.
    """
    out_lower = output.lower()
    # Gus's original: any(w in response.lower() for w in ["bigger", "larger", "greater", "9.8"])
    # The number itself is in the word list! So presence of the number = match
    if correct_answer in output and any(w in out_lower for w in ["bigger", "larger", "greater", correct_answer]):
        return "correct"
    elif incorrect_answer in output and any(w in out_lower for w in ["bigger", "larger", "greater", incorrect_answer]):
        return "incorrect"
    else:
        return "unclear"


def evaluate_gus_text(output: str, correct_answer: str, incorrect_answer: str) -> str:
    """Stricter text-based evaluation — requires answer word, handles both-present case."""
    out_lower = output.lower()
    has_correct = correct_answer in output
    has_incorrect = incorrect_answer in output
    answer_words = ["bigger", "larger", "greater"]
    has_answer_word = any(w in out_lower for w in answer_words)

    if has_correct and has_answer_word and not has_incorrect:
        return "correct"
    elif has_incorrect and has_answer_word and not has_correct:
        return "incorrect"
    elif has_correct and has_incorrect:
        return "ambiguous"
    else:
        return "unclear"


def evaluate_first_number(output: str, correct_answer: str, incorrect_answer: str) -> str:
    """Which number appears first in the output?"""
    pos_correct = output.find(correct_answer)
    pos_incorrect = output.find(incorrect_answer)
    if pos_correct == -1 and pos_incorrect == -1:
        return "neither"
    elif pos_correct == -1:
        return "incorrect_first"
    elif pos_incorrect == -1:
        return "correct_first"
    elif pos_correct < pos_incorrect:
        return "correct_first"
    else:
        return "incorrect_first"


def eval_patched_logits(patched_logits, tokenizer, correct_num: str, incorrect_num: str) -> Dict:
    """Evaluate patched logits using only discriminating (non-shared) token IDs.

    Same logic as get_logit_difference but takes pre-computed logits tensor.
    """
    correct_decimal = correct_num.split(".")[1]
    incorrect_decimal = incorrect_num.split(".")[1]

    candidates = {
        "correct": [correct_num, " " + correct_num, "." + correct_decimal,
                   " ." + correct_decimal, correct_decimal, " " + correct_decimal],
        "incorrect": [incorrect_num, " " + incorrect_num, "." + incorrect_decimal,
                     " ." + incorrect_decimal, incorrect_decimal, " " + incorrect_decimal],
    }

    resolved = {"correct": [], "incorrect": []}
    for side, token_strs in candidates.items():
        for t in token_strs:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if ids:
                resolved[side].append((t, ids[0]))

    correct_ids = {tid for _, tid in resolved["correct"]}
    incorrect_ids = {tid for _, tid in resolved["incorrect"]}
    shared_ids = correct_ids & incorrect_ids

    best = {"correct": float('-inf'), "incorrect": float('-inf')}
    for side in ["correct", "incorrect"]:
        for t, tid in resolved[side]:
            if tid in shared_ids:
                continue
            logit_val = patched_logits[tid].item()
            if logit_val > best[side]:
                best[side] = logit_val

    patched_diff = best["correct"] - best["incorrect"]
    return {
        "patched_correct_logit": best["correct"],
        "patched_incorrect_logit": best["incorrect"],
        "patched_diff": patched_diff,
        "patched_correct": best["correct"] > best["incorrect"],
    }


# ════════════════════════════════════════════════════════════════════
# Section 1: Reproduce Gus's Original GPT-2 Result
# ════════════════════════════════════════════════════════════════════

def section1_reproduce_gus(results: dict):
    """Reproduce Gus's original test exactly."""
    print("\n" + "="*70)
    print("SECTION 1: Reproduce Gus's Original GPT-2 Result")
    print("="*70)

    section = {}

    for model_name in ["gpt2", "gpt2-medium"]:
        exp = GPT2Experiment(model_name)
        prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"
        model_results = {
            "prompt": prompt,
            "temperature": 0.2,
            "max_new_tokens": 20,
            "n_trials": 10,
            "raw_outputs": [],
            "gus_original_eval": {"correct": 0, "incorrect": 0, "unclear": 0},
            "strict_eval": {"correct": 0, "incorrect": 0, "unclear": 0, "ambiguous": 0},
            "logit_eval": None,
        }

        print(f"\n  Testing {model_name} with Gus's exact config (temp=0.2, 10 runs):")
        for i in range(10):
            response = exp.generate(prompt, max_new_tokens=20, temperature=0.2)
            gus_cls = evaluate_gus_original(response, "9.8", "9.11")
            strict_cls = evaluate_gus_text(response, "9.8", "9.11")
            model_results["raw_outputs"].append(response[:100])
            model_results["gus_original_eval"][gus_cls] += 1
            model_results["strict_eval"][strict_cls] = model_results["strict_eval"].get(strict_cls, 0) + 1
            if i < 5:
                print(f"    Run {i+1}: gus=[{gus_cls:>10}] strict=[{strict_cls:>10}] {response[:60]!r}")

        # Logit-based evaluation
        logit_data = exp.get_logit_difference(prompt, "9.8", "9.11")
        model_results["logit_eval"] = logit_data

        gus = model_results["gus_original_eval"]
        strict = model_results["strict_eval"]
        total = 10
        print(f"\n  Summary for {model_name}:")
        print(f"    Gus original: correct={gus['correct']}/{total}, bug={gus['incorrect']}/{total}, unclear={gus['unclear']}/{total}")
        print(f"    Strict text:  correct={strict.get('correct',0)}/{total}, bug={strict.get('incorrect',0)}/{total}, unclear={strict.get('unclear',0)}/{total}, ambiguous={strict.get('ambiguous',0)}/{total}")
        print(f"    Logit eval:   diff={logit_data['logit_diff']:.3f} ({'correct' if logit_data['correct_preferred'] else 'BUG'})")

        section[model_name] = model_results
        exp.cleanup()

    results["section1_reproduce_gus"] = section
    save_checkpoint(results)


# ════════════════════════════════════════════════════════════════════
# Section 2: Test Hoang's Prompt on GPT-2-Small
# ════════════════════════════════════════════════════════════════════

def section2_hoang_prompt(results: dict):
    """Test Hoang's prompt format on GPT-2-Small."""
    print("\n" + "="*70)
    print("SECTION 2: Test Hoang's Prompt on GPT-2-Small")
    print("="*70)

    exp = GPT2Experiment("gpt2")
    section = {}

    for fmt_name, fmt_template in PROMPT_FORMATS.items():
        fmt_results = {
            "template": fmt_template,
            "by_x": {},
            "overall_text_eval": {"correct": 0, "incorrect": 0, "unclear": 0, "ambiguous": 0},
            "overall_logit_eval": {"correct": 0, "incorrect": 0},
        }

        print(f"\n  Format: {fmt_name}")
        print(f"  Template: {fmt_template!r}")

        for x in X_VALUES:
            prompt = fmt_template.format(x=x)
            correct_num = f"{x}.8"
            incorrect_num = f"{x}.11"

            x_data = {
                "prompt": prompt,
                "raw_outputs": [],
                "text_classifications": [],
                "logit_eval": None,
            }

            # Text-based evaluation (n=10, temp=0.0 since base model)
            for trial in range(N_TRIALS):
                response = exp.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
                cls = evaluate_gus_text(response, correct_num, incorrect_num)
                x_data["raw_outputs"].append(response[:100])
                x_data["text_classifications"].append(cls)
                fmt_results["overall_text_eval"][cls] = fmt_results["overall_text_eval"].get(cls, 0) + 1

            # Logit-based evaluation (deterministic, single run)
            logit_data = exp.get_logit_difference(prompt, correct_num, incorrect_num)
            x_data["logit_eval"] = logit_data
            if logit_data["correct_preferred"]:
                fmt_results["overall_logit_eval"]["correct"] += 1
            else:
                fmt_results["overall_logit_eval"]["incorrect"] += 1

            fmt_results["by_x"][str(x)] = x_data

            text_cls_summary = {}
            for c in x_data["text_classifications"]:
                text_cls_summary[c] = text_cls_summary.get(c, 0) + 1

            logit_sym = "✓" if logit_data["correct_preferred"] else "✗"
            print(f"    X={x}: text={text_cls_summary} | logit_diff={logit_data['logit_diff']:.3f} [{logit_sym}]")

        # Summary
        te = fmt_results["overall_text_eval"]
        le = fmt_results["overall_logit_eval"]
        total_text = sum(te.values())
        total_logit = sum(le.values())
        print(f"\n  {fmt_name} Summary:")
        print(f"    Text eval - Correct: {te.get('correct',0)}/{total_text}, Bug: {te.get('incorrect',0)}/{total_text}, Unclear: {te.get('unclear',0)}/{total_text}")
        print(f"    Logit eval - Correct: {le['correct']}/{total_logit}, Bug: {le['incorrect']}/{total_logit}")

        section[fmt_name] = fmt_results

    results["section2_hoang_prompt"] = section
    exp.cleanup()
    save_checkpoint(results)


# ════════════════════════════════════════════════════════════════════
# Section 3: Test All GPT-2 Variants
# ════════════════════════════════════════════════════════════════════

def section3_all_variants(results: dict):
    """Test all GPT-2 variants with all prompt formats."""
    print("\n" + "="*70)
    print("SECTION 3: Test All GPT-2 Variants")
    print("="*70)

    section = {}

    for model_name, model_desc in GPT2_MODELS.items():
        print(f"\n  ── {model_desc} ──")
        exp = GPT2Experiment(model_name)
        model_results = {}

        for fmt_name, fmt_template in PROMPT_FORMATS.items():
            fmt_data = {
                "text_correct": 0, "text_incorrect": 0, "text_unclear": 0,
                "logit_correct": 0, "logit_incorrect": 0,
                "by_x": {},
            }

            for x in X_VALUES:
                prompt = fmt_template.format(x=x)
                correct_num = f"{x}.8"
                incorrect_num = f"{x}.11"

                # Text eval (temp=0.0, single run since deterministic)
                response = exp.generate(prompt, temperature=0.0)
                cls = evaluate_gus_text(response, correct_num, incorrect_num)
                if cls == "correct":
                    fmt_data["text_correct"] += 1
                elif cls == "incorrect":
                    fmt_data["text_incorrect"] += 1
                else:
                    fmt_data["text_unclear"] += 1

                # Logit eval
                logit_data = exp.get_logit_difference(prompt, correct_num, incorrect_num)
                if logit_data["correct_preferred"]:
                    fmt_data["logit_correct"] += 1
                else:
                    fmt_data["logit_incorrect"] += 1

                fmt_data["by_x"][str(x)] = {
                    "raw_output": response[:100],
                    "text_cls": cls,
                    "logit_diff": logit_data["logit_diff"],
                    "logit_correct": logit_data["correct_preferred"],
                }

            model_results[fmt_name] = fmt_data
            print(f"    {fmt_name}: text={fmt_data['text_correct']}/{9} correct, "
                  f"{fmt_data['text_incorrect']}/{9} bug, {fmt_data['text_unclear']}/{9} unclear | "
                  f"logit={fmt_data['logit_correct']}/{9} correct, {fmt_data['logit_incorrect']}/{9} bug")

        section[model_name] = model_results
        exp.cleanup()

    results["section3_all_variants"] = section
    save_checkpoint(results)


# ════════════════════════════════════════════════════════════════════
# Section 4: Evaluate the Evaluation
# ════════════════════════════════════════════════════════════════════

def section4_evaluation_comparison(results: dict):
    """Compare different evaluation methods on same outputs."""
    print("\n" + "="*70)
    print("SECTION 4: Evaluate the Evaluation")
    print("="*70)

    exp = GPT2Experiment("gpt2")
    section = {"by_x": {}, "top_tokens": {}}

    print("\n  Detailed analysis of GPT-2-Small outputs and logits:")

    for x in X_VALUES:
        correct_num = f"{x}.8"
        incorrect_num = f"{x}.11"
        x_data = {}

        for fmt_name, fmt_template in PROMPT_FORMATS.items():
            prompt = fmt_template.format(x=x)
            response = exp.generate(prompt, temperature=0.0)

            # Logit analysis
            logits = exp.get_logits(prompt)
            probs = torch.softmax(logits, dim=-1)

            # Top-10 tokens
            top_probs, top_ids = probs.topk(10)
            top_tokens = []
            for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
                token_str = exp.tokenizer.decode([tid])
                top_tokens.append({"token": token_str, "prob": p, "id": tid})

            # Specific token probs
            check_tokens = [correct_num, incorrect_num, " " + correct_num, " " + incorrect_num,
                            "8", "11", " 8", " 11", ".8", ".11", " .8", " .11"]
            token_probs = {}
            for t in check_tokens:
                ids = exp.tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    token_probs[t] = probs[ids[0]].item()

            logit_data = exp.get_logit_difference(prompt, correct_num, incorrect_num)

            x_data[fmt_name] = {
                "prompt": prompt,
                "raw_output": response[:150],
                "text_eval": evaluate_gus_text(response, correct_num, incorrect_num),
                "first_number_eval": evaluate_first_number(response, correct_num, incorrect_num),
                "logit_diff": logit_data["logit_diff"],
                "logit_correct": logit_data["correct_preferred"],
                "top_10_tokens": top_tokens,
                "specific_token_probs": token_probs,
            }

        section["by_x"][str(x)] = x_data

        # Print summary for this x
        for fmt_name in PROMPT_FORMATS:
            d = x_data[fmt_name]
            print(f"\n    X={x}, {fmt_name}:")
            print(f"      Output: {d['raw_output'][:80]!r}")
            print(f"      Text eval: {d['text_eval']}, First num: {d['first_number_eval']}")
            print(f"      Logit diff: {d['logit_diff']:.3f} ({'correct' if d['logit_correct'] else 'BUG'})")
            top5 = [(t['token'], round(t['prob'], 4)) for t in d['top_10_tokens'][:5]]
            print(f"      Top-5 tokens: {top5}")

    # Aggregate comparison
    eval_methods = {"text": {"correct": 0, "bug": 0, "other": 0},
                    "logit": {"correct": 0, "bug": 0},
                    "first_number": {"correct": 0, "bug": 0, "other": 0}}

    for x_str, x_data in section["by_x"].items():
        for fmt_name, d in x_data.items():
            # Text
            if d["text_eval"] == "correct":
                eval_methods["text"]["correct"] += 1
            elif d["text_eval"] == "incorrect":
                eval_methods["text"]["bug"] += 1
            else:
                eval_methods["text"]["other"] += 1
            # Logit
            if d["logit_correct"]:
                eval_methods["logit"]["correct"] += 1
            else:
                eval_methods["logit"]["bug"] += 1
            # First number
            if d["first_number_eval"] == "correct_first":
                eval_methods["first_number"]["correct"] += 1
            elif d["first_number_eval"] == "incorrect_first":
                eval_methods["first_number"]["bug"] += 1
            else:
                eval_methods["first_number"]["other"] += 1

    section["eval_comparison"] = eval_methods
    print(f"\n  EVALUATION METHOD COMPARISON (across all X and formats):")
    for method, counts in eval_methods.items():
        print(f"    {method}: {counts}")

    results["section4_evaluation"] = section
    exp.cleanup()
    save_checkpoint(results)


# ════════════════════════════════════════════════════════════════════
# Section 5: Test Hoang's 5-Head Circuit on GPT-2-Small
# ════════════════════════════════════════════════════════════════════

def section5_hoang_circuit(results: dict):
    """Test Hoang's 5-head circuit patching on GPT-2-Small."""
    print("\n" + "="*70)
    print("SECTION 5: Test Hoang's 5-Head Circuit on GPT-2-Small")
    print("="*70)
    print(f"  Circuit heads: {HOANG_CIRCUIT_HEADS}")

    exp = GPT2Experiment("gpt2")
    n_layers = exp.model.config.n_layer  # 12
    n_heads = exp.model.config.n_head    # 12
    head_dim = exp.model.config.n_embd // n_heads  # 64

    section = {"circuit_heads": HOANG_CIRCUIT_HEADS, "by_format": {}}

    # For activation patching:
    # Clean input = reversed prompt where correct answer is natural (X.8 vs X.11)
    # Corrupted input = standard prompt where bug occurs (X.11 vs X.8)

    for fmt_name, fmt_template in PROMPT_FORMATS.items():
        fmt_results = {"by_x": {}}

        print(f"\n  Format: {fmt_name}")

        for x in X_VALUES:
            correct_num = f"{x}.8"
            incorrect_num = f"{x}.11"

            # Standard (buggy) prompt
            buggy_prompt = fmt_template.format(x=x)

            # Reversed (clean) prompt - swap the numbers
            if "hoang_compare" in fmt_name:
                clean_prompt = f"Compare two numbers: {x}.8 and {x}.11, which one is bigger?"
            elif "gus_simple" in fmt_name:
                clean_prompt = f"Which is bigger: {x}.11 or {x}.8? Answer:"
            else:
                clean_prompt = f"Q: Which is bigger: {x}.11 or {x}.8?\nA:"

            # Baseline logit diff (no patching)
            baseline = exp.get_logit_difference(buggy_prompt, correct_num, incorrect_num)

            # Get clean activations from attention heads
            clean_inputs = exp.tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
            buggy_inputs = exp.tokenizer(buggy_prompt, return_tensors="pt").to(DEVICE)

            # Save clean attention outputs
            clean_head_outputs = {}
            hooks = []

            def make_save_hook(layer_idx):
                def hook_fn(module, inp, out):
                    # out is (attn_output, attn_weights, past_key_value) or similar
                    attn_out = out[0]  # [batch, seq, hidden]
                    clean_head_outputs[layer_idx] = attn_out.detach().clone()
                return hook_fn

            for layer_idx, head_idx in HOANG_CIRCUIT_HEADS:
                attn_module = exp.model.transformer.h[layer_idx].attn
                h = attn_module.register_forward_hook(make_save_hook(layer_idx))
                hooks.append(h)

            with torch.no_grad():
                exp.model(**clean_inputs)

            for h in hooks:
                h.remove()
            hooks.clear()

            # Now run buggy prompt with patching
            def make_patch_hook(layer_idx, head_idx):
                def hook_fn(module, inp, out):
                    attn_out = out[0]  # [batch, seq, hidden]
                    if layer_idx in clean_head_outputs:
                        clean_val = clean_head_outputs[layer_idx].to(attn_out.device)
                        # Patch specific head
                        start = head_idx * head_dim
                        end = (head_idx + 1) * head_dim
                        min_seq = min(attn_out.shape[1], clean_val.shape[1])
                        patched = attn_out.clone()
                        patched[0, :min_seq, start:end] = clean_val[0, :min_seq, start:end]
                        return (patched,) + out[1:]
                    return out
                return hook_fn

            for layer_idx, head_idx in HOANG_CIRCUIT_HEADS:
                attn_module = exp.model.transformer.h[layer_idx].attn
                h = attn_module.register_forward_hook(make_patch_hook(layer_idx, head_idx))
                hooks.append(h)

            # Get patched logits
            with torch.no_grad():
                patched_outputs = exp.model(**buggy_inputs)

            patched_logits = patched_outputs.logits[0, -1, :]

            for h in hooks:
                h.remove()
            hooks.clear()

            # Evaluate patched result using discriminating tokens only
            patched_eval = eval_patched_logits(patched_logits, exp.tokenizer, correct_num, incorrect_num)
            patched_diff = patched_eval["patched_diff"]

            x_data = {
                "baseline_logit_diff": baseline["logit_diff"],
                "baseline_correct": baseline["correct_preferred"],
                "patched_logit_diff": patched_diff,
                "patched_correct": patched_eval["patched_correct"],
                "improvement": patched_diff - baseline["logit_diff"],
            }

            fmt_results["by_x"][str(x)] = x_data
            base_sym = "✓" if baseline["correct_preferred"] else "✗"
            patch_sym = "✓" if patched_eval["patched_correct"] else "✗"
            print(f"    X={x}: baseline={baseline['logit_diff']:.3f}[{base_sym}] → patched={patched_diff:.3f}[{patch_sym}] (Δ={patched_diff - baseline['logit_diff']:+.3f})")

            clean_head_outputs.clear()

        # Aggregate
        n_baseline_correct = sum(1 for v in fmt_results["by_x"].values() if v["baseline_correct"])
        n_patched_correct = sum(1 for v in fmt_results["by_x"].values() if v["patched_correct"])
        fmt_results["summary"] = {
            "baseline_correct": n_baseline_correct,
            "patched_correct": n_patched_correct,
            "total": len(X_VALUES),
        }
        print(f"  {fmt_name} summary: baseline {n_baseline_correct}/9 correct → patched {n_patched_correct}/9 correct")

        section["by_format"][fmt_name] = fmt_results

    results["section5_hoang_circuit"] = section
    exp.cleanup()
    save_checkpoint(results)


# ════════════════════════════════════════════════════════════════════
# Section 6: MLP Investigation on GPT-2
# ════════════════════════════════════════════════════════════════════

def section6_mlp_investigation(results: dict):
    """MLP patching and logit lens on GPT-2-Small."""
    print("\n" + "="*70)
    print("SECTION 6: MLP Investigation on GPT-2-Small")
    print("="*70)

    exp = GPT2Experiment("gpt2")
    n_layers = exp.model.config.n_layer  # 12
    head_dim = exp.model.config.n_embd // exp.model.config.n_head

    section = {"mlp_patching": {}, "logit_lens": {}}

    # ── Part A: MLP-only patching sweep ──
    print("\n  Part A: MLP patching sweep across all 12 layers")

    for fmt_name, fmt_template in list(PROMPT_FORMATS.items())[:2]:  # hoang + gus_simple
        fmt_results = {}

        print(f"\n    Format: {fmt_name}")

        for layer_idx in range(n_layers):
            layer_data = {"by_x": {}, "n_correct_baseline": 0, "n_correct_patched": 0}

            for x in X_VALUES:
                correct_num = f"{x}.8"
                incorrect_num = f"{x}.11"

                buggy_prompt = fmt_template.format(x=x)
                if "hoang_compare" in fmt_name:
                    clean_prompt = f"Compare two numbers: {x}.8 and {x}.11, which one is bigger?"
                else:
                    clean_prompt = f"Which is bigger: {x}.11 or {x}.8? Answer:"

                clean_inputs = exp.tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
                buggy_inputs = exp.tokenizer(buggy_prompt, return_tensors="pt").to(DEVICE)

                # Save clean MLP output
                clean_mlp_out = {}
                hooks = []

                def make_save_mlp(l_idx):
                    def hook_fn(module, inp, out):
                        clean_mlp_out[l_idx] = out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone()
                    return hook_fn

                mlp_module = exp.model.transformer.h[layer_idx].mlp
                h = mlp_module.register_forward_hook(make_save_mlp(layer_idx))
                hooks.append(h)

                with torch.no_grad():
                    exp.model(**clean_inputs)

                for h in hooks:
                    h.remove()
                hooks.clear()

                # Baseline
                baseline = exp.get_logit_difference(buggy_prompt, correct_num, incorrect_num)

                # Patch MLP
                def make_patch_mlp(l_idx):
                    def hook_fn(module, inp, out):
                        clean_val = clean_mlp_out[l_idx].to(out[0].device if isinstance(out, tuple) else out.device)
                        actual_out = out[0] if isinstance(out, tuple) else out
                        min_seq = min(actual_out.shape[1], clean_val.shape[1])
                        patched = actual_out.clone()
                        patched[0, :min_seq, :] = clean_val[0, :min_seq, :]
                        if isinstance(out, tuple):
                            return (patched,) + out[1:]
                        return patched
                    return hook_fn

                h = mlp_module.register_forward_hook(make_patch_mlp(layer_idx))
                hooks.append(h)

                with torch.no_grad():
                    patched_out = exp.model(**buggy_inputs)

                patched_logits = patched_out.logits[0, -1, :]
                for h in hooks:
                    h.remove()
                hooks.clear()

                patched_eval = eval_patched_logits(patched_logits, exp.tokenizer, correct_num, incorrect_num)
                pc = patched_eval["patched_correct_logit"]
                pi = patched_eval["patched_incorrect_logit"]
                patched_diff = patched_eval["patched_diff"]

                if baseline["correct_preferred"]:
                    layer_data["n_correct_baseline"] += 1
                if pc > pi:
                    layer_data["n_correct_patched"] += 1

                layer_data["by_x"][str(x)] = {
                    "baseline_diff": baseline["logit_diff"],
                    "patched_diff": patched_diff,
                }

                clean_mlp_out.clear()

            fmt_results[f"layer_{layer_idx}"] = layer_data
            b = layer_data["n_correct_baseline"]
            p = layer_data["n_correct_patched"]
            delta = p - b
            marker = " <<<" if delta >= 3 else ""
            print(f"      Layer {layer_idx:2d}: baseline {b}/9 → patched {p}/9 (Δ={delta:+d}){marker}")

        section["mlp_patching"][fmt_name] = fmt_results

    # ── Part B: Logit lens at key layers ──
    print("\n  Part B: Logit lens through MLP at key layers")

    for x in [1, 5, 9]:
        prompt = f"Compare two numbers: {x}.11 and {x}.8, which one is bigger?"
        inputs = exp.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Get hidden states at each layer
        with torch.no_grad():
            outputs = exp.model(**inputs, output_hidden_states=True)

        x_data = {}
        lm_head = exp.model.lm_head
        ln_f = exp.model.transformer.ln_f

        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings
            # Apply final layer norm and project to vocab
            normed = ln_f(hidden)
            layer_logits = lm_head(normed)[0, -1, :]
            layer_probs = torch.softmax(layer_logits, dim=-1)

            # Check relevant tokens
            correct_num = f"{x}.8"
            incorrect_num = f"{x}.11"
            check_tokens = [" 8", " 11", ".8", ".11", " " + correct_num, " " + incorrect_num]
            token_probs = {}
            for t in check_tokens:
                ids = exp.tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    token_probs[t] = layer_probs[ids[0]].item()

            x_data[f"layer_{layer_idx}"] = token_probs

        section["logit_lens"][f"x={x}"] = x_data
        print(f"\n    X={x}: Logit lens token probs at key layers:")
        for layer_idx in [0, 3, 6, 9, 11]:
            ldata = x_data[f"layer_{layer_idx}"]
            print(f"      L{layer_idx:2d}: {dict((k, f'{v:.4f}') for k, v in ldata.items())}")

    results["section6_mlp_investigation"] = section
    exp.cleanup()
    save_checkpoint(results)


# ════════════════════════════════════════════════════════════════════
# Summary and Output
# ════════════════════════════════════════════════════════════════════

def generate_summary(results: dict) -> str:
    """Generate the reconciliation summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT R12: GPT-2 RECONCILIATION — FINAL SUMMARY")
    lines.append("=" * 70)

    # Table 1: Reconciliation
    lines.append("\n## Reconciliation Results\n")
    lines.append("| Model | Prompt Format | Error Rate (Gus Exp10) | Error Rate (Hoang) | Error Rate (Text) | Error Rate (Logit) |")
    lines.append("|-------|--------------|----------------------|-------------------|-------------------|-------------------|")

    if "section3_all_variants" in results:
        s3 = results["section3_all_variants"]
        for model_name in GPT2_MODELS:
            if model_name in s3:
                for fmt_name in PROMPT_FORMATS:
                    if fmt_name in s3[model_name]:
                        d = s3[model_name][fmt_name]
                        gus_exp10 = "0%" if model_name in ["gpt2", "gpt2-medium"] else "N/A"
                        hoang = "100%" if model_name == "gpt2" and fmt_name in ["hoang_compare", "gus_simple"] else "N/A"
                        text_err = f"{d['text_incorrect']*100//9:.0f}%" if d['text_incorrect'] > 0 else f"{d['text_incorrect']}/9"
                        logit_err = f"{d['logit_incorrect']*100//9:.0f}%"
                        lines.append(f"| {model_name} | {fmt_name} | {gus_exp10} | {hoang} | {text_err} | {logit_err} |")

    # Evaluation comparison
    if "section4_evaluation" in results:
        s4 = results["section4_evaluation"]
        if "eval_comparison" in s4:
            lines.append("\n## Evaluation Method Comparison (GPT-2-Small, all formats, X=1-9)\n")
            for method, counts in s4["eval_comparison"].items():
                lines.append(f"  {method}: {counts}")

    # Circuit patching
    if "section5_hoang_circuit" in results:
        s5 = results["section5_hoang_circuit"]
        lines.append("\n## Hoang's 5-Head Circuit Patching Results\n")
        for fmt_name, fmt_data in s5.get("by_format", {}).items():
            s = fmt_data.get("summary", {})
            lines.append(f"  {fmt_name}: baseline {s.get('baseline_correct','?')}/9 → patched {s.get('patched_correct','?')}/9")

    # MLP
    if "section6_mlp_investigation" in results:
        s6 = results["section6_mlp_investigation"]
        lines.append("\n## MLP Patching Results (GPT-2-Small)\n")
        for fmt_name, fmt_data in s6.get("mlp_patching", {}).items():
            lines.append(f"\n  Format: {fmt_name}")
            for layer_key in sorted(fmt_data.keys(), key=lambda k: int(k.split("_")[1])):
                ld = fmt_data[layer_key]
                b = ld["n_correct_baseline"]
                p = ld["n_correct_patched"]
                delta = p - b
                if abs(delta) >= 2:
                    lines.append(f"    {layer_key}: {b}/9 → {p}/9 (Δ={delta:+d}) <<<")

    summary = "\n".join(lines)
    return summary


def main():
    print("=" * 70)
    print("EXPERIMENT R12: GPT-2 Reconciliation with Hoang's Results")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    start_time = time.time()

    # Check for checkpoint
    results = load_checkpoint()
    if results and "completed_sections" in results:
        completed = set(results["completed_sections"])
        print(f"\nResuming from checkpoint. Completed sections: {completed}")
    else:
        results = {
            "experiment": "R12_gpt2_reconciliation",
            "started": datetime.now().isoformat(),
            "device": DEVICE,
            "completed_sections": [],
        }
        completed = set()

    # Run sections
    sections = [
        ("section1", section1_reproduce_gus),
        ("section2", section2_hoang_prompt),
        ("section3", section3_all_variants),
        ("section4", section4_evaluation_comparison),
        ("section5", section5_hoang_circuit),
        ("section6", section6_mlp_investigation),
    ]

    for name, func in sections:
        if name in completed:
            print(f"\n  Skipping {name} (already completed)")
            continue
        try:
            func(results)
            results["completed_sections"].append(name)
            save_checkpoint(results)
            print(f"\n  ✓ {name} completed")
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[f"{name}_error"] = str(e)
            save_checkpoint(results)

    # Final summary
    summary = generate_summary(results)
    print("\n" + summary)

    results["summary"] = summary
    results["completed"] = datetime.now().isoformat()
    results["total_time_seconds"] = time.time() - start_time

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o:
                  int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating)
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nResults saved to: {results_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
