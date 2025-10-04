#!/usr/bin/env python3
"""
Verify with Llama 8B Instruct using same corrected methodology
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path("/home/paperspace/dev/MATS9/pythia_clustering_dynamics")
RESULTS_DIR = BASE_DIR / "results"

def main():
    """Test Llama 8B Instruct with same methodology"""

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"🔧 Loading {model_name}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    # Same prompts as Pythia test
    clean_prompt = "Which is bigger: 9.8 or 9.11?"
    buggy_prompt = "Q: Which is bigger: 9.8 or 9.11?\nA:"

    def check_bug_fixed(output_text):
        """Same bug detection logic"""
        output_lower = output_text.lower()
        correct_patterns = ["9.8 is bigger", "9.8 is larger", "9.8"]
        bug_patterns = ["9.11 is bigger", "9.11 is larger", "9.11"]

        has_correct = any(pattern in output_lower for pattern in correct_patterns)
        has_bug = any(pattern in output_lower for pattern in bug_patterns)

        return has_correct and not has_bug

    print("\n📊 Testing Llama 8B Baseline...")

    # Test clean prompt
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        clean_outputs = model.generate(
            **clean_inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    clean_response = tokenizer.decode(
        clean_outputs[0][clean_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    clean_fixed = check_bug_fixed(clean_response)

    print(f"  Clean response: '{clean_response.strip()}'")
    print(f"  Bug fixed: {clean_fixed}")

    # Test buggy prompt
    buggy_inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        buggy_outputs = model.generate(
            **buggy_inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    buggy_response = tokenizer.decode(
        buggy_outputs[0][buggy_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    buggy_fixed = check_bug_fixed(buggy_response)

    print(f"  Buggy response: '{buggy_response.strip()}'")
    print(f"  Bug fixed: {buggy_fixed}")
    print(f"  Has bug: {'✅' if not buggy_fixed else '❌'}")

    # Quick even/odd test if bug exists
    if not buggy_fixed:
        print(f"\n⚖️  Quick Even/Odd Test...")

        # Get clean activation for layer 15 (middle layer for Llama)
        target_layer = 15
        attention_module = model.model.layers[target_layer].self_attn
        saved_activation = None

        def save_hook(module, input, output):
            nonlocal saved_activation
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            saved_activation = hidden_states.detach().cpu()

        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to("cuda")
        hook = attention_module.register_forward_hook(save_hook)

        with torch.no_grad():
            model(**clean_inputs)

        hook.remove()
        print(f"  Saved Llama activation: {saved_activation.shape}")

        # Test even heads [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
        even_heads = [i for i in range(32) if i % 2 == 0]  # Llama has 32 heads

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // 32  # 32 heads in Llama

            hidden_reshaped = hidden_states.view(batch_size, seq_len, 32, head_dim)
            saved_reshaped = saved_activation.to(hidden_states.device).view(batch_size, -1, 32, head_dim)

            new_hidden = hidden_reshaped.clone()
            min_seq_len = min(seq_len, saved_reshaped.shape[1])

            for head_idx in even_heads:
                if head_idx < 32:
                    new_hidden[:, :min_seq_len, head_idx, :] = saved_reshaped[:, :min_seq_len, head_idx, :]

            new_hidden = new_hidden.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        hook = attention_module.register_forward_hook(patch_hook)

        inputs = tokenizer(buggy_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        hook.remove()

        even_fixed = check_bug_fixed(response)
        print(f"  Even heads response: '{response.strip()}'")
        print(f"  Even heads fix bug: {'✅' if even_fixed else '❌'}")

    # Save results
    results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'clean_response': clean_response.strip(),
        'buggy_response': buggy_response.strip(),
        'has_bug': not buggy_fixed,
        'clean_fixed': clean_fixed,
        'buggy_fixed': buggy_fixed
    }

    if not buggy_fixed:
        results['even_heads_test'] = {
            'response': response.strip(),
            'fixes_bug': even_fixed
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"llama_8b_verification_{timestamp}.json"

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved Llama results to {filepath}")
    print(f"\n✅ Llama 8B verification complete!")

if __name__ == "__main__":
    main()