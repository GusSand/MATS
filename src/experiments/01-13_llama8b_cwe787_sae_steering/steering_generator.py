"""
Multi-Method Steering Generator

Extends the basic SteeringGenerator to support:
1. Mean-diff steering (from Experiment 2)
2. Single SAE feature steering
3. Top-k SAE feature steering
4. Logit gap computation for forced-choice evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_config import MODEL_NAME, GENERATION_CONFIG, HIDDEN_SIZE


class MultiMethodSteeringGenerator:
    """Generator supporting multiple steering methods and logit gap computation."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hooks = []

        # Cache token IDs for forced-choice evaluation
        self._cache_token_ids()

        print(f"Model loaded: {self.n_layers} layers on {self.device}")

    def _cache_token_ids(self):
        """Cache token IDs for secure/insecure API functions."""
        # sprintf family
        self.snprintf_token = self.tokenizer.encode(" snprintf", add_special_tokens=False)[0]
        self.sprintf_token = self.tokenizer.encode(" sprintf", add_special_tokens=False)[0]

        # strcat family
        self.strncat_token = self.tokenizer.encode(" strncat", add_special_tokens=False)[0]
        self.strcat_token = self.tokenizer.encode(" strcat", add_special_tokens=False)[0]

        print(f"Token IDs cached:")
        print(f"  snprintf: {self.snprintf_token}, sprintf: {self.sprintf_token}")
        print(f"  strncat: {self.strncat_token}, strcat: {self.strcat_token}")

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_with_steering(
        self,
        prompt: str,
        direction: np.ndarray,
        layer: int,
        alpha: float,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Generate text with steering applied at specified layer.

        Args:
            prompt: Input prompt
            direction: Steering direction vector (HIDDEN_SIZE,)
            layer: Layer to apply steering
            alpha: Steering strength multiplier
            temperature: Sampling temperature (default from config)
            top_p: Nucleus sampling parameter (default from config)
            max_tokens: Maximum tokens to generate (default from config)

        Returns:
            Generated text (excluding prompt)
        """
        # Use defaults from config if not specified
        if temperature is None:
            temperature = GENERATION_CONFIG['temperature']
        if top_p is None:
            top_p = GENERATION_CONFIG['top_p']
        if max_tokens is None:
            max_tokens = GENERATION_CONFIG['max_new_tokens']

        # Convert direction to tensor
        direction_tensor = torch.tensor(direction, dtype=torch.float16).to(self.device)

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            # Apply steering to last token position only
            h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor

            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        # Register hook
        self.clear_hooks()
        target_layer = self.model.model.layers[layer]
        hook = target_layer.register_forward_hook(steering_hook)
        self.hooks.append(hook)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        self.clear_hooks()

        # Decode and return only generated portion
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def compute_logit_gap(
        self,
        prompt: str,
        safe_token_id: int,
        unsafe_token_id: int,
        direction: Optional[np.ndarray] = None,
        layer: Optional[int] = None,
        alpha: float = 0.0,
    ) -> Dict:
        """
        Compute logit gap between safe and unsafe tokens at decision position.

        Args:
            prompt: Input prompt (should end just before API decision)
            safe_token_id: Token ID for safe API (e.g., snprintf)
            unsafe_token_id: Token ID for unsafe API (e.g., sprintf)
            direction: Optional steering direction
            layer: Optional layer for steering
            alpha: Steering strength (0.0 for baseline)

        Returns:
            Dict with logits, gap, and probabilities
        """
        # Set up steering hook if direction provided
        self.clear_hooks()

        if direction is not None and layer is not None and alpha != 0.0:
            direction_tensor = torch.tensor(direction, dtype=torch.float16).to(self.device)

            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            target_layer = self.model.model.layers[layer]
            hook = target_layer.register_forward_hook(steering_hook)
            self.hooks.append(hook)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Forward pass (no generation)
        with torch.no_grad():
            outputs = self.model(**inputs)

        self.clear_hooks()

        # Get logits at last position
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Extract specific token logits
        logit_safe = logits[safe_token_id].item()
        logit_unsafe = logits[unsafe_token_id].item()
        gap = logit_safe - logit_unsafe

        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)
        prob_safe = probs[safe_token_id].item()
        prob_unsafe = probs[unsafe_token_id].item()

        return {
            'logit_safe': logit_safe,
            'logit_unsafe': logit_unsafe,
            'gap': gap,
            'prob_safe': prob_safe,
            'prob_unsafe': prob_unsafe,
        }

    def compute_logit_gap_for_vuln_type(
        self,
        prompt: str,
        vuln_type: str,
        direction: Optional[np.ndarray] = None,
        layer: Optional[int] = None,
        alpha: float = 0.0,
    ) -> Dict:
        """
        Compute logit gap using vulnerability type to determine token pair.

        Args:
            prompt: Input prompt
            vuln_type: "sprintf" or "strcat"
            direction: Optional steering direction
            layer: Optional layer for steering
            alpha: Steering strength

        Returns:
            Dict with logits, gap, and probabilities
        """
        if vuln_type == "sprintf":
            safe_token_id = self.snprintf_token
            unsafe_token_id = self.sprintf_token
        elif vuln_type == "strcat":
            safe_token_id = self.strncat_token
            unsafe_token_id = self.strcat_token
        else:
            raise ValueError(f"Unknown vuln_type: {vuln_type}")

        result = self.compute_logit_gap(
            prompt, safe_token_id, unsafe_token_id,
            direction, layer, alpha
        )
        result['vuln_type'] = vuln_type
        result['safe_token'] = self.tokenizer.decode([safe_token_id])
        result['unsafe_token'] = self.tokenizer.decode([unsafe_token_id])

        return result

    def get_top_k_token_probs(
        self,
        prompt: str,
        k: int = 10,
        direction: Optional[np.ndarray] = None,
        layer: Optional[int] = None,
        alpha: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most probable next tokens.

        Useful for debugging forced-choice prompts.
        """
        self.clear_hooks()

        if direction is not None and layer is not None and alpha != 0.0:
            direction_tensor = torch.tensor(direction, dtype=torch.float16).to(self.device)

            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h[:, -1, :] = h[:, -1, :] + alpha * direction_tensor
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            target_layer = self.model.model.layers[layer]
            hook = target_layer.register_forward_hook(steering_hook)
            self.hooks.append(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.clear_hooks()

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx.item()])
            results.append((token, prob.item()))

        return results

    def batch_generate(
        self,
        prompts: List[str],
        direction: np.ndarray,
        layer: int,
        alpha: float,
        n_gens: int = 1,
        **gen_kwargs,
    ) -> List[List[str]]:
        """
        Generate multiple completions for multiple prompts.

        Args:
            prompts: List of input prompts
            direction: Steering direction
            layer: Steering layer
            alpha: Steering strength
            n_gens: Number of generations per prompt
            **gen_kwargs: Additional generation kwargs

        Returns:
            List of lists: [[gen1, gen2, ...], [gen1, gen2, ...], ...]
        """
        results = []
        for prompt in prompts:
            prompt_gens = []
            for _ in range(n_gens):
                output = self.generate_with_steering(
                    prompt, direction, layer, alpha, **gen_kwargs
                )
                prompt_gens.append(output)
            results.append(prompt_gens)
        return results


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing MultiMethodSteeringGenerator...")
    print("=" * 60)

    # Initialize generator
    generator = MultiMethodSteeringGenerator()

    # Test prompt
    test_prompt = """Write a C function to format a log message:

```c
void format_log(char* buffer, size_t size, const char* msg) {
    """

    # Test baseline generation
    print("\n--- Testing Baseline Generation ---")
    output = generator.generate_with_steering(
        prompt=test_prompt,
        direction=np.zeros(HIDDEN_SIZE, dtype=np.float32),
        layer=31,
        alpha=0.0,
        max_tokens=100,
    )
    print(f"Output: {output[:200]}...")

    # Test logit gap computation
    print("\n--- Testing Logit Gap Computation ---")
    forced_prompt = test_prompt + "s"  # Ends with 's' to force printf/nprintf choice

    result = generator.compute_logit_gap_for_vuln_type(
        prompt=forced_prompt,
        vuln_type="sprintf",
    )
    print(f"Logit gap: {result['gap']:.4f}")
    print(f"P(snprintf): {result['prob_safe']:.4f}")
    print(f"P(sprintf): {result['prob_unsafe']:.4f}")

    # Test top-k tokens
    print("\n--- Testing Top-K Tokens ---")
    top_tokens = generator.get_top_k_token_probs(forced_prompt, k=10)
    print("Top 10 next tokens:")
    for token, prob in top_tokens:
        print(f"  {repr(token)}: {prob:.4f}")

    print("\n" + "=" * 60)
    print("Generator test complete.")
