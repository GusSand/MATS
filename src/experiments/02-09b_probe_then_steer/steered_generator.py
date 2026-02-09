"""
Hook-free steered generation.

Four options for applying steering without register_forward_hook during generation:
  Option A: Monkey-patch the layer forward (simplest)
  Option B: torch.compile the steered layer (fused kernels)
  Option C: Direct residual stream bias (most surgical)
  Option D: MLP down_proj weight bias (zero Python wrapper — true zero overhead)

Key finding from Options A-C: the overhead isn't from hook dispatch machinery — it's
from having ANY Python function wrapper in the forward path during generation, which
breaks CUDA graph optimizations. All of A/B/C still wrap a Python function and have
~100% overhead (same as hooks).

Option D bakes the steering vector directly into model weights (mlp.down_proj.bias),
requiring no Python wrapper at all.
"""
import torch
import functools
from contextlib import contextmanager
from typing import Optional


class SteeredGenerator:
    """Generate text with activation steering applied as a graph modification, not a hook."""

    def __init__(self, model, tokenizer, steering_vectors: dict, alphas: dict,
                 steering_layer: int = 31):
        """
        Args:
            model: Llama-3.1-8B-Instruct
            tokenizer: Tokenizer
            steering_vectors: {"buffer": np.array, "format_string": np.array}
            alphas: {"buffer": float, "format_string": float}
            steering_layer: Layer to apply steering (default 31)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.steering_layer = steering_layer

        # Pre-convert vectors to GPU tensors (done once, reused for all generations)
        self.steering_tensors = {}
        for key, vec in steering_vectors.items():
            self.steering_tensors[key] = torch.tensor(vec, dtype=torch.float16).to(model.device)
        self.alphas = alphas

    # ─── Option A: Monkey-patch forward ──────────────────────────────────

    @contextmanager
    def _apply_steering_monkeypatch(self, cwe_type: str):
        """
        Patches Layer 31's forward to add steering vector.
        The patched function replaces the original — no hook dispatch overhead.
        """
        layer = self.model.model.layers[self.steering_layer]
        original_forward = layer.forward
        vec = self.steering_tensors[cwe_type]
        alpha = self.alphas[cwe_type]

        @functools.wraps(original_forward)
        def steered_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            # output[0] is hidden_states — modify in-place so it works
            # regardless of whether output is a tuple, NamedTuple, or dataclass
            h = output[0]
            if h.dim() == 3:
                h[:, -1, :] += alpha * vec
            else:
                # During decode steps, h is 2D (batch, hidden) — steer all positions
                h.add_(alpha * vec)
            return output

        layer.forward = steered_forward
        try:
            yield
        finally:
            layer.forward = original_forward

    # ─── Option B: torch.compile the steered layer ───────────────────────

    @contextmanager
    def _apply_steering_compiled(self, cwe_type: str):
        """
        Use torch.compile to fuse steering into CUDA kernels.
        First call will be slow (compilation), subsequent calls will be fast.
        """
        layer = self.model.model.layers[self.steering_layer]
        original_forward = layer.forward
        vec = self.steering_tensors[cwe_type]
        alpha = self.alphas[cwe_type]

        @functools.wraps(original_forward)
        def steered_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            h = output[0]
            if h.dim() == 3:
                h[:, -1, :] += alpha * vec
            else:
                h.add_(alpha * vec)
            return output

        try:
            # Note: mode="reduce-overhead" uses CUDA graphs which are incompatible
            # with dynamic KV cache updates during autoregressive generation.
            # Use default mode instead (kernel fusion only, no CUDA graphs).
            compiled_forward = torch.compile(steered_forward)
            layer.forward = compiled_forward
        except Exception:
            layer.forward = steered_forward

        try:
            yield
        finally:
            layer.forward = original_forward

    # ─── Option C: Post-attention layernorm bias ─────────────────────────

    @contextmanager
    def _apply_steering_bias(self, cwe_type: str):
        """
        Add steering vector as a bias via post-attention layernorm patching.
        Most surgical but architecture-specific.
        """
        layer = self.model.model.layers[self.steering_layer]
        vec = self.steering_tensors[cwe_type]
        alpha = self.alphas[cwe_type]
        bias = alpha * vec

        original_post_ln = layer.post_attention_layernorm.forward

        def biased_ln_forward(x):
            out = original_post_ln(x)
            if out.dim() == 3:
                out[:, -1, :] += bias
            else:
                out.add_(bias)
            return out

        layer.post_attention_layernorm.forward = biased_ln_forward
        try:
            yield
        finally:
            layer.post_attention_layernorm.forward = original_post_ln

    # ─── Option D: MLP down_proj weight bias (zero Python wrapper) ───────

    @contextmanager
    def _apply_steering_weight_bias(self, cwe_type: str):
        """
        Bake steering vector into mlp.down_proj.bias at Layer 31.

        Zero Python wrapper overhead — the bias is computed inside the existing
        F.linear CUDA kernel, so no Python function fires per token.

        Math: Llama layer output = residual + mlp(post_attn_ln(h))
              mlp output = down_proj(act(gate(x)) * up(x))
              With bias: mlp output = down_proj(...) + bias
              → layer output = (original output) + bias   ✓ correct

        Tradeoff: steers ALL token positions during prefill (not just last).
        During decode (where >99% of generation time is spent), only 1 position
        exists, so behavior is identical to last-position-only steering.
        """
        layer = self.model.model.layers[self.steering_layer]
        vec = self.steering_tensors[cwe_type]
        alpha = self.alphas[cwe_type]
        down_proj = layer.mlp.down_proj

        # Save original bias state
        had_bias = down_proj.bias is not None
        original_bias = down_proj.bias.data.clone() if had_bias else None

        # Add steering vector as bias (or create new bias)
        steering_bias = alpha * vec
        if had_bias:
            down_proj.bias.data.add_(steering_bias)
        else:
            down_proj.bias = torch.nn.Parameter(steering_bias, requires_grad=False)

        try:
            yield
        finally:
            # Restore original state
            if had_bias:
                down_proj.bias.data = original_bias
            else:
                down_proj.bias = None

    # ─── Generation ──────────────────────────────────────────────────────

    def _get_steering_context(self, cwe_type: str, method: str = "weight_bias"):
        """Get the appropriate context manager for the steering method."""
        if method == "monkeypatch":
            return self._apply_steering_monkeypatch(cwe_type)
        elif method == "compiled":
            return self._apply_steering_compiled(cwe_type)
        elif method == "bias":
            return self._apply_steering_bias(cwe_type)
        elif method == "weight_bias":
            return self._apply_steering_weight_bias(cwe_type)
        else:
            raise ValueError(f"Unknown steering method: {method}")

    @torch.no_grad()
    def generate(self, prompt: str, cwe_type: str, method: str = "monkeypatch",
                 max_new_tokens: int = 512, temperature: float = 0.6,
                 top_p: float = 0.9, do_sample: bool = True,
                 seed: Optional[int] = None, **gen_kwargs) -> str:
        """Generate with steering applied as a graph modification, not a hook."""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with self._get_steering_context(cwe_type, method):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def generate_baseline(self, prompt: str, max_new_tokens: int = 512,
                          temperature: float = 0.6, top_p: float = 0.9,
                          do_sample: bool = True, seed: Optional[int] = None,
                          **gen_kwargs) -> str:
        """Generate without any steering (baseline)."""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
