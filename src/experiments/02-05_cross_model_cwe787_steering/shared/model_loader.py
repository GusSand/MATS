"""
Unified Model Loader for Cross-Model Experiments

Supports Llama/Mistral family models with fp16 or 8-bit quantization.
Both model families use model.model.layers[N] as hook targets.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class ModelLoader:
    """Unified model loading for Llama/Mistral family models."""

    def __init__(self, model_name: str, quantization: str = None):
        """
        Load a model with optional quantization.

        Args:
            model_name: HuggingFace model name
            quantization: None for fp16, "8bit" for 8-bit quantization
        """
        print(f"Loading model: {model_name}")
        if quantization:
            print(f"  Quantization: {quantization}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "device_map": "auto",
        }

        if quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["max_memory"] = {0: "72GiB", "cpu": "40GiB"}
        elif quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["max_memory"] = {0: "60GiB", "cpu": "60GiB"}
            load_kwargs["dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        print(f"  Loaded: {self.n_layers} layers, {self.hidden_size} hidden dim")
        self._print_memory_usage()

    def get_layer(self, idx: int):
        """Get a transformer layer by index (for hook registration)."""
        return self.model.model.layers[idx]

    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Model unloaded, GPU memory freed")
