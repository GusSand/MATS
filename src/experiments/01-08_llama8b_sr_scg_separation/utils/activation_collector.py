"""
Activation Collector for SR/SCG Separation Experiment

Extended from 01-07 experiment to support multiple security pairs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class ActivationCollector:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded: {self.n_layers} layers, {self.hidden_size} hidden dim")

        self.activations = {}
        self.hooks = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _make_layer_hook(self, layer_idx: int):
        """Create a hook that saves last-token activation."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            self.activations[layer_idx] = h[:, -1, :].detach().cpu()
        return hook_fn

    def register_all_layer_hooks(self):
        """Register hooks at all layers."""
        self.clear_hooks()
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_layer_hook(layer_idx))
            self.hooks.append(hook)

    def get_activations(self, prompt: str) -> dict:
        """Get last-token activations at all layers for a prompt."""
        self.clear_hooks()
        self.register_all_layer_hooks()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

        result = {k: v.numpy() for k, v in self.activations.items()}
        self.clear_hooks()
        return result

    def get_activations_and_logits(self, prompt: str) -> dict:
        """Get activations and next-token logits."""
        self.clear_hooks()
        self.register_all_layer_hooks()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :].cpu().numpy()
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1).cpu().numpy()

        result = {
            'activations': {k: v.numpy() for k, v in self.activations.items()},
            'logits': logits,
            'probs': probs,
            'seq_len': inputs['input_ids'].shape[1]
        }
        self.clear_hooks()
        return result

    def generate_and_classify(self, prompt: str, detection_patterns: dict,
                              temperature: float = 0.6, max_new_tokens: int = 100) -> dict:
        """Generate completion and classify using provided patterns."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = generated[len(prompt):]

        # Classify using provided patterns
        has_secure = bool(re.search(detection_patterns['secure'], output))
        has_insecure = bool(re.search(detection_patterns['insecure'], output))

        if has_secure and not has_insecure:
            label = 'secure'
        elif has_insecure:
            label = 'insecure'
        else:
            label = 'neither'

        return {
            'output': output[:300],
            'label': label,
            'has_secure': has_secure,
            'has_insecure': has_insecure
        }

    def get_token_probs(self, prompt: str, tokens: dict) -> dict:
        """Get probabilities for specific tokens.

        Args:
            prompt: Input prompt
            tokens: Dict mapping name -> token string, e.g., {'secure': ' snprintf', 'insecure': ' sprintf'}

        Returns:
            Dict with probabilities for each token
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        result = {}
        for name, token_str in tokens.items():
            token_id = self.tokenizer.encode(token_str, add_special_tokens=False)[0]
            result[f'{name}_token_id'] = token_id
            result[f'{name}_prob'] = probs[token_id].item()

        return result

    def collect_sr_dataset(self, pair_config: dict, n_samples_per_template: int = 20) -> dict:
        """
        Collect Security Recognition (SR) dataset.

        Label: 1 = secure context (has warning), 0 = neutral context (no warning)
        This measures whether the model can detect security-relevant context.
        """
        data = {layer: {'X': [], 'y': []} for layer in range(self.n_layers)}
        metadata = []

        # Secure contexts (label=1)
        for template in pair_config['secure_templates']:
            for _ in range(n_samples_per_template):
                acts = self.get_activations(template)
                for layer in range(self.n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(1)
                metadata.append({'context': 'secure', 'template': template[:50]})

        # Neutral contexts (label=0)
        for template in pair_config['neutral_templates']:
            for _ in range(n_samples_per_template):
                acts = self.get_activations(template)
                for layer in range(self.n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(0)
                metadata.append({'context': 'neutral', 'template': template[:50]})

        # Convert to numpy arrays
        for layer in range(self.n_layers):
            data[layer]['X'] = np.array(data[layer]['X'])
            data[layer]['y'] = np.array(data[layer]['y'])

        return data, metadata

    def collect_scg_dataset(self, pair_config: dict, n_samples_per_template: int = 25) -> dict:
        """
        Collect Secure Code Generation (SCG) dataset.

        Label: 1 = model outputs secure function, 0 = model outputs insecure function
        This measures whether the model will generate secure code.
        """
        data = {layer: {'X': [], 'y': []} for layer in range(self.n_layers)}
        metadata = []

        all_templates = pair_config['secure_templates'] + pair_config['neutral_templates']
        detection_patterns = pair_config['detection_patterns']

        secure_count = 0
        insecure_count = 0
        neither_count = 0

        for template in all_templates:
            for _ in range(n_samples_per_template):
                # Get activations BEFORE generation
                acts = self.get_activations(template)

                # Generate and classify
                result = self.generate_and_classify(template, detection_patterns)

                if result['label'] == 'secure':
                    label = 1
                    secure_count += 1
                elif result['label'] == 'insecure':
                    label = 0
                    insecure_count += 1
                else:
                    neither_count += 1
                    continue  # Skip samples with neither

                for layer in range(self.n_layers):
                    data[layer]['X'].append(acts[layer].squeeze())
                    data[layer]['y'].append(label)

                metadata.append({
                    'template': template[:50],
                    'output_label': result['label'],
                    'output_snippet': result['output'][:100]
                })

        # Convert to numpy arrays
        for layer in range(self.n_layers):
            if data[layer]['X']:
                data[layer]['X'] = np.array(data[layer]['X'])
                data[layer]['y'] = np.array(data[layer]['y'])
            else:
                data[layer]['X'] = np.array([]).reshape(0, self.hidden_size)
                data[layer]['y'] = np.array([])

        stats = {
            'secure': secure_count,
            'insecure': insecure_count,
            'neither': neither_count,
            'total_usable': secure_count + insecure_count
        }

        return data, metadata, stats
