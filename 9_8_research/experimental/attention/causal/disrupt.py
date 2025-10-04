import torch
import torch.nn.functional as F
from nnsight import LanguageModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class AttentionAnchoringExperiment:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model = LanguageModel(model_name, device_map="auto")
        self.layer_10_heads = list(range(32))  # Assuming 32 heads
        
    def get_baseline_attention(self, prompt, layer=10):
        """Get the natural attention pattern for a prompt"""
        with self.model.trace(prompt) as tracer:
            # Access attention weights at specified layer
            attention = self.model.model.layers[layer].self_attn.attention_weights.save()
            
        return attention.value
    
    def disrupt_begin_anchoring(self, prompt, layer=10, disruption_strength=0.8):
        """
        Intervene to redistribute BEGIN token attention to other tokens
        disruption_strength: 0 = no disruption, 1 = complete redistribution
        """
        tokens = self.model.tokenizer(prompt)
        begin_token_idx = 0  # BEGIN is always first
        
        def modify_attention(module, input, output):
            # output shape: [batch, heads, seq_len, seq_len]
            attention_weights = output[0]  # Get attention weights
            
            # For each head, reduce attention to BEGIN token
            for head_idx in range(attention_weights.shape[1]):
                # Get current attention to BEGIN
                begin_attention = attention_weights[0, head_idx, :, begin_token_idx].clone()
                
                # Reduce BEGIN attention by disruption_strength
                reduced_attention = begin_attention * (1 - disruption_strength)
                
                # Redistribute the taken attention to other tokens uniformly
                # or proportionally to their current weights
                taken_attention = begin_attention * disruption_strength
                seq_len = attention_weights.shape[-1]
                
                # Redistribute proportionally to non-BEGIN tokens
                other_indices = [i for i in range(seq_len) if i != begin_token_idx]
                if other_indices:
                    current_other = attention_weights[0, head_idx, :, other_indices]
                    current_sum = current_other.sum(dim=-1, keepdim=True)
                    
                    # Avoid division by zero
                    current_sum = torch.where(current_sum > 0, current_sum, torch.ones_like(current_sum))
                    
                    redistribution = (current_other / current_sum) * taken_attention.unsqueeze(-1)
                    attention_weights[0, head_idx, :, other_indices] += redistribution
                
                # Apply the reduced attention to BEGIN
                attention_weights[0, head_idx, :, begin_token_idx] = reduced_attention
                
                # Renormalize to ensure sum = 1
                attention_weights[0, head_idx] = F.softmax(
                    torch.log(attention_weights[0, head_idx] + 1e-10), dim=-1
                )
            
            return (attention_weights,) + output[1:]
        
        # Register hook and generate
        handle = self.model.model.layers[layer].self_attn.register_forward_hook(modify_attention)
        
        try:
            with self.model.generate(max_new_tokens=10, do_sample=False) as generator:
                with generator.invoke(prompt):
                    output = self.model.generator.output.save()
        finally:
            handle.remove()
            
        # Convert token IDs to string
        return self.model.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def run_causal_experiment(self, n_trials=100):
        """
        Main experiment: Disrupt BEGIN anchoring in Simple format
        Should cause the bug if our hypothesis is correct
        """
        results = {
            'disruption_level': [],
            'error_rate': [],
            'begin_attention': [],
            'format': []
        }
        
        # Test prompt that normally works
        simple_prompt = "Which is bigger: 9.8 or 9.11? Answer:"
        
        # Test different disruption levels
        disruption_levels = np.linspace(0, 1, 11)  # 0%, 10%, ..., 100%
        
        for disruption in disruption_levels:
            errors = []
            begin_attentions = []
            
            for _ in range(n_trials):
                if disruption == 0:
                    # Baseline: no intervention  
                    with self.model.generate(max_new_tokens=10, do_sample=False) as generator:
                        with generator.invoke(simple_prompt):
                            output_ids = self.model.generator.output.save()
                    output = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                else:
                    # Intervene to disrupt BEGIN anchoring
                    output = self.disrupt_begin_anchoring(
                        simple_prompt, 
                        disruption_strength=disruption
                    )
                
                # Check if output is correct
                is_error = "9.11" in output and "bigger" in output.lower()
                errors.append(is_error)
                
                # Measure actual BEGIN attention after intervention
                with self.model.trace(simple_prompt) as tracer:
                    if disruption > 0:
                        # Apply same intervention to measure
                        pass  # Hook is applied in the function
                    attn = self.model.model.layers[10].self_attn.attention_weights.save()
                
                # Calculate average BEGIN attention across heads
                begin_attn = attn.value[0, :, -1, 0].mean().item()  # Last token attending to BEGIN
                begin_attentions.append(begin_attn)
            
            results['disruption_level'].extend([disruption] * n_trials)
            results['error_rate'].extend(errors)
            results['begin_attention'].extend(begin_attentions)
            results['format'].extend(['Simple (Disrupted)'] * n_trials)
        
        return results

    def run_restoration_experiment(self, n_trials=100):
        """
        Opposite experiment: Restore BEGIN anchoring in Q&A format
        Should fix the bug if our hypothesis is correct
        """
        qa_prompt = "Q: Which is bigger: 9.8 or 9.11? A:"
        simple_prompt = "Which is bigger: 9.8 or 9.11? Answer:"
        
        # Get target attention pattern from Simple format
        target_attention = self.get_baseline_attention(simple_prompt, layer=10)
        target_begin_attention = target_attention[0, :, -1, 0].mean().item()
        
        results = []
        
        for restoration_strength in np.linspace(0, 1, 11):
            errors = []
            
            for _ in range(n_trials):
                if restoration_strength == 0:
                    with self.model.generate(max_new_tokens=10, do_sample=False) as generator:
                        with generator.invoke(qa_prompt):
                            output_ids = self.model.generator.output.save()
                    output = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                else:
                    # Intervene to restore BEGIN anchoring
                    output = self.restore_begin_anchoring(
                        qa_prompt,
                        target_begin_attention,
                        restoration_strength
                    )
                
                is_error = "9.11" in output and "bigger" in output.lower()
                errors.append(is_error)
            
            error_rate = sum(errors) / len(errors)
            results.append({
                'restoration': restoration_strength,
                'error_rate': error_rate,
                'ci_lower': np.percentile(errors, 2.5),
                'ci_upper': np.percentile(errors, 97.5)
            })
        
        return results
    
    def restore_begin_anchoring(self, prompt, target_attention, restoration_strength, layer=10):
        """
        Intervene to restore BEGIN token attention to target level
        restoration_strength: 0 = no restoration, 1 = full restoration to target
        """
        tokens = self.model.tokenizer(prompt)
        begin_token_idx = 0  # BEGIN is always first
        
        def modify_attention(module, input, output):
            # output shape: [batch, heads, seq_len, seq_len]
            attention_weights = output[0]  # Get attention weights
            
            # For each head, restore attention to BEGIN token
            for head_idx in range(attention_weights.shape[1]):
                # Get current attention to BEGIN
                current_begin = attention_weights[0, head_idx, :, begin_token_idx].clone()
                
                # Interpolate between current and target based on restoration_strength
                restored_attention = current_begin * (1 - restoration_strength) + target_attention * restoration_strength
                
                # Apply the restored attention
                attention_weights[0, head_idx, :, begin_token_idx] = restored_attention
                
                # Renormalize to ensure sum = 1
                attention_weights[0, head_idx] = F.softmax(
                    torch.log(attention_weights[0, head_idx] + 1e-10), dim=-1
                )
            
            return (attention_weights,) + output[1:]
        
        # Register hook and generate
        handle = self.model.model.layers[layer].self_attn.register_forward_hook(modify_attention)
        
        try:
            with self.model.generate(max_new_tokens=10, do_sample=False) as generator:
                with generator.invoke(prompt):
                    output = self.model.generator.output.save()
        finally:
            handle.remove()
            
        # Convert token IDs to string
        return self.model.tokenizer.decode(output[0], skip_special_tokens=True)