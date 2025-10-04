# Bandwidth Competition Theory: Experimental Framework

## Setup and Dependencies

```python
# Core imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Model and intervention tools
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import NNsight

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_OF_INTEREST = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Core Theory
**"Format tokens compete for attention bandwidth, disrupting information routing needed for numerical comparison. Even heads evolved robust routing patterns that preserve numerical bandwidth above the critical threshold."**

---

## Experiment Set 1: Bandwidth Competition Analysis

### 1.1 Measure Attention Bandwidth Distribution

```python
@dataclass
class TokenCategory:
    """Categorize tokens for bandwidth analysis"""
    format_tokens: List[str] = None
    numerical_tokens: List[str] = None
    other_tokens: List[str] = None
    
    def __post_init__(self):
        if self.format_tokens is None:
            self.format_tokens = ['Q', ':', 'A', 'Which', 'is', 'bigger', '?', 
                                 '<|start_header_id|>', '<|end_header_id|>', 
                                 'user', 'assistant']
        if self.numerical_tokens is None:
            self.numerical_tokens = ['9', '.', '8', '11', '0', '1', '2', '3', 
                                    '4', '5', '6', '7']

def analyze_attention_bandwidth(model, tokenizer, prompts: Dict[str, str]) -> pd.DataFrame:
    """
    Measure how attention is distributed across token categories.
    
    Expected result: 
    - Even heads maintain >40% attention on numerical tokens
    - Odd heads drop below threshold in Q&A format
    """
    results = []
    token_cat = TokenCategory()
    
    with NNsight(model) as nn:
        for format_name, prompt in prompts.items():
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            
            # Run forward pass and capture attention
            with nn.forward(inputs) as tracer:
                # Capture attention weights at Layer 10
                attention_weights = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights.save()
            
            # Analyze each head
            attn = attention_weights.value  # Shape: [batch, heads, seq_len, seq_len]
            
            for head_idx in range(attn.shape[1]):
                head_attn = attn[0, head_idx, :, :].cpu().numpy()
                
                # Categorize attention by token type
                format_attn = 0
                numerical_attn = 0
                other_attn = 0
                
                for i, token in enumerate(tokens):
                    token_str = token.replace('▁', '').strip()
                    attn_sum = head_attn[:, i].sum()
                    
                    if any(fmt in token_str for fmt in token_cat.format_tokens):
                        format_attn += attn_sum
                    elif any(num in token_str for num in token_cat.numerical_tokens):
                        numerical_attn += attn_sum
                    else:
                        other_attn += attn_sum
                
                # Normalize
                total = format_attn + numerical_attn + other_attn
                if total > 0:
                    format_attn /= total
                    numerical_attn /= total
                    other_attn /= total
                
                results.append({
                    'format': format_name,
                    'head_idx': head_idx,
                    'head_type': 'even' if head_idx % 2 == 0 else 'odd',
                    'format_bandwidth': format_attn,
                    'numerical_bandwidth': numerical_attn,
                    'other_bandwidth': other_attn
                })
    
    df = pd.DataFrame(results)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bandwidth distribution by head type
    for idx, head_type in enumerate(['even', 'odd']):
        ax = axes[idx]
        data = df[df['head_type'] == head_type]
        
        x = np.arange(len(prompts))
        width = 0.25
        
        for i, metric in enumerate(['format_bandwidth', 'numerical_bandwidth', 'other_bandwidth']):
            means = data.groupby('format')[metric].mean()
            ax.bar(x + i*width, means.values, width, label=metric.replace('_bandwidth', ''))
        
        ax.set_xlabel('Format')
        ax.set_ylabel('Bandwidth Proportion')
        ax.set_title(f'{head_type.capitalize()} Heads - Bandwidth Distribution')
        ax.set_xticks(x + width)
        ax.set_xticklabels(prompts.keys())
        ax.legend()
        ax.axhline(y=0.4, color='r', linestyle='--', label='40% threshold')
    
    plt.tight_layout()
    plt.savefig('bandwidth_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

# Test the bandwidth analysis
prompts = {
    'Simple': '9.8 or 9.11? Answer:',
    'Q&A': 'Q: Which is bigger: 9.8 or 9.11? A:',
    'Chat': '<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
}

# df_bandwidth = analyze_attention_bandwidth(model, tokenizer, prompts)
# print("\nBandwidth Analysis Summary:")
# print(df_bandwidth.groupby(['format', 'head_type'])['numerical_bandwidth'].mean())
```

### 1.2 Bandwidth Manipulation Experiment

```python
def test_bandwidth_threshold(model, tokenizer, base_prompt: str, target_bandwidth: float = 0.3):
    """
    Test if artificially modifying bandwidth allocation changes behavior.
    
    Hypothesis: Forcing even heads below 40% numerical bandwidth should break them
    Forcing odd heads above 40% numerical bandwidth should fix them
    """
    results = {'original': [], 'modified': []}
    
    with NNsight(model) as nn:
        inputs = tokenizer(base_prompt, return_tensors="pt").to(DEVICE)
        
        # Original forward pass
        with nn.forward(inputs) as tracer:
            original_output = model.output.save()
            attn_layer10 = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights
            original_attn = attn_layer10.save()
        
        results['original'] = tokenizer.decode(original_output.value[0].argmax(-1))
        
        # Modified forward pass - redistribute attention
        with nn.forward(inputs) as tracer:
            attn = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights
            
            # Custom intervention to modify bandwidth
            def redistribute_attention(attn_weights):
                modified = attn_weights.clone()
                batch_size, n_heads, seq_len, _ = modified.shape
                
                # Identify token positions
                tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                numerical_positions = []
                format_positions = []
                
                for i, token in enumerate(tokens):
                    if any(num in token for num in ['9', '8', '11', '.']):
                        numerical_positions.append(i)
                    elif any(fmt in token for fmt in ['Q', ':', 'A']):
                        format_positions.append(i)
                
                # Modify each head based on type
                for head_idx in range(n_heads):
                    if head_idx % 2 == 0:  # Even head
                        # Force to allocate only target_bandwidth to numerical
                        for pos in numerical_positions:
                            modified[0, head_idx, :, pos] *= target_bandwidth / 0.6
                        for pos in format_positions:
                            modified[0, head_idx, :, pos] *= (1 - target_bandwidth) / 0.4
                    else:  # Odd head
                        # Force to allocate 50% to numerical
                        for pos in numerical_positions:
                            modified[0, head_idx, :, pos] *= 0.5 / 0.2
                        for pos in format_positions:
                            modified[0, head_idx, :, pos] *= 0.5 / 0.8
                
                # Renormalize
                modified = modified / modified.sum(dim=-1, keepdim=True)
                return modified
            
            attn.value = redistribute_attention(attn.value)
            modified_output = model.output.save()
        
        results['modified'] = tokenizer.decode(modified_output.value[0].argmax(-1))
    
    print(f"Original output: {results['original']}")
    print(f"Modified output (bandwidth={target_bandwidth}): {results['modified']}")
    
    return results

# Test bandwidth manipulation
# results = test_bandwidth_threshold(model, tokenizer, prompts['Q&A'], target_bandwidth=0.3)
```

### 1.3 Progressive Format Injection

```python
def test_format_interference():
    """
    Test how increasing format complexity affects attention to numerical tokens.
    Should show sharp degradation at ~60% format token threshold.
    """
    test_prompts = [
        ("Minimal", "9.8 or 9.11?"),
        ("Light", "Compare: 9.8 or 9.11?"),
        ("Medium", "Q: 9.8 or 9.11?"),
        ("Heavy", "Q: Which is bigger: 9.8 or 9.11? A:"),
        ("Full", "User Question: Which number is bigger: 9.8 or 9.11? Assistant Answer:"),
    ]
    
    results = []
    
    with NNsight(model) as nn:
        for format_name, prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            with nn.forward(inputs) as tracer:
                attn = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights.save()
                output = model.output.save()
            
            # Calculate format token percentage
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            format_count = sum(1 for t in tokens if any(
                fmt in t for fmt in ['Q', ':', 'A', 'User', 'Assistant', 'Which', 'is', 'bigger', '?']
            ))
            numerical_count = sum(1 for t in tokens if any(
                num in t for num in ['9', '8', '11', '.']
            ))
            total_count = len(tokens)
            
            format_percentage = format_count / total_count
            
            # Analyze attention distribution
            attn_weights = attn.value[0].mean(dim=0)  # Average across heads
            numerical_attention = 0
            
            for i, token in enumerate(tokens):
                if any(num in token for num in ['9', '8', '11', '.']):
                    numerical_attention += attn_weights[:, i].sum().item()
            
            # Check output correctness
            output_text = tokenizer.decode(output.value[0].argmax(-1))
            is_correct = '9.8' in output_text and 'bigger' in output_text.lower()
            
            results.append({
                'format': format_name,
                'prompt_length': total_count,
                'format_percentage': format_percentage,
                'numerical_attention': numerical_attention / attn_weights.sum().item(),
                'correct': is_correct
            })
    
    df = pd.DataFrame(results)
    
    # Plot degradation curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Format percentage vs numerical attention
    ax1.scatter(df['format_percentage'], df['numerical_attention'], 
               c=df['correct'], cmap='RdYlGn', s=100)
    ax1.set_xlabel('Format Token Percentage')
    ax1.set_ylabel('Attention to Numerical Tokens')
    ax1.set_title('Format Interference Effect')
    ax1.axvline(x=0.6, color='r', linestyle='--', label='60% threshold')
    ax1.axhline(y=0.4, color='b', linestyle='--', label='40% attention threshold')
    ax1.legend()
    
    # Add format labels
    for idx, row in df.iterrows():
        ax1.annotate(row['format'], (row['format_percentage'], row['numerical_attention']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Success rate by format complexity
    ax2.bar(df['format'], df['correct'], color=['green' if c else 'red' for c in df['correct']])
    ax2.set_xlabel('Format Complexity')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Performance vs Format Complexity')
    ax2.set_xticklabels(df['format'], rotation=45)
    
    plt.tight_layout()
    plt.savefig('format_interference.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

# df_interference = test_format_interference()
# print("\nFormat Interference Results:")
# print(df_interference)
```

---

## Experiment Set 2: Information Routing Analysis

### 2.1 Attention Path Tracing

```python
def trace_information_flow(model, tokenizer, prompt: str):
    """
    Track how information flows from numerical tokens through attention patterns.
    Measure "routing efficiency" = how much reaches comparison computation.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Identify key positions
    key_positions = {}
    for i, token in enumerate(tokens):
        if '9' in token and '.' not in token:
            key_positions['integer_9'] = i
        elif '8' in token:
            key_positions['decimal_8'] = i
        elif '11' in token:
            key_positions['decimal_11'] = i
        elif '.' in token:
            key_positions['decimal_point'] = i
    
    routing_scores = defaultdict(list)
    
    with NNsight(model) as nn:
        with nn.forward(inputs) as tracer:
            # Track attention flow through multiple layers
            for layer_idx in range(LAYER_OF_INTEREST - 2, LAYER_OF_INTEREST + 3):
                attn = model.layers[layer_idx].self_attn.attention_weights.save()
    
        # Analyze routing for each head
        for layer_idx in range(LAYER_OF_INTEREST - 2, LAYER_OF_INTEREST + 3):
            attn_weights = model.layers[layer_idx].self_attn.attention_weights.value
            
            for head_idx in range(attn_weights.shape[1]):
                head_type = 'even' if head_idx % 2 == 0 else 'odd'
                head_attn = attn_weights[0, head_idx].cpu().numpy()
                
                # Calculate routing efficiency for each key position
                for pos_name, pos_idx in key_positions.items():
                    if pos_idx is not None:
                        # How much attention flows from this position to final position
                        routing_score = head_attn[-1, pos_idx]  # Attention from last token to key position
                        
                        routing_scores[f'{head_type}_{pos_name}'].append(routing_score)
    
    # Visualize routing patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    positions = ['integer_9', 'decimal_8', 'decimal_11', 'decimal_point']
    for idx, pos in enumerate(positions):
        ax = axes[idx // 2, idx % 2]
        
        even_scores = routing_scores[f'even_{pos}']
        odd_scores = routing_scores[f'odd_{pos}']
        
        ax.boxplot([even_scores, odd_scores], labels=['Even Heads', 'Odd Heads'])
        ax.set_title(f'Routing Efficiency from {pos}')
        ax.set_ylabel('Attention Score')
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Min threshold')
    
    plt.suptitle('Information Routing Analysis')
    plt.tight_layout()
    plt.savefig('routing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return routing_scores

# routing_simple = trace_information_flow(model, tokenizer, prompts['Simple'])
# routing_qa = trace_information_flow(model, tokenizer, prompts['Q&A'])
```

### 2.2 Critical Path Identification

```python
def identify_critical_paths(model, tokenizer, prompt: str):
    """
    Use gradient-based attribution to identify critical attention connections.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    model.eval()
    inputs.input_ids.requires_grad = False
    
    critical_paths = {}
    
    # Hook to capture attention gradients
    attention_gradients = []
    
    def attention_hook(module, grad_input, grad_output):
        attention_gradients.append(grad_output[0].detach())
    
    # Register hooks
    handle = model.layers[LAYER_OF_INTEREST].self_attn.register_backward_hook(attention_hook)
    
    try:
        # Forward pass
        outputs = model(**inputs)
        
        # Target: probability of correct answer (token "9")
        target_token_id = tokenizer.encode("9", add_special_tokens=False)[0]
        loss = -outputs.logits[0, -1, target_token_id]
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        if attention_gradients:
            grad = attention_gradients[0]
            
            # Identify critical connections (high gradient magnitude)
            grad_magnitude = grad.abs().mean(dim=0)
            
            # Get top-k critical paths for each head
            for head_idx in range(grad_magnitude.shape[0]):
                head_type = 'even' if head_idx % 2 == 0 else 'odd'
                head_grad = grad_magnitude[head_idx].cpu().numpy()
                
                # Find top 5 critical connections
                top_k = 5
                top_indices = np.argsort(head_grad.flatten())[-top_k:]
                
                critical_paths[f'{head_type}_head_{head_idx}'] = {
                    'indices': top_indices,
                    'importance': head_grad.flatten()[top_indices]
                }
    
    finally:
        handle.remove()
    
    return critical_paths

# critical_simple = identify_critical_paths(model, tokenizer, prompts['Simple'])
# critical_qa = identify_critical_paths(model, tokenizer, prompts['Q&A'])
```

### 2.3 Positional Encoding Analysis

```python
def analyze_positional_sensitivity():
    """
    Test if heads are position-relative or position-absolute.
    Even heads should be robust to position changes.
    """
    base_comparison = "9.8 or 9.11"
    
    position_variants = [
        ("Original", f"{base_comparison}? Answer:"),
        ("Prefix_Short", f"Compare: {base_comparison}? Answer:"),
        ("Prefix_Long", f"Please compare these numbers: {base_comparison}? Answer:"),
        ("Suffix", f"Answer: {base_comparison}?"),
        ("Wrapped", f"The question is {base_comparison}, what is the answer?"),
        ("Spaced", f"Number 1: 9.8    Number 2: 9.11    Which is bigger?"),
    ]
    
    results = []
    
    with NNsight(model) as nn:
        for variant_name, prompt in position_variants:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            with nn.forward(inputs) as tracer:
                # Capture attention patterns for all heads
                attn = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights.save()
                output = model.output.save()
            
            # Analyze attention stability
            attn_weights = attn.value[0]  # [n_heads, seq_len, seq_len]
            
            # Find numerical token positions
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            num_positions = [i for i, t in enumerate(tokens) if any(n in t for n in ['9', '8', '11', '.'])]
            
            for head_idx in range(attn_weights.shape[0]):
                head_type = 'even' if head_idx % 2 == 0 else 'odd'
                
                # Calculate attention focus on numerical tokens regardless of position
                if num_positions:
                    numerical_attention = attn_weights[head_idx, :, num_positions].sum().item()
                else:
                    numerical_attention = 0
                
                results.append({
                    'variant': variant_name,
                    'head_idx': head_idx,
                    'head_type': head_type,
                    'numerical_attention': numerical_attention,
                    'position_shift': len(prompt.split()) - len(base_comparison.split())
                })
    
    df = pd.DataFrame(results)
    
    # Calculate position invariance score
    position_invariance = df.groupby(['head_idx', 'head_type'])['numerical_attention'].std()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Position invariance by head type
    even_invariance = position_invariance[position_invariance.index.get_level_values('head_type') == 'even']
    odd_invariance = position_invariance[position_invariance.index.get_level_values('head_type') == 'odd']
    
    ax1.boxplot([even_invariance.values, odd_invariance.values], labels=['Even Heads', 'Odd Heads'])
    ax1.set_ylabel('Attention Variance (lower = more robust)')
    ax1.set_title('Positional Robustness by Head Type')
    
    # Plot 2: Heatmap of attention by position variant
    pivot = df.pivot_table(values='numerical_attention', index='head_idx', columns='variant')
    sns.heatmap(pivot, cmap='coolwarm', center=pivot.mean().mean(), ax=ax2, cbar_kws={'label': 'Numerical Attention'})
    ax2.set_title('Attention Patterns Across Position Variants')
    ax2.set_xlabel('Position Variant')
    ax2.set_ylabel('Head Index')
    
    plt.tight_layout()
    plt.savefig('positional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, position_invariance

# df_position, invariance_scores = analyze_positional_sensitivity()
# print("\nPosition Invariance Scores (lower = more robust):")
# print(f"Even heads: {invariance_scores[invariance_scores.index.get_level_values('head_type') == 'even'].mean():.4f}")
# print(f"Odd heads: {invariance_scores[invariance_scores.index.get_level_values('head_type') == 'odd'].mean():.4f}")
```

---

## Experiment Set 3: Even/Odd Specialization Validation

### 3.1 Head Similarity Analysis

```python
def measure_head_similarity(model, tokenizer, test_prompts: List[str]):
    """
    Compute similarity matrix for all 32 heads based on attention patterns.
    Should show high within-group similarity and low between-group similarity.
    """
    n_heads = 32
    similarity_matrix = np.zeros((n_heads, n_heads))
    
    # Collect attention patterns for each head across prompts
    head_patterns = defaultdict(list)
    
    with NNsight(model) as nn:
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            with nn.forward(inputs) as tracer:
                attn = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights.save()
            
            attn_weights = attn.value[0].cpu().numpy()  # [n_heads, seq_len, seq_len]
            
            for head_idx in range(n_heads):
                # Flatten attention pattern and store
                pattern = attn_weights[head_idx].flatten()
                head_patterns[head_idx].append(pattern)
    
    # Compute pairwise similarities
    for i in range(n_heads):
        for j in range(n_heads):
            # Concatenate patterns across all prompts
            patterns_i = np.concatenate(head_patterns[i])
            patterns_j = np.concatenate(head_patterns[j])
            
            # Compute correlation
            if len(patterns_i) == len(patterns_j):
                similarity = np.corrcoef(patterns_i, patterns_j)[0, 1]
                similarity_matrix[i, j] = similarity
    
    # Visualize similarity matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full similarity matrix
    im1 = ax1.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Head Index')
    ax1.set_title('Head Similarity Matrix')
    
    # Add grid to separate even/odd
    for i in range(0, n_heads, 2):
        ax1.axhline(i-0.5, color='black', linewidth=0.5, alpha=0.3)
        ax1.axvline(i-0.5, color='black', linewidth=0.5, alpha=0.3)
    
    plt.colorbar(im1, ax=ax1)
    
    # Average within/between group similarity
    even_indices = [i for i in range(n_heads) if i % 2 == 0]
    odd_indices = [i for i in range(n_heads) if i % 2 == 1]
    
    within_even = similarity_matrix[np.ix_(even_indices, even_indices)][np.triu_indices(16, k=1)]
    within_odd = similarity_matrix[np.ix_(odd_indices, odd_indices)][np.triu_indices(16, k=1)]
    between_groups = similarity_matrix[np.ix_(even_indices, odd_indices)].flatten()
    
    ax2.boxplot([within_even, within_odd, between_groups], 
                labels=['Within Even', 'Within Odd', 'Between Groups'])
    ax2.set_ylabel('Similarity')
    ax2.set_title('Group Similarity Comparison')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('head_similarity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(within_even, between_groups)
    print(f"\nStatistical Test (within-even vs between-groups):")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
    
    return similarity_matrix

# test_prompts = [
#     "9.8 or 9.11? Answer:",
#     "Q: Which is bigger: 9.8 or 9.11? A:",
#     "Compare 9.8 and 9.11:",
#     "8.7 or 8.12? Answer:",
#     "3.4 or 3.25? Answer:"
# ]
# similarity_matrix = measure_head_similarity(model, tokenizer, test_prompts)
```

### 3.2 Head Swapping Experiment

```python
def test_head_swapping(model, tokenizer, prompt: str):
    """
    Test if we can convert an odd head to behave like an even head by swapping patterns.
    This tests if specialization is in attention patterns vs value computations.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    results = {
        'original': None,
        'even_pattern_odd_values': None,
        'odd_pattern_even_values': None
    }
    
    with NNsight(model) as nn:
        # Original forward pass
        with nn.forward(inputs) as tracer:
            original_output = model.output.save()
        results['original'] = tokenizer.decode(original_output.value[0].argmax(-1))
        
        # Test 1: Apply even attention pattern to odd head values
        with nn.forward(inputs) as tracer:
            attn_module = model.layers[LAYER_OF_INTEREST].self_attn
            
            # Custom intervention
            def swap_patterns_even_to_odd(attn_weights):
                modified = attn_weights.clone()
                # Copy pattern from head 0 (even) to head 1 (odd)
                # Keep doing this for all pairs
                for even_idx in range(0, 32, 2):
                    odd_idx = even_idx + 1
                    if odd_idx < 32:
                        modified[0, odd_idx] = modified[0, even_idx].clone()
                return modified
            
            attn_module.attention_weights.value = swap_patterns_even_to_odd(
                attn_module.attention_weights.value
            )
            swapped_output = model.output.save()
        
        results['even_pattern_odd_values'] = tokenizer.decode(swapped_output.value[0].argmax(-1))
        
        # Test 2: Apply odd attention pattern to even head values
        with nn.forward(inputs) as tracer:
            attn_module = model.layers[LAYER_OF_INTEREST].self_attn
            
            def swap_patterns_odd_to_even(attn_weights):
                modified = attn_weights.clone()
                # Copy pattern from head 1 (odd) to head 0 (even)
                for even_idx in range(0, 32, 2):
                    odd_idx = even_idx + 1
                    if odd_idx < 32:
                        modified[0, even_idx] = modified[0, odd_idx].clone()
                return modified
            
            attn_module.attention_weights.value = swap_patterns_odd_to_even(
                attn_module.attention_weights.value
            )
            swapped_output = model.output.save()
        
        results['odd_pattern_even_values'] = tokenizer.decode(swapped_output.value[0].argmax(-1))
    
    print("\n=== Head Swapping Results ===")
    print(f"Original output: {results['original']}")
    print(f"Even pattern → Odd values: {results['even_pattern_odd_values']}")
    print(f"Odd pattern → Even values: {results['odd_pattern_even_values']}")
    
    return results

# swap_results = test_head_swapping(model, tokenizer, prompts['Q&A'])
```

---

## Experiment Set 4: Threshold Mechanisms

### 4.1 The 60% Threshold Deep Dive

```python
def investigate_60_percent_threshold(model, tokenizer):
    """
    Investigate why exactly 60% pattern replacement is the threshold.
    Test hypothesis: Critical features need minimum input to activate.
    """
    prompt = prompts['Q&A']
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Test different interpolation levels
    interpolation_levels = np.arange(0.0, 1.0, 0.05)
    results = []
    
    with NNsight(model) as nn:
        # Get correct (Simple) and incorrect (Q&A) patterns
        simple_inputs = tokenizer(prompts['Simple'], return_tensors="pt").to(DEVICE)
        
        with nn.forward(simple_inputs) as tracer:
            correct_pattern = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights.save()
        
        with nn.forward(inputs) as tracer:
            incorrect_pattern = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights.save()
        
        # Test each interpolation level
        for alpha in interpolation_levels:
            with nn.forward(inputs) as tracer:
                attn = model.layers[LAYER_OF_INTEREST].self_attn
                
                # Interpolate patterns
                interpolated = (1 - alpha) * incorrect_pattern.value + alpha * correct_pattern.value
                attn.attention_weights.value = interpolated
                
                output = model.output.save()
            
            output_text = tokenizer.decode(output.value[0].argmax(-1))
            is_correct = '9.8' in output_text and 'bigger' in output_text.lower()
            
            # Measure activation strength (proxy for feature activation)
            activation_norm = output.value[0, -1].norm().item()
            
            results.append({
                'interpolation': alpha,
                'correct': is_correct,
                'activation_norm': activation_norm
            })
    
    df = pd.DataFrame(results)
    
    # Find the threshold
    threshold_idx = df[df['correct']].index[0] if any(df['correct']) else len(df)
    threshold = df.iloc[threshold_idx]['interpolation'] if threshold_idx < len(df) else 1.0
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Success vs interpolation
    ax1.plot(df['interpolation'] * 100, df['correct'].astype(int), 'o-')
    ax1.axvline(x=60, color='r', linestyle='--', label='Expected 60% threshold')
    ax1.axvline(x=threshold * 100, color='g', linestyle='--', label=f'Actual threshold: {threshold*100:.1f}%')
    ax1.set_xlabel('Pattern Replacement (%)')
    ax1.set_ylabel('Success (1=correct, 0=incorrect)')
    ax1.set_title('Success vs Pattern Replacement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Activation norm vs interpolation
    ax2.plot(df['interpolation'] * 100, df['activation_norm'], 'b-', alpha=0.7)
    ax2.scatter(df['interpolation'] * 100, df['activation_norm'], 
               c=df['correct'], cmap='RdYlGn', s=50, edgecolors='black')
    ax2.axvline(x=threshold * 100, color='g', linestyle='--', label=f'Threshold: {threshold*100:.1f}%')
    ax2.set_xlabel('Pattern Replacement (%)')
    ax2.set_ylabel('Activation Norm')
    ax2.set_title('Feature Activation Strength vs Pattern Replacement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nThreshold Analysis:")
    print(f"Detected threshold: {threshold*100:.1f}%")
    print(f"Sharp transition: {'Yes' if threshold > 0.55 and threshold < 0.65 else 'No'}")
    
    return df, threshold

# df_threshold, threshold = investigate_60_percent_threshold(model, tokenizer)
```

### 4.2 The 8-Head Requirement Analysis

```python
def analyze_8_head_requirement(model, tokenizer):
    """
    Investigate why exactly 8 heads are required.
    Test if it's related to information-theoretic requirements.
    """
    prompt = prompts['Q&A']
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Test with different numbers of even heads
    results = []
    
    for n_heads in range(1, 17):
        # Try multiple random combinations for each count
        n_trials = min(50, np.math.comb(16, n_heads)) if n_heads <= 16 else 1
        
        successes = 0
        information_content = []
        
        for trial in range(int(n_trials)):
            # Select random even heads
            even_heads = [i for i in range(32) if i % 2 == 0]
            selected = np.random.choice(even_heads, n_heads, replace=False)
            
            with NNsight(model) as nn:
                # Get correct pattern from Simple format
                simple_inputs = tokenizer(prompts['Simple'], return_tensors="pt").to(DEVICE)
                
                with nn.forward(simple_inputs) as tracer:
                    correct_attn = model.layers[LAYER_OF_INTEREST].self_attn.attention_weights.save()
                
                # Apply selected heads only
                with nn.forward(inputs) as tracer:
                    attn = model.layers[LAYER_OF_INTEREST].self_attn
                    
                    def selective_replacement(attn_weights):
                        modified = attn_weights.clone()
                        for head_idx in selected:
                            modified[0, head_idx] = correct_attn.value[0, head_idx]
                        return modified
                    
                    attn.attention_weights.value = selective_replacement(attn.attention_weights.value)
                    output = model.output.save()
                
                output_text = tokenizer.decode(output.value[0].argmax(-1))
                if '9.8' in output_text and 'bigger' in output_text.lower():
                    successes += 1
                
                # Calculate information content (entropy of attention distribution)
                entropy = -torch.sum(
                    correct_attn.value[0, selected] * torch.log(correct_attn.value[0, selected] + 1e-10)
                ).item()
                information_content.append(entropy)
        
        success_rate = successes / n_trials
        avg_information = np.mean(information_content)
        
        results.append({
            'n_heads': n_heads,
            'success_rate': success_rate,
            'information_content': avg_information,
            'n_trials': n_trials
        })
    
    df = pd.DataFrame(results)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Success rate vs number of heads
    ax1.plot(df['n_heads'], df['success_rate'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect success')
    ax1.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='Complete failure')
    ax1.axvline(x=8, color='b', linestyle='--', linewidth=2, label='8-head threshold')
    ax1.fill_between([0, 7.5], 0, 1, alpha=0.2, color='red', label='Failure zone')
    ax1.fill_between([7.5, 16], 0, 1, alpha=0.2, color='green', label='Success zone')
    ax1.set_xlabel('Number of Even Heads Used')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate vs Number of Heads')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Information content vs success
    scatter = ax2.scatter(df['n_heads'], df['information_content'], 
                         c=df['success_rate'], cmap='RdYlGn', s=100, edgecolors='black')
    ax2.axvline(x=8, color='b', linestyle='--', linewidth=2, label='8-head threshold')
    ax2.set_xlabel('Number of Even Heads Used')
    ax2.set_ylabel('Information Content (bits)')
    ax2.set_title('Information Content vs Number of Heads')
    plt.colorbar(scatter, ax=ax2, label='Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('8_head_requirement.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find information threshold
    success_df = df[df['success_rate'] > 0.9]
    if not success_df.empty:
        info_threshold = success_df.iloc[0]['information_content']
        print(f"\n8-Head Requirement Analysis:")
        print(f"Minimum heads for success: {success_df.iloc[0]['n_heads']}")
        print(f"Information threshold: {info_threshold:.2f} bits")
        print(f"Information per head: {info_threshold / success_df.iloc[0]['n_heads']:.2f} bits")
    
    return df

# df_heads = analyze_8_head_requirement(model, tokenizer)
```

---

## Experiment Set 5: Intervention Validation

### 5.1 Surgical Bandwidth Repair

```python
def test_minimal_intervention(model, tokenizer):
    """
    Test if we can fix the bug by only redistributing attention from format to numerical tokens.
    This would strongly support the bandwidth competition theory.
    """
    prompt = prompts['Q&A']
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Identify token positions
    format_positions = [i for i, t in enumerate(tokens) 
                       if any(fmt in t for fmt in ['Q', ':', 'A', 'Which', 'is', 'bigger'])]
    numerical_positions = [i for i, t in enumerate(tokens) 
                          if any(num in t for num in ['9', '8', '11', '.'])]
    
    results = {}
    
    with NNsight(model) as nn:
        # Original output
        with nn.forward(inputs) as tracer:
            original_output = model.output.save()
        results['original'] = tokenizer.decode(original_output.value[0].argmax(-1))
        
        # Minimal intervention: redistribute attention
        with nn.forward(inputs) as tracer:
            attn = model.layers[LAYER_OF_INTEREST].self_attn
            
            def redistribute_attention(attn_weights):
                modified = attn_weights.clone()
                
                for head_idx in range(32):
                    if head_idx % 2 == 1:  # Only fix odd heads
                        # Calculate current bandwidth to format tokens
                        format_attn = modified[0, head_idx, :, format_positions].sum()
                        
                        # Redistribute half of format attention to numerical tokens
                        redistribution = format_attn * 0.5
                        
                        # Reduce format attention
                        modified[0, head_idx, :, format_positions] *= 0.5
                        
                        # Increase numerical attention
                        if len(numerical_positions) > 0:
                            per_num_token = redistribution / len(numerical_positions)
                            for num_pos in numerical_positions:
                                modified[0, head_idx, :, num_pos] += per_num_token / modified.shape[2]
                
                # Renormalize
                modified = modified / modified.sum(dim=-1, keepdim=True)
                return modified
            
            attn.attention_weights.value = redistribute_attention(attn.attention_weights.value)
            repaired_output = model.output.save()
        
        results['redistributed'] = tokenizer.decode(repaired_output.value[0].argmax(-1))
    
    print("\n=== Minimal Intervention Results ===")
    print(f"Original (buggy): {results['original']}")
    print(f"After redistribution: {results['redistributed']}")
    
    success = '9.8' in results['redistributed'] and 'bigger' in results['redistributed'].lower()
    print(f"Success: {success}")
    
    return results

# minimal_results = test_minimal_intervention(model, tokenizer)
```

### 5.2 Attention Mask Intervention

```python
def test_attention_masking(model, tokenizer):
    """
    Force heads to ignore format tokens through masking.
    Tests if preventing format attention fixes odd heads.
    """
    prompt = prompts['Q&A']
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Create mask for format tokens
    format_tokens = ['Q', ':', 'A', 'Which', 'is', 'bigger', '?', 'user', 'assistant']
    format_mask = torch.zeros(len(tokens))
    for i, token in enumerate(tokens):
        if any(fmt in token for fmt in format_tokens):
            format_mask[i] = 1
    
    masking_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for mask_strength in masking_levels:
        with NNsight(model) as nn:
            with nn.forward(inputs) as tracer:
                attn = model.layers[LAYER_OF_INTEREST].self_attn
                
                def apply_format_mask(attn_weights):
                    modified = attn_weights.clone()
                    
                    # Apply mask with varying strength
                    for head_idx in range(32):
                        if head_idx % 2 == 1:  # Only mask odd heads
                            for i, is_format in enumerate(format_mask):
                                if is_format:
                                    modified[0, head_idx, :, i] *= (1 - mask_strength)
                    
                    # Renormalize
                    modified = modified / (modified.sum(dim=-1, keepdim=True) + 1e-10)
                    return modified
                
                attn.attention_weights.value = apply_format_mask(attn.attention_weights.value)
                output = model.output.save()
            
            output_text = tokenizer.decode(output.value[0].argmax(-1))
            is_correct = '9.8' in output_text and 'bigger' in output_text.lower()
            
            results.append({
                'mask_strength': mask_strength,
                'output': output_text[:100],
                'correct': is_correct
            })
    
    df = pd.DataFrame(results)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(df['mask_strength'] * 100, df['correct'].astype(int), 'o-', linewidth=2, markersize=10)
    plt.xlabel('Format Token Masking (%)')
    plt.ylabel('Success')
    plt.title('Effect of Masking Format Tokens (Odd Heads Only)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    # Add annotations
    for idx, row in df.iterrows():
        plt.annotate(f"{row['mask_strength']*100:.0f}%", 
                    (row['mask_strength']*100, row['correct']), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig('attention_masking.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Attention Masking Results ===")
    print(df[['mask_strength', 'correct']])
    
    # Find minimum masking needed
    successful = df[df['correct']]
    if not successful.empty:
        min_masking = successful.iloc[0]['mask_strength']
        print(f"\nMinimum masking needed: {min_masking*100:.0f}%")
    
    return df

# masking_results = test_attention_masking(model, tokenizer)
```

---

## Summary and Statistical Analysis

```python
def run_statistical_summary():
    """
    Run key statistical tests to validate the bandwidth competition theory.
    """
    print("=" * 60)
    print("BANDWIDTH COMPETITION THEORY - STATISTICAL VALIDATION")
    print("=" * 60)
    
    # Key claims to validate
    claims = {
        "Claim 1": "Even heads maintain >40% numerical bandwidth in all formats",
        "Claim 2": "Odd heads fall below 40% numerical bandwidth in Q&A format",
        "Claim 3": "60% pattern replacement is a sharp threshold",
        "Claim 4": "Exactly 8 even heads are necessary and sufficient",
        "Claim 5": "Format tokens compete for limited attention bandwidth"
    }
    
    print("\nKey Claims to Validate:")
    for claim_id, claim_text in claims.items():
        print(f"  {claim_id}: {claim_text}")
    
    print("\n" + "=" * 60)
    print("Run the experiments above to generate data for validation")
    print("=" * 60)

# Uncomment to run after loading model
# run_statistical_summary()
```

## Usage Instructions

```python
"""
To use this experimental framework:

1. Install dependencies:
   pip install torch transformers nnsight matplotlib seaborn pandas scipy statsmodels tqdm

2. Load the model (requires GPU with ~20GB memory):
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

3. Run experiments in order:
   - Start with Experiment Set 1 to validate bandwidth competition
   - Use Set 2 to understand information routing
   - Set 3 validates even/odd specialization
   - Set 4 investigates the thresholds
   - Set 5 tests intervention strategies

4. Each experiment saves visualizations as PNG files

5. Results are returned as pandas DataFrames for further analysis

Note: Some experiments are computationally intensive and may take 10-30 minutes each.
"""
```
