#!/usr/bin/env python3
"""
Analyze Attention Architecture Differences
==========================================

This script investigates the specific attention mechanisms used by Llama, Pythia, and Gemma
to understand why the even/odd head specialization pattern generalizes to some models but not others.
"""

import torch
from transformers import AutoConfig, AutoModel
import json
from datetime import datetime

def analyze_model_architecture(model_name: str) -> dict:
    """Analyze the attention architecture of a specific model"""

    print(f"\nAnalyzing {model_name}...")
    print("=" * 50)

    try:
        config = AutoConfig.from_pretrained(model_name)

        # Basic architecture info
        analysis = {
            'model_name': model_name,
            'model_type': config.model_type,
            'num_attention_heads': getattr(config, 'num_attention_heads', 'Unknown'),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', 'Unknown'),
            'hidden_size': getattr(config, 'hidden_size', 'Unknown'),
            'attention_mechanism': {},
            'architectural_features': {}
        }

        print(f"Model Type: {config.model_type}")
        print(f"Attention Heads: {analysis['num_attention_heads']}")
        print(f"Hidden Layers: {analysis['num_hidden_layers']}")
        print(f"Hidden Size: {analysis['hidden_size']}")

        # Check for Grouped Query Attention (GQA)
        if hasattr(config, 'num_key_value_heads'):
            analysis['attention_mechanism']['grouped_query_attention'] = True
            analysis['attention_mechanism']['num_key_value_heads'] = config.num_key_value_heads
            analysis['attention_mechanism']['gqa_groups'] = analysis['num_attention_heads'] // config.num_key_value_heads
            print(f"✅ Grouped Query Attention: {config.num_key_value_heads} KV heads, {analysis['attention_mechanism']['gqa_groups']} groups")
        else:
            analysis['attention_mechanism']['grouped_query_attention'] = False
            print("❌ Standard Multi-Head Attention (no GQA)")

        # Check for other attention features
        attention_features = [
            'attention_dropout',
            'attention_bias',
            'scale_attn_weights',
            'use_cache',
            'rope_scaling',
            'rope_theta',
            'sliding_window',
            'attention_window_size'
        ]

        for feature in attention_features:
            if hasattr(config, feature):
                value = getattr(config, feature)
                analysis['attention_mechanism'][feature] = value
                print(f"  {feature}: {value}")

        # Model-specific features
        if config.model_type == 'llama':
            # Llama-specific features
            if hasattr(config, 'rope_theta'):
                analysis['architectural_features']['rope_theta'] = config.rope_theta
            if hasattr(config, 'rms_norm_eps'):
                analysis['architectural_features']['normalization'] = 'RMSNorm'
                analysis['architectural_features']['rms_norm_eps'] = config.rms_norm_eps
            print("  Architecture: Llama (RMSNorm, RoPE)")

        elif config.model_type == 'gpt_neox':
            # Pythia/GPT-NeoX specific features
            if hasattr(config, 'layer_norm_eps'):
                analysis['architectural_features']['normalization'] = 'LayerNorm'
                analysis['architectural_features']['layer_norm_eps'] = config.layer_norm_eps
            if hasattr(config, 'rotary_pct'):
                analysis['architectural_features']['rotary_pct'] = config.rotary_pct
            print("  Architecture: GPT-NeoX (LayerNorm, Partial RoPE)")

        elif config.model_type == 'gemma':
            # Gemma-specific features
            if hasattr(config, 'rms_norm_eps'):
                analysis['architectural_features']['normalization'] = 'RMSNorm'
                analysis['architectural_features']['rms_norm_eps'] = config.rms_norm_eps
            if hasattr(config, 'rope_theta'):
                analysis['architectural_features']['rope_theta'] = config.rope_theta
            if hasattr(config, 'attention_bias'):
                analysis['architectural_features']['attention_bias'] = config.attention_bias
            print("  Architecture: Gemma (RMSNorm, RoPE, specialized attention)")

        # Check head dimensions
        if analysis['num_attention_heads'] != 'Unknown' and analysis['hidden_size'] != 'Unknown':
            head_dim = analysis['hidden_size'] // analysis['num_attention_heads']
            analysis['attention_mechanism']['head_dim'] = head_dim
            print(f"  Head Dimension: {head_dim}")

        # Check for any custom attention implementations
        custom_features = [
            'attention_head_type',
            'multi_query_attention',
            'attention_layers',
            'attention_types'
        ]

        for feature in custom_features:
            if hasattr(config, feature):
                value = getattr(config, feature)
                analysis['architectural_features'][feature] = value
                print(f"  Custom: {feature} = {value}")

        return analysis

    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        return {'model_name': model_name, 'error': str(e)}

def compare_attention_mechanisms(analyses: list) -> dict:
    """Compare attention mechanisms across models"""

    print("\n" + "=" * 70)
    print("ATTENTION MECHANISM COMPARISON")
    print("=" * 70)

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'models_analyzed': [a['model_name'] for a in analyses if 'error' not in a],
        'attention_comparison': {},
        'key_differences': [],
        'hypothesis_for_pattern_differences': ""
    }

    # Compare key attention features
    features_to_compare = [
        'grouped_query_attention',
        'num_key_value_heads',
        'head_dim',
        'attention_bias',
        'normalization'
    ]

    print("\nKEY ATTENTION FEATURES:")
    print("-" * 30)

    for feature in features_to_compare:
        print(f"\n{feature.upper()}:")
        feature_values = {}

        for analysis in analyses:
            if 'error' in analysis:
                continue

            model_name = analysis['model_name'].split('/')[-1]

            # Look in both attention_mechanism and architectural_features
            value = analysis['attention_mechanism'].get(feature,
                   analysis['architectural_features'].get(feature, 'Not specified'))

            feature_values[model_name] = value
            print(f"  {model_name}: {value}")

        comparison['attention_comparison'][feature] = feature_values

    # Analyze differences
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES ANALYSIS")
    print("=" * 70)

    # GQA Analysis
    gqa_models = []
    standard_models = []

    for analysis in analyses:
        if 'error' in analysis:
            continue

        model_name = analysis['model_name'].split('/')[-1]
        if analysis['attention_mechanism'].get('grouped_query_attention', False):
            gqa_models.append(model_name)
        else:
            standard_models.append(model_name)

    print(f"\nGrouped Query Attention (GQA):")
    print(f"  ✅ Uses GQA: {gqa_models}")
    print(f"  ❌ Standard MHA: {standard_models}")

    if gqa_models and standard_models:
        comparison['key_differences'].append({
            'feature': 'Grouped Query Attention',
            'gqa_models': gqa_models,
            'standard_models': standard_models,
            'potential_impact': 'GQA groups heads functionally, may affect specialization patterns'
        })

    # Head count analysis
    print(f"\nAttention Head Counts:")
    head_counts = {}
    for analysis in analyses:
        if 'error' in analysis:
            continue
        model_name = analysis['model_name'].split('/')[-1]
        head_count = analysis['num_attention_heads']
        head_counts[model_name] = head_count
        print(f"  {model_name}: {head_count} heads")

    # Normalization analysis
    print(f"\nNormalization Methods:")
    norm_methods = {}
    for analysis in analyses:
        if 'error' in analysis:
            continue
        model_name = analysis['model_name'].split('/')[-1]
        norm_method = analysis['architectural_features'].get('normalization', 'Unknown')
        norm_methods[model_name] = norm_method
        print(f"  {model_name}: {norm_method}")

    # Generate hypothesis
    print("\n" + "=" * 70)
    print("HYPOTHESIS FOR PATTERN DIFFERENCES")
    print("=" * 70)

    hypothesis_parts = []

    # GQA hypothesis
    if gqa_models and standard_models:
        if 'Llama-3.1-8B' in gqa_models and 'pythia-160m' in standard_models:
            hypothesis_parts.append(
                "GQA vs Standard MHA doesn't explain the pattern - both Llama (GQA) and Pythia (standard) show even/odd specialization"
            )

    # Model family hypothesis
    gemma_different = any('gemma' in analysis['model_name'].lower() for analysis in analyses
                         if 'error' not in analysis)
    if gemma_different:
        hypothesis_parts.append(
            "Gemma's different attention implementation (despite using RMSNorm like Llama) may prevent even/odd specialization"
        )

    # Training hypothesis
    hypothesis_parts.append(
        "The pattern may be more related to training data/methodology than pure architecture - "
        "Llama and Pythia may have similar training dynamics that encourage even/odd specialization"
    )

    # Head count hypothesis
    unique_head_counts = set(head_counts.values())
    if len(unique_head_counts) > 1:
        hypothesis_parts.append(
            f"Different head counts ({list(unique_head_counts)}) may affect the emergence of specialization patterns"
        )

    comparison['hypothesis_for_pattern_differences'] = " | ".join(hypothesis_parts)

    for i, part in enumerate(hypothesis_parts, 1):
        print(f"{i}. {part}")

    return comparison

def main():
    """Analyze attention architectures across models"""

    models_to_analyze = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "EleutherAI/pythia-160m",
        "google/gemma-2b"
    ]

    print("ATTENTION ARCHITECTURE ANALYSIS")
    print("=" * 70)
    print("Investigating why even/odd patterns generalize to some models but not others")

    # Analyze each model
    analyses = []
    for model_name in models_to_analyze:
        analysis = analyze_model_architecture(model_name)
        analyses.append(analysis)

    # Compare mechanisms
    comparison = compare_attention_mechanisms(analyses)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"attention_architecture_analysis_{timestamp}.json"

    results = {
        'individual_analyses': analyses,
        'comparison': comparison
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Key architectural differences identified that may explain")
    print("why even/odd head specialization patterns vary across models.")
    print("See JSON file for complete technical details.")

if __name__ == "__main__":
    main()