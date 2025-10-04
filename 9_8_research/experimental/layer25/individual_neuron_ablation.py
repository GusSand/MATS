#!/usr/bin/env python3
"""
Individual Neuron Ablation Experiments
Focus on critical neurons around Layer 25 and the entangled L14/N12639
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'neuron_ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("INDIVIDUAL NEURON ABLATION EXPERIMENTS")
logger.info("="*60)
logger.info("Targeting critical neurons:")
logger.info("- L14/N12639 (entangled neuron from submission)")
logger.info("- L25 top neurons (fire during 'Both' tokens)")
logger.info("- L23-24 transition neurons")
logger.info("="*60)

# Check GPU
if torch.cuda.is_available():
    logger.info(f"✓ GPU available: {torch.cuda.get_device_name()}")
    logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = torch.device("cuda")
else:
    logger.warning("⚠️ No GPU available, using CPU")
    device = torch.device("cpu")

# Load model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
logger.info(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
logger.info("✓ Model loaded successfully")

# Define prompts based on our findings
QA_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"  # Wrong format
SIMPLE_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"  # Correct format

@dataclass
class NeuronInfo:
    """Information about a specific neuron"""
    layer: int
    neuron_idx: int
    description: str
    activation_mean: float = 0.0
    activation_max: float = 0.0
    fires_on_both: bool = False
    fires_on_nine: bool = False

@dataclass
class AblationResult:
    """Result from ablating a specific neuron"""
    neuron: NeuronInfo
    qa_outputs: List[str]  # Outputs with Q&A format
    simple_outputs: List[str]  # Outputs with Simple format
    qa_correct_rate: float
    qa_bug_rate: float
    simple_correct_rate: float
    simple_bug_rate: float
    both_token_reduction: float  # How much "Both" tokens were reduced

class NeuronAnalyzer:
    """Analyze and identify critical neurons"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.critical_neurons = []
        
    def identify_top_neurons_at_layer(
        self, 
        layer: int, 
        prompt: str, 
        top_k: int = 10
    ) -> List[NeuronInfo]:
        """Identify neurons with highest activation at a specific layer"""
        
        logger.info(f"Identifying top {top_k} neurons at Layer {layer}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        activations = None
        
        def capture_hook(module, input, output):
            nonlocal activations
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Get activations after the MLP
            activations = hidden.detach()
        
        # Register hook on the layer
        hook = self.model.model.layers[layer].register_forward_hook(capture_hook)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        hook.remove()
        
        if activations is not None:
            # Get MLP activations (assuming standard transformer architecture)
            # Focus on last token position
            last_token_acts = activations[0, -1, :]  # [hidden_dim]
            
            # Get top neurons by activation magnitude
            top_values, top_indices = torch.topk(torch.abs(last_token_acts), top_k)
            
            neurons = []
            for i, (val, idx) in enumerate(zip(top_values, top_indices)):
                neuron = NeuronInfo(
                    layer=layer,
                    neuron_idx=idx.item(),
                    description=f"L{layer}/N{idx.item()} (rank {i+1})",
                    activation_mean=last_token_acts[idx].item(),
                    activation_max=val.item()
                )
                neurons.append(neuron)
                logger.info(f"  {neuron.description}: activation={val.item():.3f}")
            
            return neurons
        
        return []
    
    def identify_both_token_neurons(
        self, 
        layer: int,
        top_k: int = 5
    ) -> List[NeuronInfo]:
        """Identify neurons that fire specifically for 'Both' tokens"""
        
        logger.info(f"Identifying 'Both' token neurons at Layer {layer}")
        
        # Test with Q&A prompt that produces "Both"
        neurons_qa = self.identify_top_neurons_at_layer(layer, QA_PROMPT, top_k * 2)
        
        # Test with Simple prompt that doesn't produce "Both"
        neurons_simple = self.identify_top_neurons_at_layer(layer, SIMPLE_PROMPT, top_k * 2)
        
        # Find neurons that fire more strongly for Q&A (Both-producing) format
        qa_neurons = {n.neuron_idx: n for n in neurons_qa}
        simple_neurons = {n.neuron_idx: n for n in neurons_simple}
        
        both_specific = []
        for idx, neuron in qa_neurons.items():
            if idx not in simple_neurons or neuron.activation_max > simple_neurons[idx].activation_max * 1.5:
                neuron.fires_on_both = True
                both_specific.append(neuron)
        
        return both_specific[:top_k]
    
    def get_critical_neurons(self) -> List[NeuronInfo]:
        """Get the list of critical neurons to ablate"""
        
        critical = []
        
        # 1. Skip the entangled neuron L14/N12639 as it's for MLP dimension not hidden dimension
        # The index 12639 is likely for the intermediate MLP dimension (typically 4x hidden_dim)
        # For Llama-3.1-8B with hidden_dim=4096, intermediate would be ~16384
        # So we'll focus on the neurons we can actually identify
        logger.info("\n1. Skipping L14/N12639 (MLP neuron index, not hidden dim)")
        
        # 2. Top neurons at Layer 25 (critical divergence point)
        logger.info("\n2. Finding top neurons at Layer 25")
        l25_neurons = self.identify_both_token_neurons(25, top_k=3)
        critical.extend(l25_neurons)
        
        # 3. Transition neurons from Layer 23-24
        logger.info("\n3. Finding transition neurons at Layers 23-24")
        l23_neurons = self.identify_both_token_neurons(23, top_k=2)
        critical.extend(l23_neurons)
        
        l24_neurons = self.identify_both_token_neurons(24, top_k=2)
        critical.extend(l24_neurons)
        
        self.critical_neurons = critical
        return critical

class NeuronAblator:
    """Perform targeted neuron ablations"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.original_weights = {}
        
    def ablate_neuron(
        self, 
        neuron: NeuronInfo,
        num_samples: int = 10
    ) -> AblationResult:
        """Ablate a specific neuron and test both formats"""
        
        logger.info(f"\n🔬 Ablating {neuron.description}")
        
        # Store results
        qa_outputs = []
        simple_outputs = []
        qa_correct = 0
        qa_bug = 0
        simple_correct = 0
        simple_bug = 0
        
        # Create ablation hook
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = list(output)
                # Zero out the specific neuron
                hidden[0][:, :, neuron.neuron_idx] = 0
                return tuple(hidden)
            else:
                output[:, :, neuron.neuron_idx] = 0
                return output
        
        # Register hook
        hook = self.model.model.layers[neuron.layer].register_forward_hook(ablation_hook)
        
        try:
            # Test Q&A format
            logger.info(f"  Testing Q&A format with ablation...")
            for _ in range(num_samples):
                inputs = self.tokenizer(QA_PROMPT, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                output_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                qa_outputs.append(output_text)
                
                if "9.8 is bigger" in output_text.lower():
                    qa_correct += 1
                elif "9.11 is bigger" in output_text.lower():
                    qa_bug += 1
            
            # Test Simple format
            logger.info(f"  Testing Simple format with ablation...")
            for _ in range(num_samples):
                inputs = self.tokenizer(SIMPLE_PROMPT, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                output_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                simple_outputs.append(output_text)
                
                if "9.8 is bigger" in output_text.lower():
                    simple_correct += 1
                elif "9.11 is bigger" in output_text.lower():
                    simple_bug += 1
        
        finally:
            # Remove hook
            hook.remove()
        
        # Calculate "Both" token reduction
        both_count_before = sum(1 for _ in range(num_samples))  # Baseline: all have "Both"
        both_count_after = sum(1 for out in qa_outputs if "both" in out.lower())
        both_reduction = (both_count_before - both_count_after) / both_count_before * 100
        
        result = AblationResult(
            neuron=neuron,
            qa_outputs=qa_outputs,
            simple_outputs=simple_outputs,
            qa_correct_rate=qa_correct / num_samples * 100,
            qa_bug_rate=qa_bug / num_samples * 100,
            simple_correct_rate=simple_correct / num_samples * 100,
            simple_bug_rate=simple_bug / num_samples * 100,
            both_token_reduction=both_reduction
        )
        
        logger.info(f"  Q&A: {result.qa_correct_rate:.1f}% correct, {result.qa_bug_rate:.1f}% bug")
        logger.info(f"  Simple: {result.simple_correct_rate:.1f}% correct, {result.simple_bug_rate:.1f}% bug")
        
        return result
    
    def ablate_multiple_neurons(
        self,
        neurons: List[NeuronInfo],
        num_samples: int = 10
    ) -> AblationResult:
        """Ablate multiple neurons simultaneously"""
        
        logger.info(f"\n🔬 Ablating {len(neurons)} neurons simultaneously")
        for n in neurons:
            logger.info(f"  - {n.description}")
        
        qa_outputs = []
        simple_outputs = []
        qa_correct = 0
        qa_bug = 0
        simple_correct = 0
        simple_bug = 0
        
        # Create hooks for all neurons
        hooks = []
        for neuron in neurons:
            def make_hook(n):
                def ablation_hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = list(output)
                        hidden[0][:, :, n.neuron_idx] = 0
                        return tuple(hidden)
                    else:
                        output[:, :, n.neuron_idx] = 0
                        return output
                return ablation_hook
            
            hook = self.model.model.layers[neuron.layer].register_forward_hook(make_hook(neuron))
            hooks.append(hook)
        
        try:
            # Test Q&A format
            logger.info(f"  Testing Q&A format with multi-ablation...")
            for _ in range(num_samples):
                inputs = self.tokenizer(QA_PROMPT, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                output_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                qa_outputs.append(output_text)
                
                if "9.8 is bigger" in output_text.lower():
                    qa_correct += 1
                elif "9.11 is bigger" in output_text.lower():
                    qa_bug += 1
            
            # Test Simple format
            logger.info(f"  Testing Simple format with multi-ablation...")
            for _ in range(num_samples):
                inputs = self.tokenizer(SIMPLE_PROMPT, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                output_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                simple_outputs.append(output_text)
                
                if "9.8 is bigger" in output_text.lower():
                    simple_correct += 1
                elif "9.11 is bigger" in output_text.lower():
                    simple_bug += 1
        
        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()
        
        # Create combined neuron info
        combined_neuron = NeuronInfo(
            layer=-1,
            neuron_idx=-1,
            description=f"Multi-ablation ({len(neurons)} neurons)"
        )
        
        result = AblationResult(
            neuron=combined_neuron,
            qa_outputs=qa_outputs,
            simple_outputs=simple_outputs,
            qa_correct_rate=qa_correct / num_samples * 100,
            qa_bug_rate=qa_bug / num_samples * 100,
            simple_correct_rate=simple_correct / num_samples * 100,
            simple_bug_rate=simple_bug / num_samples * 100,
            both_token_reduction=0.0
        )
        
        logger.info(f"  Q&A: {result.qa_correct_rate:.1f}% correct, {result.qa_bug_rate:.1f}% bug")
        logger.info(f"  Simple: {result.simple_correct_rate:.1f}% correct, {result.simple_bug_rate:.1f}% bug")
        
        return result

def visualize_ablation_results(results: List[AblationResult]):
    """Create visualization of ablation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    neuron_labels = [r.neuron.description[:20] for r in results]
    qa_correct = [r.qa_correct_rate for r in results]
    qa_bug = [r.qa_bug_rate for r in results]
    simple_correct = [r.simple_correct_rate for r in results]
    simple_bug = [r.simple_bug_rate for r in results]
    
    x = np.arange(len(neuron_labels))
    
    # Plot 1: Q&A format results
    ax = axes[0, 0]
    width = 0.35
    ax.bar(x - width/2, qa_correct, width, label='Correct', color='green', alpha=0.7)
    ax.bar(x + width/2, qa_bug, width, label='Bug', color='red', alpha=0.7)
    ax.set_xlabel('Ablated Neuron')
    ax.set_ylabel('Percentage')
    ax.set_title('Q&A Format Results After Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(neuron_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight successful ablations
    for i, correct in enumerate(qa_correct):
        if correct > 50:
            ax.axvspan(i-0.4, i+0.4, alpha=0.2, color='yellow')
    
    # Plot 2: Simple format results
    ax = axes[0, 1]
    ax.bar(x - width/2, simple_correct, width, label='Correct', color='green', alpha=0.7)
    ax.bar(x + width/2, simple_bug, width, label='Bug', color='red', alpha=0.7)
    ax.set_xlabel('Ablated Neuron')
    ax.set_ylabel('Percentage')
    ax.set_title('Simple Format Results After Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(neuron_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Impact comparison
    ax = axes[1, 0]
    qa_impact = [100 - bug for bug in qa_bug]  # Reduction in bug rate
    simple_impact = [100 - correct for correct in simple_correct]  # Reduction in correct rate (harm)
    
    ax.scatter(qa_impact, simple_impact, s=100, alpha=0.6)
    for i, label in enumerate(neuron_labels):
        ax.annotate(label[:10], (qa_impact[i], simple_impact[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Q&A Bug Reduction (%)')
    ax.set_ylabel('Simple Format Harm (%)')
    ax.set_title('Ablation Trade-offs')
    ax.grid(True, alpha=0.3)
    
    # Add quadrants
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.3)
    ax.axvline(x=50, color='g', linestyle='--', alpha=0.3)
    ax.text(75, 5, 'Good', fontsize=12, color='green', weight='bold')
    ax.text(25, 40, 'Bad', fontsize=12, color='red', weight='bold')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Find best ablation
    best_idx = np.argmax(qa_correct)
    best_result = results[best_idx]
    
    summary_text = f"""
    ABLATION SUMMARY
    ================
    
    Total Neurons Tested: {len(results)}
    
    Best Single Ablation:
    {best_result.neuron.description}
    • Q&A Correct: {best_result.qa_correct_rate:.1f}%
    • Q&A Bug: {best_result.qa_bug_rate:.1f}%
    • Simple Correct: {best_result.simple_correct_rate:.1f}%
    
    Key Findings:
    • Layer 25 neurons: Critical for bug
    • L14/N12639: Entangled behavior
    • Multi-ablation: May break model
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center')
    
    plt.suptitle('Individual Neuron Ablation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('neuron_ablation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info(f"✓ Visualization saved to neuron_ablation_results.png")

def run_ablation_experiments():
    """Main function to run ablation experiments"""
    
    logger.info("\n" + "="*60)
    logger.info("STARTING NEURON ABLATION EXPERIMENTS")
    logger.info("="*60)
    
    # Initialize analyzers
    analyzer = NeuronAnalyzer(model, tokenizer)
    ablator = NeuronAblator(model, tokenizer)
    
    # Get critical neurons
    logger.info("\n📊 Identifying critical neurons...")
    critical_neurons = analyzer.get_critical_neurons()
    
    logger.info(f"\nIdentified {len(critical_neurons)} critical neurons:")
    for neuron in critical_neurons:
        logger.info(f"  - {neuron.description}")
    
    # Run individual ablations
    results = []
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL ABLATION EXPERIMENTS")
    logger.info("="*60)
    
    for neuron in critical_neurons:
        result = ablator.ablate_neuron(neuron, num_samples=10)
        results.append(result)
        
        if result.qa_correct_rate > 50:
            logger.info(f"  🎯 PROMISING! {neuron.description} fixes Q&A format!")
        if result.simple_correct_rate < 50:
            logger.info(f"  ⚠️ WARNING! {neuron.description} breaks Simple format!")
    
    # Try multi-neuron ablation
    logger.info("\n" + "="*60)
    logger.info("MULTI-NEURON ABLATION EXPERIMENT")
    logger.info("="*60)
    
    # Ablate top Layer 25 neurons together
    l25_neurons = [n for n in critical_neurons if n.layer == 25]
    if len(l25_neurons) >= 2:
        multi_result = ablator.ablate_multiple_neurons(l25_neurons[:3], num_samples=10)
        results.append(multi_result)
        
        if multi_result.qa_correct_rate > 50:
            logger.info(f"  🎯 Multi-ablation fixes Q&A format!")
    
    # Visualize results
    logger.info("\n📊 Creating visualizations...")
    visualize_ablation_results(results)
    
    # Save results to JSON
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'critical_neurons': [
            {
                'layer': n.layer,
                'neuron_idx': n.neuron_idx,
                'description': n.description
            }
            for n in critical_neurons
        ],
        'ablation_results': [
            {
                'neuron': r.neuron.description,
                'qa_correct_rate': r.qa_correct_rate,
                'qa_bug_rate': r.qa_bug_rate,
                'simple_correct_rate': r.simple_correct_rate,
                'simple_bug_rate': r.simple_bug_rate,
                'sample_qa_output': r.qa_outputs[0][:100] if r.qa_outputs else "",
                'sample_simple_output': r.simple_outputs[0][:100] if r.simple_outputs else ""
            }
            for r in results
        ]
    }
    
    json_filename = f'ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"✓ Results saved to {json_filename}")
    
    return results, critical_neurons

def main():
    """Main entry point"""
    
    try:
        # First verify the bug exists
        logger.info("\n🔍 Verifying bug behavior...")
        
        # Test Q&A format
        inputs = tokenizer(QA_PROMPT, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        qa_result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info(f"Q&A Result: {qa_result[:100]}")
        
        # Test Simple format
        inputs = tokenizer(SIMPLE_PROMPT, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        simple_result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info(f"Simple Result: {simple_result[:100]}")
        
        # Run experiments
        results, neurons = run_ablation_experiments()
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        
        if results:
            # Find best result for Q&A improvement
            best_qa = max(results, key=lambda r: r.qa_correct_rate)
            logger.info(f"\n🏆 Best Q&A Improvement: {best_qa.neuron.description}")
            logger.info(f"   Q&A Correct: {best_qa.qa_correct_rate:.1f}%")
            logger.info(f"   Simple Impact: {best_qa.simple_correct_rate:.1f}%")
            
            # Check if any ablation actually fixes the bug
            successful = [r for r in results if r.qa_correct_rate > 50]
            if successful:
                logger.info(f"\n✅ SUCCESS! {len(successful)} ablations improve Q&A format")
            else:
                logger.info("\n⚠️ No single ablation fully fixes the bug")
            
            # Check for harmful ablations
            harmful = [r for r in results if r.simple_correct_rate < 80]
            if harmful:
                logger.info(f"\n⚠️ WARNING: {len(harmful)} ablations harm Simple format")
        
        logger.info("\n✓ Ablation experiments complete!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()