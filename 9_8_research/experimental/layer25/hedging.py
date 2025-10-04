import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime
import json

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'hedging_intervention_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("HEDGING INTERVENTION EXPERIMENTS")
logger.info("="*60)
logger.info("Based on findings from logitlens analysis:")
logger.info("- Temperature MUST be 0.0 for deterministic behavior")
logger.info("- Layer 25 is the critical divergence point")
logger.info("- Q&A format produces wrong answers, Simple format produces correct answers")
logger.info("="*60)

# Check GPU
if torch.cuda.is_available():
    logger.info(f"✓ GPU available: {torch.cuda.get_device_name()}")
    logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    logger.warning("⚠️ No GPU available, using CPU")

# Load model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
logger.info(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
logger.info("✓ Model loaded successfully")

# Define prompts based on our findings
# From SUMMARY.md: Q&A format gives wrong answer, Simple format gives correct answer
QA_PROMPT = "Q: Which is bigger: 9.8 or 9.11?\nA:"  # Wrong format (produces "9.11 is bigger")
SIMPLE_PROMPT = "Which is bigger: 9.8 or 9.11?\nAnswer:"  # Correct format (produces "9.8 is bigger")

# Chat format for completeness (deterministic gives wrong, stochastic gives empty)
CHAT_PROMPT = """<|start_header_id|>user<|end_header_id|>
Which is bigger: 9.8 or 9.11?
<|start_header_id|>assistant<|end_header_id|>"""

logger.info("\nPrompt formats:")
logger.info(f"  Wrong format (Q&A): '{QA_PROMPT}'")
logger.info(f"  Correct format (Simple): '{SIMPLE_PROMPT}'")
logger.info(f"  Chat format: [truncated for display]")

@dataclass
class InterventionResult:
    """Store results from an intervention experiment"""
    description: str
    correct_rate: float
    bug_rate: float
    incoherent_rate: float
    both_token_rate: float  # New: track "Both" tokens
    outputs: List[str]
    token_probs: Dict[str, List[float]]  # Track specific token probabilities

class HedgingInterventions:
    """
    Targeted interventions for the "Both" → "9" transition at Layer 25
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Get token IDs for key tokens
        self.token_ids = {
            '9': tokenizer.encode('9', add_special_tokens=False)[0],
            '8': tokenizer.encode('8', add_special_tokens=False)[0],
            '11': tokenizer.encode('11', add_special_tokens=False)[0],
            'Both': tokenizer.encode('Both', add_special_tokens=False)[0],
            'both': tokenizer.encode('both', add_special_tokens=False)[0],
        }
        
        # Add space variants
        for key in list(self.token_ids.keys()):
            space_key = ' ' + key
            try:
                self.token_ids[space_key] = tokenizer.encode(space_key, add_special_tokens=False)[0]
            except:
                pass
    
    def analyze_hedging_zone(self, prompt: str, layers_to_check: List[int] = None):
        """
        Analyze where "Both" tokens appear and their probabilities
        """
        if layers_to_check is None:
            layers_to_check = list(range(20, 26))  # Focus on hedging zone
        
        logger.debug(f"Analyzing hedging zone for prompt type: {prompt[:20]}...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        layer_probs = {}
        
        for layer_idx in layers_to_check:
            # Hook to capture hidden states
            hidden_state = None
            
            def capture_hook(module, input, output):
                nonlocal hidden_state
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
            
            # Register hook
            hook = self.model.model.layers[layer_idx].register_forward_hook(capture_hook)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(**inputs)
            
            hook.remove()
            
            # Apply LN and get logits
            if hidden_state is not None:
                normalized = self.model.model.norm(hidden_state)
                logits = self.model.lm_head(normalized)
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                
                # Track probabilities of key tokens
                layer_probs[layer_idx] = {
                    'Both': probs[self.token_ids.get('Both', 0)].item(),
                    'both': probs[self.token_ids.get('both', 0)].item(),
                    '9': probs[self.token_ids.get('9', 0)].item(),
                    '8': probs[self.token_ids.get('8', 0)].item(),
                }
        
        return layer_probs

    def intervention_1_suppress_both_tokens(
        self,
        bad_prompt: str,
        layers: List[int] = None,
        suppression_factor: float = 0.01,
        num_samples: int = 20
    ) -> InterventionResult:
        """
        Intervention 1: Directly suppress "Both" token probabilities
        """
        if layers is None:
            layers = [20, 21, 22, 23, 24, 25]
        
        logger.info(f"\n🔬 Intervention 1: Suppress 'Both' tokens at layers {layers}")
        logger.info(f"   Suppression factor: {suppression_factor}")
        logger.info(f"   Running {num_samples} samples with deterministic generation...")
        
        results = {
            'outputs': [],
            'correct': 0,
            'bug': 0,
            'incoherent': 0,
            'has_both': 0
        }
        
        for _ in range(num_samples):
            inputs = self.tokenizer(bad_prompt, return_tensors="pt").to(self.model.device)
            
            # Hook to modify logits
            def suppress_both_hook(module, input, output):
                # Modify logits to suppress "Both" tokens
                for token_key in ['Both', 'both', ' Both', ' both']:
                    if token_key in self.token_ids:
                        token_id = self.token_ids[token_key]
                        output[:, :, token_id] *= suppression_factor
                return output
            
            # Register hooks on LM head for specified layers
            hooks = []
            for layer_idx in layers:
                hook = self.model.lm_head.register_forward_hook(suppress_both_hook)
                hooks.append(hook)
            
            # Generate with DETERMINISTIC settings (critical for bug reproduction!)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.0,  # MUST be 0.0 for deterministic behavior
                    do_sample=False,  # MUST be False to avoid sampling
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Analyze output
            output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            results['outputs'].append(output_text)
            
            # Check which number is stated as bigger - be more precise!
            if "9.8 is bigger" in output_text.lower() or "9.8 is larger" in output_text.lower():
                results['correct'] += 1
            elif "9.11 is bigger" in output_text.lower() or "9.11 is larger" in output_text.lower():
                results['bug'] += 1
            else:
                results['incoherent'] += 1
            
            if "both" in output_text.lower():
                results['has_both'] += 1
        
        return InterventionResult(
            description=f"Suppress 'Both' at L{layers}",
            correct_rate=results['correct'] / num_samples * 100,
            bug_rate=results['bug'] / num_samples * 100,
            incoherent_rate=results['incoherent'] / num_samples * 100,
            both_token_rate=results['has_both'] / num_samples * 100,
            outputs=results['outputs'],
            token_probs={}
        )

    def intervention_2_boost_commitment_token(
        self,
        bad_prompt: str,
        target_token: str = '9',
        boost_layer: int = 25,
        boost_factor: float = 10.0,
        num_samples: int = 20
    ) -> InterventionResult:
        """
        Intervention 2: Boost probability of commitment token ("9") at Layer 25
        """
        print(f"\n🔬 Intervention 2: Boost '{target_token}' token at layer {boost_layer}")
        print(f"   Boost factor: {boost_factor}x")
        
        results = {
            'outputs': [],
            'correct': 0,
            'bug': 0,
            'incoherent': 0,
            'has_both': 0
        }
        
        for _ in range(num_samples):
            inputs = self.tokenizer(bad_prompt, return_tensors="pt").to(self.model.device)
            
            # Hook to boost specific token at specific layer
            def boost_token_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                # Apply LN and get logits
                normalized = self.model.model.norm(hidden)
                logits = self.model.lm_head(normalized)
                
                # Boost target token
                if target_token in self.token_ids:
                    token_id = self.token_ids[target_token]
                    logits[:, :, token_id] *= boost_factor
                
                return output
            
            # Register hook at specific layer
            hook = self.model.model.layers[boost_layer].register_forward_hook(boost_token_hook)
            
            # Generate with DETERMINISTIC settings (critical for bug reproduction!)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.0,  # MUST be 0.0 for deterministic behavior
                    do_sample=False,  # MUST be False to avoid sampling
                    pad_token_id=tokenizer.eos_token_id
                )
            
            hook.remove()
            
            # Analyze output
            output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            results['outputs'].append(output_text)
            
            # Check which number is stated as bigger - be more precise!
            if "9.8 is bigger" in output_text.lower() or "9.8 is larger" in output_text.lower():
                results['correct'] += 1
            elif "9.11 is bigger" in output_text.lower() or "9.11 is larger" in output_text.lower():
                results['bug'] += 1
            else:
                results['incoherent'] += 1
            
            if "both" in output_text.lower():
                results['has_both'] += 1
        
        return InterventionResult(
            description=f"Boost '{target_token}' at L{boost_layer}",
            correct_rate=results['correct'] / num_samples * 100,
            bug_rate=results['bug'] / num_samples * 100,
            incoherent_rate=results['incoherent'] / num_samples * 100,
            both_token_rate=results['has_both'] / num_samples * 100,
            outputs=results['outputs'],
            token_probs={}
        )

    def intervention_3_redirect_both_to_nine(
        self,
        bad_prompt: str,
        redirect_layers: List[int] = None,
        num_samples: int = 20
    ) -> InterventionResult:
        """
        Intervention 3: When model wants to say "Both", redirect to "9"
        """
        if redirect_layers is None:
            redirect_layers = [23, 24, 25]
        
        print(f"\n🔬 Intervention 3: Redirect 'Both' → '9' at layers {redirect_layers}")
        
        results = {
            'outputs': [],
            'correct': 0,
            'bug': 0,
            'incoherent': 0,
            'has_both': 0,
            'redirects_triggered': 0
        }
        
        for _ in range(num_samples):
            inputs = self.tokenizer(bad_prompt, return_tensors="pt").to(self.model.device)
            redirect_count = 0
            
            def redirect_hook(module, input, output):
                nonlocal redirect_count
                # Get logits
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                normalized = self.model.model.norm(hidden)
                logits = self.model.lm_head(normalized)
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                
                # Check if "Both" is likely
                both_prob = 0
                for token_key in ['Both', 'both']:
                    if token_key in self.token_ids:
                        both_prob += probs[self.token_ids[token_key]].item()
                
                if both_prob > 0.1:  # If "Both" is likely
                    redirect_count += 1
                    # Suppress "Both" and boost "9"
                    for token_key in ['Both', 'both']:
                        if token_key in self.token_ids:
                            logits[:, :, self.token_ids[token_key]] -= 10
                    
                    if '9' in self.token_ids:
                        logits[:, :, self.token_ids['9']] += 5
                
                return output
            
            # Register hooks
            hooks = []
            for layer_idx in redirect_layers:
                hook = self.model.model.layers[layer_idx].register_forward_hook(redirect_hook)
                hooks.append(hook)
            
            # Generate with DETERMINISTIC settings (critical for bug reproduction!)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.0,  # MUST be 0.0 for deterministic behavior
                    do_sample=False,  # MUST be False to avoid sampling
                    pad_token_id=tokenizer.eos_token_id
                )
            
            for hook in hooks:
                hook.remove()
            
            # Track redirects
            if redirect_count > 0:
                results['redirects_triggered'] += 1
            
            # Analyze output
            output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            results['outputs'].append(output_text)
            
            # Check which number is stated as bigger - be more precise!
            if "9.8 is bigger" in output_text.lower() or "9.8 is larger" in output_text.lower():
                results['correct'] += 1
            elif "9.11 is bigger" in output_text.lower() or "9.11 is larger" in output_text.lower():
                results['bug'] += 1
            else:
                results['incoherent'] += 1
            
            if "both" in output_text.lower():
                results['has_both'] += 1
        
        print(f"   Redirects triggered in {results['redirects_triggered']}/{num_samples} samples")
        
        return InterventionResult(
            description=f"Redirect Both→9 at L{redirect_layers}",
            correct_rate=results['correct'] / num_samples * 100,
            bug_rate=results['bug'] / num_samples * 100,
            incoherent_rate=results['incoherent'] / num_samples * 100,
            both_token_rate=results['has_both'] / num_samples * 100,
            outputs=results['outputs'],
            token_probs={}
        )

    def intervention_4_transplant_commitment_pattern(
        self,
        bad_prompt: str,
        good_prompt: str,
        commitment_layers: List[int] = None,
        num_samples: int = 20
    ) -> InterventionResult:
        """
        Intervention 4: Transplant the commitment pattern from good to bad format
        Specifically target the transition from hedging to commitment
        """
        if commitment_layers is None:
            commitment_layers = [24, 25, 26]  # The commitment zone
        
        print(f"\n🔬 Intervention 4: Transplant commitment pattern at layers {commitment_layers}")
        
        results = {
            'outputs': [],
            'correct': 0,
            'bug': 0,
            'incoherent': 0,
            'has_both': 0
        }
        
        # First, analyze what happens in good format at these layers
        good_inputs = self.tokenizer(good_prompt, return_tensors="pt").to(self.model.device)
        good_patterns = {}
        
        for layer_idx in commitment_layers:
            hidden_state = None
            
            def capture_hook(module, input, output):
                nonlocal hidden_state
                if isinstance(output, tuple):
                    hidden_state = output[0].detach()
                else:
                    hidden_state = output.detach()
            
            hook = self.model.model.layers[layer_idx].register_forward_hook(capture_hook)
            
            with torch.no_grad():
                _ = self.model(**good_inputs)
            
            hook.remove()
            
            # Store the "commitment pattern" - the final token's hidden state
            if hidden_state is not None:
                good_patterns[layer_idx] = hidden_state[:, -1, :].clone()
        
        # Now apply this pattern during bad prompt generation
        for _ in range(num_samples):
            inputs = self.tokenizer(bad_prompt, return_tensors="pt").to(self.model.device)
            
            def transplant_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = list(output)
                        # Blend the commitment pattern
                        hidden[0][:, -1, :] = 0.7 * hidden[0][:, -1, :] + 0.3 * good_patterns[layer_idx].to(hidden[0].device)
                        return tuple(hidden)
                    else:
                        output[:, -1, :] = 0.7 * output[:, -1, :] + 0.3 * good_patterns[layer_idx].to(output.device)
                        return output
                return hook
            
            # Register hooks
            hooks = []
            for layer_idx in commitment_layers:
                hook = self.model.model.layers[layer_idx].register_forward_hook(transplant_hook(layer_idx))
                hooks.append(hook)
            
            # Generate with DETERMINISTIC settings (critical for bug reproduction!)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.0,  # MUST be 0.0 for deterministic behavior
                    do_sample=False,  # MUST be False to avoid sampling
                    pad_token_id=tokenizer.eos_token_id
                )
            
            for hook in hooks:
                hook.remove()
            
            # Analyze output
            output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            results['outputs'].append(output_text)
            
            # Check which number is stated as bigger - be more precise!
            if "9.8 is bigger" in output_text.lower() or "9.8 is larger" in output_text.lower():
                results['correct'] += 1
            elif "9.11 is bigger" in output_text.lower() or "9.11 is larger" in output_text.lower():
                results['bug'] += 1
            else:
                results['incoherent'] += 1
            
            if "both" in output_text.lower():
                results['has_both'] += 1
        
        return InterventionResult(
            description=f"Transplant commitment L{commitment_layers}",
            correct_rate=results['correct'] / num_samples * 100,
            bug_rate=results['bug'] / num_samples * 100,
            incoherent_rate=results['incoherent'] / num_samples * 100,
            both_token_rate=results['has_both'] / num_samples * 100,
            outputs=results['outputs'],
            token_probs={}
        )

def verify_bug_behavior():
    """Verify that the bug manifests as expected with temperature=0.0"""
    logger.info("Testing Q&A format (should give WRONG answer)...")
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
    logger.info(f"  Q&A Result: {qa_result[:100]}")
    
    logger.info("Testing Simple format (should give CORRECT answer)...")
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
    logger.info(f"  Simple Result: {simple_result[:100]}")
    
    # Verify results match expectations
    if "9.11" in qa_result and "bigger" in qa_result.lower():
        logger.info("  ✓ Q&A format produces WRONG answer as expected")
    else:
        logger.warning(f"  ⚠️ Q&A format did not produce expected wrong answer!")
    
    if "9.8" in simple_result and "bigger" in simple_result.lower():
        logger.info("  ✓ Simple format produces CORRECT answer as expected")
    else:
        logger.warning(f"  ⚠️ Simple format did not produce expected correct answer!")

def run_hedging_interventions():
    """
    Main function to run all hedging-specific interventions
    """
    logger.info("\n" + "="*60)
    logger.info("STARTING HEDGING → COMMITMENT INTERVENTION EXPERIMENTS")
    logger.info("="*60)
    
    try:
        interventions = HedgingInterventions(model, tokenizer)
        logger.info("✓ Intervention handler initialized")
        
        # First, verify the bug exists with our prompts
        logger.info("\n🔍 Verifying bug behavior...")
        verify_bug_behavior()
        
        # Analyze the hedging zone
        logger.info("\n📊 Analyzing hedging zone in wrong format (Q&A)...")
        qa_hedging = interventions.analyze_hedging_zone(QA_PROMPT)
        
        logger.info("\nHedging probabilities by layer (Q&A format - WRONG):")
        for layer, probs in qa_hedging.items():
            logger.info(f"  L{layer}: Both={probs['Both']:.3f}, both={probs['both']:.3f}, "
                       f"9={probs['9']:.3f}, 8={probs['8']:.3f}")
            # Highlight Layer 25 - the critical divergence point
            if layer == 25:
                logger.info(f"  ⚠️ L25 is the CRITICAL DIVERGENCE POINT!")
        
        logger.info("\n📊 Analyzing commitment zone in correct format (Simple)...")
        simple_commitment = interventions.analyze_hedging_zone(SIMPLE_PROMPT)
        
        logger.info("\nCommitment probabilities by layer (Simple format - CORRECT):")
        for layer, probs in simple_commitment.items():
            logger.info(f"  L{layer}: Both={probs['Both']:.3f}, both={probs['both']:.3f}, "
                       f"9={probs['9']:.3f}, 8={probs['8']:.3f}")
            if layer == 25:
                logger.info(f"  ✓ L25 shows commitment to '9' in correct format!")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise
    
    # Run interventions
    all_results = []
    
    # Test 1: Suppress "Both" tokens
    result1 = interventions.intervention_1_suppress_both_tokens(
        QA_PROMPT,
        layers=[22, 23, 24, 25],
        suppression_factor=0.01
    )
    all_results.append(result1)
    print_result(result1)
    
    # Test 2: Boost "9" token at layer 25
    result2 = interventions.intervention_2_boost_commitment_token(
        QA_PROMPT,
        target_token='9',
        boost_layer=25,
        boost_factor=10.0
    )
    all_results.append(result2)
    print_result(result2)
    
    # Test 3: Redirect "Both" to "9"
    result3 = interventions.intervention_3_redirect_both_to_nine(
        QA_PROMPT,
        redirect_layers=[23, 24, 25]
    )
    all_results.append(result3)
    print_result(result3)
    
    # Test 4: Transplant commitment pattern
    result4 = interventions.intervention_4_transplant_commitment_pattern(
        QA_PROMPT,
        SIMPLE_PROMPT,
        commitment_layers=[24, 25, 26]
    )
    all_results.append(result4)
    print_result(result4)
    
    # Visualize results
    visualize_hedging_results(all_results)
    
    return all_results

def print_result(result: InterventionResult):
    """Pretty print intervention results"""
    logger.info(f"\n📈 Results for: {result.description}")
    logger.info(f"   ✓ Correct: {result.correct_rate:.1f}%")
    logger.info(f"   ✗ Bug: {result.bug_rate:.1f}%")
    logger.info(f"   ? Incoherent: {result.incoherent_rate:.1f}%")
    logger.info(f"   'Both' tokens: {result.both_token_rate:.1f}%")
    
    if result.correct_rate > 50:
        logger.info("   🎯 PROMISING RESULT! Intervention shows >50% success rate!")
    
    if result.outputs:
        logger.info(f"   Sample output: {result.outputs[0][:100]}...")

def visualize_hedging_results(results: List[InterventionResult]):
    """Create visualization of hedging intervention results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    descriptions = [r.description for r in results]
    correct_rates = [r.correct_rate for r in results]
    bug_rates = [r.bug_rate for r in results]
    incoherent_rates = [r.incoherent_rate for r in results]
    both_rates = [r.both_token_rate for r in results]
    
    x = np.arange(len(descriptions))
    
    # Plot 1: Success rates
    width = 0.2
    ax1.bar(x - 1.5*width, correct_rates, width, label='Correct', color='green', alpha=0.7)
    ax1.bar(x - 0.5*width, bug_rates, width, label='Bug', color='red', alpha=0.7)
    ax1.bar(x + 0.5*width, incoherent_rates, width, label='Incoherent', color='gray', alpha=0.7)
    ax1.bar(x + 1.5*width, both_rates, width, label="'Both' tokens", color='orange', alpha=0.7)
    
    ax1.set_xlabel('Intervention Type')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Hedging → Commitment Intervention Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d[:20] + '...' if len(d) > 20 else d for d in descriptions], 
                        rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight successful interventions
    for i, correct in enumerate(correct_rates):
        if correct > 50:
            ax1.axvspan(i-0.4, i+0.4, alpha=0.2, color='yellow')
    
    # Plot 2: Focus on "Both" token reduction
    ax2.scatter(both_rates, correct_rates, s=100, alpha=0.6)
    for i, desc in enumerate(descriptions):
        ax2.annotate(f"Int{i+1}", (both_rates[i], correct_rates[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel("'Both' Token Rate (%)")
    ax2.set_ylabel('Correct Rate (%)')
    ax2.set_title('Trade-off: Reducing Hedging vs Correctness')
    ax2.grid(True, alpha=0.3)
    
    # Add ideal zone
    ax2.axhspan(80, 100, alpha=0.1, color='green', label='Target zone')
    ax2.axvspan(0, 20, alpha=0.1, color='blue')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('hedging_intervention_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        results = run_hedging_interventions()
        
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        
        if results:
            # Find best intervention
            best_result = max(results, key=lambda r: r.correct_rate)
            logger.info(f"\n🏆 Best Intervention: {best_result.description}")
            logger.info(f"   Achieved {best_result.correct_rate:.1f}% correct rate")
            logger.info(f"   Reduced bug rate to {best_result.bug_rate:.1f}%")
            
            # Save results to JSON for analysis
            results_dict = {
                'timestamp': datetime.now().isoformat(),
                'interventions': [
                    {
                        'description': r.description,
                        'correct_rate': r.correct_rate,
                        'bug_rate': r.bug_rate,
                        'incoherent_rate': r.incoherent_rate,
                        'both_token_rate': r.both_token_rate,
                        'sample_outputs': r.outputs[:3] if r.outputs else []
                    }
                    for r in results
                ]
            }
            
            with open(f'hedging_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info("\n✓ Results saved to JSON file")
            
            if best_result.correct_rate > 80:
                logger.info("\n✅ SUCCESS: Found intervention that fixes the hedging behavior!")
            elif best_result.correct_rate > 50:
                logger.info("\n🔶 MODERATE SUCCESS: Interventions improve behavior significantly")
            else:
                logger.info("\n⚠️  Limited success: Interventions show promise but need refinement")
        else:
            logger.error("No results obtained from interventions")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise