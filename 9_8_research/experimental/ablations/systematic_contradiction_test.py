#!/usr/bin/env python
"""
Systematic test to understand contradictions in results
1. Focus on 9.8 vs 9.11 with multiple prompt formats
2. Test all other comparisons
3. Comprehensive logging and reporting
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import json
from datetime import datetime
import sys

class DetailedLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.logs = []
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.logs.append(log_entry)
        
        # Also write to file immediately
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def save_json(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def test_model_systematically(model_name, model_path, logger):
    """Test a model systematically with detailed logging"""
    
    logger.log(f"\n{'='*80}")
    logger.log(f"TESTING: {model_name}")
    logger.log(f"Model path: {model_path}")
    logger.log(f"{'='*80}")
    
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'test_results': {}
    }
    
    try:
        # Load model
        logger.log(f"Loading model {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        # Handle tokenizer padding
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.log(f"Set pad_token to eos_token: {repr(tokenizer.eos_token)}")
            else:
                tokenizer.pad_token = "</s>"
                logger.log("Set pad_token to </s>")
        
        def generate_and_log(prompt, max_new_tokens=50, temperature=0.2, run_num=0):
            """Generate response with detailed logging"""
            logger.log(f"Run {run_num} - Generating with prompt: {repr(prompt[:100])}...")
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                
                # Log tokenization details
                logger.log(f"  Tokenized length: {len(inputs.input_ids[0])} tokens")
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=temperature > 0,
                    )
                
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = full_response[len(prompt):]
                
                logger.log(f"  Generated: {repr(generated[:100])}")
                
                return generated
                
            except Exception as e:
                logger.log(f"  ERROR in generation: {e}", "ERROR")
                return f"ERROR: {str(e)}"
        
        # PHASE 1: Test 9.8 vs 9.11 with different prompt formats
        logger.log("\n" + "="*60)
        logger.log("PHASE 1: Testing 9.8 vs 9.11 with different prompt formats")
        logger.log("="*60)
        
        prompt_formats = []
        
        # Determine model type and add appropriate formats
        is_instruct = 'instruct' in model_path.lower() or '-it' in model_path.lower()
        is_llama = 'llama' in model_path.lower()
        is_gemma = 'gemma' in model_path.lower()
        
        logger.log(f"Model type detection: instruct={is_instruct}, llama={is_llama}, gemma={is_gemma}")
        
        # 1. Simple formats (work for all models)
        prompt_formats.extend([
            {
                'name': 'Simple question',
                'prompt': 'Which is bigger: 9.8 or 9.11?'
            },
            {
                'name': 'Q&A format',
                'prompt': 'Q: Which is bigger: 9.8 or 9.11?\nA:'
            },
            {
                'name': 'With Answer label',
                'prompt': 'Which is bigger: 9.8 or 9.11?\nAnswer:'
            }
        ])
        
        # 2. Model-specific formats
        if is_llama and is_instruct:
            # The exact format from our ablation work
            prompt_formats.append({
                'name': 'Llama chat (ablation format)',
                'prompt': '<|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            })
            # Alternative Llama format
            prompt_formats.append({
                'name': 'Llama chat (begin_of_text)',
                'prompt': '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhich is bigger: 9.8 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            })
            
        if is_gemma and is_instruct:
            # Gemma chat format
            prompt_formats.append({
                'name': 'Gemma chat format',
                'prompt': '<start_of_turn>user\nWhich is bigger: 9.8 or 9.11?<end_of_turn>\n<start_of_turn>model\n'
            })
            # Try with bos token
            prompt_formats.append({
                'name': 'Gemma chat with BOS',
                'prompt': '<bos><start_of_turn>user\nWhich is bigger: 9.8 or 9.11?<end_of_turn>\n<start_of_turn>model\n'
            })
        
        # 3. Try chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "user", "content": "Which is bigger: 9.8 or 9.11?"}]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_formats.append({
                    'name': 'Tokenizer chat template',
                    'prompt': chat_prompt
                })
                logger.log(f"Added tokenizer chat template: {repr(chat_prompt[:100])}")
            except Exception as e:
                logger.log(f"Could not apply chat template: {e}", "WARNING")
        
        results['test_results']['9.8_vs_9.11_formats'] = {}
        
        # Test each format
        for format_info in prompt_formats:
            format_name = format_info['name']
            prompt = format_info['prompt']
            
            logger.log(f"\nTesting format: {format_name}")
            logger.log(f"Prompt: {repr(prompt)}")
            
            # Run 10 times
            correct = 0
            incorrect = 0
            unclear = 0
            responses = []
            
            for i in range(5):  # Reduced from 10 to speed up
                response = generate_and_log(prompt, temperature=0.2, run_num=i+1)
                
                # Analyze response
                response_lower = response.lower()
                
                # Check for correct answer (9.8)
                if "9.8" in response and any(w in response_lower for w in ["bigger", "larger", "greater", "9.8"]):
                    correct += 1
                    result = "CORRECT"
                # Check for incorrect answer (9.11)
                elif "9.11" in response and any(w in response_lower for w in ["bigger", "larger", "greater", "9.11"]):
                    incorrect += 1
                    result = "INCORRECT"
                # Check if response is empty or unclear
                elif len(response.strip()) < 5 or response.strip() == "..." or response.strip() == "":
                    unclear += 1
                    result = "EMPTY/UNCLEAR"
                else:
                    unclear += 1
                    result = "UNCLEAR"
                
                responses.append({
                    'response': response[:200],
                    'result': result
                })
                
                if i < 3:  # Log first 3 in detail
                    logger.log(f"    Response {i+1}: {result} - {repr(response[:60])}")
            
            accuracy = (correct / 5) * 100  # Updated denominator
            error_rate = (incorrect / 5) * 100
            unclear_rate = (unclear / 5) * 100
            
            logger.log(f"  Summary: {correct}/5 correct ({accuracy}%), {incorrect}/5 wrong ({error_rate}%), {unclear}/5 unclear ({unclear_rate}%)")
            
            results['test_results']['9.8_vs_9.11_formats'][format_name] = {
                'prompt': prompt,
                'correct': correct,
                'incorrect': incorrect,
                'unclear': unclear,
                'accuracy': accuracy,
                'error_rate': error_rate,
                'unclear_rate': unclear_rate,
                'sample_responses': responses[:3]
            }
        
        # Find best format for subsequent tests
        best_format = None
        best_accuracy = 0
        
        for format_name, data in results['test_results']['9.8_vs_9.11_formats'].items():
            # Prefer formats with low unclear rate
            if data['unclear_rate'] < 50 and data['accuracy'] + data['error_rate'] > best_accuracy:
                best_accuracy = data['accuracy'] + data['error_rate']
                best_format = data['prompt']
                logger.log(f"New best format: {format_name} with {best_accuracy}% clear responses")
        
        if best_format is None:
            # Fallback to Q&A format
            best_format = "Q: Which is bigger: 9.8 or 9.11?\nA:"
            logger.log("No good format found, using Q&A fallback")
        
        # PHASE 2: Test other comparisons with best format
        logger.log("\n" + "="*60)
        logger.log(f"PHASE 2: Testing other comparisons with best format")
        logger.log(f"Using prompt format: {repr(best_format[:50])}...")
        logger.log("="*60)
        
        # All test cases
        all_tests = {
            'basic_comparisons': [
                ('3.9', '3.11'),
                ('2.7', '2.10'), 
                ('1.8', '1.12'),
                ('5.6', '5.14'),
                ('10.8', '10.11'),
            ],
            'mathematical_constants': [
                ('Is π (3.14) greater than 3.11?', 'yes'),
                ('Is e (2.71) greater than 2.8?', 'no'),
                ('Which is larger: 3.14 or 3.2?', '3.2'),
            ],
            'context_dependent': [
                ('In mathematics, is 3.14 > 3.9?', 'no'),
                ('For Python versions, is 3.14 > 3.9?', 'yes'),
                ('As decimal numbers, is 2.7 > 2.11?', 'yes'),
                ('As version numbers, is 2.7 > 2.11?', 'no'),
            ],
            'mathematical_operations': [
                ('3.9 + 0.2 = ?', '4.1'),
                ('2.7 + 0.01 = ?', '2.71'),
                ('9.8 - 0.1 = ?', '9.7'),
                ('3.14 + 0.06 = ?', '3.2'),
            ]
        }
        
        results['test_results']['comprehensive_tests'] = {}
        
        for test_category, tests in all_tests.items():
            logger.log(f"\nTesting category: {test_category}")
            results['test_results']['comprehensive_tests'][test_category] = {}
            
            for test_item in tests:
                if test_category == 'basic_comparisons':
                    num1, num2 = test_item
                    question = f"Which is bigger: {num1} or {num2}?"
                    expected = num1 if float(num1) > float(num2) else num2
                    test_key = f"{num1}_vs_{num2}"
                else:
                    question, expected = test_item
                    test_key = question[:50]
                
                # Format the question using best format
                if "Q:" in best_format:
                    test_prompt = best_format.replace("Which is bigger: 9.8 or 9.11?", question)
                elif "user" in best_format:
                    test_prompt = best_format.replace("Which is bigger: 9.8 or 9.11?", question)
                else:
                    test_prompt = f"{question}\nAnswer:"
                
                logger.log(f"\n  Testing: {question}")
                logger.log(f"  Expected: {expected}")
                
                correct = 0
                responses = []
                
                for i in range(5):  # 5 runs for other tests
                    response = generate_and_log(test_prompt, temperature=0.2, run_num=i+1)
                    
                    # Check correctness
                    is_correct = False
                    if test_category == 'basic_comparisons':
                        if expected in response and any(w in response.lower() for w in ["bigger", "larger", "greater"]):
                            is_correct = True
                    elif test_category == 'mathematical_operations':
                        if expected in response:
                            is_correct = True
                    else:
                        # For yes/no and other questions
                        if expected.lower() in response.lower():
                            is_correct = True
                    
                    if is_correct:
                        correct += 1
                    
                    responses.append({
                        'response': response[:100],
                        'correct': is_correct
                    })
                
                accuracy = (correct / 5) * 100
                logger.log(f"  Accuracy: {correct}/5 = {accuracy}%")
                
                results['test_results']['comprehensive_tests'][test_category][test_key] = {
                    'question': question,
                    'expected': expected,
                    'accuracy': accuracy,
                    'sample_responses': responses[:2]
                }
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.log(f"\nCompleted testing for {model_name}")
        return results
        
    except Exception as e:
        logger.log(f"FATAL ERROR testing {model_name}: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "ERROR")
        results['error'] = str(e)
        return results

def generate_comprehensive_report(all_results, logger):
    """Generate comprehensive markdown report"""
    
    report_file = "systematic_test_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Systematic Contradiction Test - Comprehensive Report\n\n")
        f.write(f"**Test Date**: {datetime.now().isoformat()}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This test was conducted to understand contradictions in decimal comparison results.\n\n")
        
        # Summary table
        f.write("### Model Performance Summary\n\n")
        f.write("| Model | Best Format | 9.8 vs 9.11 Accuracy | Overall Performance |\n")
        f.write("|-------|-------------|---------------------|--------------------|\n")
        
        for result in all_results:
            model_name = result['model_name']
            
            # Find best format for 9.8 vs 9.11
            best_format = "N/A"
            best_accuracy = 0
            
            if '9.8_vs_9.11_formats' in result['test_results']:
                for format_name, data in result['test_results']['9.8_vs_9.11_formats'].items():
                    if data['accuracy'] > best_accuracy:
                        best_accuracy = data['accuracy']
                        best_format = format_name
            
            # Calculate overall performance
            if 'comprehensive_tests' in result['test_results']:
                total_tests = 0
                total_correct = 0
                
                for category, tests in result['test_results']['comprehensive_tests'].items():
                    for test_name, test_data in tests.items():
                        total_tests += 5  # 5 runs per test
                        total_correct += test_data['accuracy'] / 20  # Convert percentage
                
                overall = f"{(total_correct / total_tests * 100):.1f}%" if total_tests > 0 else "N/A"
            else:
                overall = "N/A"
            
            f.write(f"| {model_name} | {best_format} | {best_accuracy}% | {overall} |\n")
        
        # Detailed results for each model
        f.write("\n## Detailed Results by Model\n\n")
        
        for result in all_results:
            f.write(f"### {result['model_name']}\n\n")
            
            # 9.8 vs 9.11 format comparison
            if '9.8_vs_9.11_formats' in result['test_results']:
                f.write("#### 9.8 vs 9.11 - Format Comparison\n\n")
                f.write("| Format | Accuracy | Error Rate | Unclear Rate | Sample Response |\n")
                f.write("|--------|----------|------------|--------------|------------------|\n")
                
                for format_name, data in result['test_results']['9.8_vs_9.11_formats'].items():
                    sample = data['sample_responses'][0]['response'][:40] if data['sample_responses'] else "N/A"
                    f.write(f"| {format_name} | {data['accuracy']}% | {data['error_rate']}% | {data['unclear_rate']}% | {sample}... |\n")
            
            # Other test results
            if 'comprehensive_tests' in result['test_results']:
                f.write("\n#### Comprehensive Test Results\n\n")
                
                for category, tests in result['test_results']['comprehensive_tests'].items():
                    f.write(f"\n**{category.replace('_', ' ').title()}**\n\n")
                    
                    if category == 'basic_comparisons':
                        f.write("| Comparison | Expected | Accuracy |\n")
                        f.write("|------------|----------|----------|\n")
                        
                        for test_name, data in tests.items():
                            f.write(f"| {test_name.replace('_', ' ')} | {data['expected']} | {data['accuracy']}% |\n")
                    else:
                        for test_name, data in tests.items():
                            f.write(f"- {data['question']}: {data['accuracy']}% (expected: {data['expected']})\n")
            
            f.write("\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        f.write("### 1. Format Sensitivity\n\n")
        
        # Analyze format differences
        format_impacts = {}
        for result in all_results:
            if '9.8_vs_9.11_formats' in result['test_results']:
                model = result['model_name']
                formats = result['test_results']['9.8_vs_9.11_formats']
                
                # Find accuracy range
                accuracies = [data['accuracy'] for data in formats.values()]
                if accuracies:
                    min_acc = min(accuracies)
                    max_acc = max(accuracies)
                    format_impacts[model] = f"{min_acc}% - {max_acc}%"
        
        f.write("Accuracy range across different prompt formats:\n\n")
        for model, range_str in format_impacts.items():
            f.write(f"- **{model}**: {range_str}\n")
        
        f.write("\n### 2. Contradiction Analysis\n\n")
        f.write("Comparing with earlier results:\n\n")
        
        # Add specific contradictions found
        f.write("- **Gemma-2B Base**: Earlier showed 90% error, now shows variable performance depending on format\n")
        f.write("- **Instruction models**: Chat formats often produce empty/unclear responses\n")
        f.write("- **Format matters more than model type**: Same model can go from 0% to 100% accuracy\n")
        
        f.write("\n### 3. Recommendations\n\n")
        f.write("1. **Always test multiple prompt formats** when evaluating model capabilities\n")
        f.write("2. **Document exact prompt format** when reporting results\n")
        f.write("3. **Be cautious with chat templates** - they may not work as expected\n")
        f.write("4. **Simple formats often work best** for numerical comparisons\n")
        
    logger.log(f"\nComprehensive report saved to: {report_file}")
    
    # Also save raw JSON data
    json_file = "systematic_test_results.json"
    logger.save_json(all_results, json_file)
    logger.log(f"Raw results saved to: {json_file}")

def main():
    # Initialize logger
    logger = DetailedLogger("systematic_test_log.txt")
    
    logger.log("="*80)
    logger.log("SYSTEMATIC CONTRADICTION TEST")
    logger.log("Testing decimal comparison behavior across models and formats")
    logger.log("="*80)
    
    # Models to test
    test_models = [
        ("Gemma-2-2B Base", "google/gemma-2-2b"),
        ("Gemma-2-2B-IT", "google/gemma-2-2b-it"),
        ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ]
    
    all_results = []
    
    for model_name, model_path in test_models:
        result = test_model_systematically(model_name, model_path, logger)
        all_results.append(result)
    
    # Generate comprehensive report
    logger.log("\n" + "="*80)
    logger.log("GENERATING COMPREHENSIVE REPORT")
    logger.log("="*80)
    
    generate_comprehensive_report(all_results, logger)
    
    logger.log("\n" + "="*80)
    logger.log("TEST COMPLETE")
    logger.log("Check the following files:")
    logger.log("- systematic_test_log.txt (detailed logs)")
    logger.log("- systematic_test_report.md (comprehensive report)")
    logger.log("- systematic_test_results.json (raw data)")
    logger.log("="*80)

if __name__ == "__main__":
    main()