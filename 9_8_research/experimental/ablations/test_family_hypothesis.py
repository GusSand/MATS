#!/usr/bin/env python
"""
Test the family-dependent hypothesis: 
- Gemma: instruction tuning FIXES the bug
- Llama: instruction tuning CREATES the bug
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import json
from datetime import datetime

def test_model_comprehensive(model_name, model_path, model_type='base', num_runs=10):
    """Comprehensive test of a model across different decimal comparison scenarios"""
    
    print(f"\n{'='*60}")
    print(f"🤖 Testing {model_name} ({model_type})")
    print(f"{'='*60}")
    
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    try:
        # Load model
        print(f"Loading {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
        
        def generate(prompt, max_new_tokens=50, temperature=0.2):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
            
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = full[len(prompt):]
            return generated
        
        def get_prompt(question, model_family, model_type):
            """Get the appropriate prompt format for the model"""
            if model_family == 'llama' and model_type == 'instruct':
                # Use the format that triggers the bug in Llama
                return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif model_family == 'gemma' and model_type == 'instruct':
                # Use Gemma's chat format
                if hasattr(tokenizer, 'apply_chat_template'):
                    messages = [{"role": "user", "content": question}]
                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            else:
                # Base models use Q&A format
                return f"Q: {question}\nA:"
        
        # Determine model family
        model_family = 'llama' if 'llama' in model_path.lower() else 'gemma'
        
        # Test 1: Basic decimal comparisons (including 9.8 vs 9.11)
        print("\n📊 Test 1: Basic Decimal Comparisons")
        basic_comparisons = [
            ('9.8', '9.11'),
            ('3.9', '3.11'),
            ('2.7', '2.10'),
            ('1.8', '1.12'),
            ('5.6', '5.14'),
        ]
        
        results['tests']['basic_comparisons'] = {}
        
        for num1, num2 in basic_comparisons:
            correct_answer = num1 if float(num1) > float(num2) else num2
            question = f"Which is bigger: {num1} or {num2}?"
            prompt = get_prompt(question, model_family, model_type)
            
            correct_count = 0
            incorrect_count = 0
            examples = []
            
            for i in range(num_runs):
                response = generate(prompt, temperature=0.2)
                
                # Check if response contains correct answer
                if correct_answer in response and any(w in response.lower() for w in ["bigger", "larger", "greater", correct_answer]):
                    correct_count += 1
                    symbol = "✓"
                else:
                    wrong_answer = num2 if correct_answer == num1 else num1
                    if wrong_answer in response and any(w in response.lower() for w in ["bigger", "larger", "greater", wrong_answer]):
                        incorrect_count += 1
                        symbol = "✗"
                    else:
                        symbol = "?"
                
                if i < 3:
                    examples.append((symbol, response[:60].strip()))
            
            accuracy = correct_count / num_runs * 100
            error_rate = incorrect_count / num_runs * 100
            
            results['tests']['basic_comparisons'][f"{num1}_vs_{num2}"] = {
                'correct_answer': correct_answer,
                'accuracy': accuracy,
                'error_rate': error_rate,
                'examples': examples
            }
            
            print(f"  {num1} vs {num2}: {accuracy:.0f}% correct, {error_rate:.0f}% wrong")
        
        # Test 2: Mathematical constants
        print("\n📊 Test 2: Mathematical Constants")
        math_constants_tests = [
            "Is π (3.14) greater than 3.11?",
            "Is e (2.71) greater than 2.8?",
            "Compare 3.14 with 3.2: which is larger?",
        ]
        
        results['tests']['mathematical_constants'] = {}
        
        for test_question in math_constants_tests:
            prompt = get_prompt(test_question, model_family, model_type)
            
            # Determine expected answer
            if "π (3.14) greater than 3.11" in test_question:
                expected = "yes"  # 3.14 > 3.11
            elif "e (2.71) greater than 2.8" in test_question:
                expected = "no"   # 2.71 < 2.8
            elif "3.14 with 3.2" in test_question:
                expected = "3.2"  # 3.2 > 3.14
            
            correct_count = 0
            examples = []
            
            for i in range(num_runs):
                response = generate(prompt, temperature=0.2)
                response_lower = response.lower()
                
                # Check correctness based on expected answer
                if expected == "yes" and any(w in response_lower for w in ["yes", "true", "correct", "greater", "π is greater"]):
                    correct_count += 1
                    symbol = "✓"
                elif expected == "no" and any(w in response_lower for w in ["no", "false", "not", "less", "smaller"]):
                    correct_count += 1
                    symbol = "✓"
                elif expected == "3.2" and "3.2" in response and any(w in response_lower for w in ["3.2", "larger", "bigger"]):
                    correct_count += 1
                    symbol = "✓"
                else:
                    symbol = "✗"
                
                if i < 3:
                    examples.append((symbol, response[:60].strip()))
            
            accuracy = correct_count / num_runs * 100
            
            results['tests']['mathematical_constants'][test_question] = {
                'expected': expected,
                'accuracy': accuracy,
                'examples': examples
            }
            
            print(f"  {test_question[:30]}...: {accuracy:.0f}% correct")
        
        # Test 3: Ambiguous contexts
        print("\n📊 Test 3: Context-Dependent Comparisons")
        context_tests = [
            ('In mathematics, is 3.14 > 3.9?', 'no'),  # Clearly decimal
            ('For Python versions, is 3.14 > 3.9?', 'yes'),  # Version context
            ('As decimal numbers, is 2.7 > 2.11?', 'yes'),  # 2.7 > 2.11
            ('As version numbers, is 2.7 > 2.11?', 'no'),   # v2.11 > v2.7
        ]
        
        results['tests']['context_dependent'] = {}
        
        for test_question, expected in context_tests:
            prompt = get_prompt(test_question, model_family, model_type)
            
            correct_count = 0
            examples = []
            
            for i in range(num_runs):
                response = generate(prompt, temperature=0.2)
                response_lower = response.lower()
                
                if expected == "yes" and any(w in response_lower for w in ["yes", "true", "correct", "greater"]):
                    correct_count += 1
                    symbol = "✓"
                elif expected == "no" and any(w in response_lower for w in ["no", "false", "not", "less", "smaller"]):
                    correct_count += 1
                    symbol = "✓"
                else:
                    symbol = "✗"
                
                if i < 3:
                    examples.append((symbol, response[:60].strip()))
            
            accuracy = correct_count / num_runs * 100
            
            results['tests']['context_dependent'][test_question] = {
                'expected': expected,
                'accuracy': accuracy,
                'examples': examples
            }
            
            print(f"  {test_question[:40]}...: {accuracy:.0f}% correct")
        
        # Test 4: Mathematical operations
        print("\n📊 Test 4: Mathematical Operations")
        math_ops = [
            ('3.9 + 0.2 = ?', '4.1'),
            ('2.7 + 0.01 = ?', '2.71'),
            ('9.8 - 0.1 = ?', '9.7'),
            ('3.14 + 0.06 = ?', '3.2'),
        ]
        
        results['tests']['mathematical_operations'] = {}
        
        for operation, expected in math_ops:
            prompt = get_prompt(operation, model_family, model_type)
            
            correct_count = 0
            examples = []
            
            for i in range(num_runs):
                response = generate(prompt, temperature=0.2)
                
                if expected in response:
                    correct_count += 1
                    symbol = "✓"
                else:
                    symbol = "✗"
                
                if i < 3:
                    examples.append((symbol, response[:60].strip()))
            
            accuracy = correct_count / num_runs * 100
            
            results['tests']['mathematical_operations'][operation] = {
                'expected': expected,
                'accuracy': accuracy,
                'examples': examples
            }
            
            print(f"  {operation}: {accuracy:.0f}% correct (expected {expected})")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"  Error testing {model_name}: {e}")
        results['error'] = str(e)
        return results

def generate_markdown_report(results, output_file):
    """Generate a detailed markdown report for a model"""
    
    with open(output_file, 'w') as f:
        f.write(f"# Test Results: {results['model_name']}\n\n")
        f.write(f"**Model Path**: `{results['model_path']}`\n")
        f.write(f"**Model Type**: {results['model_type']}\n")
        f.write(f"**Test Date**: {results['timestamp']}\n\n")
        
        if 'error' in results:
            f.write(f"## Error\n\n{results['error']}\n")
            return
        
        # Basic comparisons
        f.write("## 1. Basic Decimal Comparisons\n\n")
        f.write("| Comparison | Correct Answer | Accuracy | Error Rate | Examples |\n")
        f.write("|------------|----------------|----------|------------|----------|\n")
        
        for test, data in results['tests'].get('basic_comparisons', {}).items():
            nums = test.replace('_vs_', ' vs ')
            examples = ' / '.join([f"{sym} {ex[:20]}..." for sym, ex in data['examples'][:2]])
            f.write(f"| {nums} | {data['correct_answer']} | {data['accuracy']:.0f}% | {data['error_rate']:.0f}% | {examples} |\n")
        
        # Mathematical constants
        f.write("\n## 2. Mathematical Constants\n\n")
        f.write("| Test | Expected | Accuracy | Examples |\n")
        f.write("|------|----------|----------|----------|\n")
        
        for test, data in results['tests'].get('mathematical_constants', {}).items():
            examples = ' / '.join([f"{sym} {ex[:20]}..." for sym, ex in data['examples'][:2]])
            f.write(f"| {test[:40]}... | {data['expected']} | {data['accuracy']:.0f}% | {examples} |\n")
        
        # Context-dependent
        f.write("\n## 3. Context-Dependent Comparisons\n\n")
        f.write("| Context | Expected | Accuracy | Examples |\n")
        f.write("|---------|----------|----------|----------|\n")
        
        for test, data in results['tests'].get('context_dependent', {}).items():
            examples = ' / '.join([f"{sym} {ex[:20]}..." for sym, ex in data['examples'][:2]])
            f.write(f"| {test[:50]}... | {data['expected']} | {data['accuracy']:.0f}% | {examples} |\n")
        
        # Mathematical operations
        f.write("\n## 4. Mathematical Operations\n\n")
        f.write("| Operation | Expected | Accuracy | Examples |\n")
        f.write("|-----------|----------|----------|----------|\n")
        
        for test, data in results['tests'].get('mathematical_operations', {}).items():
            examples = ' / '.join([f"{sym} {ex[:20]}..." for sym, ex in data['examples'][:2]])
            f.write(f"| {test} | {data['expected']} | {data['accuracy']:.0f}% | {examples} |\n")
        
        # Summary
        f.write("\n## Summary\n\n")
        
        # Calculate overall accuracy
        total_tests = 0
        total_correct = 0
        
        for category in ['basic_comparisons', 'mathematical_constants', 'context_dependent', 'mathematical_operations']:
            if category in results['tests']:
                for test, data in results['tests'][category].items():
                    total_tests += 10  # num_runs
                    total_correct += data['accuracy'] / 10  # Convert percentage back to count
        
        overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
        
        f.write(f"**Overall Accuracy**: {overall_accuracy:.1f}%\n\n")
        
        # Key findings
        f.write("### Key Findings\n\n")
        
        # Check basic decimal comparison performance
        basic_results = results['tests'].get('basic_comparisons', {})
        if basic_results:
            high_error_tests = [test for test, data in basic_results.items() if data['error_rate'] > 50]
            if high_error_tests:
                f.write(f"- **Decimal Bug Present**: High error rates on {', '.join(high_error_tests)}\n")
            else:
                f.write("- **No Clear Decimal Bug**: Low error rates on basic comparisons\n")
        
        # Check context sensitivity
        context_results = results['tests'].get('context_dependent', {})
        if context_results:
            f.write(f"- **Context Sensitivity**: Model shows {'good' if sum(d['accuracy'] for d in context_results.values())/len(context_results) > 50 else 'poor'} context understanding\n")
        
        # Check mathematical operations
        math_ops_results = results['tests'].get('mathematical_operations', {})
        if math_ops_results:
            f.write(f"- **Mathematical Operations**: {sum(d['accuracy'] for d in math_ops_results.values())/len(math_ops_results):.0f}% accuracy on arithmetic\n")

def main():
    print("🔬 TESTING FAMILY-DEPENDENT HYPOTHESIS")
    print("="*60)
    
    # Model pairs to test
    model_pairs = [
        ('Gemma-2-2B Base', 'google/gemma-2-2b', 'base'),
        ('Gemma-2-2B-IT', 'google/gemma-2-2b-it', 'instruct'),
        ('Llama-3.1-8B Base', 'meta-llama/Llama-3.1-8B', 'base'),
        ('Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 'instruct'),
    ]
    
    all_results = []
    
    # Create output directory
    os.makedirs('test_results', exist_ok=True)
    
    for model_name, model_path, model_type in model_pairs:
        results = test_model_comprehensive(model_name, model_path, model_type)
        all_results.append(results)
        
        # Generate individual report
        output_file = f"test_results/{model_name.replace(' ', '_').replace('.', '_').lower()}_results.md"
        generate_markdown_report(results, output_file)
        print(f"\n📄 Report saved to: {output_file}")
    
    # Generate comparison report
    print("\n\n" + "="*60)
    print("📊 GENERATING COMPARISON REPORT")
    print("="*60)
    
    with open('test_results/comparison_summary.md', 'w') as f:
        f.write("# Family-Dependent Hypothesis Test Results\n\n")
        f.write("## Hypothesis\n\n")
        f.write("- **Gemma**: Instruction tuning FIXES the decimal bug\n")
        f.write("- **Llama**: Instruction tuning CREATES the decimal bug\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Model | Type | 9.8 vs 9.11 | Overall Accuracy | Hypothesis Support |\n")
        f.write("|-------|------|-------------|------------------|--------------------|\n")
        
        for result in all_results:
            if 'error' not in result:
                # Get 9.8 vs 9.11 result
                basic = result['tests'].get('basic_comparisons', {})
                nine_eight_test = basic.get('9.8_vs_9.11', {})
                nine_eight_acc = nine_eight_test.get('accuracy', 'N/A')
                
                # Calculate overall accuracy
                total = sum(
                    data.get('accuracy', 0) 
                    for category in result['tests'].values() 
                    for data in category.values()
                )
                count = sum(
                    len(category) 
                    for category in result['tests'].values()
                )
                overall = total / count if count > 0 else 0
                
                # Determine hypothesis support
                if 'gemma' in result['model_path'].lower():
                    if result['model_type'] == 'instruct' and overall > 80:
                        support = "✅ IT fixes bug"
                    elif result['model_type'] == 'base' and overall < 80:
                        support = "✅ Base has bug"
                    else:
                        support = "❓ Mixed"
                else:  # Llama
                    if result['model_type'] == 'instruct' and overall < 50:
                        support = "✅ IT creates bug"
                    elif result['model_type'] == 'base' and overall > 50:
                        support = "✅ Base is better"
                    else:
                        support = "❓ Mixed"
                
                f.write(f"| {result['model_name']} | {result['model_type']} | {nine_eight_acc}% | {overall:.1f}% | {support} |\n")
    
    print("📄 Comparison report saved to: test_results/comparison_summary.md")

if __name__ == "__main__":
    main()