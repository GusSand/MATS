# prepare_locally.py
"""
Prepare everything locally to minimize GPU rental time
"""

import os
import json
import numpy as np

def create_experiment_config():
    """
    Create all configs so we don't waste GPU time on setup
    """
    
    config = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'prompts': {
            'buggy': "<|start_header_id|>user<|end_header_id|>\nWhich is bigger: 9.9 or 9.11?\n<|start_header_id|>assistant<|end_header_id|>",
            'clean': "Q: Which is bigger: 9.9 or 9.11?\nA:",
        },
        'positive_controls': {
            'counting': {
                'prompt': "How many times does the letter 'r' appear in 'strawberry'?",
                'correct_answer': "3",
                'fix_layers': [5, 6, 7]
            },
            'arithmetic': {
                'prompt': "What is 17 × 13?",
                'correct_answer': "221",
                'fix_layers': [15, 16, 17]
            },
            'ioi': {
                'prompt': "Alice and Bob went to the store. Bob gave a gift to",
                'correct_answer': "Alice",
                'fix_heads': [(9, 9), (10, 0)]
            },
            'river': {
                'prompt': "You are at the sea. To reach the mountain where the river starts, go upstream or downstream?",
                'correct_answer': "upstream",
                'fix_layers': [12, 13, 14]
            }
        },
        'ablation_values': list(np.linspace(0, -5, 200)),  # 200 points
        'neurons_to_ablate': {
            7: [1978],
            13: [10352], 
            14: [13315, 2451, 12639],
            15: [3136, 5076, 421]
        }
    }
    
    with open('experiment_config.json', 'w') as f:
        json.dump(config, f)
    
    print("Config saved. Upload this to the GPU server.")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)

if __name__ == "__main__":
    create_experiment_config()