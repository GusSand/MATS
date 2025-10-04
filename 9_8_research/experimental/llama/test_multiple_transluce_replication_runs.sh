#!/bin/bash
# Test accuracy over multiple runs using the existing script

source transluce_env_312/bin/activate

echo "Testing 'bible verses' ablation accuracy (temp=0.2)"
echo "=================================================="

# Counters
unaltered_correct=0
ablated_correct=0
total_runs=20

for i in $(seq 1 $total_runs); do
    echo -ne "\rRun $i/$total_runs... "
    
    # Run and capture output
    output=$(python transluce_replication.py 2>&1)
    
    # Extract the answers (looking for the actual response lines)
    unaltered=$(echo "$output" | grep -A 20 "Unaltered (Flawed Pathway):" | grep -E "^(9\.8|9\.11)" | head -1)
    ablated=$(echo "$output" | grep -A 20 "Intervened (Concept Ablated):" | grep -E "^(9\.8|9\.11)" | head -1)
    
    # Check if correct (9.8 IS bigger than 9.11, so "9.11 is bigger" is WRONG)
    if [[ "$unaltered" == *"9.11 is bigger"* ]] || [[ "$unaltered" == *"9.11 is larger"* ]]; then
        # This is INCORRECT
        echo -n "✗"
    else
        ((unaltered_correct++))
        echo -n "✓"
    fi
    
    echo -n " | "
    
    if [[ "$ablated" == *"9.11 is bigger"* ]] || [[ "$ablated" == *"9.11 is larger"* ]]; then
        # This is INCORRECT
        echo -n "✗"
    else
        ((ablated_correct++))
        echo -n "✓"
    fi
done

echo -e "\n"
echo "Results:"
echo "--------"
echo "Unaltered: $unaltered_correct/$total_runs = $(( unaltered_correct * 100 / total_runs ))%"
echo "Ablated:   $ablated_correct/$total_runs = $(( ablated_correct * 100 / total_runs ))%"
echo "Improvement: $(( (ablated_correct - unaltered_correct) * 100 / total_runs ))%"