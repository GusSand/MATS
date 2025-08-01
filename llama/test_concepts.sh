#!/bin/bash
# Test different concepts to see which ones actually help

source transluce_env_312/bin/activate

echo "Testing different concept ablations..."
echo "=================================="

# Test each concept
for concept in "bible verses" "September 11" "dates" "version numbers" "numerical comparisons"; do
    echo -e "\n🔍 Testing: $concept"
    echo "-------------------"
    
    # Update the concept in the script
    sed -i "s/CONCEPT_TO_ABLATE = .*/CONCEPT_TO_ABLATE = \"$concept\"/" transluce_replication.py
    
    # Run and capture just the results
    python transluce_replication.py 2>&1 | grep -A 10 "FINAL RESULTS" | grep -E "(Original|Ablated|Answer)" | tail -4
    
    echo ""
done

# Reset to original
sed -i 's/CONCEPT_TO_ABLATE = .*/CONCEPT_TO_ABLATE = "bible verses"/' transluce_replication.py