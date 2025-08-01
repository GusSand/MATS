#!/bin/bash
# Script to run the transluce replication with proper environment

# Activate virtual environment
source transluce_env/bin/activate

# Set PYTHONPATH to include observatory libraries
export PYTHONPATH="${PYTHONPATH}:/home/paperspace/dev/MATS9/observatory/lib/neurondb"
export PYTHONPATH="${PYTHONPATH}:/home/paperspace/dev/MATS9/observatory/lib/util"
export PYTHONPATH="${PYTHONPATH}:/home/paperspace/dev/MATS9/observatory/lib/investigator"
export PYTHONPATH="${PYTHONPATH}:/home/paperspace/dev/MATS9/observatory/lib/explanations"
export PYTHONPATH="${PYTHONPATH}:/home/paperspace/dev/MATS9/observatory/lib/activations"

# Run the original script
python transluce_replication.py