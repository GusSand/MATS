"""
Experiment: CodeQL Scoring Prototype
Date: 2026-01-14

Goal: Test if CodeQL can be used to score LLM-generated C code for CWE-787
(buffer overflow) vulnerabilities, replacing regex-based scoring.

Methodology:
1. Sample outputs from LOBO experiment (secure/insecure/other by regex)
2. Wrap each snippet in compilable C file
3. Create CodeQL database
4. Run CWE-787 queries
5. Compare CodeQL labels to regex labels
"""

from pathlib import Path

# Paths
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
QUERIES_DIR = EXPERIMENT_DIR / "queries"
WRAPPED_CODE_DIR = DATA_DIR / "wrapped_code"

# Source data
LOBO_RESULTS_DIR = Path(__file__).parent.parent / "01-12_llama8b_cwe787_lobo_steering" / "data"
LOBO_FOLD_RESULTS = LOBO_RESULTS_DIR / "fold_results"

# Sampling config
SAMPLES_PER_CATEGORY = 10  # 10 secure, 10 insecure, 10 other
ALPHA_TO_SAMPLE = "3.5"  # Sample from highest steering strength

# CWE queries to run
CWE_QUERIES = [
    "cpp/overflow-destination",  # Buffer overflow in destination
    "cpp/very-likely-overrunning-write",  # Overrunning write
    "cpp/unbounded-write",  # Unbounded write
]

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
QUERIES_DIR.mkdir(exist_ok=True)
WRAPPED_CODE_DIR.mkdir(exist_ok=True)
