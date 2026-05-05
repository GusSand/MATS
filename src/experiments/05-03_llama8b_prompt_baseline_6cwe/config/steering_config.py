"""Steering directions and best-α per CWE for the combined experiment.

Best-α values are LOBO-verified strict_secure_rate maxima from canonical
experiments. Direction vectors come from the same canonical mean-difference
extractions used in those experiments. Layer = 31 (last hidden layer of
Llama-3.1-8B-Instruct) for all CWEs.
"""

from pathlib import Path

MATS_ROOT = Path("/home/paperspace/MATS")

# Layer index for steering hook (matches all canonical Llama-8B experiments)
STEER_LAYER = 31

# Per-CWE direction vector paths and best alpha (LOBO-verified)
STEER_CONFIG = {
    "CWE-787": {
        "direction_path": MATS_ROOT / "src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/direction_cwe787_L31_20260206_031901.npy",
        "alpha": 3.5,
        "lobo_strict_secure_rate": 0.524,  # 52.4% per 01-12 LOBO @ α=3.5
    },
    "CWE-119": {
        "direction_path": MATS_ROOT / "src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/direction_cwe119_L31_20260206_031901.npy",
        "alpha": 1.0,
        "lobo_strict_secure_rate": 0.384,  # 38.4% per 02-05 CWE-119 LOBO @ α=1.0
    },
    "CWE-134": {
        "direction_path": MATS_ROOT / "src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/direction_cwe134_L31_20260206_031901.npy",
        "alpha": 3.0,
        "lobo_strict_secure_rate": 0.749,  # 74.9% per 02-13 CWE-134 LOBO @ α=3.0
    },
    "CWE-89": {
        "direction_path": MATS_ROOT / "src/experiments/02-10_python_cwe_steering/data/direction_cwe89_L31_20260210_015359.npy",
        "alpha": 5.0,
        "lobo_strict_secure_rate": 0.703,  # 70.3% per 02-10 CWE-89 LOBO @ α=5.0
    },
    "CWE-78": {
        "direction_path": MATS_ROOT / "src/experiments/02-10_python_cwe_steering/data/direction_cwe78_L31_20260210_015359.npy",
        "alpha": 5.0,
        "lobo_strict_secure_rate": 0.220,  # 22.0% per 02-10 CWE-78 LOBO @ α=5.0
    },
    "CWE-79": {
        "direction_path": MATS_ROOT / "src/experiments/02-10_python_cwe_steering/data/direction_cwe79_L31_20260210_015359.npy",
        "alpha": 5.0,
        "lobo_strict_secure_rate": 0.305,  # 30.5% per 02-10 CWE-79 LOBO @ α=5.0
    },
}


# Combined experiment conditions (extends config/phrasings.py)
# Each one combines a prompting condition + steering at the CWE's best α.
COMBINED_CONDITIONS = [
    {"id": "baseline_steer",     "base_condition": "baseline",     "steer": True},
    {"id": "sys_verbose_steer",  "base_condition": "sys_verbose",  "steer": True},
    {"id": "usr_verbose_steer",  "base_condition": "usr_verbose",  "steer": True},
]
