"""Dataset paths and per-CWE schema mapping for the prompt-engineering baseline."""

from pathlib import Path

MATS_ROOT = Path("/home/paperspace/MATS")
DATASETS_DIR = MATS_ROOT / "src/experiments/02-05_cross_cwe_steering/datasets"

CWE_LIST = ["CWE-787", "CWE-119", "CWE-134", "CWE-89", "CWE-78", "CWE-79"]

C_CWES = {"CWE-787", "CWE-119", "CWE-134"}
PY_CWES = {"CWE-89", "CWE-78", "CWE-79"}


# Adversarial-variant prompt sources (n=105 each, 7 base_ids x 15 variations)
ADVERSARIAL_PATHS = {
    "CWE-787": MATS_ROOT / "src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl",
    "CWE-119": DATASETS_DIR / "cwe119/data/cwe119_expanded_20260207_024627.jsonl",
    "CWE-134": DATASETS_DIR / "cwe134/data/cwe134_expanded_20260207_024627.jsonl",
    "CWE-89":  DATASETS_DIR / "cwe89/data/cwe89_expanded_20260209_221808.jsonl",
    "CWE-78":  DATASETS_DIR / "cwe78/data/cwe78_expanded_20260209_221808.jsonl",
    "CWE-79":  DATASETS_DIR / "cwe79/data/cwe79_expanded_20260209_221808.jsonl",
}

# Neutral deployment sets (21 each, 7 prompts x 3 CWEs per language)
NEUTRAL_C_PATH  = MATS_ROOT / "src/experiments/02-05_cross_cwe_steering/neutral_eval/data/neutral_eval_prompts.jsonl"
NEUTRAL_PY_PATH = DATASETS_DIR / "neutral_eval/data/neutral_python_prompts.jsonl"


# Field names — schemas differ between C and Python adversarial prompts
def get_prompt_field(cwe: str, variant: str) -> str:
    """Return the JSONL field name holding the prompt text for the given CWE family + variant.

    variant: 'adversarial' (insecure-asking), 'secure' (secure-asking, S7), 'neutral' (deployment)
    """
    if variant == "neutral":
        return "prompt"
    if cwe in C_CWES:
        return {"adversarial": "vulnerable", "secure": "secure"}[variant]
    return {"adversarial": "insecure_prompt", "secure": "secure_prompt"}[variant]


def get_id_field(cwe: str) -> str:
    return "id" if cwe in C_CWES else "pair_id"


def needs_completion_prefix(cwe: str, variant: str) -> bool:
    """Whether to prepend 'Complete the following X function...' to the user message.

    Canonical convention:
    - C adversarial/secure: prompt is already a 'Task: Write a C function...' instruction → no prefix
    - C neutral: prompt is just a code prefix → add 'Complete the following C function...' prefix
    - Python all variants: prompt is a code prefix (function with docstring) → add Python completion prefix
    """
    if cwe in C_CWES:
        return variant == "neutral"
    return True  # Python always needs the prefix


def completion_prefix(cwe: str) -> str:
    lang = "C" if cwe in C_CWES else "Python"
    return f"Complete the following {lang} function. Only write the function body, no explanation.\n\n"
