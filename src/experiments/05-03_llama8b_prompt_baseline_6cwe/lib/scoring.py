"""Per-CWE scoring shim. Wires up:

- C CWEs (787, 119, 134): regex secure_pattern / insecure_pattern from each
  adversarial prompt's `detection` field. For neutral C prompts (no embedded
  detection), patterns are OR-combined across all adversarial pairs of the
  same CWE at module load.
- Python CWEs (89, 78, 79): module-level scorer functions from
  src/experiments/02-05_cross_cwe_steering/datasets/cwe{89,78,79}/scoring.py
  (same scorers used by the SVEN comparison committed to research_journal.md).
"""

import json
import re
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
sys.path.insert(0, str(CONFIG_DIR))
import datasets as ds  # noqa: E402

# Wire up Python scorers via the SVEN-comparison path layout
sys.path.insert(0, str(ds.DATASETS_DIR))
from cwe89.scoring import score_cwe89  # noqa: E402
from cwe78.scoring import score_cwe78  # noqa: E402
from cwe79.scoring import score_cwe79  # noqa: E402


# ─── C scoring ──────────────────────────────────────────────────────────────

def _score_c_with_detection(code: str, detection: dict) -> str:
    secure_pat = detection.get("secure_pattern", "")
    insecure_pat = detection.get("insecure_pattern", "")
    has_secure = bool(re.search(secure_pat, code)) if secure_pat else False
    has_insecure = bool(re.search(insecure_pat, code)) if insecure_pat else False
    if has_secure and not has_insecure:
        return "secure"
    if has_insecure:
        return "insecure"
    return "other"


def _build_cwe_level_c_patterns():
    """OR-combine secure_pattern / insecure_pattern across all adversarial pairs of each C CWE."""
    out = {}
    for cwe in ds.C_CWES:
        rows = [json.loads(l) for l in open(ds.ADVERSARIAL_PATHS[cwe])]
        secure_pats = sorted({r["detection"].get("secure_pattern", "") for r in rows} - {""})
        insecure_pats = sorted({r["detection"].get("insecure_pattern", "") for r in rows} - {""})
        out[cwe] = {
            "secure_pattern": "(?:" + "|".join(secure_pats) + ")" if secure_pats else "",
            "insecure_pattern": "(?:" + "|".join(insecure_pats) + ")" if insecure_pats else "",
        }
    return out


_CWE_LEVEL_C_PATTERNS = _build_cwe_level_c_patterns()


# ─── Public entry points ────────────────────────────────────────────────────

PYTHON_SCORERS = {
    "CWE-89": score_cwe89,
    "CWE-78": score_cwe78,
    "CWE-79": score_cwe79,
}


def score_completion(code: str, cwe: str, prompt_row: dict, variant: str) -> str:
    """Return one of: 'secure', 'insecure', 'other'.

    For C adversarial/secure variants, uses per-prompt detection patterns
    embedded in prompt_row. For C neutral, uses CWE-level OR-combined
    patterns. For Python, calls the dedicated per-CWE scorer.

    Python scorers return objects with .label or strings depending on module;
    we normalize to {'secure','insecure','other'}.
    """
    if cwe in ds.C_CWES:
        if variant == "neutral":
            return _score_c_with_detection(code, _CWE_LEVEL_C_PATTERNS[cwe])
        return _score_c_with_detection(code, prompt_row.get("detection", {}))

    # Python
    result = PYTHON_SCORERS[cwe](code)
    label = getattr(result, "label", result)
    if isinstance(label, str) and label in {"secure", "insecure", "other"}:
        return label
    raise ValueError(f"Unexpected Python scorer return for {cwe}: {result!r}")
