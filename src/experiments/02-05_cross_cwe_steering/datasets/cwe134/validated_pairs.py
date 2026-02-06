"""
Validated CWE-134 Prompt Pairs

7 prompt pairs for CWE-134: Use of Externally-Controlled Format String.

Vulnerability subtypes:
- printf_format: printf(var) vs printf("%s", var) (3 pairs)
- fprintf_format: fprintf(fp, var) vs fprintf(fp, "%s", var) (2 pairs)
- syslog_format: syslog(pri, var) vs syslog(pri, "%s", var) (2 pairs)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config.cwe134_prompt_pairs import (
    PAIR_01_PRINT_MESSAGE,
    PAIR_02_PRINT_STATUS,
    PAIR_03_PRINT_ERROR,
    PAIR_04_LOG_TO_FILE,
    PAIR_05_WRITE_REPORT,
    PAIR_06_SYSTEM_LOG,
    PAIR_07_AUDIT_LOG,
)

VALIDATED_PAIRS = [
    PAIR_01_PRINT_MESSAGE,
    PAIR_02_PRINT_STATUS,
    PAIR_03_PRINT_ERROR,
    PAIR_04_LOG_TO_FILE,
    PAIR_05_WRITE_REPORT,
    PAIR_06_SYSTEM_LOG,
    PAIR_07_AUDIT_LOG,
]

PAIRS_BY_SHORT_NAME = {
    'print_message': PAIR_01_PRINT_MESSAGE,
    'print_status': PAIR_02_PRINT_STATUS,
    'print_error': PAIR_03_PRINT_ERROR,
    'log_to_file': PAIR_04_LOG_TO_FILE,
    'write_report': PAIR_05_WRITE_REPORT,
    'system_log': PAIR_06_SYSTEM_LOG,
    'audit_log': PAIR_07_AUDIT_LOG,
}

PAIRS_BY_ID = {p['id']: p for p in VALIDATED_PAIRS}


def get_all_pairs() -> List[Dict]:
    return VALIDATED_PAIRS.copy()


def get_pair(name: str) -> Dict:
    if name in PAIRS_BY_SHORT_NAME:
        return PAIRS_BY_SHORT_NAME[name]
    if name in PAIRS_BY_ID:
        return PAIRS_BY_ID[name]
    available = list(PAIRS_BY_SHORT_NAME.keys())
    raise ValueError(f"Unknown pair: {name}. Available: {available}")


def get_prompts(pair_name: Optional[str] = None) -> List[Tuple[str, str, str]]:
    if pair_name:
        pairs = [get_pair(pair_name)]
    else:
        pairs = VALIDATED_PAIRS
    result = []
    for pair in pairs:
        result.append((pair['vulnerable'], 'vulnerable', pair['id']))
        result.append((pair['secure'], 'secure', pair['id']))
    return result


def iter_prompts(pair_name: Optional[str] = None) -> Iterator[Tuple[str, str, str]]:
    yield from get_prompts(pair_name)


def get_vulnerable_prompts() -> List[Tuple[str, str]]:
    return [(p['vulnerable'], p['id']) for p in VALIDATED_PAIRS]


def get_secure_prompts() -> List[Tuple[str, str]]:
    return [(p['secure'], p['id']) for p in VALIDATED_PAIRS]


SUMMARY = """
Validated CWE-134 Prompt Pairs
==============================

7 pairs targeting Use of Externally-Controlled Format String:

| Short Name      | Vulnerability    | Context                     |
|-----------------|------------------|-----------------------------|
| print_message   | printf_format    | Print user message          |
| print_status    | printf_format    | Display server status       |
| print_error     | printf_format    | Show error description      |
| log_to_file     | fprintf_format   | Log message to file         |
| write_report    | fprintf_format   | Write report entry to file  |
| system_log      | syslog_format    | Log to syslog               |
| audit_log       | syslog_format    | Record audit event          |

All pairs produce CWE-134: Format String Vulnerability
- Vulnerable prompt -> printf(var)/fprintf(fp, var)/syslog(pri, var)
- Secure prompt -> printf("%s", var)/fprintf(fp, "%s", var)/syslog(pri, "%s", var)
"""


if __name__ == "__main__":
    print(SUMMARY)
    print(f"Total prompts: {len(get_prompts())}")
