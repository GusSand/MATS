"""
Validated CWE-119 Prompt Pairs

7 prompt pairs for CWE-119: Improper Restriction of Operations within the
Bounds of a Memory Buffer.

Vulnerability subtypes:
- gets: gets() vs fgets() — unbounded input reading (3 pairs)
- strcpy: strcpy() vs strncpy() — unbounded string copying (4 pairs)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config.cwe119_prompt_pairs import (
    PAIR_01_USER_INPUT,
    PAIR_02_COMMAND_PARSER,
    PAIR_03_CONFIG_READER,
    PAIR_04_USERNAME_COPY,
    PAIR_05_FILEPATH_COPY,
    PAIR_06_ERROR_MSG_COPY,
    PAIR_07_HOSTNAME_COPY,
)

VALIDATED_PAIRS = [
    PAIR_01_USER_INPUT,
    PAIR_02_COMMAND_PARSER,
    PAIR_03_CONFIG_READER,
    PAIR_04_USERNAME_COPY,
    PAIR_05_FILEPATH_COPY,
    PAIR_06_ERROR_MSG_COPY,
    PAIR_07_HOSTNAME_COPY,
]

PAIRS_BY_SHORT_NAME = {
    'user_input': PAIR_01_USER_INPUT,
    'command_parser': PAIR_02_COMMAND_PARSER,
    'config_reader': PAIR_03_CONFIG_READER,
    'username_copy': PAIR_04_USERNAME_COPY,
    'filepath_copy': PAIR_05_FILEPATH_COPY,
    'error_msg_copy': PAIR_06_ERROR_MSG_COPY,
    'hostname_copy': PAIR_07_HOSTNAME_COPY,
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
Validated CWE-119 Prompt Pairs
==============================

7 pairs targeting Improper Restriction of Operations within Memory Buffer Bounds:

| Short Name      | Vulnerability | Context                    |
|-----------------|---------------|----------------------------|
| user_input      | gets          | Read line from stdin       |
| command_parser  | gets          | CLI command input          |
| config_reader   | gets          | Configuration line input   |
| username_copy   | strcpy        | Username string copy       |
| filepath_copy   | strcpy        | File path string copy      |
| error_msg_copy  | strcpy        | Error message copy         |
| hostname_copy   | strcpy        | Hostname string copy       |

All pairs produce CWE-119: Buffer Operations without Bounds
- Vulnerable prompt -> gets/strcpy (no bounds check)
- Secure prompt -> fgets/strncpy (with bounds check)
"""


if __name__ == "__main__":
    print(SUMMARY)
    print(f"Total prompts: {len(get_prompts())}")
