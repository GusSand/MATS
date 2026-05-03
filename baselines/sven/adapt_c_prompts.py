"""
Adapt C-language instruction-style prompts to code-completion format for CodeGen.

Our C CWE prompts (119, 134, 787) are instruction-style:
    "Task: Write a C function that reads a line of user input...
     Requirements:
     - Function signature: void read_input(char* buffer)
     - Use gets() for simple, direct input reading
     ..."

CodeGen expects code-completion format:
    #include <stdio.h>

    /* Read a line of user input into the buffer */
    void read_input(char* buffer) {

This script converts mechanically by extracting the function signature and
composing a minimal C stub. The detection patterns in the original JSON
still apply to the generated code — we're only changing the prompt, not
the evaluation criteria.
"""

import json
import re
import sys
from pathlib import Path


# Standard includes by vulnerability type
INCLUDES_BY_VULN_TYPE = {
    # CWE-119: buffer read overflow
    "gets": "#include <stdio.h>\n#include <string.h>",
    "strcpy": "#include <stdio.h>\n#include <string.h>",
    # CWE-134: format string
    "printf_format": "#include <stdio.h>",
    "fprintf_format": "#include <stdio.h>",
    "syslog_format": "#include <stdio.h>\n#include <syslog.h>",
    # CWE-787: buffer write overflow
    "sprintf": "#include <stdio.h>\n#include <string.h>",
    "strcat": "#include <stdio.h>\n#include <string.h>",
}


def extract_function_signature(prompt_text: str) -> str | None:
    """Extract function signature from 'Function signature: ...' line."""
    match = re.search(
        r"[Ff]unction\s+(?:signature|prototype|declaration):\s*(.+?)(?:\n|$)",
        prompt_text,
    )
    if match:
        sig = match.group(1).strip()
        # Remove trailing punctuation
        sig = sig.rstrip(".")
        return sig
    return None


def extract_task_description(prompt_text: str) -> str:
    """Extract the task description from 'Task: ...' or first line."""
    match = re.search(r"(?:Task|Objective|Create|Design|Implement|Write|Develop):\s*(.+?)(?:\n|$)", prompt_text)
    if match:
        desc = match.group(1).strip().rstrip(".")
        return desc
    # Fallback: first non-empty line
    for line in prompt_text.split("\n"):
        line = line.strip()
        if line:
            return line.rstrip(".")
    return "Implementation"


def adapt_prompt(prompt_text: str, vulnerability_type: str) -> str:
    """Convert an instruction-style C prompt to a code-completion stub."""
    sig = extract_function_signature(prompt_text)
    if sig is None:
        # Some prompts don't have "Function signature:" — try to find it in the text
        # Look for common C function patterns
        match = re.search(r"((?:void|int|char\s*\*|size_t|FILE\s*\*)\s+\w+\s*\([^)]*\))", prompt_text)
        if match:
            sig = match.group(1).strip()
        else:
            return None  # Can't adapt — flag for manual review

    desc = extract_task_description(prompt_text)
    includes = INCLUDES_BY_VULN_TYPE.get(vulnerability_type, "#include <stdio.h>")

    # Build code-completion stub
    stub = f"{includes}\n\n/* {desc} */\n{sig} {{\n"
    return stub


def adapt_dataset(input_path: str, output_path: str) -> dict:
    """Adapt all prompts in a JSONL file. Returns stats."""
    adapted = []
    failed = []

    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)
            vuln_type = entry["vulnerability_type"]

            # Adapt both vulnerable and secure prompts
            vuln_adapted = adapt_prompt(entry["vulnerable"], vuln_type)
            sec_adapted = adapt_prompt(entry["secure"], vuln_type)

            if vuln_adapted is None or sec_adapted is None:
                failed.append(entry["id"])
                # Keep original as fallback but flag it
                adapted_entry = {
                    **entry,
                    "codegen_vulnerable": entry["vulnerable"],
                    "codegen_secure": entry["secure"],
                    "adaptation_failed": True,
                }
            else:
                adapted_entry = {
                    **entry,
                    "codegen_vulnerable": vuln_adapted,
                    "codegen_secure": sec_adapted,
                    "adaptation_failed": False,
                }
            adapted.append(adapted_entry)

    with open(output_path, "w") as f:
        for entry in adapted:
            f.write(json.dumps(entry) + "\n")

    return {
        "total": len(adapted),
        "adapted": len(adapted) - len(failed),
        "failed": len(failed),
        "failed_ids": failed,
    }


def main():
    base = Path(__file__).resolve().parent.parent.parent  # MATS root

    datasets = [
        (
            base / "src/experiments/02-05_cross_cwe_steering/datasets/cwe119/data/cwe119_expanded_20260207_024627.jsonl",
            "CWE-119",
        ),
        (
            base / "src/experiments/02-05_cross_cwe_steering/datasets/cwe134/data/cwe134_expanded_20260207_024627.jsonl",
            "CWE-134",
        ),
        (
            base / "src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl",
            "CWE-787",
        ),
    ]

    output_dir = Path(__file__).resolve().parent / "adapted_prompts"
    output_dir.mkdir(exist_ok=True)

    for input_path, cwe_name in datasets:
        output_path = output_dir / f"{cwe_name.lower()}_codegen.jsonl"
        stats = adapt_dataset(str(input_path), str(output_path))
        print(f"{cwe_name}: {stats['adapted']}/{stats['total']} adapted, {stats['failed']} failed")
        if stats["failed_ids"]:
            print(f"  Failed: {stats['failed_ids']}")

    # Also copy Python CWE datasets as-is (they're already code-completion format)
    # Just add codegen_insecure/codegen_secure fields that mirror insecure_prompt/secure_prompt
    py_datasets = [
        (
            base / "src/experiments/02-05_cross_cwe_steering/datasets/cwe89/data/cwe89_expanded_20260209_221808.jsonl",
            "CWE-89",
        ),
        (
            base / "src/experiments/02-05_cross_cwe_steering/datasets/cwe78/data/cwe78_expanded_20260209_221808.jsonl",
            "CWE-78",
        ),
        (
            base / "src/experiments/02-05_cross_cwe_steering/datasets/cwe79/data/cwe79_expanded_20260209_221808.jsonl",
            "CWE-79",
        ),
    ]

    for input_path, cwe_name in py_datasets:
        output_path = output_dir / f"{cwe_name.lower()}_codegen.jsonl"
        count = 0
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                entry = json.loads(line)
                # Python prompts use insecure_prompt/secure_prompt keys
                adapted_entry = {
                    **entry,
                    "codegen_vulnerable": entry["insecure_prompt"],
                    "codegen_secure": entry["secure_prompt"],
                    "adaptation_failed": False,
                }
                fout.write(json.dumps(adapted_entry) + "\n")
                count += 1
        print(f"{cwe_name}: {count}/{count} copied (already code-completion format)")


if __name__ == "__main__":
    main()
