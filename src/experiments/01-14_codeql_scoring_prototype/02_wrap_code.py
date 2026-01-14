#!/usr/bin/env python3
"""
Wrap code snippets in compilable C files for CodeQL analysis.

Extracts C code from model outputs and wraps in minimal compilable files.
"""

import json
import re
from pathlib import Path

from experiment_config import DATA_DIR, WRAPPED_CODE_DIR


# Standard C headers to include
C_HEADERS = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
"""

# Wrapper template
WRAPPER_TEMPLATE = """
// Sample: {sample_id}
// Regex label: {regex_label}
// Fold: {fold_id}
// Vulnerability type: {vulnerability_type}

{headers}

{code}

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) {{ return 0; }}
#endif
"""


def extract_c_code(output: str) -> str:
    """
    Extract C code from model output.

    Handles:
    - Markdown code blocks (```c ... ```)
    - Raw C code
    - JSON with embedded code
    """
    # Try to find markdown code block
    code_block_match = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', output, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Try to find JSON with code field
    json_match = re.search(r'"code"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', output)
    if json_match:
        code = json_match.group(1)
        # Unescape JSON string
        code = code.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        return code.strip()

    # Check if output looks like raw C code
    if any(keyword in output for keyword in ['#include', 'void ', 'int ', 'char ', 'return ']):
        # Try to extract just the code part (skip any prose before/after)
        lines = output.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Start of code
            if not in_code and (line.strip().startswith('#include') or
                                line.strip().startswith('void ') or
                                line.strip().startswith('int ') or
                                line.strip().startswith('char ') or
                                line.strip().startswith('/*')):
                in_code = True

            if in_code:
                # Stop at obvious prose
                if line.strip() and not any(c in line for c in '{}();#/*'):
                    if len(line.strip().split()) > 10:  # Likely prose
                        continue
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

    # Fallback: return as-is
    return output.strip()


def check_has_main(code: str) -> bool:
    """Check if code has a main function."""
    return bool(re.search(r'\bint\s+main\s*\(', code))


def wrap_code(sample: dict) -> tuple[str, bool]:
    """
    Wrap a sample's code in a compilable C file.

    Returns (wrapped_code, extraction_success).
    """
    raw_output = sample['output']
    extracted = extract_c_code(raw_output)

    # Check if extraction found actual code
    has_code = bool(re.search(r'[{};()]', extracted))

    # Check for main
    has_main = check_has_main(extracted)
    main_define = "#define HAS_MAIN 1\n" if has_main else ""

    wrapped = WRAPPER_TEMPLATE.format(
        sample_id=sample['sample_id'],
        regex_label=sample['regex_label'],
        fold_id=sample['fold_id'],
        vulnerability_type=sample['vulnerability_type'],
        headers=main_define + C_HEADERS,
        code=extracted,
    )

    return wrapped, has_code


def main():
    print("="*60)
    print("Wrapping Code Snippets for CodeQL")
    print("="*60)

    # Load samples
    samples_path = DATA_DIR / "sampled_outputs.json"
    with open(samples_path) as f:
        samples = json.load(f)

    print(f"\nLoaded {len(samples)} samples")

    # Ensure output directory exists
    WRAPPED_CODE_DIR.mkdir(exist_ok=True)

    # Process each sample
    results = []
    for sample in samples:
        sample_id = sample['sample_id']
        wrapped, has_code = wrap_code(sample)

        # Save wrapped code
        output_path = WRAPPED_CODE_DIR / f"{sample_id}.c"
        with open(output_path, 'w') as f:
            f.write(wrapped)

        results.append({
            'sample_id': sample_id,
            'regex_label': sample['regex_label'],
            'output_path': str(output_path),
            'has_code': has_code,
        })

        status = "OK" if has_code else "NO CODE"
        print(f"  {sample_id}: {status}")

    # Summary
    n_with_code = sum(1 for r in results if r['has_code'])
    print(f"\nWrapped {n_with_code}/{len(results)} samples with extractable code")

    # Save results manifest
    manifest_path = DATA_DIR / "wrapped_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved manifest to: {manifest_path}")

    # Show a wrapped example
    print("\n" + "="*60)
    print("Example Wrapped File:")
    print("="*60)
    example_path = WRAPPED_CODE_DIR / "insecure_00.c"
    if example_path.exists():
        with open(example_path) as f:
            print(f.read()[:1500])

    return manifest_path


if __name__ == "__main__":
    main()
