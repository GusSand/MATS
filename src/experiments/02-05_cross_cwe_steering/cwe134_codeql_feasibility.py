#!/usr/bin/env python3
"""
CodeQL Feasibility Check for CWE-134 Outputs at α=1.5

Samples 30 outputs from CWE-134 LOBO/pilot results at α=1.5,
wraps them as standalone C files, and tries to compile with gcc.
Reports compilation rate to determine if CodeQL validation is feasible.

Reuses: extract_c_code() logic from 01-14_codeql_scoring_prototype/02_wrap_code.py
"""

import json
import re
import os
import subprocess
import tempfile
import random
from pathlib import Path

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
FOLD_DIR = SCRIPT_DIR / "experiment_cwe134_llama8b/data/fold_results"
ALPHA = "1.5"
N_SAMPLES = 30

C_HEADERS = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <syslog.h>
"""


def extract_c_code(output: str) -> str:
    """Extract C code from model output. Reused from 01-14_codeql_scoring_prototype."""
    code_block_match = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', output, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    if any(keyword in output for keyword in ['#include', 'void ', 'int ', 'char ', 'return ']):
        lines = output.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if not in_code and (line.strip().startswith('#include') or
                                line.strip().startswith('void ') or
                                line.strip().startswith('int ') or
                                line.strip().startswith('char ') or
                                line.strip().startswith('/*')):
                in_code = True
            if in_code:
                if line.strip() and not any(c in line for c in '{}();#/*'):
                    if len(line.strip().split()) > 10:
                        continue
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()

    return output.strip()


def wrap_as_c_file(code: str, sample_id: str) -> str:
    """Wrap extracted code in a compilable C file."""
    has_main = bool(re.search(r'\bint\s+main\s*\(', code))
    has_includes = '#include' in code

    parts = []
    parts.append(f"// Sample: {sample_id}")

    if not has_includes:
        parts.append(C_HEADERS)

    # Add syslog.h if syslog is used but not included
    if 'syslog' in code and '#include <syslog.h>' not in code:
        parts.append('#include <syslog.h>')

    parts.append(code)

    if not has_main:
        parts.append("\nint main(void) { return 0; }")

    return '\n'.join(parts)


def try_compile(c_source: str) -> tuple:
    """Try to compile C source with gcc. Returns (success, errors)."""
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
        f.write(c_source)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['gcc', '-fsyntax-only', '-w', tmp_path],
            capture_output=True, text=True, timeout=10
        )
        success = result.returncode == 0
        errors = result.stderr.strip() if not success else ""
        return success, errors
    except subprocess.TimeoutExpired:
        return False, "compilation timeout"
    finally:
        os.unlink(tmp_path)


def main():
    print("=" * 60)
    print("CodeQL Feasibility Check: CWE-134 at α=1.5")
    print("=" * 60)

    # Load all fold results (full LOBO + pilot)
    fold_files = sorted(FOLD_DIR.glob("*fold_pair_*_2026*.json"))

    print(f"Found {len(fold_files)} fold files")

    all_samples = []
    for fp in fold_files:
        with open(fp) as f:
            fold = json.load(f)
        fold_id = fold['fold_id']
        if ALPHA in fold['alpha_results']:
            for item in fold['alpha_results'][ALPHA]:
                item['fold_id'] = fold_id
                all_samples.append(item)

    print(f"Total samples at α={ALPHA}: {len(all_samples)}")

    # Sample 30
    if len(all_samples) > N_SAMPLES:
        samples = random.sample(all_samples, N_SAMPLES)
    else:
        samples = all_samples
    print(f"Sampled: {len(samples)}")

    # Process each sample
    results = []
    compile_ok = 0
    compile_fail = 0
    no_code = 0

    print(f"\n{'#':>3} {'ID':<40} {'Label':<10} {'Code?':>5} {'Compiles?':>9}")
    print("-" * 72)

    for i, s in enumerate(samples):
        sample_id = s.get('id', f"sample_{i}")
        code = extract_c_code(s['output'])
        has_code = bool(re.search(r'[{};()]', code))

        if has_code:
            wrapped = wrap_as_c_file(code, sample_id)
            success, errors = try_compile(wrapped)
        else:
            success = False
            errors = "no extractable code"
            wrapped = ""

        if not has_code:
            no_code += 1
            status = "NO CODE"
        elif success:
            compile_ok += 1
            status = "OK"
        else:
            compile_fail += 1
            status = "FAIL"

        results.append({
            'sample_id': sample_id,
            'strict_label': s['strict_label'],
            'has_code': has_code,
            'compiles': success,
            'errors': errors[:200],
            'wrapped_code': wrapped[:500],
        })

        print(f"{i+1:>3} {sample_id:<40} {s['strict_label']:<10} {'Y' if has_code else 'N':>5} {status:>9}")

    n = len(samples)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples:    {n}")
    print(f"Has code:         {n - no_code}/{n} ({(n-no_code)/n*100:.1f}%)")
    print(f"Compiles:         {compile_ok}/{n} ({compile_ok/n*100:.1f}%)")
    print(f"Compile failures: {compile_fail}/{n} ({compile_fail/n*100:.1f}%)")
    print(f"No code:          {no_code}/{n} ({no_code/n*100:.1f}%)")

    if compile_ok / n > 0.5:
        print(f"\n*** FEASIBLE: {compile_ok/n*100:.1f}% > 50% compile rate ***")
        print(f"*** CodeQL validation is worth pursuing ***")
    else:
        print(f"\n*** NOT FEASIBLE: {compile_ok/n*100:.1f}% <= 50% compile rate ***")
        print(f"*** CodeQL validation may not be practical ***")

    # Show common errors
    error_samples = [r for r in results if not r['compiles'] and r['has_code']]
    if error_samples:
        print(f"\n{'='*60}")
        print(f"COMMON COMPILATION ERRORS (showing up to 5)")
        print(f"{'='*60}")
        for r in error_samples[:5]:
            print(f"\n  {r['sample_id']}:")
            print(f"    {r['errors'][:300]}")

    # Save results
    output_dir = SCRIPT_DIR / "cross_cwe_analysis" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'alpha': float(ALPHA),
        'n_samples': n,
        'compile_ok': compile_ok,
        'compile_fail': compile_fail,
        'no_code': no_code,
        'compile_rate': compile_ok / n,
        'feasible': compile_ok / n > 0.5,
        'samples': results,
    }
    results_path = output_dir / "cwe134_codeql_feasibility.json"
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
