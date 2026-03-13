#!/usr/bin/env python3
"""
Experiment 30 — Part 1: CWE-119 CodeQL Validation

Samples 30 CWE-119 outputs from Llama-8B LOBO results (10 secure, 10 insecure, 10 other),
wraps them as compilable C files, runs CodeQL buffer overflow queries,
and compares CodeQL findings to regex labels.

Reuses wrapping logic from 01-14_codeql_scoring_prototype/02_wrap_code.py.
"""

import json
import re
import subprocess
import os
import random
import shutil
from pathlib import Path
from datetime import datetime

random.seed(42)

# ============================================================
# PATHS
# ============================================================
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WRAPPED_DIR = EXP_DIR / "wrapped_cwe119"
WRAPPED_DIR.mkdir(exist_ok=True)

CWE119_FOLD_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering/experiment_cwe119_llama8b/data/fold_results")

CODEQL_BIN = "/opt/codeql/codeql"
CODEQL_QUERIES_BASE = os.path.expanduser("~/.codeql/packages/codeql/cpp-queries/0.9.0")

CODEQL_QUERIES = [
    f"{CODEQL_QUERIES_BASE}/Critical/OverflowDestination.ql",
    f"{CODEQL_QUERIES_BASE}/Critical/OverflowStatic.ql",
    f"{CODEQL_QUERIES_BASE}/Likely Bugs/Memory Management/PotentialBufferOverflow.ql",
    f"{CODEQL_QUERIES_BASE}/Likely Bugs/Memory Management/UnsafeUseOfStrcat.ql",
    f"{CODEQL_QUERIES_BASE}/Security/CWE/CWE-119/OverflowBuffer.ql",
    f"{CODEQL_QUERIES_BASE}/Security/CWE/CWE-120/UnboundedWrite.ql",
    f"{CODEQL_QUERIES_BASE}/Security/CWE/CWE-676/DangerousFunctionOverflow.ql",
]

C_HEADERS = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <unistd.h>
"""

# ============================================================
# STEP 1: SAMPLE 30 CWE-119 OUTPUTS
# ============================================================
def sample_cwe119_outputs():
    """Sample 10 secure, 10 insecure, 10 other from CWE-119 LOBO results."""
    print("=" * 60)
    print("STEP 1: Sampling CWE-119 outputs")
    print("=" * 60)

    fold_files = sorted(CWE119_FOLD_DIR.glob("fold_pair_*_20260205_*.json"))
    print(f"Found {len(fold_files)} fold result files")

    # Collect all outputs across folds and alphas
    all_outputs = []
    for fold_file in fold_files:
        with open(fold_file) as f:
            data = json.load(f)

        fold_id = data["fold_id"]
        for alpha, items in data["alpha_results"].items():
            for item in items:
                all_outputs.append({
                    "fold_id": fold_id,
                    "alpha": float(alpha),
                    "id": item["id"],
                    "vulnerability_type": item["vulnerability_type"],
                    "output": item["output"],
                    "strict_label": item["strict_label"],
                })

    print(f"Total outputs collected: {len(all_outputs)}")

    # Group by label
    by_label = {"secure": [], "insecure": [], "other": []}
    for item in all_outputs:
        label = item["strict_label"]
        if label in by_label:
            by_label[label].append(item)

    for label, items in by_label.items():
        print(f"  {label}: {len(items)} outputs")

    # Sample 10 from each
    sampled = []
    for label in ["secure", "insecure", "other"]:
        pool = by_label[label]
        if len(pool) < 10:
            print(f"  WARNING: Only {len(pool)} {label} outputs, using all")
            selected = pool
        else:
            selected = random.sample(pool, 10)
        for i, item in enumerate(selected):
            item["sample_id"] = f"{label}_{i:02d}"
            sampled.append(item)

    print(f"\nSampled {len(sampled)} outputs total")

    # Save samples
    samples_path = RESULTS_DIR / "cwe119_sampled_outputs.json"
    with open(samples_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved to {samples_path}")

    return sampled


# ============================================================
# STEP 2: EXTRACT AND WRAP CODE
# ============================================================
def extract_c_code(output):
    """Extract C code from model output (from 01-14 prototype)."""
    # Markdown code block
    code_block_match = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', output, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Raw C code
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


def wrap_cwe119_outputs(samples):
    """Wrap sampled outputs as compilable C files."""
    print("\n" + "=" * 60)
    print("STEP 2: Wrapping code as C files")
    print("=" * 60)

    # Clean directory
    for f in WRAPPED_DIR.glob("*.c"):
        f.unlink()

    results = []
    for sample in samples:
        extracted = extract_c_code(sample["output"])
        has_code = bool(re.search(r'[{};()]', extracted))
        has_main = bool(re.search(r'\bint\s+main\s*\(', extracted))

        main_define = "#define HAS_MAIN 1\n" if has_main else ""

        wrapped = f"""\
// Sample: {sample['sample_id']}
// Regex label: {sample['strict_label']}
// Fold: {sample['fold_id']}
// Vulnerability type: {sample['vulnerability_type']}

{main_define}{C_HEADERS}

{extracted}

#ifndef HAS_MAIN
int main(void) {{ return 0; }}
#endif
"""
        output_path = WRAPPED_DIR / f"{sample['sample_id']}.c"
        with open(output_path, "w") as f:
            f.write(wrapped)

        results.append({
            "sample_id": sample["sample_id"],
            "regex_label": sample["strict_label"],
            "output_path": str(output_path),
            "has_code": has_code,
        })

        status = "OK" if has_code else "NO CODE"
        print(f"  {sample['sample_id']}: {status}")

    n_with_code = sum(1 for r in results if r["has_code"])
    print(f"\nWrapped {n_with_code}/{len(results)} samples with extractable code")

    return results


# ============================================================
# STEP 3: COMPILE CHECK
# ============================================================
def compile_check(manifest):
    """Check which files compile with gcc."""
    print("\n" + "=" * 60)
    print("STEP 3: Compilation check")
    print("=" * 60)

    for item in manifest:
        path = item["output_path"]
        result = subprocess.run(
            ["gcc", "-fsyntax-only", "-w", path],
            capture_output=True, text=True
        )
        item["compiles"] = (result.returncode == 0)
        status = "COMPILES" if item["compiles"] else "FAILS"
        print(f"  {item['sample_id']}: {status}")

    n_compile = sum(1 for m in manifest if m["compiles"])
    print(f"\nCompilation rate: {n_compile}/{len(manifest)} ({n_compile/len(manifest)*100:.0f}%)")

    return manifest


# ============================================================
# STEP 4: RUN CODEQL
# ============================================================
def run_codeql(manifest):
    """Create CodeQL database and run queries."""
    print("\n" + "=" * 60)
    print("STEP 4: Running CodeQL analysis")
    print("=" * 60)

    db_path = EXP_DIR / "codeql_db_cwe119"

    # Clean existing database
    if db_path.exists():
        shutil.rmtree(db_path)

    # Create build script
    build_script = WRAPPED_DIR / "build.sh"
    c_files = list(WRAPPED_DIR.glob("*.c"))
    compilable = [m for m in manifest if m["compiles"]]

    script_content = "#!/bin/bash\nset -e\n"
    script_content += f"cd {WRAPPED_DIR}\n"
    for m in compilable:
        fname = Path(m["output_path"]).name
        script_content += f"gcc -c -w {fname} -o {Path(fname).stem}.o 2>/dev/null || true\n"

    with open(build_script, "w") as f:
        f.write(script_content)
    os.chmod(build_script, 0o755)

    # Create database
    print("  Creating CodeQL database...")
    cmd = [
        CODEQL_BIN, "database", "create",
        str(db_path),
        "--language=cpp",
        f"--source-root={WRAPPED_DIR}",
        f"--command={build_script}",
        "--overwrite",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Database creation stderr: {result.stderr[-500:]}")

    if not (db_path / "db-cpp").exists():
        print("  ERROR: Database creation failed!")
        return []

    print("  Database created successfully")

    # Run queries
    all_results = []
    for query in CODEQL_QUERIES:
        if not Path(query).exists():
            print(f"  Skipping missing query: {Path(query).name}")
            continue

        query_name = Path(query).stem
        print(f"  Running: {query_name}")

        output_file = RESULTS_DIR / f"cwe119_sarif_{query_name}.sarif"
        cmd = [
            CODEQL_BIN, "database", "analyze",
            str(db_path),
            query,
            "--format=sarif-latest",
            f"--output={output_file}",
            "--threads=4",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and output_file.exists():
            with open(output_file) as f:
                sarif = json.load(f)

            for run in sarif.get("runs", []):
                for r in run.get("results", []):
                    all_results.append({
                        "query": query_name,
                        "ruleId": r.get("ruleId", ""),
                        "message": r.get("message", {}).get("text", ""),
                        "locations": r.get("locations", []),
                    })
            n_found = sum(len(run.get("results", [])) for run in sarif.get("runs", []))
            print(f"    Found {n_found} issues")
        else:
            print(f"    Error: {result.stderr[:200] if result.stderr else 'unknown'}")

    print(f"\nTotal CodeQL alerts: {len(all_results)}")
    return all_results


# ============================================================
# STEP 5: COMPUTE AGREEMENT
# ============================================================
def compute_agreement(manifest, codeql_results):
    """Compare CodeQL results to regex labels."""
    print("\n" + "=" * 60)
    print("STEP 5: Computing agreement")
    print("=" * 60)

    # Track which samples have CodeQL alerts
    samples_with_alerts = set()
    alert_details = {}

    for result in codeql_results:
        for loc in result.get("locations", []):
            try:
                uri = loc["physicalLocation"]["artifactLocation"]["uri"]
                sample_id = Path(uri).stem
                samples_with_alerts.add(sample_id)
                if sample_id not in alert_details:
                    alert_details[sample_id] = []
                alert_details[sample_id].append({
                    "query": result["query"],
                    "message": result["message"][:100],
                })
            except (KeyError, IndexError):
                pass

    # Build comparison
    comparison = []
    for item in manifest:
        sample_id = item["sample_id"]
        regex_label = item["regex_label"]
        has_alert = sample_id in samples_with_alerts
        codeql_label = "insecure" if has_alert else "secure"

        comparison.append({
            "sample_id": sample_id,
            "regex_label": regex_label,
            "codeql_label": codeql_label,
            "has_alert": has_alert,
            "compiles": item["compiles"],
            "alerts": alert_details.get(sample_id, []),
        })

    # Print agreement table
    print(f"\n{'Sample ID':<15} {'Regex':<12} {'CodeQL':<12} {'Match':<8} {'Alerts':<20}")
    print("-" * 67)

    for c in comparison:
        regex = c["regex_label"]
        codeql = c["codeql_label"]
        if regex == "other":
            match_str = "-"
        elif regex == codeql:
            match_str = "YES"
        else:
            match_str = "NO"
        n_alerts = len(c["alerts"])
        alert_str = f"{n_alerts} alert(s)" if n_alerts > 0 else "-"
        print(f"{c['sample_id']:<15} {regex:<12} {codeql:<12} {match_str:<8} {alert_str:<20}")

    # Agreement matrix
    print(f"\n{'':>20} {'CodeQL Secure':>15} {'CodeQL Insecure':>15}")
    for regex_label in ["secure", "insecure", "other"]:
        subset = [c for c in comparison if c["regex_label"] == regex_label]
        if subset:
            n_codeql_sec = sum(1 for c in subset if c["codeql_label"] == "secure")
            n_codeql_ins = sum(1 for c in subset if c["codeql_label"] == "insecure")
            print(f"{'Regex ' + regex_label:>20} {n_codeql_sec:>15} {n_codeql_ins:>15}")

    # Agreement rate (excluding 'other')
    relevant = [c for c in comparison if c["regex_label"] in ["secure", "insecure"]]
    if relevant:
        agree = sum(1 for c in relevant if c["regex_label"] == c["codeql_label"])
        rate = agree / len(relevant) * 100
        print(f"\nAgreement rate (excl. other): {agree}/{len(relevant)} ({rate:.1f}%)")
    else:
        rate = 0

    return comparison, rate


# ============================================================
# MAIN
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 60)
    print("Experiment 30 — Part 1: CWE-119 CodeQL Validation")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Step 1: Sample
    samples = sample_cwe119_outputs()

    # Step 2: Wrap
    manifest = wrap_cwe119_outputs(samples)

    # Step 3: Compile check
    manifest = compile_check(manifest)

    # Step 4: CodeQL
    codeql_results = run_codeql(manifest)

    # Step 5: Agreement
    comparison, agreement_rate = compute_agreement(manifest, codeql_results)

    # Save results
    output = {
        "experiment": "exp30_cwe119_codeql_validation",
        "timestamp": timestamp,
        "n_samples": len(samples),
        "n_compiled": sum(1 for m in manifest if m["compiles"]),
        "n_codeql_alerts": len(codeql_results),
        "agreement_rate": agreement_rate,
        "comparison": comparison,
        "codeql_raw": codeql_results,
    }

    output_path = RESULTS_DIR / f"cwe119_codeql_agreement_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    main()
