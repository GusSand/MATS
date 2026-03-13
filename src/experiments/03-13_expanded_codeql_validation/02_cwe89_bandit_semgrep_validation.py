#!/usr/bin/env python3
"""
Experiment 30 — Part 2: CWE-89 Bandit/Semgrep Validation

Samples 30 CWE-89 outputs from Llama-8B baseline results (10 secure, 10 insecure, 10 other),
wraps them as Python files, runs Bandit and Semgrep SQL injection checks,
and compares findings to regex labels.
"""

import json
import re
import subprocess
import os
import random
from pathlib import Path
from datetime import datetime

random.seed(42)

# ============================================================
# PATHS
# ============================================================
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WRAPPED_DIR = EXP_DIR / "wrapped_cwe89"
WRAPPED_DIR.mkdir(exist_ok=True)

CWE89_BASELINE_PATH = Path("/home/paperspace/MATS/src/experiments/02-10_python_cwe_steering/results/baseline_full_rescored_20260210_021448.json")


# ============================================================
# STEP 1: SAMPLE 30 CWE-89 OUTPUTS
# ============================================================
def sample_cwe89_outputs():
    """Sample 10 secure, 10 insecure, 10 other from CWE-89 baseline results."""
    print("=" * 60)
    print("STEP 1: Sampling CWE-89 outputs")
    print("=" * 60)

    with open(CWE89_BASELINE_PATH) as f:
        data = json.load(f)

    # Use insecure_results which has a mix of labels (after rescoring)
    all_outputs = data["insecure_results"]
    print(f"Total outputs available: {len(all_outputs)}")

    # Group by label
    by_label = {"secure": [], "insecure": [], "other": []}
    for item in all_outputs:
        label = item["label"]
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
            sampled.append({
                "sample_id": f"{label}_{i:02d}",
                "regex_label": label,
                "pair_id": item.get("pair_id", item.get("id", "")),
                "output": item["output"],
            })

    print(f"\nSampled {len(sampled)} outputs total")

    # Save
    samples_path = RESULTS_DIR / "cwe89_sampled_outputs.json"
    with open(samples_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved to {samples_path}")

    return sampled


# ============================================================
# STEP 2: WRAP AS PYTHON FILES
# ============================================================
def extract_python_code(output):
    """Extract Python code from model output."""
    # Markdown code block
    code_block_match = re.search(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Raw Python code
    if any(keyword in output for keyword in ['import ', 'def ', 'class ', 'cursor', 'execute', 'SELECT']):
        lines = output.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if not in_code and (line.strip().startswith('import ') or
                                line.strip().startswith('from ') or
                                line.strip().startswith('def ') or
                                line.strip().startswith('class ')):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()

    return output.strip()


def wrap_cwe89_outputs(samples):
    """Wrap sampled outputs as Python files."""
    print("\n" + "=" * 60)
    print("STEP 2: Wrapping code as Python files")
    print("=" * 60)

    # Clean directory
    for f in WRAPPED_DIR.glob("*.py"):
        f.unlink()

    results = []
    for sample in samples:
        extracted = extract_python_code(sample["output"])

        # Add standard imports if missing
        imports = ""
        if "import sqlite3" not in extracted and "sqlite3" not in extracted:
            imports += "import sqlite3\n"

        code = f"""\
# Sample: {sample['sample_id']}
# Regex label: {sample['regex_label']}
# Pair: {sample['pair_id']}

{imports}{extracted}
"""

        output_path = WRAPPED_DIR / f"{sample['sample_id']}.py"
        with open(output_path, "w") as f:
            f.write(code)

        has_sql = bool(re.search(r'(?:select|insert|update|delete)', extracted, re.IGNORECASE))
        results.append({
            "sample_id": sample["sample_id"],
            "regex_label": sample["regex_label"],
            "output_path": str(output_path),
            "has_sql": has_sql,
        })

        status = "HAS SQL" if has_sql else "NO SQL"
        print(f"  {sample['sample_id']}: {status}")

    n_with_sql = sum(1 for r in results if r["has_sql"])
    print(f"\nWrapped {n_with_sql}/{len(results)} samples with SQL code")

    return results


# ============================================================
# STEP 3: RUN BANDIT
# ============================================================
def run_bandit(manifest):
    """Run Bandit SQL injection checks."""
    print("\n" + "=" * 60)
    print("STEP 3: Running Bandit")
    print("=" * 60)

    bandit_output = RESULTS_DIR / "cwe89_bandit_raw.json"

    cmd = [
        "bandit", "-r", str(WRAPPED_DIR),
        "-f", "json",
        "-o", str(bandit_output),
        "--severity-level", "low",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Bandit returns non-zero if it finds issues — this is expected
    print(f"  Bandit exit code: {result.returncode}")

    if not bandit_output.exists():
        print(f"  WARNING: No bandit output generated")
        print(f"  stderr: {result.stderr[:500]}")
        return {}

    with open(bandit_output) as f:
        bandit_data = json.load(f)

    # Map results to samples
    sample_alerts = {}
    for finding in bandit_data.get("results", []):
        filename = Path(finding["filename"]).stem
        if filename not in sample_alerts:
            sample_alerts[filename] = []
        sample_alerts[filename].append({
            "test_id": finding["test_id"],
            "test_name": finding["test_name"],
            "severity": finding["issue_severity"],
            "confidence": finding["issue_confidence"],
            "text": finding["issue_text"][:100],
        })

    # Filter for SQL injection specifically (B608)
    sql_injection_alerts = {}
    for sample_id, alerts in sample_alerts.items():
        sql_alerts = [a for a in alerts if a["test_id"] == "B608"]
        if sql_alerts:
            sql_injection_alerts[sample_id] = sql_alerts

    print(f"\n  Total Bandit findings: {len(bandit_data.get('results', []))}")
    print(f"  Samples with any alert: {len(sample_alerts)}")
    print(f"  Samples with SQL injection alert (B608): {len(sql_injection_alerts)}")

    return {"all_alerts": sample_alerts, "sql_injection_alerts": sql_injection_alerts}


# ============================================================
# STEP 4: RUN SEMGREP
# ============================================================
def run_semgrep(manifest):
    """Run Semgrep SQL injection checks."""
    print("\n" + "=" * 60)
    print("STEP 4: Running Semgrep")
    print("=" * 60)

    semgrep_output = RESULTS_DIR / "cwe89_semgrep_raw.json"

    # Try multiple rule configs
    # Use --config auto which includes SQL injection rules (sqlalchemy, formatted-sql-query)
    cmd = [
        "semgrep", "scan",
        "--config", "auto",
        str(WRAPPED_DIR),
        "--json",
        "-o", str(semgrep_output),
        "--no-git-ignore",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print(f"  Semgrep exit code: {result.returncode}")

    if not semgrep_output.exists():
        print("  Semgrep failed to produce output")
        return {}

    with open(semgrep_output) as f:
        semgrep_data = json.load(f)

    # Map results to samples
    sample_alerts = {}
    for finding in semgrep_data.get("results", []):
        filename = Path(finding["path"]).stem
        if filename not in sample_alerts:
            sample_alerts[filename] = []
        sample_alerts[filename].append({
            "check_id": finding.get("check_id", ""),
            "message": finding.get("extra", {}).get("message", "")[:100],
            "severity": finding.get("extra", {}).get("severity", ""),
        })

    # Filter for SQL injection specifically (includes sqlalchemy, formatted-sql-query rules)
    sql_alerts = {}
    for sample_id, alerts in sample_alerts.items():
        sql = [a for a in alerts if any(kw in a["check_id"].lower() for kw in ["sql", "injection"])]
        if sql:
            sql_alerts[sample_id] = sql

    print(f"\n  Total Semgrep findings: {len(semgrep_data.get('results', []))}")
    print(f"  Samples with any alert: {len(sample_alerts)}")
    print(f"  Samples with SQL injection alert: {len(sql_alerts)}")

    return {"all_alerts": sample_alerts, "sql_injection_alerts": sql_alerts}


# ============================================================
# STEP 5: COMPUTE AGREEMENT
# ============================================================
def compute_agreement(manifest, bandit_results, semgrep_results):
    """Compare static analysis results to regex labels."""
    print("\n" + "=" * 60)
    print("STEP 5: Computing agreement")
    print("=" * 60)

    bandit_sql = bandit_results.get("sql_injection_alerts", {})
    bandit_all = bandit_results.get("all_alerts", {})
    semgrep_sql = semgrep_results.get("sql_injection_alerts", {})
    semgrep_all = semgrep_results.get("all_alerts", {})

    comparison = []
    for item in manifest:
        sample_id = item["sample_id"]
        regex_label = item["regex_label"]

        has_bandit_sql = sample_id in bandit_sql
        has_bandit_any = sample_id in bandit_all
        has_semgrep_sql = sample_id in semgrep_sql
        has_semgrep_any = sample_id in semgrep_all

        bandit_label = "insecure" if has_bandit_sql else "secure"
        semgrep_label = "insecure" if has_semgrep_sql else "secure"
        # Combined: insecure if either tool flags it
        combined_label = "insecure" if (has_bandit_sql or has_semgrep_sql) else "secure"

        comparison.append({
            "sample_id": sample_id,
            "regex_label": regex_label,
            "bandit_label": bandit_label,
            "semgrep_label": semgrep_label,
            "combined_label": combined_label,
            "bandit_sql_alerts": bandit_sql.get(sample_id, []),
            "bandit_all_alerts": bandit_all.get(sample_id, []),
            "semgrep_sql_alerts": semgrep_sql.get(sample_id, []),
            "semgrep_all_alerts": semgrep_all.get(sample_id, []),
        })

    # Print comparison table
    print(f"\n{'Sample ID':<15} {'Regex':<10} {'Bandit':<10} {'Semgrep':<10} {'Combined':<10}")
    print("-" * 55)
    for c in comparison:
        print(f"{c['sample_id']:<15} {c['regex_label']:<10} {c['bandit_label']:<10} "
              f"{c['semgrep_label']:<10} {c['combined_label']:<10}")

    # Agreement matrices
    for tool_name, tool_key in [("Bandit", "bandit_label"), ("Semgrep", "semgrep_label"), ("Combined", "combined_label")]:
        print(f"\n{tool_name} Agreement Matrix:")
        print(f"{'':>20} {tool_name + ' Secure':>15} {tool_name + ' Insecure':>15}")
        for regex_label in ["secure", "insecure", "other"]:
            subset = [c for c in comparison if c["regex_label"] == regex_label]
            if subset:
                n_sec = sum(1 for c in subset if c[tool_key] == "secure")
                n_ins = sum(1 for c in subset if c[tool_key] == "insecure")
                print(f"{'Regex ' + regex_label:>20} {n_sec:>15} {n_ins:>15}")

        relevant = [c for c in comparison if c["regex_label"] in ["secure", "insecure"]]
        if relevant:
            agree = sum(1 for c in relevant if c["regex_label"] == c[tool_key])
            rate = agree / len(relevant) * 100
            print(f"  Agreement rate (excl. other): {agree}/{len(relevant)} ({rate:.1f}%)")

    return comparison


# ============================================================
# MAIN
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 60)
    print("Experiment 30 — Part 2: CWE-89 Bandit/Semgrep Validation")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Step 1: Sample
    samples = sample_cwe89_outputs()

    # Step 2: Wrap
    manifest = wrap_cwe89_outputs(samples)

    # Step 3: Bandit
    bandit_results = run_bandit(manifest)

    # Step 4: Semgrep
    semgrep_results = run_semgrep(manifest)

    # Step 5: Agreement
    comparison = compute_agreement(manifest, bandit_results, semgrep_results)

    # Compute overall rates
    relevant = [c for c in comparison if c["regex_label"] in ["secure", "insecure"]]
    bandit_rate = sum(1 for c in relevant if c["regex_label"] == c["bandit_label"]) / max(len(relevant), 1) * 100
    semgrep_rate = sum(1 for c in relevant if c["regex_label"] == c["semgrep_label"]) / max(len(relevant), 1) * 100
    combined_rate = sum(1 for c in relevant if c["regex_label"] == c["combined_label"]) / max(len(relevant), 1) * 100

    # Save results
    output = {
        "experiment": "exp30_cwe89_bandit_semgrep_validation",
        "timestamp": timestamp,
        "n_samples": len(samples),
        "agreement_rates": {
            "bandit": bandit_rate,
            "semgrep": semgrep_rate,
            "combined": combined_rate,
        },
        "comparison": comparison,
    }

    output_path = RESULTS_DIR / f"cwe89_bandit_semgrep_agreement_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    main()
