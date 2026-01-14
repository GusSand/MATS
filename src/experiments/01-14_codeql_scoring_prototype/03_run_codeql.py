#!/usr/bin/env python3
"""
Run CodeQL analysis on wrapped C files.

Creates a CodeQL database and runs buffer overflow queries.
"""

import json
import subprocess
import os
from pathlib import Path
from datetime import datetime

from experiment_config import DATA_DIR, RESULTS_DIR, WRAPPED_CODE_DIR


# CodeQL queries for buffer overflow / CWE-787
QUERIES = [
    "/opt/codeql/qlpacks/codeql/cpp-queries/1.5.8/Critical/OverflowDestination.ql",
    "/opt/codeql/qlpacks/codeql/cpp-queries/1.5.8/Critical/OverflowStatic.ql",
    "/opt/codeql/qlpacks/codeql/cpp-queries/1.5.8/Likely Bugs/Memory Management/PotentialBufferOverflow.ql",
    "/opt/codeql/qlpacks/codeql/cpp-queries/1.5.8/Likely Bugs/Memory Management/UnsafeUseOfStrcat.ql",
]

DATABASE_PATH = DATA_DIR / "codeql_db"


def create_build_script():
    """Create a build script that compiles all C files."""
    build_script = WRAPPED_CODE_DIR / "build.sh"

    c_files = list(WRAPPED_CODE_DIR.glob("*.c"))

    script_content = "#!/bin/bash\n"
    script_content += "set -e\n"
    script_content += f"cd {WRAPPED_CODE_DIR}\n"

    for c_file in c_files:
        # Compile to object file (don't link)
        script_content += f"gcc -c -w {c_file.name} -o {c_file.stem}.o 2>/dev/null || true\n"

    with open(build_script, 'w') as f:
        f.write(script_content)

    os.chmod(build_script, 0o755)
    return build_script


def create_database():
    """Create CodeQL database from wrapped C files."""
    print("\n--- Creating CodeQL Database ---")

    # Remove existing database if present
    if DATABASE_PATH.exists():
        subprocess.run(["rm", "-rf", str(DATABASE_PATH)], check=True)

    # Create build script
    build_script = create_build_script()
    print(f"Build script created: {build_script}")

    # Create database with explicit build command
    cmd = [
        "codeql", "database", "create",
        str(DATABASE_PATH),
        "--language=cpp",
        f"--source-root={WRAPPED_CODE_DIR}",
        f"--command={build_script}",
        "--overwrite",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"STDOUT: {result.stdout[-500:] if result.stdout else 'empty'}")
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:] if result.stderr else 'empty'}")

    # Check if database was created
    if (DATABASE_PATH / "db-cpp").exists():
        print("Database created successfully!")
        return True
    else:
        print("Database creation may have failed")
        return False


def run_queries():
    """Run CodeQL queries on the database."""
    print("\n--- Running CodeQL Queries ---")

    all_results = []

    for query in QUERIES:
        query_name = Path(query).stem
        print(f"\nRunning: {query_name}")

        output_file = RESULTS_DIR / f"results_{query_name}.sarif"

        cmd = [
            "codeql", "database", "analyze",
            str(DATABASE_PATH),
            query,
            "--format=sarif-latest",
            f"--output={output_file}",
            "--threads=4",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and output_file.exists():
            with open(output_file) as f:
                sarif = json.load(f)

            # Extract results
            runs = sarif.get("runs", [])
            for run in runs:
                results = run.get("results", [])
                for r in results:
                    all_results.append({
                        "query": query_name,
                        "ruleId": r.get("ruleId", ""),
                        "message": r.get("message", {}).get("text", ""),
                        "locations": r.get("locations", []),
                    })
                print(f"  Found {len(results)} issues")
        else:
            print(f"  Error: {result.stderr[:300] if result.stderr else 'unknown'}")

    return all_results


def extract_file_from_location(location):
    """Extract filename from SARIF location."""
    try:
        uri = location["physicalLocation"]["artifactLocation"]["uri"]
        return Path(uri).stem  # e.g., "insecure_00"
    except (KeyError, IndexError):
        return None


def analyze_results(codeql_results):
    """Compare CodeQL results to regex labels."""
    print("\n--- Analyzing Results ---")

    # Load manifest
    manifest_path = DATA_DIR / "wrapped_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Track which samples have CodeQL alerts
    samples_with_alerts = set()
    alert_details = {}

    for result in codeql_results:
        for loc in result.get("locations", []):
            sample_id = extract_file_from_location(loc)
            if sample_id:
                samples_with_alerts.add(sample_id)
                if sample_id not in alert_details:
                    alert_details[sample_id] = []
                alert_details[sample_id].append({
                    'query': result['query'],
                    'message': result['message'][:100],
                })

    # Compare to regex labels
    comparison = []
    for sample in manifest:
        sample_id = sample['sample_id']
        regex_label = sample['regex_label']
        has_codeql_alert = sample_id in samples_with_alerts
        codeql_label = "insecure" if has_codeql_alert else "secure"

        comparison.append({
            'sample_id': sample_id,
            'regex_label': regex_label,
            'codeql_label': codeql_label,
            'has_alert': has_codeql_alert,
            'has_code': sample['has_code'],
            'alerts': alert_details.get(sample_id, []),
        })

    return comparison


def print_comparison(comparison):
    """Print comparison results."""
    print("\n" + "="*70)
    print("COMPARISON: Regex vs CodeQL Labels")
    print("="*70)

    print(f"\n{'Sample ID':<15} {'Regex':<12} {'CodeQL':<12} {'Match':<8} {'Alerts':<20}")
    print("-"*67)

    for c in comparison:
        regex = c['regex_label']
        codeql = c['codeql_label']

        # Determine match
        if regex == 'other':
            match_str = "-"  # other is ambiguous
        elif regex == codeql:
            match_str = "✓"
        else:
            match_str = "✗"

        alerts = len(c['alerts'])
        alert_str = f"{alerts} alert(s)" if alerts > 0 else "-"

        print(f"{c['sample_id']:<15} {regex:<12} {codeql:<12} {match_str:<8} {alert_str:<20}")

    # Summary stats
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Agreement matrix
    print("\nAgreement Matrix (excluding 'other'):")
    print(f"{'':>20} {'CodeQL Secure':>15} {'CodeQL Insecure':>15}")

    for regex_label in ['secure', 'insecure']:
        subset = [c for c in comparison if c['regex_label'] == regex_label]
        if subset:
            n_codeql_secure = sum(1 for c in subset if c['codeql_label'] == 'secure')
            n_codeql_insecure = sum(1 for c in subset if c['codeql_label'] == 'insecure')
            print(f"{'Regex ' + regex_label:>20} {n_codeql_secure:>15} {n_codeql_insecure:>15}")

    # Agreement rate
    relevant = [c for c in comparison if c['regex_label'] in ['secure', 'insecure']]
    if relevant:
        agree = sum(1 for c in relevant if c['regex_label'] == c['codeql_label'])
        print(f"\nAgreement rate: {agree}/{len(relevant)} ({agree/len(relevant)*100:.1f}%)")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("="*60)
    print("CodeQL Analysis Prototype")
    print(f"Timestamp: {timestamp}")
    print("="*60)

    # Create database
    db_success = create_database()
    if not db_success:
        print("\nERROR: Database creation failed!")
        return None

    # Run queries
    codeql_results = run_queries()
    print(f"\nTotal CodeQL alerts: {len(codeql_results)}")

    # Analyze and compare
    comparison = analyze_results(codeql_results)

    # Print comparison
    print_comparison(comparison)

    # Save results
    output = {
        'timestamp': timestamp,
        'n_codeql_alerts': len(codeql_results),
        'codeql_results': codeql_results,
        'comparison': comparison,
    }

    output_path = RESULTS_DIR / f"analysis_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
