#!/usr/bin/env python3
"""Preflight for one Slurm experiment directory: dry_run_check.py <exp_dir> [extra files...]. Exit 1 on any problem.
Run from the repo root with relative paths. Checks: everything that is copied is committed and pushed; run.sbatch parses,
sets account/partition/walltime/output and no --qos; the last design-review line of the experiment file is PASS, its
report is committed and names this script's sha256; smoke_cpu.log is committed and matches the script."""
import hashlib
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
from check_reviews import header, sections  # noqa: E402

d, extra = sys.argv[1].rstrip("/"), sys.argv[2:]
sh = lambda *a: subprocess.run(a, capture_output=True, text=True)
bad = []
for f in [d] + extra:
    if os.path.isabs(f) or f.startswith(".."):
        sys.exit(f"PREFLIGHT: {f} must be relative to the repo root")
script, sb, log = (os.path.join(d, f) for f in ("run_experiment.py", "run.sbatch", "smoke_cpu.log"))
text = open(sb).read()
sha = hashlib.sha256(open(script, "rb").read()).hexdigest()
m = re.search(r"docs/experiments/[\w.-]+\.md", text)
doc_path = m.group(0) if m else None
doc = open(doc_path).read() if doc_path and os.path.exists(doc_path) else ""
review = None
lines = re.findall(r"^\*\*Design review:\*\*\s*(.+)$", doc, re.M)
if not lines or not lines[-1].startswith("PASS"):
    bad.append("the last design-review line does not start with PASS")
else:
    r = re.search(r"docs/reviews/[\w./-]+\.md", lines[-1])
    review = r.group(0) if r else None
    if not review or not os.path.exists(review):
        bad.append("the last design review has no saved report")
    else:
        sec = sections(open(review).read())
        if sha not in sec.get("Report", ""):
            bad.append("the last design report does not name this script's sha256")
        if not (header(sec["_head"], "Verdict") or "").startswith("PASS"):
            bad.append("the last design report's own verdict is not PASS")
needs = set(re.findall(r"^# Needs: (\S+)$", text, re.M))
if needs != set(extra):
    bad.append(f"extra files differ from run.sbatch's Needs lines: {sorted(needs ^ set(extra))}")
for f in [script, sb, log, *extra] + [x for x in (doc_path, review) if x]:
    if sh("git", "ls-files", "--error-unmatch", f).returncode:
        bad.append(f"{f} is not tracked")
    elif sh("git", "status", "--porcelain", "--", f).stdout.strip():
        bad.append(f"{f} has uncommitted changes")
if sh("git", "fetch", "-q").returncode:
    bad.append("git fetch failed")
if sh("git", "rev-parse", "HEAD").stdout != sh("git", "rev-parse", "origin/main").stdout:
    bad.append("HEAD is not origin/main (push or pull first)")
p = sh("bash", "-n", sb)
if p.returncode:
    bad.append("run.sbatch does not parse: " + p.stderr.strip())
for pat in ("--account=torch_pr_173_tandon_advanced", "--partition=", "--time=", "--output="):
    if pat not in text:
        bad.append(f"run.sbatch lacks {pat}")
if re.search(r"^#SBATCH --qos", text, re.M):
    bad.append("run.sbatch sets --qos (rejected on Torch)")
if not (os.path.exists(log) and sha[:16] in open(log).read()):
    bad.append("smoke_cpu.log missing or from a different script version")
print("\n".join("PREFLIGHT: " + b for b in bad) or f"PREFLIGHT OK {sha}")
sys.exit(bool(bad))
