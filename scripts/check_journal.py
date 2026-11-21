#!/usr/bin/env python3
"""Fail when docs/research_journal.md falls behind docs/experiments/.

Two checks:
  1. Every Slurm job ID written in an experiment file also appears in the
     journal (the "every job gets one journal line" rule).
  2. Every experiment file that has run (it holds a job ID, or a Results /
     Job log section with content) is named in the journal by file name.

Exceptions go in docs/journal_allowlist.txt, one per line, with a reason:
    20261010    a date inside a path, not a job
    08-08_old_file.md    pre-checker; journaled as "Phase 0 baseline"
An entry without a reason is rejected.

    python scripts/check_journal.py            # working tree
    python scripts/check_journal.py --staged   # index (used by the pre-commit hook)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = "docs/research_journal.md"
EXPERIMENTS = "docs/experiments"
ALLOWLIST = "docs/journal_allowlist.txt"

# Torch Slurm job IDs are 8 digits, optionally with an array suffix (_3).
# Neighbours that are word characters, '.', '/' or '-' mean the digits are part
# of a path, hash, version or date, not a job ID.
JOB_RE = re.compile(r"(?<![\w./-])(\d{8})(?:_\d+)?(?![\w/-])")
RAN_HEADING = re.compile(r"^#{2,4} .*\b(Results?|Job log)\b", re.I | re.M)
PLACEHOLDER = re.compile(r"^\(?(none|after run|tbd|pending)\b", re.I)


def read(path, staged):
    if staged:
        r = subprocess.run(["git", "show", f":{path}"], cwd=ROOT,
                           capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    p = ROOT / path
    return p.read_text() if p.exists() else None


def experiment_files(staged):
    if staged:
        out = subprocess.run(["git", "ls-files", "--", EXPERIMENTS], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        return sorted(f for f in out.split() if f.endswith(".md"))
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / EXPERIMENTS).glob("*.md"))


def load_allowlist(staged, errors):
    allowed = {}
    for n, line in enumerate((read(ALLOWLIST, staged) or "").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, reason = line.partition(" ")
        if not reason.strip():
            errors.append(f"{ALLOWLIST}:{n}: '{key}' has no reason")
        allowed[key] = reason.strip()
    return allowed


def has_run_section(text):
    """A Results or Job log section with at least one non-placeholder line."""
    for m in RAN_HEADING.finditer(text):
        body = text[m.end():]
        nxt = re.search(r"^#{1,4} ", body, re.M)
        body = body[: nxt.start()] if nxt else body
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        if any(not PLACEHOLDER.match(l) for l in lines):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staged", action="store_true", help="check the git index, not the working tree")
    args = ap.parse_args()

    errors = []
    journal = read(JOURNAL, args.staged)
    if journal is None:
        print(f"JOURNAL CHECK: {JOURNAL} missing")
        return 1
    allowed = load_allowlist(args.staged, errors)

    for f in experiment_files(args.staged):
        text = read(f, args.staged) or ""
        name = Path(f).name
        ids = sorted(set(JOB_RE.findall(text)))
        for jid in ids:
            if jid not in journal and jid not in allowed:
                line = text[: text.find(jid)].count("\n") + 1
                errors.append(f"{f}:{line}: job {jid} is not in the journal")
        ran = bool(ids) or has_run_section(text)
        if ran and name not in journal and Path(f).stem not in journal and name not in allowed:
            errors.append(f"{f}: has results or jobs but no journal entry names it")

    if errors:
        print("\n".join(errors))
        print(f"JOURNAL BEHIND: {len(errors)} problem(s). Add journal lines "
              f"(or an allowlist entry with a reason in {ALLOWLIST}).")
        return 1
    print("JOURNAL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
