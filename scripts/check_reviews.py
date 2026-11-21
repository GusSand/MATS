#!/usr/bin/env python3
"""Fail when an experiment ran or was called verified without an independent review on file.

The doer never signs off its own work. Three checks:
  1. Design review before compute. An experiment file that holds a Slurm job
     ID or a filled Results / Job log section needs a line
         **Design review:** PASS|FIXED — fresh subagent, YYYY-MM-DD — docs/reviews/<name>.md
     whose last occurrence says PASS or FIXED (REJECTED blocks), and the linked
     report must exist.
  2. Verification before a finding. A journal entry below the gate marker
         <!-- review-gate: entries below need saved verifier reports -->
     whose heading says VERIFIED (any VERIFIED-WITH-... form) must name its
     experiment files, and each one needs a line
         **Verification:** VERIFIED|VERIFIED-WITH-FLAGS — fresh subagent, YYYY-MM-DD — docs/reviews/<name>.md
     with an existing report.
  3. The report is the reviewer's, verbatim. Every report added or changed (in
     the index with --staged, else against HEAD) names an Agent-ID, and the
     Claude Code transcript of that subagent must show: the "## Brief given"
     section is the agent's first message; the "## Report" section is its final
     message; the header Verdict appears in that report; the agent made no
     Edit/Write outside docs/reviews/, outputs/, scripts/tmp/ and ran no git
     commit/push; the Transcript-SHA256 header matches the copy kept in
     outputs/review_transcripts/. On a machine without ~/.claude/projects the
     transcript checks are skipped with a warning.

Write a report from a finished subagent (extracts brief and report verbatim,
keeps a transcript copy, fills the hash):
    python scripts/check_reviews.py --list-agents
    python scripts/check_reviews.py --save-report <agent-id> docs/reviews/<stem>_verify.md --verdict "VERIFIED-WITH-FLAGS"
Then add a "## Doer response" section below the report if needed.

Exceptions go in docs/review_allowlist.txt, one per line, with a reason:
  - a job ID exempts that one job from check 1;
  - an experiment file name exempts the file's no-job results from check 1
    (new job IDs in it are still checked).
An entry without a reason is rejected.

    python scripts/check_reviews.py            # working tree
    python scripts/check_reviews.py --staged   # index (used by the pre-commit hook)
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = "docs/research_journal.md"
EXPERIMENTS = "docs/experiments"
REVIEWS = "docs/reviews"
ALLOWLIST = "docs/review_allowlist.txt"
COPIES = "outputs/review_transcripts"
MARKER = "<!-- review-gate: entries below need saved verifier reports -->"
TRANSCRIPTS = Path.home() / ".claude" / "projects"
REVIEWER_MAY_WRITE = ("docs/reviews/", "outputs/", "scripts/tmp/")

# Same job-ID rule as check_journal.py: 8-digit Torch Slurm IDs, optional array suffix.
JOB_RE = re.compile(r"(?<![\w./-])(\d{8})(?:_\d+)?(?![\w/-])")
RAN_HEADING = re.compile(r"^#{2,4} .*\b(Results?|Job log)\b", re.I | re.M)
PLACEHOLDER = re.compile(r"^\(?(none|after run|tbd|pending)\b", re.I)
DESIGN_LINE = re.compile(r"^\*\*Design review:\*\*\s*(PASS|FIXED|REJECTED)\b.*?(docs/reviews/[\w./-]+\.md)", re.M)
VERIFY_LINE = re.compile(r"^\*\*Verification:\*\*\s*(VERIFIED(?:-WITH-[A-Z]+)?|FAILED|UNVERIFIED)\b.*?(docs/reviews/[\w./-]+\.md)", re.M)
VERIFIED_HEADING = re.compile(r"(?<!UN)VERIFIED")
EXP_NAME = re.compile(r"\b(\d\d-\d\d_[\w-]*\w(?:\.(?!md\b)[\w-]*\w)*)")  # full stem; drops .md and a trailing dot


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def read(path, staged):
    if staged:
        r = git("show", f":{path}")
        return r.stdout if r.returncode == 0 else None
    p = ROOT / path
    return p.read_text() if p.exists() else None


def exists(path, staged):
    if staged:
        return git("cat-file", "-e", f":{path}").returncode == 0
    return (ROOT / path).is_file()


def experiment_files(staged):
    if staged:
        out = git("ls-files", "--", EXPERIMENTS).stdout
        return sorted(f for f in out.split() if f.endswith(".md"))
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / EXPERIMENTS).glob("*.md"))


def changed_reports(staged):
    if staged:
        out = git("diff", "--cached", "--name-only", "--diff-filter=AM", "--", REVIEWS).stdout.split()
    else:
        out = git("diff", "HEAD", "--name-only", "--diff-filter=AM", "--", REVIEWS).stdout.split()
        out += git("ls-files", "--others", "--exclude-standard", "--", REVIEWS).stdout.split()
    return sorted({f for f in out if f.endswith(".md") and not f.endswith("README.md")})


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
    for m in RAN_HEADING.finditer(text):
        body = text[m.end():]
        nxt = re.search(r"^#{1,4} ", body, re.M)
        body = body[: nxt.start()] if nxt else body
        if any(not PLACEHOLDER.match(l.strip()) for l in body.splitlines() if l.strip()):
            return True
    return False


# ---- transcripts -----------------------------------------------------------

def norm(s):
    return " ".join((s or "").split())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_transcript(aid):
    """The live Claude Code transcript, else the copy kept in outputs/."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
    for pattern in (f"{slug}/*/subagents/agent-{aid}.jsonl", f"*/*/subagents/agent-{aid}.jsonl"):
        hits = sorted(TRANSCRIPTS.glob(pattern))
        if hits:
            return hits[0]
    copy = ROOT / COPIES / f"agent-{aid}.jsonl"
    return copy if copy.is_file() else None


def blocks(rec):
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return c or []


def parse_transcript(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    brief, final, writes, git_ops = None, None, [], []
    for r in recs:
        if r.get("type") == "user" and brief is None:
            texts = [b.get("text", "") for b in blocks(r) if b.get("type") == "text"]
            if texts:
                brief = "\n".join(texts)
        if r.get("type") != "assistant":
            continue
        for b in blocks(r):
            if b.get("type") == "text" and b.get("text", "").strip():
                final = b["text"]
            if b.get("type") != "tool_use":
                continue
            name, inp = b.get("name"), b.get("input") or {}
            if name == "SubagentHandback":
                final = inp.get("message", final)
            elif name in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
                writes.append(inp.get("file_path") or inp.get("notebook_path") or "")
            elif name == "Bash" and re.search(r"\bgit\s+(commit|push)\b", inp.get("command", "")):
                git_ops.append(inp["command"][:80])
    return brief, final, writes, git_ops


def sections(text):
    """Split on the three fixed headings only; reports often hold '## ' headings of their own."""
    head, _, rest = text.partition("\n## Brief given\n")
    brief, _, rest = rest.partition("\n## Report\n")
    i = rest.rfind("\n## Doer response\n")
    report, doer = (rest[:i], rest[i:]) if i != -1 else (rest, "")
    return {"_head": head, "Brief given": brief, "Report": report, "Doer response": doer}


def header(text, key):
    m = re.search(rf"^{key}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def check_report_provenance(path, staged, errors, warnings):
    text = read(path, staged) or ""
    sec = sections(text)
    aid = header(sec["_head"], "Agent-ID")
    verdict = header(sec["_head"], "Verdict")
    if not aid:
        errors.append(f"{path}: no 'Agent-ID:' header; write it with --save-report <agent-id>")
        return
    if not TRANSCRIPTS.is_dir() and not (ROOT / COPIES / f"agent-{aid}.jsonl").is_file():
        warnings.append(f"{path}: no transcripts on this machine; agent {aid} not checked")
        return
    tpath = find_transcript(aid)
    if tpath is None:
        errors.append(f"{path}: no transcript for agent {aid} (not in ~/.claude/projects or {COPIES})")
        return
    brief, final, writes, git_ops = parse_transcript(tpath)
    if norm(sec.get("Brief given")) != norm(brief):
        errors.append(f"{path}: '## Brief given' differs from agent {aid}'s first message")
    if norm(sec.get("Report")) != norm(final):
        errors.append(f"{path}: '## Report' differs from agent {aid}'s final message (must be verbatim)")
    if not verdict or norm(verdict).lower() not in norm(final).lower():
        errors.append(f"{path}: header 'Verdict: {verdict}' does not appear in the agent's own report")
    for w in sorted(set(writes)):
        rel = Path(w).resolve()
        try:
            rel = str(rel.relative_to(ROOT))
        except ValueError:
            continue  # outside the repo (scratchpad): allowed
        if not rel.startswith(REVIEWER_MAY_WRITE):
            errors.append(f"{path}: reviewer {aid} edited {rel}; a reviewer may not change the work it reviews")
    for g in git_ops:
        errors.append(f"{path}: reviewer {aid} ran '{g}'")
    copy = ROOT / COPIES / f"agent-{aid}.jsonl"
    want = header(sec["_head"], "Transcript-SHA256")
    if not copy.is_file():
        errors.append(f"{path}: transcript copy {COPIES}/agent-{aid}.jsonl missing; rerun --save-report")
    elif want != sha256(copy):
        errors.append(f"{path}: Transcript-SHA256 does not match {COPIES}/agent-{aid}.jsonl")


def list_agents():
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
    rows = []
    for meta in TRANSCRIPTS.glob(f"{slug}/*/subagents/agent-*.meta.json"):
        m = json.loads(meta.read_text())
        aid = meta.name[len("agent-"):-len(".meta.json")]
        rows.append((meta.stat().st_mtime, aid, m.get("finished"), m.get("description", "")))
    for _, aid, done, desc in sorted(rows)[-30:]:
        print(f"{aid}  {'done   ' if done else 'running'}  {desc}")


def save_report(aid, out, verdict):
    tpath = find_transcript(aid)
    if tpath is None:
        sys.exit(f"no transcript for agent {aid}")
    meta = tpath.with_name(f"agent-{aid}.meta.json")
    if meta.is_file() and not json.loads(meta.read_text()).get("finished"):
        sys.exit(f"agent {aid} has not finished")
    brief, final, _, _ = parse_transcript(tpath)
    if not final:
        sys.exit(f"agent {aid} has no final message")
    if norm(verdict).lower() not in norm(final).lower():
        sys.exit(f"'{verdict}' does not appear in the agent's report; use the agent's own verdict words")
    copy = ROOT / COPIES / f"agent-{aid}.jsonl"
    copy.parent.mkdir(parents=True, exist_ok=True)
    if tpath != copy:
        shutil.copyfile(tpath, copy)
    out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"Reviewer: fresh subagent (Claude Code)\nAgent-ID: {aid}\nVerdict: {verdict}\n"
        f"Transcript-SHA256: {sha256(copy)}\n\n## Brief given\n\n{brief.strip()}\n\n"
        f"## Report\n\n{final.strip()}\n")
    print(f"wrote {out.relative_to(ROOT)}; transcript copy {COPIES}/agent-{aid}.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staged", action="store_true", help="check the git index, not the working tree")
    ap.add_argument("--list-agents", action="store_true", help="list this repo's recent subagents")
    ap.add_argument("--save-report", nargs=2, metavar=("AGENT_ID", "OUT"), help="write a report from a transcript")
    ap.add_argument("--verdict", help="with --save-report: the agent's own verdict words")
    args = ap.parse_args()
    if args.list_agents:
        return list_agents()
    if args.save_report:
        if not args.verdict:
            sys.exit("--save-report needs --verdict")
        return save_report(*args.save_report, args.verdict)

    errors, warnings = [], []
    allowed = load_allowlist(args.staged, errors)
    files = experiment_files(args.staged)
    texts = {f: read(f, args.staged) or "" for f in files}

    def report_exists(report, where):
        if not exists(report, args.staged):
            errors.append(f"{where}: review report {report} does not exist")

    # 1. Design review before compute or results.
    for f, text in texts.items():
        name = Path(f).name
        ids = sorted(set(JOB_RE.findall(text)))
        new_ids = [i for i in ids if i not in allowed]
        results = has_run_section(text) and name not in allowed
        if not new_ids and not results:
            continue
        reviews = DESIGN_LINE.findall(text)
        if not reviews:
            what = f"jobs {', '.join(new_ids)}" if new_ids else "results"
            errors.append(f"{f}: has {what} but no '**Design review:**' line")
            continue
        verdict, report = reviews[-1]
        if verdict == "REJECTED":
            errors.append(f"{f}: latest design review is REJECTED; fix and re-review before running")
        report_exists(report, f)

    # 2. Verification before a VERIFIED journal entry.
    journal = read(JOURNAL, args.staged) or ""
    if MARKER in journal:
        tail = journal.split(MARKER, 1)[1]
        for block in re.split(r"^(?=## )", tail, flags=re.M):
            heading = block.split("\n", 1)[0]
            if not heading.startswith("## ") or not VERIFIED_HEADING.search(heading):
                continue
            named = {Path(f).stem for f in files} & set(EXP_NAME.findall(block))
            if not named:
                errors.append(f"{JOURNAL}: entry '{heading[3:70]}' says VERIFIED but names no experiment file")
            for stem in sorted(named):
                f = f"{EXPERIMENTS}/{stem}.md"
                checks = [r for v, r in VERIFY_LINE.findall(texts[f]) if v.startswith("VERIFIED")]
                if not checks:
                    errors.append(f"{f}: journal entry '{heading[3:70]}' says VERIFIED but the file has no "
                                  f"'**Verification:** VERIFIED…' line")
                for r in checks:
                    report_exists(r, f)
    else:
        errors.append(f"{JOURNAL}: gate marker missing; append the line: {MARKER}")

    # 3. Each new or changed report is the named subagent's own, verbatim.
    for r in changed_reports(args.staged):
        check_report_provenance(r, args.staged, errors, warnings)

    for w in warnings:
        print("warning:", w)
    if errors:
        print("\n".join(errors))
        print(f"REVIEWS MISSING: {len(errors)} problem(s). A fresh subagent that did not write the work "
              f"must review it; save its report with --save-report (exceptions: {ALLOWLIST}, with a reason).")
        return 1
    print("REVIEWS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
