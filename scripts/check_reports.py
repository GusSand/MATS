"""Check that docs/reports/ is fresh against the files it cites.

Each page ends with a SOURCES comment that maps numbers to repo files:
    <!-- SOURCES: number → file:line
    92.2 → results/summary.md:22 ; docs/experiments/08-09_x.md:40
    -->
`--stamp` hashes every cited file into a meta tag in the page head:
    <meta name="report-sources" content="results/summary.md@1a2b3c4d5e ...">
The check then fails when:
  - a cited file changed after the page was stamped, or is gone;
  - the SOURCES comment cites a file that the meta tag does not hold (cited but never stamped);
  - a docs/experiments/*.md file is cited by no page and is not in index.html's
    <meta name="report-excluded" content="docs/experiments/a.md ...">;
  - a page holds a private path or links to a local page that does not exist.
Pages listed in index.html's <meta name="report-legacy" content="docs/reports/a.html ..."> are reports
that predate this kit (e.g. a frozen dated snapshot). They may not be edited to add a SOURCES comment,
so only that one check is skipped for them; the private-path and dead-link checks still apply.
The journal and decisions.md are never hashed: new entries go at the bottom, so old entries and the
line numbers that cite them do not change. BACKLOG.md and INBOX.md ARE hashed: new items go at the top,
which shifts every line below, so a line citation into them can go stale silently. Hashes, not dates: a page touched without being reconciled
still fails. This catches staleness only; it does not check that a page is correct.

    python scripts/check_reports.py                       # exit 1 on any problem
    python scripts/check_reports.py --stamp docs/reports/q05_steering.html [...]
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "docs" / "reports"
EXPERIMENTS = ROOT / "docs" / "experiments"
APPEND_ONLY = {"docs/research_journal.md", "docs/decisions.md"}  # bottom-appended only
PRIVATE = re.compile(r"/Users/|/home/|/scratch/|\$SCRATCH|/private/tmp|/state/partition")
SOURCES = re.compile(r"<!--\s*SOURCES(.*?)-->", re.S)
PATH = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.(?:md|json|jsonl|csv|tsv|txt|py|yaml|yml|tex|bib|svg|html))")
META = re.compile(r'<meta name="report-sources" content="([^"]*)">\n?')


def pages():
    return [p for p in sorted(REPORTS.rglob("*.html")) if not p.name.startswith("_")]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def cited(html: str) -> set[str]:
    out = set()
    for block in SOURCES.findall(html):
        for p in PATH.findall(block):
            if p not in APPEND_ONLY and (ROOT / p).is_file():
                out.add(p)
    return out


def stamped(html: str) -> dict[str, str]:
    m = META.search(html)
    if not m:
        return {}
    return dict(item.rsplit("@", 1) for item in m.group(1).split())


def stamp(page: Path) -> None:
    html = page.read_text()
    tag = '<meta name="report-sources" content="{}">\n'.format(
        " ".join(f"{p}@{digest(ROOT / p)}" for p in sorted(cited(html))))
    if META.search(html):
        html = META.sub(lambda _: tag, html, count=1)
    elif "</title>" in html:
        html = html.replace("</title>\n", "</title>\n" + tag, 1)
    else:
        sys.exit(f"{page}: no <title> line to put the meta tag after")
    page.write_text(html)
    print(f"stamped {page.relative_to(ROOT)}: {len(cited(html))} sources")


def index_meta(name: str) -> set[str]:
    index = REPORTS / "index.html"
    if not index.is_file():
        return set()
    m = re.search(rf'<meta name="{name}" content="([^"]*)">', index.read_text())
    return set(m.group(1).split()) if m else set()


def check() -> list[str]:
    problems, covered = [], set()
    legacy = index_meta("report-legacy")
    for page in pages():
        rel, html = page.relative_to(ROOT), page.read_text()
        cites, stamps = cited(html), stamped(html)
        # A cited-but-unstamped file is already reported below, so count it as covered here too
        # (otherwise every new page's experiment files are reported twice before stamping).
        covered |= cites | set(stamps)
        if not SOURCES.search(html) and str(rel) not in legacy:
            problems.append(f"{rel}: no <!-- SOURCES --> comment")
        for p in sorted(cites - set(stamps)):
            problems.append(f"{rel}: cites {p} but was never stamped against it")
        for p, h in sorted(stamps.items()):
            if not (ROOT / p).is_file():
                problems.append(f"{rel}: source {p} is gone")
            elif digest(ROOT / p) != h:
                problems.append(f"{rel}: STALE, {p} changed since the page was reconciled")
        for m in PRIVATE.finditer(html):
            problems.append(f"{rel}: private path {m.group(0)!r} at char {m.start()}")
        for href in re.findall(r'href="([^"#:]+\.html)', html):
            if not (page.parent / href).is_file():
                problems.append(f"{rel}: links to missing page {href}")
    excluded = index_meta("report-excluded")
    for exp in sorted(EXPERIMENTS.glob("*.md")):
        p = str(exp.relative_to(ROOT))
        if p not in covered and p not in excluded:
            problems.append(f"{p}: no report page cites it (add it to a page, or to report-excluded on index.html)")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", nargs="+", type=Path, metavar="PAGE")
    args = ap.parse_args()
    if args.stamp:
        for page in args.stamp:
            stamp(page.resolve())
    if not (REPORTS / "index.html").is_file():
        print("REPORTS: none yet (no docs/reports/index.html); nothing to check")
        sys.exit(0)
    problems = check()
    for line in problems:
        print(line)
    print("REPORTS OK" if not problems else f"REPORTS STALE: {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)
