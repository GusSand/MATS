"""Prepare docs/reports/ for publishing as one private multi-file Artifact.

The Artifact's main page must not carry its own <!doctype>/<html>/<head>/<body> wrapper, so this
writes a stripped copy of index.html to --out, and copies every other page and asset there as-is
(the Artifact tool only reads files under the session's working directory or scratchpad, so one
self-contained folder works from any session). Runs check_reports.py first and refuses to continue
if it fails.

    python scripts/publish_reports.py --out <scratchpad>/publish_<project>
Then publish with the Artifact tool: file_path=<out>/index.html, root=<out>, files = the list this
prints, and url = the URL in docs/reports/README.md (omit url only on the first publish).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "docs" / "reports"
WRAPPER = re.compile(r'^(<!doctype html>|<html[^>]*>|</?head>|</?body>|</html>|<meta charset[^>]*>|<meta name="viewport"[^>]*>)\s*$',
                     re.I)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if subprocess.run([sys.executable, str(ROOT / "scripts" / "check_reports.py")]).returncode:
        sys.exit("check_reports.py failed; fix the reports before publishing")
    lines = (REPORTS / "index.html").read_text().splitlines()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text("\n".join(l for l in lines if not WRAPPER.match(l)) + "\n")
    files = sorted(str(p.relative_to(REPORTS)) for p in REPORTS.rglob("*")
                   if p.is_file() and p.suffix in {".html", ".css", ".svg", ".png"}
                   and p != REPORTS / "index.html" and not p.name.startswith("_"))
    for f in files:
        (args.out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPORTS / f, args.out / f)
        # Pages other than the main one are served as written; one without a doctype (e.g. a frozen
        # legacy snapshot) would render in quirks mode, so the published copy gets one,
        # with a charset and viewport.
        if f.endswith(".html") and not (args.out / f).read_text()[:200].lstrip().lower().startswith("<!doctype"):
            (args.out / f).write_text('<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n'
                                    + (args.out / f).read_text())
    print(f"wrote {args.out}")
    print(json.dumps(files))
