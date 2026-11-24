# scripts

Only scripts needed to reproduce results live here. Temporary ones go in `scripts/tmp/` (git-ignored).

| Script | What it does |
| --- | --- |
| `make_mindmap.py` | Renders `docs/mindmap.svg` from its `BRANCHES` table. `--check` exits 1 if the SVG is stale. Only verified results become leaves. |
| `check_reports.py` | Freshness gate for `docs/reports/`. `--stamp PAGE` hashes the files a page's SOURCES comment cites; the check exits 1 when one changes later, when an experiment file is cited by no page, or on a private path or dead link. |
| `publish_reports.py` | `--out DIR`: runs the check, writes the stripped `index.html` for the Artifact tool, prints the `files` list. |
| `link_report_terms.py` | Links the first mention of each benchmark, model and method per report page. Shared defaults plus `docs/reports/terms.tsv`. Idempotent. |
| `check_journal.py` | Journal freshness gate. Exits 1 when a Slurm job ID in `docs/experiments/*.md` is missing from the journal, or an experiment file with jobs or results is not named in it. `--staged` checks the git index. Exceptions: `docs/journal_allowlist.txt`, each with a reason. |
| `check_reviews.py` | Independent-review gate. A job ID or result needs a `**Design review:**` line linking a saved report in `docs/reviews/`; a VERIFIED journal entry needs `**Verification:**` lines in the files it names. Format: `docs/reviews/README.md`. Each new report is checked against its subagent's transcript (verbatim brief and report, own verdict, no edits to the reviewed work). `--save-report <agent-id> <out> --verdict "…"` writes one; `--list-agents` lists recent subagents. |
| `hooks/pre-commit` | Runs `check_journal.py --staged` and `check_reviews.py --staged` on every commit. `new-project.sh` installs it; in an existing clone run `git config core.hooksPath scripts/hooks`. |
| `dry_run_check.py` | Preflight for one experiment directory: committed, pushed, sbatch parses, the last design review is PASS (not FIXED) and names the script hash, extra files equal the sbatch `# Needs:` lines, smoke log matches the script. |
| `submit.sh` | The only way to submit: preflight, copy to Torch without `--delete`, compare hashes, `sbatch`. |
