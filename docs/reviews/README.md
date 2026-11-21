# Independent reviews

One file per review, written by a fresh subagent that did not write the work. `scripts/check_reviews.py`
(run by the pre-commit hook) requires them, and checks each new or changed one against the subagent's
Claude Code transcript.

- Design review, before any compute: `<experiment-stem>_design.md` (add `_v2`, `_amend3`… for later rounds).
- Verification, before a VERIFIED journal entry: `<experiment-stem>_verify.md` (same suffix rule).

## How to write one

Never type a report by hand. When the subagent has finished:

    python scripts/check_reviews.py --list-agents
    python scripts/check_reviews.py --save-report <agent-id> docs/reviews/<stem>_verify.md --verdict "<the agent's own verdict words>"

This writes the header (`Reviewer`, `Agent-ID`, `Verdict`, `Transcript-SHA256`), the `## Brief given`
section (the agent's first message) and the `## Report` section (its final message), both verbatim, and
keeps a transcript copy in `outputs/review_transcripts/` (local, git-ignored). The doer may add a
`## Doer response` section below (what was fixed, what was declined and why). Nothing else may change.

The hook fails when: the report or brief differs from the transcript; the header verdict is not in the
agent's own words; the agent edited repo files outside `docs/reviews/`, `outputs/` or `scripts/tmp/`, or
ran git commit/push (so a doer subagent cannot review its own work); or the transcript hash differs.
Transcripts live only on the machine that ran the session, so other clones skip these checks with a
warning.

## Brief rules

- Give the reviewer the files and the pre-registered criteria, not your conclusions or your result scripts.
- Ask it not to quote raw harmful prompts or responses: IDs and paraphrase only. Reports are committed.

## Linking from the experiment file

    **Design review:** FIXED — fresh subagent, 2026-10-31 — docs/reviews/<stem>_design.md
    **Verification:** VERIFIED-WITH-FLAGS — fresh subagent, 2026-10-31 — docs/reviews/<stem>_verify.md

Design-review depth: the full adversarial checklist (research-workflow, "Design review") for runs over
~20 GPU-h; for smaller runs at least checklist items 1, 2, 4 and 5 plus "is the matching control
pre-registered?".
