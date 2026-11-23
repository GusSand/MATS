#!/usr/bin/env python3
"""Render docs/mindmap.svg — goal-oriented mind map of this investigation.

Structure: root = the driving question; one branch per sub-question, colored by status
(green answered / amber partial / gray open); the first leaf is the current answer, the
following leaves are the evidence behind it.

MAINTENANCE RULE — this is enforced, not advisory. `check_project_state.py` re-renders from
BRANCHES and fails if `docs/mindmap.svg` differs, so the pre-commit hook catches a stale map
the same way it catches a stale living report. When a task closes, edit BRANCHES in the same
change set and run:

    python scripts/make_mindmap.py

`--check` renders without writing and exits 1 if the file on disk is stale. That is what the
project-state checker calls.

Only VERIFIED results become leaves. An unverified number does not go on the map.
"""

import sys
from pathlib import Path

GOAL = "GOAL — Which model components cause format-dependent failures such as 9.8 vs 9.11, and which interventions actually repair them?"
ROOT_TITLE = "MATS"
ROOT_SUB = "Even Heads (arXiv:2508.19414) and its v2, Locating Is Not Repairing"

# (sub-question, status: answered|partial|open, [answer, evidence...])
# First leaf = current answer (highlighted); the rest = the evidence behind it.
# Only VERIFIED results become leaves. Keep branch labels under ~38 characters
# or they collide with the leaf column.
BRANCHES = [
    ("Do L10 heads carry the repair?", "answered", [
        "NO — true-head prompt patch of all 32 heads fails (R13)",
        "prefill_right (primary): 0 of 1,307 configs correct",
        "prefill left-aligned: all 0, even-16 0, rand8 23/200",
        "old protocol re-run on A100: R8 1,000/1,000 exact",
        "verdicts identical on L40S and A100 (0 primary diffs)",
    ]),
    ("Do Mistral's head claims hold?", "answered", [
        "NO — no single head or subset repairs (R15)",
        "0 of 1,024 single heads; 0 of 524 subsets at L2/L17",
        "only layer 7 repairs; draft named 2, 17, 13, 10, 11",
        "5-layer repair not specific: 7/10 random sets too",
    ]),
    ("Do Pythia/Gemma have the effect?", "partial", [
        "Gemma base & IT: no format-dependent bug (R14)",
        "Pythia L6 true-head repair fails; Sept not reproduced",
        "Pythia old-protocol gate 7/10 (INBOX 2)",
    ]),
    ("What made the old head results?", "partial", [
        "post-W_O block patch + decode-step writes, jointly",
        "post_wo prompt-only: even-16 correct, all blocks not",
        "decode-step BOS writes alone: 0 of 1,307 correct",
        "pre_wo all-steps: all 32 correct, even-16 not",
    ]),
    ("Where does Llama decide 9.8 vs 9.11?", "partial", [
        "resid at the '11' token, layers 4-14 (r 0.55-0.85)",
        "read out by layer-14 attention into the answer",
        "' Which' -> 'Which' at layer 0 alone repairs",
        "open: specific content vs salience of '11'",
    ]),
    ("Does position 0 explain the GPU gap?", "answered", [
        "NO: A100 keep0 gives 119 of 120 R8, not 83",
        "A100 donor on L40S: 86 of 83, gap is receiver-side",
        "inject gate failed (4 of 1,466 coords): raw only",
        "placebo invalid on both GPUs: fragility not tested",
    ]),
]

STATUS = {
    "answered": dict(edge="#3f9e6e", fill="#e2f4e8", label="#1f6b45",
                     ans="#6fcf9e", glyph="✓", word="answered"),
    "partial":  dict(edge="#d99a3b", fill="#fbf0dc", label="#8a5a12",
                     ans="#e6b36b", glyph="◐", word="partial"),
    "open":     dict(edge="#8a8a92", fill="#f2f0e6", label="#4a4a42",
                     ans="#a9a9b2", glyph="○", word="open"),
}

BG = "#0b0b0e"
LEAF_TEXT = "#c9cbd4"
W = 2080
LEAF_H = 62          # vertical slot per leaf
BLOCK_GAP = 74       # gap between branch blocks
MARGIN = 70
ROOT_X, ROOT_W, ROOT_H = 48, 270, 92
BR_X = 560           # branch node left edge
LEAF_X = 1140        # leaf text left edge
UL_END = W - 60      # underline right edge
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"

OUT = Path(__file__).resolve().parent.parent / "docs" / "mindmap.svg"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def header(p):
    """Goal banner + status tally. Returns header height."""
    gy = 46
    p.append(
        f'<text x="{W / 2:.0f}" y="{gy}" text-anchor="middle" font-size="25" '
        f'font-weight="700" fill="#f0eee6">{esc(GOAL)}</text>')
    counts = {k: sum(1 for _, s, _ in BRANCHES if s == k) for k in STATUS}
    parts = []
    for k in ("answered", "partial", "open"):
        s = STATUS[k]
        parts.append(f'<tspan fill="{s["ans"]}">{s["glyph"]} {counts[k]} {s["word"]}</tspan>')
    sep = '<tspan fill="#55555c">   ·   </tspan>'
    p.append(
        f'<text x="{W / 2:.0f}" y="{gy + 32}" text-anchor="middle" '
        f'font-size="19">{sep.join(parts)}</text>')
    return gy + 56


def render():
    """Return the SVG text. Pure — writes nothing."""
    p = [None, None]  # svg open tag + bg, filled once height is known
    header_h = header(p)

    blocks, y = [], header_h + 20
    for label, status, leaves in BRANCHES:
        h = len(leaves) * LEAF_H
        blocks.append((label, status, leaves, y, h))
        y += h + BLOCK_GAP
    total_h = y - BLOCK_GAP + MARGIN
    root_cy = header_h + 20 + (total_h - header_h - 20 - MARGIN) / 2

    p[0] = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" '
        f'viewBox="0 0 {W} {total_h}" font-family="{FONT}">')
    p[1] = f'<rect width="{W}" height="{total_h}" fill="{BG}"/>'

    edges, nodes = [], []
    for label, status, leaves, by, bh in blocks:
        c = STATUS[status]
        node_label = f'{c["glyph"]} {label}'
        bcy = by + bh / 2
        bw = round(len(node_label) * 11.6 + 48)
        # root -> branch
        x0, x1 = ROOT_X + ROOT_W, BR_X
        mx = (x0 + x1) / 2
        edges.append(
            f'<path d="M {x0} {root_cy:.0f} C {mx:.0f} {root_cy:.0f} '
            f'{mx:.0f} {bcy:.0f} {x1} {bcy:.0f}" fill="none" '
            f'stroke="{c["edge"]}" stroke-width="2.2" stroke-opacity="0.85"/>')
        nodes.append(
            f'<rect x="{BR_X}" y="{bcy - 33:.0f}" width="{bw}" height="66" rx="15" '
            f'fill="{c["fill"]}" stroke="{c["edge"]}" stroke-width="1.5"/>'
            f'<text x="{BR_X + bw / 2:.0f}" y="{bcy + 8:.0f}" text-anchor="middle" '
            f'font-size="23" font-weight="600" fill="{c["label"]}">{esc(node_label)}</text>')
        # branch -> leaves (first leaf = the answer, highlighted)
        for i, leaf in enumerate(leaves):
            ly = by + i * LEAF_H + LEAF_H / 2
            uy = ly + 12
            x0b, x1b = BR_X + bw, LEAF_X - 8
            mxb = x0b + (x1b - x0b) * 0.55
            edges.append(
                f'<path d="M {x0b} {bcy:.0f} C {mxb:.0f} {bcy:.0f} '
                f'{mxb:.0f} {uy:.0f} {x1b} {uy:.0f}" fill="none" '
                f'stroke="{c["edge"]}" stroke-width="1.8" stroke-opacity="0.7"/>')
            if i == 0:
                fill, weight, size = c["ans"], 700, 21
            else:
                fill, weight, size = LEAF_TEXT, 400, 20
            nodes.append(
                f'<text x="{LEAF_X}" y="{ly + 5:.0f}" font-size="{size}" '
                f'font-weight="{weight}" fill="{fill}">{esc(leaf)}</text>'
                f'<line x1="{LEAF_X - 8}" y1="{uy:.0f}" x2="{UL_END}" y2="{uy:.0f}" '
                f'stroke="{c["edge"]}" stroke-width="1.4" stroke-opacity="0.55"/>')

    p += edges + nodes
    p.append(
        f'<rect x="{ROOT_X}" y="{root_cy - ROOT_H / 2:.0f}" width="{ROOT_W}" '
        f'height="{ROOT_H}" rx="16" fill="#f7f5ee"/>'
        f'<text x="{ROOT_X + ROOT_W / 2}" y="{root_cy - 6:.0f}" text-anchor="middle" '
        f'font-size="26" font-weight="700" fill="#1c1c1e">{esc(ROOT_TITLE)}</text>'
        f'<text x="{ROOT_X + ROOT_W / 2}" y="{root_cy + 26:.0f}" text-anchor="middle" '
        f'font-size="19" fill="#55555c">{esc(ROOT_SUB)}</text>')
    p.append("</svg>")
    return "\n".join(p)


def main():
    svg = render()
    if "--check" in sys.argv:
        if not OUT.exists():
            print(f"MINDMAP STALE — {OUT} does not exist; run python scripts/make_mindmap.py",
                  file=sys.stderr)
            return 1
        if OUT.read_text() != svg:
            print("MINDMAP STALE — docs/mindmap.svg does not match BRANCHES; "
                  "run python scripts/make_mindmap.py", file=sys.stderr)
            return 1
        print("MINDMAP OK")
        return 0
    OUT.write_text(svg)
    counts = {k: sum(1 for _, s, _ in BRANCHES if s == k) for k in STATUS}
    print(f"wrote {OUT} — {len(BRANCHES)} branches "
          f"({counts['answered']} answered, {counts['partial']} partial, {counts['open']} open)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
