"""Link the first mention of each benchmark, model, dataset and method in docs/reports/*.html.

Idempotent: a term that is already linked on a page is skipped. Text inside <svg>, <a>, <code>,
<title>, <h1>-<h3>, <summary>, the masthead and HTML comments is never touched.
TERMS below are shared defaults (common safety benchmarks, models and methods). A project adds or
overrides terms in docs/reports/terms.tsv, one "term<TAB>url" per line ("term<TAB>-" drops a default); check new URLs against the
project's refs.bib or the paper itself, never from memory.

    python scripts/link_report_terms.py
"""
import re
from pathlib import Path

REPORTS = Path(__file__).resolve().parent.parent / "docs" / "reports"

# Longest names first so "WildGuardMix" wins over "WildGuard".
TERMS = [
    ("Qwen2.5-72B-Instruct", "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct"),
    ("WildGuardMix", "https://huggingface.co/datasets/allenai/wildguardmix"),
    ("WildGuard", "https://arxiv.org/abs/2406.18495"),
    ("WildJailbreak", "https://arxiv.org/abs/2406.18510"),
    ("JailbreakTrigger", "https://arxiv.org/abs/2401.05561"),
    ("TrustLLM", "https://arxiv.org/abs/2401.05561"),
    ("XSTest", "https://arxiv.org/abs/2308.01263"),
    ("HarmBench", "https://arxiv.org/abs/2402.04249"),
    ("StrongREJECT", "https://arxiv.org/abs/2402.10260"),
    ("CoCoNot", "https://arxiv.org/abs/2407.12043"),
    ("URIAL", "https://arxiv.org/abs/2312.01552"),
    ("DAN", "https://arxiv.org/abs/2308.03825"),
    ("MMLU", "https://arxiv.org/abs/2009.03300"),
    ("GSM8K", "https://arxiv.org/abs/2110.14168"),
    ("IFEval", "https://arxiv.org/abs/2311.07911"),
    ("TracIn", "https://arxiv.org/abs/2002.08484"),
    ("Petri", "https://github.com/safety-research/petri"),
    ("safety-eval", "https://github.com/allenai/safety-eval"),
    ("Llama-3.1-8B", "https://huggingface.co/meta-llama/Llama-3.1-8B"),
    ("Tülu 3", "https://arxiv.org/abs/2411.15124"),
    ("OLMo 2", "https://arxiv.org/abs/2501.00656"),
    ("OLMo 3", "https://arxiv.org/abs/2512.13961"),
    ("DPO", "https://arxiv.org/abs/2305.18290"),
    ("RLVR", "https://arxiv.org/abs/2411.15124"),
    ("LoRA", "https://arxiv.org/abs/2106.09685"),
]
TERMS_FILE = REPORTS / "terms.tsv"
if TERMS_FILE.is_file():
    extra = dict(line.split("\t", 1) for line in TERMS_FILE.read_text().splitlines()
                 if line.strip() and not line.startswith("#"))
    # A url of "-" drops a shared default for this project.
    TERMS = [(t, u) for t, u in TERMS if t not in extra] + [(t, u.strip()) for t, u in extra.items() if u.strip() != "-"]
TERMS.sort(key=lambda tu: -len(tu[0]))
SKIP = {"svg", "a", "code", "title", "h1", "h2", "h3", "summary", "style", "script", "head"}
TOKEN = re.compile(r"(<!--.*?-->|<[^>]+>)", re.S)


def link_page(html: str) -> tuple[str, int]:
    done = {t for t, url in TERMS if f'href="{url}"' in html and re.search(rf">{re.escape(t)}</a>", html)}
    out, stack, n = [], [], 0
    masthead_depth = None
    for part in TOKEN.split(html):
        if not part:
            continue
        if part.startswith("<!--"):
            out.append(part)
            continue
        m = re.match(r"<(/?)([a-zA-Z0-9]+)", part)
        if m:
            closing, tag = m.group(1) == "/", m.group(2).lower()
            if not closing and not part.endswith("/>") and tag not in {"meta", "link", "br", "img", "input", "hr"}:
                stack.append(tag)
                if 'class="masthead"' in part and masthead_depth is None:
                    masthead_depth = len(stack)
            elif closing and tag in stack:
                while stack and stack.pop() != tag:
                    pass
                if masthead_depth and len(stack) < masthead_depth:
                    masthead_depth = None
            out.append(part)
            continue
        if any(t in SKIP for t in stack) or masthead_depth:
            out.append(part)
            continue
        text, anchors = part, []
        for term, url in TERMS:
            if term in done:
                continue
            # \x00k\x00 placeholders keep later (shorter) terms from matching inside an inserted link.
            pat = re.compile(rf"(?<![\w\-\x00]){re.escape(term)}(?!\w)")
            mm = pat.search(text)
            if mm:
                anchors.append(f'<a href="{url}" target="_blank" rel="noopener">{term}</a>')
                text = text[: mm.start()] + f"\x00{len(anchors) - 1}\x00" + text[mm.end():]
                done.add(term)
                n += 1
        for k, a in enumerate(anchors):
            text = text.replace(f"\x00{k}\x00", a, 1)
        out.append(text)
    return "".join(out), n


def legacy_pages() -> set[Path]:
    """Pages listed in index.html's report-legacy meta predate this kit and must not be edited."""
    index = REPORTS / "index.html"
    if not index.is_file():
        return set()
    m = re.search(r'<meta name="report-legacy" content="([^"]*)">', index.read_text())
    root = REPORTS.parent.parent
    return {(root / rel).resolve() for rel in m.group(1).split()} if m else set()


if __name__ == "__main__":
    legacy = legacy_pages()
    for p in sorted(REPORTS.rglob("*.html")):
        if p.name.startswith("_") or p.resolve() in legacy:
            continue
        new, n = link_page(p.read_text())
        if n:
            p.write_text(new)
        print(f"{p.relative_to(REPORTS)}: {n} links added")
