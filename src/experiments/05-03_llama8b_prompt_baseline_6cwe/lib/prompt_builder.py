"""Build chat-template messages for the prompt-engineering baseline.

Uniformly applies the Llama-3.1 chat template across all 6 CWEs (see architect §B).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
import datasets as ds


def build_user_text(prompt_row: dict, cwe: str, variant: str, user_prepend: str) -> str:
    """Build the user-message content for one (prompt, condition) combo."""
    field = ds.get_prompt_field(cwe, variant)
    raw = prompt_row[field]
    if ds.needs_completion_prefix(cwe, variant):
        body = ds.completion_prefix(cwe) + raw
    else:
        body = raw
    return user_prepend + body


def build_messages(prompt_row: dict, cwe: str, variant: str, condition: dict) -> list:
    """Return the list-of-dicts ready for tokenizer.apply_chat_template.

    condition: a dict from config.phrasings.CONDITIONS
    """
    user_text = build_user_text(prompt_row, cwe, variant, condition["user_prepend"])
    msgs = []
    if condition["system"] is not None:
        msgs.append({"role": "system", "content": condition["system"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def build_prompt_string(tokenizer, prompt_row: dict, cwe: str, variant: str, condition: dict) -> str:
    """Apply chat template to produce the final tokenizable string."""
    msgs = build_messages(prompt_row, cwe, variant, condition)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
