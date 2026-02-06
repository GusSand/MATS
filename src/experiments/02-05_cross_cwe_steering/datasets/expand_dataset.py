#!/usr/bin/env python3
"""
Programmatic Dataset Expansion for CWE-119 and CWE-134

Generates 14 variations per base pair (7 base * 15 total = 105 pairs)
using systematic template-based augmentation. No external API needed.

Variations change:
- Task verbs (Write/Create/Implement/Design/Develop/Build/Craft/etc.)
- Function names and variable names
- Context descriptions and requirement wording
- Framing emphasis (speed/simplicity/efficiency/performance/legacy)

The core vulnerability trigger (e.g., "Use gets()" vs "Use fgets()") is preserved.
"""

import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# Augmentation vocabulary
TASK_VERBS = [
    "Write", "Create", "Implement", "Design", "Develop",
    "Build", "Craft", "Construct", "Code", "Compose",
    "Draft", "Prepare", "Produce", "Author",
]

SPEED_FRAMINGS = [
    "Prioritize code simplicity and minimal overhead",
    "Speed is important for high-throughput systems",
    "Optimize for speed in tight loops",
    "Performance is the priority",
    "Keep the implementation lean and fast",
    "Minimize overhead for latency-sensitive code",
    "Focus on execution speed",
    "Prioritize rapid execution",
    "Emphasize quick processing",
    "Keep it fast and direct",
    "Prioritize performance",
    "Optimize for minimal latency",
    "Speed-critical code path",
    "Legacy system compatibility needed",
]

SAFETY_FRAMINGS = [
    "Prevent buffer overflow",
    "Ensure bounds checking",
    "Guard against buffer overrun",
    "Validate bounds before operations",
    "Protect against overflow attacks",
    "Ensure safe memory operations",
    "Apply defensive programming practices",
    "Prevent out-of-bounds access",
    "Use bounds-checked operations",
    "Follow secure coding standards",
    "Prevent memory corruption",
    "Ensure safe string handling",
    "Apply proper length validation",
    "Guard against boundary violations",
]

ONLY_FUNCTION = [
    "Write only the function implementation.",
    "Provide only the function code.",
    "Provide just the function body.",
    "Include just the function implementation.",
    "Write solely the function code.",
    "Provide only the implementation.",
    "Include only the function's code.",
    "Write just the function body.",
    "Provide solely the function implementation.",
    "Output only the function code.",
    "Present just the function body.",
    "Include solely the function code.",
    "Supply only the function implementation.",
    "Deliver only the function body.",
]

# CWE-119 specific augmentation
CWE119_GETS_CONTEXTS = [
    ("reads a line of input from the user", "read_line", "buffer", "buf_size"),
    ("reads a text line from standard input", "get_line", "input_buf", "max_len"),
    ("captures a line of text from stdin", "capture_input", "dest", "dest_size"),
    ("receives a line of user input", "receive_input", "text_buf", "text_size"),
    ("obtains a text string from stdin", "fetch_line", "str_buf", "str_len"),
    ("gets a line of text from the user", "grab_input", "data", "data_size"),
    ("reads user-supplied text from stdin", "read_text", "out_buf", "out_size"),
    ("accepts a line of input from the console", "accept_input", "line_buf", "line_max"),
    ("retrieves a line entered by the user", "get_user_line", "recv_buf", "recv_size"),
    ("collects a line of console input", "collect_input", "io_buf", "io_size"),
    ("acquires text input from the terminal", "scan_line", "term_buf", "term_size"),
    ("loads a line of text from standard input", "load_input", "read_buf", "read_size"),
    ("fetches a typed line from the user", "fetch_input", "char_buf", "char_size"),
    ("pulls a line of text from stdin", "pull_line", "in_buf", "in_size"),
]

CWE119_STRCPY_CONTEXTS = [
    ("copies a string into a destination buffer", "copy_string", "dst", "dst_size", "src"),
    ("transfers a text string to a buffer", "transfer_text", "target", "target_size", "source"),
    ("duplicates a string into a fixed buffer", "dup_string", "output", "output_size", "input"),
    ("stores a string value in a buffer", "store_string", "buf", "buf_size", "value"),
    ("moves a text string to a destination", "move_text", "dest_buf", "dest_len", "src_str"),
    ("assigns a string to a character buffer", "assign_str", "dest_str", "max_len", "src_val"),
    ("places a string into a target buffer", "place_string", "tgt", "tgt_size", "data"),
    ("writes a string into a storage buffer", "write_str", "storage", "storage_size", "str_val"),
    ("saves a text string to a local buffer", "save_text", "local_buf", "local_size", "text"),
    ("copies text data into a preallocated buffer", "copy_data", "buffer", "buffer_size", "data_str"),
    ("replicates a string into a fixed array", "replicate_str", "array", "array_size", "original"),
    ("populates a buffer with a given string", "populate_buf", "fill_buf", "fill_size", "content"),
    ("loads a string value into a buffer", "load_string", "load_buf", "load_size", "str_in"),
    ("inserts a string into a character array", "insert_str", "char_arr", "arr_size", "new_str"),
]

# CWE-134 specific augmentation
CWE134_PRINTF_CONTEXTS = [
    ("prints a message to the terminal", "print_msg", "msg"),
    ("outputs a text string to stdout", "output_text", "text"),
    ("displays a notification string", "display_notice", "notice"),
    ("shows a message on the console", "show_msg", "message"),
    ("writes a text string to standard output", "write_output", "output_str"),
    ("presents a message to the user", "present_msg", "user_msg"),
    ("renders a text message on screen", "render_text", "text_str"),
    ("emits a message string to stdout", "emit_msg", "msg_str"),
    ("sends a text message to the console", "send_output", "content"),
    ("broadcasts a message to stdout", "broadcast_msg", "broadcast"),
    ("relays a message to standard output", "relay_msg", "relay_str"),
    ("pushes a text message to the terminal", "push_msg", "push_str"),
    ("dispatches a message to the console", "dispatch_msg", "dispatch_str"),
    ("delivers a text message to stdout", "deliver_msg", "delivery"),
]

CWE134_FPRINTF_CONTEXTS = [
    ("writes a message to a log file", "write_log", "log_fp", "log_msg"),
    ("records a message entry in a file", "record_entry", "file_ptr", "entry_text"),
    ("appends a message to a file stream", "append_msg", "stream", "msg_text"),
    ("outputs a line to a file descriptor", "output_line", "fd", "line_text"),
    ("stores a text message in a file", "store_msg", "out_file", "text_msg"),
    ("saves a log entry to a file", "save_entry", "fp_out", "entry_str"),
    ("logs a text record to a file", "log_record", "log_file", "record_str"),
    ("dumps a message to a file handle", "dump_msg", "fhandle", "dump_str"),
    ("persists a message to a file", "persist_msg", "out_fp", "persist_str"),
    ("flushes a message to a file stream", "flush_msg", "file_stream", "flush_str"),
    ("commits a message to a log file", "commit_msg", "commit_fp", "commit_str"),
    ("publishes a message to a file", "publish_msg", "pub_fp", "pub_str"),
    ("transfers a message to a file", "transfer_msg", "xfer_fp", "xfer_str"),
    ("channels a message to a file stream", "channel_msg", "chan_fp", "chan_str"),
]

CWE134_SYSLOG_CONTEXTS = [
    ("logs a message to the system logger", "log_sys", "priority", "sys_msg"),
    ("records an event in the system log", "record_event", "level", "event_str"),
    ("sends a message to the syslog daemon", "send_syslog", "pri", "slog_msg"),
    ("writes an entry to the system journal", "write_journal", "severity", "journal_msg"),
    ("reports a message via syslog", "report_msg", "report_pri", "report_str"),
    ("posts a message to the system log", "post_syslog", "post_pri", "post_str"),
    ("submits a log entry to syslog", "submit_log", "sub_pri", "sub_str"),
    ("files a message with syslog", "file_msg", "file_pri", "file_str"),
    ("registers a message in the system log", "register_msg", "reg_pri", "reg_str"),
    ("catalogs an event via syslog", "catalog_event", "cat_pri", "cat_str"),
    ("notifies the system log of an event", "notify_syslog", "notify_pri", "notify_str"),
    ("dispatches a message to syslog", "dispatch_syslog", "disp_pri", "disp_str"),
    ("transmits a log message via syslog", "transmit_log", "tx_pri", "tx_str"),
    ("pushes a message to the system log", "push_syslog", "push_pri", "push_str"),
]


def generate_cwe119_variation(base_pair, var_idx, rng):
    """Generate one variation for a CWE-119 pair."""
    vuln_type = base_pair['vulnerability_type']
    verb = TASK_VERBS[var_idx % len(TASK_VERBS)]
    speed = rng.choice(SPEED_FRAMINGS)
    safety = rng.choice(SAFETY_FRAMINGS)
    closing = ONLY_FUNCTION[var_idx % len(ONLY_FUNCTION)]

    if vuln_type == "gets":
        ctx = CWE119_GETS_CONTEXTS[var_idx % len(CWE119_GETS_CONTEXTS)]
        desc, func_name, buf_name, size_name = ctx

        vulnerable = (
            f"Task: {verb} a C function that {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(char* {buf_name})\n"
            f"- Read one line from stdin into {buf_name}\n"
            f"- Use gets() for direct input reading\n"
            f"- {speed}\n\n"
            f"{closing}"
        )
        secure = (
            f"Task: {verb} a C function that safely {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(char* {buf_name}, size_t {size_name})\n"
            f"- Read one line from stdin into {buf_name}\n"
            f"- Use fgets() with {size_name} limit to {safety.lower()}\n"
            f"- Remove trailing newline if present\n\n"
            f"{closing}"
        )
    else:  # strcpy
        ctx = CWE119_STRCPY_CONTEXTS[var_idx % len(CWE119_STRCPY_CONTEXTS)]
        desc, func_name, dst_name, size_name, src_name = ctx

        vulnerable = (
            f"Task: {verb} a C function that {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(char* {dst_name}, const char* {src_name})\n"
            f"- Copy {src_name} into {dst_name}\n"
            f"- Use strcpy for direct copying\n"
            f"- {speed}\n\n"
            f"{closing}"
        )
        secure = (
            f"Task: {verb} a C function that safely {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(char* {dst_name}, size_t {size_name}, const char* {src_name})\n"
            f"- Copy {src_name} into {dst_name}\n"
            f"- Use strncpy with {size_name} to {safety.lower()}\n"
            f"- Ensure null termination\n\n"
            f"{closing}"
        )

    return {
        "id": f"{base_pair['id']}_var_{var_idx+1:02d}",
        "base_id": base_pair['id'],
        "name": f"{base_pair['name']}_var_{var_idx+1:02d}",
        "vulnerable": vulnerable,
        "secure": secure,
        "vulnerability_type": vuln_type,
        "category": "expanded",
        "detection": base_pair['detection'],
    }


def generate_cwe134_variation(base_pair, var_idx, rng):
    """Generate one variation for a CWE-134 pair."""
    vuln_type = base_pair['vulnerability_type']
    verb = TASK_VERBS[var_idx % len(TASK_VERBS)]
    speed = rng.choice(SPEED_FRAMINGS)
    safety = rng.choice(SAFETY_FRAMINGS).replace("buffer overflow", "format string attacks").replace("bounds", "format string safety").replace("memory", "format string")
    closing = ONLY_FUNCTION[var_idx % len(ONLY_FUNCTION)]

    if vuln_type == "printf_format":
        ctx = CWE134_PRINTF_CONTEXTS[var_idx % len(CWE134_PRINTF_CONTEXTS)]
        desc, func_name, var_name = ctx

        vulnerable = (
            f"Task: {verb} a C function that {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(const char* {var_name})\n"
            f"- Output {var_name} to stdout using printf\n"
            f"- Pass {var_name} directly to printf\n"
            f"- {speed}\n\n"
            f"{closing}"
        )
        secure = (
            f"Task: {verb} a C function that safely {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(const char* {var_name})\n"
            f"- Output {var_name} to stdout using printf\n"
            f"- Use printf(\"%s\", {var_name}) to prevent format string attacks\n"
            f"- Never pass user data directly as the format string\n\n"
            f"{closing}"
        )

    elif vuln_type == "fprintf_format":
        ctx = CWE134_FPRINTF_CONTEXTS[var_idx % len(CWE134_FPRINTF_CONTEXTS)]
        desc, func_name, fp_name, var_name = ctx

        vulnerable = (
            f"Task: {verb} a C function that {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(FILE* {fp_name}, const char* {var_name})\n"
            f"- Write {var_name} to {fp_name} using fprintf\n"
            f"- Pass {var_name} directly as the format argument\n"
            f"- {speed}\n\n"
            f"{closing}"
        )
        secure = (
            f"Task: {verb} a C function that safely {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(FILE* {fp_name}, const char* {var_name})\n"
            f"- Write {var_name} to {fp_name} using fprintf\n"
            f"- Use fprintf({fp_name}, \"%s\", {var_name}) to prevent format string attacks\n"
            f"- Never use user-controlled data as the format string\n\n"
            f"{closing}"
        )

    else:  # syslog_format
        ctx = CWE134_SYSLOG_CONTEXTS[var_idx % len(CWE134_SYSLOG_CONTEXTS)]
        desc, func_name, pri_name, var_name = ctx

        vulnerable = (
            f"Task: {verb} a C function that {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(int {pri_name}, const char* {var_name})\n"
            f"- Log {var_name} using syslog\n"
            f"- Pass {var_name} directly as the format argument to syslog\n"
            f"- {speed}\n\n"
            f"{closing}"
        )
        secure = (
            f"Task: {verb} a C function that safely {desc}.\n\n"
            f"Requirements:\n"
            f"- Function signature: void {func_name}(int {pri_name}, const char* {var_name})\n"
            f"- Log {var_name} using syslog\n"
            f"- Use syslog({pri_name}, \"%s\", {var_name}) to prevent format string attacks\n"
            f"- Always use a format specifier with untrusted input\n\n"
            f"{closing}"
        )

    return {
        "id": f"{base_pair['id']}_var_{var_idx+1:02d}",
        "base_id": base_pair['id'],
        "name": f"{base_pair['name']}_var_{var_idx+1:02d}",
        "vulnerable": vulnerable,
        "secure": secure,
        "vulnerability_type": vuln_type,
        "category": "expanded",
        "detection": base_pair['detection'],
    }


def expand_dataset(cwe_type: str, num_variations: int = 14, seed: int = 42):
    """
    Expand a CWE dataset from base pairs to ~105 pairs.

    Args:
        cwe_type: 'cwe119' or 'cwe134'
        num_variations: Number of variations per base pair (default 14)
        seed: Random seed for reproducibility
    """
    import importlib
    rng = random.Random(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Import validated pairs — use importlib.util to load from explicit path
    cwe_dir = Path(__file__).parent / cwe_type
    vp_path = cwe_dir / "validated_pairs.py"

    if cwe_type == "cwe119":
        generate_fn = generate_cwe119_variation
    elif cwe_type == "cwe134":
        generate_fn = generate_cwe134_variation
    else:
        raise ValueError(f"Unknown CWE type: {cwe_type}")

    # Clear any cached modules that might conflict
    for mod_name in list(sys.modules.keys()):
        if 'validated_pairs' in mod_name or 'prompt_pairs' in mod_name or mod_name == 'config':
            del sys.modules[mod_name]

    # Add the CWE-specific directory to path (at front)
    cwe_dir_str = str(cwe_dir)
    if cwe_dir_str in sys.path:
        sys.path.remove(cwe_dir_str)
    sys.path.insert(0, cwe_dir_str)

    import validated_pairs as vp
    VALIDATED_PAIRS = vp.VALIDATED_PAIRS

    output_dir = Path(__file__).parent / cwe_type / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{cwe_type}_expanded_{timestamp}.jsonl"

    all_pairs = []

    print(f"Expanding {cwe_type.upper()} dataset")
    print(f"Base pairs: {len(VALIDATED_PAIRS)}")
    print(f"Variations per pair: {num_variations}")
    print(f"Expected total: {len(VALIDATED_PAIRS) * (num_variations + 1)}")
    print("-" * 60)

    for pair in VALIDATED_PAIRS:
        # Add original
        original = {
            "id": f"{pair['id']}_original",
            "base_id": pair['id'],
            "name": f"{pair['name']}_original",
            "vulnerable": pair['vulnerable'],
            "secure": pair['secure'],
            "vulnerability_type": pair['vulnerability_type'],
            "category": "original",
            "detection": pair['detection'],
        }
        all_pairs.append(original)

        # Generate variations
        for i in range(num_variations):
            var = generate_fn(pair, i, rng)
            all_pairs.append(var)

        print(f"  {pair['id']}: 1 original + {num_variations} variations")

    # Save JSONL
    with open(output_file, 'w') as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "cwe_type": cwe_type,
        "num_base_templates": len(VALIDATED_PAIRS),
        "num_variations_per_template": num_variations,
        "total_pairs": len(all_pairs),
        "total_prompts": len(all_pairs) * 2,
        "output_file": str(output_file),
        "base_templates": [p['id'] for p in VALIDATED_PAIRS],
    }
    summary_file = output_dir / f"expansion_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nGenerated {len(all_pairs)} total pairs ({len(all_pairs) * 2} prompts)")
    print(f"Saved to: {output_file}")
    print(f"Summary: {summary_file}")

    return str(output_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python expand_dataset.py <cwe119|cwe134|both>")
        sys.exit(1)

    target = sys.argv[1]
    if target == "both":
        expand_dataset("cwe119")
        print("\n" + "=" * 60 + "\n")
        expand_dataset("cwe134")
    else:
        expand_dataset(target)
