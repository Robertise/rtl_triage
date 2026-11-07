#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Common Output JSON Parser
Shared between Triage Agent and Baseline configurations.

Preregistration rules enforced:
- If the entire response parses as JSON after stripping at most one outer Markdown code fence
  (```json ... ``` or ``` ... ```), handle as standard.
- Otherwise, search for ```json code blocks in the response:
  - Exactly 1 block: parse that block and ignore text outside the fence.
  - 0 or >1 blocks: status = FAILED.
- Never searches for unfenced JSON, never attempts JSON repairs, never re-prompts the LLM.
- Normalizes `file` to basename (e.g., "rtl/fifo.v" -> "fifo.v").
- Requires exact fields and types:
  - Agent: file (str), line (int > 0, not bool), bug_type (str), root_cause (str), patch (str), verified (bool).
  - Baseline: file (str), line (int > 0, not bool), bug_type (str), root_cause (str), patch (str) (no `verified`).
- If `bug_type` is a valid non-empty string outside the 7 preregistered bug types,
  the run is NOT marked FAILED (status remains COMPLETED, `is_known_bug_type` is False so Acc_cls = 0).
- If required fields are missing, have wrong types, or JSON is invalid -> status = FAILED.
- If `stop_reason == "max_tokens"` and output is incomplete/invalid -> status = FAILED and `max_tokens_stop = True`.
- Never re-prompts the LLM to fix formatting.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

VALID_BUG_TYPES = frozenset({
    "off_by_one",
    "reset_polarity",
    "missing_branch",
    "bit_width_mismatch",
    "blocking_nonblocking",
    "fifo_flag",
    "fsm_invalid_transition",
})

AGENT_REQUIRED_FIELDS = ("file", "line", "bug_type", "root_cause", "patch", "verified")
BASELINE_REQUIRED_FIELDS = ("file", "line", "bug_type", "root_cause", "patch")

FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n")
FENCE_CLOSE_RE = re.compile(r"\n?\s*```$")


@dataclass
class ParseResult:
    valid: bool
    status: str  # "COMPLETED" or "FAILED"
    data: dict[str, Any] | None
    error: str = ""
    max_tokens_stop: bool = False
    is_known_bug_type: bool = False


def strip_one_markdown_fence(raw_text: str) -> str:
    """
    Removes at most ONE outer Markdown code fence wrapping the text.
    Does not strip multiple nested fences.
    """
    trimmed = raw_text.strip()
    if trimmed.startswith("```") and trimmed.endswith("```"):
        open_match = FENCE_OPEN_RE.match(trimmed)
        if open_match:
            inner = trimmed[open_match.end():]
            close_match = FENCE_CLOSE_RE.search(inner)
            if close_match:
                return inner[:close_match.start()].strip()
    return trimmed


def normalize_filename(file_val: str) -> str:
    """Normalizes a file path string to its basename (e.g. 'rtl/fifo.v' -> 'fifo.v')."""
    cleaned = file_val.strip().replace("\\", "/")
    if cleaned.endswith("/"):
        return ""
    return Path(cleaned).name


def parse_model_output(
    raw_text: str,
    mode: str = "agent",
    stop_reason: str = "end_turn",
) -> ParseResult:
    """
    Parses and validates final model JSON response for either 'agent' or 'baseline' mode.
    """
    if mode not in ("agent", "baseline"):
        raise ValueError(f"Unsupported parser mode: {mode!r}")

    is_max_tokens = (stop_reason == "max_tokens")

    if not isinstance(raw_text, str) or not raw_text.strip():
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error="Empty model output.",
            max_tokens_stop=is_max_tokens,
        )

    unwrapped = strip_one_markdown_fence(raw_text)

    parsed = None
    try:
        parsed = json.loads(unwrapped)
    except json.JSONDecodeError as direct_exc:
        # If full text cannot be parsed directly (e.g. conversational preamble/postscript),
        # search for ```json code blocks in raw_text.
        json_blocks = re.findall(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if len(json_blocks) == 1:
            try:
                parsed = json.loads(json_blocks[0].strip())
            except json.JSONDecodeError as block_exc:
                return ParseResult(
                    valid=False,
                    status="FAILED",
                    data=None,
                    error=f"Invalid JSON in ```json block: {block_exc}",
                    max_tokens_stop=is_max_tokens,
                )
        elif len(json_blocks) > 1:
            return ParseResult(
                valid=False,
                status="FAILED",
                data=None,
                error=f"Ambiguous model output: found {len(json_blocks)} ```json code blocks, expected at most 1.",
                max_tokens_stop=is_max_tokens,
            )
        else:
            return ParseResult(
                valid=False,
                status="FAILED",
                data=None,
                error=f"Invalid JSON: {direct_exc}",
                max_tokens_stop=is_max_tokens,
            )

    if not isinstance(parsed, dict):
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error=f"Top-level JSON value must be an object, got {type(parsed).__name__}.",
            max_tokens_stop=is_max_tokens,
        )

    required_fields = AGENT_REQUIRED_FIELDS if mode == "agent" else BASELINE_REQUIRED_FIELDS
    missing = [f for f in required_fields if f not in parsed]
    if missing:
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error=f"Missing required field(s): {missing}",
            max_tokens_stop=is_max_tokens,
        )

    # Validate `file`
    raw_file = parsed.get("file")
    if not isinstance(raw_file, str) or not raw_file.strip():
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error="Field 'file' must be a non-empty string.",
            max_tokens_stop=is_max_tokens,
        )
    norm_file = normalize_filename(raw_file)
    if not norm_file:
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error=f"Field 'file' could not be normalized to a valid basename: {raw_file!r}",
            max_tokens_stop=is_max_tokens,
        )

    # Validate `line` (must be int, not bool, > 0)
    raw_line = parsed.get("line")
    if isinstance(raw_line, bool) or not isinstance(raw_line, int) or raw_line <= 0:
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error=f"Field 'line' must be a positive integer, got {raw_line!r} ({type(raw_line).__name__}).",
            max_tokens_stop=is_max_tokens,
        )

    # Validate `bug_type` (must be non-empty string; unknown label allowed without failing run)
    raw_bug_type = parsed.get("bug_type")
    if not isinstance(raw_bug_type, str) or not raw_bug_type.strip():
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error="Field 'bug_type' must be a non-empty string.",
            max_tokens_stop=is_max_tokens,
        )
    clean_bug_type = raw_bug_type.strip()
    is_known_bug_type = clean_bug_type in VALID_BUG_TYPES

    # Validate `root_cause`
    raw_root_cause = parsed.get("root_cause")
    if not isinstance(raw_root_cause, str) or not raw_root_cause.strip():
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error="Field 'root_cause' must be a non-empty string.",
            max_tokens_stop=is_max_tokens,
        )

    # Validate `patch`
    raw_patch = parsed.get("patch")
    if not isinstance(raw_patch, str):
        return ParseResult(
            valid=False,
            status="FAILED",
            data=None,
            error="Field 'patch' must be a string.",
            max_tokens_stop=is_max_tokens,
        )

    normalized_data: dict[str, Any] = {
        "file": norm_file,
        "line": raw_line,
        "bug_type": clean_bug_type,
        "root_cause": raw_root_cause.strip(),
        "patch": raw_patch,
    }

    if mode == "agent":
        raw_verified = parsed.get("verified")
        if not isinstance(raw_verified, bool):
            return ParseResult(
                valid=False,
                status="FAILED",
                data=None,
                error=f"Field 'verified' must be a boolean, got {raw_verified!r} ({type(raw_verified).__name__}).",
                max_tokens_stop=is_max_tokens,
            )
        normalized_data["verified"] = raw_verified

    return ParseResult(
        valid=True,
        status="COMPLETED",
        data=normalized_data,
        error="",
        max_tokens_stop=False,
        is_known_bug_type=is_known_bug_type,
    )
