#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Common Unified Diff Patcher
Used by both Triage Agent (apply_patch tool and workspace diff generation)
and Baseline evaluation.

Rules enforced:
- Parses standard unified diff hunks (@@ -start,len +start,len @@).
- Locates hunk application position strictly by exact code context and removed lines;
  line numbers in @@ headers are treated as reference only.
- If a hunk's context/removed block matches 0 locations or >1 locations, rejects the patch.
- Never performs fuzzy/approximate matching.
"""

from dataclasses import dataclass
import difflib
import re
from pathlib import Path

HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n")
FENCE_CLOSE_RE = re.compile(r"\n?\s*```$")


@dataclass
class Hunk:
    ref_old_start: int
    ref_old_len: int
    ref_new_start: int
    ref_new_len: int
    old_lines: list[str]
    new_lines: list[str]
    has_changes: bool


@dataclass
class PatchResult:
    success: bool
    patched_text: str
    error: str = ""
    target_file: str | None = None
    hunks_applied: int = 0


def strip_single_code_fence(text: str) -> str:
    """Strips at most one outer Markdown code fence wrapping a diff or JSON block."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        open_match = FENCE_OPEN_RE.match(stripped)
        if open_match:
            inner = stripped[open_match.end():]
            close_match = FENCE_CLOSE_RE.search(inner)
            if close_match:
                return inner[:close_match.start()]
    return text


def parse_unified_diff(diff_text: str) -> tuple[list[Hunk], str | None, str]:
    """
    Parses a unified diff into a list of Hunks, optional target filename, and error message.
    Returns (hunks, target_file, error). If error is non-empty, parsing failed.
    """
    if not isinstance(diff_text, str) or not diff_text.strip():
        return [], None, "Empty diff provided."

    cleaned = strip_single_code_fence(diff_text)
    raw_lines = cleaned.splitlines()

    target_file: str | None = None
    hunks: list[Hunk] = []
    current_hunk: Hunk | None = None
    in_hunk = False

    for idx, line in enumerate(raw_lines, start=1):
        if not in_hunk:
            if line.startswith("--- "):
                path_part = line[4:].strip().split("\t")[0]
                if path_part and path_part != "/dev/null":
                    if path_part.startswith("a/"):
                        path_part = path_part[2:]
                    target_file = path_part
                continue
            if line.startswith("+++ "):
                path_part = line[4:].strip().split("\t")[0]
                if path_part and path_part != "/dev/null":
                    if path_part.startswith("b/"):
                        path_part = path_part[2:]
                    target_file = path_part
                continue
            if line.startswith("diff ") or line.startswith("index "):
                continue
            m = HUNK_HEADER_RE.match(line)
            if m:
                ref_old_start = int(m.group(1))
                ref_old_len = int(m.group(2)) if m.group(2) is not None else 1
                ref_new_start = int(m.group(3))
                ref_new_len = int(m.group(4)) if m.group(4) is not None else 1
                current_hunk = Hunk(
                    ref_old_start=ref_old_start,
                    ref_old_len=ref_old_len,
                    ref_new_start=ref_new_start,
                    ref_new_len=ref_new_len,
                    old_lines=[],
                    new_lines=[],
                    has_changes=False,
                )
                hunks.append(current_hunk)
                in_hunk = True
                continue
            if not line.strip():
                continue
            return [], target_file, f"Line {idx}: Unexpected content outside hunk header: {line!r}"
        else:
            m = HUNK_HEADER_RE.match(line)
            if m:
                ref_old_start = int(m.group(1))
                ref_old_len = int(m.group(2)) if m.group(2) is not None else 1
                ref_new_start = int(m.group(3))
                ref_new_len = int(m.group(4)) if m.group(4) is not None else 1
                current_hunk = Hunk(
                    ref_old_start=ref_old_start,
                    ref_old_len=ref_old_len,
                    ref_new_start=ref_new_start,
                    ref_new_len=ref_new_len,
                    old_lines=[],
                    new_lines=[],
                    has_changes=False,
                )
                hunks.append(current_hunk)
                continue

            if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("diff "):
                return [], target_file, "Multi-file diffs are not supported in a single patch operation."

            assert current_hunk is not None
            if line.startswith(" "):
                content = line[1:]
                current_hunk.old_lines.append(content)
                current_hunk.new_lines.append(content)
            elif line == "":
                # Blank context line where trailing space was omitted
                current_hunk.old_lines.append("")
                current_hunk.new_lines.append("")
            elif line.startswith("-"):
                content = line[1:]
                current_hunk.old_lines.append(content)
                current_hunk.has_changes = True
            elif line.startswith("+"):
                content = line[1:]
                current_hunk.new_lines.append(content)
                current_hunk.has_changes = True
            elif line.startswith("\\"):
                # "\ No newline at end of file" marker
                continue
            else:
                return [], target_file, f"Line {idx}: Malformed hunk line (must start with ' ', '+', '-', or '\\\\'): {line!r}"

    if not hunks:
        return [], target_file, "No valid unified diff hunks (@@ ... @@) found."

    for h_idx, h in enumerate(hunks, start=1):
        if not h.has_changes:
            return [], target_file, f"Hunk {h_idx} contains no additions or deletions."
        if not h.old_lines:
            return [], target_file, f"Hunk {h_idx} has no context or removed lines to anchor position."

    return hunks, target_file, ""


def _find_exact_subsequence_matches(haystack: list[str], needle: list[str]) -> list[int]:
    """Finds all 0-based start indices where needle matches haystack exactly."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return []
    matches = []
    limit = len(haystack) - n + 1
    for i in range(limit):
        if haystack[i : i + n] == needle:
            matches.append(i)
    return matches


def apply_unified_diff(original_text: str, diff_text: str) -> PatchResult:
    """
    Applies a unified diff to original_text using exact context matching.
    Line numbers in hunk headers are reference only.
    Rejects if any hunk matches 0 locations or >1 locations, or if hunks overlap.
    """
    hunks, target_file, err = parse_unified_diff(diff_text)
    if err:
        return PatchResult(success=False, patched_text=original_text, error=err, target_file=target_file)

    orig_lines = original_text.splitlines()
    had_trailing_newline = original_text.endswith("\n") or original_text == ""

    # Locate every hunk on orig_lines first to verify uniqueness and non-overlapping ranges
    located_spans: list[tuple[int, int, Hunk, int]] = []
    for h_idx, hunk in enumerate(hunks, start=1):
        matches = _find_exact_subsequence_matches(orig_lines, hunk.old_lines)
        if len(matches) == 0:
            return PatchResult(
                success=False,
                patched_text=original_text,
                error=f"Hunk {h_idx} (around reference line {hunk.ref_old_start}) did not match any location in target file.",
                target_file=target_file,
            )
        if len(matches) > 1:
            return PatchResult(
                success=False,
                patched_text=original_text,
                error=f"Ambiguous hunk {h_idx}: context matched {len(matches)} locations in target file (lines {[m + 1 for m in matches]}).",
                target_file=target_file,
            )
        start_idx = matches[0]
        end_idx = start_idx + len(hunk.old_lines)
        located_spans.append((start_idx, end_idx, hunk, h_idx))

    # Sort spans by start index and verify no overlap
    located_spans.sort(key=lambda item: item[0])
    for i in range(len(located_spans) - 1):
        cur_start, cur_end, _, cur_id = located_spans[i]
        next_start, next_end, _, next_id = located_spans[i + 1]
        if cur_end > next_start:
            return PatchResult(
                success=False,
                patched_text=original_text,
                error=f"Overlapping hunks {cur_id} and {next_id} detected.",
                target_file=target_file,
            )

    # Reconstruct patched lines
    result_lines: list[str] = []
    cursor = 0
    for start_idx, end_idx, hunk, _ in located_spans:
        result_lines.extend(orig_lines[cursor:start_idx])
        result_lines.extend(hunk.new_lines)
        cursor = end_idx
    result_lines.extend(orig_lines[cursor:])

    patched_text = "\n".join(result_lines)
    if had_trailing_newline and result_lines:
        patched_text += "\n"

    return PatchResult(
        success=True,
        patched_text=patched_text,
        error="",
        target_file=target_file,
        hunks_applied=len(hunks),
    )


def create_unified_diff(original_text: str, modified_text: str, filename: str = "file.v") -> str:
    """
    Creates a canonical unified diff between original_text and modified_text.
    Automatically ensures sufficient context lines so that apply_unified_diff
    uniquely locates every hunk.
    """
    if original_text == modified_text:
        return ""

    clean_name = Path(filename.replace("\\", "/")).name
    orig_lines = original_text.splitlines(keepends=True)
    mod_lines = modified_text.splitlines(keepends=True)

    for context_lines in (3, 5, 10, 20, max(len(orig_lines), 1)):
        diff_str = "".join(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{clean_name}",
                tofile=f"b/{clean_name}",
                n=context_lines,
            )
        )
        check = apply_unified_diff(original_text, diff_str)
        if check.success and check.patched_text == modified_text:
            return diff_str

    return "".join(
        difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{clean_name}",
            tofile=f"b/{clean_name}",
            n=max(len(orig_lines), 3),
        )
    )
