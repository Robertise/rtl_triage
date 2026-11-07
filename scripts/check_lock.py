#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Lockfile Generator & Integrity Checker
Computes and verifies SHA-256 checksums for all locked Gate 3 artifacts:
  - bugs/catalog.yaml
  - labels/labels.json
  - all files in bugs/variants/
  - all files in rtl/
  - all files in tb/visible/
  - all files in tb/heldout/
  - bugs/inject.py
  - bugs/validate.py
  - docs/preregistration.md

Exits with code 0 when all hashes and file sets match labels/labels.lock,
or non-zero (1) if any file is modified, missing, or unexpected.
"""

import argparse
import hashlib
import sys
from pathlib import Path

EXPLICIT_FILES = [
    "bugs/catalog.yaml",
    "labels/labels.json",
    "bugs/inject.py",
    "bugs/validate.py",
    "docs/preregistration.md",
]

TRACKED_DIRS = [
    "bugs/variants",
    "rtl",
    "tb/visible",
    "tb/heldout",
]


def sha256_file(path: Path) -> str:
    """Computes the raw byte SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_tracked_files(root: Path) -> tuple[list[str], list[str]]:
    """
    Returns (sorted_rel_paths, missing_required_paths) relative to root
    using POSIX forward slashes in deterministic lexicographical order.
    """
    rel_paths = set()
    missing = []

    for rel in EXPLICIT_FILES:
        p = root / rel
        if p.is_file():
            rel_paths.add(rel)
        else:
            missing.append(rel)

    for d_rel in TRACKED_DIRS:
        d_path = root / d_rel
        if not d_path.is_dir():
            missing.append(d_rel)
            continue
        for p in d_path.rglob("*"):
            if p.is_file():
                if p.name == ".gitkeep":
                    continue
                rel_posix = p.relative_to(root).as_posix()
                rel_paths.add(rel_posix)

    return sorted(rel_paths), sorted(missing)


def write_lockfile(root: Path, lock_path: Path) -> int:
    """Generates labels/labels.lock deterministically without timestamps."""
    rel_paths, missing = collect_tracked_files(root)
    if missing:
        for m in missing:
            sys.stderr.write(f"[ERROR] Required path missing: {m}\n")
        return 1

    lines = []
    for rel in rel_paths:
        digest = sha256_file(root / rel)
        lines.append(f"{digest}  {rel}\n")

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)

    print(f"[LOCK CREATED] Wrote {len(lines)} SHA-256 entries to {lock_path.relative_to(root).as_posix() if lock_path.is_relative_to(root) else lock_path}")
    return 0


def parse_lockfile(lock_path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Parses lockfile lines of format '<sha256>  <rel_path>'."""
    entries = []
    errors = []
    with open(lock_path, "r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                errors.append(f"Line {idx}: Invalid lockfile format: {line!r}")
                continue
            entries.append((parts[1], parts[0]))
    return entries, errors


def check_lockfile(root: Path, lock_path: Path) -> int:
    """Verifies current repository state against labels/labels.lock."""
    if not lock_path.is_file():
        print(f"[FAIL] Lockfile not found: {lock_path}")
        return 1

    entries, parse_errors = parse_lockfile(lock_path)
    if parse_errors:
        for err in parse_errors:
            print(f"[FAIL] {err}")
        return 1

    if not entries:
        print(f"[FAIL] Lockfile is empty: {lock_path}")
        return 1

    locked_paths = [rel for rel, _ in entries]
    if locked_paths != sorted(locked_paths):
        print("[FAIL] Lockfile paths are not in canonical sorted order.")
        return 1

    locked_map = dict(entries)
    current_paths, missing_required = collect_tracked_files(root)

    discrepancies = []

    for m in missing_required:
        discrepancies.append(f"MISSING REQUIRED: {m}")

    # Check all locked entries
    for rel, expected_hash in entries:
        target = root / rel
        if not target.is_file():
            discrepancies.append(f"MISSING FILE: {rel}")
        else:
            actual_hash = sha256_file(target)
            if actual_hash != expected_hash:
                discrepancies.append(
                    f"HASH MISMATCH: {rel} (expected {expected_hash[:12]}..., got {actual_hash[:12]}...)"
                )

    # Check for unexpected extra files in tracked directories
    locked_set = set(locked_map.keys())
    for rel in current_paths:
        if rel not in locked_set:
            discrepancies.append(f"UNEXPECTED FILE: {rel}")

    if discrepancies:
        print(f"[FAIL] Lock verification failed with {len(discrepancies)} discrepancy(ies):")
        for d in discrepancies:
            print(f"  - {d}")
        return 1

    print(f"[PASS] All {len(entries)} locked files verified against {lock_path.relative_to(root).as_posix() if lock_path.is_relative_to(root) else lock_path} (SHA-256 match 100%).")
    return 0


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate or verify labels/labels.lock SHA-256 checksums.")
    parser.add_argument("--root", default=str(default_root), help="Repository root directory")
    parser.add_argument("--lockfile", default=None, help="Path to labels.lock (default: <root>/labels/labels.lock)")
    parser.add_argument("--write", action="store_true", help="Generate/overwrite labels/labels.lock")

    args = parser.parse_args()
    root = Path(args.root).resolve()
    lock_path = Path(args.lockfile).resolve() if args.lockfile else (root / "labels" / "labels.lock")

    if args.write:
        sys.exit(write_lockfile(root, lock_path))
    else:
        sys.exit(check_lockfile(root, lock_path))


if __name__ == "__main__":
    main()
