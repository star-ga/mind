#!/usr/bin/env python3
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HEADER = """// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

"""

def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "target" in parts or ".git" in parts:
        return True
    return False

def add_header_to_file(path: Path):
    text = path.read_text(encoding="utf-8")
    if "STARGA Inc. and MIND Language Contributors" in text:
        return

    if text.startswith("#!"):
        lines = text.splitlines(keepends=True)
        first = lines[0]
        rest = "".join(lines[1:])
        new_text = first + HEADER + rest
    else:
        new_text = HEADER + text

    path.write_text(new_text, encoding="utf-8")
    print(f"Updated: {path}")

def main():
    for root, dirs, files in os.walk(REPO_ROOT):
        root_path = Path(root)
        if should_skip(root_path):
            continue
        for name in files:
            if name.endswith(".rs"):
                add_header_to_file(root_path / name)

if __name__ == "__main__":
    main()
