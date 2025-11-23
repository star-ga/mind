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

    # Already has our header — nothing to do
    if "STARGA Inc. and MIND Language Contributors" in text:
        return

    lines = text.splitlines(keepends=True)
    if not lines:
        return

    idx = 0

    # 1) Optional shebang: "#!/usr/bin/env ..." etc.
    if lines[idx].startswith("#!") and not lines[idx].startswith("#!["):
        idx += 1

    # 2) All consecutive crate-level attributes of the form "#![cfg(...)]"
    while idx < len(lines) and lines[idx].lstrip().startswith("#!["):
        idx += 1

    # 3) Build new text: shebang + attributes + HEADER + the rest
    new_text = "".join(lines[:idx]) + HEADER + "".join(lines[idx:])

    path.write_text(new_text, encoding="utf-8")

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
