#!/usr/bin/env python3
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HEADER = """// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

"""

MIT_MARKERS = (
    "MIT License",
    "SPDX-License-Identifier: MIT",
    "Permission is hereby granted, free of charge",
)

def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "target" in parts or ".git" in parts:
        return True
    return False

def strip_mit_header(lines: list[str], idx: int) -> list[str]:
    """Remove an existing MIT header block directly after the insertion point."""

    if idx >= len(lines):
        return lines

    start = idx
    # Only attempt to strip simple line-comment blocks used by the old MIT header.
    if not lines[start].lstrip().startswith("//"):
        return lines

    end = start
    while end < len(lines) and lines[end].lstrip().startswith("//"):
        end += 1

    block_text = "".join(lines[start:end])
    if any(marker in block_text for marker in MIT_MARKERS):
        return lines[:start] + lines[end:]

    return lines


def add_header_to_file(path: Path):
    text = path.read_text(encoding="utf-8")

    # Already has our header — nothing to do
    if "Licensed under the Apache License, Version 2.0" in text:
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

    # 2.5) If an MIT header is present after the insertion point, drop it first.
    lines = strip_mit_header(lines, idx)

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
