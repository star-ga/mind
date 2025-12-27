#!/usr/bin/env python3
# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_SCAN_LINES = 60

APACHE_MARKER = "Licensed under the Apache License, Version 2.0"
MIT_MARKER = "Permission is hereby granted, free of charge"
LEGACY_COMMUNITY_MARKER = "MIND Language"
MIT_MARKERS = (
    "MIT License",
    "SPDX-License-Identifier: MIT",
    MIT_MARKER,
    LEGACY_COMMUNITY_MARKER,
)


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return "target" in parts or ".git" in parts


def detect_comment_prefix(ext: str) -> str:
    if ext == ".rs":
        return "//"
    if ext == ".py":
        return "#"
    raise ValueError(f"Unsupported extension: {ext}")


def find_year(lines: list[str]) -> str:
    for line in lines[:MAX_SCAN_LINES]:
        match = re.search(r"Copyright\s+(\d{4})\s+STARGA Inc\.", line)
        if match:
            return match.group(1)
    return str(datetime.datetime.now().year)


def build_header(year: str, prefix: str) -> str:
    lines = [
        f"{prefix} Copyright {year} STARGA Inc.",
        f"{prefix} {APACHE_MARKER} (the \"License\");",
        f"{prefix} you may not use this file except in compliance with the License.",
        f"{prefix} You may obtain a copy of the License at",
        f"{prefix}     http://www.apache.org/licenses/LICENSE-2.0",
        f"{prefix}",
        f"{prefix} Unless required by applicable law or agreed to in writing, software",
        f"{prefix} distributed under the License is distributed on an \"AS IS\" BASIS,",
        f"{prefix} WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        f"{prefix} See the License for the specific language governing permissions and",
        f"{prefix} limitations under the License.",
        "",
    ]
    return "\n".join(lines) + "\n"


def strip_legacy_header(lines: list[str], idx: int, prefix: str) -> list[str]:
    if idx >= len(lines):
        return lines

    if not lines[idx].lstrip().startswith(prefix):
        return lines

    end = idx
    while end < len(lines) and lines[end].lstrip().startswith(prefix):
        end += 1

    block_text = "".join(lines[idx:end])
    if any(marker in block_text for marker in MIT_MARKERS):
        return lines[:idx] + lines[end:]

    return lines


def add_header_to_file(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    original_lines = list(lines)

    if not lines:
        return

    prefix = detect_comment_prefix(path.suffix)

    idx = 0

    if lines[idx].startswith("#!") and not lines[idx].startswith("#!["):
        idx += 1

    if path.suffix == ".py" and idx < len(lines):
        encoding_re = re.compile(r"#.*coding[:=]\s*[-\w.]+")
        if encoding_re.match(lines[idx]):
            idx += 1

    while idx < len(lines) and lines[idx].lstrip().startswith("#!["):
        idx += 1

    probe_lines: list[str] = []
    probe_limit = min(len(lines), idx + MAX_SCAN_LINES)
    probe_idx = idx
    while probe_idx < probe_limit:
        line = lines[probe_idx]
        if line.strip() == "" or line.lstrip().startswith(prefix):
            probe_lines.append(line)
            probe_idx += 1
            continue
        break

    header_probe = "".join(probe_lines)
    if APACHE_MARKER in header_probe:
        return

    lines = strip_legacy_header(lines, idx, prefix)

    year = find_year(original_lines)
    header = build_header(year, prefix)
    new_text = "".join(lines[:idx]) + header + "".join(lines[idx:])

    path.write_text(new_text, encoding="utf-8")


def main():
    for root, dirs, files in os.walk(REPO_ROOT):
        root_path = Path(root)
        if should_skip(root_path):
            continue
        for name in files:
            path = root_path / name
            if path.suffix in {".rs", ".py"}:
                add_header_to_file(path)


if __name__ == "__main__":
    main()
