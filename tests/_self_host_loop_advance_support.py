#!/usr/bin/env python3
"""Shared fixtures for the self-host loop advancement controls.

This module contains no control cases and is imported by the two focused case
modules and the stable self_host_loop_advance_test.py runner.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SMOKE_DIR = REPO / "examples" / "mindc_mind"
PREFLIGHT = REPO / "scripts" / "preflight.sh"
sys.path.insert(0, str(SMOKE_DIR))

# Importing the module under test must not build anything or exit; that property
# is asserted for real in case_import_does_not_resolve_or_build_the_oracle, from
# a CLEAN subprocess where this import has not already happened.
#
# The guard below exists because that case cannot report a regression it is
# killed by: if import-time resolution comes back, THIS import dies first and the
# whole file exits before a single case runs. Measured with the eager
# `resolve_so()` restored: exit 1 with the resolver's refusal and not one `[FAIL]`
# line — correct-but-anonymous, and easy to misread as "no failures". Naming it
# costs four lines and keeps the non-zero exit.
try:
    import self_host_loop_smoke as loop  # noqa: E402
except SystemExit as exc:  # pragma: no cover - only on the regression
    print("[FAIL] import: importing self_host_loop_smoke must not exit "
          f"(the oracle is being resolved at import time): {exc}")
    raise

FAILURES: list[str] = []


def describe(value: object) -> str:
    if isinstance(value, bytes):
        return f"bytes(len={len(value)}, sha256={hashlib.sha256(value).hexdigest()})"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(describe(item) for item in value) + "]"
    text = repr(value)
    return text if len(text) <= 240 else text[:237] + "..."


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={describe(got)} want={describe(want)}")
    if not ok:
        FAILURES.append(label)


# ── synthetic material ─────────────────────────────────────────────────────

def elf(tag: bytes, size: int = 5000) -> bytes:
    """A byte blob that satisfies `is_static_elf` and is unique per `tag`.

    Shaped, not compiled: magic at 0, ET_EXEC at 16, padded past the 4096-byte
    floor. It is a stand-in for an emitted image so the harness's comparisons
    have something to compare; it is not an executable and is not compiler
    output.
    """
    head = b"\x7fELF" + b"\x00" * 12 + b"\x02\x00" + tag
    return head + b"\x00" * (size - len(head))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


OLD_SEED = elf(b"old-seed")
NEW_POINT = elf(b"new-fixed-point")
OTHER = elf(b"something-else")
# The two-hop shape: the OLD seed compiles the new source into a compiler that
# still carries the old codegen (INTERMEDIATE), and that compiler then emits the
# new fixed point. INTERMEDIATE != NEW_POINT is exactly what --hops 2 admits.
INTERMEDIATE = elf(b"intermediate-stage1")
SRC = b"// synthetic combined source\n"
IMAGE = b"\x00" * 16 + SRC
USER_LO = 7


class StubOracle:
    """Stands in for `ResolvedSo` at the ctypes boundary.

    Carries the same three things `do_advance` consults — existence, freshness,
    provenance — so a stale or absent oracle can be presented without building
    one.
    """

    def __init__(self, *, present: bool = True, provenance: str = "fresh-build") -> None:
        self._present = present
        self.provenance = provenance
        self.detail = f"stub oracle ({provenance})"
        self.name = "stub-oracle.so"

    def exists(self) -> bool:
        return self._present

    @property
    def is_fresh(self) -> bool:
        return self.provenance in ("fresh-build", "env-verified")

    def __str__(self) -> str:  # used in the refusal messages
        return "<stub oracle>"


class Sandbox:
    """A throwaway fixture pair with the module's paths pointed at it.

    Restores every patched module attribute on exit so one case cannot leak
    into the next, and never touches the real tracked fixture.
    """

    def __init__(self) -> None:
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="advance-control-"))
        self.elf_path = self.dir / "stage1.elf"
        self.manifest_path = self.dir / "MANIFEST.txt"
        self.elf_path.write_bytes(OLD_SEED)
        self.elf_path.chmod(0o755)
        self.manifest_path.write_text("# original manifest\nstage1.elf\t1\tdeadbeef\n")
        self.before_elf = self.elf_path.read_bytes()
        self.before_manifest = self.manifest_path.read_bytes()
        self._saved: dict[str, object] = {}

    def patch(self, **kw: object) -> None:
        for k, v in kw.items():
            self._saved.setdefault(k, getattr(loop, k))
            setattr(loop, k, v)

    def __enter__(self) -> "Sandbox":
        self.patch(_FROZEN=self.elf_path, _FROZEN_MANIFEST=self.manifest_path)
        return self

    def __exit__(self, *exc: object) -> None:
        for k, v in self._saved.items():
            setattr(loop, k, v)
        shutil.rmtree(self.dir, ignore_errors=True)

    def unchanged(self) -> bool:
        """Both published files still byte-identical to their pre-run state."""
        return (
            self.elf_path.read_bytes() == self.before_elf
            and self.manifest_path.read_bytes() == self.before_manifest
        )


def captured(fn, *args, **kwargs) -> tuple[object, str]:
    """Call `fn` with stdout captured; return (result, printed text).

    An exception is NOT swallowed: a control that expects a verdict must see a
    traceback as a failure, not as a missing line of output.
    """
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


def receipt_rows(text: str) -> dict[str, str]:
    """The `# <key>\\t<value>` receipt rows of a manifest, as a dict."""
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("# ") and "\t" in line:
            key, value = line[2:].split("\t", 1)
            rows[key] = value
    return rows


def stub_loop(stages, log: list[bytes], *, on_call=None, before_call=None):
    """Replace `run_elf` with a recorder that returns the given stage chain.

    Records the BYTES OF THE FILE it was asked to run, not its path. The harness
    now executes a private snapshot of the captured old seed rather than the live
    fixture, so "the old seed ran first" has to be asserted by content — which is
    the stronger claim anyway, and the one the recorded
    `advanced_from_seed_sha256` is supposed to describe.

    An entry in `stages` may be an exception instance, which is raised instead of
    returned; that is how a stage timeout or spawn error is presented.

    `before_call` fires BEFORE the file is read, `on_call` after — the two sides
    of the time-of-check/time-of-use window a case may need to open.
    """
    seq = list(stages)

    def _run(path: pathlib.Path, stdin_image: bytes) -> bytes:
        if before_call is not None:
            before_call(len(log) + 1)
        log.append(pathlib.Path(path).read_bytes())
        if on_call is not None:
            on_call(len(log))
        item = seq[min(len(log) - 1, len(seq) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    return _run
