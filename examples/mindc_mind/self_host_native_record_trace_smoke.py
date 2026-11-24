"""Canonical-trace contract for record-field reads, incl. the fixed-array case.

WHAT THIS GATE PROVES
---------------------
`flatten_expr_env`'s `ast_index` arm admits ONE call-bearing base: the
`__mind_load_i64(recv + j*8)` subtree `field_build_load_call` synthesises for a
resolved `recv.field` receiver. Opening a previously fail-closed refusal is only
safe if the bytes that come out are the RIGHT bytes, so every claim here is
measured against `mindc --emit-mic3` and against the ordinary evaluator:

  1. REFERENCE BYTE EQUALITY — a `trace` fixture's pure-MIND `selftest_nb_mic3`
     module is compared BYTE-FOR-BYTE with the reference's. "Non-empty" is not a
     result; the exact bytes are.
  2. RUNTIME CLOSURE — the same source goes through `selftest_native_elf_u`, the
     note-carrying executable path (which itself needs the canonical trace to
     build), and the emitted ELF is RUN. Its exit status must equal the recorded
     value, and `reference.mind` restates each value as an `#[test]` over the
     ordinary evaluator so the expectation is the reference's, not this gate's.
  3. DETERMINISM — every canonical trace is built twice and must be identical.
     This is a regression control, not a formality: the prefold passes rewrite
     against `build_src_intrinsics`' copy when the module source does not itself
     spell `__mind_alloc` / `__mind_store_i64` / `__mind_load_i64`, and while the
     strtab/emit passes still read the ORIGINAL buffer those synthesised callee
     spans resolved past its end — the same program emitted different bytes on
     every call inside one process.
  4. SOURCE BINDING — every expectation lives in `MANIFEST.txt` keyed by the
     fixture's size + sha256, and the manifest must name exactly the fixtures on
     disk in both directions. Edit a fixture without re-deriving its row and the
     gate fails LOUD, so recorded evidence cannot drift from its input.
  5. VALIDITY — EVERY fixture, in every role, is fed to `mindc check` and paired
     with a malformed-input control. A zero-byte expectation is only evidence
     about a refusal if the input was valid MIND to begin with; a `blocked` or
     `refuse` fixture corrupted into a syntax error would otherwise emit 0 bytes
     on both paths and keep passing forever. The sweep covers all four roles,
     not just the `trace` positives.

WHAT IS STILL BLOCKED, AND WHY THAT IS PINNED HERE
--------------------------------------------------
The reference lowers a record with a fixed-array field INLINE — one canonical
slot per element — from CONSTRUCTION onward, and this tree's struct-literal
desugar still lays every field out one slot wide. Measured against
`mindc --emit-mic3`: the bare construction `let r = R { pre: 5, xs: [11,22,33] }`
is 129 self-host bytes against 170; `r.xs[1] + r.post` is 203 against 279 (the
reference stream also carries an `__mind_oob_check` guard call this path never
emits); an all-scalar record of the same shape is byte-EQUAL.

The precondition therefore has to sit on BOTH paths, and the scope of the claim
matters:

  * `srt_canonical_layout_modelled` via `field_owner_layout_modelled` /
    `fixed_field_array_key` closes the field READ.
  * the same predicate via `slit_ctor_layout_key` closes the struct-literal
    CONSTRUCTION desugar.

The read guard alone was NOT enough, and nothing here caught that: every
`blocked` fixture also performs a read (`r.xs[1]`, `let ys = r.xs`), so the read
guard fired first and a program that only CONSTRUCTS a fixed-array record was
never exercised on its own — it emitted the 129 wrong bytes measured above.
`construct_only.mind` is that program, and it is why `blocked` now means the
construction as well as the read.

`blocked` rows pin that: the fixture must produce ZERO bytes on BOTH the
canonical and the executable path. When the layout gap closes they will start
emitting, this gate will go red, and the fixtures get re-derived as `trace`
positives — which is the point. A gap recorded as a checked artifact cannot be
mistaken for a gap that was handled.

Re-deriving the manifest after an intentional change:
    python3 examples/mindc_mind/self_host_native_record_trace_smoke.py --bless
and REVIEW the resulting diff — it rewrites the recorded evidence.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_DATA = _HERE / "testdata" / "native_record_trace"
_MANIFEST = _DATA / "MANIFEST.txt"

sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402
from _selfhost_sources import core_limit_argv  # noqa: E402

_TRACE, _BLOCKED, _REFUSE, _EVALUATOR = "trace", "blocked", "refuse", "evaluator"

_HEADER = (
    "# Canonical record-trace fixtures — recorded evidence bound to its input.\n"
    "# A row is valid only for the exact source bytes named by size+sha256; edit a\n"
    "# fixture and the gate fails until its row is re-derived (--bless).\n"
    "# role: trace     = canonical bytes must equal --emit-mic3; the executable must exit as recorded\n"
    "#       blocked   = must stay 0 bytes on BOTH paths (fixed-array record layout not modelled yet)\n"
    "#       refuse    = must stay 0 bytes on BOTH paths (wrong type / unproven owner / constant OOB)\n"
    "#       evaluator = ordinary `mindc test` fixture; expected_exit is the test count\n"
    "# name\trole\tsize_bytes\tsource_sha256\ttrace_sha256\texpected_exit\n"
)

#: Element mutation on the byte-equal control: fixture, needle, replacement, exit.
_MUTATION = ("scalar_owner", b"mid: 22", b"mid: 23", 30)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Row:
    """One manifest row: what a fixture is, and the evidence bound to it."""

    __slots__ = ("role", "size", "src_sha", "trace_sha", "expected")

    def __init__(self, role, size, src_sha, trace_sha, expected):
        self.role = role
        self.size = size
        self.src_sha = src_sha
        self.trace_sha = trace_sha
        self.expected = expected


def _load_manifest() -> dict[str, Row]:
    rows: dict[str, Row] = {}
    for line in _MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, role, size, src_sha, trace_sha, exit_code = line.split("\t")
        if role not in (_TRACE, _BLOCKED, _REFUSE, _EVALUATOR):
            raise AssertionError(f"{name}: unknown manifest role {role!r}")
        rows[name] = Row(
            role,
            int(size),
            src_sha,
            None if trace_sha == "-" else trace_sha,
            None if exit_code == "-" else int(exit_code),
        )
    return rows


def _fixtures_on_disk() -> set[str]:
    return {p.stem for p in _DATA.glob("*.mind")}


def _check_coverage(rows: dict[str, Row]) -> None:
    on_disk = _fixtures_on_disk()
    declared = set(rows)
    if on_disk != declared:
        raise AssertionError(
            "MANIFEST.txt does not name exactly the fixtures on disk: "
            f"unrecorded={sorted(on_disk - declared)} stale={sorted(declared - on_disk)}"
        )
    print(f"  PASS manifest covers {len(on_disk)} fixtures in both directions")


def _source(name: str, rows: dict[str, Row]) -> bytes:
    """Fixture bytes, REFUSED unless they are the bytes the row was derived from."""
    data = (_DATA / f"{name}.mind").read_bytes()
    row = rows[name]
    got = _sha256(data)
    if len(data) != row.size or got != row.src_sha:
        raise AssertionError(
            f"{name}.mind is not the source its manifest row was derived from: "
            f"size {len(data)} (recorded {row.size}), sha256 {got} (recorded {row.src_sha})"
        )
    return data


def _state_bytes(state: int) -> bytes:
    def read_i64(addr: int, off: int = 0) -> int:
        return ctypes.cast(addr + off, ctypes.POINTER(ctypes.c_int64))[0]

    handle = read_i64(state) if state else 0
    if not handle:
        return b""
    return ctypes.string_at(read_i64(handle), read_i64(handle, 8))


def _mind_mic3(lib: ctypes.CDLL, source: bytes) -> bytes:
    buf = ctypes.create_string_buffer(source, len(source))
    state = lib.selftest_nb_mic3(ctypes.cast(buf, ctypes.c_void_p).value, len(source), 0)
    return _state_bytes(state)


def _mind_mic3_stable(lib: ctypes.CDLL, source: bytes, label: str) -> bytes:
    """The canonical trace, built twice. Differing bytes are a determinism break."""
    first = _mind_mic3(lib, source)
    second = _mind_mic3(lib, source)
    if first != second:
        raise AssertionError(
            f"{label}: canonical trace is NOT deterministic — "
            f"{len(first)}B/{_sha256(first)} then {len(second)}B/{_sha256(second)}"
        )
    return first


def _mind_elf(lib: ctypes.CDLL, source: bytes) -> bytes:
    buf = ctypes.create_string_buffer(source, len(source))
    state = lib.selftest_native_elf_u(
        ctypes.cast(buf, ctypes.c_void_p).value, len(source), 0
    )
    return _state_bytes(state)


def _mindc() -> pathlib.Path:
    mindc = pathlib.Path(os.environ.get("MINDC_BIN", _REPO / "target/release/mindc"))
    if not mindc.is_file():
        raise AssertionError(f"reference mindc missing: {mindc}")
    return mindc


def _rust_mic3(mindc: pathlib.Path, source: bytes, scratch: pathlib.Path) -> bytes:
    src = scratch / "case.mind"
    out = scratch / "case.mic3"
    src.write_bytes(source)
    if out.exists():
        out.unlink()
    proc = subprocess.run(
        [str(mindc), "--emit-mic3", str(out), str(src)],
        cwd=_REPO,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0 or not out.is_file():
        raise AssertionError(
            f"reference MIC3 refused rc={proc.returncode}: {proc.stderr[-800:]!r}"
        )
    return out.read_bytes()


def _reference_consumes(mindc: pathlib.Path, source: bytes, path: pathlib.Path) -> None:
    """Prove the reference compiler reads these exact bytes. The fixture tree is
    VCS-ignored (excluded from the production example walk), so without this
    control nothing observes that a fixture is valid input at all; the
    malformed-append leg is the positive control that the check CAN fail."""
    command = [str(mindc), "check", "--no-fmt", "--no-lint", str(path)]
    path.write_bytes(source)
    checked = subprocess.run(command, capture_output=True, timeout=60, check=False)
    if checked.returncode != 0:
        raise AssertionError(
            f"{path.name}: reference check refused: {checked.stderr[-800:]!r}"
        )
    path.write_bytes(source + b"\nfn deliberately_malformed( {\n")
    malformed = subprocess.run(command, capture_output=True, timeout=60, check=False)
    path.write_bytes(source)
    if malformed.returncode == 0:
        raise AssertionError(f"{path.name}: reference check did not consume its input")
    print(f"  PASS reference consumes {path.name} and rejects its malformed control")


def _run(elf: bytes, path: pathlib.Path) -> int:
    path.write_bytes(elf)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return subprocess.run(core_limit_argv([str(path)]), timeout=30, check=False).returncode


def _evaluator(mindc: pathlib.Path, name: str, expected: int) -> None:
    """Anchor the executed values in the ORDINARY evaluator before the self-host
    artifact is asked to reproduce them."""
    proc = subprocess.run(
        [str(mindc), "test", str(_DATA / f"{name}.mind"), "--threads", "1"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    transcript = proc.stdout + proc.stderr
    if proc.returncode != 0 or f"{expected} passed; 0 failed" not in transcript:
        raise AssertionError(
            f"evaluator {name} failed rc={proc.returncode}: {transcript[-800:]}"
        )
    print(f"  PASS evaluator {name} {expected}/{expected}")


def _bless(lib: ctypes.CDLL, mindc: pathlib.Path) -> int:
    """Re-derive MANIFEST.txt. Roles are preserved from the existing manifest —
    promoting a `blocked` fixture to a `trace` positive is a REVIEWED edit, never
    something a re-bless does on its own."""
    previous = _load_manifest() if _MANIFEST.is_file() else {}
    lines = [_HEADER]
    with tempfile.TemporaryDirectory(prefix="mind-record-trace-bless-") as td:
        scratch = pathlib.Path(td)
        for name in sorted(_fixtures_on_disk()):
            data = (_DATA / f"{name}.mind").read_bytes()
            prior = previous.get(name)
            role = prior.role if prior is not None else _REFUSE
            trace, expected = "-", "-"
            if role == _EVALUATOR:
                expected = str(prior.expected)
            elif role == _TRACE:
                mine = _mind_mic3_stable(lib, data, name)
                oracle = _rust_mic3(mindc, data, scratch)
                if not mine or mine != oracle:
                    raise AssertionError(
                        f"{name}: refusing to record a trace that is empty or differs "
                        f"from the reference (mine={len(mine)}B oracle={len(oracle)}B)"
                    )
                elf = _mind_elf(lib, data)
                if not elf:
                    raise AssertionError(f"{name}: canonical trace without an executable")
                trace = _sha256(mine)
                expected = str(_run(elf, scratch / f"{name}.elf"))
            lines.append(
                f"{name}\t{role}\t{len(data)}\t{_sha256(data)}\t{trace}\t{expected}\n"
            )
    _MANIFEST.write_text("".join(lines), encoding="utf-8")
    print(f"blessed {_MANIFEST} — REVIEW THE DIFF")
    return 0


def main(argv: list[str]) -> int:
    so = resolve_so()
    lib = ctypes.CDLL(str(so))
    for name in ("selftest_nb_mic3", "selftest_native_elf_u"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int64
        fn.argtypes = [ctypes.c_int64] * 3

    mindc = _mindc()
    if "--bless" in argv:
        return _bless(lib, mindc)

    rows = _load_manifest()
    _check_coverage(rows)

    by_role: dict[str, list[str]] = {}
    for name, row in rows.items():
        by_role.setdefault(row.role, []).append(name)
    for role in (_TRACE, _BLOCKED, _REFUSE, _EVALUATOR):
        if not by_role.get(role):
            raise AssertionError(f"manifest records no {role} fixture — gate is vacuous")

    for name in sorted(by_role[_EVALUATOR]):
        _source(name, rows)
        expected = rows[name].expected
        if expected is None:
            raise AssertionError(f"{name}: evaluator row needs a test count")
        _evaluator(mindc, name, expected)

    seen: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="mind-record-trace-") as td:
        scratch = pathlib.Path(td)
        # VALIDITY SWEEP — EVERY fixture, every role. Running this over the
        # `trace` positives alone left the zero-byte roles unobserved: a
        # `blocked`/`refuse` fixture corrupted into a syntax error still emits 0
        # bytes on both paths, so its row would keep passing while proving
        # nothing about the refusal it names. That is precisely the failure this
        # gate's own docstring is written against, so the check covers all roles.
        for name in sorted(rows):
            _reference_consumes(mindc, _source(name, rows), scratch / f"{name}.mind")

        for name in sorted(by_role[_TRACE]):
            row = rows[name]
            source = _source(name, rows)

            mine = _mind_mic3_stable(lib, source, name)
            oracle = _rust_mic3(mindc, source, scratch)
            if mine != oracle:
                raise AssertionError(
                    f"{name}: canonical trace differs from the reference "
                    f"mine={len(mine)}B/{_sha256(mine)} "
                    f"oracle={len(oracle)}B/{_sha256(oracle)}"
                )
            if _sha256(mine) != row.trace_sha:
                raise AssertionError(
                    f"{name}: canonical trace drifted from its recorded evidence "
                    f"got={_sha256(mine)} recorded={row.trace_sha}"
                )
            print(
                f"  PASS trace {name} {len(mine)}B sha256={row.trace_sha} "
                f"== reference, deterministic"
            )

            elf = _mind_elf(lib, source)
            if not elf:
                raise AssertionError(f"{name}: canonical trace without an executable")
            rc = _run(elf, scratch / f"{name}.elf")
            if rc != row.expected:
                raise AssertionError(f"{name}: executed exit {rc}, recorded {row.expected}")
            print(f"  PASS executed {name} exit={rc} elf={len(elf)}B")
            seen[name] = mine

        base, needle, replacement, mutated_exit = _MUTATION
        if rows[base].role != _TRACE:
            raise AssertionError(f"mutation control {base} is not a trace positive")
        source = _source(base, rows)
        if source.count(needle) != 1:
            raise AssertionError(
                f"mutation control is stale: {needle!r} occurs "
                f"{source.count(needle)}x in {base}.mind (want exactly 1)"
            )
        mutated = source.replace(needle, replacement)
        mutated_mic3 = _mind_mic3_stable(lib, mutated, f"{base}+mutation")
        if not mutated_mic3 or mutated_mic3 == seen[base]:
            raise AssertionError("element mutation did not change the canonical bytes")
        if mutated_mic3 != _rust_mic3(mindc, mutated, scratch):
            raise AssertionError("mutated canonical trace differs from the reference")
        mutated_elf = _mind_elf(lib, mutated)
        if not mutated_elf:
            raise AssertionError("element mutation produced no executable")
        rc = _run(mutated_elf, scratch / "mutated.elf")
        if rc != mutated_exit:
            raise AssertionError(
                f"element mutation executed exit {rc}, expected {mutated_exit}"
            )
        print(
            f"  PASS mutation {needle.decode()} -> {replacement.decode()} "
            f"changes the canonical bytes and executes {rc}"
        )

        for role in (_BLOCKED, _REFUSE):
            for name in sorted(by_role[role]):
                source = _source(name, rows)
                trace = _mind_mic3_stable(lib, source, name)
                elf = _mind_elf(lib, source)
                if trace or elf:
                    raise AssertionError(
                        f"{name} ({role}): expected 0 bytes on both paths, got "
                        f"trace={len(trace)}B elf={len(elf)}B. If the fixed-array "
                        f"record layout now lowers canonically, re-derive this "
                        f"fixture as a `trace` positive."
                    )
                print(f"  PASS {role} {name} trace=0B elf=0B")

    print(
        f"ALL PASS canonical record trace: {len(by_role[_TRACE])} reference-equal "
        f"traces executed, {len(by_role[_BLOCKED])} blocked, "
        f"{len(by_role[_REFUSE])} refusals, 1 mutation, "
        f"{len(by_role[_EVALUATOR])} evaluator anchor"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
