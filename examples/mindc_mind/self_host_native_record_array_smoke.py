"""Focused pure-MIND native fixed-array record-field gate."""

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
_DATA = _HERE / "testdata" / "native_record_array"

sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402
from _selfhost_sources import core_limit_argv  # noqa: E402


def _compile(lib: ctypes.CDLL, source: bytes, trace_hash: bytes = bytes(32)) -> bytes:
    src = ctypes.create_string_buffer(source, len(source))
    note = ctypes.create_string_buffer(trace_hash, 32)
    state = lib.selftest_native_elf_h(
        ctypes.cast(src, ctypes.c_void_p).value,
        len(source),
        ctypes.cast(note, ctypes.c_void_p).value,
    )

    def read_i64(addr: int, off: int = 0) -> int:
        return ctypes.cast(addr + off, ctypes.POINTER(ctypes.c_int64))[0]

    handle = read_i64(state)
    if handle == 0:
        return b""
    return ctypes.string_at(read_i64(handle), read_i64(handle, 8))


def _run(elf: bytes, path: pathlib.Path) -> int:
    path.write_bytes(elf)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return subprocess.run(core_limit_argv([str(path)]), check=False).returncode


def _source(name: str) -> bytes:
    return (_DATA / f"{name}.mind").read_bytes()


def _require_refusal(lib: ctypes.CDLL, name: str) -> None:
    got = _compile(lib, _source(name))
    if got:
        raise AssertionError(
            f"{name}: expected 0-byte refusal, emitted {len(got)} bytes"
        )
    print(f"  PASS refusal {name}")


def _reference_tests() -> None:
    mindc = pathlib.Path(os.environ.get("MINDC_BIN", _REPO / "target/release/mindc"))
    if not mindc.is_file():
        raise AssertionError(f"reference mindc missing: {mindc}")
    for name, expected in (("reference", 6), ("reference_u8", 3)):
        proc = subprocess.run(
            [str(mindc), "test", str(_DATA / f"{name}.mind"), "--threads", "1"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        transcript = proc.stdout + proc.stderr
        marker = f"{expected} passed; 0 failed"
        if proc.returncode != 0 or marker not in transcript:
            raise AssertionError(
                f"ordinary evaluator {name} failed rc={proc.returncode}: "
                f"{transcript[-800:]}"
            )
        print(f"  PASS evaluator {name} {expected}/{expected}")


def _scalar_identity(lib: ctypes.CDLL) -> None:
    source = (
        b"fn add(a: i64, b: i64) -> i64 {\n"
        b"    return a + b;\n"
        b"}\n"
        b"fn main() -> i64 {\n"
        b"    return add(2, 3);\n"
        b"}\n"
    )
    oracle = (_HERE / "testdata/native_elf_oracle/add.elf").read_bytes()
    note = oracle[-52:]
    if note[12:16] != b"MIND":
        raise AssertionError("frozen scalar oracle has no MIND note")
    got = _compile(lib, source, note[20:52])
    if got != oracle:
        raise AssertionError(
            "scalar native bytes changed: "
            f"got={hashlib.sha256(got).hexdigest()} "
            f"want={hashlib.sha256(oracle).hexdigest()}"
        )
    print(f"  PASS scalar bytes {hashlib.sha256(got).hexdigest()[:16]}")


def _check_positive(source: bytes, path: pathlib.Path) -> None:
    # Test data are excluded from the production example walk. Copy the exact
    # bytes outside that ignored tree and prove this path is actually consumed.
    mindc = pathlib.Path(os.environ.get("MINDC_BIN", _REPO / "target/release/mindc"))
    command = [str(mindc), "check", "--no-fmt", "--no-lint", str(path)]
    path.write_bytes(source)
    checked = subprocess.run(command, capture_output=True, timeout=30, check=False)
    if checked.returncode != 0:
        raise AssertionError(f"{path.name}: reference check refused: {checked.stderr[-800:]!r}")
    path.write_bytes(source + b"\nfn deliberately_malformed( {\n")
    malformed = subprocess.run(command, capture_output=True, timeout=30, check=False)
    path.write_bytes(source)
    if malformed.returncode == 0:
        raise AssertionError(f"{path.name}: reference check did not consume its input")
    print(f"  PASS reference check and input-consumption control {path.name}")


def main() -> int:
    so = resolve_so()
    lib = ctypes.CDLL(str(so))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3

    _reference_tests()
    _scalar_identity(lib)

    positives = {
        "direct": 29,
        "copy_out": 117,
        "owners": 35,
        "computed_i64": 25,
        "computed_i64_bitwise": 25,
        "record_param_loop": 13,
        "record_param_copy": 23,
        "record_return": 5,
        "record_transport_owners": 37,
        "u8_direct_loop": 118,
        "u8_copy_out": 155,
        "u8_record_return": 144,
    }
    seen: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="mind-native-record-array-") as td:
        scratch = pathlib.Path(td)
        for name, expected in positives.items():
            _check_positive(_source(name), scratch / f"{name}.mind")
            first = _compile(lib, _source(name))
            second = _compile(lib, _source(name))
            if not first or first != second:
                raise AssertionError(f"{name}: empty or non-repeatable native bytes")
            rc = _run(first, scratch / f"{name}.elf")
            if rc != expected:
                raise AssertionError(f"{name}: exit {rc}, expected {expected}")
            seen[name] = first
            print(
                f"  PASS native {name} exit={rc} bytes={len(first)} "
                f"sha256={hashlib.sha256(first).hexdigest()[:16]}"
            )

        runtime = _compile(lib, _source("runtime_oob"))
        if not runtime or _run(runtime, scratch / "runtime_oob.elf") != 77:
            raise AssertionError("runtime_oob: expected emitted ELF exit 77")
        print("  PASS runtime_oob exit=77")

        u8_runtime = _compile(lib, _source("u8_runtime_oob"))
        if not u8_runtime or _run(u8_runtime, scratch / "u8_runtime_oob.elf") != 77:
            raise AssertionError("u8_runtime_oob: expected emitted ELF exit 77")
        print("  PASS u8_runtime_oob exit=77")

        mutated = _source("direct").replace(b"post: 7", b"post: 9")
        mutated_elf = _compile(lib, mutated)
        if not mutated_elf or mutated_elf == seen["direct"]:
            raise AssertionError("value mutation did not change native bytes")
        if _run(mutated_elf, scratch / "mutated.elf") != 31:
            raise AssertionError("value mutation did not change executed result to 31")
        print("  PASS discriminating value mutation exit=31")

        type_mutated = _source("direct").replace(b"xs: [i64; 3]", b"xs: [i16; 3]")
        if _compile(lib, type_mutated):
            raise AssertionError("descriptor mutation i64->i16 did not refuse")
        print("  PASS discriminating descriptor mutation refuses")

        owner_mutated = _source("record_transport_owners").replace(
            b"return read_a(a) + read_b(b);",
            b"return read_a(b) + read_b(b);",
        )
        if _compile(lib, owner_mutated):
            raise AssertionError("record argument owner mutation did not refuse")
        print("  PASS discriminating record-owner mutation refuses")

        extent_mutated = _source("record_return").replace(
            b"xs: [i64; 2]", b"xs: [i64; 3]"
        )
        if _compile(lib, extent_mutated):
            raise AssertionError("record return extent mutation did not refuse")
        print("  PASS discriminating return-extent mutation refuses")

        u8_value_mutated = _source("u8_direct_loop").replace(b"300]", b"301]")
        u8_value_elf = _compile(lib, u8_value_mutated)
        if not u8_value_elf or u8_value_elf == seen["u8_direct_loop"]:
            raise AssertionError("u8 value mutation did not change native bytes")
        if _run(u8_value_elf, scratch / "u8_value_mutated.elf") != 119:
            raise AssertionError("u8 value mutation did not change executed result to 119")
        print("  PASS discriminating u8 value mutation exit=119")

        u8_kind_mutated = _source("u8_direct_loop").replace(b"[u8; 4]", b"[i64; 4]")
        u8_kind_elf = _compile(lib, u8_kind_mutated)
        if not u8_kind_elf or u8_kind_elf == seen["u8_direct_loop"]:
            raise AssertionError("u8->i64 descriptor mutation did not change bytes")
        if _run(u8_kind_elf, scratch / "u8_kind_mutated.elf") != 182:
            raise AssertionError("u8->i64 descriptor mutation did not change result to 182")
        print("  PASS discriminating u8 descriptor mutation exit=182")

    for name in (
        "constant_oob",
        "narrow_refuse",
        "alias_refuse",
        "field_store_refuse",
        "nonliteral_refuse",
        "zero_extent_refuse",
        "extent_cap_refuse",
        "extent_overflow_refuse",
        "aggregate_refuse",
        "i64_field_float_refuse",
        "i64_field_aggregate_refuse",
        "i64_field_comparison_refuse",
        "i64_field_not_refuse",
        "wrong_owner_param_refuse",
        "wrong_owner_return_refuse",
        "return_extent_refuse",
        "duplicate_owner_refuse",
        "u8_float_refuse",
        "u8_aggregate_refuse",
        "u8_comparison_refuse",
        "u8_not_refuse",
        "u8_unproven_let_refuse",
        "u8_alias_refuse",
        "u8_wrong_owner_refuse",
        "u8_constant_oob_refuse",
    ):
        _require_refusal(lib, name)

    print("PASS native record-array gate: 12 ELF positives, 25 refusals, 6 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
