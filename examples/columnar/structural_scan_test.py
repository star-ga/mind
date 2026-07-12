#!/usr/bin/env python3
"""Runnable verification for examples/columnar/structural_scan_json.mind.

Phase 1 of the deterministic multi-format columnar front-end: the scalar
structural oracle for JSON. This harness compiles the .mind oracle with
`mindc --emit-shared` and drives both entry points through ctypes over a
hand-verified corpus, asserting the string-aware structural count and the
order-sensitive index checksum against an independent scalar reference.

The corpus includes adversarial documents where structural characters ({ , })
appear INSIDE a JSON string (must NOT be counted) and where a backslash-escaped
quote must not close the string. This is the fail-closed byte-identity ground
truth the Phase-2 SIMD/NEON structural pack must reproduce byte-for-byte.

Usage:
    python3 examples/columnar/structural_scan_test.py [path/to/mindc]
Deterministic: no clock, no randomness; output is a pure function of the bytes.
"""
import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "structural_scan_json.mind"
REPO = HERE.parent.parent  # examples/columnar -> repo root
DEFAULT_MINDC = REPO / "target" / "debug" / "mindc"

MASK = (1 << 64) - 1


def s64(x: int) -> int:
    x &= MASK
    return x - (1 << 64) if x >= (1 << 63) else x


def reference(bs: bytes) -> tuple[int, int]:
    """Independent scalar oracle: string-aware structural count + i64 checksum."""
    i = in_str = count = acc = 0
    n = len(bs)
    while i < n:
        b = bs[i]
        if in_str:
            if b == 0x5C:          # backslash escapes the next byte
                i += 2
            else:
                if b == 0x22:      # unescaped quote closes the string
                    in_str = 0
                i += 1
        else:
            if b == 0x22:          # unescaped quote opens a string
                in_str = 1
            else:
                if b in (0x7B, 0x7D, 0x5B, 0x5D, 0x3A, 0x2C):  # { } [ ] : ,
                    count += 1
                    acc = (acc * 1000003 + i) & MASK
            i += 1
    return count, s64(acc)


def straddle_doc(min_len: int) -> bytes:
    """Build a JSON array whose structural bytes land on and ACROSS every
    16/32/64-byte chunk seam — the Phase-2 SIMD kernel's chunk boundaries.

    Each repeat unit is a string element carrying structural chars that MUST NOT
    count ({ } [ ] : ,) followed by a bare integer element and a REAL structural
    comma that MUST count. The units are dense and the string elements cross
    chunk seams, so an in-string structural byte sits astride every 16/32/64-byte
    boundary: the SIMD in-string mask has to survive a chunk-boundary carry to
    reproduce reference(). Deterministic: a pure function of min_len.
    """
    parts = [b"["]
    total = 1
    while total < min_len:
        for chunk in (b'"a{b}c:d,e[f]g",', b"1234567890,"):
            parts.append(chunk)
            total += len(chunk)
    parts.append(b"0]")
    return b"".join(parts)


CORPUS = [
    ("obj plain",            b'{"a":1,"b":2}'),
    ("arr plain",            b'[1,2,3]'),
    ("adversarial str {,}",  b'{"k":"a,b{c}"}'),
    ("adversarial esc \\\"", b'{"x":"a\\"b","y":9}'),
    ("nested",               b'{"o":{"p":[1,2]},"q":3}'),
    ("empty",                b''),
    # --- F-1-1 hardening: backslash-run parity + chunk-straddle -------------
    # (a) EVEN backslash run (one escaped backslash) then a REAL close-quote,
    #     then structural chars that MUST count.
    ("even bslash run",      b'{"x":"a\\\\","y":1}'),
    # (b) ODD run: escaped-backslash + escaped-quote — the string stays open
    #     through the middle quote and closes only on the real trailing quote.
    ("odd bslash run",       b'{"k":"\\\\\\"","z":[1]}'),
    # (c) trailing backslash as the FINAL byte: the escape-skip must step i past
    #     the buffer end without over-reading.
    ("trailing bslash",      b'{"k":"abc\\'),
    # (d) a \uXXXX escape whose hex spells a structural char (0x7d = '}') — the
    #     literal escape bytes are non-structural and MUST NOT count.
    ("uXXXX structural hex", b'{"u":"\\u007d","v":2}'),
    # (e) chunk-seam straddle: one doc >64 bytes, one >4096 bytes, structural
    #     chars (in- and out-of-string) crossing every 16/32/64-byte seam.
    (">64B straddle",        straddle_doc(80)),
    (">4096B straddle",      straddle_doc(4200)),
]


def main() -> int:
    mindc = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MINDC
    if not mindc.exists():
        print(f"mindc not found: {mindc}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        so = Path(td) / "structural_scan_json.so"
        cp = subprocess.run(
            [str(mindc), str(SRC), "--emit-shared", str(so)],
            capture_output=True, text=True,
        )
        if cp.returncode != 0 or not so.exists():
            print(cp.stdout)
            print(cp.stderr, file=sys.stderr)
            print(f"compile failed (exit {cp.returncode})", file=sys.stderr)
            return 1

        lib = ctypes.CDLL(str(so))
        for name in ("structural_scan", "structural_index_checksum"):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_int64
            fn.argtypes = [ctypes.c_int64, ctypes.c_int64]

        ok = True
        hdr = f"{'doc':22} {'len':>4} {'scan':>5} {'ref':>5} {'checksum':>22} {'ref_cksum':>22}  match"
        print(hdr)
        for name, bs in CORPUS:
            if bs:
                buf = ctypes.create_string_buffer(bs, len(bs))
                addr = ctypes.addressof(buf)
            else:
                addr = 0
            sc = lib.structural_scan(addr, len(bs))
            ck = lib.structural_index_checksum(addr, len(bs))
            rc, ra = reference(bs)
            match = (sc == rc and ck == ra)
            ok = ok and match
            print(f"{name:22} {len(bs):>4} {sc:>5} {rc:>5} {ck:>22} {ra:>22}  "
                  f"{'OK' if match else 'FAIL'}")

    print("ALL_MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
