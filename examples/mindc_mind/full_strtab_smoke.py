"""
Self-host FULL STRTAB COLLECTION smoke (cutover fold, step 2).

Proves the recursive emit's FULL string-table collection — the step-1 body pass
(build_module_strtab: instr strings) THEN the step-2 registry pass
(build_module_registry_strtab: struct type + field names) — equals the real
`mindc --emit-mic3` string table for a field-using module, end to end.

The registry strings are appended in BTreeMap-sorted (alphabetical by struct
name) order, exactly as the Rust oracle's collect_struct_def_strings iterates
module.struct_defs (a BTreeMap<String, Vec<String>>) — NOT declaration order.
Verified live against `mindc --emit-mic3` when the binary is available (the
golden is regenerated from the real artifact, guarding against staleness).

This exercises the additive, collection-only exports in main.mind
(selftest_full_strtab_{count,len,byte}). It does NO mic@3 byte-output and is
fully isolated from the canary (mindc_compile -> emit_fn_def stays stubbed), so
fixed_point_smoke.py stays byte-identical.

Run:  python3 examples/mindc_mind/full_strtab_smoke.py
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# Repo root = .../mind ; the release mindc oracle.
_REPO = _HERE.parent.parent
_MINDC = pathlib.Path(
    os.environ.get("MINDC_BIN", str(_REPO / "target" / "release" / "mindc")))

LOAD = b"__mind_load_i64"

# Each case: (source, [expected full interned strtab in canonical order]).
# Canonical order = instr strings (per-fn first-seen) THEN registry strings
# (structs sorted by name; each: type name then field names in declared order).
CASES = [
    # Single struct, one field read. Instr: get_a, s, __mind_load_i64.
    # Registry: Point, a, b.
    (b"struct Point {\n    a: i64,\n    b: i64,\n}\n"
     b"pub fn get_a(s: Point) -> i64 {\n    s.a\n}\n",
     [b"get_a", b"s", LOAD, b"Point", b"a", b"b"]),

    # Two structs declared OUT of alphabetical order (Zeta then Alpha). Instr
    # strings follow fn order; registry strings sort: Alpha(p,q) then Zeta(x,y).
    (b"struct Zeta {\n    x: i64,\n    y: i64,\n}\n"
     b"struct Alpha {\n    p: i64,\n    q: i64,\n}\n"
     b"pub fn get_zx(z: Zeta) -> i64 {\n    z.x\n}\n"
     b"pub fn get_ap(a: Alpha) -> i64 {\n    a.p\n}\n",
     [b"get_zx", b"z", LOAD, b"get_ap", b"a",
      b"Alpha", b"p", b"q", b"Zeta", b"x", b"y"]),

    # Single struct, three fields, two reads (load name deduped once).
    (b"struct Tri {\n    m: i64,\n    n: i64,\n    o: i64,\n}\n"
     b"pub fn mn(t: Tri) -> i64 {\n    t.m + t.n\n}\n",
     [b"mn", b"t", LOAD, b"Tri", b"m", b"n", b"o"]),

    # Three structs declared C, A, B -> registry sorts A, B, C.
    (b"struct Cc {\n    c0: i64,\n}\n"
     b"struct Aa {\n    a0: i64,\n}\n"
     b"struct Bb {\n    b0: i64,\n}\n"
     b"pub fn rc(c: Cc) -> i64 {\n    c.c0\n}\n",
     [b"rc", b"c", LOAD, b"Aa", b"a0", b"Bb", b"b0", b"Cc", b"c0"]),
]


def _read_uleb(b, i):
    res = 0
    shift = 0
    while True:
        byte = b[i]
        i += 1
        res |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return res, i


def _decode_strtab(mic3_bytes):
    assert mic3_bytes[:4] == b"MIC3", mic3_bytes[:4]
    i = 5  # magic(4) + version(1)
    n, i = _read_uleb(mic3_bytes, i)
    out = []
    for _ in range(n):
        ln, i = _read_uleb(mic3_bytes, i)
        out.append(mic3_bytes[i:i + ln])
        i += ln
    return out


def _oracle_strtab(src):
    """Real --emit-mic3 string table for `src`, or None if mindc unavailable."""
    if not _MINDC.exists():
        return None
    with tempfile.TemporaryDirectory() as d:
        srcp = pathlib.Path(d) / "case.mind"
        outp = pathlib.Path(d) / "case.mic3"
        srcp.write_bytes(src)
        r = subprocess.run(
            [str(_MINDC), "--emit-mic3", str(outp), str(srcp)],
            capture_output=True)
        if r.returncode != 0 or not outp.exists():
            return None
        return _decode_strtab(outp.read_bytes())


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    # Cross-verify every hard-coded golden against the live oracle first (guards
    # golden staleness). A drift here is a HARD FAIL.
    oracle_seen = False
    for src, want in CASES:
        oracle = _oracle_strtab(src)
        if oracle is None:
            continue
        oracle_seen = True
        if oracle != want:
            label = src.split(b"\n")[0].decode()
            print(f"[FAIL] oracle mismatch for golden: {label}")
            print(f"        golden: {want}")
            print(f"        oracle: {oracle}")
            return 1
    if oracle_seen:
        print(f"[ok ] goldens cross-verified vs real --emit-mic3 ({_MINDC})")
    else:
        print(f"[warn] mindc oracle not found at {_MINDC}; using static goldens")

    lib = ctypes.CDLL(str(SO))
    for fn in ("selftest_full_strtab_count",
               "selftest_full_strtab_len",
               "selftest_full_strtab_byte"):
        getattr(lib, fn).restype = ctypes.c_int64
    lib.selftest_full_strtab_count.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.selftest_full_strtab_len.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    lib.selftest_full_strtab_byte.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for src, want_names in CASES:
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        n = lib.selftest_full_strtab_count(addr, len(src))
        got_names = []
        for i in range(n):
            ln = lib.selftest_full_strtab_len(addr, len(src), i)
            name = bytes(
                lib.selftest_full_strtab_byte(addr, len(src), i, j) & 0xFF
                for j in range(ln)
            )
            got_names.append(name)
        ok = got_names == want_names
        status = "ok " if ok else "FAIL"
        label = src.split(b"\n")[0].decode()
        print(f"[{status}] {label} ...")
        if not ok:
            print(f"        want: {want_names}")
            print(f"        got:  {got_names}")
            failures += 1

    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print(f"\n{len(CASES)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
