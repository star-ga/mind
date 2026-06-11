"""
Self-host RECURSIVE-EMIT STRTAB COLLECTION smoke (cutover fold, step 1).

Proves the recursive emit's string-collection pass (build_module_strtab ->
collect_fn_strings -> collect_body_strings -> collect_expr_strings) interns the
synthetic "__mind_load_i64" call name when an ast_field (kind 17) appears in a
function body, in the EXACT first-seen order the Rust oracle's
collect_instr_strings(Instr::Call) produces.

This exercises the additive collection-only path in main.mind
(selftest_collect_field_strings_{count,len,byte} exports). It does NO mic@3
byte-output and is fully isolated from the canary (mindc_compile ->
lower_program -> emit_fn_def stays stubbed), so fixed_point_smoke.py stays
byte-identical.

These are self-consistency unit goldens: for each module source the expected
interned string table is the FIRST-SEEN traversal order — per top-level fn in
sequence: fn name, each param name, then any body Call name (here the synthetic
"__mind_load_i64" for each field read, interned once, idempotent). Struct type
and field names are NOT part of the body traversal (the oracle interns those in
the separate struct_defs registry pass), so they do not appear here.

Run:  python3 examples/mindc_mind/collect_field_strings_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

LOAD = b"__mind_load_i64"

# Each case: (source, [expected interned string names in first-seen order]).
CASES = [
    # Single field read: fn name, param name, then the load-call name.
    (b"pub fn get_a(s: Point) -> i64 { s.a }\n",
     [b"get_a", b"s", LOAD]),
    # Field at index > 0 still lowers to one __mind_load_i64 Call (CONST+BINOP
    # intern nothing) — same three strings.
    (b"pub fn get_b(s: Point) -> i64 { s.b }\n",
     [b"get_b", b"s", LOAD]),
    # Two field reads in one body intern the load name ONCE (idempotent dedup).
    (b"pub fn sum2(s: Point) -> i64 { s.a + s.b }\n",
     [b"sum2", b"s", LOAD]),
    # A let-bound field read: collect recurses the let init (an ast_field).
    (b"pub fn la(s: Point) -> i64 { let t: i64 = s.a; t }\n",
     [b"la", b"s", LOAD]),
    # No field access -> the load name is NOT interned (only fn + param names).
    (b"pub fn add(a: i64, b: i64) -> i64 { a + b }\n",
     [b"add", b"a", b"b"]),
    # Two fns, second has a field read: first-seen order across the module.
    (b"pub fn id(x: i64) -> i64 { x }\npub fn gx(s: Point) -> i64 { s.a }\n",
     [b"id", b"x", b"gx", b"s", LOAD]),
    # The load name from fn0 is shared (interned once) when fn1 also reads.
    (b"pub fn g0(s: Point) -> i64 { s.a }\npub fn g1(t: Point) -> i64 { t.b }\n",
     [b"g0", b"s", LOAD, b"g1", b"t"]),
]


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    for fn in ("selftest_collect_field_strings_count",
               "selftest_collect_field_strings_len",
               "selftest_collect_field_strings_byte"):
        getattr(lib, fn).restype = ctypes.c_int64
    lib.selftest_collect_field_strings_count.argtypes = [
        ctypes.c_int64, ctypes.c_int64]
    lib.selftest_collect_field_strings_len.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    lib.selftest_collect_field_strings_byte.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for src, want_names in CASES:
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        n = lib.selftest_collect_field_strings_count(addr, len(src))
        got_names = []
        for i in range(n):
            ln = lib.selftest_collect_field_strings_len(addr, len(src), i)
            name = bytes(
                lib.selftest_collect_field_strings_byte(addr, len(src), i, j) & 0xFF
                for j in range(ln)
            )
            got_names.append(name)
        ok = got_names == want_names
        status = "ok " if ok else "FAIL"
        label = src.split(b"\n")[0].decode()
        print(f"[{status}] {label}")
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
