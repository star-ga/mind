#!/usr/bin/env python3
"""
RI-E-1 — native-ELF ELEMENT-WISE TENSOR ADD *value* smoke.

`a + b` over two `[i64; N]` arrays lowers to a SINGLE whole-array BinOp{Add} in
the IR (eval/lower.rs does not expand it elementwise), so the native-ELF backend
owns the expansion. Before RI-E-1 that binop either GP-added two base POINTERS (a
running-ELF WRONG VALUE) or, after the RI-E-0a fix, refused. This smoke is the
gate for the CORRECT emit.

There is NO frozen native byte-oracle for tensors (the deleted Rust native
backend rejected them), so the gate is EXECUTION CORRECTNESS: emit the ELF from
the pure-MIND emitter (`selftest_native_elf_h`), write it, chmod +x, RUN it, and
assert the process exit code. A successful build proves nothing here — only the
executed artifact's value counts.

Case kinds:
  * "OK"  -> the ELF must be non-empty, run, and exit with the wanted code.
  * "0B"  -> the emitter must REFUSE (zero bytes). Fail-closed is always
             acceptable; a running ELF with a wrong value never is.

Exit: 0 all pass, 1 drift/wrong value, 2 blocked (no .so).

Run:
  MINDC_SO=/path/libmindc_mind.so python3 \
      examples/mindc_mind/self_host_native_tensor_ewadd_value_smoke.py
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

EXIT_PASS = 0
EXIT_DRIFT = 1
EXIT_BLOCKED = 2

_Int64Ptr = ctypes.POINTER(ctypes.c_int64)


def _rd(addr, off=0):
    return int(ctypes.cast(addr + off, _Int64Ptr)[0])


def emit_elf(lib, src):
    """Emit the native ELF for `src` through the pure-MIND emitter."""
    raw = src.encode()
    sbuf = ctypes.create_string_buffer(raw, len(raw))
    hbuf = ctypes.create_string_buffer(bytes(32), 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(sbuf, ctypes.c_void_p).value,
        len(raw),
        ctypes.cast(hbuf, ctypes.c_void_p).value,
    )
    handle = _rd(es, 0)
    if handle == 0:
        return b""
    length = _rd(handle, 8)
    if length <= 0:
        return b""
    return ctypes.string_at(_rd(handle, 0), length)


def run_elf(elf):
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / "ewadd_artifact"
    p.write_bytes(elf)
    p.chmod(0o755)
    return subprocess.run([str(p)], timeout=15).returncode


# The core RI-E-1 shape, spelled out so the failure mode is unambiguous.
_ABC = (
    "fn main() -> i64 {\n"
    "    let a: [i64; 3] = [1, 2, 3];\n"
    "    let b: [i64; 3] = [4, 5, 6];\n"
    "    let c: [i64; 3] = a + b;\n"
    "%s"
    "}\n"
)

CASES = [
    # --- THE task gate: [1,2,3] + [4,5,6] == [5,7,9], and c[0] == 5 ---
    ("ewadd c[0] == 5", _ABC % "    return c[0];\n", 5, "OK"),
    ("ewadd c[1] == 7", _ABC % "    return c[1];\n", 7, "OK"),
    ("ewadd c[2] == 9", _ABC % "    return c[2];\n", 9, "OK"),
    # All three elements at once (a partial/short write cannot hide behind one probe).
    (
        "ewadd whole vector == [5,7,9]",
        _ABC
        % (
            "    if c[0] == 5 {\n"
            "        if c[1] == 7 {\n"
            "            if c[2] == 9 {\n"
            "                return 42;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return 1;\n"
        ),
        42,
        "OK",
    ),
    ("ewadd sum of result == 21", _ABC % "    return c[0] + c[1] + c[2];\n", 21, "OK"),
    # --- the operands must NOT be clobbered by the result buffer ---
    (
        "operands intact after ewadd",
        _ABC % "    return a[2] * 10 + b[0];\n",
        34,
        "OK",
    ),
    # --- DIRECT array literals as operands (no let binding) ---
    (
        "direct literal operands",
        "fn main() -> i64 {\n    let c: [i64; 3] = [1, 2, 3] + [4, 5, 6];\n    return c[1];\n}\n",
        7,
        "OK",
    ),
    # --- longer vectors exercise the per-element SSA-slot lockstep (frame sizing) ---
    (
        "n=8 last element",
        "fn main() -> i64 {\n"
        "    let a: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];\n"
        "    let b: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];\n"
        "    let c: [i64; 8] = a + b;\n"
        "    return c[7];\n"
        "}\n",
        88,
        "OK",
    ),
    (
        "n=8 first element",
        "fn main() -> i64 {\n"
        "    let a: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];\n"
        "    let b: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];\n"
        "    let c: [i64; 8] = a + b;\n"
        "    return c[0];\n"
        "}\n",
        11,
        "OK",
    ),
    # --- surrounding scalar code must still be correct (frame not undersized) ---
    (
        "scalar work after ewadd",
        "fn main() -> i64 {\n"
        "    let a: [i64; 3] = [1, 2, 3];\n"
        "    let b: [i64; 3] = [4, 5, 6];\n"
        "    let c: [i64; 3] = a + b;\n"
        "    let s: i64 = c[0] + c[1];\n"
        "    let t: i64 = s * 2;\n"
        "    return t + c[2];\n"
        "}\n",
        33,
        "OK",
    ),
    # deferred (fail-CLOSED, NOT a miscompile): CHAINING an ewadd result back into
    # another ewadd. The result `c` is let-bound to a BinOp node, so its let
    # descriptor is not an `[...]` literal and its length is not statically
    # resolvable -> the admission predicate rejects and the module refuses (0
    # bytes). Upgrade path: teach nb_let_descriptor / nb_tensor_desc to carry a
    # derived array descriptor (length + element dtype) through an admitted ewadd,
    # then this becomes an "OK" case returning 24.
    (
        "refuse chained ewadd (result operand, deferred)",
        "fn main() -> i64 {\n"
        "    let a: [i64; 2] = [1, 2];\n"
        "    let b: [i64; 2] = [10, 20];\n"
        "    let c: [i64; 2] = a + b;\n"
        "    let d: [i64; 2] = a + c;\n"
        "    return d[1];\n"
        "}\n",
        None,
        "0B",
    ),
    # Same class spelled inline — a nested binop operand is equally unresolvable.
    (
        "refuse nested ewadd expression (deferred)",
        "fn main() -> i64 {\n"
        "    let a: [i64; 2] = [1, 2];\n"
        "    let b: [i64; 2] = [10, 20];\n"
        "    let c: [i64; 2] = (a + b) + b;\n"
        "    return c[0];\n"
        "}\n",
        None,
        "0B",
    ),
    # --- scalar binops MUST be untouched by the tensor arm ---
    ("scalar add unaffected", "fn main() -> i64 {\n    return 6 + 7;\n}\n", 13, "OK"),
    (
        "scalar add over array elements unaffected",
        "fn main() -> i64 {\n    let a: [i64; 3] = [1, 2, 3];\n    return a[0] + a[2];\n}\n",
        4,
        "OK",
    ),
    # --- FAIL-CLOSED: every non-admissible tensor binop must still refuse (0 bytes) ---
    (
        "refuse tensor MUL",
        "fn main() -> i64 {\n"
        "    let a: [i64; 3] = [1, 2, 3];\n"
        "    let b: [i64; 3] = [4, 5, 6];\n"
        "    let c: [i64; 3] = a * b;\n"
        "    return c[0];\n"
        "}\n",
        None,
        "0B",
    ),
    (
        "refuse tensor SUB",
        "fn main() -> i64 {\n"
        "    let a: [i64; 3] = [1, 2, 3];\n"
        "    let b: [i64; 3] = [4, 5, 6];\n"
        "    let c: [i64; 3] = a - b;\n"
        "    return c[0];\n"
        "}\n",
        None,
        "0B",
    ),
    (
        "refuse array + scalar",
        "fn main() -> i64 {\n"
        "    let a: [i64; 3] = [1, 2, 3];\n"
        "    let s: i64 = 4;\n"
        "    let c: [i64; 3] = a + s;\n"
        "    return c[0];\n"
        "}\n",
        None,
        "0B",
    ),
    (
        "refuse length mismatch",
        "fn main() -> i64 {\n"
        "    let a: [i64; 2] = [1, 2];\n"
        "    let b: [i64; 3] = [4, 5, 6];\n"
        "    let c: [i64; 2] = a + b;\n"
        "    return c[0];\n"
        "}\n",
        None,
        "0B",
    ),
    (
        "refuse float element tier",
        "fn main() -> f64 {\n"
        "    let a: [f64; 2] = [1.0, 2.0];\n"
        "    let b: [f64; 2] = [4.0, 5.0];\n"
        "    let c: [f64; 2] = a + b;\n"
        "    return c[0];\n"
        "}\n",
        None,
        "0B",
    ),
    # ------------------------------------------------------------------
    # HARDENING (adversarial). Every case below was ADMITTED by the first
    # RI-E-1 admission gate and emitted a RUNNING ELF WITH A WRONG VALUE —
    # the fail-open silent-miscompile class. The gate now proves the element
    # kind (nb_ewadd_elem_ok) instead of trusting a dtype classifier that
    # defaults everything it does not recognise to INT.
    # ------------------------------------------------------------------
    # POINTER ELEMENTS #1 — a nested array. Element 0 is an ast_array_lit,
    # which the AST-only classifier fell through to "INT": the emit GP-added
    # the two heap ROW POINTERS and stored the sum as c[0].
    (
        "refuse nested-array elements (pointer add)",
        "fn main() -> i64 {\n"
        "    let a = [[1, 2], [3, 4]];\n"
        "    let b = [[5, 6], [7, 8]];\n"
        "    let c = a + b;\n"
        "    return 7;\n"
        "}\n",
        None,
        "0B",
    ),
    # POINTER ELEMENTS #2 — the same pointer laundered through an IDENT. No
    # dtype classifier can catch this one (a row pointer IS dtype-INT); only
    # the let-env aggregate-descriptor check closes it.
    (
        "refuse ident-bound array elements (laundered pointer add)",
        "fn main() -> i64 {\n"
        "    let r: [i64; 2] = [1, 2];\n"
        "    let a = [r, r];\n"
        "    let b = [r, r];\n"
        "    let c = a + b;\n"
        "    return 7;\n"
        "}\n",
        None,
        "0B",
    ),
    # POINTER ELEMENTS #3 — struct-literal elements.
    (
        "refuse struct-literal elements (pointer add)",
        "struct P {\n"
        "    x: i64,\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let a = [P { x: 1 }, P { x: 2 }];\n"
        "    let b = [P { x: 3 }, P { x: 4 }];\n"
        "    let c = a + b;\n"
        "    return 7;\n"
        "}\n",
        None,
        "0B",
    ),
    # POINTER ELEMENTS #4 — string-literal elements.
    (
        "refuse string-literal elements (pointer add)",
        "fn main() -> i64 {\n"
        '    let a = ["ab", "cd"];\n'
        '    let b = ["ef", "gh"];\n'
        "    let c = a + b;\n"
        "    return 7;\n"
        "}\n",
        None,
        "0B",
    ),
    # FLOAT ELEMENTS THROUGH AN IDENT — nb_array_elems ACCEPTS this array as a
    # genuine f64 array (it uses the table-aware classifier), but the first
    # admission gate used the AST-only one, saw a bare ident, called it INT and
    # GP-added the raw IEEE-754 bit patterns. Only a SYNTACTIC float literal was
    # caught before; the identical value behind a name was not.
    (
        "refuse ident-bound f64 elements (raw-bit add)",
        "fn main() -> i64 {\n"
        "    let p: f64 = 1.5;\n"
        "    let q: f64 = 2.5;\n"
        "    let a = [p, q];\n"
        "    let b = [p, q];\n"
        "    let c = a + b;\n"
        "    return 7;\n"
        "}\n",
        None,
        "0B",
    ),
    # The float-through-ident refusal must NOT be a blanket ident refusal: an
    # INT-typed ident element is still a legitimate scalar and must keep working.
    (
        "admit int-bound ident elements",
        "fn main() -> i64 {\n"
        "    let p: i64 = 1;\n"
        "    let q: i64 = 2;\n"
        "    let a = [p, q];\n"
        "    let b = [p, q];\n"
        "    let c = a + b;\n"
        "    return c[0] * 10 + c[1];\n"
        "}\n",
        24,
        "OK",
    ),
    # ------------------------------------------------------------------
    # COUNT/EMIT LET-ENV SYMMETRY (the lockstep root fix). nb_stmt's assign arm
    # rebinds NAME with slit_node 0, ERASING its array descriptor; nb_count_stmt
    # now performs the identical rebind, and nb_count_stmt's while arm now
    # restores the count scope after the body exactly as nb_emit_while does.
    # Emit and count therefore resolve the SAME descriptors at the SAME points.
    # ------------------------------------------------------------------
    # Re-assigned operand: the descriptor is gone from BOTH envs -> fail closed
    # (never a length-0/garbage expansion). Deferred upgrade: carry the
    # descriptor through an ident-RHS assign, which makes this an OK case.
    (
        "refuse ewadd after operand reassignment (env symmetry)",
        "fn main() -> i64 {\n"
        "    let mut a: [i64; 3] = [1, 2, 3];\n"
        "    let b: [i64; 3] = [4, 5, 6];\n"
        "    let t: [i64; 3] = a;\n"
        "    a = t;\n"
        "    let c = a + b;\n"
        "    return c[0];\n"
        "}\n",
        None,
        "0B",
    ),
    # An UNRELATED assignment before the ewadd must not disturb it (the rebind
    # is name-scoped, and the count arm mints no id for it).
    (
        "ewadd after unrelated assignment",
        "fn main() -> i64 {\n"
        "    let mut z: i64 = 1;\n"
        "    z = z + 4;\n"
        "    let a: [i64; 3] = [1, 2, 3];\n"
        "    let b: [i64; 3] = [4, 5, 6];\n"
        "    let c: [i64; 3] = a + b;\n"
        "    return c[0] + z;\n"
        "}\n",
        10,
        "OK",
    ),
    # ewadd INSIDE a while body, with a carried assign in the same body: the
    # body-local result let now dies at loop exit in the COUNT env exactly as it
    # does in the EMIT env. A frame undersize here is an SSA slot collision.
    (
        "ewadd inside while body (scope restore)",
        "fn main() -> i64 {\n"
        "    let a: [i64; 3] = [1, 2, 3];\n"
        "    let b: [i64; 3] = [4, 5, 6];\n"
        "    let mut i: i64 = 0;\n"
        "    let mut acc: i64 = 0;\n"
        "    while i < 3 {\n"
        "        let c: [i64; 3] = a + b;\n"
        "        acc = acc + c[0];\n"
        "        i = i + 1;\n"
        "    }\n"
        "    return acc;\n"
        "}\n",
        15,
        "OK",
    ),
    # Same loop, but the ewadd result is read through a second body-local let —
    # more body-local bindings to drop, and scalar work after the loop that must
    # still land on the counted frame.
    (
        "ewadd in while body then scalar work after loop",
        "fn main() -> i64 {\n"
        "    let a: [i64; 2] = [10, 20];\n"
        "    let b: [i64; 2] = [1, 2];\n"
        "    let mut i: i64 = 0;\n"
        "    let mut acc: i64 = 0;\n"
        "    while i < 2 {\n"
        "        let c: [i64; 2] = a + b;\n"
        "        let d: i64 = c[0] + c[1];\n"
        "        acc = acc + d;\n"
        "        i = i + 1;\n"
        "    }\n"
        "    let e: i64 = acc * 2;\n"
        "    return e + 1;\n"
        "}\n",
        133,
        "OK",
    ),
]


def main():
    so = os.environ.get("MINDC_SO")
    if not so or not pathlib.Path(so).exists():
        print("BLOCKED: set MINDC_SO to the built libmindc_mind.so")
        return EXIT_BLOCKED
    lib = ctypes.CDLL(so)
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3

    ok = True
    for label, src, want, kind in CASES:
        try:
            elf = emit_elf(lib, src)
        except Exception as exc:  # noqa: BLE001 - emitter fault is a hard fail
            print(f"  [FAIL] {label}: emitter raised {exc!r}")
            ok = False
            continue
        if kind == "0B":
            good = len(elf) == 0
            got = f"{len(elf)} bytes"
        else:
            if not elf:
                good, got = False, "REFUSED (0 bytes)"
            else:
                try:
                    rc = run_elf(elf)
                    got = f"rc={rc}"
                    good = rc == want
                except Exception as exc:  # noqa: BLE001
                    good, got = False, f"run error {exc!r}"
            got = f"{got} want rc={want}"
        if not good:
            ok = False
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: {got}")

    print("ALL PASS" if ok else "SOME FAIL")
    return EXIT_PASS if ok else EXIT_DRIFT


if __name__ == "__main__":
    sys.exit(main())
