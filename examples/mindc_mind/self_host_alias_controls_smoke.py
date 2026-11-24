"""Focused native alias-environment controls.

Each case is compiled twice: by the current pure-MIND stage1 executable and by
the independently built Rust/evaluator shared library.  The generated native
ELFs must both run to the expected exit value and agree byte-for-byte.  The
cases target exact-name lookup, immutable branch frames, width joins, and
declared-width value-if initializers.
"""

import ctypes
import hashlib
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import _stdlib_manifest  # noqa: E402
from _selfhost_sources import start_child  # noqa: E402
from _selfhost_sources import core_limit_argv  # noqa: E402

SO = pathlib.Path(os.environ["MINDC_SO"])
SEED = pathlib.Path(os.environ.get(
    "MINDC_NATIVE_ELF",
    str(HERE / "testdata" / "selfhost_loop" / "stage1.elf"),
))
P = ctypes.POINTER(ctypes.c_int64)


def i64(addr, off=0):
    return int(ctypes.cast(addr + off, P)[0])


def emit_so(lib, combined, user_lo):
    buf = ctypes.create_string_buffer(combined, len(combined))
    es = lib.selftest_native_elf_u(
        ctypes.cast(buf, ctypes.c_void_p).value, len(combined), user_lo,
    )
    if not es:
        return b""
    sh = i64(es, 0)
    if not sh:
        return b""
    n = i64(sh, 8)
    return ctypes.string_at(i64(sh, 0), n) if n > 0 else b""


def emit_seed(seed, image):
    p = start_child(
        [str(seed)], image, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120, check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"seed exit {p.returncode}: {p.stderr[:160]!r}")
    return p.stdout


def run_elf(blob):
    with tempfile.NamedTemporaryFile(prefix="mind-alias-control-", delete=False) as f:
        path = pathlib.Path(f.name)
        f.write(blob)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    try:
        return subprocess.run(core_limit_argv([str(path)]), timeout=8, check=False).returncode
    finally:
        path.unlink(missing_ok=True)


CASES = [
    (
        "djb2_ab_ba_exact_names",
        "fn compute() -> i64 {\n"
        "    let Ab: i8 = 1;\n"
        "    let BA: i64 = 1000;\n"
        "    return Ab + BA;\n"
        "}\nfn main() -> i64 { return compute(); }\n",
        233,  # 1001 modulo the process exit byte
    ),
    (
        "branch_growth_same_name_shadow",
        "fn compute(c: i64) -> i64 {\n"
        "    let x: i8 = 7;\n"
        "    if c == 1 {\n"
        "        let x: i16 = 99;\n"
        "        if c == 1 { let x: i32 = 101; x; }\n"
        "    }\n"
        "    return x;\n"
        "}\nfn main() -> i64 { return compute(1); }\n",
        7,
    ),
    (
        "sibling_nested_scope_isolation",
        "fn compute(c: i64) -> i64 {\n"
        "    let x: i8 = 11;\n"
        "    if c == 1 {\n"
        "        let sibling: i16 = 22;\n"
        "        if c == 1 { let nested: i32 = 33; nested; }\n"
        "    }\n"
        "    if c == 1 { let other: i64 = 44; other; }\n"
        "    return x;\n"
        "}\nfn main() -> i64 { return compute(1); }\n",
        11,
    ),
    (
        "typed_wide_join_preserves_outer",
        "fn compute(c: i64) -> i64 {\n"
        "    let x: i8 = 5;\n"
        "    let y: i64 = if c == 1 {\n"
        "        let x: i64 = 700;\n"
        "        x\n"
        "    } else {\n"
        "        x\n"
        "    };\n"
        "    return y;\n"
        "}\nfn main() -> i64 { return compute(0); }\n",
        5,
    ),
    (
        "alias_value_if_declared_width_predicate",
        "type Byte = i8;\n"
        "fn compute(c: i64) -> i64 {\n"
        "    let y: Byte = if c == 1 { 300 } else { 300 };\n"
        "    if y > 255 { return 1; }\n"
        "    return 0;\n"
        "}\nfn main() -> i64 { return compute(1); }\n",
        0,
    ),
    (
        "builtin_value_if_declared_width_predicate",
        "fn compute(c: i64) -> i64 {\n"
        "    let y: i8 = if c == 1 { 300 } else { 300 };\n"
        "    if y > 255 { return 1; }\n"
        "    return 0;\n"
        "}\nfn main() -> i64 { return compute(1); }\n",
        0,
    ),
    (
        "called_and_uncalled_alias_isolation",
        "type Byte = i8;\n"
        "fn unused() -> i64 {\n"
        "    let hidden: Byte = if 1 == 1 { 300 } else { 300 };\n"
        "    if hidden > 255 { return 99; }\n"
        "    return 98;\n"
        "}\n"
        "fn compute() -> i64 {\n"
        "    let visible: Byte = if 1 == 1 { 7 } else { 7 };\n"
        "    return visible;\n"
        "}\nfn main() -> i64 { return compute(); }\n",
        7,
    ),
]


def main():
    if not SO.is_file() or not SEED.is_file():
        print("BLOCKED: missing explicit MINDC_SO/MINDC_NATIVE_ELF")
        return 2
    std = _stdlib_manifest.seed_blob()
    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_u.restype = ctypes.c_int64
    lib.selftest_native_elf_u.argtypes = [ctypes.c_int64] * 3
    all_ok = True
    for name, program, expected in CASES:
        combined = std + program.encode()
        image = __import__("struct").pack("<qq", len(std), len(combined)) + combined
        pure = emit_seed(SEED, image)
        oracle = emit_so(lib, combined, len(std))
        pure_rc = run_elf(pure) if pure[:4] == b"\x7fELF" else None
        oracle_rc = run_elf(oracle) if oracle[:4] == b"\x7fELF" else None
        equal = pure == oracle
        ok = equal and pure_rc == expected and oracle_rc == expected
        all_ok &= ok
        print(
            f"{'PASS' if ok else 'FAIL'} {name} pure={len(pure)}:{hashlib.sha256(pure).hexdigest()[:16]} "
            f"oracle={len(oracle)}:{hashlib.sha256(oracle).hexdigest()[:16]} "
            f"equal={equal} rc={pure_rc}/{oracle_rc} want={expected}"
        )
    print("ALL PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


raise SystemExit(main())
