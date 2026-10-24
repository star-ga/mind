"""
Self-host NATIVE main(argc, argv) smoke (W4b) -- proves the pure-MIND native-ELF
emitter binds an entry `fn main(argc: i64, argv: i64)` to the REAL process
argc/argv, with argv in the C `char **` representation (argv + 8*i holds the
pointer to the NUL-terminated argv[i]; argv[argc] is the NULL terminator) --
the same representation the MLIR/clang build hands `main`, and the same pointer
`__mind_argv(i)` returns for argv[i].

Before W4b the entry spilled rdi/rsi into the parameter slots, and at `_start`
those registers are not argc/argv: every such program ran with argc = argv = 0
(the bridge refused the shape as `native.entry_main_has_params`).

Oracles (independent of the emitter under test):
  * the Python-computed expectation (exit code AND stdout) from the argv list the
    harness itself passes to the child, and
  * the MLIR build of the SAME source (`mindc build --backend mlir
    --emit=binary`), run with the same argv.
Every positive case must agree with BOTH. Negative cases (entry shapes the
binding does not cover) must be REFUSED by the self-host emitter (empty image),
never emitted with zeroed parameters.

The self-host image is produced exactly like the seeded production path: the
bundled stdlib is seeded ahead of the program (`selftest_native_elf_u(std ++
prog, len, len(std))`, the mindfuzz_self_host.py contract).

Run:
  MINDC_SO=<self-host .so> MINDC_BIN=<mindc> \
    python3 examples/mindc_mind/self_host_native_main_args_smoke.py
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "scripts"))
from gate_assert import check  # noqa: E402  (one [PASS]/[FAIL] line per case)
from _selfhost_so import resolve_so  # noqa: E402

_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]

_RD = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]  # noqa: E731


def std_blob() -> bytes:
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    return b"\n".join(parts) + b"\n"


def mindc_bin() -> pathlib.Path:
    return pathlib.Path(os.environ.get("MINDC_BIN", str(_REPO / "target" / "release" / "mindc")))


class SelfHost:
    """Seeded pure-MIND native-ELF emission (the production native path)."""

    def __init__(self, so_path):
        self.lib = ctypes.CDLL(str(so_path))
        fn = self.lib.selftest_native_elf_u
        fn.restype = ctypes.c_int64
        fn.argtypes = [ctypes.c_int64] * 3
        self.std = std_blob()

    def elf(self, prog: str) -> bytes:
        src = self.std + prog.encode()
        sb = ctypes.create_string_buffer(src, len(src))
        es = self.lib.selftest_native_elf_u(ctypes.cast(sb, ctypes.c_void_p).value, len(src), len(self.std))
        if not es:
            return b""
        sh = _RD(es, 0)
        if not sh:
            return b""
        n = _RD(sh, 8)
        return ctypes.string_at(_RD(sh, 0), n) if n > 0 else b""


def mlir_binary(prog: str, d: pathlib.Path, name: str):
    """Build `prog` with the MLIR backend; return (path or None, diagnostic)."""
    src = d / f"{name}.mind"
    src.write_text(prog)
    out = d / f"{name}.mlir.bin"
    r = subprocess.run(
        [str(mindc_bin()), "build", str(src), "--release", "--backend", "mlir",
         "--emit=binary", f"--out={out}"],
        capture_output=True, text=True, timeout=600, cwd=str(d))
    if r.returncode != 0 or not out.exists():
        return None, (r.stderr or r.stdout).strip()[-400:]
    return out, ""


def write_exec(path: pathlib.Path, data: bytes) -> pathlib.Path:
    path.write_bytes(data)
    path.chmod(0o755)
    return path


def run(path: pathlib.Path, tail):
    try:
        r = subprocess.run([str(path), *tail], capture_output=True, timeout=20)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return "timeout", b""


# (name, source, [argv tails], expected(tail) -> (exit, stdout)[, mlir_oracle=True])
POSITIVE = [
    (
        "argc_returned",
        "fn main(argc: i64, argv: i64) -> i64 {\n"
        "    return argc;\n"
        "}\n",
        [[], ["a"], ["1", "2", "3", "4", "5"]],
        lambda t: (1 + len(t), b""),
    ),
    (
        "argv1_first_byte",
        "fn main(argc: i64, argv: i64) -> i64 {\n"
        "    if argc < 2 {\n"
        "        return 200;\n"
        "    }\n"
        "    let a: i64 = __mind_load_i64(argv + 8);\n"
        "    return __mind_load_i8(a);\n"
        "}\n",
        [[], ["Q"], ["zeta", "x", "y", "z", "w"]],
        lambda t: (200 if not t else ord(t[0][0]), b""),
    ),
    (
        "argv_echo_all",
        "fn slen(p: i64) -> i64 {\n"
        "    let mut n: i64 = 0;\n"
        "    while __mind_load_i8(p + n) != 0 {\n"
        "        n = n + 1;\n"
        "    }\n"
        "    return n;\n"
        "}\n"
        "fn emit(p: i64) -> i64 {\n"
        "    return __mind_write(1, p, slen(p), 0 - 1);\n"
        "}\n"
        "fn main(argc: i64, argv: i64) -> i64 {\n"
        "    let mut i: i64 = 1;\n"
        "    let mut total: i64 = 0;\n"
        "    while i < argc {\n"
        "        total = total + emit(__mind_load_i64(argv + i * 8));\n"
        "        i = i + 1;\n"
        "    }\n"
        "    return total;\n"
        "}\n",
        [[], ["hello"], ["a", "bc", "def", "ghij", "k"]],
        lambda t: (sum(len(x) for x in t), "".join(t).encode()),
    ),
    (
        "argv_null_terminated",
        "fn main(argc: i64, argv: i64) -> i64 {\n"
        "    if __mind_load_i64(argv + argc * 8) != 0 {\n"
        "        return 9;\n"
        "    }\n"
        "    if __mind_load_i64(argv) == 0 {\n"
        "        return 8;\n"
        "    }\n"
        "    return 7;\n"
        "}\n",
        [[], ["x"], ["1", "2", "3", "4", "5"]],
        lambda t: (7, b""),
    ),
    (
        "params_agree_with_intrinsics",
        "fn main(argc: i64, argv: i64) -> i64 {\n"
        "    if argc != __mind_argc() {\n"
        "        return 3;\n"
        "    }\n"
        "    if argc > 1 {\n"
        "        if __mind_argv(1) != __mind_load_i64(argv + 8) {\n"
        "            return 4;\n"
        "        }\n"
        "    }\n"
        "    return argc * 10 + __mind_argc();\n"
        "}\n",
        [[], ["p"], ["1", "2", "3", "4", "5"]],
        lambda t: (11 * (1 + len(t)), b""),
        False,  # __mind_argc/__mind_argv are native-only intrinsics (no MLIR build)
    ),
    (
        "callee_params_untouched",
        "fn mix(a: i64, b: i64) -> i64 {\n"
        "    return a * 2 + b;\n"
        "}\n"
        "fn main(argc: i64, argv: i64) -> i64 {\n"
        "    return mix(argc, 5) + mix(3, 4);\n"
        "}\n",
        [[], ["y"], ["1", "2", "3", "4", "5"]],
        lambda t: ((1 + len(t)) * 2 + 5 + 10, b""),
    ),
    (
        "argc_only_param",
        "fn main(argc: i64) -> i64 {\n"
        "    return argc + 40;\n"
        "}\n",
        [[], ["a", "b"], ["1", "2", "3", "4", "5"]],
        lambda t: (41 + len(t), b""),
    ),
]

# Entry shapes the binding does not cover: the self-host emitter must refuse
# (empty image), never emit a program whose extra parameters read garbage.
NEGATIVE = [
    (
        "three_params_refused",
        "fn main(argc: i64, argv: i64, envp: i64) -> i64 {\n"
        "    return argc;\n"
        "}\n",
    ),
    (
        "float_param_refused",
        "fn main(x: f64) -> i64 {\n"
        "    return 1;\n"
        "}\n",
    ),
]


def main() -> int:
    so = resolve_so()
    sh = SelfHost(so)
    print(f"[main-args smoke] so={so} mindc={mindc_bin()}")
    ok = True
    with tempfile.TemporaryDirectory(prefix="w4b-") as td:
        d = pathlib.Path(td)
        for case in POSITIVE:
            name, prog, tails, expect = case[:4]
            use_mlir = case[4] if len(case) > 4 else True
            elf = sh.elf(prog)
            if not elf:
                ok = False
                print(f"  FAIL  {name}: self-host refused (empty image)")
                continue
            native = write_exec(d / f"{name}.native.elf", elf)
            mbin, diag = mlir_binary(prog, d, name) if use_mlir else (None, "")
            if use_mlir and mbin is None:
                ok = False
                print(f"  FAIL  {name}: MLIR oracle build failed: {diag}")
                continue
            for tail in tails:
                want = expect(tail)
                got_n = run(native, tail)
                # Native-only intrinsics: the Python model is the sole oracle.
                got_m = run(mbin, tail) if use_mlir else want
                good = got_n == want and got_m == want
                ok = ok and good
                check(good, f"{name} extra_args={len(tail)}: "
                      f"native=(exit {got_n[0]}, stdout {got_n[1]!r}) "
                      + (f"mlir=(exit {got_m[0]}, stdout {got_m[1]!r}) " if use_mlir else "mlir=n/a ")
                      + f"expected=(exit {want[0]}, stdout {want[1]!r})")
        for name, prog in NEGATIVE:
            elf = sh.elf(prog)
            good = len(elf) == 0
            ok = ok and good
            check(good, f"{name}: self-host image "
                  f"{len(elf)}B (expected 0B refusal)")
    if ok:
        print("\nALL PASS  (native main(argc, argv) == OS argc/argv == MLIR)")
        return 0
    print("\nFAIL  main(argc, argv) smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
