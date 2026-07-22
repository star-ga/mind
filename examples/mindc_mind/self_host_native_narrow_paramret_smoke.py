#!/usr/bin/env python3
"""Byte-behavior smoke for narrow-int (i8/i16/i32) PARAM + RETURN auto-wrap in the
pure-MIND native-ELF path.

The nb_* native emitter wraps a narrow-declared param IN PLACE in its home slot at
entry (nb_wrap_params_w, driven by nb_width_of_ann) and wraps a narrow-declared
return just before the epilogue (nb_wrap_ret_w, driven by frt_lookup_width off the
`-> iN {` signature). This proves both ends end-to-end, and — with sign-sensitive
i64-returning cases that the return-wrap CANNOT mask — proves the param wrap really
happens at entry (not just the return wrap).

Every expected exit code is computed by an INDEPENDENT Python two's-complement
width-wrap reference (`wrap` below); the emitted native ELF must match it exactly.
A control i64 fn (no wrap) confirms full-width fns are unaffected. A headline case
lands on exit 42.

Env: MINDC_SO (prebuilt .so) or MINDC_BIN (default mindc).
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")


def wrap(v: int, w: int) -> int:
    """Two's-complement signed wrap of v to width w (w in {8,16,32,64})."""
    if w >= 64:
        # keep within i64 (the reference callers stay in range)
        v &= (1 << 64) - 1
        return v - (1 << 64) if v & (1 << 63) else v
    mask = (1 << w) - 1
    v &= mask
    return v - (1 << w) if v & (1 << (w - 1)) else v


def ref_exit(arg: int, pw: int, op: str, k: int, rw: int) -> int:
    """Reference for `fn f(x: i<pw>) -> i<rw> { return x <op> k; }` called f(arg)."""
    xp = wrap(arg, pw)          # param wrap at entry
    if op == "+":
        t = xp + k
    elif op == "/":
        # MIND / native uses truncating (toward zero) signed division
        t = int(xp / k) if (xp < 0) != (k < 0) and xp % k != 0 else xp // k
    else:
        raise ValueError(op)
    r = wrap(t, rw)             # return wrap before epilogue
    return r & 0xFF             # process exit code is the low unsigned byte


def ref_exit_sign(arg: int, pw: int) -> int:
    """Reference for `fn f(x: i<pw>) -> i64 { if x < 0 {return 1;} return 0; }`."""
    return 1 if wrap(arg, pw) < 0 else 0


def build_so() -> str:
    so = os.environ.get("MINDC_SO")
    if so:
        return so
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    r = subprocess.run([mindc, MAIN_MIND, "--emit-shared", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stderr[-3000:])
        sys.exit(1)
    return out


def main() -> int:
    so = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        return 1
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_h"):
        print("FAIL: selftest_native_elf_h absent")
        return 1
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]

    def emit(src: str) -> bytes:
        b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
        h = ctypes.create_string_buffer(b"\x00" * 32, 32)
        es = lib.selftest_native_elf_h(
            ctypes.cast(b, ctypes.c_void_p).value, len(src.encode()),
            ctypes.cast(h, ctypes.c_void_p).value,
        )
        sh = rd(es, 0)
        ln = rd(sh, 8)
        return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""

    def run(src: str):
        elf = emit(src)
        if not elf:
            return None, b"", 0
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "m.elf"
            p.write_bytes(elf)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            r = subprocess.run([str(p)], timeout=10, capture_output=True)
            return r.returncode, r.stdout, len(elf)

    # (label, src, want_exit, want_stdout)
    def add_case(name, pw, rw, arg, k=1):
        src = (f"fn f(x: i{pw}) -> i{rw} {{ return x + {k}; }}\n"
               f"fn main() -> i64 {{ return f({arg}); }}\n")
        return (name, src, ref_exit(arg, pw, "+", k, rw), b"")

    def sign_case(name, pw, arg):
        src = (f"fn f(x: i{pw}) -> i64 {{ if x < 0 {{ return 1; }} return 0; }}\n"
               f"fn main() -> i64 {{ return f({arg}); }}\n")
        return (name, src, ref_exit_sign(arg, pw), b"")

    def div_case(name, pw, arg, k):
        src = (f"fn f(x: i{pw}) -> i64 {{ return x / {k}; }}\n"
               f"fn main() -> i64 {{ return f({arg}); }}\n")
        return (name, src, ref_exit(arg, pw, "/", k, 64), b"")

    cases = [
        # headline: i8 param+ret wrap lands exactly on 42
        add_case("i8 param+ret add: f(41) -> 42", 8, 8, 41),
        # i8 param wraps negative, return wraps too
        add_case("i8 param+ret add: f(200) -> wrap", 8, 8, 200),
        # i16 param+ret wrap
        add_case("i16 param+ret add: g(70000) -> wrap", 16, 16, 70000),
        # i32 param+ret wrap (small, exercises movsxd path)
        add_case("i32 param+ret add: h(5) -> 6", 32, 32, 5),
        # --- param-wrap ISOLATED (return is i64, cannot mask the param high bits) ---
        sign_case("i8 param SIGN isolate: f(200) -> neg -> 1", 8, 200),
        sign_case("i8 param SIGN isolate: f(41) -> pos -> 0", 8, 41),
        sign_case("i16 param SIGN isolate: g(40000) -> neg -> 1", 16, 40000),
        sign_case("i32 param SIGN isolate: h(3000000000) -> neg -> 1", 32, 3000000000),
        div_case("i8 param DIV isolate: f(200)/2 -> -28 -> 228", 8, 200, 2),
    ]
    # control: i64 fn — NO wrap, must be unchanged.
    controls = [
        ("i64 control add: f(41) -> 42",
         "fn f(x: i64) -> i64 { return x + 1; }\nfn main() -> i64 { return f(41); }\n",
         42, b""),
        ("i64 control SIGN: f(200) -> pos -> 0 (no wrap)",
         "fn f(x: i64) -> i64 { if x < 0 { return 1; } return 0; }\nfn main() -> i64 { return f(200); }\n",
         0, b""),
    ]

    all_ok = True
    saw_headline_42 = False
    wrapped_ran = 0
    for name, src, want, want_out in cases:
        rc, out, ln = run(src)
        ok = rc == want and out == want_out and ln > 0
        all_ok = all_ok and ok
        if ok and ln > 0:
            wrapped_ran += 1
        if "-> 42" in name and want == 42:
            saw_headline_42 = saw_headline_42 or (ok and rc == 42)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: exit {rc} (want {want}), "
              f"stdout {out!r} (want {want_out!r}), elf {ln}B")

    ctrl_ran = 0
    for name, src, want, want_out in controls:
        rc, out, ln = run(src)
        ok = rc == want and out == want_out and ln > 0
        all_ok = all_ok and ok
        if ok:
            ctrl_ran += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: exit {rc} (want {want}), "
              f"stdout {out!r} (want {want_out!r}), elf {ln}B")

    if not saw_headline_42:
        print("FAIL: vacuous (headline exit==42 case did not pass)")
        return 1
    if wrapped_ran < 4:
        print("FAIL: vacuous (too few narrow param/return cases ran)")
        return 1
    if ctrl_ran < 1:
        print("FAIL: vacuous (no i64 control ran)")
        return 1
    if all_ok:
        print("ALL PASS  narrow-int (i8/i16/i32) param + return auto-wrap matches the "
              "independent width-wrap reference; i64 controls unaffected")
        return 0
    print("FAIL  narrow param/return wrap mismatch vs reference")
    return 1


if __name__ == "__main__":
    sys.exit(main())
