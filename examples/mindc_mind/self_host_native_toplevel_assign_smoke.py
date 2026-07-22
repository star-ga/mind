#!/usr/bin/env python3
"""INDEPENDENCE_ROADMAP Phase-C follow-up — TOP-LEVEL STRAIGHT-LINE i64 REASSIGN
(`let w: i64 = 100; w = w + 100;`) now emits+runs in the no-feed native-ELF path.

Prior state (roadmap C2 note + self_host_native_autowrap_smoke.py lines 33-36): a
straight-line reassignment with NO surrounding while-loop returned the EMPTY EmitState
from `selftest_native_elf` — the loop-carried form (`while .. { w = w + 1; }`) worked
but the top-level form failed CLOSED. Root cause was NOT the native `nb_*` emitter (which
already lowers a top-level assign via its env-rebind arm) but the mic@3 trace-hash gate
that fronts `selftest_native_elf`: `selftest_native_elf_u` computes `nb_trace_hash` FIRST
and returns es_new() when it is 0, and the mic@3 statement-sequence planner
(`flatten_stmt_seq`) had NO arm for a top-level `ast_assign` — the statement fell through
to the bare-expression arm, which rejected the (non-expression) assign node and failed the
whole mic@3 build closed, zeroing the note.

The fix adds a top-level-assign arm to `flatten_stmt_seq` that reuses the type-0 `let`
plan path exactly: flatten the RHS (child2) against the CURRENT env, then append a
SHADOWING env entry re-using the target name (child0) so letenv_lookup's last-match-wins
resolves later reads to the reassigned slot. A reassignment emits NO dedicated Assign
instruction (eval/lower.rs rebinds body_env in place), so the mic@3 body is byte-identical
to a `let w2 = w + 100` — verified below against `mindc --emit-mic3` for every fixture.

There is NO frozen native-ELF byte-oracle for arbitrary user programs (the deleted Rust
src/native backend captured only the fixed self-host oracle). The gates here are therefore
(1) mic@3 byte-identity vs the live Rust `--emit-mic3` oracle for each fixture, and
(2) native-ELF EXECUTION correctness on the CPU against an INDEPENDENT Python reference —
each value fixture returns its raw computed value (compared mod-256 to the Python model),
plus a non-fakeable MARKER=42 discrimination fixture whose FULL-64-bit `err == 0` compare
runs BEFORE the exit-code truncation (a wrong lowering leaves a non-zero err and exits 43).
A loop-carried control (already supported) confirms the shared mechanism is intact.

Usage:
  MINDC_SO=/path/to.so MINDC_BIN=/path/to/mindc python3 self_host_native_toplevel_assign_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

MARKER = 42


def mind_native_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    enc = src.encode()
    buf = ctypes.create_string_buffer(enc, len(enc))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(enc))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def mind_mic3(lib, src: str) -> bytes:
    """main.mind's OWN mic@3 emit for a standalone fixture (user_lo=0)."""
    fn = lib.selftest_nb_mic3
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    enc = src.encode()
    buf = ctypes.create_string_buffer(enc, len(enc))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(enc), 0)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def oracle_mic3(mindc: str, src: str, tmp: pathlib.Path) -> bytes:
    p = tmp / "case.mind"
    p.write_text(src)
    out = tmp / "case.mic3"
    r = subprocess.run(
        [mindc, str(p), "--emit-mic3", str(out)], capture_output=True, text=True
    )
    if r.returncode != 0 or not out.exists():
        return b""
    return out.read_bytes()


def run_elf(elf: bytes, tmp: pathlib.Path) -> tuple:
    p = tmp / "case.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([str(p)], capture_output=True)
    return r.returncode, r.stdout


def sixty4(v: int) -> int:
    """two's-complement i64 wrap of a Python int (models i64 arithmetic)."""
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >= (1 << 63) else v


def py_straightline(init: int, ops: list) -> int:
    """INDEPENDENT reference: fold a straight-line i64 reassignment sequence.
    ops = [(op, operand_or_'w'), ...]; 'w' means self-reference (w = w * w)."""
    w = init
    for op, operand in ops:
        rhs = w if operand == "w" else operand
        if op == "+":
            w = sixty4(w + rhs)
        elif op == "-":
            w = sixty4(w - rhs)
        elif op == "*":
            w = sixty4(w * rhs)
        else:
            raise ValueError(op)
    return w


def build_value_src(init: int, ops: list) -> str:
    """`let w = init; w = w OP x; ...; w` — returns the raw final value."""
    lines = [f"    let w: i64 = {init};"]
    for op, operand in ops:
        rhs = "w" if operand == "w" else str(operand)
        lines.append(f"    w = w {op} {rhs};")
    lines.append("    w")
    body = "\n".join(lines)
    return f"fn main() -> i64 {{\n{body}\n}}\n"


def py_value_if(c: int, then_v: int, else_v: int) -> int:
    """INDEPENDENT reference: `w = if c > 0 { then_v } else { else_v }`."""
    return then_v if c > 0 else else_v


def build_value_if_src(c: int, then_v: int, else_v: int) -> str:
    """`let c=..; let w=0; w = if c > 0 { then_v } else { else_v }; w` — raw final w.
    Exercises the value-if RHS in a top-level reassignment (the deferred sub-case)."""
    return (
        "fn main() -> i64 {\n"
        f"    let c: i64 = {c};\n"
        "    let w: i64 = 0;\n"
        f"    w = if c > 0 {{ {then_v} }} else {{ {else_v} }};\n"
        "    w\n"
        "}\n"
    )


def build_marker_value_if_src(c: int, then_v: int, else_v: int, expected: int) -> str:
    """Non-fakeable: the value-if reassign then a FULL-64-bit `w == expected` compare
    that runs BEFORE the exit-code mod-256 truncation (wrong lowering exits 43)."""
    return (
        "fn main() -> i64 {\n"
        f"    let c: i64 = {c};\n"
        "    let w: i64 = 0;\n"
        f"    w = if c > 0 {{ {then_v} }} else {{ {else_v} }};\n"
        f"    if w == {expected} {{\n"
        f"        {MARKER}\n"
        "    } else {\n"
        "        43\n"
        "    }\n"
        "}\n"
    )


def build_marker_src(init: int, ops: list, expected: int) -> str:
    """`let w = init; w = ...; if w == expected { 42 } else { 43 }` — non-fakeable:
    the FULL-64-bit compare runs before the exit-code mod-256 truncation."""
    lines = [f"    let w: i64 = {init};"]
    for op, operand in ops:
        rhs = "w" if operand == "w" else str(operand)
        lines.append(f"    w = w {op} {rhs};")
    lines.append(f"    if w == {expected} {{")
    lines.append(f"        {MARKER}")
    lines.append("    } else {")
    lines.append("        43")
    lines.append("    }")
    body = "\n".join(lines)
    return f"fn main() -> i64 {{\n{body}\n}}\n"


# Loop-carried CONTROL (already supported): 5 iterations of w += 10 -> 50; the same
# env-rebind mechanism the top-level arm reuses. Proves the shared path is intact.
SRC_LOOP_CONTROL = f"""fn main() -> i64 {{
    let w: i64 = 0;
    let i: i64 = 0;
    while i < 5 {{
        w = w + 10;
        i = i + 1;
    }}
    if w == 50 {{
        {MARKER}
    }} else {{
        43
    }}
}}
"""

# Straight-line reassignment shapes: (name, init, ops).
SHAPES = [
    ("single", 100, [("+", 100)]),               # 100 + 100 = 200 (the roadmap gap)
    ("chained", 100, [("+", 100), ("*", 2)]),    # 200 then *2 = 400
    ("selfref", 5, [("-", 1), ("*", "w"), ("+", 2)]),  # 4, 16, 18
    ("many", 1, [("+", 10), ("*", 3), ("-", 7), ("+", 100)]),  # 11,33,26,126
]

# Value-if RHS reassignment shapes (the DEFERRED sub-case now closed):
# `let w=0; w = if c > 0 { then_v } else { else_v };` — BOTH branches, both signs of c.
VALUE_IF_SHAPES = [
    ("vif-then", 1, 10, 20),    # c>0  -> w = 10
    ("vif-else", -1, 10, 20),   # c<=0 -> w = 20
    ("vif-else0", 0, 10, 20),   # c==0 -> w = 20 (boundary)
    ("vif-then42", 7, 42, 99),  # c>0  -> w = 42 (raw-exit lands on MARKER value)
]


def main() -> int:
    so = str(resolve_so())  # MINDC_SO verbatim, else fresh build (never stale)
    mindc = os.environ.get("MINDC_BIN", "./target/release/mindc")
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf"):
        print("FAIL  selftest_native_elf: symbol absent")
        return 1
    if not hasattr(lib, "selftest_nb_mic3"):
        print("FAIL  selftest_nb_mic3: symbol absent")
        return 1

    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        # --- top-level straight-line reassignment shapes ---
        for name, init, ops in SHAPES:
            expected = py_straightline(init, ops)  # INDEPENDENT Python reference

            # (a) raw-value fixture: native ELF exit == py_reference (mod 256)
            vsrc = build_value_src(init, ops)
            velf = mind_native_elf(lib, vsrc)
            v_is_elf = len(velf) > 120 and velf[:4] == b"\x7fELF"
            if not v_is_elf:
                print(f"  FAIL  {name}: TOP-LEVEL assign still fails closed "
                      f"(empty/invalid ELF len={len(velf)})")
                all_ok = False
                continue
            v_exit, v_out = run_elf(velf, tmp)
            want_exit = expected & 0xFF
            v_ok = v_exit == want_exit and v_out == b""

            # (b) mic@3 byte-identity of the raw-value fixture vs the Rust oracle
            mine = mind_mic3(lib, vsrc)
            orc = oracle_mic3(mindc, vsrc, tmp)
            mic_ok = len(orc) > 0 and mine == orc

            # (c) non-fakeable MARKER discrimination (full-64-bit compare pre-truncation)
            msrc = build_marker_src(init, ops, expected)
            melf = mind_native_elf(lib, msrc)
            m_is_elf = len(melf) > 120 and melf[:4] == b"\x7fELF"
            m_exit = run_elf(melf, tmp)[0] if m_is_elf else None
            m_ok = m_is_elf and m_exit == MARKER

            ok = v_ok and mic_ok and m_ok
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {name:8} value_exit={v_exit} "
                f"(py_ref={want_exit}) mic3_id={mic_ok} marker_exit={m_exit} "
                f"(elf {len(velf)}B, zero MLIR/LLVM)"
            )

        # --- value-if RHS reassignment shapes (deferred sub-case now closed) ---
        for name, c, then_v, else_v in VALUE_IF_SHAPES:
            expected = py_value_if(c, then_v, else_v)  # INDEPENDENT Python reference

            # (a) raw-value fixture: native ELF exit == py_reference (mod 256), zero stdout
            vsrc = build_value_if_src(c, then_v, else_v)
            velf = mind_native_elf(lib, vsrc)
            v_is_elf = len(velf) > 120 and velf[:4] == b"\x7fELF"
            if not v_is_elf:
                print(f"  FAIL  {name}: value-if RHS assign fails closed "
                      f"(empty/invalid ELF len={len(velf)})")
                all_ok = False
                continue
            v_exit, v_out = run_elf(velf, tmp)
            want_exit = expected & 0xFF
            v_ok = v_exit == want_exit and v_out == b""

            # (b) mic@3 byte-identity of the raw-value fixture vs the Rust oracle
            mine = mind_mic3(lib, vsrc)
            orc = oracle_mic3(mindc, vsrc, tmp)
            mic_ok = len(orc) > 0 and mine == orc

            # (c) non-fakeable MARKER discrimination -> exit==42 iff w==expected
            msrc = build_marker_value_if_src(c, then_v, else_v, expected)
            melf = mind_native_elf(lib, msrc)
            m_is_elf = len(melf) > 120 and melf[:4] == b"\x7fELF"
            m_exit, m_out = run_elf(melf, tmp) if m_is_elf else (None, None)
            m_ok = m_is_elf and m_exit == MARKER and m_out == b""

            ok = v_ok and mic_ok and m_ok
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {name:10} value_exit={v_exit} "
                f"(py_ref={want_exit}) mic3_id={mic_ok} marker_exit={m_exit} "
                f"(elf {len(velf)}B, zero MLIR/LLVM)"
            )

        # --- loop-carried control (already supported) ---
        celf = mind_native_elf(lib, SRC_LOOP_CONTROL)
        c_is_elf = len(celf) > 120 and celf[:4] == b"\x7fELF"
        c_exit = run_elf(celf, tmp)[0] if c_is_elf else None
        c_ok = c_is_elf and c_exit == MARKER
        all_ok = all_ok and c_ok
        print(
            f"  {'PASS' if c_ok else 'FAIL'}  {'loop-ctrl':8} exit={c_exit} "
            f"expected={MARKER}  (loop-carried assign control, still works)"
        )

    if all_ok:
        print(
            "ALL PASS  top-level i64 reassignment — straight-line (`w=w+100;`) AND "
            "value-if RHS (`w = if c>0 {A} else {B};`) — EMITS a runnable native ELF "
            "and runs correct; the mic@3 trace-hash note is byte-identical to the Rust "
            "--emit-mic3 oracle for every fixture (flatten_stmt_seq top-level-assign "
            "arm + its type-7 value-if sub-case, additive to the self-compile)"
        )
        return 0
    print(
        "FAIL  a top-level assign fixture failed to emit, ran to the wrong value, or its "
        "mic@3 diverged from the oracle — do NOT guess (report the native exit above)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
