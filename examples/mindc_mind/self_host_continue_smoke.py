#!/usr/bin/env python3
"""Regression gate for #308 (closes #286's break/continue fixtures): the pure-MIND
native-ELF compiler must NEVER silently miscompile a loop with a dead statement after
an unconditional `continue` into an INFINITE LOOP.

Root cause it guards: parse_for desugars `for i in 0..N { continue; }` to
`let i=0; while i<N { i=i+1; continue; i=i+1 }`, appending a tail increment that is
DEAD after the unconditional continue. Before the fix, nb_while_carry's last-wins carry
latched the loop-carried `i` onto that dead slot, so the live increment wrote a slot the
condition never re-read -> i never advanced -> hang. The fix (nb_truncate_dead) drops
unreachable statements after a top-level `continue` for both the carry pre-walk and the
emit. This smoke compiles each fixture via the frozen pure-MIND stage1.elf and asserts
the emitted ELF TERMINATES with the expected exit — a timeout is a hard FAIL (a wedged
CI red beats a silently-hanging miscompile shipping green).

Exit: 0 all pass; 1 a fixture hung or returned wrong; 2 BLOCKED (missing stage1.elf).
"""
import struct
import subprocess
import pathlib
import os
import tempfile
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE1 = HERE / "testdata" / "selfhost_loop" / "stage1.elf"
STD_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring", "json", "map",
    "net", "process", "reactor", "regex", "ring", "sha256", "string", "time", "toml",
    "tui", "vec",
]


def std_blob() -> bytes:
    return b"\n".join((REPO / "std" / f"{m}.mind").read_bytes() for m in STD_MODULES) + b"\n"


def compile_and_run(prog: str, run_timeout: int = 6):
    std = std_blob()
    comb = std + prog.encode()
    img = struct.pack("<qq", len(std), len(comb)) + comb
    r = subprocess.run([str(STAGE1)], input=img, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60)
    out = r.stdout
    if len(out) < 4 or out[:4] != b"\x7fELF":
        return ("no-elf", len(out), r.returncode)
    with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
        f.write(out)
        p = f.name
    os.chmod(p, 0o755)
    try:
        rc = subprocess.run([p], timeout=run_timeout).returncode
        return ("exit", rc, None)
    except subprocess.TimeoutExpired:
        return ("HANG", None, None)
    finally:
        os.unlink(p)


# (name, program, expected_exit) — each MUST compile to an ELF that terminates.
FIXTURES = [
    ("for_bare_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { continue; } return s; }", 0),
    ("for_continue_then_stmt",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { continue; s = s + 1; } return s; }", 0),
    ("for_stmt_then_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { s = s + 0; continue; } return s; }", 0),
    ("while_dead_tail_continue",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 3 { i = i + 1; continue; i = i + 1; } return s; }", 0),
    ("while_continue_no_dead_tail",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 5 { i = i + 1; continue; } return i; }", 5),
    ("for_sum_no_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { s = s + i; } return s; }", 10),
    ("for_break",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..10 { if i == 5 { break; } s = s + 1; } return s; }", 5),
    # Fable audit Finding 2: a dead statement after a BARE break poisons the loop-carried
    # var's POST-LOOP value (last-wins carry records the never-run dead slot). Must return 1.
    ("break_dead_tail_value",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { i = i + 1; break; i = i + 1; } return i; }", 1),
]
# No-hang guard (Fable audit Finding 1): shapes that must NEVER hang — either fail-closed
# (0 bytes, honest) or terminate. Nested loops currently fail-close upstream, but a future
# nested-for lowering must not reintroduce the dead-tail-continue hang.
NO_HANG = [
    ("nested_for_continue", "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..2 { continue; } } return s; }"),
    ("nested_while_dead_tail", "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { i = i + 1; let mut k: i64 = 0; while k < 2 { k = k + 1; continue; k = k + 1; } } return i; }"),
]


def main() -> int:
    if not STAGE1.exists():
        print(f"BLOCKED: frozen stage1.elf missing at {STAGE1}")
        return 2
    fails = []
    for name, prog, want in FIXTURES:
        kind, val, comp = compile_and_run(prog)
        if kind == "HANG":
            fails.append(f"{name}: HUNG (infinite loop — the #308 silent miscompile)")
        elif kind == "no-elf":
            fails.append(f"{name}: compiler emitted no ELF (bytes={val}, comp_exit={comp})")
        elif val != want:
            fails.append(f"{name}: exit {val}, expected {want}")
        else:
            print(f"  ok    {name}: exit {val}")
    for name, prog in NO_HANG:
        kind, val, comp = compile_and_run(prog)
        if kind == "HANG":
            fails.append(f"{name}: HUNG (must fail-close or terminate, never hang)")
        else:
            print(f"  ok    {name}: no-hang ({kind} {val})")
    if fails:
        print("FAIL  self-host continue/break regression gate (#308/#286):")
        for f in fails:
            print("   -", f)
        return 1
    print("PASS  self-host continue/break: no dead-tail-continue hang; all fixtures terminate correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
