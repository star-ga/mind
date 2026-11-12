#!/usr/bin/env python3
"""Regression gate for #308 (closes #286's break/continue fixtures): the pure-MIND
native-ELF compiler must NEVER silently miscompile a loop with a dead statement after
an unconditional `continue` into an INFINITE LOOP — at ANY loop nesting depth.

Root cause it guards: parse_for desugars `for i in 0..N { continue; }` to
`let i=0; while i<N { i=i+1; continue; i=i+1 }`, appending a tail increment that is
DEAD after the unconditional continue. Before the fix, nb_while_carry's last-wins carry
latched the loop-carried `i` onto that dead slot, so the live increment wrote a slot the
condition never re-read -> i never advanced -> hang. The fix (nb_truncate_dead) drops
unreachable statements after a top-level `continue` for both the carry pre-walk and the
emit.

Nested loops: the emit truncated an inner loop's body, but the ENCLOSING loop's carry
pre-walk (nb_while_carry_wnest) and the count walk (nb_count_stmt's while arm) still
walked the inner body at its RAW length. They threaded next_id through the dead tail,
so every outer carried var assigned after the inner loop got a predicted slot past the
one the emit really writes; the outer condition re-read a never-updated slot and the
OUTER loop hung (`for i in 0..3 { for j in 0..2 { continue; } }`). Every walker now uses
the same truncated length. The NESTED fixtures below are the regression gate for that.

Exit edges: a carried var's working slot is its LAST body write, and the header (or
the post-loop code) re-reads that slot. A conditional `continue`/`break` taken AFTER
an earlier write and BEFORE the last one left the live value in the earlier slot, so
`i = i + 1; if i == 5 { continue; } i = i + 0;` re-entered the loop with the previous
iteration's i and never terminated, and accumulators lost updates. The emitter now
writes every carried var back to its working slot on each exit edge
(nb_loops_exit_writeback). The EXIT-EDGE WRITE-BACK fixtures below gate that, at one
level and across nested loops.

DEAD_TAIL_IDENTICAL proves the truncation is applied by EVERY id-threading walker, not
just by enough of them for the loop to terminate: a loop body with a dead tail must
compile to the SAME code segment (first PT_LOAD, so the trace-hash note — which hashes
the mic@3 that keeps the dead tail — is excluded) as the same program without it. A
walker that still reserved ids for the dead tail shows up as a larger frame.

This smoke compiles each fixture via the frozen pure-MIND stage1.elf (what
`mindc build --backend native` runs) and asserts the emitted ELF TERMINATES with the
expected exit — a timeout is a hard FAIL (a wedged CI red beats a silently-hanging
miscompile shipping green). An optional argument names a DIFFERENT native compiler ELF
(e.g. one derived from the current main.mind by the frozen seed) so a main.mind fix can
be gated before the frozen seed is advanced.

Exit: 0 all pass; 1 a fixture hung, refused, returned wrong, or emitted different code;
2 BLOCKED (missing compiler ELF).
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


def compile_elf(compiler: pathlib.Path, prog: str):
    """(elf_bytes or None, compiler_exit)."""
    std = std_blob()
    comb = std + prog.encode()
    img = struct.pack("<qq", len(std), len(comb)) + comb
    r = subprocess.run([str(compiler)], input=img, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    out = r.stdout
    if len(out) < 4 or out[:4] != b"\x7fELF":
        return None, r.returncode
    return out, r.returncode


def compile_and_run(compiler: pathlib.Path, prog: str, run_timeout: int = 6):
    out, comp = compile_elf(compiler, prog)
    if out is None:
        return ("no-elf", 0, comp)
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


def code_segment(elf: bytes) -> bytes:
    """Bytes of the first PT_LOAD (ELF header + code); excludes the MIND trace note."""
    phoff = struct.unpack_from("<Q", elf, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", elf, 54)
    for k in range(phnum):
        p_type, _flags, p_offset = struct.unpack_from("<IIQ", elf, phoff + k * phentsize)
        p_filesz = struct.unpack_from("<Q", elf, phoff + k * phentsize + 32)[0]
        if p_type == 1:  # PT_LOAD
            return elf[p_offset:p_offset + p_filesz]
    raise ValueError("no PT_LOAD segment")


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
    # audit Finding 2: a dead statement after a BARE break poisons the loop-carried
    # var's POST-LOOP value (last-wins carry records the never-run dead slot). Must return 1.
    ("break_dead_tail_value",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { i = i + 1; break; i = i + 1; } return i; }", 1),
    ("while_dead_tail_then_post_read",
     "fn main() -> i64 { let mut j: i64 = 0; while j < 2 { j = j + 1; continue; j = j + 1; } let t: i64 = j + 5; return t; }", 7),
    # ---- NESTED: the enclosing loop's carry/count must skip the inner dead tail ----
    ("nested_for_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..2 { continue; } } return s; }", 0),
    ("nested_for_continue_then_outer_stmt",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..2 { continue; } s = s + 1; } return s; }", 3),
    ("nested_for_count_then_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..2 { s = s + 1; continue; } } return s; }", 6),
    ("nested_for_inner_reads_outer_var",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..3 { s = s + i * j; continue; } } return s; }", 9),
    ("nested_for_both_levels_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..2 { s = s + 1; continue; } s = s + 10; continue; } return s; }", 36),
    ("nested_for_outer_continue_only",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..2 { s = s + 1; } continue; } return s; }", 6),
    ("triple_for_continue_mid_stmt",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..2 { for j in 0..2 { for k in 0..2 { continue; } s = s + 1; } } return s; }", 4),
    ("triple_for_count_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..2 { for j in 0..3 { for k in 0..4 { s = s + 1; continue; } } } return s; }", 24),
    ("for_continue_in_while",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 4 { for j in 0..3 { s = s + j; continue; } i = i + 1; } return s; }", 12),
    ("while_continue_in_for",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { let mut j: i64 = 0; while j < 4 { j = j + 1; s = s + 1; continue; } } return s; }", 12),
    ("nested_while_dead_tail",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { i = i + 1; let mut k: i64 = 0; while k < 2 { k = k + 1; continue; k = k + 1; } } return i; }", 3),
    ("nested_while_dead_tail_outer_incr_after",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; continue; j = j + 1; } i = i + 1; } return i; }", 3),
    # the dead tail writes an OUTER var: it must neither be carried nor shift the slots
    ("nested_while_dead_tail_writes_outer",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; s = s + 1; continue; s = s + 100; } i = i + 1; } return s; }", 6),
    ("nested_while_break_dead_tail",
     "fn main() -> i64 { let mut i: i64 = 0; let mut t: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; break; j = j + 1; } t = t + j; i = i + 1; } return t * 10 + i; }", 33),
    ("triple_while_dead_tail",
     "fn main() -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; while i < 2 { let mut j: i64 = 0; while j < 2 { let mut k: i64 = 0; while k < 2 { k = k + 1; continue; k = k + 5; } j = j + 1; s = s + k; } i = i + 1; } return s * 10 + i; }", 82),
    # ---- adversarial: each HUNG through the pre-fix compiler; values cross-checked
    # against the Rust MLIR path (`mindc --emit-shared`), which agrees on every one ----
    # an outer carried var written BEFORE and AFTER the inner loop, reading the inner counter
    ("outer_var_before_and_after_inner",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 3 { s = s + 1; let mut j: i64 = 0; while j < 2 { j = j + 1; continue; j = j + 9; } s = s + j; i = i + 1; } return s; }", 9),
    # two sibling inner loops: the second's slots must follow the first's LIVE length
    ("two_sequential_inner_fors_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..2 { s = s + 1; continue; } for k in 0..4 { s = s + 10; continue; } } return s; }", 126),
    # an outer var written ONLY in the dead tail: never carried, never changed
    ("dead_tail_only_write_not_carried",
     "fn main() -> i64 { let mut z: i64 = 0; let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; continue; z = 7; } i = i + 1; } return z * 10 + i; }", 3),
    ("quad_for_continue",
     "fn main() -> i64 { let mut s: i64 = 0; for a in 0..2 { for b in 0..2 { for c in 0..2 { for d in 0..3 { s = s + 1; continue; } s = s + 10; } } } return s; }", 104),
    ("for_outer_continue_inner_while_break_tail",
     "fn main() -> i64 { let mut t: i64 = 0; for i in 0..4 { let mut j: i64 = 0; while j < 5 { j = j + 2; break; j = j + 1; } t = t + j; continue; } return t; }", 8),
    # a dead `let` in the tail must not reserve a binding or a slot
    ("inner_dead_tail_let",
     "fn main() -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; s = s + j; continue; let q: i64 = 5; s = s + q; } i = i + 1; } return s; }", 9),
    # dead tails at BOTH levels (the outer one after its own `continue`)
    ("outer_while_dead_tail_after_inner",
     "fn main() -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; while i < 3 { for j in 0..2 { s = s + 1; continue; } i = i + 1; continue; i = i + 50; } return s * 10 + i; }", 63),
    # ---- EXIT-EDGE WRITE-BACK: a carried var written BEFORE a conditional
    # `continue`/`break` and AGAIN after it. The header (and the post-loop code)
    # read the var's working slot = its LAST write; on the exit edge the live
    # value still sits in the EARLIER write's slot. Without the write-back the
    # `continue` re-enters the loop with the previous iteration's value — the
    # induction variable never advances (HANG) or an accumulator loses the
    # update (wrong value). Values cross-checked against the Rust MLIR path. ----
    ("while_continue_induction_written_after",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 10 { i = i + 1; if i == 5 { continue; } i = i + 0; } return i; }", 10),
    ("while_continue_write_before_and_after",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 3 { i = i + 1; s = s + 1; if i == 2 { continue; } s = s + 100; } return s - 100; }", 103),
    ("while_break_write_before_and_after",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 10 { i = i + 1; s = s + 1; if i == 3 { break; } s = s + 100; } return s - 200; }", 3),
    ("break_post_loop_reads_live_counter",
     "fn main() -> i64 { let mut j: i64 = 0; let mut u: i64 = 0; while j < 20 { j = j + 2; u = j + 0; if j == 8 { break; } j = j + 1; } return j * 10 + u; }", 88),
    ("continue_two_stale_carried_vars",
     "fn main() -> i64 { let mut a: i64 = 0; let mut b: i64 = 0; let mut i: i64 = 0; while i < 4 { i = i + 1; a = a + 1; b = b + 2; if i == 2 { continue; } a = a + 10; b = b + 20; } return a + b; }", 102),
    # the working slot is an if-MERGE slot (the var's last write is inside an if)
    ("continue_post_is_if_merge_slot",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 3 { i = i + 1; s = s + 1; if i == 2 { continue; } if i >= 0 { s = s + 10; } } return s; }", 23),
    # nested: the OUTER `continue` follows an inner loop that wrote the outer var
    ("outer_continue_after_inner_write",
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 3 { i = i + 1; for j in 0..2 { s = s + j; continue; } if i == 2 { continue; } s = s + 100; } return s; }", 203),
    # nested: the INNER loop's own exit edges inside an outer range-for
    ("inner_continue_write_back_in_for",
     "fn main() -> i64 { let mut t: i64 = 0; for k in 0..3 { let mut j: i64 = 0; while j < 4 { j = j + 1; t = t + 1; if j == 2 { continue; } t = t + 10; } } return t; }", 102),
    ("inner_break_write_back_in_for",
     "fn main() -> i64 { let mut t: i64 = 0; for k in 0..2 { let mut j: i64 = 0; while j < 10 { j = j + 1; t = t + 1; if j == 3 { break; } t = t + 100; } } return t - 400; }", 6),
]
# No-hang guard: shapes the native backend may REFUSE (0 bytes, honest fail-closed —
# an if-wrapped inner loop, a `continue` on an if-branch that also writes) but must
# NEVER hang and must never return a wrong value if it does emit.
NO_HANG = [
    ("if_wrapped_inner_while_dead_tail",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { if i >= 0 { let mut j: i64 = 0; while j < 2 { j = j + 1; continue; j = j + 1; } } i = i + 1; } return i; }", 3),
    ("nested_for_continue_in_if",
     "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..4 { if j == 1 { continue; } s = s + 1; } } return s; }", 9),
    # `return` inside a nested loop, followed by a dead tail (the MLIR path returns 40)
    ("nested_while_return_dead_tail",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; return i + 40; j = j + 5; } i = i + 1; } return 99; }", 40),
]
# (name, program WITH an unreachable tail, the SAME program without it): the native
# emitter must produce an identical code segment for both.
DEAD_TAIL_IDENTICAL = [
    ("single_while_continue_tail",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { i = i + 1; continue; i = i + 1; } return i; }",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { i = i + 1; continue; } return i; }"),
    ("nested_while_continue_tail",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; continue; j = j + 1; } i = i + 1; } return i; }",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; continue; } i = i + 1; } return i; }"),
    ("nested_while_break_tail",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; break; j = j + 1; } i = i + 1; } return i; }",
     "fn main() -> i64 { let mut i: i64 = 0; while i < 3 { let mut j: i64 = 0; while j < 2 { j = j + 1; break; } i = i + 1; } return i; }"),
]


def main(argv: list[str]) -> int:
    compiler = pathlib.Path(argv[0]).resolve() if argv else STAGE1
    if not compiler.exists():
        print(f"BLOCKED: native compiler ELF missing at {compiler}")
        return 2
    print(f"compiler = {compiler}")
    fails = []
    for name, prog, want in FIXTURES:
        kind, val, comp = compile_and_run(compiler, prog)
        if kind == "HANG":
            fails.append(f"{name}: HUNG (infinite loop — the #308 silent miscompile)")
        elif kind == "no-elf":
            fails.append(f"{name}: compiler emitted no ELF (comp_exit={comp})")
        elif val != want:
            fails.append(f"{name}: exit {val}, expected {want}")
        else:
            print(f"  ok    {name}: exit {val}")
    for name, prog, want in NO_HANG:
        kind, val, comp = compile_and_run(compiler, prog)
        if kind == "HANG":
            fails.append(f"{name}: HUNG (must fail-close or terminate, never hang)")
        elif kind == "exit" and val != want:
            fails.append(f"{name}: exit {val}, expected {want} or a fail-closed refusal")
        else:
            print(f"  ok    {name}: no-hang ({kind} {val if kind == 'exit' else comp})")
    for name, with_tail, without in DEAD_TAIL_IDENTICAL:
        a, ca = compile_elf(compiler, with_tail)
        b, cb = compile_elf(compiler, without)
        if a is None or b is None:
            fails.append(f"{name}: compiler emitted no ELF (comp_exit={ca}/{cb})")
        elif code_segment(a) != code_segment(b):
            fails.append(f"{name}: dead tail changed the emitted code "
                         f"({len(code_segment(a))}B vs {len(code_segment(b))}B) — a "
                         f"walker still reserves ids for the unreachable tail")
        else:
            print(f"  ok    {name}: dead tail emits identical code ({len(code_segment(a))}B)")
    if fails:
        print("FAIL  self-host continue/break regression gate (#308/#286):")
        for f in fails:
            print("   -", f)
        return 1
    print("PASS  self-host continue/break: no dead-tail-continue hang at any loop nesting; "
          "all fixtures terminate correctly; dead tails emit identical code")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
