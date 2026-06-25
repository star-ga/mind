"""
Self-host NATIVE-ELF smoke (Rust-independence #14, PHASE 1.3) — proves the
pure-MIND front-end can emit a static x86-64 ELF64 byte-identical to the Rust
`mind-native` backend (src/native/mod.rs) for the SCALAR i64 subset.

This is the FIRST increment of porting src/native into main.mind. It exercises
the additive `selftest_native_elf_h` export (SECTION 4c in main.mind), which is
ISOLATED from the mic@1 canary / mic@3 flip — so the keystone stays consistent.

The oracle is regenerated LIVE from a freshly-built `mind-native` binary (guards
golden staleness). NO fake wins — the pass requires a byte-for-byte match of the
pure-MIND-emitted ELF vs the Rust `mind-native` ELF, AND the pure-MIND ELF must
run and exit with the fixture's value (add(2,3) = 5).

The 32-byte ir_trace_hash that anchors the ELF's PT_NOTE is not yet ported to
pure MIND (FOLLOW-ON increment); the harness reads it from the oracle ELF's note
and passes it in, so the note is byte-identical. The instruction stream, the four
phdrs, and the ELF skeleton are ALL emitted in pure MIND.

Run:  python3 examples/mindc_mind/self_host_native_elf_smoke.py
"""

import ctypes
import os
import pathlib
import stat
import subprocess
import sys

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# The mind-native oracle binary (built with --features native-backend). CI / local
# point MIND_NATIVE_BIN at the freshly-built debug/release binary.
_REPO = _HERE.parent.parent
_DEFAULT_NATIVE = pathlib.Path("/tmp/mind-native-target/debug/mind-native")
MIND_NATIVE = pathlib.Path(os.environ.get("MIND_NATIVE_BIN", str(_DEFAULT_NATIVE)))

# The native-port ladder: each fixture is lowered by the pure-MIND
# `selftest_native_elf_h` AND by the Rust `mind-native` oracle; the gate requires a
# byte-for-byte match of the two ELFs AND that the pure-MIND ELF runs + exits with
# the fixture's value. Each rung adds the emit_seq arms it needs (ported from
# src/native/mod.rs). PHASE 1.3 of Rust-independence #14.
FIXTURES = [
    # add/main — the scalar slice (ConstI64/BinOp-add/Param/Call/Return). exit 5.
    (
        "add",
        (
            "fn add(a: i64, b: i64) -> i64 {\n"
            "    return a + b;\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return add(2, 3);\n"
            "}\n"
        ),
        5,
    ),
    # if_ret — `If` control flow (cmp/setcc/je/jmp + a diverging then-branch +
    # synthetic-else dst-copy) + the Eq comparison op. exit 1.
    (
        "if_ret",
        (
            "fn f(c: i64) -> i64 {\n"
            "    if c == 0 {\n"
            "        return 1;\n"
            "    }\n"
            "    return 2;\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return f(0);\n"
            "}\n"
        ),
        1,
    ),
    # value_if — an if-EXPRESSION bound to a `let` (`let m = if a > b { a } else
    # { b }`): both branches FALL THROUGH (the if_ret then-branch diverged), the
    # then_result/else_result are the branch VALUES (params a/b, not the leading
    # ConstI64), a join `jmp` over the else-block, and a let-env so `return m`
    # resolves to the if's dst slot. Exercises the Gt comparison op too. exit 7.
    (
        "value_if",
        (
            "fn f(a: i64, b: i64) -> i64 {\n"
            "    let m: i64 = if a > b { a } else { b };\n"
            "    return m;\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return f(3, 7);\n"
            "}\n"
        ),
        7,
    ),
    # recursion — RECURSIVE intra-module calls + a non-value `if` statement with a
    # DIVERGING then-branch and NO else (`if n < 2 { return n; }` then a
    # fall-through `return`), the `Lt` comparison (cmp/setl/movzx), and `Sub`/`Add`
    # BinOps (fib(n-1) + fib(n-2)). Exercises the same emit_seq arms already ported
    # for if_ret/value_if — the `If`-statement path (nb_if_stmt), comparison ops,
    # and the linker's PC-relative call patching — over a self-referential callee.
    # exit 13 (fib(7)).
    (
        "recursion",
        (
            "fn fib(n: i64) -> i64 {\n"
            "    if n < 2 {\n"
            "        return n;\n"
            "    }\n"
            "    return fib(n - 1) + fib(n - 2);\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return fib(7);\n"
            "}\n"
        ),
        13,
    ),
    # struct_field — the Option-C i64-handle struct ABI: a `struct P { x, y }`
    # literal `P { x: 7, y: 9 }` lowers (eval/lower.rs StructLit all-i64 path) to
    # CONST 8 ; CONST n_fields ; MUL ; __mind_alloc(size) -> ptr ; per-field
    # __mind_store_i64(addr, val) with offset 8*i ; the field read `p.x` to a
    # __mind_load_i64(ptr + 8*idx). Exercises SECTION 4c's new ast_struct_lit /
    # ast_field arms + the inlined bump-arena alloc/store/load intrinsics, all
    # byte-identical to the Rust mind-native ELF. exit 7 (p.x).
    (
        "struct_field",
        (
            "struct P {\n"
            "    x: i64,\n"
            "    y: i64,\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    let p: P = P { x: 7, y: 9 };\n"
            "    return p.x;\n"
            "}\n"
        ),
        7,
    ),
]


def oracle_elf(src: str, tmp: pathlib.Path) -> bytes:
    src_path = tmp / "fixture_add.mind"
    elf_path = tmp / "rust.elf"
    src_path.write_text(src)
    subprocess.run(
        [str(MIND_NATIVE), str(src_path), str(elf_path)],
        capture_output=True,
        check=True,
    )
    return elf_path.read_bytes()


# ---------------------------------------------------------------------------
# SEEDED whole-module rung (Phase 1.3 #14, stdlib-seeding fix).
#
# The Rust `mind-native` backend, before lowering the user file, SEEDS the
# bundled standard library so `use std.*` free-function callees resolve
# (src/bin/mind-native.rs + src/project/stdlib.rs::STDLIB_MIND_SOURCES). It
# parses each std/*.mind separately and prepends their items, then dead-code
# prunes to the user file's call graph. The pure-MIND emitter mirrors this in
# the 4-arg `selftest_native_elf_hb(src, len, hash, user_lo)`: it lexes ONE
# combined buffer (the std sources concatenated AHEAD of the user file) and
# treats `user_lo` — the byte length of the seeded prefix — as the seam, so
# every fn whose name starts at/after `user_lo` is a prune root.
#
# This rung concatenates the SAME 21 stdlib modules in the SAME order Rust uses
# (STDLIB_MIND_SOURCES — alphabetical by `std.X`, with std.llvm + std.mlir
# excluded, exactly as the Rust bundle excludes them), passes the seam offset
# as `user_lo`, and diffs the pure-MIND ELF against the Rust `mind-native`
# oracle for the WHOLE self-host compiler (examples/mindc_mind/main.mind).
#
# Gate semantics: the goal is byte-identity, but the rung's PASS bar is that
# the pure-MIND emitter gets PAST the stdlib-seeding blocker — i.e. it seeds +
# prunes + lays the SAME fn set and emits a code region byte-identical to the
# oracle for at least `_SEEDED_CODE_PREFIX_FLOOR` bytes. A seeding/prune/header
# regression would collapse that identical prefix to near-zero, hard-failing
# the rung; a real advance can only raise the floor. When the prefix reaches
# full byte-identity, the rung reports the closed fixed point. No fake wins:
# the floor is a published lower bound on already-verified progress, never a
# success the emitter has not actually earned.
_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]
# Byte length of the byte-identical code-region PREFIX the pure-MIND emitter is
# known to reach today. The PHASE 1.3 WHILE-loop + Assign + Break/Continue port
# (nb_emit_while / nb_if_*_merged / the merge-phi machinery in main.mind SECTION 4c)
# pushed this from 5130 (which ended just before std.string's first loop-bearing fn)
# to 11703 — string_push_str and every now-supported while/assign/value-if-merge fn
# emit byte-identically to the Rust `mind-native` oracle. PHASE 1.3 then cleared the
# fn#53 (is_digit) F2 nested-region exit-merge floor @ 0x2ed7: an if-STATEMENT whose
# then/else block tail is a NESTED `if`/`while` statement-expression now threads that
# nested region's EXIT/dst id up as the branch's then_result/else_result (mirroring
# lower.rs 3915-4015's `then_result = lower_expr(other)` in the `other =>` arm),
# instead of reusing the stale leading-const0 placeholder. That pushed the floor to
# 23637. PHASE 1.3 #14 then cleared the `scan` (main.mind:599) F2 outer-var-through-
# NESTED-region merge floor @ 0x5d75 — three fixes that compound on the same call
# path: (1) nb_branch_writes now records a nested `if`-STATEMENT tail's branch_bindings
# (its merged_names) as writes of the ENCLOSING branch, so each enclosing `if`
# allocates the merge phi the oracle does (scan's frame 0x890 -> the byte-identical
# 0x920, +18 slots); (2) nb_arith_rax_mem ports the branchless guarded signed `/`
# sequence (emit_div_mod_guarded, src/native 218+) so `vec_len(toks) / 3` in tok_count
# emits idiv, not imul; (3) nb_emit_params copies the 7th+ stack-passed parameter from
# `[rbp + 16 + 8*(i-6)]` instead of dropping it. Together these pushed the floor to
# 61282. PHASE 1.3 #14 then ported the CALLER-side System-V stack-arg sequence
# (nb_emit_stack_pad / nb_emit_stackargs / nb_emit_stack_cleanup, mirroring
# src/native/mod.rs Call arm 935-954): a call with >6 args now emits `sub rsp,8`
# (odd-pad), pushes args[6:] right-to-left, then `add rsp,cleanup` — recovering
# ~29855 B of the prior 30603 B shortfall (the whole-image delta dropped from
# -30603 B to -748 B). The seeded ELF's `add rsp,imm32` cleanup distribution and
# `sub rsp,8` pad count are now byte-for-byte identical to the oracle.
#
# The floor stays at 61282 because the byte-identical PREFIX is still capped at
# 0xf082 — a FORWARD call-rel32 to a callee placed 748 B earlier in the mind image
# than in the oracle. That residual -748 B is NOT a stack-arg or string-table/id
# gap (the prompt's "movabs imm differs by 19 at 0x12dfc" was a misattribution).
# It is the F2 value-if MERGE-PLACEHOLDER gap: in exactly 7 downstream fns (each
# short by a multiple of 17 B = one elided `movabs rax,0 ; mov [slot],rax`
# const0), the oracle emits THEN-side placeholder ConstI64(0)s for merged names
# that the branch only carries through a NESTED `if` tail's branch_bindings, while
# the pure-MIND nb_* native backend treats them as branch-DEFINED and resolves the
# then-edge value to a let-env slot that misses to slot 0 ([rbp-8]). The first
# occurrence is at oracle 0x3b43e (3 placeholders + 4 merge stores). The rule is
# CONTEXT-DEPENDENT (a blanket "nested-export -> placeholder" narrowing regressed
# `scan` back to 0x5d75), so it needs a per-edge analysis of lower.rs's join-block
# phi predecessors before nb_merge_fill_defs can mirror it byte-exact.
#
# CLOSED 2026-06-25 (#14 Phase 1.3 FINAL-NUT): the per-edge merge-placeholder
# predicate is now byte-exact. The last 748 B (44 elided ConstI64(0) placeholders
# across 7 fns) came from let-env POLLUTION, not a wrong predicate: nb shares ONE
# monotonically-growing let-env across branch scopes, whereas eval/lower.rs lowers
# each branch in a sub-IR with a CLONED env. A branch-local `let` (or a value-if's
# internal merge phi used as a let-INIT) therefore leaked into the shared env, and
# a later SIBLING/INNER merge's `in_outer` test wrongly saw it as carried-from-outer
# and elided the placeholder. Fix = scope the let-env exactly like lower.rs:
#   (1) dst-id pre-walks restore lcell (count-only, transient);
#   (2) merged emitters restore to outer_n before nb_rebind_merges (post-if env =
#       incoming + branch_bindings only);
#   (3) a value-if-EXPRESSION let-INIT restores after the init (only X = dst escapes,
#       never the if's internal phis), in BOTH the emit let-arm and nb_count_stmt;
#   (4) nb_count_stmt's if-arm truncates branch-locals + re-binds merged names.
# The whole seeded image is now byte-identical (the `got == oracle` branch fires);
# the floor below is a regression backstop — raise it, never lower it.
_SEEDED_CODE_PREFIX_FLOOR = 1031503
# Where the ELF code image begins: 64-byte ehdr + 4 * 56-byte phdrs.
_ELF_CODE_START = 0x120


def _seeded_buffer() -> tuple[bytes, int]:
    """Concatenate the 21 bundled stdlib sources (STDLIB_MIND_SOURCES order)
    ahead of main.mind. Returns (combined_bytes, user_lo) where user_lo is the
    byte length of the seeded std prefix — the seam the native emitter slices on.
    """
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    # Newline-join keeps tokens from merging across the file boundaries, so the
    # single-buffer lex yields the same item order as Rust's per-module parse +
    # `combined.items.extend`.
    std_blob = b"\n".join(parts) + b"\n"
    user = (_HERE / "main.mind").read_bytes()
    return std_blob + user, len(std_blob)


def _mind_seeded_elf(lib, combined: bytes, user_lo: int, trace_hash: bytes) -> bytes:
    src_buf = ctypes.create_string_buffer(combined, len(combined))
    hash_buf = ctypes.create_string_buffer(trace_hash, 32)
    es = lib.selftest_native_elf_hb(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(combined),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
        user_lo,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def seeded_main_rung(lib, tmp: pathlib.Path) -> int:
    """Run the stdlib-seeded whole-module rung on main.mind. Returns 0 on PASS
    (past the seeding blocker, prefix >= floor, or byte-identical), 1 on a
    seeding/prune regression below the floor."""
    if not (_HERE / "main.mind").exists():
        print("  SKIP  seeded rung: main.mind not present")
        return 0

    lib.selftest_native_elf_hb.restype = ctypes.c_int64
    lib.selftest_native_elf_hb.argtypes = [ctypes.c_int64] * 4

    # Oracle: mind-native on main.mind (it seeds the stdlib internally).
    elf_path = tmp / "oracle_main.elf"
    subprocess.run(
        [str(MIND_NATIVE), str(_HERE / "main.mind"), str(elf_path)],
        capture_output=True,
        check=True,
    )
    oracle = elf_path.read_bytes()
    note = oracle[-52:]
    if note[12:16] != b"MIND":
        print(f"  FAIL  seeded rung: oracle note missing MIND name: {note[12:20]!r}")
        return 1
    trace_hash = note[20:52]

    combined, user_lo = _seeded_buffer()
    got = _mind_seeded_elf(lib, combined, user_lo, trace_hash)

    if got == oracle:
        print(
            f"  PASS  seeded main.mind native ELF BYTE-IDENTICAL "
            f"({len(oracle)} bytes) — stdlib-seeding fixed point CLOSED"
        )
        return 0

    # Not yet byte-identical: measure the byte-identical code-region prefix to
    # confirm we are PAST the seeding/prune. The ELF header up to the code region
    # is identical EXCEPT the p_filesz/p_memsz/segment-size phdr fields, which are
    # pure functions of the total image size and so legitimately differ while the
    # code stream itself diverges downstream — that is NOT a seeding regression.
    # The real, non-fakeable seeding-OK proof is the multi-function byte-identical
    # CODE prefix: it can only hold if the same seeded fn set was pruned, ordered,
    # and emitted identically through the first divergent fn.
    n = min(len(got), len(oracle))
    # Structural header fields that must stay identical regardless of code size:
    # ELF magic + ident (0..16), type/machine/version/entry/phoff (16..0x40), and
    # the phdr type/flags/offset/vaddr/align fields (everything in 0x40..0x120
    # except the four size-bearing 0x60/0x68/0x80/0x88-region words, which the
    # diff above shows are the only legitimate size-dependent deltas). We assert
    # the ELF magic + e_machine + e_entry are intact as a cheap seeding sanity
    # check, then lean on the code-region floor for the real proof.
    magic_ok = got[:0x14] == oracle[:0x14] and got[0x18:0x40] == oracle[0x18:0x40]
    prefix = 0
    for i in range(_ELF_CODE_START, n):
        if got[i] != oracle[i]:
            break
        prefix += 1
    first = _ELF_CODE_START + prefix
    blk = (
        f"first divergence at 0x{first:x}: "
        f"mind={got[first]:#04x} oracle={oracle[first]:#04x}"
        if first < n
        else f"common prefix identical to len {n}"
    )
    if not magic_ok:
        print(
            f"  FAIL  seeded rung: ELF magic/machine/entry diverged — stdlib "
            f"seeding REGRESSED (mind {len(got)} B vs oracle {len(oracle)} B); {blk}"
        )
        return 1
    if prefix < _SEEDED_CODE_PREFIX_FLOOR:
        print(
            f"  FAIL  seeded rung: identical code-region prefix {prefix} B "
            f"< floor {_SEEDED_CODE_PREFIX_FLOOR} B — REGRESSION; {blk}"
        )
        return 1

    print(
        f"  PASS  seeded main.mind PAST stdlib-seeding blocker: ELF magic/machine/entry "
        f"intact, {prefix} B of code byte-identical (through the while/assign/merge "
        f"loop-bearing fns; floor {_SEEDED_CODE_PREFIX_FLOOR}); next blocker @ {blk}"
    )
    print(
        "        (known next gaps: a downstream frame-size/slot-count under-count "
        "(oracle `sub rsp,0x920` vs mind `0x890`) + the F2 outer-var-MUTATION rebind "
        "+ the div/mod/shift binops — see the deferred markers at nb_lower_fn / "
        "nb_branch_writes / nb_arith_rax_mem in main.mind; not fully byte-identical YET)"
    )
    return 0


def mind_elf(lib, src: bytes, trace_hash: bytes) -> bytes:
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(trace_hash, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def first_diverge(a: bytes, b: bytes) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            lo, hi = max(0, i - 8), min(n, i + 16)
            return (
                f"FIRST DIVERGE at offset {i} (0x{i:x}): "
                f"mind={a[i]:#04x} rust={b[i]:#04x}\n"
                f"  mind: {a[lo:hi].hex()}\n  rust: {b[lo:hi].hex()}"
            )
    return f"length mismatch: mind={len(a)} rust={len(b)} (common prefix identical)"


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0
    if not MIND_NATIVE.exists():
        print(
            f"SKIP: mind-native oracle binary not found at {MIND_NATIVE} "
            "(build with: CARGO_TARGET_DIR=/tmp/mind-native-target "
            "cargo build --features native-backend --bin mind-native)"
        )
        return 0

    import tempfile

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, fixture, expected_exit in FIXTURES:
            rust = oracle_elf(fixture, tmp)
            # The ir_trace_hash anchoring the note: last 52 bytes = 12-byte nhdr +
            # "MIND\0\0\0\0" (8) + 32-byte hash.
            note = rust[-52:]
            if note[12:16] != b"MIND":
                print(f"ERROR: oracle ELF note missing MIND name: {note[12:20]!r}")
                return 1
            trace_hash = note[20:52]

            got = mind_elf(lib, fixture.encode(), trace_hash)

            ok = got == rust
            print(
                f"  {'PASS' if ok else 'FAIL'}  {name} native ELF "
                f"({len(rust)} oracle bytes / {len(got)} mind bytes)"
            )
            if not ok:
                print(first_diverge(got, rust))
                return 1

            # Run the pure-MIND-emitted ELF; it must exit with the fixture's value.
            code = run_elf(got, tmp)
            run_ok = code == expected_exit
            print(
                f"  {'PASS' if run_ok else 'FAIL'}  {name} pure-MIND ELF runs + exits "
                f"{code} (expected {expected_exit})"
            )
            if not run_ok:
                return 1

        # SEEDED whole-module rung: compile main.mind WITH the bundled stdlib
        # seeded (same set+order as Rust mind-native) and diff vs the oracle.
        print("\n[seeded whole-module rung: main.mind + bundled stdlib]")
        if seeded_main_rung(lib, tmp) != 0:
            return 1

    print(
        "\nALL PASS  (byte-identical to Rust mind-native + runs for every fixture; "
        "seeded whole-module rung past the stdlib-seeding blocker)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
