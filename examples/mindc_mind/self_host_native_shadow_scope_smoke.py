#!/usr/bin/env python3
"""Lexical scoping of nested bindings in the native seed (block / loop-body shadows).

A `let` inside an if/else block or a while body that re-declares a name from an
enclosing scope (an outer `let`, a parameter) must get its own storage for that
block's lifetime: reads and writes inside the block resolve to the innermost
binding, and the outer binding is untouched once the block ends. The frozen seed
before this fix merged the two (`let mut x = 5; if c { let mut x = 9; if c { x = 40; } }
return x;` exited 40 natively, 5 under MLIR), which is why the Rust bridge refuses
every such program (native.block_shadow / native.loop_body_shadow).

What runs:
  1. stage1 is built from THIS tree's main.mind by the frozen pure-MIND seed
     (testdata/selfhost_loop/stage1.elf; no Rust, no MLIR in that step).
  2. Hand fixtures (the shapes the bridge guard was written from, a break/continue
     taken while a loop-body shadow of a carried name is in scope, typed i64/i32/f64
     shadows) and a seeded generated corpus mixing shadowing with
     while/if/else/break/continue/return at block depths 1-4, plus a helper fn whose
     parameters are shadowed.
  3. Each program is fed to stage1 (stdin protocol [user_lo][src_len][std blob +
     user source]); the emitted ELF is run and its exit code compared with a Python
     model that interprets the generated program with lexical scopes (an oracle
     independent of the emitter's slot allocation); a wrong exit is a FAIL. A
     refusal (rc != 0, no ELF) is fail-closed only when the program's alpha-renamed
     twin (every nested shadow renamed away, same meaning) is refused too, i.e. the
     native control-flow subset refused it. A refusal the twin does not share is
     split with the MINDC_SO oracle library: when the code generator alone builds it
     and the ELF exits with the model's value while only the mic@3 trace-hash note
     builder (the Rust emit_mic3 twin) refuses, it is note_only_refusal (counted,
     allowed: fail-closed, and not a scoping defect of the code generator); anything
     else -- or no MINDC_SO -- is shadow_only_refusal and a FAIL.
  4. With --mlir (or SHADOW_MLIR=1) every program is also built with `mindc build`
     (MLIR backend) and its exit compared with the model's. The MLIR backend carries
     a loop-body shadow out of the loop when a break/continue is taken while it is
     in scope (the fixtures' comments give its exits); a disagreement on a program
     the model saw take such an exit is MLIR-KNOWN-DEFECT (counted, allowed); any
     other disagreement is MLIR-DISAGREE and fails above --max-mlir-disagree
     (default 0). Every hand fixture is built with MLIR as well: it must exit with
     the model's value, or, for the fixtures listed in MLIR_DEFECT_FIXTURES, with
     the recorded defect exit (so a change in either backend is noticed).
     The mlab_* fixtures are the re-declaration shapes of the ten MIND-Lab
     programs (a block-local name re-declared after the block, the same name in
     both branches, in sibling loop bodies); they are not shadows to the Rust
     guard and must keep separate storage under every compiler.
  5. stage1 must carry the scoped-nested-bindings capability mark
     (`nb_cap_scoped_nested_bindings`: movabs rax, "nbscope1"), the bytes the Rust
     bridge looks for before admitting the proven shadow shapes.
  6. With --bridge (or SHADOW_BRIDGE=1) the real `mindc build --backend native`
     path is exercised. With MINDC_NATIVE_ELF=stage1 every natively built
     generated program must build and exit with the model's value, and every
     fixture must either do so or be refused (exit 3, no artifact) by a fence
     other than the shadow guard (native profile, closure); the range-for
     fixtures (BRIDGE_UNPROVEN) must be refused by the shadow guard. The
     embedded default seed (no MINDC_NATIVE_ELF; MINDC_BIN built from this tree)
     is held to the contract its own bytes select: a seed carrying the mark is a
     scoping compiler and meets exactly the stage1 contract above; an unmarked
     seed may run no fixture to a wrong value, and every fixture the scoping
     compiler built must be refused by the shadow guard (native.block_shadow /
     native.loop_body_shadow).

Flags: --count N (generated programs, default 200), --seed S (default 1234),
       --min-compared M (default 60: natively-built-and-compared floor),
       --mlir, --jobs J.
Env:   SHADOW_MAIN_MIND (main.mind to build stage1 from; used to show RED on a base
       copy), MINDC_BIN (MLIR leg).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import hashlib
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _stdlib_manifest import seed_blob  # noqa: E402
from _shadow_scope_lang import (  # noqa: E402
    Gen, alpha_rename, has_nested_shadow, model_exit, render)

FROZEN_SEED = HERE / "testdata" / "selfhost_loop" / "stage1.elf"
REPO = HERE.parent.parent
# movabs rax, imm64 with the immediate's little-endian bytes "nbscope1"; must equal
# SCOPED_BINDINGS_MARK in src/build/native_bridge/seed.rs.
MARK = b"\x48\xb8nbscope1"
DRIVER = HERE / "selfhost_driver.mind"
# ---------------------------------------------------------------- stage1 / runs
def build_stage1(cache: pathlib.Path) -> pathlib.Path:
    blob = seed_blob()
    main = pathlib.Path(os.environ.get("SHADOW_MAIN_MIND", str(HERE / "main.mind")))
    combined = blob + main.read_bytes() + b"\n" + DRIVER.read_bytes()
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    key = hashlib.sha256(FROZEN_SEED.read_bytes() + img).hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"stage1_{key}.elf"
    if out.exists():
        return out
    r = subprocess.run([str(FROZEN_SEED)], input=img, capture_output=True, timeout=1800)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise RuntimeError(f"frozen seed failed to build stage1: rc={r.returncode}")
    tmp = cache / f".{key}.{os.getpid()}"
    tmp.write_bytes(r.stdout)
    os.chmod(tmp, 0o755)
    os.replace(tmp, out)
    return out


def native_exit(stage1: pathlib.Path, src: str, wd: pathlib.Path, name: str):
    """('ok', exit) | ('refused', rc) | ('crash', detail)."""
    blob = seed_blob()
    combined = blob + src.encode()
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    r = subprocess.run([str(stage1)], input=img, capture_output=True, timeout=600)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        if r.returncode < 0:
            return "crash", f"stage1 signal {r.returncode}"
        return "refused", r.returncode
    p = wd / f"{name}.nat"
    p.write_bytes(r.stdout)
    os.chmod(p, 0o755)
    try:
        e = subprocess.run([str(p)], capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "crash", "elf did not exit within 60 s (hang)"
    if e.returncode < 0:
        return "crash", f"elf signal {e.returncode}"
    return "ok", e.returncode


def so_lib():
    """The MINDC_SO oracle library (main.mind built by the Rust compiler), or None."""
    so = os.environ.get("MINDC_SO")
    if not so:
        return None
    lib = ctypes.CDLL(so)
    lib.selftest_native_elf_hb.restype = ctypes.c_int64
    lib.selftest_native_elf_hb.argtypes = [ctypes.c_int64] * 4
    lib.nb_build_mic3.restype = ctypes.c_int64
    lib.nb_build_mic3.argtypes = [ctypes.c_int64] * 3
    return lib


def split_refusal(lib, src: str, wd: pathlib.Path, name: str):
    """Which half of stage1 refused `src`: the code generator or the trace-hash note.

    stage1 emits an ELF only when both the native code generator
    (selftest_native_elf_hb) and the pure-MIND mic@3 note builder (nb_build_mic3,
    the twin of the Rust emit_mic3 the PT_NOTE hashes) accept the program. Run each
    half alone through the oracle library: ('note', exit) when the code generator
    builds it (the ELF, run with a placeholder note, exits `exit`) and only the
    note builder refuses; ('codegen', None) when the code generator refuses."""
    blob = seed_blob()
    comb = blob + src.encode()
    sb = ctypes.create_string_buffer(comb, len(comb))
    hb = ctypes.create_string_buffer(b"\x11" * 32, 32)
    addr = ctypes.cast(sb, ctypes.c_void_p).value
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]  # noqa: E731
    es = lib.selftest_native_elf_hb(addr, len(comb), ctypes.cast(hb, ctypes.c_void_p).value,
                                    len(blob))
    elf = ctypes.string_at(rd(rd(es, 0), 0), rd(rd(es, 0), 8))
    note = rd(rd(lib.nb_build_mic3(addr, len(comb), len(blob)), 0), 8)
    if not elf.startswith(b"\x7fELF") or note != 0:
        return "codegen", None
    p = wd / f"{name}.cg"
    p.write_bytes(elf)
    os.chmod(p, 0o755)
    r = subprocess.run([str(p)], capture_output=True, timeout=60)
    return "note", r.returncode


def mindc_bin() -> str:
    """Compiler for --bridge / --mlir. A missing file is a hard exit.

    The no-arg leg never calls this: it builds stage1 from the frozen seed.
    A set MINDC_BIN or MINDC that is not a file, and an unset handle with no
    release binary in this tree, both stop here. Returning a missing path
    would build stage1 first and only fail once the bridge leg ran.
    """
    env = os.environ.get("MINDC_BIN") or os.environ.get("MINDC")
    if env:
        if not pathlib.Path(env).is_file():
            raise SystemExit(
                f"FAIL: mindc not found at {env} (from $MINDC_BIN/$MINDC)")
        return env
    from _selfhost_so import resolve_mindc
    return resolve_mindc()


def bridge_exit(src: str, wd: pathlib.Path, name: str, compiler):
    """`mindc build --backend native`; compiler=None -> the embedded seed.
    ('ok', exit) | ('refused', rc, stderr_tail)."""
    d = wd / f"br_{name}"
    d.mkdir(exist_ok=True)
    f = d / "main.mind"
    f.write_text(src)
    out = d / "out.elf"
    env = dict(os.environ, MINDC_STD_DIR=str(REPO / "std"))
    env.pop("MINDC_NATIVE_ELF", None)
    if compiler is not None:
        env["MINDC_NATIVE_ELF"] = str(compiler)
    b = subprocess.run([mindc_bin(), "build", str(f), "--release", "--backend", "native",
                        "--out", str(out)], capture_output=True, timeout=900, cwd=str(d),
                       env=env)
    if b.returncode != 0 or not out.exists():
        return "refused", b.returncode, b.stderr.decode(errors="replace")[-600:], out.exists()
    os.chmod(out, 0o755)
    r = subprocess.run([str(out)], capture_output=True, timeout=60)
    return "ok", r.returncode


def mlir_exit(src: str, wd: pathlib.Path, name: str):
    mindc = mindc_bin()
    f = wd / f"{name}.mind"
    f.write_text(src)
    out = wd / f"{name}.mlir.bin"
    b = subprocess.run([mindc, "build", str(f), "--release", "--out", str(out)],
                       capture_output=True, timeout=900, cwd=str(wd))
    if b.returncode != 0 or not out.exists():
        return "build-failed", (b.stdout + b.stderr)[-300:]
    r = subprocess.run([str(out)], capture_output=True, timeout=60)
    return "ok", r.returncode


# ---------------------------------------------------------------- fixtures
FIXTURES = {
    # The shapes the bridge's shadow guard was written from (frozen seed got 40/11/40/150).
    "if2_write": ("fn main() -> i64 { let c: i64 = 1; let mut x: i64 = 5; if c == 1 { "
                  "let mut x: i64 = 9; if c == 1 { x = 40; } } return x; }", 5),
    "if_loop_write": ("fn main() -> i64 { let c: i64 = 1; let mut x: i64 = 5; if c == 1 { "
                      "let mut x: i64 = 9; let mut i: i64 = 0; while i < 2 { x = x + 1; "
                      "i = i + 1; } } return x; }", 5),
    "param_if2": ("fn f(x: i64) -> i64 { if x > 0 { let mut x: i64 = 9; if x > 0 { x = 40; } } "
                  "return x; } fn main() -> i64 { return f(3); }", 3),
    "loop_body_mut": ("fn main() -> i64 { let mut s: i64 = 7; let mut i: i64 = 0; let mut t: i64 = 0; "
                      "while i < 4 { let mut s: i64 = i; s = s * 3; t = t + s; i = i + 1; } "
                      "return s * 100 + t; }", (7 * 100 + 18) & 255),
    "loop_body_guard_doc": ("fn main() -> i64 { let mut x: i64 = 5; let mut k: i64 = 0; while k < 2 { "
                            "k = k + 1; let mut x: i64 = 7; x = 3; } return x; }", 5),
    "inner_reads_inner": ("fn main() -> i64 { let mut x: i64 = 5; let mut r: i64 = 0; if x > 1 { "
                          "let mut x: i64 = 9; x = x + 1; r = x; } return x * 100 + r; }",
                          (510) & 255),
    "nested_loop_counter_shadow": ("fn main() -> i64 { let mut i: i64 = 0; let mut t: i64 = 0; "
                                   "while i < 3 { let mut j: i64 = 0; while j < 2 { "
                                   "let i: i64 = j + 10; t = t + i; j = j + 1; } i = i + 1; } "
                                   "return t + i; }", 66),
    "break_reads_shadow": ("fn main() -> i64 { let mut x: i64 = 1; let mut i: i64 = 0; while i < 10 { "
                           "let x: i64 = i * 2; if x > 6 { break; } i = i + 1; } return x * 10 + i; }",
                           14),
    "else_shadow_write": ("fn main() -> i64 { let mut x: i64 = 2; if x > 5 { x = 1; } else { "
                          "let mut x: i64 = 30; if x > 3 { x = 99; } } return x; }", 2),
    "depth4_write": ("fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 2 { "
                     "k = k + 1; if k > 0 { let mut x: i64 = 50; let mut j: i64 = 0; while j < 2 { "
                     "j = j + 1; if j == 1 { x = x + 7; } } x = x + 1; } x = x + 2; } return x; }", 5),
    "loop_keyword": ("fn main() -> i64 { let mut x: i64 = 5; let mut k: i64 = 0; loop { k = k + 1; "
                     "let mut x: i64 = 7; x = 3; if k >= 2 { break; } } return x; }", 5),
    "continue_shadow": ("fn main() -> i64 { let mut x: i64 = 4; let mut k: i64 = 0; let mut t: i64 = 0; "
                        "while k < 5 { k = k + 1; let mut x: i64 = k * 3; if x > 6 { continue; } "
                        "x = x + 100; t = t + x; } return t + x; }", (103 + 106 + 4) & 255),
    # An outer write BEFORE the loop-body shadow, then a nested `if` writing the
    # shadow: the if's write must stay on the per-iteration binding (the shape that
    # kills a mutant dropping the loop-body shadow carry restore).
    "assign_then_shadow_if_write": ("fn main() -> i64 { let mut x: i64 = 8; let mut k: i64 = 0; "
                                    "while k < 3 { k = k + 1; x = x + 1; let mut x: i64 = 50; "
                                    "if k > 0 { x = 21; } } return x; }", 11),
    "assign_then_shadow_if_rw": ("fn main() -> i64 { let mut x: i64 = 8; let mut y: i64 = 2; "
                                 "let mut k: i64 = 0; while k < 3 { k = k + 1; let mut y: i64 = y + 39; "
                                 "x = y + 19; let mut x: i64 = x + 10; if 27 > 25 { y = x - y; "
                                 "x = y; x = 21; } } return x * 2 + y; }", ((41 + 19) * 2 + 2) & 255),
    "return_inner_shadow": ("fn f(x: i64) -> i64 { let mut y: i64 = 3; if x > 0 { let y: i64 = x * 7; "
                            "if y > 10 { return y + 1; } } return y; } "
                            "fn main() -> i64 { return f(2) * 10 + f(1); }", (15 * 10 + 3) & 255),
}


# A break/continue taken while a loop-body `let` shadows a CARRIED name (one the
# body assigns): the exit must carry the outer binding's value, not the shadow's.
# Before this fix the native seed refused every such function (fn-wide poison);
# the MLIR backend builds them but carries the shadow out of the loop (MLIR exit in
# the comment, a known MLIR defect -- the lexical value is `want`).
FIXTURES.update({
    # MLIR 9
    "continue_edge_assign_before_shadow": (
        "fn main() -> i64 { let mut x: i64 = 5; let y: i64 = 9; let mut k: i64 = 0; "
        "while k < 4 { k = k + 1; x = x; let x: i64 = y; if 28 != 0 { continue; } } "
        "return x; }", 5),
    # MLIR 100
    "break_after_shadow": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 5 { k = k + 1; "
        "x = x + 10; let x: i64 = 100; if k == 3 { break; } } return x; }", 31),
    # MLIR 51
    "continue_mut_shadow": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 4 { k = k + 1; "
        "x = x + 1; let mut x: i64 = 50; x = x + 1; if k > 1 { continue; } x = x + 5; } "
        "return x; }", 5),
    # MLIR 40
    "two_carried_one_shadowed": (
        "fn main() -> i64 { let mut x: i64 = 0; let mut y: i64 = 0; let mut k: i64 = 0; "
        "while k < 5 { k = k + 1; x = x + 1; y = y + 2; let y: i64 = 0; "
        "if k == 2 { continue; } if k == 4 { break; } } return x * 10 + y; }", 48),
    # MLIR 117 (the shadow's 31 carried out: 373 & 255)
    "shadow_rw_then_break": (
        "fn main() -> i64 { let mut x: i64 = 3; let mut t: i64 = 0; let mut k: i64 = 0; "
        "while k < 9 { k = k + 1; x = x + 1; let mut x: i64 = k * 10; x = x + 1; t = t + x; "
        "if t > 40 { break; } } return x * 10 + t; }", (60 + 63) & 255),
    # MLIR 1
    "else_break_after_shadow": (
        "fn main() -> i64 { let mut x: i64 = 0; let mut k: i64 = 0; while k < 9 { k = k + 1; "
        "x = x + 4; let mut x: i64 = 1; if k < 3 { x = x; } else { break; } } return x; }", 12),
    # i32 carried var (wraps at the second add), i64 shadow; MLIR 1
    "i32_carried_i64_shadow_break": (
        "fn main() -> i64 { let mut x: i32 = 2147483640; let mut k: i64 = 0; while k < 3 { "
        "k = k + 1; x = x + 5; let x: i64 = 1; if k == 2 { break; } } return x as i64; }",
        (2147483650 - (1 << 32)) & 255),
    "inner_loop_break_with_shadow": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 3 { k = k + 1; "
        "x = x + 1; let mut j: i64 = 0; while j < 5 { j = j + 1; let x: i64 = j * 100; "
        "if j == 2 { break; } } } return x; }", 4),
    "carried_var_is_the_shadow": (
        "fn main() -> i64 { let c: i64 = 1; let mut r: i64 = 0; let mut k: i64 = 0; "
        "let mut x: i64 = 1; if c == 1 { let mut x: i64 = 10; while k < 5 { k = k + 1; "
        "x = x + 1; if k == 3 { break; } } r = x; } return x * 100 + r; }", (100 + 13) & 255),
    # typed shadows (i64 / i32 / f64 across the pair)
    "i32_shadow_of_i64": (
        "fn main() -> i64 { let c: i64 = 1; let mut x: i64 = 1; if c == 1 { let mut x: i32 = 7; "
        "x = x + 1; } x = x + 4294967295; return x >> 32; }", 1),
    "i64_shadow_of_i32": (
        "fn main() -> i64 { let c: i64 = 1; let mut r: i64 = 0; let mut x: i32 = 1; "
        "if c == 1 { let mut x: i64 = 4294967296; x = x + 1; r = x >> 32; } "
        "return r * 10 + (x as i64); }", 11),
    "i32_loop_shadow_wraps": (
        "fn main() -> i64 { let mut x: i64 = 4294967300; let mut t: i64 = 0; let mut k: i64 = 0; "
        "while k < 2 { k = k + 1; let mut x: i32 = 2147483647; x = x + 1; t = t + 1; } "
        "return (x >> 32) + t; }", 3),
    "f64_shadow_of_i64": (
        "fn main() -> i64 { let c: i64 = 1; let mut x: i64 = 3; if c == 1 { let x: f64 = 2.5; "
        "let y: f64 = x * 2.0; } x = x + 1; return x; }", 4),
})

# Range-for loops whose variable is bound outside the loop or re-declared in its
# body: the desugar must scope VAR to the loop (the counter still advances, the
# outer binding is untouched after it). Before this fix stage1 hung on the first
# two, returned 33 and 76 on the next two and refused the last; the MLIR build
# exits with `want` except on the last (103: its continue edge carries the branch
# shadow into the loop variable, the known MLIR defect above).
FIXTURES.update({
    "for_body_let_shadows_var": (
        "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { let i: i64 = 10; s = s + i; } "
        "return s; }", 30),
    "nested_for_same_var": (
        "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for i in 0..2 { s = s + i; } "
        "s = s + 100; } return s; }", 303 & 255),
    "for_var_shadows_param": (
        "fn f(i: i64) -> i64 { let mut s: i64 = 0; for i in 0..3 { s = s + i; } "
        "return i * 10 + s; } fn main() -> i64 { return f(4); }", 43),
    "for_var_shadows_outer_let": (
        "fn main() -> i64 { let i: i64 = 7; let mut s: i64 = 0; let c: i64 = 1; "
        "if c == 1 { for i in 0..3 { s = s + i; } s = s + i; } return i * 10 + s; }", 80),
    "sibling_fors_reuse_var": (
        "fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { s = s + i; } "
        "for i in 0..2 { s = s + i * 10; } return s; }", 13),
    "for_continue_branch_let_var": (
        "fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { if i > 2 { let i: i64 = 100; "
        "s = s + i; continue; } s = s + i; } return s; }", 203),
})

# The re-declaration shapes the ten MIND-Lab programs (current-ten export) use, with
# their memory intrinsics replaced by arithmetic: a block-local `let` followed by an
# outer `let` of the same name after the block (ec_add `tail`, nn_add `i`), the same
# name in the then- and else-branch (ec_integer `b`), and the same name in sibling
# while bodies (nn_decimal / nn_add `b`, `value`). None of them is a shadow (the Rust
# guard sees none); each must keep its own storage. The four with a `return` or
# `break` inside the loop and the if/else-assign merge are refused by the native
# control-flow subset with or without the repeated name (TWINS).
FIXTURES.update({
    "mlab_ec_add_tail": (
        "fn f(key: i64, start: i64) -> i64 { if key == 0 { let tail: i64 = start + 1; "
        "if tail == 0 { return 7; } else { return tail * 2; } } let mut old: i64 = start; "
        "let mut tail: i64 = 0; while old != 0 { if key != 0 { if old == 3 { return 100 + tail; } } "
        "tail = tail + old; old = old - 1; } if tail == 0 { return 9; } else { return tail; } } "
        "fn main() -> i64 { return f(0, 4) + f(1, 2) * 3 + f(1, 5); }", 128),
    "mlab_ec_add_tail_noret": (
        "fn f(key: i64, start: i64) -> i64 { if key == 0 { let tail: i64 = start + 1; "
        "return tail * 2; } let mut old: i64 = start; let mut tail: i64 = 0; while old != 0 { "
        "tail = tail + old; old = old - 1; } return tail; } "
        "fn main() -> i64 { return f(0, 4) + f(1, 5); }", 25),
    "mlab_ec_integer_b": (
        "fn f(first: i64, c: i64) -> i64 { let mut start: i64 = 10; if first == 48 { "
        "start = start + 1; let b: i64 = c + 1; if b >= 48 { return 1; } } else { "
        "let mut b: i64 = c; while b < 57 { start = start + 1; b = b + 2; } } return start; } "
        "fn main() -> i64 { return f(48, 3) * 10 + f(50, 50) + f(48, 60); }", 125),
    "mlab_ec_integer_b_merge": (
        "fn f(first: i64, c: i64) -> i64 { let mut start: i64 = 10; if first == 48 { "
        "let b: i64 = c + 1; start = start + b; } else { let mut b: i64 = c; while b < 57 { "
        "start = start + 1; b = b + 2; } } return start; } "
        "fn main() -> i64 { return f(48, 3) * 10 + f(50, 50); }", 154),
    "mlab_nn_decimal_b": (
        "fn f(n: i64, base: i64) -> i64 { let mut i: i64 = 0; let mut digits: i64 = 0; "
        "while i < n { let b: i64 = base + i; if b > 57 { break; } digits = digits + 1; i = i + 1; } "
        "let mut frac: i64 = 0; if i < n { i = i + 1; while i < n { let b: i64 = base + i * 2; "
        "if b > 70 { break; } frac = frac + 1; i = i + 1; } } let mut e: i64 = 0; if i < n { "
        "while i < n { let b: i64 = i; if b > 100 { return 1; } e = e + b; i = i + 1; } } "
        "return digits * 100 + frac * 10 + e; } fn main() -> i64 { return f(12, 50) % 256; }", 63),
    "mlab_sibling_loops_b": (
        "fn f(n: i64) -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; while i < n { "
        "let b: i64 = i * 2; if b > 9 { break; } s = s + b; i = i + 1; } while i < n { "
        "let b: i64 = i * 3; s = s + b; i = i + 1; } return s; } "
        "fn main() -> i64 { return f(8); }", 74),
    "mlab_nn_add_i_value": (
        "fn f(a: i64, b: i64) -> i64 { if a == b { let n: i64 = 3; let mut carry: i64 = 0; "
        "let mut i: i64 = 0; while i < n { let value: i64 = a + carry + i; carry = value - a; "
        "i = i + 1; } if carry != 0 { i = i + 1; } return i * 10 + carry; } "
        "let mut borrow: i64 = 0; let mut i: i64 = 0; while i < 4 { "
        "let mut value: i64 = a - b - borrow; if value < 0 { value = value + 10; borrow = 1; } "
        "i = i + value; } return i + borrow; } "
        "fn main() -> i64 { return f(2, 2) + f(1, 3) * 7; }", 106),
})
# A loop exit resolves each carried name to the binding made before the first
# body-local DECLARATION of that name; whether a let-env entry is a declaration is
# read from the source text before the name (`let`, `let mut`, `for`). A `//`
# comment ending in one of those words right before a plain assignment must not
# make that assignment look like a declaration (no shadow anywhere: the exit
# would otherwise drop the assignment), and a comment between `let` and the name
# must not hide a real declaration.
FIXTURES.update({
    "comment_let_before_assign_continue": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 4 { k = k + 1;\n"
        "    // no new let\n    x = x + 10; if k > 1 { continue; } x = x + 1; } return x; }", 42),
    "comment_for_before_assign_break": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 5 { k = k + 1;\n"
        "    // wait for\n    x = x + 10; if k == 3 { break; } x = x + 1; } return x; }", 33),
    "comment_let_mut_before_assign_break": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 5 { k = k + 1;\n"
        "    // keep let mut\n    x = x + 10; if k == 3 { break; } x = x + 1; } return x; }", 33),
    "comment_slashes_in_string_line": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 5 { k = k + 1;\n"
        "    let c: i64 = '\"'; // let\n    x = x + 10; if k == 3 { break; } x = x + c - 33; } "
        "return x; }", 33),
    "comment_between_let_and_name": (
        "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 5 { k = k + 1; "
        "x = x + 10; let\n    // the per-iteration copy\n    x: i64 = 100; if k == 3 { break; } } "
        "return x; }", 31),
    # The byte-neutral range-for desugar's step `i = i + 1` (spliced before each
    # `continue`) is an assignment whose target span is the header's `i`, which
    # follows `for`; the native control-flow subset refuses this loop shape today
    # (so does its explicit `while` form). Once it is admitted the step must not be
    # read as a declaration at the continue (the counter would never advance): it
    # must build to 3, or stay refused -- never hang (see nb_lets_lookup_carried).
    "for_continue_step": (
        "fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { if i > 2 { continue; } "
        "s = s + i; } return s; }", 3),
})
MLAB_TWINS = {
    "mlab_ec_add_tail": FIXTURES["mlab_ec_add_tail"][0].replace(
        "let tail: i64 = start + 1; if tail == 0 { return 7; } else { return tail * 2; }",
        "let tl: i64 = start + 1; if tl == 0 { return 7; } else { return tl * 2; }"),
    "mlab_ec_integer_b": FIXTURES["mlab_ec_integer_b"][0].replace(
        "let b: i64 = c + 1; if b >= 48", "let bb: i64 = c + 1; if bb >= 48"),
    "mlab_ec_integer_b_merge": FIXTURES["mlab_ec_integer_b_merge"][0].replace(
        "let b: i64 = c + 1; start = start + b;", "let bb: i64 = c + 1; start = start + bb;"),
    "mlab_nn_decimal_b": FIXTURES["mlab_nn_decimal_b"][0].replace(
        "let b: i64 = base + i * 2; if b > 70", "let b2: i64 = base + i * 2; if b2 > 70").replace(
        "let b: i64 = i; if b > 100 { return 1; } e = e + b;",
        "let b3: i64 = i; if b3 > 100 { return 1; } e = e + b3;"),
    "mlab_sibling_loops_b": FIXTURES["mlab_sibling_loops_b"][0].replace(
        "let b: i64 = i * 3; s = s + b;", "let b2: i64 = i * 3; s = s + b2;"),
}
assert all(MLAB_TWINS[k] != FIXTURES[k][0] for k in MLAB_TWINS)

# Fixtures the native control-flow subset refuses whatever the names (an `if` with
# a nested loop inside a loop, an if/else assign merge, `loop`, `else if`): each
# twin renames the shadow away; the fixture may be refused only while its twin is.
TWINS = {
    "else_shadow_write": "fn main() -> i64 { let mut x: i64 = 2; if x > 5 { x = 1; } else { "
                         "let mut w: i64 = 30; if w > 3 { w = 99; } } return x; }",
    "depth4_write": "fn main() -> i64 { let mut x: i64 = 1; let mut k: i64 = 0; while k < 2 { "
                    "k = k + 1; if k > 0 { let mut w: i64 = 50; let mut j: i64 = 0; while j < 2 { "
                    "j = j + 1; if j == 1 { w = w + 7; } } w = w + 1; } x = x + 2; } return x; }",
    "loop_keyword": "fn main() -> i64 { let mut x: i64 = 5; let mut k: i64 = 0; loop { k = k + 1; "
                    "let mut w: i64 = 7; w = 3; if k >= 2 { break; } } return x; }",
    "for_continue_branch_let_var": "fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { "
                                   "if i > 2 { let w: i64 = 100; s = s + w; continue; } "
                                   "s = s + i; } return s; }",
}
TWINS.update(MLAB_TWINS)
# no shadow to rename: refused is acceptable only while the program itself is
TWINS["for_continue_step"] = FIXTURES["for_continue_step"][0]

# The MLIR build's exit for the fixtures that take a loop exit while a body-local
# shadow is in scope (the comments above): the shadow's value leaks out of the loop.
MLIR_DEFECT_FIXTURES = {
    "continue_edge_assign_before_shadow": 9, "break_after_shadow": 100,
    "continue_mut_shadow": 51, "two_carried_one_shadowed": 40,
    "shadow_rw_then_break": 117, "else_break_after_shadow": 1,
    "i32_carried_i64_shadow_break": 1, "for_continue_branch_let_var": 103,
    "comment_between_let_and_name": 100,
}

# The historical embedded seed (testdata/selfhost_loop/stage1.elf, abb0622b) dropped
# the first of two assignments to a carried name at a `continue`/`break` taken
# between them -- no shadow involved (`while k < 4 { k = k + 1; x = x + 10; if k > 1 {
# continue; } x = x + 1; }` exits 12, lexically 42). main.mind lowers it correctly
# and the two-hop reseed that embeds the nblane01/nbscope1 marks ships that, so a
# marked embedded seed gets no allowance; the bridge leg accepts this recorded exit
# only from an UNMARKED embedded seed.
EMBEDDED_SEED_DEFECT = {
    "comment_let_before_assign_continue": 12, "comment_for_before_assign_break": 23,
    "comment_let_mut_before_assign_break": 23, "comment_slashes_in_string_line": 23,
}


# Fixtures whose shadow the Rust bridge guard keeps unproven (range-for variables
# and bodies): refused with native.loop_body_shadow / native.block_shadow
# whatever compiler runs. And the fixture the guard sees no shadow in at all.
BRIDGE_UNPROVEN = {"for_body_let_shadows_var", "nested_for_same_var", "for_var_shadows_param",
                   "for_var_shadows_outer_let", "for_continue_branch_let_var"}
NO_GUARD_SHADOW = ({"sibling_fors_reuse_var", "for_continue_step", "comment_let_before_assign_continue",
                    "comment_for_before_assign_break", "comment_let_mut_before_assign_break",
                    "comment_slashes_in_string_line"}
                   | {k for k in FIXTURES if k.startswith("mlab_")})


def shadow_refusal(got) -> bool:
    return (got[0] == "refused" and got[1] == 3 and not got[3]
            and ("native.block_shadow" in got[2] or "native.loop_body_shadow" in got[2]))


def check_fixtures(st, wd, lib):
    fails = 0
    for k, (src, want) in FIXTURES.items():
        st_, v = native_exit(st, src, wd, f"fx_{k}")
        ok = st_ == "ok" and v == want
        note = ""
        if st_ == "refused" and k in TWINS:
            tw = native_exit(st, TWINS[k], wd, f"fxt_{k}")
            ok = tw[0] == "refused"
            note = f" (twin without the shadow: {tw[0]}:{tw[1]})"
        elif st_ == "refused":
            # allowed only when the code generator alone builds it to `want` and
            # just the mic@3 trace-hash note builder refuses (see split_refusal)
            split = split_refusal(lib, src, wd, f"fx_{k}") if lib else ("no-so", None)
            ok = split == ("note", want)
            note = f" (split: {split[0]}:{split[1]})"
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  fixture {k}: native={st_}:{v} want={want}{note}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    # seed 1234 / count 200 compares 69 natively (the rest are refused by the
    # native control-flow subset, each with its shadow-free twin refused too)
    ap.add_argument("--min-compared", type=int, default=60)
    ap.add_argument("--mlir", action="store_true", default=os.environ.get("SHADOW_MLIR") == "1")
    ap.add_argument("--jobs", type=int, default=4)
    # The MLIR backend's known defect (a break/continue taken while a loop-body
    # `let` shadows an outer name carries the shadow out) is counted apart as
    # mlir_known_defect; any other MLIR-vs-model disagreement counts here.
    ap.add_argument("--max-mlir-disagree", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--bridge", action="store_true",
                    default=os.environ.get("SHADOW_BRIDGE") == "1")
    a = ap.parse_args(argv)
    # --bridge and --mlir both compile with mindc. Fail closed before the
    # frozen-seed stage1 build, which does not need a compiler and would
    # otherwise hide a missing binary behind a long, then-green, no-arg leg.
    if a.bridge or a.mlir:
        mindc_bin()
    td = pathlib.Path(tempfile.mkdtemp(prefix="shadow_scope_"))
    st = build_stage1(pathlib.Path(tempfile.gettempdir()) / "shadow_scope_stage1_cache")
    print(f"stage1: {st.name}")
    lib = so_lib()
    fails = check_fixtures(st, td, lib)
    has_mark = MARK in st.read_bytes()
    fails += not has_mark
    print(f"  {'PASS' if has_mark else 'FAIL'}  stage1 carries the scoped-nested-bindings mark "
          f"{MARK!r}")
    if a.mlir:
        # every fixture's lexical value under the MLIR build too; the fixtures whose
        # comments give a different MLIR exit carry a shadow out through a loop exit
        # (the known MLIR defect) and must still show it, so a fix is noticed
        for k, (src, want) in FIXTURES.items():
            ml = mlir_exit(src, td, f"fxm_{k}")
            if k in MLIR_DEFECT_FIXTURES:
                ok = ml == ("ok", MLIR_DEFECT_FIXTURES[k])
                tag = f"MLIR-KNOWN-DEFECT exit (lexical {want})"
            else:
                ok = ml == ("ok", want)
                tag = f"want={want}"
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  mlir fixture {k}: {ml} {tag}")
    if a.bridge:
        # which contract the embedded leg is held to is decided by the seed's own
        # bytes (the mark the Rust bridge reads), never assumed
        embedded_scopes = MARK in FROZEN_SEED.read_bytes()
        print(f"  NOTE  embedded default seed {'carries' if embedded_scopes else 'lacks'} "
              f"the scoped-nested-bindings mark {MARK!r}")
        for k, (src, want) in FIXTURES.items():
            got = bridge_exit(src, td, f"m_{k}", st)
            if k in BRIDGE_UNPROVEN:
                # the Rust guard does not admit this shadow shape for any compiler
                ok = shadow_refusal(got)
            else:
                # built to the model's value, or refused (no artifact) by a fence
                # other than the shadow guard (the native profile, the closure)
                ok = got == ("ok", want) or (got[0] == "refused" and got[1] == 3 and not got[3]
                                             and not shadow_refusal(got))
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  bridge(stage1) {k}: {got[:2]} want={want}"
                  + ("" if ok else f"\n{got[2] if len(got) > 2 else ''}"))
            emb = bridge_exit(src, td, f"e_{k}", None)
            tag = ""
            if embedded_scopes:
                # the reseeded embedded seed is a scoping compiler: the stage1
                # contract above, with no recorded-defect allowance
                if k in BRIDGE_UNPROVEN:
                    ok = shadow_refusal(emb)
                else:
                    ok = emb == ("ok", want) or (emb[0] == "refused" and emb[1] == 3
                                                 and not emb[3] and not shadow_refusal(emb))
            else:
                # the unmarked embedded seed never runs to a wrong value; where the
                # scoping compiler built the program, only the shadow guard stops it
                ok = emb == ("ok", want) or (emb[0] == "refused" and emb[1] == 3 and not emb[3])
                if got[0] == "ok" and k not in NO_GUARD_SHADOW:
                    ok = shadow_refusal(emb)
            if not ok and not embedded_scopes and emb == ("ok", EMBEDDED_SEED_DEFECT.get(k)):
                # the frozen seed's own miscompile (fixed in main.mind, shipped by
                # the next reseed): counted and named, not a scoping failure here
                ok, tag = True, f"  EMBEDDED-SEED-KNOWN-DEFECT (lexical {want})"
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  bridge(embedded seed) {k}: {emb[:2]}{tag}"
                  + ("" if ok else f"\n{emb[2:]}"))
    rng = random.Random(a.seed)
    progs = []
    while len(progs) < a.count:
        fns = Gen(rng, ("full", "lean", "exit", "noret", "exit")[len(progs) % 5]).program()
        if not has_nested_shadow(fns):
            continue
        trace = {}
        want = model_exit(fns, trace)
        twin = alpha_rename(fns)
        # the renamer must preserve the meaning and remove every nested shadow
        assert model_exit(twin) == want and not has_nested_shadow(twin), render(fns)
        progs.append((len(progs), fns, render(fns), want, render(twin),
                      bool(trace.get("exit_shadow"))))

    def one(p):
        i, fns, src, want, twin_src, exit_shadow = p
        nat = native_exit(st, src, td, f"g{i}")
        # a refusal is fail-closed only if the shadow-free twin is refused as well
        tw = native_exit(st, twin_src, td, f"t{i}") if nat[0] == "refused" else None
        ml = mlir_exit(src, td, f"g{i}") if a.mlir else None
        br = bridge_exit(src, td, f"g{i}", st) if a.bridge and nat[0] == "ok" else None
        return i, src, want, nat, ml, br, tw, exit_shadow

    stats = dict(compared=0, refused=0, note_only_refusal=0, shadow_only_refusal=0,
                 mismatch=0, crash=0,
                 exit_with_shadow=0, mlir_ok=0, mlir_mismatch=0, mlir_known_defect=0,
                 mlir_build_failed=0, bridge_ok=0, bridge_bad=0)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for i, src, want, nat, ml, br, tw, exit_shadow in ex.map(one, progs):
            stats["exit_with_shadow"] += exit_shadow
            if br is not None:
                if br == ("ok", want):
                    stats["bridge_ok"] += 1
                else:
                    stats["bridge_bad"] += 1
                    print(f"  FAIL  gen{i}: bridge(stage1) {br} want={want}\n{src}")
            if nat[0] == "ok":
                stats["compared"] += 1
                if nat[1] != want:
                    stats["mismatch"] += 1
                    print(f"  FAIL  gen{i}: native exit {nat[1]} != model {want}\n{src}")
            elif nat[0] == "refused":
                stats["refused"] += 1
                if tw[0] != "refused":
                    split = split_refusal(lib, src, td, f"g{i}") if lib else ("no-so", None)
                    if split == ("note", want):
                        # the code generator scoped it and ran to the model's value;
                        # only the mic@3 trace-hash note builder refused (fail-closed)
                        stats["note_only_refusal"] += 1
                        print(f"  NOTE  gen{i}: refused only by the mic@3 note builder; the "
                              f"code generator alone exits {want} (= model)")
                    else:
                        stats["shadow_only_refusal"] += 1
                        print(f"  FAIL  gen{i}: refused natively, its shadow-free twin built "
                              f"({tw}), split={split} want={want}\n{src}")
            else:
                stats["crash"] += 1
                print(f"  FAIL  gen{i}: {nat[1]}\n{src}")
            if ml is not None:
                if ml[0] == "ok":
                    stats["mlir_ok"] += 1
                    if ml[1] != want and exit_shadow:
                        # the known MLIR defect: the exit edge carries the shadow
                        stats["mlir_known_defect"] += 1
                        print(f"  MLIR-KNOWN-DEFECT  gen{i}: MLIR exit {ml[1]} != model {want} "
                              f"(native {nat}; a loop exit with a body-local shadow in scope)")
                    elif ml[1] != want:
                        stats["mlir_mismatch"] += 1
                        print(f"  MLIR-DISAGREE  gen{i}: MLIR exit {ml[1]} != model {want} "
                              f"(native {nat})\n{src}")
                else:
                    stats["mlir_build_failed"] += 1
                    print(f"  NOTE  gen{i}: MLIR build failed: {ml[1]!r}")
            if a.dump:
                d = pathlib.Path(a.dump)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"gen{i}.mind").write_text(src)
    print(f"generated: count={a.count} seed={a.seed} " +
          " ".join(f"{k}={v}" for k, v in stats.items()))
    bad = (fails + stats["mismatch"] + stats["crash"] + stats["bridge_bad"]
           + stats["shadow_only_refusal"])
    if stats["mlir_mismatch"] > a.max_mlir_disagree:
        print(f"  FAIL  {stats['mlir_mismatch']} MLIR-vs-model disagreements "
              f"(> --max-mlir-disagree {a.max_mlir_disagree})")
        bad += 1
    if stats["compared"] < a.min_compared:
        print(f"  FAIL  only {stats['compared']} programs built natively and were compared "
              f"(< --min-compared {a.min_compared})")
        bad += 1
    print("ALL PASS" if bad == 0 else f"FAILED ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
