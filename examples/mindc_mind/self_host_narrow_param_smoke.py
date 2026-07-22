#!/usr/bin/env python3
"""Native-ELF smoke for narrow-width (i8/i16/i32) function PARAMETERS carried by a loop.

A narrow param REASSIGNED inside a TOP-LEVEL `while` now lowers CORRECTLY by the
pure-MIND nb native path: nb_while_carry mints the same fresh width-wrap slot the
emit does for a narrow carried assign (post_id = the wrap slot, next_id advanced
+1), so the carried var's working slot and every later carried var stay
frame-consistent with the emit + the frame count (nb_count_stmt's assign arm). The
value re-wraps two's-complement each iteration and post-loop reads resolve to the
live slot — no stale value, no hang, no frame undercount. Previously ALL narrow
param + loop shapes fail-closed; this proves the top-level-carry sub-shape now
EMITS and RUNS to the value an INDEPENDENT Python reference computes.

A narrow param REASSIGNED by a while NESTED inside an `if` branch now ALSO lowers
correctly: the enclosing if-region's merge machinery collects the nested while's
top-level live_writes as branch writes (nb_branch_writes' ast_while arm ->
nb_while_live_writes), allocates a merge phi per carried name, and nb_rebind_merges
binds them into the post-if let-env, so the post-loop read resolves the carried
value; the narrow width-wrap slot is minted+counted identically on both the
nb_while_carry pre-walk and the nb_count_stmt assign arm via the param-width
let-env binding. Verified at if-depth 1 AND 2, in the then- and else-branch, and
when the enclosing if is not taken. Gated by nb_region_while_carries_narrow, which
permits a top-level narrow carry, an if-nested narrow carry, OR a narrow carry in a
DIRECT nested while (to any depth) — but descends into a `while` body ONLY through
DIRECT nested whiles (nb_body_while_carries_narrow_direct, no `if` arm), matching
exactly what the loop-carry promotion (nb_while_carry_wnest) handles. An inner while
wrapped in an `if` INSIDE a loop body (`while{ if{ while{ narrow += 1 } } }`) is NOT
promoted to the outer carry, so it stays FAIL-CLOSED (asserted in IF_WRAPPED_WHILE
below) — permitting it would silently drop the carry. The pre-existing i64
`while{ if{ x += 1 } }` carry-drop gap (a plain if-wrapped assign in a loop, no
narrow param, so not gated by this guard at all) is a SEPARATE generic-path bug,
documented below (I64_IF_IN_WHILE_GAP) and left for a follow-up — not fixed here.

F2 FIXED: a narrow param REASSIGNED by a while NESTED inside ANOTHER while (nested
loops mutating an OUTER carried var) now lowers CORRECTLY. nb_while_carry descends
into the inner while, computes each inner-carried var's post_id (its inner-loop exit
slot), and PROMOTES every one that is an OUTER (pre-loop) var into the OUTER carry
table with that inner post_id — the native analogue of lower.rs
last_region_exit_rebindings + record_loop_mut over `env`. The outer copy-in then
seeds the inner-exit slot ONCE before the header (so the value accumulates across
outer iterations) and the inner copy-in becomes a no-op. nb_count_carried descends
identically so the exit_id frame slots stay in lockstep with the emit (a mismatch
would corrupt the exact runtime value — proven below by while{while{}} 3x4=12,
triple-nest 2x2x2=8, i8/i16/i32/i64 + overflow wrap). The bug was width-independent
(i64 miscompiled too — returned 2 for a 2x2 loop, want 4); it is now correct for
every width. Body-local `let mut` counters (bound after the outer loop) are NOT
promoted (they reset each iteration), matching lower.rs's `env.contains_key` guard.

One sub-shape remains genuinely broken and MUST still fail closed (empty ELF):
  * a narrow read-only param CARRIED ACROSS a top-level loop then returned — the
    narrow-param loop-carry machinery only records a REASSIGNED carry slot; a
    read-only narrow param read post-loop (with or without a trailing cast/binop)
    is not yet threaded. Independent of the cast: even `return x as i64` (no binop)
    is refused when a loop is present. Refused here (loud, no silent miscompile).

The `(x as i64) + c` CAST-IN-BINOP lowering itself is now FIXED: parse_postfix_rest
consumes `as TYPE` in postfix position so the cast binds tighter than any infix op
and the trailing binop operand is lowered, not dropped. Previously it SILENTLY
mis-lowered — the `+ c` became dead code after `ret`, so `(x as i64)+3` of f(10)
returned 10 not 13. Asserted below (CAST_BINOP) over narrow (i8/i16/i32) params
widened into +/* binops with NO loop (the shape the bug reproduces on): emit + run
== the independent Python value.

Asserts: (1) the top-level-carry shapes EMIT and run to the Python-reference exit
(incl. two's-complement overflow wrap); (2) the now-fixed no-loop cast-in-binop
shapes EMIT and run correct; (3) the two broken sub-shapes emit an EMPTY ELF
(refused); (4) i64 + narrow-no-loop controls still emit + run (no over-rejection).
Guarded on >=1 of each so it cannot pass vacuously.

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


def wrap(v: int, bits: int) -> int:
    """Independent two's-complement wrap-to-width reference (no compiler involved)."""
    m = 1 << bits
    v &= m - 1
    if v >= (m >> 1):
        v -= m
    return v


def _carry_ref(start: int, iters: int, bits: int) -> int:
    """Reference model of `x: iN` reassigned `x = x + 1` `iters` times, then read."""
    x = start
    for _ in range(iters):
        x = wrap(x + 1, bits)
    return wrap(x, bits)


# EMIT + RUN correct: a narrow param reassigned by a TOP-LEVEL while now carries
# correctly. `exp` is computed by the independent Python reference above, then the
# process exit is the low byte of it.
CARRY = [
    ("i8 param reassigned in top-level loop, f(10) +3",
     _carry_ref(10, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i16 param reassigned in top-level loop, f(100) +5",
     _carry_ref(100, 5, 16),
     "fn f(x: i16) -> i64 {\n    let mut c: i64 = 0;\n    while c < 5 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(100); }\n"),
    ("i8 param OVERFLOW two's-complement wrap, f(126) +3 -> -127",
     _carry_ref(126, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(126); }\n"),
]
# EMIT + RUN correct: the now-fixed cast-in-binop. A narrow (i8/i16/i32) param is
# widened via `(x as i64)` and fed into a +/* binop with NO loop — the exact shape
# the dropped-`+c` bug reproduced on. `exp` is the plain integer value; process exit
# is its low byte. Previously each of these SILENTLY returned the un-added operand.
CAST_BINOP = [
    ("i8 param (x as i64)+3, f(10)", 13,
     "fn f(x: i8) -> i64 {\n    return (x as i64) + 3;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i8 param (x as i64)*2+1, f(4)", 9,
     "fn f(x: i8) -> i64 {\n    return (x as i64) * 2 + 1;\n}\nfn main() -> i64 { return f(4); }\n"),
    ("i16 param (x as i64)+7, f(100)", 107,
     "fn f(x: i16) -> i64 {\n    return (x as i64) + 7;\n}\nfn main() -> i64 { return f(100); }\n"),
    ("i32 param (x as i64)+2, f(40)", 42,
     "fn f(x: i32) -> i64 {\n    return (x as i64) + 2;\n}\nfn main() -> i64 { return f(40); }\n"),
    ("i8 param no-paren `x as i64 + 5` (as binds tighter than +), f(10)", 15,
     "fn f(x: i8) -> i64 {\n    return x as i64 + 5;\n}\nfn main() -> i64 { return f(10); }\n"),
]
# EMIT + RUN correct: NARROWING casts `as i8/i16/i32` in a binop. The operand is an
# i64 param holding the FULL value; `(y as iN)` truncates it two's-complement to N
# bits (native movsx rax,al/ax / movsxd rax,eax — the same wrap the __mind_wrap_iN
# intrinsics use), then composes as a normal integer binop operand. `exp` is the low
# byte of an INDEPENDENT Python two's-complement-wrap reference (wrap(), above), NOT
# the compiler. Previously narrowing casts fed into a binop were fail-closed (a plain
# `as i64` passthrough would be WRONG for a narrowing target); now they truncate.
NARROW_CAST = [
    ("(y as i8)+1 f(300) -> wrap(300,8)=44, +1", (wrap(300, 8) + 1) & 0xFF,
     "fn f(y: i64) -> i64 {\n    return (y as i8) + 1;\n}\nfn main() -> i64 { return f(300); }\n"),
    ("(y as i8)+1 f(200) OVERFLOW -> wrap(200,8)=-56, +1", (wrap(200, 8) + 1) & 0xFF,
     "fn f(y: i64) -> i64 {\n    return (y as i8) + 1;\n}\nfn main() -> i64 { return f(200); }\n"),
    ("(y as i16)+7 f(70000) -> wrap(70000,16)=4464, +7", (wrap(70000, 16) + 7) & 0xFF,
     "fn f(y: i64) -> i64 {\n    return (y as i16) + 7;\n}\nfn main() -> i64 { return f(70000); }\n"),
    ("(y as i32)+2 f(4294967301) -> high bits masked, wrap=5, +2", (wrap(4294967301, 32) + 2) & 0xFF,
     "fn f(y: i64) -> i64 {\n    return (y as i32) + 2;\n}\nfn main() -> i64 { return f(4294967301); }\n"),
    ("no-paren `y as i8 + 5` (as binds tighter than +) f(300)", (wrap(300, 8) + 5) & 0xFF,
     "fn f(y: i64) -> i64 {\n    return y as i8 + 5;\n}\nfn main() -> i64 { return f(300); }\n"),
    ("(y as i8)*2 f(200) -> wrap(200,8)=-56, *2=-112", (wrap(200, 8) * 2) & 0xFF,
     "fn f(y: i64) -> i64 {\n    return (y as i8) * 2;\n}\nfn main() -> i64 { return f(200); }\n"),
    ("return (y as i8) f(200) (no binop, widened i64 return) -> -56", wrap(200, 8) & 0xFF,
     "fn f(y: i64) -> i64 {\n    return (y as i8);\n}\nfn main() -> i64 { return f(200); }\n"),
]
# EMIT + RUN correct (ITEM 2): a narrow-declared LOCAL (`let mut y: i8`) carried by a
# TOP-LEVEL while — the c27a766 twin-slot narrow-carry mechanism keyed on a narrow
# carried assign TARGET, which covers a narrow LOCAL, not just a narrow param. `exp`
# is the independent Python reference: y re-wraps two's-complement each iteration.
NARROW_LOCAL_LOOP = [
    ("i8 LOCAL y=0 reassigned in top-level loop x3", _carry_ref(0, 3, 8),
     "fn f() -> i64 {\n    let mut y: i8 = 0;\n    let mut c: i64 = 0;\n    while c < 3 {\n        y = y + 1;\n        c = c + 1;\n    }\n    return y as i64;\n}\nfn main() -> i64 { return f(); }\n"),
    ("i8 LOCAL y=125 OVERFLOW two's-complement wrap x5 -> -126", _carry_ref(125, 5, 8),
     "fn f() -> i64 {\n    let mut y: i8 = 125;\n    let mut c: i64 = 0;\n    while c < 5 {\n        y = y + 1;\n        c = c + 1;\n    }\n    return y as i64;\n}\nfn main() -> i64 { return f(); }\n"),
]
# EMIT + RUN correct: a narrow param REASSIGNED by a while NESTED inside an `if`
# branch. The if-region merge machinery threads the nested while's carry rebinding out
# through the post-if exit env (nb_region_while_carries_narrow now permits it). `exp`
# is the independent Python carry reference (the loop runs iff the enclosing if is
# taken). Covers if-depth 1 & 2, else-branch, overflow wrap, and an untaken if.
NESTED_CARRY = [
    ("i8 param while-in-IF carry, f(10) if c<1 while c<4 x+=1", _carry_ref(10, 4, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    if c < 1 {\n        while c < 4 {\n            x = x + 1;\n            c = c + 1;\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i8 param while-in-IF OVERFLOW two's-complement wrap, f(126) x3", _carry_ref(126, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    if c < 1 {\n        while c < 3 {\n            x = x + 1;\n            c = c + 1;\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(126); }\n"),
    ("i16 param while-in-IF carry, f(100) x5", _carry_ref(100, 5, 16),
     "fn f(x: i16) -> i64 {\n    let mut c: i64 = 0;\n    if c < 1 {\n        while c < 5 {\n            x = x + 1;\n            c = c + 1;\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(100); }\n"),
    ("i8 param while in IF-in-IF (depth 2), f(10) x3", _carry_ref(10, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    if 1 < 2 {\n        if c < 1 {\n            while c < 3 {\n                x = x + 1;\n                c = c + 1;\n            }\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i8 param while in ELSE-branch, f(10) x4", _carry_ref(10, 4, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    if c > 5 {\n        c = c + 1;\n    } else {\n        while c < 4 {\n            x = x + 1;\n            c = c + 1;\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i8 param while-in-IF, enclosing if NOT taken, f(10) -> unchanged", _carry_ref(10, 0, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 5;\n    if c < 1 {\n        while c < 4 {\n            x = x + 1;\n            c = c + 1;\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
]
# EMIT + RUN correct (F2 FIXED): a var REASSIGNED by a while NESTED inside another
# while (nested loops mutating an OUTER carried var). `exp` is the independent Python
# carry reference. Covers i8/i16/i32/i64, multiple iteration counts, overflow wrap,
# and a triple-nested loop. Formerly REFUSED (and, unguarded for i64, MISCOMPILED to
# 2 instead of 4). Now emits + runs to the reference value at every width.
def _nested_ref(iters_outer: int, iters_inner: int, start: int, bits: int) -> int:
    x = start
    for _ in range(iters_outer):
        for _ in range(iters_inner):
            x = wrap(x + 1, bits)
    return wrap(x, bits)


NESTED_LOOP = [
    ("i64 while-in-while 2x2 x+=1 -> 4", _nested_ref(2, 2, 0, 64),
     "fn f() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 2 {\n        let mut c: i64 = 0;\n        while c < 2 { x = x + 1; c = c + 1; }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(); }\n"),
    ("i64 while-in-while 3x4 -> 12", _nested_ref(3, 4, 0, 64),
     "fn f() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 3 {\n        let mut c: i64 = 0;\n        while c < 4 { x = x + 1; c = c + 1; }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(); }\n"),
    ("i8 param while-in-while carry, f(0) 2x2 -> 4", _nested_ref(2, 2, 0, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 2 {\n        let mut d: i64 = 0;\n        while d < 2 {\n            x = x + 1;\n            d = d + 1;\n        }\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
    ("i8 param while-in-while OVERFLOW two's-complement wrap, f(126) 2x2", _nested_ref(2, 2, 126, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 2 {\n        let mut d: i64 = 0;\n        while d < 2 {\n            x = x + 1;\n            d = d + 1;\n        }\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(126); }\n"),
    ("i16 param while-in-while 3x3=+9, f(100) -> 109", _nested_ref(3, 3, 100, 16),
     "fn f(x: i16) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        let mut d: i64 = 0;\n        while d < 3 {\n            x = x + 1;\n            d = d + 1;\n        }\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(100); }\n"),
    ("i32 param while-in-while 2x5=+10, f(40) -> 50", _nested_ref(2, 5, 40, 32),
     "fn f(x: i32) -> i64 {\n    let mut c: i64 = 0;\n    while c < 2 {\n        let mut d: i64 = 0;\n        while d < 5 {\n            x = x + 1;\n            d = d + 1;\n        }\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(40); }\n"),
    ("i64 TRIPLE-nested while 2x2x2 -> 8", 8,
     "fn f() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 2 {\n        let mut b: i64 = 0;\n        while b < 2 {\n            let mut c: i64 = 0;\n            while c < 2 { x = x + 1; c = c + 1; }\n            b = b + 1;\n        }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(); }\n"),
    ("i8 param while-in-while, outer reset of inner counter each iter, f(10) 3x2 -> 16", _nested_ref(3, 2, 10, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        let mut d: i64 = 0;\n        while d < 2 {\n            x = x + 1;\n            d = d + 1;\n        }\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
]
# REFUSE: the remaining genuinely-broken sub-shape must emit an EMPTY ELF.
# EMIT + RUN correct (SUB-STEP B): a var DIRECTLY assigned inside an `if` INSIDE a loop
# body (`while{ if{ x += 1 } }`) now carries across the loop. nb_while_carry_ifnest records
# the branch-assigned OUTER var with post_id = the if's merge slot (pulled from the SHARED
# nb_count_stmt total, NOT re-derived), so the loop copy-in/exit thread it. Covers i8/i16/
# i32/i64 (+ overflow), always-taken and cond-selective. `exp` is the independent Python
# carry reference (x increments only on iterations where the inner if is taken). This is the
# fix for the former I64 `while{ if{ x+=1 } }` gap (Finding #2, single-branch part).
IF_IN_LOOP_CARRY = [
    ("i8 param while{ if(a<10, taken){ x+=1 } } x3, f(10) -> 13", _carry_ref(10, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a < 10 { x = x + 1; }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i8 param while{ if(taken){ x+=1 } } OVERFLOW f(126) x3 -> -127", _carry_ref(126, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a < 10 { x = x + 1; }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(126); }\n"),
    ("i16 param while{ if(taken){ x+=1 } } x5 f(100) -> 105", _carry_ref(100, 5, 16),
     "fn f(x: i16) -> i64 {\n    let mut a: i64 = 0;\n    while a < 5 {\n        if a < 10 { x = x + 1; }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(100); }\n"),
    ("i32 param while{ if(taken){ x+=1 } } x4 f(40) -> 44", _carry_ref(40, 4, 32),
     "fn f(x: i32) -> i64 {\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 10 { x = x + 1; }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(40); }\n"),
    # two DIFFERENT narrow params, each written in EXACTLY ONE branch (x in then, y in else) —
    # each is a genuine single-branch carry (the sibling does not write it), so both carry
    # correctly. x=2 (a=0,1), y=2 (a=2,3) -> x*10+y = 22. (The write-aware per-param check
    # permits this precisely; a coarse "any narrow assign per branch" count would over-refuse.)
    ("i8 two-diff-params while{ if{ x+=1 } else { y+=1 } } 4 iters -> 22", 22,
     "fn f(x: i8, y: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { x = x + 1; } else { y = y + 1; }\n        a = a + 1;\n    }\n    return (x as i64) * 10 + (y as i64);\n}\nfn main() -> i64 { return f(0, 0); }\n"),
    ("i8 param while{ if(a<2, selective){ x+=1 } } 5 iters f(10) -> 12", _carry_ref(10, 2, 8),
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 5 {\n        if a < 2 { x = x + 1; }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i64 param while{ if(taken){ x+=1 } } x3, f(10) -> 13 (I64 gap Finding #2 FIXED)", 13,
     "fn f(x: i64) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a < 10 { x = x + 1; }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(10); }\n"),
]
REFUSE = [
    ("i8 read-only param carried across a top-level loop then (x as i64)+c "
     "(narrow read-only loop-carry gap — NOT the cast, which now composes; "
     "even `return x as i64` w/ a loop is refused)",
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        c = c + 1;\n    }\n    return (x as i64) + c;\n}\nfn main() -> i64 { return f(10); }\n"),
]
# REFUSE (regression guard for commit 0b5f489): a narrow param reassigned by a while
# wrapped in an `if` INSIDE a loop body (`while{ if{ while{ x += 1 } } }`). The inner
# while is NOT promoted to the OUTER loop's carry (nb_while_carry_nonassign only
# descends into a DIRECT nested while — nb_while_carry_wnest — never through an if), so
# the narrow var is dropped across the outer loop. Before 0b5f489 this matched neither
# guard arm -> refused (correct); 0b5f489's F2 fix let the guard's while-body recursion
# reach the inner while THROUGH the if arm and wrongly PERMIT it, silently miscompiling
# (independently confirmed got=2 want=6). The guard's while-body descent is now
# direct-while-only (nb_body_while_carries_narrow_direct), so this stays fail-closed.
# MUST emit an EMPTY ELF for every narrow width.
IF_WRAPPED_WHILE = [
    ("i8 param while{ if{ while{ x+=1 } } } 3x2 (if-wrapped inner while — carry NOT "
     "promoted to outer loop, must fail-closed)",
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        let mut d: i64 = 0;\n        if a < 10 {\n            while d < 2 { x = x + 1; d = d + 1; }\n        }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
    ("i16 param while{ if{ while{ x+=1 } } } 3x2 (if-wrapped inner while — must "
     "fail-closed)",
     "fn f(x: i16) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        let mut d: i64 = 0;\n        if a < 10 {\n            while d < 2 { x = x + 1; d = d + 1; }\n        }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
    ("i32 param while{ if{ while{ x+=1 } } } 3x2 (if-wrapped inner while — must "
     "fail-closed)",
     "fn f(x: i32) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        let mut d: i64 = 0;\n        if a < 10 {\n            while d < 2 { x = x + 1; d = d + 1; }\n        }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
]
# REFUSE (blind-review XN class, regression guard for the UNCOMMITTED Sub-step B): a narrow
# param assigned TOP-LEVEL in one if branch while the SIBLING branch ALSO writes it via
# NESTED control (`else { if{ x+=5 } }` / `else { while{ x+=1 } }`). A non-recursive
# "single-branch" check misclassified these as clean single-branch and carried x while
# DROPPING the sibling's nested write — a running ELF with the WRONG value (XN1 emitted
# exit 7 want 12). The sibling-write check is now RECURSIVE (nb_block_writes_rec), so any
# sibling write at any depth fails these closed. MUST emit an EMPTY ELF for every width.
XN_SIBLING_NESTED_WRITE = [
    ("i8 while{ if{ x+=1 } else { if{ x+=5 } } } — else writes x via nested if (must "
     "fail-closed, was leaking exit 7 want 12)",
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { x = x + 1; } else { if a < 10 { x = x + 5; } }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
    ("i8 while{ if{ if{ x+=5 } } else { x+=1 } } — then writes x via nested if (mirror, "
     "must fail-closed)",
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { if a < 10 { x = x + 5; } } else { x = x + 1; }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
    ("i8 while{ if{ x+=1 } else { while{ x+=1 } } } — else writes x via nested while (must "
     "fail-closed)",
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 4 {\n        let mut d: i64 = 0;\n        if a < 2 { x = x + 1; } else { while d < 3 { x = x + 1; d = d + 1; } }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
    ("i8 while{ if{ x+=1 } else { if{ while{ x+=1 } } } } — else writes x via if>while "
     "(deep, must fail-closed)",
     "fn f(x: i8) -> i64 {\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { x = x + 1; } else { if a < 10 { let mut d: i64 = 0; while d < 2 { x = x + 1; d = d + 1; } } }\n        a = a + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(0); }\n"),
]
# SUB-STEP B FIXED the single-branch part of Finding #2: i64/narrow `while{ if{ x+=1 } }`
# now carries correct (returns 3, asserted in IF_IN_LOOP_CARRY above). Two related shapes
# remain, tracked as follow-ups, NOT asserted correct here:
#   * Sub-step C: i64/narrow `while{ if{ while{ x+=1 } } }` (if-WRAPPED inner while) still
#     drops the carry (got=2 want=6) — the inner while is not promoted to the outer loop.
#     Stays fail-closed for narrow (IF_WRAPPED_WHILE below); the i64 form is a known gap.
#   * A separate, pre-existing nb_if_stmt_merged merge-read bug: a var assigned in BOTH
#     branches of an if (`if{ x+=1 } else { x+=2 }`) mis-reads on the else side (reproduces
#     standalone: `let mut x=5; if 1<0 {x=x+1} else {x=x+2}` -> 2 not 7). nb_while_carry_ifnest
#     deliberately does NOT promote a both-branch-same-var (XOR filter), so that shape is
#     unchanged (not newly miscompiled). Awaits a merge-read fix; out of Sub-step B scope.
I64_IF_IN_WHILE_GAP_NOTE = (
    "REMAINING (not this fix, pre-existing gaps): i64 while{ if{ while{ x+=1 } } } returns "
    "2 not 6 (Sub-step C, if-wrapped inner while); i64 while{ if{ if{ x+=1 } } } returns 1 "
    "not 3 (if-in-if — x written only via nested if in ONE branch, not a top-level "
    "single-branch carry, unpromoted); both-branch-same-var if merge-read bug "
    "(if{x+=1}else{x+=2}) — all tracked as follow-ups"
)
# WORK controls: i64 loops are unaffected; a narrow param with NO loop lowers via the
# entry width-wrap driver (must NOT be over-rejected).
I64_CONTROLS = [
    ("i64 param reassigned in loop", 13,
     "fn f(x: i64) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i64 add", 5,
     "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\nfn main() -> i64 { return add(2, 3); }\n"),
    ("narrow param NO loop (read/return works via wrap driver — must NOT be rejected)", 5,
     "fn pw(x: i32) -> i64 {\n    return x;\n}\nfn main() -> i64 { return pw(5); }\n"),
]


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    r = subprocess.run([mindc, MAIN_MIND, "--emit-shared", out], capture_output=True, text=True)
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

    all_ok = True
    carried = 0
    cast_ok = 0
    narrow_cast_ok = 0
    narrow_local_ok = 0
    nested_ok = 0
    nested_loop_ok = 0
    if_in_loop_ok = 0
    xn_refused = 0
    refused = 0
    if_wrapped_refused = 0
    ran = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        def run_elf(elf: bytes) -> int:
            p = tmp / "m.elf"
            p.write_bytes(elf)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            return subprocess.run([str(p)], timeout=10).returncode

        for label, exp, src in CARRY:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  narrow carry OVER-REJECTED: {label} (emit 0B, want run)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            carried += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  narrow param carried by loop: {label} "
                  f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, exp, src in CAST_BINOP:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  cast-in-binop REFUSED (want emit+run): {label} (emit 0B)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            cast_ok += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  cast-in-binop now composes: {label} "
                  f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, exp, src in NARROW_CAST:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  narrowing cast REFUSED (want emit+run): {label} (emit 0B)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            narrow_cast_ok += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  narrowing cast truncates + composes: {label} "
                  f"-> exit {rc} (python-wrap-ref byte {want})")

        for label, exp, src in NARROW_LOCAL_LOOP:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  narrow LOCAL carry OVER-REJECTED: {label} (emit 0B, want run)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            narrow_local_ok += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  narrow LOCAL carried by loop (item 2): {label} "
                  f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, exp, src in NESTED_CARRY:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  nested-if carry OVER-REJECTED: {label} (emit 0B, want run)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            nested_ok += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  narrow param carried by if-nested loop: {label} "
                  f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, exp, src in NESTED_LOOP:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  nested-loop carry OVER-REJECTED: {label} (emit 0B, want run)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            nested_loop_ok += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  var carried by while-nested-in-while (F2 fixed): {label} "
                  f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, exp, src in IF_IN_LOOP_CARRY:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  if-in-loop carry OVER-REJECTED: {label} (emit 0B, want run)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            if_in_loop_ok += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  var carried by if-in-loop (Sub-step B): {label} "
                  f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, src in REFUSE:
            elf = emit(src)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            refused += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  still-broken sub-shape refused: {label} "
                  f"(emit {len(elf)}B, want 0 — fail-closed, NOT run)")

        for label, src in IF_WRAPPED_WHILE:
            elf = emit(src)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            if_wrapped_refused += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  if-wrapped inner while refused "
                  f"(0b5f489 regression guard): {label} (emit {len(elf)}B, want 0 — "
                  f"fail-closed, NOT run)")

        for label, src in XN_SIBLING_NESTED_WRITE:
            elf = emit(src)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            xn_refused += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  sibling-nested-write refused "
                  f"(XN silent-miscompile guard): {label} (emit {len(elf)}B, want 0 — "
                  f"fail-closed, NOT run)")

        print(f"  NOTE  documented pre-existing gap (NOT fixed here): {I64_IF_IN_WHILE_GAP_NOTE}")

        for label, exp, src in I64_CONTROLS:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  control OVER-REJECTED: {label} (emit 0B)")
                all_ok = False
                continue
            rc = run_elf(elf)
            ok = rc == exp
            all_ok = all_ok and ok
            ran += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  control still works: {label} -> exit {rc} (want {exp})")

    if carried < 1:
        print("FAIL: vacuous (no narrow-param carry ran)")
        return 1
    if cast_ok < 1:
        print("FAIL: vacuous (no cast-in-binop shape emitted + ran)")
        return 1
    if narrow_cast_ok < 1:
        print("FAIL: vacuous (no narrowing-cast shape emitted + ran)")
        return 1
    if narrow_local_ok < 1:
        print("FAIL: vacuous (no narrow-local-in-loop shape emitted + ran)")
        return 1
    if nested_ok < 1:
        print("FAIL: vacuous (no narrow-param if-nested-loop carry emitted + ran)")
        return 1
    if nested_loop_ok < 1:
        print("FAIL: vacuous (no while-nested-in-while carry emitted + ran)")
        return 1
    if if_in_loop_ok < len(IF_IN_LOOP_CARRY):
        print("FAIL: vacuous/incomplete (a Sub-step B if-in-loop carry did not run correct)")
        return 1
    if refused < 1:
        print("FAIL: vacuous (no broken sub-shape refused)")
        return 1
    if if_wrapped_refused < len(IF_WRAPPED_WHILE):
        print("FAIL: an if-wrapped inner-while shape was PERMITTED (0b5f489 regression "
              "— must be fail-closed for every narrow width)")
        return 1
    if xn_refused < len(XN_SIBLING_NESTED_WRITE):
        print("FAIL: a sibling-nested-write shape was PERMITTED (XN silent-miscompile "
              "regression — the sibling branch writes the carried var, must be fail-closed)")
        return 1
    if ran < 1:
        print("FAIL: vacuous (no i64 control ran)")
        return 1
    if all_ok:
        print("ALL PASS  narrow-width params/locals carried by a top-level loop AND by "
              "a while NESTED inside an if-branch (any if-depth, then/else, overflow, "
              "untaken) emit + run correct (two's-complement wrap, no stale/hang) vs an "
              "independent Python ref, the widening cast-in-binop `(x as i64)+c` "
              "composes, NARROWING casts `(y as i8/i16/i32)` in a binop truncate "
              "two's-complement (movsx) + compose, a var carried by a while NESTED in "
              "another while (any width, overflow, triple-nest) now emits + runs correct "
              "(F2 fixed), while the narrow read-only loop-carry sub-shape AND an inner "
              "while wrapped in an `if` inside a loop body (0b5f489 regression guard) "
              "stay fail-closed and i64 fns are unaffected")
        return 0
    print("FAIL  narrow-param carry smoke mis-behaved")
    return 1


if __name__ == "__main__":
    sys.exit(main())
