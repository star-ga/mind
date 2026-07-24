#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND B1 E2003 rule — unknown call.

Ports resolve.rs's Node::Call arm (UNKNOWN_CALL_CODE, E2003,
resolve.rs:774-832): a call whose BARE callee resolves through NO symbol
source. The exact Rust condition — the else-if chain's final arm — is
`!call_resolvable(callee)` (resolve.rs:572-634):

    call_resolvable(name) =
         ident_resolvable(name):
           scopes.contains(name)          [D2 — live local binding]
        || syms.name_resolvable(name)     [D1 module decls + prelude
                                           OR D3 bundled-std exports]
        || name == "bytes"
        || name.contains("::")            [qualified — never bare-undef]
        || cm_symbol_exported_res(name)   [vacuous single-file]
      || name.starts_with("__mind_")      [blanket intrinsic namespace]
      || name.starts_with("tensor.")      [folded callee — dotted shape]
      || name == "gen_deref"
      || name in BARE_BUILTINS            [34-name empirically-swept set]
      || std_surface_intrinsic_arity(name).is_some()   [E2024 table: "byte"]
      || cm_lookup_fn(name).is_some()     [vacuous single-file]

Precedence at the Call arm (grounded): unknown_variant (E2008, `::` head) →
is_fn_value_call (E2012, callee resolves ONLY as a local binding) →
!call_resolvable (E2003). Both earlier arms imply call_resolvable is TRUE, so
E2003 == !call_resolvable at a bare-callee position. The E2012-vs-E2003
boundary is exactly the scope axis: a live local binding (and nothing else)
fires E2012, never E2003; no binding and no union accept fires E2003.

Live-grounded boundaries (probe battery 2026-07-23):
  * keyword-spelled callees in EXPRESSION position (`let z = return(1)`,
    `loop(1)`, `break(1)`, `in(1)`, `as(1)`, `mut(1)`) parse as Call and
    FIRE E2003; statement-position keyword occurrences (`while (c) {`,
    `return (3)`, `match (v) {`) are header/statement consumption — never;
  * FOLDED-KEYWORD callees (the self-host lexer emits `fn`/`if`/`else`/
    `let`/`use`/`pub`/`import` as dedicated tk_kw_* kinds, NOT tk_ident —
    live mindc lexes them as plain idents): `fn`/`let`/`use`/`import`/`pub`
    FIRE E2003 in expression + match-arm-body position (statement position
    is decl/import-parse consumed — no Call); `else` ALSO fires at a bare
    statement start (after `{`/`;`/newline/a non-if `}`) with the single
    no-fire shape being the consumed `} else (…)` of an if/else-if arm
    (incl. newline-separated); `if` NEVER fires (if-expression header).
    Routed through tc_uc_kw_callee by token KIND — the round-2 fail-OPEN
    class this smoke pins RED-without-the-route;
  * `true(1)` is a value-literal head, never a Call — no diagnostic;
  * a dotted `head.rest(args)` is a MethodCall whose receiver is walked as a
    value USE (`undefmod.undefn(1)` fires E2002 on the head, never E2003) or
    a folded `tensor.*` callee (blanket-accepted, `tensor.zork(1)` is clean);
  * a module-level call is OUTSIDE resolve_fn_body — live fires E2001 there;
  * the resolve.rs module doc-comment's "unresolved imports SUPPRESS
    undefined-call diagnostics" (resolve.rs:45-50, 333-335) is STALE — no
    suppression exists in the implementation: E2003 fires under BOTH
    `import std.vec` and an unresolvable non-std import (pinned below as
    positive classify cases + a live sentinel).

The pure-MIND twin `selftest_tc_unknown_call(src, src_len, pos, std_src,
std_len)` takes the fixture source, the byte offset of the queried CALLEE
ident, and the CONCATENATED bundled-std source (the D3 caller-supplied
model). Returns 1 (E2003 fires) / 0 (the callee resolves, or E2003
structurally never fires at this shape) / -3 (fail-closed sentinel: not a
classifiable bare-callee position).

Three-way agreement, machine-checked (no hand-authored verdict table), with
LIVE AUTHORITATIVE ON EVERY CASE and leg 2 fully independent of the port's
position classifier (Rule 3a):
  1. MIND core over (source, pos, std concat).
  2. The E2003 RESOLUTION UNION recomputed independently in Python — the D2
     frame semantics + the D1/D3 decl-set mirrors + prelude + the literal
     bare-name accepts, with BARE_BUILTINS re-extracted from resolve.rs and
     STD_SURFACE_INTRINSICS re-extracted from type_checker/mod.rs on every
     run, and ZERO position/shape guards (it never replicates tc_uc_shape's
     token heuristics; it presumes the fixture's declared callee position
     and computes only resolvability).
  3. The LIVE `mindc check` oracle: E2003 present at exactly the query
     line:col <=> verdict 1, checked for EVERY case including never/decline.

Per-case modes (position-classification claims are validated ONLY by live):
  classify — MIND must answer 0/1 and got == leg2 == live.
  kwfire   — folded-keyword callees (`use(1)` etc.): the leg-2 D2 mirror
             structurally cannot classify positions at tokens its tokenizer
             folds as keywords, so leg 2 is DROPPED for exactly these cases
             and LIVE is the sole oracle (Rule 3a's drop-leg-2 branch):
             got == 1 AND live-E2003 == 1.
  never    — structural never-fire shapes (qualified `Head::…(`, dotted
             `head.rest(` — resolve.rs's own verdict): got == 0 AND
             live-E2003 == 0 (leg 2 not consulted: the folded name is out of
             the bare-name union's domain).
  decline  — out-of-domain positions (decl names, method names, keyword
             statement occurrences, literal heads, non-call idents,
             top-level calls, BOUND-keyword callees — live E2012's domain):
             got == -3 AND live-E2003 == 0. Declining where live FIRES is
             the fail-open bug this mode catches — RED.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_undeclared_assign_smoke.py (E2009); the leg-2 D1/D2/
D3 Python mirrors are imported from the E2002 smoke unchanged and the
BARE_BUILTINS / intrinsics extractors from the E2012 smoke unchanged (all
already independent of every MIND-side classifier).
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")
REPO = os.path.dirname(os.path.dirname(HERE))
RESOLVE_RS = os.path.join(REPO, "src", "type_checker", "resolve.rs")
STD_DIR = os.path.join(REPO, "std")

sys.path.insert(0, HERE)
import self_host_tc_unknown_ident_smoke as e2002  # noqa: E402
import self_host_tc_fn_value_call_smoke as e2012  # noqa: E402

PRELUDE = e2002.PRELUDE
collect_decl_names = e2002.collect_decl_names
scope_verdict = e2002.scope_verdict
bundled_modules = e2002.bundled_modules
bare_builtins = e2012.bare_builtins
intrinsics_table = e2012.intrinsics_table


# ── source-of-truth guard: the Call arm must keep the grounded else-if
# chain (E2008 → E2012 → E2003 via !call_resolvable) and call_resolvable
# must keep its union arms — any drift = this smoke is stale ────────────────
def call_arm_source_guard():
    with open(RESOLVE_RS) as f:
        rs = f.read()
    m = re.search(
        r"Node::Call \{ callee, args, span \} => \{(.*?)\n            \}",
        rs,
        re.S,
    )
    if not m:
        print("FAIL: Node::Call arm not found in resolve.rs — drifted")
        sys.exit(1)
    arm = m.group(1)
    for needle in (
        "self.unknown_variant(callee)",
        "self.is_fn_value_call(callee)",
        "!self.call_resolvable(callee)",
    ):
        if needle not in arm:
            print(f"FAIL: Call-arm chain lost `{needle}` — re-ground the port")
            sys.exit(1)
    i_var = arm.index("self.unknown_variant(callee)")
    i_fv = arm.index("self.is_fn_value_call(callee)")
    i_cr = arm.index("!self.call_resolvable(callee)")
    if not i_var < i_fv < i_cr:
        print("FAIL: Call-arm precedence reordered — re-ground the port")
        sys.exit(1)
    m = re.search(r"fn call_resolvable\(&self, name: &str\) -> bool \{(.*?)\n    \}", rs, re.S)
    if not m:
        print("FAIL: call_resolvable not found in resolve.rs — drifted")
        sys.exit(1)
    body = m.group(1)
    for needle in (
        "self.ident_resolvable(name)",
        'name.contains("::")',
        'name.starts_with("__mind_")',
        'name.starts_with("tensor.")',
        'name == "gen_deref"',
        "BARE_BUILTINS.binary_search(&name)",
        "std_surface_intrinsic_arity(name)",
        "cm_lookup_fn(name)",
    ):
        if needle not in body:
            print(f"FAIL: call_resolvable lost `{needle}` — re-ground the port")
            sys.exit(1)


# ── fixtures ────────────────────────────────────────────────────────────────
S_UNDEF = """\
fn main() -> i64 {
    zork(1)
    return 0
}
"""

S_UNDEF_EXPR = """\
fn main() -> i64 {
    let v = zork(1)
    return 0
}
"""

S_UNDEF_NESTED = """\
fn main() -> i64 {
    if 1 == 1 {
        zork(1)
    }
    return 0
}
"""

S_UNDEF_COND = """\
fn main() -> i64 {
    if zork(1) == 1 {
        return 1
    }
    return 0
}
"""

S_UNDEF_WHILE = """\
fn main() -> i64 {
    while zork(1) == 1 {
        return 1
    }
    return 0
}
"""

S_UNDEF_ARG = """\
fn foo(a: i64) -> i64 { return a }
fn main() -> i64 {
    return foo(zork(1))
}
"""

S_SPACED = """\
fn main() -> i64 {
    zork (1)
    return 0
}
"""

S_ARM_BODY = """\
fn main() -> i64 {
    let v = 1
    match v {
        1 => zork(1),
        _ => 0,
    }
    return 0
}
"""

S_CLOSED = """\
fn main() -> i64 {
    if 1 == 1 {
        let g = 1
    }
    g(2)
    return 0
}
"""

S_TYPO = """\
fn main() -> i64 {
    let v = vec_neww()
    return 0
}
"""

S_KW_RETURN = """\
fn main() -> i64 {
    let z = return(1)
    return 0
}
"""

S_KW_LOOP = """\
fn main() -> i64 {
    let z = loop(1)
    return 0
}
"""

S_KW_BREAK = """\
fn main() -> i64 {
    let z = break(1)
    return 0
}
"""

S_KW_IN = """\
fn main() -> i64 {
    let z = in(1)
    return 0
}
"""

S_KW_AS = """\
fn main() -> i64 {
    let z = as(1)
    return 0
}
"""

S_KW_MUT = """\
fn main() -> i64 {
    let z = mut(1)
    return 0
}
"""

S_IMPORT_STD = """\
import std.vec
fn main() -> i64 {
    zork(1)
    return 0
}
"""

S_IMPORT_NONSTD = """\
import mylib.stuff
fn main() -> i64 {
    zork(1)
    return 0
}
"""

S_LOCALFN = """\
fn foo() -> i64 { return 1 }
fn main() -> i64 {
    return foo()
}
"""

S_LATEFN = """\
fn main() -> i64 {
    return late_fn()
}
fn late_fn() -> i64 { return 7 }
"""

S_STD = """\
fn main() -> i64 {
    let v = vec_new()
    return 0
}
"""

S_BUILTIN = """\
fn main() -> i64 {
    let x = sqrt(4.0)
    return 0
}
"""

S_PRELUDE_C = """\
fn main() -> i64 {
    let s = Some(5)
    return 0
}
"""

S_MINDALLOC = """\
fn main() -> i64 {
    let p = __mind_alloc(8)
    return 0
}
"""

S_MINDTYPO = """\
fn main() -> i64 {
    let p = __mind_allocc(8)
    return 0
}
"""

S_BYTE = """\
fn main() -> i64 {
    let b = byte(65)
    return 0
}
"""

S_BYTES = """\
fn main() -> i64 {
    let b = bytes(4)
    return 0
}
"""

S_GENDEREF = """\
fn main() -> i64 {
    let g = gen_deref(0)
    if g == 0 {
        return 0
    }
    return 1
}
"""

S_FNVALUE = """\
fn main() -> i64 {
    let f = 1
    f(2)
    return 0
}
"""

S_FNVALUE_PARAM = """\
fn apply(h: i64) -> i64 {
    return h(1)
}
fn main() -> i64 {
    return apply(3)
}
"""

S_SHADOW_BUILTIN = """\
fn main() -> i64 {
    let sqrt = 1
    let y = sqrt(4.0)
    return 0
}
"""

S_STRUCT_CTOR = """\
struct P {
    x: i64,
}
fn main() -> i64 {
    let p = P(1)
    return 0
}
"""

S_EXTERN = """\
extern "C" {
    fn getpid() -> i64
}
fn main() -> i64 {
    return getpid()
}
"""

S_QUAL = """\
fn main() -> i64 {
    let v = Foo::bar()
    return 0
}
"""

S_ENUM_BADVAR = """\
enum Mode {
    On(i64),
    Off,
}
fn main() -> i64 {
    let m = Mode::Zap(1)
    return 0
}
"""

S_DOTTED = """\
fn main() -> i64 {
    undefmod.undefn(1)
    return 0
}
"""

S_TENSOR_BAD = """\
fn main() -> i64 {
    let t = tensor.zork(1)
    return 0
}
"""

S_TENSOR_OK = """\
fn main() -> i64 {
    let t = tensor.zeros(f32, (2, 2))
    return 0
}
"""

S_METHOD = """\
fn main() -> i64 {
    let x = 5
    x.zork(1)
    return 0
}
"""

S_FNDECL = """\
fn zork(a: i64) -> i64 { return a }
fn main() -> i64 {
    return 0
}
"""

S_WHILE_HDR = """\
fn main() -> i64 {
    let mut i = 0
    while (i < 2) {
        i = i + 1
    }
    return i
}
"""

S_MATCH_HDR = """\
fn main() -> i64 {
    let v = 1
    match (v) {
        1 => 2,
        _ => 0,
    }
    return 0
}
"""

S_RETURN_HDR = """\
fn main() -> i64 {
    return (3)
}
"""

S_TRUE_CALL = """\
fn main() -> i64 {
    let z = true(1)
    return 0
}
"""

S_NONCALL = """\
fn main() -> i64 {
    let a = 1
    let b = zz + 1
    return b
}
"""

S_TOPLEVEL = """\
let g = zork(1)
fn main() -> i64 {
    return 0
}
"""

# ── folded-keyword callees (the fail-OPEN class round 2 closed): the
# SELF-HOST lexer folds fn/if/else/let/use/pub/import to tk_kw_* kinds, but
# live mindc lexes them as plain idents and fires E2003 on a call in
# expression / arm-body position (live-probed truth table 2026-07-23) ───────
S_KW_IMPORT = """\
fn main() -> i64 {
    let z = import(1)
    return 0
}
"""

S_KW_USE = """\
fn main() -> i64 {
    let z = use(1)
    return 0
}
"""

S_KW_PUB = """\
fn main() -> i64 {
    let z = pub(1)
    return 0
}
"""

S_KW_ELSE = """\
fn main() -> i64 {
    let z = else(1)
    return 0
}
"""

S_KW_FN = """\
fn main() -> i64 {
    let z = fn(1)
    return 0
}
"""

S_KW_LET = """\
fn main() -> i64 {
    let z = let(1)
    return 0
}
"""

S_KW_RET_USE = """\
fn main() -> i64 {
    return use(1)
}
"""

S_KW_ARM_USE = """\
fn main() -> i64 {
    let v = 1
    match v {
        1 => use(1),
        _ => 0,
    }
    return 0
}
"""

S_KW_ELSE_STMT = """\
fn main() -> i64 {
    else(1)
    return 0
}
"""

S_KW_ELSE_WHILE = """\
fn main() -> i64 {
    while 1 == 2 {
        return 1
    }
    else(1)
    return 0
}
"""

S_KW_USE_IMPORT_STMT = """\
use std.vec
fn main() -> i64 {
    return 0
}
"""

S_KW_PUB_STMT = """\
pub fn helper() -> i64 { return 1 }
fn main() -> i64 {
    return helper()
}
"""

S_KW_ELSE_CONSUMED = """\
fn main() -> i64 {
    if 1 == 1 { return 1 } else (2)
    return 0
}
"""

S_KW_ELSE_CONSUMED_NL = """\
fn main() -> i64 {
    if 1 == 2 {
        return 1
    }
    else(1)
    return 0
}
"""

S_KW_FN_STMT = """\
fn main() -> i64 {
    fn(1)
    return 0
}
"""

S_KW_USE_STMT = """\
fn main() -> i64 {
    use(1)
    return 0
}
"""

S_KW_IF_CALL = """\
fn main() -> i64 {
    let z = if(1)
    return 0
}
"""

S_KW_USE_TOPLEVEL = """\
let g = use(1)
fn main() -> i64 {
    return 0
}
"""

S_KW_BOUND = """\
fn main() -> i64 {
    let use = 5
    let z = use(1)
    return 0
}
"""

S_KW_BOUND_MUT = """\
fn main() -> i64 {
    let mut use = 5
    let z = use(1)
    return 0
}
"""

S_KW_CLOSED = """\
fn main() -> i64 {
    if 1 == 1 {
        let use = 5
    }
    let z = use(1)
    return 0
}
"""

CASES = [
    # classify positives — E2003 fires; got == leg2 == live == 1:
    ("classify", S_UNDEF, "zork", 0, "undefined callee, stmt pos -> E2003"),
    ("classify", S_UNDEF_EXPR, "zork", 0, "undefined callee, let RHS -> E2003"),
    ("classify", S_UNDEF_NESTED, "zork", 0, "undefined callee, nested block -> E2003"),
    ("classify", S_UNDEF_COND, "zork", 0, "undefined callee, if-cond -> E2003"),
    ("classify", S_UNDEF_WHILE, "zork", 0, "undefined callee, while-cond -> E2003"),
    ("classify", S_UNDEF_ARG, "zork", 0, "undefined callee, arg position -> E2003"),
    ("classify", S_SPACED, "zork", 0, "spaced `zork (1)` is still a call -> E2003"),
    ("classify", S_ARM_BODY, "zork", 0, "match-arm body callee -> E2003"),
    ("classify", S_CLOSED, "g(2", 0, "closed inner-scope binding -> E2003"),
    ("classify", S_TYPO, "vec_neww", 0, "near-std typo -> E2003"),
    ("classify", S_KW_RETURN, "return(1", 0, "`return(1)` expr pos -> E2003"),
    ("classify", S_KW_LOOP, "loop(1", 0, "`loop(1)` expr pos -> E2003"),
    ("classify", S_KW_BREAK, "break(1", 0, "`break(1)` expr pos -> E2003"),
    ("classify", S_KW_IN, "in(1", 0, "`in(1)` expr pos -> E2003"),
    ("classify", S_KW_AS, "as(1", 0, "`as(1)` expr pos -> E2003"),
    ("classify", S_KW_MUT, "mut(1", 0, "`mut(1)` expr pos -> E2003"),
    ("classify", S_IMPORT_STD, "zork", 0,
     "std import does NOT suppress -> E2003 (stale-doc pin)"),
    ("classify", S_IMPORT_NONSTD, "zork", 0,
     "unresolved non-std import does NOT suppress -> E2003 (stale-doc pin)"),
    # classify negatives — the callee resolves; got == leg2 == live == 0:
    ("classify", S_LOCALFN, "foo()", 1, "module fn callee -> no (D1)"),
    ("classify", S_LATEFN, "late_fn()", 0, "call BEFORE fn decl -> no (D1)"),
    ("classify", S_STD, "vec_new", 0, "std export callee -> no (D3)"),
    ("classify", S_BUILTIN, "sqrt", 0, "BARE_BUILTINS callee -> no"),
    ("classify", S_PRELUDE_C, "Some", 0, "prelude constructor -> no (D1)"),
    ("classify", S_MINDALLOC, "__mind_alloc", 0, "__mind_ prefix -> no"),
    ("classify", S_MINDTYPO, "__mind_allocc", 0,
     "misspelled __mind_ still blanket-accepted (E2024 warns) -> no"),
    ("classify", S_BYTE, "byte(65", 0, "E2024 intrinsic table (`byte`) -> no"),
    ("classify", S_BYTES, "bytes(4", 0, "`bytes` value-type callee -> no"),
    ("classify", S_GENDEREF, "gen_deref", 0, "`gen_deref` builtin -> no"),
    ("classify", S_FNVALUE, "f(2", 0,
     "local VALUE binding called -> E2012 domain, NOT E2003 (boundary)"),
    ("classify", S_FNVALUE_PARAM, "h(1", 0,
     "param called -> E2012 domain, NOT E2003 (boundary)"),
    ("classify", S_SHADOW_BUILTIN, "sqrt(4.0", 0, "local shadows builtin -> no (D2)"),
    ("classify", S_STRUCT_CTOR, "P(1", 0, "struct decl-name callee -> no (D1)"),
    ("classify", S_EXTERN, "getpid()", 1, "extern-block fn callee -> no (D1)"),
    # never — structural never-fire shapes; got == 0 AND live-E2003 == 0:
    ("never", S_QUAL, "Foo", 0, "qualified `Foo::bar()` head (`::` accept)"),
    ("never", S_ENUM_BADVAR, "Mode::Zap", 0,
     "variant-typo head fires E2008, never E2003 (precedence)"),
    ("never", S_DOTTED, "undefmod", 0,
     "dotted head is a MethodCall receiver (E2002 domain)"),
    ("never", S_TENSOR_BAD, "tensor", 0, "tensor.* folded callee (blanket accept)"),
    ("never", S_TENSOR_OK, "tensor", 0, "tensor.zeros head (blanket accept)"),
    # kwfire — FOLDED-KEYWORD callees (round-2 class): the self-host lexer
    # emits these as tk_kw_* (non-ident) tokens, live FIRES E2003. LIVE is
    # the sole oracle here (Rule 3a's drop-leg-2 branch): the leg-2 D2
    # mirror structurally cannot classify positions at tokens its tokenizer
    # folds as keywords, so got == live == 1 is the contract:
    ("kwfire", S_KW_IMPORT, "import(1", 0, "`import(1)` expr pos -> E2003 (folded tk_kw_use)"),
    ("kwfire", S_KW_USE, "use(1", 0, "`use(1)` expr pos -> E2003 (folded tk_kw_use)"),
    ("kwfire", S_KW_PUB, "pub(1", 0, "`pub(1)` expr pos -> E2003 (folded tk_kw_pub)"),
    ("kwfire", S_KW_ELSE, "else(1", 0, "`else(1)` expr pos -> E2003 (folded tk_kw_else)"),
    ("kwfire", S_KW_FN, "fn(1", 0, "`fn(1)` expr pos -> E2003 (folded tk_kw_fn)"),
    ("kwfire", S_KW_LET, "let(1", 0, "`let(1)` expr pos -> E2003 (folded tk_kw_let)"),
    ("kwfire", S_KW_RET_USE, "use(1", 0, "`return use(1)` -> E2003 (folded, return-atom)"),
    ("kwfire", S_KW_ARM_USE, "use(1", 0, "arm-body `=> use(1)` -> E2003 (folded)"),
    ("kwfire", S_KW_ELSE_STMT, "else(1", 0,
     "bare stmt-start `else(1)` -> E2003 (else with no if-arm is not consumed)"),
    ("kwfire", S_KW_ELSE_WHILE, "else(1", 0,
     "`else(1)` after a WHILE block -> E2003 (non-if `}` does not consume)"),
    ("kwfire", S_KW_CLOSED, "use(1", 0,
     "closed-scope keyword binding -> E2003 (the binding popped)"),
    # decline — folded-keyword statement occurrences (live-probed no-fire):
    ("decline", S_KW_USE_IMPORT_STMT, "use std", 0, "`use std.vec` import stmt"),
    ("decline", S_KW_PUB_STMT, "pub fn", 0, "`pub fn` visibility keyword"),
    ("decline", S_KW_ELSE_CONSUMED, "else (2", 0,
     "consumed `} else (…)` of an if-arm (if-chain parser swallows it)"),
    ("decline", S_KW_ELSE_CONSUMED_NL, "else(1", 0,
     "newline-separated `}\\nelse(` after an if-arm is STILL consumed"),
    ("decline", S_KW_FN_STMT, "fn(1", 0, "stmt-start `fn(1)` (decl-parse consumed)"),
    ("decline", S_KW_USE_STMT, "use(1", 0, "stmt-start `use(1)` (import-parse consumed)"),
    ("decline", S_KW_IF_CALL, "if(1", 0, "`if(1)` never fires (if-expression header)"),
    ("decline", S_KW_USE_TOPLEVEL, "use(1", 0, "top-level `use(1)` (E2001 domain)"),
    ("decline", S_KW_BOUND, "use(1", 0,
     "BOUND keyword callee (`let use = 5`) -> live E2012, not E2003"),
    ("decline", S_KW_BOUND_MUT, "use(1", 0,
     "mut-BOUND keyword callee -> live E2012, not E2003"),
    # decline — out of E2003's domain; got == -3 AND live-E2003 == 0:
    ("decline", S_METHOD, "zork", 0, "method NAME after `.`"),
    ("decline", S_FNDECL, "zork", 0, "fn DECLARATION name before `(`"),
    ("decline", S_WHILE_HDR, "while (", 0, "`while (` header keyword"),
    ("decline", S_MATCH_HDR, "match (", 0, "`match (` header keyword"),
    ("decline", S_RETURN_HDR, "return (", 0, "`return (3)` stmt keyword"),
    ("decline", S_TRUE_CALL, "true(1", 0, "bool-literal head (parse shape)"),
    ("decline", S_NONCALL, "zz", 0, "non-callee ident (E2002 domain)"),
    ("decline", S_TOPLEVEL, "zork", 0, "module-level call (E2001 domain)"),
]


# ── leg 2: the E2003 RESOLUTION UNION, recomputed independently ─────────────
# ZERO position/shape guards here — this leg NEVER replicates tc_uc_shape's
# classifier. It presumes the fixture's queried position is a bare-callee
# (the "classify"-mode contract, validated against LIVE) and recomputes only
# call_resolvable's bare-name union: D2 scope frames + D1 decls/prelude + D3
# std exports + "bytes"/"gen_deref" + the "__mind_" prefix + BARE_BUILTINS
# (re-extracted from resolve.rs) + STD_SURFACE_INTRINSICS (re-extracted from
# type_checker/mod.rs). "::"/"tensor." are dotted/qualified SHAPES — out of
# the bare-name domain, handled by the never-mode cases.
def call_resolution_verdict(src, pos, std_set, bare, intrin):
    name_m = re.match(r"[A-Za-z_]\w*", src[pos:])
    if not name_m:
        return -99
    name = name_m.group(0)
    sv = scope_verdict(src, pos)
    if sv == 1:
        return 0
    if sv != 0:
        return -99  # the D2 mirror could not reach the position — fail loud
    if name in PRELUDE:
        return 0
    if name in collect_decl_names(src):
        return 0
    if name in std_set:
        return 0
    if name in ("bytes", "gen_deref"):
        return 0
    if name.startswith("__mind_"):
        return 0
    if name in bare:
        return 0
    if name in intrin:
        return 0
    return 1


# ── leg 3: the live mindc oracle (position-matched E2003 presence) ──────────
DIAG_RE = re.compile(
    r":(\d+):(\d+): error: ([^\n]*?)\[type_check::(E2001|E2002|E2003|E2008|E2012)\]"
)
NAME_RE = re.compile(r"`([^`]+)`")


def live_diags(mindc, src, workdir, idx):
    path = os.path.join(workdir, f"case_{idx}.mind")
    with open(path, "w") as f:
        f.write(src)
    r = subprocess.run([mindc, "check", path], capture_output=True, text=True)
    out = []
    for m in DIAG_RE.finditer(r.stdout + r.stderr):
        nm = NAME_RE.search(m.group(3))
        out.append((int(m.group(1)), int(m.group(2)), m.group(4),
                    nm.group(1) if nm else None))
    return out


def live_codes_at(mindc, src, pos, workdir, idx):
    line = src.count("\n", 0, pos) + 1
    col = pos - (src.rfind("\n", 0, pos) + 1) + 1
    return {c for (ln, co, c, _) in live_diags(mindc, src, workdir, idx)
            if ln == line and co == col}


def live_verdict(mindc, src, pos, workdir, idx):
    line = src.count("\n", 0, pos) + 1
    col = pos - (src.rfind("\n", 0, pos) + 1) + 1
    name_m = re.match(r"[A-Za-z_]\w*", src[pos:])
    name = name_m.group(0) if name_m else None
    for ln, co, code, nm in live_diags(mindc, src, workdir, idx):
        if code != "E2003" or ln != line:
            continue
        if co == col:
            return 1
        if name is not None and nm == name:
            return 1
    return 0


def pos_of(src, needle, occ):
    hits = [m.start() for m in re.finditer(re.escape(needle), src)]
    return hits[occ]


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so, False
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    cmd = [mindc, MAIN_MIND, "--emit-shared", out]
    print("BUILD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        sys.exit(1)
    return out, True


def main():
    so, built = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        sys.exit(1)
    lib = ctypes.CDLL(so)
    fn = lib.selftest_tc_unknown_call
    fn.argtypes = [ctypes.c_int64] * 5
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")

    call_arm_source_guard()
    bare = bare_builtins()
    intrin = intrinsics_table()
    mods = bundled_modules()
    concat = b""
    std_set = set()
    for m in mods:
        path = os.path.join(STD_DIR, m + ".mind")
        with open(path, "rb") as f:
            s = f.read()
        if not s.endswith(b"\n"):
            s += b"\n"
        concat += s
        std_set |= collect_decl_names(s.decode())
    print(f"tables: Call-arm chain verified, {len(bare)} bare builtins, "
          f"{len(intrin)} intrinsics, {len(mods)} std modules "
          f"({len(concat)} bytes, {len(std_set)} exports)")

    sb = ctypes.create_string_buffer(concat, len(concat))
    sp = ctypes.cast(sb, ctypes.c_void_p).value

    def mind_verdict(src, pos):
        b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
        return fn(ctypes.cast(b, ctypes.c_void_p).value,
                  len(src.encode()), pos, sp, len(concat))

    total = fails = positives = negatives = live_checked = 0
    with tempfile.TemporaryDirectory() as workdir:
        # Live sentinels: the diagnostic surface this rule's boundaries rest
        # on must hold with position matching intact —
        #   (1) E2003 at the undefined callee;
        #   (2) a local-value callee fires E2012 and NOT E2003 (the sibling
        #       boundary this port must never cross);
        #   (3) a local-enum variant-typo callee fires E2008 and NOT E2003
        #       (the precedence arm);
        #   (4) an unresolvable non-std import does NOT suppress E2003 (the
        #       stale doc-comment pin, live-grounded).
        if live_verdict(mindc, S_UNDEF, pos_of(S_UNDEF, "zork", 0),
                        workdir, "s_undef") != 1:
            print("FAIL: live sentinel — E2003 not reported at the "
                  "undefined callee")
            sys.exit(1)
        f_codes = live_codes_at(mindc, S_FNVALUE, pos_of(S_FNVALUE, "f(2", 0),
                                workdir, "s_fv")
        if f_codes != {"E2012"}:
            print(f"FAIL: live sentinel — fn-value callee reported "
                  f"{sorted(f_codes)}, expected exactly E2012")
            sys.exit(1)
        v_codes = live_codes_at(mindc, S_ENUM_BADVAR,
                                pos_of(S_ENUM_BADVAR, "Mode::Zap", 0),
                                workdir, "s_var")
        if v_codes != {"E2008"}:
            print(f"FAIL: live sentinel — variant-typo callee reported "
                  f"{sorted(v_codes)}, expected exactly E2008")
            sys.exit(1)
        if live_verdict(mindc, S_IMPORT_NONSTD,
                        pos_of(S_IMPORT_NONSTD, "zork", 0),
                        workdir, "s_imp") != 1:
            print("FAIL: live sentinel — E2003 suppressed under an "
                  "unresolved import (the stale doc-comment came TRUE; "
                  "re-ground the port)")
            sys.exit(1)
        print("live sentinels: E2003 callee + E2012-not-E2003 boundary + "
              "E2008-not-E2003 precedence + no-import-suppression (4/4)")
        # Folded-keyword E2012 boundary sentinel: live mindc CONTEXTUALLY
        # binds keyword names (`let use = 5` is legal — the folded words are
        # reserved only in the SELF-HOST lexer), and a bound-keyword callee
        # then fires E2012, never E2003. The port mirrors this through the
        # tc_sf_kw_bindable binder extension; pin the live surface.
        k_codes = live_codes_at(mindc, S_KW_BOUND,
                                pos_of(S_KW_BOUND, "use(1", 0),
                                workdir, "s_kwb")
        if k_codes != {"E2012"}:
            print(f"FAIL: live sentinel — bound-keyword callee reported "
                  f"{sorted(k_codes)}, expected exactly E2012")
            sys.exit(1)
        print("live sentinel: bound-keyword callee fires E2012 (1/1)")

        for idx, (mode, src, needle, occ, note) in enumerate(CASES):
            pos = pos_of(src, needle, occ)
            got = mind_verdict(src, pos)
            live = live_verdict(mindc, src, pos, workdir, idx)
            live_checked += 1
            if mode == "classify":
                exp = call_resolution_verdict(src, pos, std_set, bare, intrin)
                ok = got == exp == live
                exp_s = str(exp)
            elif mode == "kwfire":
                # Folded-keyword callee: the leg-2 D2 mirror cannot reach a
                # keyword-token position (Rule 3a drop-leg-2 branch) — LIVE
                # is the sole, non-fakeable oracle and must FIRE.
                exp = 1
                ok = got == 1 and live == 1
                exp_s = "k1"
            elif mode == "never":
                exp = 0
                ok = got == 0 and live == 0
                exp_s = "n0"
            else:  # decline — allowed ONLY where live-E2003 does not fire
                exp = -3
                ok = got == -3 and live == 0
                exp_s = "d-3"
            total += 1
            if (mode == "classify" and exp == 1) or mode == "kwfire":
                positives += 1
            elif mode == "classify" and exp == 0:
                negatives += 1
            if not ok:
                fails += 1
            mark = "ok " if ok else "DIFF"
            print(f"  {mark} got={got} rule={exp_s} live={live} pos={pos} "
                  f"[{mode}] {note}")

        # ── FULL keyword × callee-position MATRIX (round-3 class closure) ──
        # Every keyword the self-host lexer can see — the Rust parser's
        # 23-word stmt_keyword table (parser/mod.rs:198) + the context words
        # in/as/mut + match + the bool literals — in BOTH expression and
        # statement callee position, LIVE-keyed: live fires → the port MUST
        # answer 1; live quiet → the port MUST NOT answer 1 (0 / -3 both
        # sound declines). Any future keyword divergence in either direction
        # goes RED here. (Live 2026-07-23 ground truth: EVERY keyword fires
        # in expr position except if/region/match/true/false; ONLY in/as/mut
        # fire in stmt position.)
        KW_MATRIX = ["fn", "if", "for", "let", "pub", "use", "enum", "loop",
                     "type", "break", "const", "print", "while", "assert",
                     "export", "extern", "import", "module", "region",
                     "return", "struct", "continue", "invariant",
                     "in", "as", "mut", "match", "true", "false"]
        m_fired = m_quiet = 0
        for kw in KW_MATRIX:
            for shape in ("expr", "stmt"):
                if shape == "expr":
                    msrc = ("fn m() -> i64 {\n    let z = " + kw +
                            "(1)\n    return 0\n}\n")
                else:
                    msrc = ("fn m() -> i64 {\n    " + kw +
                            "(1)\n    return 0\n}\n")
                mpos = msrc.index(kw + "(1")
                lv = live_verdict(mindc, msrc, mpos, workdir,
                                  f"kw_{kw}_{shape}")
                pv = mind_verdict(msrc, mpos)
                total += 1
                live_checked += 1
                bad = (lv == 1 and pv != 1) or (lv == 0 and pv == 1)
                if lv == 1:
                    m_fired += 1
                else:
                    m_quiet += 1
                if bad:
                    fails += 1
                    print(f"  DIFF kw-matrix `{kw}` [{shape}] port={pv} "
                          f"live={lv}")
        print(f"kw-matrix: {len(KW_MATRIX)}x2 cells, fired={m_fired} "
              f"quiet={m_quiet}, divergences included in fails")
        if m_fired < 20:
            print("FAIL: kw-matrix vacuous (live fired < 20 cells)")
            sys.exit(1)

    print(f"unknown_call: cases={total} positives={positives} "
          f"negatives={negatives} live_checked={live_checked} fails={fails}")
    if positives < 25 or negatives < 12 or live_checked < 60:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2003 rule diverges from the Rust rule")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
