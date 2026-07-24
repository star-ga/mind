#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND B1 E2009 rule — undeclared assign.

Ports resolve.rs's Node::Assign arm (UNDECLARED_ASSIGN_CODE, E2009,
resolve.rs:860-882): an assignment statement `name = expr` (including the
compound `name OP= expr` desugar, parser/mod.rs:1360-1439 — same bare-ident
LHS shape) whose l-value NAME is bound NOWHERE. The exact Rust condition:

    E2009 fires iff  !scopes.contains(name)        [D2 — live local binding]
                  && !syms.name_resolvable(name)   [D1 module decls + prelude
                                                    OR D3 bundled-std exports]

CRUCIAL grounded differences from E2002's ident_resolvable (live-verified,
probe battery 2026-07-23): the E2009 union has NO "bytes" accept
(`bytes = 5` FIRES), NO "::" auto-accept (a qualified target resolves ONLY
through the decl set's local `Enum::Variant` insertion or a qualified std
export — `Foo::Bar = 5`, `Mode::Zap = 5`, `A::B::C = 5` and even
`Option::Some = 5` all FIRE; prelude injects only the 6 BARE names), and no
cross-module arm (vacuous single-file). Field/index assignment targets are
SEPARATE AST nodes (FieldAssign / IndexAssign) whose receiver is walked as a
value USE — E2002 domain, never E2009. A compound `x += 1` desugars to
Assign(x, x+1): the cloned LHS fires E2002 AND the target fires E2009 at the
same position. `x == 1` is a comparison (one tk_eqeq token) — never Assign.
`mut` / `in` / `as` are NOT parser statement keywords: `in = 5` etc. parse
as a plain Assign and FIRE (they are exempt from the keyword decline).

The pure-MIND twin `selftest_tc_undeclared_assign(src, src_len, pos,
std_src, std_len)` takes the fixture source, the byte offset of the queried
ASSIGN-TARGET ident, and the CONCATENATED bundled-std source (the D3
caller-supplied model). Returns 1 (E2009 fires) / 0 (the target resolves) /
-3 (fail-closed sentinel: not an assign-target position — value use, binder,
member name, comparison, keyword occurrence, mid-expression shape).

Three-way agreement, machine-checked (no hand-authored verdict table), with
LIVE AUTHORITATIVE ON EVERY CASE and leg 2 fully independent of the port's
position classifier (Rule 3a):
  1. MIND core over (source, pos, std concat).
  2. The E2009 RESOLUTION UNION recomputed independently in Python — the D2
     frame semantics + the D1/D3 decl-set mirrors + prelude, with ZERO
     position/shape guards (it never replicates tc_ua_target_end's token
     heuristics; it presumes the fixture's declared assign-target position
     and computes only resolvability, with the E2009-specific union: no
     "bytes", no "::" auto-accept, qualified via decl-set insertion only).
  3. The LIVE `mindc check` oracle: E2009 present at exactly the query
     line:col <=> verdict 1, checked for EVERY case including declines.

Per-case modes (position-classification claims are validated ONLY by live):
  classify — MIND must answer 0/1 and got == leg2 == live.
  decline  — out-of-domain positions (value uses, binders, member names,
             comparisons, keyword occurrences, parse-error shapes): got ==
             -3 AND live-E2009 == 0. Declining where live FIRES is the
             fail-open bug this mode exists to catch — such a case is RED.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_unknown_ident_smoke.py (E2002); the leg-2a/2b D1/D3/
D2 Python mirrors are imported from it unchanged (they are already
independent of every MIND-side classifier).
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

PRELUDE = e2002.PRELUDE
collect_decl_names = e2002.collect_decl_names
scope_verdict = e2002.scope_verdict
bundled_modules = e2002.bundled_modules

# ── source-of-truth guard: the Assign arm's predicate must keep the exact
# two-term union (a third accept appearing there = this smoke is stale) ─────
def assign_arm_source_guard():
    with open(RESOLVE_RS) as f:
        rs = f.read()
    m = re.search(
        r"Node::Assign \{ name, value, span \} => \{(.*?)\n            \}",
        rs,
        re.S,
    )
    if not m:
        print("FAIL: Node::Assign arm not found in resolve.rs — drifted")
        sys.exit(1)
    arm = m.group(1)
    if "!self.scopes.contains(name) && !self.syms.name_resolvable(name)" not in arm:
        print("FAIL: Assign-arm predicate drifted from "
              "scopes.contains && name_resolvable — re-ground the port")
        sys.exit(1)
    if "undeclared_assign: true" not in arm:
        print("FAIL: Assign arm no longer selects undeclared_assign (E2009)")
        sys.exit(1)


# ── fixtures ────────────────────────────────────────────────────────────────
S_UNDEF = """\
fn main() -> i64 {
    conter = 1
    return 0
}
"""

S_BOUND = """\
fn main() -> i64 {
    let mut counter = 0
    counter = 1
    return counter
}
"""

S_PARAM = """\
fn set(x: i64) -> i64 {
    x = 1
    return x
}
fn main() -> i64 {
    return set(0)
}
"""

S_MODLET = """\
let mut g_state = 0
fn main() -> i64 {
    g_state = 5
    return 0
}
"""

S_LATE = """\
fn main() -> i64 {
    late = 1
    return 0
}
let mut late = 0
"""

S_FNNAME = """\
fn foo() -> i64 { return 1 }
fn main() -> i64 {
    foo = 5
    return 0
}
"""

S_CONST = """\
const LIM = 9
fn main() -> i64 {
    LIM = 7
    return 0
}
"""

S_STD = """\
fn main() -> i64 {
    vec_new = 5
    return 0
}
"""

S_PRELUDE_T = """\
fn main() -> i64 {
    Some = 5
    return 0
}
"""

S_BYTES = """\
fn main() -> i64 {
    bytes = 5
    return 0
}
"""

S_BEFORE = """\
fn main() -> i64 {
    x = 1
    let x = 2
    return x
}
"""

S_CLOSED = """\
fn main() -> i64 {
    if 1 == 1 {
        let y = 2
    }
    y = 3
    return 0
}
"""

S_COMPOUND = """\
fn main() -> i64 {
    kount += 1
    return 0
}
"""

S_COMPOUND_OK = """\
fn main() -> i64 {
    let mut k = 0
    k += 1
    return k
}
"""

S_SHL = """\
fn main() -> i64 {
    zz <<= 2
    return 0
}
"""

S_AMP_OK = """\
fn main() -> i64 {
    let mut b = 3
    b &= 1
    return b
}
"""

S_ARM = """\
fn main() -> i64 {
    let v = 1
    match v {
        n => zz = 5,
        _ => 0,
    }
    return 0
}
"""

S_ARM_BINDER = """\
fn main() -> i64 {
    let v = 1
    match v {
        n => n = 5,
        _ => 0,
    }
    return 0
}
"""

S_IN = """\
fn main() -> i64 {
    in = 5
    return 0
}
"""

S_AS = """\
fn main() -> i64 {
    as = 5
    return 0
}
"""

S_MUT = """\
fn main() -> i64 {
    mut = 5
    return 0
}
"""

S_SHADOW_STD = """\
fn main() -> i64 {
    let mut vec_new = 0
    vec_new = 5
    return vec_new
}
"""

S_ENUM_OK = """\
enum Mode {
    On,
    Off,
}
fn main() -> i64 {
    Mode::On = 5
    return 0
}
"""

S_ENUM_BADVAR = """\
enum Mode {
    On,
    Off,
}
fn main() -> i64 {
    Mode::Zap = 5
    return 0
}
"""

S_QUAL_UNDEF = """\
fn main() -> i64 {
    Foo::Bar = 5
    return 0
}
"""

S_QUAL_DEEP = """\
fn main() -> i64 {
    A::B::C = 5
    return 0
}
"""

S_QUAL_PRELUDE = """\
fn main() -> i64 {
    Option::Some = 5
    return 0
}
"""

S_FIELD = """\
fn main() -> i64 {
    s.field = 1
    return 0
}
"""

S_INDEX = """\
fn main() -> i64 {
    xs[0] = 1
    return 0
}
"""

S_EQEQ = """\
fn main() -> i64 {
    let q = 1
    if nope == 1 {
        return q
    }
    return 0
}
"""

S_LET_BINDER = """\
fn main() -> i64 {
    let w = 1
    return w
}
"""

S_MIDEXPR = """\
fn main() -> i64 {
    let a = 1
    a + zz = 5
    return 0
}
"""

S_NEWLINE = """\
fn main() -> i64 {
    zz
    = 5
    return 0
}
"""

S_KW_RETURN = """\
fn main() -> i64 {
    return = 5
    return 0
}
"""

S_KW_LOOP = """\
fn main() -> i64 {
    loop = 5
    return 0
}
"""

S_BOOL = """\
fn main() -> i64 {
    true = 5
    return 0
}
"""

S_PAT_BINDER = """\
fn main() -> i64 {
    let v = 1
    match v {
        n => 1,
        _ => 0,
    }
    return 0
}
"""

S_FIELD_OK = """\
struct P {
    x: i64,
}
fn main() -> i64 {
    let mut p = P { x: 1 }
    p.x = 5
    return p.x
}
"""

S_QUAL_SPACED = """\
fn main() -> i64 {
    Foo :: Bar = 5
    return 0
}
"""

CASES = [
    # classify — assign-target positions; got == leg2 == live on 0/1:
    ("classify", S_UNDEF, "conter", 0, "undefined target -> E2009"),
    ("classify", S_BOUND, "counter = 1", 0, "bound local target -> no (D2)"),
    ("classify", S_PARAM, "x = 1", 0, "param target -> no (D2)"),
    ("classify", S_MODLET, "g_state = 5", 0, "module-level let -> no (D1)"),
    ("classify", S_LATE, "late = 1", 0, "module let AFTER fn -> no (D1)"),
    ("classify", S_FNNAME, "foo = 5", 0, "fn name target -> no (D1)"),
    ("classify", S_CONST, "LIM = 7", 0, "const target -> no (D1)"),
    ("classify", S_STD, "vec_new = 5", 0, "std export target -> no (D3)"),
    ("classify", S_PRELUDE_T, "Some = 5", 0, "prelude target -> no (D1)"),
    ("classify", S_BYTES, "bytes", 0, "bytes target -> E2009 (NO bytes accept)"),
    ("classify", S_BEFORE, "x = 1", 0, "use-before-let target -> E2009"),
    ("classify", S_CLOSED, "y = 3", 0, "closed inner-scope binding -> E2009"),
    ("classify", S_COMPOUND, "kount", 0, "compound += undefined -> E2009"),
    ("classify", S_COMPOUND_OK, "k +=", 0, "compound += bound -> no (D2)"),
    ("classify", S_SHL, "zz", 0, "compound <<= undefined -> E2009"),
    ("classify", S_AMP_OK, "b &=", 0, "compound &= bound -> no (D2)"),
    ("classify", S_ARM, "zz", 0, "match-arm body assign -> E2009"),
    ("classify", S_ARM_BINDER, "n = 5", 0, "arm binder target -> no (D2)"),
    ("classify", S_IN, "in =", 0, "`in` target (NOT a stmt kw) -> E2009"),
    ("classify", S_AS, "as =", 0, "`as` target (NOT a stmt kw) -> E2009"),
    ("classify", S_MUT, "mut =", 0, "`mut` target (NOT a stmt kw) -> E2009"),
    ("classify", S_SHADOW_STD, "vec_new = 5", 0, "local shadows std -> no (D2)"),
    ("classify", S_ENUM_OK, "Mode::On =", 0, "declared Enum::Variant -> no"),
    ("classify", S_ENUM_BADVAR, "Mode::Zap", 0, "unknown variant target -> E2009"),
    ("classify", S_QUAL_UNDEF, "Foo::Bar", 0, "undeclared qualified -> E2009"),
    ("classify", S_QUAL_DEEP, "A::B::C", 0, "deep qualified -> E2009"),
    ("classify", S_QUAL_PRELUDE, "Option::Some", 0,
     "prelude is BARE-only -> E2009"),
    # decline — out of E2009's domain; got == -3 AND live-E2009 == 0:
    ("decline", S_FIELD, "field", 0, "FieldAssign member name"),
    ("decline", S_FIELD, "s.field", 0, "FieldAssign receiver (E2002 domain)"),
    ("decline", S_INDEX, "xs[0]", 0, "IndexAssign receiver (E2002 domain)"),
    ("decline", S_EQEQ, "nope", 0, "`==` comparison, not assign"),
    ("decline", S_LET_BINDER, "w = 1", 0, "let-binding name"),
    ("decline", S_BOUND, "counter = 0", 0, "let mut binding name"),
    ("decline", S_CONST, "LIM = 9", 0, "const decl name"),
    ("decline", S_MIDEXPR, "zz", 0, "mid-expression LHS (parse error)"),
    ("decline", S_NEWLINE, "zz", 0, "newline-split `=` (parse error)"),
    ("decline", S_KW_RETURN, "return =", 0, "`return` stmt keyword"),
    ("decline", S_KW_LOOP, "loop", 0, "`loop` stmt keyword"),
    ("decline", S_BOOL, "true", 0, "bool literal target (parse error)"),
    ("decline", S_PAT_BINDER, "n =>", 0, "match-pattern binder (`=>` tail)"),
    ("decline", S_FIELD_OK, "x = 5", 0, "field assign member via receiver"),
    ("decline", S_QUAL_SPACED, "Foo ::", 0, "spaced `::` (parse error)"),
]

# ── leg 2: the E2009 RESOLUTION UNION, recomputed independently ─────────────
# ZERO position/shape guards here — this leg NEVER replicates
# tc_ua_target_end's classifier. It presumes the fixture's queried position
# is an assign target (the "classify"-mode contract, validated against LIVE)
# and recomputes only the Assign arm's union: D2 scope frames + D1
# decls/prelude + D3 std exports. Deliberately ABSENT vs E2002: the "bytes"
# accept and the "::" auto-accept (a qualified name resolves only if the
# decl-set mirror inserted it — local `Enum::Variant` — or it is a qualified
# std export).
def resolution_verdict(src, pos, std_set):
    name_m = re.match(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)*", src[pos:])
    if not name_m:
        return -99
    name = name_m.group(0)
    if "::" in name:
        if name in collect_decl_names(src):
            return 0
        if name in std_set:
            return 0
        return 1
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
    return 1


# ── leg 3: the live mindc oracle (position-matched E2009 presence) ──────────
# NOTE 1: scan ALL diagnostics at the position — a compound `x += 1` reports
# E2002 AND E2009 at the SAME line:col, so first-match-wins would be wrong.
# NOTE 2: the Node::Assign SPAN starts at the enclosing STATEMENT start; in a
# match-arm body (`n => zz = 5`) that is the arm PATTERN's column, not the
# assign target's (live-verified: E2009 for `zz` reported at the `n` column).
# The verdict therefore matches an E2009 on the SAME LINE whose exact column
# OR whose quoted target name matches the query — still fully live-derived.
DIAG_RE = re.compile(
    r":(\d+):(\d+): error: ([^\n]*?)\[type_check::(E2002|E2003|E2008|E2009|E2012)\]"
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
    name_m = re.match(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)*", src[pos:])
    name = name_m.group(0) if name_m else None
    for ln, co, code, nm in live_diags(mindc, src, workdir, idx):
        if code != "E2009" or ln != line:
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
    fn = lib.selftest_tc_undeclared_assign
    fn.argtypes = [ctypes.c_int64] * 5
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")

    assign_arm_source_guard()
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
    print(f"tables: Assign-arm predicate verified, {len(mods)} std modules "
          f"({len(concat)} bytes, {len(std_set)} exports)")

    sb = ctypes.create_string_buffer(concat, len(concat))
    sp = ctypes.cast(sb, ctypes.c_void_p).value

    def mind_verdict(src, pos):
        b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
        return fn(ctypes.cast(b, ctypes.c_void_p).value,
                  len(src.encode()), pos, sp, len(concat))

    total = fails = positives = negatives = live_checked = 0
    with tempfile.TemporaryDirectory() as workdir:
        # Live sentinels: both directions must reach resolve with position
        # matching intact — an E2009 hit at the undefined target, and a
        # FieldAssign receiver that reports E2002 (NOT E2009): the
        # separate-node discipline this rule's domain boundary rests on.
        if live_verdict(mindc, S_UNDEF, pos_of(S_UNDEF, "conter", 0),
                        workdir, "s_in") != 1:
            print("FAIL: live sentinel — E2009 not reported at the "
                  "undefined assign target")
            sys.exit(1)
        s_codes = live_codes_at(mindc, S_FIELD, pos_of(S_FIELD, "s.field", 0),
                                workdir, "s_out")
        if s_codes != {"E2002"}:
            print(f"FAIL: live sentinel — FieldAssign receiver reported "
                  f"{sorted(s_codes)}, expected exactly E2002")
            sys.exit(1)
        # The compound both-fire sentinel: `x += 1` must report E2002 AND
        # E2009 at the same position (the desugar-clone, live-grounded).
        c_codes = live_codes_at(mindc, S_COMPOUND,
                                pos_of(S_COMPOUND, "kount", 0),
                                workdir, "s_comp")
        if c_codes != {"E2002", "E2009"}:
            print(f"FAIL: live sentinel — compound target reported "
                  f"{sorted(c_codes)}, expected E2002+E2009")
            sys.exit(1)
        print("live sentinels: E2009 target + E2002-only receiver + "
              "compound dual-fire (3/3)")

        for idx, (mode, src, needle, occ, note) in enumerate(CASES):
            pos = pos_of(src, needle, occ)
            got = mind_verdict(src, pos)
            live = live_verdict(mindc, src, pos, workdir, idx)
            live_checked += 1
            if mode == "classify":
                exp = resolution_verdict(src, pos, std_set)
                ok = got == exp == live
                exp_s = str(exp)
            else:  # decline — allowed ONLY where live-E2009 does not fire
                exp = -3
                ok = got == -3 and live == 0
                exp_s = "d-3"
            total += 1
            if mode == "classify" and exp == 1:
                positives += 1
            elif mode == "classify" and exp == 0:
                negatives += 1
            if not ok:
                fails += 1
            mark = "ok " if ok else "DIFF"
            print(f"  {mark} got={got} rule={exp_s} live={live} pos={pos} "
                  f"[{mode}] {note}")

    print(f"undeclared_assign: cases={total} positives={positives} "
          f"negatives={negatives} live_checked={live_checked} fails={fails}")
    if positives < 10 or negatives < 10 or live_checked < 40:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2009 rule diverges from the Rust rule")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
