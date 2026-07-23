#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND D1 module DECL-NAME SET.

Ports the resolve.rs `collect_decl_names` set (the scope-frame engine
foundation for the E2002/E2003/E2009/E2012 rule ports) plus the
`collect_module_syms` prelude injection (resolve.rs:344): the names a
module's TOP-LEVEL items declare — fn / struct / enum (name + per variant
BOTH the bare name AND the qualified `Enum::Variant` form) / let (incl.
`let mut`) / const / type / `extern const NAME` / `extern "C" { fn NAME }`
block fns — recursing into the Block a `module NAME { ... }` body unwraps
into (the module NAME itself is dropped by the parser, so it is NOT a
member), plus the hardwired prelude Result/Option/Ok/Err/Some/None. Names
inside fn bodies, struct fields, enum payload fields, params, strings and
comments are NOT members. Duplicate decls merge (BTreeSet insert).

The pure-MIND twin `selftest_tc_decl_names(src, src_len, name, name_len)`
takes the FULL module source buffer (it builds the set itself, via the real
self-host lexer) plus the queried name — so the set construction, not just
the final membership bit, is MIND-computed.

Three-way agreement, machine-checked (no hand table):
  1. MIND core over (source, name).
  2. The exact Rust rule recomputed in Python: a token-level mirror of
     collect_decl_names' AST arms (comments stripped, strings tokenized as
     units, non-decl braces opaque, module/extern blocks transparent, enum
     variant entry scan mirroring collect_enum_variants).
  3. The LIVE `mindc check` oracle where applicable (bare names off the std
     surface): the name is referenced in value position from a probe fn and
     E2002 (UNKNOWN_IDENT_CODE) absence == membership, because resolve's
     `name_resolvable` consults exactly this set (plus std exports).
     Fail-closed: a sentinel definitely-undefined probe MUST yield E2002 or
     the whole smoke fails (proves the live leg actually reaches resolve).

Also regenerates the prelude packed-word constants from the strings and
cross-checks the MIND packer against them.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_unknown_variant_smoke.py.
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")

PRELUDE = {"Result", "Option", "Ok", "Err", "Some", "None"}

SRC_A = """\
fn top_fn(par_a: i64) -> i64 {
    let inner_let = par_a;
    return inner_let;
}
pub fn pub_fn() -> i64 {
    return 1;
}
struct Pt { fx: i64, fy: i64 }
enum Col { Red, Green(i64), Blu { pw: i64 } }
enum Col { Extra }
let top_let = 3;
let mut top_mut = 4;
const TOP_CONST: i64 = 5;
type Alias = i64;
extern "C" { fn ext_read(p: i64) -> i64; fn ext_write(p: i64) -> i64; }
extern const LUT: [i64; 4]
module inner {
    fn mod_fn() -> i64 {
        return 2;
    }
    struct ModS { ma: i64 }
}
fn body_trap() -> i64 {
    let hidden_local = 1;
    // fn ghost_comment() -> i64 { return 9; }
    return hidden_local;
}
"""

SRC_B = """\
fn only_fn() -> i64 {
    let s_local = "fn ghost_str() -> i64 { enum GhostE { G } }";
    return 0;
}
let brace_trap = "{ fn ghost_brace() -> i64 { return 1; } }";
enum Late { V1 }
module deep.sub {
    fn dot_fn() -> i64 {
        return 7;
    }
}
"""

# (source, name, live) — live=True: the E2002 value-position probe verdict is
# well-defined (bare name, not a std-surface export). Expected membership is
# NEVER hand-authored: it comes from the recomputed Rust rule (leg 2) and must
# equal legs 1 and 3.
CASES = [
    # SRC_A positives
    (SRC_A, "top_fn", True),
    (SRC_A, "pub_fn", True),
    (SRC_A, "Pt", True),
    (SRC_A, "Col", True),
    (SRC_A, "Red", True),
    (SRC_A, "Green", True),
    (SRC_A, "Blu", True),
    (SRC_A, "Extra", True),
    (SRC_A, "Col::Red", False),
    (SRC_A, "Col::Extra", False),
    (SRC_A, "top_let", True),
    (SRC_A, "top_mut", True),
    (SRC_A, "TOP_CONST", True),
    (SRC_A, "Alias", True),
    (SRC_A, "ext_read", True),
    (SRC_A, "ext_write", True),
    (SRC_A, "LUT", True),
    (SRC_A, "mod_fn", True),
    (SRC_A, "ModS", True),
    # prelude
    (SRC_A, "Result", True),
    (SRC_A, "Option", True),
    (SRC_A, "Ok", True),
    (SRC_A, "Err", True),
    (SRC_A, "Some", True),
    (SRC_A, "None", True),
    # SRC_A negatives
    (SRC_A, "inner", True),        # module NAME is dropped by the parser
    (SRC_A, "inner_let", True),    # fn-body local
    (SRC_A, "hidden_local", True),
    (SRC_A, "par_a", True),        # param
    (SRC_A, "fx", True),           # struct field
    (SRC_A, "pw", True),           # enum payload field
    (SRC_A, "ghost_comment", True),
    (SRC_A, "Missing", True),
    (SRC_A, "Col::Blue", False),   # unknown variant, qualified
    (SRC_A, "Col::Red::Deep", False),  # deeper path never inserted
    (SRC_A, "Pt::fx", False),      # non-enum head never inserted
    # SRC_B
    (SRC_B, "only_fn", True),
    (SRC_B, "brace_trap", True),
    (SRC_B, "Late", True),
    (SRC_B, "V1", True),
    (SRC_B, "Late::V1", False),
    (SRC_B, "dot_fn", True),
    (SRC_B, "deep", True),         # dotted module path segment dropped
    (SRC_B, "sub", True),
    (SRC_B, "ghost_str", True),    # fn-name inside a body string
    (SRC_B, "GhostE", True),       # enum decl inside a string
    (SRC_B, "ghost_brace", True),  # fn-name inside a TOP-LEVEL string w/ braces
    (SRC_B, "s_local", True),
]

TOKEN_RE = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'|[A-Za-z_]\w*|\S')


def is_word(t):
    return re.fullmatch(r"[A-Za-z_]\w*", t) is not None


def tokenize(src):
    return TOKEN_RE.findall(re.sub(r"//[^\n]*", "", src))


def collect_decl_names(src):
    """Recompute resolve.rs collect_decl_names over the source: top-level decl
    arms only, module/extern-block recursion, enum variants bare + qualified,
    non-decl braces opaque."""
    toks = tokenize(src)
    n = len(toks)
    names = set()

    def skip_brace(j):  # j just after '{'
        d = 1
        while j < n and d > 0:
            if toks[j] == "{":
                d += 1
            elif toks[j] == "}":
                d -= 1
            j += 1
        return j

    i = 0
    while i < n:
        t = toks[i]
        if t == "fn" and i + 1 < n and is_word(toks[i + 1]):
            names.add(toks[i + 1])  # Node::FnDef (body brace opaque below)
            i += 2
            continue
        if t == "let":
            j = i + 1
            if j + 1 < n and toks[j] == "mut" and is_word(toks[j + 1]):
                j += 1
            if j < n and is_word(toks[j]):
                names.add(toks[j])  # Node::Let
                i = j + 1
                continue
            i += 1
            continue
        if t in ("struct", "const", "type") and i + 1 < n and is_word(toks[i + 1]):
            names.add(toks[i + 1])  # StructDef / Const / TypeAlias
            i += 2
            continue
        if t == "enum" and i + 2 < n and is_word(toks[i + 1]) and toks[i + 2] == "{":
            ename = toks[i + 1]
            names.add(ename)  # Node::EnumDef
            j, d, p, expect = i + 3, 1, 0, True
            while j < n and d > 0:
                c = toks[j]
                if c == "{":
                    d += 1
                elif c == "}":
                    d -= 1
                elif c == "(":
                    p += 1
                elif c == ")":
                    p -= 1
                elif d == 1 and p == 0:
                    if c == ",":
                        expect = True
                    elif expect and is_word(c):
                        names.add(c)  # bare variant name
                        names.add(f"{ename}::{c}")  # qualified form
                        expect = False
                j += 1
            i = j
            continue
        if t == "extern":
            if i + 2 < n and toks[i + 1] == "const" and is_word(toks[i + 2]):
                names.add(toks[i + 2])  # Node::ExternConst
                i += 3
                continue
            j = i + 1
            while j < n and toks[j] != "{":
                j += 1
            i = j + 1  # ExternBlock: interior fns counted by the fn arm
            continue
        if t == "module" and i + 1 < n and is_word(toks[i + 1]):
            j = i + 2
            while j + 1 < n and toks[j] == "." and is_word(toks[j + 1]):
                j += 2
            if j < n and toks[j] == "{":
                j += 1  # Block recursion: transparent, NAME dropped
            i = j
            continue
        if t == "{":
            i = skip_brace(i + 1)  # non-decl brace: opaque
            continue
        i += 1
    return names


def rust_member(src, name):
    if name in PRELUDE:
        return 1
    return 1 if name in collect_decl_names(src) else 0


def pack8(s):
    v = 0
    for i, c in enumerate(s.encode()):
        v |= c << (8 * i)
    return v


def live_member(mindc, src, name, workdir, idx):
    probe = src + (
        "\nfn __dn_probe() -> i64 {\n    let v = " + name + ";\n    return 0;\n}\n"
    )
    p = os.path.join(workdir, f"case_{idx}.mind")
    with open(p, "w") as f:
        f.write(probe)
    r = subprocess.run([mindc, "check", p], capture_output=True, text=True)
    return 0 if "E2002" in (r.stdout + r.stderr) else 1


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
    fn = lib.selftest_tc_decl_names
    fn.argtypes = [ctypes.c_int64] * 4
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")

    # Prelude packed-word constants: regenerate from the strings and require
    # them to match the values baked into tc_dn_is_prelude (via a membership
    # probe on an EMPTY module, where ONLY the prelude can answer 1).
    empty = ctypes.create_string_buffer(b"\n", 1)
    for s in sorted(PRELUDE):
        nb = ctypes.create_string_buffer(s.encode(), len(s))
        got = fn(ctypes.cast(empty, ctypes.c_void_p).value, 1,
                 ctypes.cast(nb, ctypes.c_void_p).value, len(s))
        if got != 1:
            print(f"FAIL: prelude {s!r} (pack {pack8(s)}) not a member of the "
                  f"empty module (got={got})")
            sys.exit(1)
    print(f"prelude: 6/6 members of the empty module (packed words verified)")

    total = fails = positives = negatives = live_checked = 0
    with tempfile.TemporaryDirectory() as workdir:
        # Fail-closed sentinel: the live leg must actually reach resolve.
        for si, base in ((0, SRC_A), (1, SRC_B)):
            if live_member(mindc, base, "__dn_definitely_missing", workdir,
                           f"sentinel_{si}") != 0:
                print("FAIL: live-oracle sentinel did not produce E2002 — "
                      "`mindc check` is not reaching the resolve pass")
                sys.exit(1)
        print("live sentinel: E2002 fires for a definitely-undefined name (2/2)")

        for idx, (src, name, live_ok) in enumerate(CASES):
            exp = rust_member(src, name)
            sb = ctypes.create_string_buffer(src.encode(), len(src.encode()))
            nb = ctypes.create_string_buffer(name.encode(), len(name.encode()))
            got = fn(
                ctypes.cast(sb, ctypes.c_void_p).value,
                len(src.encode()),
                ctypes.cast(nb, ctypes.c_void_p).value,
                len(name.encode()),
            )
            if live_ok:
                live = live_member(mindc, src, name, workdir, idx)
                live_checked += 1
                ok = got == exp == live
                live_s = str(live)
            else:
                live = None
                ok = got == exp
                live_s = "-"
            total += 1
            if exp == 1:
                positives += 1
            else:
                negatives += 1
            if not ok:
                fails += 1
            mark = "ok " if ok else "DIFF"
            which = "A" if src is SRC_A else "B"
            print(f"  {mark} got={got} rule={exp} live={live_s} src={which} {name!r}")

    print(
        f"decl_names: cases={total} positives={positives} "
        f"negatives={negatives} live_checked={live_checked} fails={fails}"
    )
    if positives < 5 or negatives < 5 or live_checked < 10:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND decl-name set diverges from the Rust rule")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
