#!/usr/bin/env python3
"""RI-D1a gate (task #313): `mindc build --backend=native` must FAIL-CLOSED with an
EXACT diagnostic that NAMES the unsupported construct on stderr and exits NON-ZERO —
never the old silent 0-byte-stdout-exit-0. The pure-MIND self-host driver
(selfhost_driver.mind) scans the user region and writes the diagnostic; the Rust
bridge (run_native_backend_bridge, src/bin/mindc.rs) surfaces stage1.elf's stderr.

Four invariants, all through the SHIPPING `mindc build --backend=native` path
(spawns the frozen pure-MIND stage1.elf — zero MLIR/LLVM/clang):

  (1) SUPPORTED (scalar)  -> exit 0, an ELF is written, it RUNS to the expected code.
  (2) UNSUPPORTED tensor  -> exit != 0, NO artifact, stderr names "tensor".
  (3) UNSUPPORTED trait   -> exit != 0, NO artifact, the parser diagnostic names
      the construct and carries file/source/caret context.
  (4) MALFORMED source    -> exit != 0, NO artifact, the same structured parser
      diagnostic carries file/source/caret context.
  (5) SECOND FENCE        -> a whole-array binop fed straight to the pure-MIND
      compiler (no Rust fence in front) emits no ELF, while an element-read
      program fed the same way runs to its value. The compiler is --stage1, or
      one hop of the self-host loop: the committed seed compiling THIS tree.
      The case prints the sha256 of the compiler it fed and of the committed
      seed, so a mutation run shows it ran a compiler other than the seed.

  (6) LANE FRAME      -> the compiler prints its key table (MINDKEY) equal to testdata/lane_keys.txt;
      a refusal of the lane is ONE stderr line with exit 3, no artifact and no other backend named
      (the renamed string refusals fed as three programs, the rodata window at its limit, and nine reason words forced through the
      MINDHOOK test input, each with a hand-computed line and column, in a fn, outside any fn,
      in the std region, past the image); the same programs give the same bytes with and without
      the MINDSLF self-mode magic. A construct the native parser stops on, together with a refused
      literal, is named on the failure path through `mindc build` (the host puts the std prefix in
      front of the user text): the raw-newline key at the user line, the first of two such literals,
      and a non-quote token that keeps the generic text. An unknown escape after that construct is
      named by a direct feed, because the host refuses the escape before the compiler runs.
      A user region that starts with a line feed counts that line: one forced word, and one real
      refusal through mindc build, both at a hand-counted line.
      A body-less top-level `fn` declaration does not hold a later site: a bad const after
      `extern "C" fn getpid() -> i64;` names no function, and the same literal inside a later
      function names that function. Both keys are checked on a direct feed and on the std-prefixed
      image. `mindc build` of that brace-less declaration stops in the host parser (expected '{').
      An extern block around the declaration is forwarded for a top-level raw-newline const and
      names no function. A const nested in a later function is the host's nested-item refusal,
      and that refusal names the later function. An unknown escape stops in the host parser.

  --stage1 PATH  the native compiler for every case: (1)-(4) run mindc with
                 MINDC_NATIVE_ELF=PATH, (5) feeds PATH directly.

This is the regression coverage for the fail-close diagnostic: without it, a
future driver/bridge change could silently revert to the 0-byte generic message
(UNSUPPORTED_FEATURE_DIAGNOSTICS regressing from EXACT), and nothing else catches it.

Exit: 0 all pass; 1 a gate failed; 2 BLOCKED (missing mindc / stage1.elf).

Run:  python3 examples/mindc_mind/self_host_native_diag_smoke.py
"""
import argparse
import hashlib
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from _selfhost_sources import HOOK_MAGIC, SELF_MAGIC, capture_bundle, lane_key_table, start_child  # noqa: E402
MINDC = pathlib.Path(os.environ.get("MINDC_BIN", str(_REPO / "target" / "release" / "mindc")))
STAGE1 = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"

SCALAR = "fn main() -> i64 { return 7 + 35; }\n"
TENSOR = "fn sink(x: Tensor<f32, [4]>) -> i64 { return 0; }\nfn main() -> i64 { return 0; }\n"
TRAIT = "trait Greet { fn hi(self) -> i64; }\nfn main() -> i64 { return 0; }\n"
MALFORMED = "fn main( -> i64 { return 0; }\n"
WHOLE_ARRAY = "fn main() -> i64 { let a = [8, 6]; let b = [2, 3]; let c = a + b; return c[0]; }\n"
ELEMENTS = "fn main() -> i64 { let a = [5, 6]; return a[1] + a[0]; }\n"


def tree_stage1() -> bytes:
    """One hop of the self-host loop: the committed seed compiles this tree."""
    sys.path.insert(0, str(_HERE))
    import self_host_loop_smoke as loop  # noqa: E402
    _combined, image, _user_lo = loop.build_seed()
    return loop.run_elf(STAGE1, image)


def compile_direct(stage1: pathlib.Path, src: str) -> bytes:
    """Feed an unseeded image straight to the compiler; return its stdout."""
    body = src.encode()
    image = struct.pack("<qq", 0, len(body)) + body
    r = start_child([str(stage1)], image, capture_output=True, timeout=600)
    return r.stdout


def run_elf_bytes(elf: bytes):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(elf)
        p = f.name
    os.chmod(p, 0o755)
    try:
        return subprocess.run([p], timeout=10).returncode
    finally:
        os.unlink(p)


def feed(compiler: pathlib.Path, body: str, user_lo: int = 0, prefix: bytes = b""):
    """(exit code, stdout, stderr text) of one direct feed: `prefix`, [user_lo][src_len], the source."""
    data = body.encode()
    r = start_child([str(compiler)], prefix + struct.pack("<qq", user_lo, len(data)) + data,
                       capture_output=True, timeout=600)
    return r.returncode, r.stdout, r.stderr.decode(errors="replace")


def lane_word(key: int, site: int) -> int:
    return key * 4294967296 + site


HEAD = "error[native-lane]: the program is outside candidate native-lane-v1 ("
TAIL = " No artifact written.\n"
PROG = "fn helper(a: i64) -> i64 {\n    return a + 1;\n}\nfn main() -> i64 {\n    return helper(1);\n}\n"
STD = "fn std_one(a: i64) -> i64 {\n    return a;\n}\n"
UTF = "fn main() -> i64 {\n    print(\"\u00e9\"); return 0;\n}\n"
# The three renamed refusals, fed directly: (source, key, text after the key; the sites are by hand).
RENAMED = [
    ("raw_newline", "fn main() -> i64 {\n    print(\"ab\ncd\");\n    return 0;\n}\n", "lane.string.raw_newline",
     " in fn `main` at line 2, column 11 of the composed user image."),
    ("escape", "fn helper() -> i64 { return 1; }\nfn main() -> i64 {\n    print(\"a\\qb\");\n    return 0;\n}\n",
     "lane.string.escape", " in fn `main` at line 3, column 11 of the composed user image."),
    ("escape_toplevel", "const T: i64 = \"a\\x41\";\nfn main() -> i64 { return T; }\n", "lane.string.escape",
     " at line 1, column 16 of the composed user image."),
]


def window_program(n: int, size: int) -> str:
    """`import std.string;` and a chain of n fns, each `return string_len("...") + <next fn>();`
    (a literal of `size` bytes each; the last fn has no call), and a main that calls the first."""
    lines = ["import std.string;"]
    for k in range(n):
        nxt = f" + g{k + 1}()" if k + 1 < n else ""
        lines.append(f'fn g{k}() -> i64 {{ return string_len("{"A" * size}"){nxt}; }}')
    lines.append("fn main() -> i64 { return g0(); }")
    return "\n".join(lines) + "\n"


# The rows of frame_checks. The last row counts them, so a row skipped without a line turns it red.
FRAME_ROWS = 36


def frame_checks(direct: pathlib.Path) -> int:
    """Invariant (6): the lane frame. Returns the number of failed rows; every row runs on every call."""
    fails = 0
    ran = 0

    def row(label: str, ok: bool, detail: str = "") -> None:
        nonlocal fails, ran
        fails += not ok
        ran += 1
        print(f"  {'PASS' if ok else 'FAIL'}  frame {label}{': ' + detail if detail and not ok else ''}")

    committed = (_HERE / "testdata" / "lane_keys.txt").read_bytes()
    row("MINDKEY table == testdata/lane_keys.txt", lane_key_table(direct) == committed and committed.count(b"\n") == 26)
    for name, src, key, where in RENAMED:
        rc, out, err = feed(direct, src)
        want = HEAD + key + ")" + where + TAIL
        row(f"renamed {name}: one line, exit 3, {key}", rc == 3 and out == b"" and err == want, repr((rc, err)))
        row(f"renamed {name}: no other backend named", "mlir" not in err.lower() and "--backend" not in err)
    # Forced words: (label, std, user, key, byte offset in the composed image, text after the key). Every
    # expectation below is typed by hand from the source text, not computed by the compiler's rule.
    def at(text: str, sub: str, extra: int = 0) -> int:
        return len(text[:text.index(sub)].encode()) + extra

    names = {16: "lane.source.string", 17: "lane.type.mismatch", 64: "lane.extern.unknown_symbol", 27: "lane.key.27",
             23: "lane.source.index", 39: "lane.string.use", 18: "lane.binop.non_scalar", 40: "lane.string.raw_newline",
             67: "lane.extern.callconv"}
    cases = [
        ("fn start", "", PROG, 16, 0, " in fn `helper` at line 1, column 1 of the composed user image."),
        ("in a fn", "", PROG, 17, at(PROG, "a + 1"), " in fn `helper` at line 2, column 12 of the composed user image."),
        ("second fn", "", PROG, 64, at(PROG, "helper(1)"),
         " in fn `main` at line 5, column 12 of the composed user image. Symbol `helper`."),
        ("closing brace", "", PROG, 27, at(PROG, "}\nfn main"), " in fn `helper` at line 3, column 1 of the composed user image."),
        ("after the fn", "", PROG, 23, at(PROG, "}\nfn main", 1), " at line 3, column 2 of the composed user image."),
        ("std region", STD, PROG, 39, at(STD, "a;"), " in fn `std_one` in the std region."),
        ("user after std", STD, PROG, 18, len(STD) + at(PROG, "helper(1)"),
         " in fn `main` at line 5, column 12 of the composed user image."),
        ("past the image", "", PROG, 40, 1000000, "."),
        ("utf-8 column", "", UTF, 67, at(UTF, "return 0"), " in fn `main` at line 2, column 17 of the composed user image."),
    ]
    for label, std, user, key, site, where in cases:
        rc, out, err = feed(direct, std + user, len(std.encode()), HOOK_MAGIC + struct.pack("<q", lane_word(key, site)))
        want = HEAD + names[key] + ")" + where + TAIL
        row(f"hook word {label}: {names[key]}", rc == 3 and out == b"" and err == want, repr((rc, err)))
    # The rodata window: 129 literals of 65,000 bytes (8,385,000) fit the 8,388,608-byte window, the 130th does not. The
    # 129-fn chain builds; the 130-fn chain is refused at the last literal (g129: line 131, column 38 by hand: 2 + 129
    # lines, and "fn g129() -> i64 { return string_len(" is 37 bytes; the literal is the argument of a String parameter). The seed committed before this release gave the generic text for it; the committed seeds name the key.
    blob = capture_bundle(_REPO).std_prefix.decode()
    rc, out, err = feed(direct, blob + window_program(129, 65000), len(blob.encode()))
    row("rodata window: a chain of 129 fns with a literal of 65,000 bytes each builds", rc == 0 and out[:4] == b"\x7fELF", repr((rc, err[-160:])))
    src = window_program(130, 65000)
    rc, out, err = feed(direct, blob + src, len(blob.encode()))
    want = HEAD + "lane.string.literal_capacity) in fn `g129` at line 131, column 38 of the composed user image." + TAIL
    row("rodata window: lane.string.literal_capacity at the 130th literal", rc == 3 and out == b"" and err == want
        and src.index('"', src.index("fn g129()")) - src.index("fn g129()") == 37, repr((rc, err[-160:])))
    # The test word 0 and the self magic change no byte of any program that compiles or is refused.
    for label, std, user in (("scalar", "", SCALAR), ("elements", "", ELEMENTS), ("two fns", "", PROG), ("std + user", STD, PROG),
                             ("refused array", "", WHOLE_ARRAY), ("refused tensor", "", TENSOR)):
        base = feed(direct, std + user, len(std.encode()))
        row(f"MINDSLF changes nothing: {label}", feed(direct, std + user, len(std.encode()), SELF_MAGIC) == base and base[0] in (0, 3))
        row(f"hook word 0 changes nothing: {label}",
            feed(direct, std + user, len(std.encode()), HOOK_MAGIC + struct.pack("<q", 0)) == base)
        # Self mode switches the rules off: a forced word is ignored when the self magic follows it.
        row(f"self mode ignores the hook word: {label}", feed(direct, std + user, len(std.encode()),
            HOOK_MAGIC + struct.pack("<q", lane_word(16, 0)) + SELF_MAGIC) == base)
    counted = ran
    row(f"every frame row ran ({counted} of {FRAME_ROWS})", counted == FRAME_ROWS)
    return fails


# Failure path (nb_report_unsupported): the native parser stops on `loop` and a refused literal is
# still named. Line and column are counted by hand from the user text. `mindc build` composes the
# std prefix, so a site packed relative to the user region lands in the std region.
RAW_ONE = (
    "fn main() -> i64 {\n"
    "    print(\"ab\ncd\");\n"
    "    let mut i: i64 = 0;\n"
    "    loop { i = i + 1; if i > 3 { break; } }\n"
    "    return i;\n"
    "}\n"
)
RAW_TWO = (
    "fn main() -> i64 {\n"
    "    print(\"ab\ncd\");\n"
    "    print(\"ab\ncd\");\n"
    "    let mut i: i64 = 0;\n"
    "    loop { i = i + 1; if i > 3 { break; } }\n"
    "    return i;\n"
    "}\n"
)
TILDE = "fn main() -> i64 {\n    let x: i64 = ~1;\n    return x;\n}\n"
ESC_LOOP = (
    "fn helper() -> i64 { return 1; }\n"
    "fn main() -> i64 {\n"
    "    print(\"a\\qb\");\n"
    "    let mut i: i64 = 0;\n"
    "    loop { i = i + 1; if i > 3 { break; } }\n"
    "    return i;\n"
    "}\n"
)
GENERIC = "error[backend-native]: unsupported construct; use --backend=mlir\n"
# A leading line feed is line 1. `s.cap` is line 3, column 12 (four spaces, then `return `).
LEAD_CAP = (
    "\nfn take(s: String) -> i64 {\n"
    "    return s.cap;\n"
    "}\n"
    "fn main() -> i64 { return take(\"x\"); }\n"
)
# 1 byte of line feed, 19 bytes of `fn main() -> i64 {\n`, four spaces: `return` is byte 24, line 3 column 5.
LEAD_FN = "\nfn main() -> i64 {\n    return 1;\n}\n"
CAP_WHERE = " in fn `take` at line 3, column 12 of the composed user image."
LEAD_WHERE = " in fn `main` at line 3, column 5 of the composed user image."
RAW_WHERE = " in fn `main` at line 2, column 11 of the composed user image."
ESC_WHERE = " in fn `main` at line 3, column 11 of the composed user image."
# A body-less declaration then a bad const. The quote of `const T` is column 16 on line 2
# with no indent, and column 20 when the const line has four spaces. Hand-counted.
DECL = "extern \"C\" fn getpid() -> i64;\n"
DECL_ESC = DECL + "const T: i64 = \"a\\qb\";\nfn main() -> i64 { return 0; }\n"
DECL_RAW = DECL + "const T: i64 = \"ab\ncd\";\nfn main() -> i64 { return 0; }\n"
LATER_ESC = DECL + "fn later() -> i64 {\n    const T: i64 = \"a\\qb\";\n    return 0;\n}\nfn main() -> i64 { return 0; }\n"
LATER_RAW = DECL + "fn later() -> i64 {\n    const T: i64 = \"ab\ncd\";\n    return 0;\n}\nfn main() -> i64 { return 0; }\n"
DECL_WHERE = " at line 2, column 16 of the composed user image."
LATER_WHERE = " in fn `later` at line 3, column 20 of the composed user image."
# A semicolon inside a signature's brackets is not the end of the function.
BRACKET = "fn main() -> [i64; 1] {\n    const T: i64 = \"a\\qb\";\n    return 0;\n}\n"
BRACKET_WHERE = " in fn `main` at line 2, column 20 of the composed user image."
# A semicolon inside a parameter list is not the end of the function.
PAREN = "fn main(x: i64;) -> i64 {\n    const T: i64 = \"a\\qb\";\n    return x;\n}\n"
PAREN_WHERE = " in fn `main` at line 2, column 20 of the composed user image."
# The host parser accepts a body-less fn only inside an extern block, and forwards a raw
# line feed to the compiler. The quote is column 16 on the const line, column 20 inside later.
BLOCK = "extern \"C\" {\n    fn getpid() -> i64;\n}\n"
BLOCK_RAW = BLOCK + "const T: i64 = \"ab\ncd\";\nfn main() -> i64 { return 0; }\n"
BLOCK_LATER = BLOCK + "fn later() -> i64 {\n    const T: i64 = \"ab\ncd\";\n    return 0;\n}\nfn main() -> i64 { return 0; }\n"
BLOCK_WHERE = " at line 4, column 16 of the composed user image."
BLOCK_LATER_WHERE = " in fn `later` at line 5, column 20 of the composed user image."
BLOCK_ESC = BLOCK + "const T: i64 = \"a\\qb\";\nfn main() -> i64 { return 0; }\n"
BLOCK_LATER_ESC = BLOCK + "fn later() -> i64 {\n    const T: i64 = \"a\\qb\";\n    return 0;\n}\nfn main() -> i64 { return 0; }\n"
# The host names a const nested in a function at the `const` keyword (four spaces, column 5).
NESTED_WHERE = "lane.source.nested_item) in fn `later` at line 5, column 5 of the composed user image."


def _line_col_at(src: str, at: int) -> tuple[int, int]:
    """1-based line and column of byte `at`, counted on the user text."""
    return src[:at].count("\n") + 1, at - src.rfind("\n", 0, at)


# The rows of string_fail_path. The last row counts them, so a row skipped without a line turns it red.
FAIL_ROWS = 45


def string_fail_path(direct: pathlib.Path) -> int:
    """Rows of the failure path. Returns the number of failed rows. `mindc build` uses `direct`."""
    fails = 0
    ran = 0

    def row(label: str, ok: bool, detail: str = "") -> None:
        nonlocal fails, ran
        fails += not ok
        ran += 1
        print(f"  {'PASS' if ok else 'FAIL'}  fail-path {label}{': ' + detail if detail and not ok else ''}")

    # The opening quote of each raw-newline literal is line 2, column 11 (four spaces, then `print(`).
    # The second literal's closing quote is line 5, column 3; the first quote is the one that is named.
    # The tilde is line 2, column 18. The escape's opening quote is line 3, column 11.
    q = chr(34)
    row("fixture: one raw newline opens at line 2 column 11", _line_col_at(RAW_ONE, RAW_ONE.index(q)) == (2, 11))
    close_two = RAW_TWO.index(");\n    let") - 1
    row("fixture: first quote at line 2 column 11, second close at line 5 column 3",
        _line_col_at(RAW_TWO, RAW_TWO.index(q)) == (2, 11) and RAW_TWO[close_two] == q
        and _line_col_at(RAW_TWO, close_two) == (5, 3))
    row("fixture: tilde is line 2 column 18", _line_col_at(TILDE, TILDE.index("~")) == (2, 18))
    esc_q = ESC_LOOP.index("print(") + len("print(")
    row("fixture: unknown escape opens at line 3 column 11",
        ESC_LOOP[esc_q] == q and _line_col_at(ESC_LOOP, esc_q) == (3, 11))
    cap_at = LEAD_CAP.index("s.cap")
    ret_at = LEAD_FN.index("return")
    row("fixture: leading newline puts s.cap at line 3 column 12", _line_col_at(LEAD_CAP, cap_at) == (3, 12))
    row("fixture: leading newline puts return at byte 24, line 3 column 5",
        ret_at == 24 and _line_col_at(LEAD_FN, ret_at) == (3, 5))
    saved = ENV.get("MINDC_NATIVE_ELF")
    ENV["MINDC_NATIVE_ELF"] = str(direct)
    try:
        for label, src in (("one raw newline", RAW_ONE), ("first of two raw newlines", RAW_TWO)):
            rc, err, art = build_native(src)
            want = HEAD + "lane.string.raw_newline)" + RAW_WHERE + TAIL
            row(f"{label}: mindc build names the first quote", rc == 3 and art is None and want in err
                and "in the std region" not in err, repr((rc, art is not None, err)))
        rc, err, art = build_native(TILDE)
        row("tilde: generic text, not a string key", rc == 3 and art is None and GENERIC in err
            and "lane.string" not in err, repr((rc, err)))
        rc, err, art = build_native(LEAD_CAP)
        want = HEAD + "lane.string.cap)" + CAP_WHERE + TAIL
        row("leading newline: mindc build counts the blank line", rc == 3 and art is None and want in err,
            repr((rc, err)))
    finally:
        if saved is None:
            ENV.pop("MINDC_NATIVE_ELF", None)
        else:
            ENV["MINDC_NATIVE_ELF"] = saved
    rc, out, err = feed(direct, ESC_LOOP)
    want = HEAD + "lane.string.escape)" + ESC_WHERE + TAIL
    row("unknown escape after loop: direct feed names lane.string.escape",
        rc == 3 and out == b"" and err == want, repr((rc, err)))
    row("unknown escape after loop: no other backend named", "mlir" not in err.lower() and "--backend" not in err)
    rc, out, err = feed(direct, LEAD_FN, 0, HOOK_MAGIC + struct.pack("<q", lane_word(40, ret_at)))
    want = HEAD + "lane.string.raw_newline)" + LEAD_WHERE + TAIL
    row("leading newline: forced word counts the blank line", rc == 3 and out == b"" and err == want,
        repr((rc, err)))

    def const_quote(src: str) -> int:
        at = src.index("const T:")
        return at + src[at:].index(q)

    row("fixture: decl escape quote is line 2 column 16", _line_col_at(DECL_ESC, const_quote(DECL_ESC)) == (2, 16))
    row("fixture: decl raw quote is line 2 column 16", _line_col_at(DECL_RAW, const_quote(DECL_RAW)) == (2, 16))
    row("fixture: later escape quote is line 3 column 20", _line_col_at(LATER_ESC, const_quote(LATER_ESC)) == (3, 20))
    row("fixture: later raw quote is line 3 column 20", _line_col_at(LATER_RAW, const_quote(LATER_RAW)) == (3, 20))
    row("fixture: bracket signature quote is line 2 column 20",
        _line_col_at(BRACKET, const_quote(BRACKET)) == (2, 20))
    row("fixture: paren signature quote is line 2 column 20",
        _line_col_at(PAREN, const_quote(PAREN)) == (2, 20))
    row("fixture: extern block raw quote is line 4 column 16",
        _line_col_at(BLOCK_RAW, const_quote(BLOCK_RAW)) == (4, 16))
    row("fixture: extern block later quote is line 5 column 20",
        _line_col_at(BLOCK_LATER, const_quote(BLOCK_LATER)) == (5, 20))
    blob = capture_bundle(_REPO).std_prefix

    def check_image(label: str, src: str, key: str, where: str) -> None:
        want_line = HEAD + key + ")" + where + TAIL
        rc, out, err = feed(direct, src)
        row(f"{label}: direct", rc == 3 and out == b"" and err == want_line, repr((rc, err)))
        rc, out, err = feed(direct, blob.decode() + src, len(blob))
        row(f"{label}: std image", rc == 3 and out == b"" and err == want_line, repr((rc, err)))

    check_image("body-less decl then escape const", DECL_ESC, "lane.string.escape", DECL_WHERE)
    check_image("body-less decl then raw const", DECL_RAW, "lane.string.raw_newline", DECL_WHERE)
    check_image("body-less decl then escape inside later", LATER_ESC, "lane.string.escape", LATER_WHERE)
    check_image("body-less decl then raw inside later", LATER_RAW, "lane.string.raw_newline", LATER_WHERE)
    check_image("semicolon inside a signature bracket", BRACKET, "lane.string.escape", BRACKET_WHERE)
    check_image("semicolon inside a parameter list", PAREN, "lane.string.escape", PAREN_WHERE)
    # Direct feed of the extern-block form: the `fn` sits inside braces, so it is not the
    # declaration that holds a later site. The top-level const names no function; the const
    # inside `later` names `later`.
    check_image("extern block then raw const", BLOCK_RAW, "lane.string.raw_newline", BLOCK_WHERE)
    check_image("extern block then raw inside later", BLOCK_LATER, "lane.string.raw_newline", BLOCK_LATER_WHERE)
    # mindc build uses the same compiler. The brace-less declaration and an unknown escape
    # stop in the host parser. A top-level raw const after an extern block is forwarded.
    # A const nested in a function is the host's nested-item refusal, which names that function.
    saved_build = ENV.get("MINDC_NATIVE_ELF")
    ENV["MINDC_NATIVE_ELF"] = str(direct)
    try:
        rc, err, art = build_native(BLOCK_RAW)
        want_line = HEAD + "lane.string.raw_newline)" + BLOCK_WHERE + TAIL
        row("extern block then raw const: mindc build", rc == 3 and art is None and want_line in err
            and "in fn `" not in err and "in the std region" not in err, repr((rc, art is not None, err)))
        rc, err, art = build_native(BLOCK_LATER)
        row("extern block then raw inside later: mindc build names the later function",
            rc == 3 and art is None and NESTED_WHERE in err and "in fn `getpid`" not in err
            and "lane.string" not in err, repr((rc, art is not None, err)))
        for label, src, marker in (
                ("extern block then escape const", BLOCK_ESC, "unknown string escape"),
                ("extern block then escape inside later", BLOCK_LATER_ESC, "unknown string escape")):
            rc, err, art = build_native(src)
            row(f"{label}: mindc build stops in the host parser",
                rc == 3 and art is None and marker in err and "E1001" in err
                and "in fn `getpid`" not in err and "lane.string" not in err, repr((rc, err[:240])))
        for label, src in (
                ("brace-less decl then escape const", DECL_ESC),
                ("brace-less decl then raw const", DECL_RAW),
                ("brace-less decl then escape inside later", LATER_ESC),
                ("brace-less decl then raw inside later", LATER_RAW)):
            rc, err, art = build_native(src)
            row(f"{label}: mindc build does not name the declaration",
                rc == 3 and art is None and "in fn `getpid`" not in err and "E1001" in err
                and "lane.string" not in err, repr((rc, err[:240])))
    finally:
        if saved_build is None:
            ENV.pop("MINDC_NATIVE_ELF", None)
        else:
            ENV["MINDC_NATIVE_ELF"] = saved_build
    counted = ran
    row(f"every fail-path row ran ({counted} of {FAIL_ROWS})", counted == FAIL_ROWS)
    return fails


def build_native(src: str):
    """mindc build --backend=native <src>; return (exit_code, stderr_text, artifact_bytes_or_None)."""
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "p.mind"
        op = pathlib.Path(td) / "p.bin"
        sp.write_text(src)
        r = subprocess.run(
            [str(MINDC), "build", "--backend=native", f"--out={op}", str(sp)],
            capture_output=True, text=True, timeout=120, env=ENV,
        )
        art = op.read_bytes() if op.exists() and op.stat().st_size > 0 else None
        return r.returncode, (r.stderr or ""), art


ENV = dict(os.environ)

# The legs of main. The verdict lines of a green run: each leg, each row table with its count row,
# the legs' count row and the ALL PASS line. Every runner call site demands VERDICTS as its floor.
MAIN_LEGS = 5
VERDICTS = MAIN_LEGS + 1 + FRAME_ROWS + 1 + FAIL_ROWS + 1 + 1


def main() -> int:
    global STAGE1
    ap = argparse.ArgumentParser(description="RI-D1a native diagnostic gate")
    ap.add_argument("--stage1", type=pathlib.Path, default=None,
                    help="native compiler for every case (default: the committed seed)")
    args = ap.parse_args()
    if args.stage1 is not None:
        STAGE1 = args.stage1.resolve()
        ENV["MINDC_NATIVE_ELF"] = str(STAGE1)
    print(f"  stage1 = {STAGE1}")
    if not MINDC.exists():
        print(f"BLOCKED: mindc not found at {MINDC}")
        return 2
    if not STAGE1.exists():
        print(f"BLOCKED: frozen stage1.elf not found at {STAGE1}")
        return 2

    fails = 0
    legs = 0

    def leg(ok: bool, text: str) -> None:
        nonlocal fails, legs
        fails += 0 if ok else 1
        legs += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")

    # (1) SUPPORTED scalar -> exit 0 + ELF that runs to 42
    ec, err, art = build_native(SCALAR)
    ran = None
    if art is not None and art[:4] == b"\x7fELF":
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(art)
            p = f.name
        os.chmod(p, 0o755)
        try:
            ran = subprocess.run([p], timeout=10).returncode
        finally:
            os.unlink(p)
    ok = (ec == 0 and art is not None and art[:4] == b"\x7fELF" and ran == 42)
    leg(ok, f"scalar: exit={ec} "
        f"elf={'YES' if art else 'NO'} result={ran}(want 42)")

    # (2) UNSUPPORTED tensor -> non-zero, no artifact, stderr names the construct
    ec, err, art = build_native(TENSOR)
    ok = (ec != 0 and art is None and "tensor" in err.lower())
    leg(ok, f"tensor: exit={ec} artifact={'YES' if art else 'NO'} "
        f"names_tensor={'tensor' in err.lower()}  stderr={err.strip()[:70]!r}")

    # (3) UNSUPPORTED trait -> non-zero, no artifact, exact parser context
    ec, err, art = build_native(TRAIT)
    trait_context = all(part in err for part in (
        "error[parse][E1001]", "p.mind:1:", TRAIT.splitlines()[0], "^"
    ))
    ok = (ec != 0 and art is None and "trait" in err.lower() and trait_context)
    leg(ok, f"trait:  exit={ec} artifact={'YES' if art else 'NO'} "
        f"names_trait={'trait' in err.lower()} context={trait_context}  "
        f"stderr={err.strip()[:100]!r}")

    # (4) MALFORMED source -> structured parse diagnostic with source context
    ec, err, art = build_native(MALFORMED)
    malformed_context = all(part in err for part in (
        "error[parse][E1001]", "p.mind:1:", MALFORMED.strip(), "^"
    ))
    ok = (ec != 0 and art is None and malformed_context)
    leg(ok, f"malformed: exit={ec} "
        f"artifact={'YES' if art else 'NO'} context={malformed_context}  "
        f"stderr={err.strip()[:100]!r}")

    # (5) SECOND FENCE -> the pure-MIND compiler itself refuses an array operand
    with tempfile.TemporaryDirectory() as td:
        if args.stage1 is not None:
            direct = STAGE1
        else:
            direct = pathlib.Path(td) / "stage1.elf"
            direct.write_bytes(tree_stage1())
            direct.chmod(0o755)
        fed_sha = hashlib.sha256(direct.read_bytes()).hexdigest()
        refused = compile_direct(direct, WHOLE_ARRAY)
        control = compile_direct(direct, ELEMENTS)
        got = run_elf_bytes(control) if control[:4] == b"\x7fELF" else None
    wrote = refused[:4] == b"\x7fELF"
    ok = not wrote and got == 11
    leg(ok, f"second fence: whole-array binop "
        f"elf={'YES' if wrote else 'NO'}(want NO)  element read result={got}(want 11)")
    seed = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
    seed_sha = hashlib.sha256(seed.read_bytes()).hexdigest()
    print(f"        compiler sha256={fed_sha}  committed seed sha256={seed_sha}")

    # (6) LANE FRAME on the same compiler the second fence fed
    with tempfile.TemporaryDirectory() as td:
        if args.stage1 is not None:
            direct = STAGE1
        else:
            direct = pathlib.Path(td) / "stage1.elf"
            direct.write_bytes(tree_stage1())
            direct.chmod(0o755)
        fails += frame_checks(direct)
        fails += string_fail_path(direct)
    ok = legs == MAIN_LEGS
    fails += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  every main leg ran ({legs} of {MAIN_LEGS})")
    if fails:
        print(f"FAIL: {fails} RI-D1a diagnostic gate(s) failed")
        return 1
    print("ALL PASS  (RI-D1a: native backend fail-closes with an EXACT construct-naming "
          "diagnostic + non-zero exit; supported scalar still compiles+runs; zero MLIR/LLVM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
