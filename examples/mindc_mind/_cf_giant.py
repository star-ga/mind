#!/usr/bin/env python3
"""Program generators of self_host_native_cf_smoke (census X1 capacity cases).

Each generator returns (source, expected) where expected is the (exit status,
stdout) an independent evaluation of the same program gives (computed here in
Python, never by the compiler), or ("refused", 3) for a program that must be
refused cleanly. The case lists pair each program with whether a clean rc-3
refusal is accepted. Not a gate: importing it runs nothing.
"""
import bisect

import _cf_corpus as C

BOUNDED = (768 << 20, 384 << 20, 1 << 20)  # arena, stack, guard (self_host_bounded_profile_smoke)


def giant_program(arms: int) -> tuple:
    """X1 shape: one function holding a huge else-if chain inside a loop."""
    lines = ["fn pick(x: i64) -> i64 {", "    let mut r: i64 = 0;"]
    for k in range(arms):
        kw = "if" if k == 0 else "} else if"
        lines.append(f"    {kw} x == {k} {{")
        lines.append(f"        r = {(k * 7 + 3) % 1000};")
    lines += ["    } else {", "        r = 0 - 1;", "    }", "    r", "}",
              "fn main() -> i64 {", "    let mut i: i64 = 0;", "    let mut s: i64 = 0;",
              f"    while i < {arms + 3} {{",
              "        s = s + pick(i * 3);",
              "        if s > 1000000 && i > 2 {", "            return 0 - 2;", "        }",
              "        i = i + 1;", "    }", "    s % 251", "}"]
    s = 0
    for i in range(arms + 3):
        x = i * 3
        s += (x * 7 + 3) % 1000 if x < arms else -1
        if s > 1000000 and i > 2:
            return "\n".join(lines) + "\n", (C.wrap(-2) & 255, b"")
    return "\n".join(lines) + "\n", (C.tmod(s, 251) & 255, b"")


def straight_line_program(blocks: int) -> tuple:
    """Census X1 shape (MIND-Lab nr_literals): one straight-line function of
    `blocks` x {let, early-return if, store} — 600+ statements. The mic@3 twin's
    statement planner had a fixed 512-record table and segfaulted the compiler
    on it."""
    lines = ["fn space(c: i64, n: i64) -> i64 {", "    if n > 0 {", "        return __mind_alloc(n);",
             "    }", "    0", "}", "fn table(c: i64) -> i64 {",
             f"    let t: i64 = space(c, {8 * blocks});", "    if t == 0 { return 0; }"]
    for k in range(blocks):
        lines.append(f"    let p{k}: i64 = space(c, 8);")
        lines.append(f"    if p{k} == 0 {{ return 0; }}")
        lines.append(f"    __mind_store_i64(p{k}, {k * 3 + 1});")
        lines.append(f"    __mind_store_i64(t + {8 * k}, __mind_load_i64(p{k}));")
    lines += ["    t", "}", "fn main() -> i64 {", "    let t: i64 = table(1);", "    let mut s: i64 = 0;",
              "    let mut i: i64 = 0;", f"    while i < {blocks} {{",
              "        s = s + __mind_load_i64(t + i * 8);", "        i = i + 1;", "    }", "    s % 251", "}"]
    return "\n".join(lines) + "\n", (sum(k * 3 + 1 for k in range(blocks)) % 251, b"")


def reassign_program(n: int, shape: str) -> tuple:
    """Census X1 let-env capacity: every parameter, `let` and reassignment of
    the SSA path appends one entry to its 512-entry let-env, which had no bound
    check — `n` straight-line reassignments (n >= 511), n let/assign pairs
    (n >= 256) or n reassignments in a loop body overran it and segfaulted
    stage1. Such a function must compile (HOME) and run correctly."""
    lines = ["fn f(a: i64) -> i64 {", "    let mut s: i64 = a;"]
    s, reps = 1, 1
    if shape == "loop":
        lines.append("    let mut i: i64 = 0;")
        lines.append("    while i < 3 {")
        reps = 3
    for k in range(n):
        if shape == "pairs":
            lines += [f"    let x{k}: i64 = s + {k};", f"    s = x{k} % 1000;"]
        else:
            lines.append(f"    s = (s + {k}) % 1000;")
    if shape == "loop":
        lines += ["        i = i + 1;", "    }"]
    lines += ["    s", "}", "fn main() -> i64 { f(1) % 251 }"]
    for _ in range(reps):
        for k in range(n):
            s = (s + k) % 1000
    return "\n".join(lines) + "\n", (s % 251, b"")


def dense_tree_program(depth: int, pad: int = 0) -> tuple:
    """Census X1 shape at scale (MIND-Lab cs_ucd_mask is a 2006-node if/else tree
    over a code point): one function holding a balanced if/else tree of `depth`
    levels (2**depth leaf returns, 4-space indentation), optionally padded with
    `pad` comment lines inside the body. Its source span, not its node count, is
    what used to size the HOME tables, so the indentation alone pushed the
    bounded compiler's stack past its guard."""
    lim = 1114112
    leaves = []

    def leaf(k: int) -> int:
        return (k * 2654435761 + 97) % 251

    lines = ["fn mask(cp: i64) -> i64 {", "    if cp < 0 || cp > 1114111 { return 0; }"]
    lines += [f"    // capacity padding line {i}: words if while let && || in a comment" for i in range(pad)]

    def tree(lo: int, hi: int, d: int, ind: int) -> None:
        sp = "    " * ind
        if d == 0:
            leaves.append((lo, hi))
            lines.append(f"{sp}return {leaf(len(leaves) - 1)};")
            return
        mid = (lo + hi) // 2
        lines.append(f"{sp}if cp < {mid} {{")
        tree(lo, mid, d - 1, ind + 1)
        lines.append(f"{sp}}} else {{")
        tree(mid, hi, d - 1, ind + 1)
        lines.append(f"{sp}}}")

    tree(0, lim, depth, 1)
    lines += ["}", "fn main() -> i64 {", "    let mut cp: i64 = 0 - 3;", "    let mut acc: i64 = 0;",
              "    while cp < 1114130 {", "        acc = (acc * 131 + mask(cp) + 7) % 1000003;",
              "        cp = cp + 37;", "    }", "    return acc % 251;", "}"]

    lows = [lo for lo, _ in leaves]

    def mask(cp: int) -> int:
        if cp < 0 or cp > 1114111:
            return 0
        k = bisect.bisect_right(lows, cp) - 1
        assert leaves[k][0] <= cp < leaves[k][1]
        return leaf(k)

    acc, cp = 0, -3
    while cp < 1114130:
        acc = (acc * 131 + mask(cp) + 7) % 1000003
        cp += 37
    return "\n".join(lines) + "\n", (acc % 251, b"")


def flat_program(n: int, shape: str) -> tuple:
    """Census X1 giant FLAT function: one block of `n` statements.
    inc:    `s = s + 1;` x n (straight-line reassignments of one name)
    assign: `s = (s + k) % 1000;` x n (the same, ~183 B of code per statement:
            40000 of them pass the image's 4 MiB code window, a clean rc 3)
    lets:   `let xK: i64 = xJ + k % 7;` x n (a chain of n bindings)
    loop:   the assign list as the body of a 2-trip while loop.
    Every statement-list walk of the compiler took one stack frame per
    statement (the mic@3 twin's planner frame alone is ~14 KB, and its env
    lookup at statement i recursed over the i bindings before it): the bounded
    compiler (384 MiB stack) died with SIGSEGV at ~26k straight-line statements."""
    trips = 2 if shape == "loop" else 1
    ind = "        " if shape == "loop" else "    "
    lines = ["fn f(a: i64) -> i64 {", "    let mut s: i64 = a;"]
    if shape == "loop":
        lines += ["    let mut i: i64 = 0;", "    while i < 2 {"]
    v = 1
    if shape == "lets":
        lines.append("    let x0: i64 = s;")
        for k in range(1, n):
            lines.append(f"    let x{k}: i64 = x{k - 1} + {k % 7};")
            v += k % 7
        lines.append(f"    s = x{n - 1} % 1000;")
        v %= 1000
    elif shape == "inc":
        lines += ["    s = s + 1;"] * n
        v += n
    else:
        for k in range(n):
            lines.append(f"{ind}s = (s + {k % 997}) % 1000;")
        for _ in range(trips):
            for k in range(n):
                v = (v + k % 997) % 1000
    if shape == "loop":
        lines += ["        i = i + 1;", "    }"]
    lines += ["    s", "}", "fn main() -> i64 { f(1) % 251 }"]
    return "\n".join(lines) + "\n", (v % 251, b"")


def deep_program(kind: str, d: int) -> tuple:
    """Census X1 DEEP trees: one expression or block nest `d` levels deep inside
    f (see deep_cases)."""
    s = 1
    if kind == "add":
        body = ["    s = " + " + ".join(["1"] * d) + ";"]
        s = d
    elif kind == "parens":
        body = ["    s = " + "(" * d + "s + 1" + ")" * d + ";"]
        s = 2
    elif kind == "calls":
        body = ["    s = " + "g(" * d + "s" + ")" * d + ";"]
        for _ in range(d):
            s = (s + 1) % 1000
    elif kind == "neg":
        body = ["    s = " + "0 - (" * d + "s" + ")" * d + ";"]
        s = 1 if d % 2 == 0 else -1
    elif kind == "and":
        body = ["    if " + " && ".join(["s > 0"] * d) + " { s = s + 5; }"]
        s = 6
    elif kind == "as":
        body = ["    s = s" + " as i64" * d + ";"]
    elif kind == "uminus":
        body = ["    s = " + "-" * (2 * (d // 2)) + "s;"]
    elif kind == "nestif":
        body = ["    if s > 0 {" * d, "    s = s + 7;", "    }" * d]
        s = 8
    elif kind == "vif":
        body = ["    s = " + "if s > 0 { " * d + "s + 2" + " } else { 0 }" * d + ";"]
        s = 3
    elif kind == "elsevif":
        body = ["    s = " + "if s < 0 { 0 } else { " * d + "s + 2" + " }" * d + ";"]
        s = 3
    elif kind == "while":
        body = ["    let mut c: i64 = 0;", "    while c < 1 {" * d, "    c = c + 1; s = s + 4;", "    }" * d]
        s = 5
    elif kind == "for":
        body = ["    for i%d in 0..1 {" % k for k in range(d)] + ["    s = s + 4;", "    }" * d]
        s = 5
    elif kind == "whileif":
        body = ["    let mut c: i64 = 0;", "    while c < 1 { if c < 1 {" * (d // 2),
                "    c = c + 1; s = s + 4;", "    } }" * (d // 2)]
        s = 5
    elif kind == "condadd":
        body = ["    if s" + " + 1" * d + " > 0 { s = 5; }"]
        s = 5
    elif kind == "vifbranch":
        body = ["    s = if s > 0 { s" + " + 1" * d + " } else { 0 };"]
        s = 1 + d
    elif kind == "whilecond":
        body = ["    let mut c: i64 = 0;", "    while c" + " + 0" * d + " < 1 { c = c + 1; s = s + 2; }"]
        s = 3
    elif kind == "not":
        body = ["    if " + "!" * (2 * (d // 2)) + "(s > 0) { s = s + 5; }"]
        s = 6
    elif kind == "condvif":
        e = "s"
        for _ in range(d):
            e = f"if ({e}) > 0 {{ 1 }} else {{ 0 }}"
        body = [f"    s = {e};"]
    else:
        raise ValueError(kind)
    src = "\n".join(["fn g(x: i64) -> i64 { (x + 1) % 1000 }", "fn f(a: i64) -> i64 {",
                     "    let mut s: i64 = a;"] + body + ["    s", "}", "fn main() -> i64 { f(1) % 251 }"]) + "\n"
    return src, (C.tmod(s, 251) & 255, b"")


def deep_cases():
    """(label, (source, expected), may_refuse, seeded): deep trees in the bounded
    production profile. Each must compile and run to the Python value, or (may_refuse)
    be refused cleanly (rc 3); never a signal, never past the compile timeout."""
    return [
        ("40000-term `+` chain in one expression", deep_program("add", 40000), True, True),
        ("40000 nested parentheses", deep_program("parens", 40000), True, True),
        ("40000-deep nested calls g(g(...))", deep_program("calls", 40000), True, True),
        ("40000-deep `0 - (...)` chain", deep_program("neg", 40000), True, True),
        ("40000-term `&&` chain in an if condition", deep_program("and", 40000), True, True),
        ("40000-deep `as i64` chain", deep_program("as", 40000), True, True),
        ("40000-deep unary minus chain", deep_program("uminus", 40000), True, True),
        ("40000 nested if statements", deep_program("nestif", 40000), True, True),
        ("15 nested value-ifs (2^15 mic@3 twin probes)", deep_program("vif", 15), False, True),
        ("15-arm else-if value chain", deep_program("elsevif", 15), False, True),
        ("300 nested value-ifs", deep_program("vif", 300), False, True),
        ("300 nested while loops", deep_program("while", 300), False, True),
        ("4000-term `+` chain (under the budget)", deep_program("add", 4000), False, True),
        ("8000 nested parentheses (under the budget)", deep_program("parens", 8000), False, True),
        ("2000 nested if statements (under the budget)", deep_program("nestif", 2000), False, True),
        ("10000 if expressions nested in parenthesised conditions", deep_program("condvif", 10000), True, True),
        # Long expressions past the mic@3 twin's 256-entry descriptor tables
        # (no bound check: arena corruption, SIGSEGV); each must compile and run.
        ("1000-term `+` chain in an if condition", deep_program("condadd", 1000), False, True),
        ("300-term value-if branch value", deep_program("vifbranch", 300), False, True),
        ("300-term while condition", deep_program("whilecond", 300), False, True),
        ("400-deep `!` chain in an if condition", deep_program("not", 400), False, True),
        # Nested loops past the twin's while-depth cap (12) but under the
        # statement cap (8192): every twin level re-walked the whole nest with
        # tables sized from the function (depth x size), so 600..4000 nested
        # while loops exhausted the 768 MiB arena (exit 70, neither a
        # compile nor the rc-3 refusal); 10000 passed only because the
        # statement cap routed it past the twin. Each must compile and run.
        ("600 nested while loops", deep_program("while", 600), False, True),
        ("1000 nested while loops", deep_program("while", 1000), False, True),
        ("4000 nested while loops", deep_program("while", 4000), False, True),
        ("600 nested for loops", deep_program("for", 600), False, True),
        ("1000-deep alternating while/if nest", deep_program("whileif", 1000), False, True),
    ]


def deep_v1_cases():
    """(label, (source, ("refused", 3)), seeded) for the BOUNDED compiler fed a
    default-layout (v1) image: its own stack is 384 MiB whatever layout the
    image requests, so a tree past that budget must be refused, never a signal."""
    return [("bounded compiler, default-layout image: 40000-term `+` chain",
             (deep_program("add", 40000)[0], ("refused", 3)), True)]


def deep_default_cases():
    """(label, (source, expected), may_refuse, seeded) for the default-profile
    compiler (4 GiB stack): a 300000-term chain is past even that budget."""
    return [("300000-term `+` chain (past the default 4 GiB stack budget)",
             deep_program("add", 300000), True, True)]


def flat_cases():
    """(label, (source, expected), may_refuse, seeded): giant flat statement
    lists in the bounded production profile. Each must compile and run to the
    Python value, or (may_refuse) be refused cleanly (rc 3); never a signal."""
    return [
        ("40000 straight-line statements in one function (census X1 giant flat list)",
         flat_program(40000, "inc"), False, True),
        # 20000 bindings pass the native alias environment's 4096-binding lookup
        # depth (alias_cap: a structured refusal), so this one may refuse; 4000
        # stay under it and must compile (the mic@3 twin's let-chain tables were
        # a fixed 512 bindings / 4096 descriptors and silently overran).
        ("20000 chained lets in one function", flat_program(20000, "lets"), True, True),
        ("4000 chained lets in one function", flat_program(4000, "lets"), False, True),
        ("40000 heavier straight-line statements (past the 4 MiB code window)",
         flat_program(40000, "assign"), True, True),
        ("30000 statements in one while body", flat_program(30000, "loop"), True, True),
    ]


FLAT_POSITIONS = ("body", "then", "else", "while", "vif", "calls", "ifseq", "lets", "shadow", "mixed",
                  "and3", "or3")
# Tokens one statement of each position costs (the bounded compiler budgets the
# whole program at 640 B of stack per token, nb_stack_short).
FLAT_TOKENS = {"body": 6, "then": 6, "else": 6, "while": 6, "vif": 6, "calls": 7, "ifseq": 12,
               "lets": 9, "shadow": 9, "mixed": 8, "and3": 20, "or3": 20}
# Mid size per position, and whether a clean rc 3 is accepted there (measured:
# 20000 chained lets pass the native alias environment's 4096-binding lookup
# depth, a structured refusal; every other position compiles and runs).
FLAT_MID = {"body": 40000, "then": 40000, "else": 40000, "while": 40000, "vif": 40000, "calls": 40000,
            "ifseq": 20000, "lets": 20000, "shadow": 40000, "mixed": 20000, "and3": 25000, "or3": 25000}
FLAT_MID_MAY_REFUSE = ("lets", "mixed", "and3", "or3")
# Sizes that fit every budget, so they must compile and run: 6000 three-term
# `&&` / `||` conditions emit ~3.5 MB of code (under the 4 MiB code window).
FLAT_SMALL = {"and3": 6000, "or3": 6000}
FLAT_CPU = 240  # CPU seconds per compile; every case takes < 30 s


def flat_budget_tokens() -> int:
    """Program tokens the bounded compiler admits (nb_stack_short: 640 B per
    token plus 16 MiB of its 384 MiB stack), less the seeded std prefix
    (~57k tokens) and a margin."""
    return ((BOUNDED[1] - (16 << 20)) // 640) - 60000


def flat_matrix_program(pos: str, n: int) -> tuple:
    """One function holding a flat list of `n` statements in statement position
    `pos`, and its value computed here (independent of the compiler)."""
    L = ["fn g(x: i64) -> i64 { (x + 1) % 1000 }", "fn f(a: i64) -> i64 {", "    let mut s: i64 = a;"]
    v = 1
    if pos in ("body", "then", "else", "while", "vif"):
        stmt = {"body": "    s = s + 1;", "vif": "        t = t + 1;"}.get(pos, "        s = s + 1;")
        head = {"body": [], "then": ["    if a > 0 {"], "else": ["    if a < 0 { s = 0; } else {"],
                "while": ["    let mut i: i64 = 0;", "    while i < 2 {"],
                "vif": ["    s = if a > 0 {", "        let mut t: i64 = s;"]}[pos]
        tail = {"body": [], "then": ["    }"], "else": ["    }"], "while": ["        i = i + 1;", "    }"],
                "vif": ["        t", "    } else { 0 };"]}[pos]
        L += head + [stmt] * n + tail
        v += 2 * n if pos == "while" else n
    elif pos == "calls":
        L += ["    s = g(s);"] * n
        for _ in range(n):
            v = (v + 1) % 1000
    elif pos == "ifseq":
        L += ["    if s > 0 { s = s + 1; }"] * n
        v += n
    elif pos == "lets":
        L.append("    let x0: i64 = s;")
        for k in range(1, n):
            L.append(f"    let x{k}: i64 = x{k - 1} + {k % 7};")
            v += k % 7
        L.append(f"    s = x{n - 1} % 1000;")
        v %= 1000
    elif pos == "shadow":
        for k in range(n):
            L.append(f"    let s: i64 = s + {k % 5};")
            v += k % 5
        L.append("    s = s % 1000;")
        v %= 1000
    elif pos == "and3":
        # Census A1 x X1: a three-term `&&` condition in each of n if statements.
        # The HOME lowering desugars each into nested ifs (~26 KB of arena per
        # statement); 24000..27000 of them exhausted the bounded compiler's
        # 768 MiB arena (exit 70) before any refusal could run.
        for k in range(n):
            L.append(f"    if s > 0 && s < {k + 100000} && a > 0 {{ s = s + 1; }}")
            v += 1
    elif pos == "or3":
        for k in range(n):
            L.append(f"    if s < 0 || a < 0 || s > {k % 3} {{ s = s + 1; }}")
            if v < 0 or v > k % 3:
                v += 1
    elif pos == "mixed":
        y = 0
        for k in range(n):
            if k % 2 == 0:
                L.append(f"    let y{k}: i64 = s + {k % 3};")
                y = v + k % 3
            else:
                L.append(f"    s = y{k - 1} % 1000;")
                v = y % 1000
    else:
        raise ValueError(pos)
    L += ["    s", "}", "fn main() -> i64 { f(1) % 251 }"]
    return "\n".join(L) + "\n", (v % 251, b"")


def flat_matrix_cases():
    """(label, (source, expected), may_refuse) over FLAT_POSITIONS x {mid size,
    the largest size the token budget admits}."""
    out = []
    for pos in FLAT_POSITIONS:
        mid = FLAT_MID[pos]
        out.append((f"{pos}: {mid} statements", flat_matrix_program(pos, mid), pos in FLAT_MID_MAY_REFUSE))
        if pos in FLAT_SMALL:
            out.append((f"{pos}: {FLAT_SMALL[pos]} statements (fits every budget)",
                        flat_matrix_program(pos, FLAT_SMALL[pos]), False))
        top = flat_budget_tokens() // FLAT_TOKENS[pos]
        out.append((f"{pos}: {top} statements (token-budget boundary)", flat_matrix_program(pos, top), True))
    return out


def bounded_cases():
    """(label, (source, expected), may_refuse, seeded) for the bounded production compiler.
    Each must compile and run to the Python value; a may_refuse case may instead
    be refused cleanly (rc 3). A compiler killed by a signal is always a FAIL."""
    return [
        ("dense if/else tree, 2048 leaves (census X1 cs_ucd_mask shape at 7x)",
         dense_tree_program(11), False, True),
        ("dense if/else tree, 1024 leaves + 2600 comment lines in the body",
         dense_tree_program(10, pad=2600), False, True),
    ] + flat_cases() + deep_cases() + extreme_cases()


def extreme_cases():
    """(label, (source, expected), may_refuse, seeded) whole-program extremes, run
    in BOTH the default and the bounded profile: each must compile and run to the
    Python value or be refused cleanly (rc 3); an expected ("refused", 3) must
    refuse. Each SIGSEGVed stage1 before nb_stack_short, the halving
    nb_region_has and the halving source scanners (census X1). The UNSEEDED image
    (user_lo = 0, no std blob) matters: without the std prefix the source holds
    no `__mind_alloc` / `__mind_store_i64` / `__mind_load_i64` literal, so the
    trace-hash front end scanned (and copied) the whole source one stack frame
    per byte (find_alloc_name_span: 1.57M frames at the bounded 384 MiB stack's
    end on a 21 MB program, 3M+ frames on the 18.8 MB tree in the default
    profile; a 1.96 MB tree already overflowed the bounded stack). The 18.8 MB
    tree runs seeded only: unseeded it reaches the same scanners as the 21 MB
    unseeded case below, at twice the compile time."""
    pad = refused_padded_program(320000)
    return [
        # ~786k tokens (18.8 MB): past the bounded profile's whole-program stack
        # budget (nb_stack_short); the front end's per-token walk overflowed the
        # 384 MiB stack here (SIGSEGV), so that compiler must refuse it cleanly.
        ("dense if/else tree, 65536 leaves (18.8 MB, past the bounded stack budget)",
         dense_tree_program(16), True, True),
        # A 21 MB program that is refused (a Tensor parameter, after 320k comment
        # lines): the refusal reporter scanned the user region one stack frame
        # per byte and overflowed the stack (SIGSEGV) instead of exiting rc 3.
        ("21 MB comment-padded program with an unsupported construct (must refuse)",
         pad, False, True),
        ("21 MB comment-padded program with an unsupported construct, unseeded image (must refuse)",
         pad, False, False),
        # 8192 leaves (1.96 MB): fits every budget, so it must compile and run.
        ("dense if/else tree, 8192 leaves (1.96 MB), unseeded image", dense_tree_program(13), False, False),
    ]


def refused_padded_program(lines: int):
    pad = "// padding comment line for the refusal reporter scan xxxxxxxxxxxx\n" * lines
    src = ("fn main() -> i64 { return 0; }\n" + pad
           + "fn sink(x: Tensor<f32, [4]>) -> i64 { return 0; }\n")
    return src, ("refused", 3)


REFUSE_ADDR = """fn f(n: i64) -> i64 {
    let mut x: i64 = 0;
    let r: i64 = &x;
    while n > 0 {
        if n == 3 || n == 5 {
            return x;
        }
        n = n - 1;
    }
    x
}
fn main() -> i64 { return f(4); }
"""




def many_fn_program(n: int, shape: str) -> tuple:
    """Census X1 at program scale: `n` small functions, main calls the last.
    inc:     `let mut s = a; s = s + 1; s` (the SSA path's fixed per-function
             tables cost ~105 KB of arena each: 8100 of them exhausted the
             bounded compiler's 768 MiB arena, exit 70)
    lets100: a 100-binding let chain each (10000 of them = 27 MB of source: the
             lexer exhausted the arena on the token stream, exit 70, before the
             stack budget could refuse it)."""
    L, v = [], 0
    for j in range(n):
        if shape == "inc":
            L.append(f"fn h{j}(a: i64) -> i64 {{\n    let mut s: i64 = a;\n    s = s + 1;\n    s\n}}")
            v = 2
        elif shape == "lets100":
            body = "".join(f"    let x{i}: i64 = x{i - 1} + {i % 7};\n" for i in range(1, 100))
            L.append(f"fn h{j}(a: i64) -> i64 {{\n    let x0: i64 = a;\n{body}    x99\n}}")
            v = 1 + sum(i % 7 for i in range(1, 100))
        else:
            raise ValueError(shape)
    L.append(f"fn main() -> i64 {{ h{n - 1}(1) % 251 }}")
    return "\n".join(L) + "\n", (v % 251, b"")


def long_token_program(kind: str, n: int) -> tuple:
    """Census X1, one giant TOKEN: a string literal, identifier, digit run or
    comment line of `n` bytes. The lexer walked each one stack frame per byte
    (480 B per byte for a string literal: SIGSEGV at ~839 KB in the bounded
    profile) and the token-count stack budget cannot see a single token."""
    head = "fn f(a: i64) -> i64 {\n"
    tail = "fn main() -> i64 { f(1) % 251 }\n"
    if kind == "str":
        body = f'    let s: i64 = __mind_write(1, "{"A" * n}", 0, 0 - 1);\n    s + 1\n}}\n'
        return head + body + tail, (1, b"")
    if kind == "ident":
        name = "x" * n
        return head + f"    let {name}: i64 = a + 1;\n    {name}\n}}\n" + tail, (2, b"")
    if kind == "digits":
        return head + f"    a + {'0' * n}7\n}}\n" + tail, (8, b"")
    if kind == "comment":
        return head + f"    // {'c' * n}\n    a + 2\n}}\n" + tail, (3, b"")
    raise ValueError(kind)


def balanced_sum_program(depth: int) -> tuple:
    """One expression statement: a balanced `+` tree of 2**depth `(s + 1)`
    leaves. Its nesting depth is only `depth`, so the nesting budget admits it,
    but every node costs the expression emitter arena (~2..5 KB each)."""
    def tree(d: int) -> str:
        return "(s + 1)" if d == 0 else f"({tree(d - 1)} + {tree(d - 1)})"
    src = ("fn f(a: i64) -> i64 {\n    let mut s: i64 = a;\n    s = " + tree(depth)
           + ";\n    s\n}\nfn main() -> i64 { f(1) % 251 }\n")
    return src, ((2 << depth) % 251, b"")


def capacity_cases():
    """(label, (source, expected), may_refuse, seeded): whole-program capacity in
    the bounded profile past the per-token stack budget: many functions, one
    giant token, one wide expression. Each must compile and run to the Python
    value or be refused cleanly (rc 3 with a diagnostic, no ELF): never a
    signal, never the arena trap (exit 70)."""
    return [
        ("8100 small functions (SSA tables, ~105 KB of arena each)", many_fn_program(8100, "inc"), True, True),
        ("10000 functions x 100 lets (27 MB)", many_fn_program(10000, "lets100"), True, True),
        ("1 MB string literal", long_token_program("str", 1000000), True, True),
        ("1 MB string literal, unseeded image", long_token_program("str", 1000000), True, False),
        ("4 MB identifier", long_token_program("ident", 4000000), True, True),
        ("4 MB digit run", long_token_program("digits", 4000000), True, True),
        ("4 MB comment line", long_token_program("comment", 4000000), False, True),
        ("500 KB string literal (under the token-length budget)", long_token_program("str", 500000), False, True),
        ("500 KB identifier (under the token-length budget)", long_token_program("ident", 500000), False, True),
        ("balanced 65536-leaf `+` tree in one expression", balanced_sum_program(16), True, True),
        ("balanced 1024-leaf `+` tree in one expression", balanced_sum_program(10), False, True),
    ]


def default_capacity_cases():
    """(label, (source, expected), may_refuse, seeded) for the DEFAULT-profile
    compiler (1 GiB arena, 4 GiB stack): its stack budget admits ~6.7M tokens,
    so the arena binds first; the default layout traps an exhausted arena with
    ud2 (SIGILL), so the stage must refuse before it gets there."""
    return [
        ("10000 functions x 100 lets (27 MB): the lexer ran the arena out (SIGILL)",
         many_fn_program(10000, "lets100"), True, True),
        ("1 MB string literal", long_token_program("str", 1000000), True, True),
        ("balanced 65536-leaf `+` tree in one expression", balanced_sum_program(16), True, True),
    ]
