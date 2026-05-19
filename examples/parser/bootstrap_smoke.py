"""
Phase 6.5 Stage 2 — pure-MIND parser cdylib bootstrap smoke harness.

Loads libmindc_lexer.so + libmindc_parser.so via ctypes, runs:
  1. lex(fixture_buf, buf_len) -> Vec handle
  2. parse(vec_handle, buf_addr) -> i64 AST root (heap-record address)
Walks the AST heap-record tree and compares against the expected tree
documented in EXPECTED.md.

AST node heap-record layout (Option-C ABI, RFC 0005 P0e, 7×i64 = 56 bytes):
  offset  0 : kind      (ast_* tag, 1–12)
  offset  8 : span_lo   (source-byte start)
  offset 16 : span_hi   (source-byte end)
  offset 24 : child0    (i64 subnode addr, Vec addr for lists, or 0)
  offset 32 : child1    (i64 subnode addr or 0)
  offset 40 : child2    (i64 subnode addr or 0)
  offset 48 : aux       (op tag / item count / packed flags)

Variable-length child lists (block stmts, fn params, call args, program items)
ride in child0 as the Vec base address, with aux holding the logical count.

Vec heap-record layout (Option-C ABI):
  offset  0 : data_ptr  (i64) — base address of the i64 element array
  offset  8 : length    (i64) — element count
  offset 16 : capacity  (i64) — allocated capacity
"""

import ctypes
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent.resolve()
_MIND = _HERE.parent.parent  # repo root
LEXER_SO_PATH  = _HERE.parent / "lexer" / "libmindc_lexer.so"
PARSER_SO_PATH = _HERE / "libmindc_parser.so"
FIXTURE_PATH   = _HERE / "fixture.mind"

# ---------------------------------------------------------------------------
# AST kind tags (must match main.mind ast_* constants)
# ---------------------------------------------------------------------------

AST_INT_LIT  = 1
AST_IDENT    = 2
AST_BINOP    = 3
AST_CALL     = 4
AST_FN_DEF   = 5
AST_LET      = 6
AST_USE      = 7
AST_RETURN   = 8
AST_PARAM    = 9
AST_BLOCK    = 10
AST_PROGRAM  = 11
AST_PAREN    = 12

AST_NAMES = {
    1: "ast_int_lit", 2: "ast_ident",   3: "ast_binop", 4: "ast_call",
    5: "ast_fn_def",  6: "ast_let",     7: "ast_use",   8: "ast_return",
    9: "ast_param",   10: "ast_block",  11: "ast_program", 12: "ast_paren",
}

# Operator tags (aux field of ast_binop)
OP_NAMES = {1: "op_add", 2: "op_sub", 3: "op_mul", 4: "op_div",
            5: "op_lt",  6: "op_gt",  7: "op_eq"}

# ---------------------------------------------------------------------------
# Low-level heap-record reader
# ---------------------------------------------------------------------------

_Int64Ptr = ctypes.POINTER(ctypes.c_int64)


@dataclass
class AstNode:
    kind: int
    span_lo: int
    span_hi: int
    child0: int
    child1: int
    child2: int
    aux: int
    addr: int  # heap address (for debugging)

    def kind_name(self) -> str:
        return AST_NAMES.get(self.kind, f"ast_unknown({self.kind})")


def read_node(addr: int) -> AstNode:
    """Read a 7×i64 AST heap record from `addr`."""
    p = ctypes.cast(addr, _Int64Ptr)
    return AstNode(
        kind=int(p[0]),
        span_lo=int(p[1]),
        span_hi=int(p[2]),
        child0=int(p[3]),
        child1=int(p[4]),
        child2=int(p[5]),
        aux=int(p[6]),
        addr=addr,
    )


def read_child_list(data_ptr: int, count: int) -> list[int]:
    """
    Return list of AST node addresses from a child-list slot.

    `data_ptr` is the raw element-array base address stored in child0 of
    list-bearing AST nodes (program, block, fn_def, call).  It is the value
    returned by `vec_addr(acc)` in main.mind, which equals
    `__mind_load_i64(acc + 0)` — the backing-store pointer, NOT the Vec
    header address.  Reading it as a Vec header (three-field struct) would
    give garbage length / capacity values.
    """
    if data_ptr == 0 or count == 0:
        return []
    elem_p = ctypes.cast(data_ptr, _Int64Ptr)
    return [int(elem_p[i]) for i in range(count)]


# ---------------------------------------------------------------------------
# AST tree representation for comparison
# ---------------------------------------------------------------------------

@dataclass
class AstTree:
    """Structural representation of an AST node for comparison purposes."""
    kind: int
    span_lo: int       # used for idents (byte slice comparison via fixture)
    span_hi: int
    children: list     # list of AstTree  (ordered, semantic children)
    aux: int           # op tag / count / is_pub encoding

    def kind_name(self) -> str:
        return AST_NAMES.get(self.kind, f"ast_unknown({self.kind})")

    def src(self, fixture_bytes: bytes) -> str:
        return fixture_bytes[self.span_lo:self.span_hi].decode("latin-1")


def walk_ast(addr: int, fixture_bytes: bytes) -> AstTree:
    """
    Recursively walk the AST heap-record tree starting at `addr`.
    Returns an AstTree mirroring the structure from EXPECTED.md.
    """
    n = read_node(addr)

    if n.kind == AST_PROGRAM:
        # child0 = Vec addr of items, aux = items_len
        items_addrs = read_child_list(n.child0, n.aux)
        children = [walk_ast(a, fixture_bytes) for a in items_addrs]
        return AstTree(AST_PROGRAM, n.span_lo, n.span_hi, children, n.aux)

    if n.kind == AST_USE:
        # child0 = head ident addr, child1 = tail ident addr
        head = walk_ast(n.child0, fixture_bytes)
        tail = walk_ast(n.child1, fixture_bytes)
        return AstTree(AST_USE, n.span_lo, n.span_hi, [head, tail], 0)

    if n.kind == AST_FN_DEF:
        # child0 = name ident, child1 = Vec addr params, child2 = body block
        # aux = params_len * 2 + is_pub
        name = walk_ast(n.child0, fixture_bytes)
        params_len = n.aux >> 1
        is_pub = n.aux & 1
        params_addrs = read_child_list(n.child1, params_len)
        params = [walk_ast(a, fixture_bytes) for a in params_addrs]
        body = walk_ast(n.child2, fixture_bytes)
        return AstTree(AST_FN_DEF, n.span_lo, n.span_hi,
                       [name] + params + [body], n.aux)

    if n.kind == AST_PARAM:
        # child0 = name ident, child1 = type ident
        name = walk_ast(n.child0, fixture_bytes)
        ty   = walk_ast(n.child1, fixture_bytes)
        return AstTree(AST_PARAM, n.span_lo, n.span_hi, [name, ty], 0)

    if n.kind == AST_BLOCK:
        # child0 = Vec addr of stmts, aux = stmts_len
        stmts_addrs = read_child_list(n.child0, n.aux)
        children = [walk_ast(a, fixture_bytes) for a in stmts_addrs]
        return AstTree(AST_BLOCK, n.span_lo, n.span_hi, children, n.aux)

    if n.kind == AST_LET:
        # child0 = name ident, child1 = type ident, child2 = init expr
        name = walk_ast(n.child0, fixture_bytes)
        ty   = walk_ast(n.child1, fixture_bytes)
        init = walk_ast(n.child2, fixture_bytes)
        return AstTree(AST_LET, n.span_lo, n.span_hi, [name, ty, init], 0)

    if n.kind == AST_RETURN:
        # child0 = value expr
        value = walk_ast(n.child0, fixture_bytes)
        return AstTree(AST_RETURN, n.span_lo, n.span_hi, [value], 0)

    if n.kind == AST_BINOP:
        # child0 = lhs, child1 = rhs, aux = op tag
        lhs = walk_ast(n.child0, fixture_bytes)
        rhs = walk_ast(n.child1, fixture_bytes)
        return AstTree(AST_BINOP, n.span_lo, n.span_hi, [lhs, rhs], n.aux)

    if n.kind == AST_CALL:
        # child0 = callee ident, child1 = Vec addr args, aux = args_count
        callee = walk_ast(n.child0, fixture_bytes)
        args_addrs = read_child_list(n.child1, n.aux)
        args = [walk_ast(a, fixture_bytes) for a in args_addrs]
        return AstTree(AST_CALL, n.span_lo, n.span_hi, [callee] + args, n.aux)

    if n.kind == AST_IDENT:
        return AstTree(AST_IDENT, n.span_lo, n.span_hi, [], 0)

    if n.kind == AST_INT_LIT:
        return AstTree(AST_INT_LIT, n.span_lo, n.span_hi, [], 0)

    if n.kind == AST_PAREN:
        inner = walk_ast(n.child0, fixture_bytes)
        return AstTree(AST_PAREN, n.span_lo, n.span_hi, [inner], 0)

    # Unknown — leaf
    return AstTree(n.kind, n.span_lo, n.span_hi, [], n.aux)


# ---------------------------------------------------------------------------
# Expected AST (from EXPECTED.md)
#
# The fixture file has 3 top-level items:
#   1. use std.vec;
#   2. pub fn add(x: i64, y: i64) -> i64 { let z: i64 = x + y; return z; }
#   3. pub fn compute(x: i64, y: i64, z: i64) -> i64 {
#          let r: i64 = x + y * z;
#          add(r, x)
#      }
#
# Byte offsets are taken from the lexer's actual token stream on this fixture.
# (The fixture has 635 bytes; declarations start at byte 452 after the comments.)
# ---------------------------------------------------------------------------

def build_expected(fixture_bytes: bytes) -> AstTree:
    """Construct the expected AST tree from EXPECTED.md."""

    def ident(lo: int, hi: int) -> AstTree:
        return AstTree(AST_IDENT, lo, hi, [], 0)

    def binop(lhs: AstTree, rhs: AstTree, op: int, lo: int, hi: int) -> AstTree:
        return AstTree(AST_BINOP, lo, hi, [lhs, rhs], op)

    def param(n_lo: int, n_hi: int, t_lo: int, t_hi: int) -> AstTree:
        return AstTree(AST_PARAM, n_lo, t_hi, [ident(n_lo, n_hi), ident(t_lo, t_hi)], 0)

    # ── item 1: use std.vec; ────────────────────────────────────────────────
    # Token offsets (from lexer output on this fixture):
    #   tok[0] = (kw_use,  452, 455)
    #   tok[1] = (ident,   456, 459)  "std"
    #   tok[2] = (dot,     459, 460)  "."
    #   tok[3] = (ident,   460, 463)  "vec"
    #   tok[4] = (semi,    463, 464)
    use_node = AstTree(AST_USE, 452, 463,
                       [ident(456, 459), ident(460, 463)], 0)

    # ── item 2: pub fn add(x: i64, y: i64) -> i64 { ... } ──────────────────
    # Name ident at tok[7] = (ident, 473, 476) "add"
    # params:
    #   param x:i64  — name tok[9]=(477,478), type tok[11]=(480,483)
    #   param y:i64  — name tok[13]=(485,486), type tok[15]=(488,491)
    # body block tokens start at tok[19] = (lbrace, 500, 501)
    #   stmt 1: let z: i64 = x + y;
    #     name tok[21]=(510,511) "z", type tok[23]=(513,516) "i64"
    #     init: x+y  lhs tok[25]=(519,520), rhs tok[27]=(523,524)
    #   stmt 2: return z;
    #     value tok[30]=(537,538) "z"
    # block closes tok[32]=(rbrace,540,541)

    add_let_init = binop(
        ident(519, 520),   # x
        ident(523, 524),   # y
        1,                 # op_add
        519, 524,
    )
    add_let = AstTree(AST_LET, 506, 524,
                      [ident(510, 511), ident(513, 516), add_let_init], 0)
    add_return = AstTree(AST_RETURN, 530, 538,
                         [ident(537, 538)], 0)
    add_block = AstTree(AST_BLOCK, 500, 541, [add_let, add_return], 2)

    add_fn = AstTree(
        AST_FN_DEF, 470, 541,
        [
            ident(473, 476),                       # name "add"
            param(477, 478, 480, 483),             # x: i64
            param(485, 486, 488, 491),             # y: i64
            add_block,
        ],
        2 * 2 + 1,   # params_len=2, is_pub=1  → aux = 5
    )

    # ── item 3: pub fn compute(x: i64, y: i64, z: i64) -> i64 { ... } ──────
    # Name ident at tok[35]=(550,557) "compute"
    # params:
    #   param x:i64  tok[37]=(558,559), tok[39]=(561,564)
    #   param y:i64  tok[41]=(566,567), tok[43]=(569,572)
    #   param z:i64  tok[45]=(574,575), tok[47]=(577,580)
    # body block start tok[51]=(lbrace,589,590)
    #   stmt 1: let r: i64 = x + y * z;
    #     name tok[53]=(599,600) "r", type tok[55]=(602,605) "i64"
    #     init: x + (y * z)   — Pratt: * binds tighter
    #       lhs tok[57]=(608,609) "x"
    #       rhs: y*z  lhs tok[59]=(612,613), rhs tok[61]=(616,617)
    #   stmt 2: add(r, x)  — call expression
    #     callee tok[63]=(623,626) "add"
    #     args: tok[65]=(627,628) "r", tok[67]=(630,631) "x"
    # block close tok[69]=(rbrace,633,634)

    mul_node = binop(
        ident(612, 613),   # y
        ident(616, 617),   # z
        3,                 # op_mul
        612, 617,
    )
    compute_let_init = binop(
        ident(608, 609),   # x
        mul_node,
        1,                 # op_add
        608, 617,
    )
    compute_let = AstTree(AST_LET, 595, 617,
                          [ident(599, 600), ident(602, 605), compute_let_init], 0)

    call_callee = ident(623, 626)   # "add"
    call_node = AstTree(
        AST_CALL, 623, 632,
        [call_callee, ident(627, 628), ident(630, 631)],
        2,  # args_count
    )
    compute_block = AstTree(AST_BLOCK, 589, 634, [compute_let, call_node], 2)

    compute_fn = AstTree(
        AST_FN_DEF, 547, 634,
        [
            ident(550, 557),                       # name "compute"
            param(558, 559, 561, 564),             # x: i64
            param(566, 567, 569, 572),             # y: i64
            param(574, 575, 577, 580),             # z: i64
            compute_block,
        ],
        3 * 2 + 1,   # params_len=3, is_pub=1  → aux = 7
    )

    return AstTree(AST_PROGRAM, 0, 0, [use_node, add_fn, compute_fn], 3)


# ---------------------------------------------------------------------------
# Pretty printer for diff output
# ---------------------------------------------------------------------------

def _fmt_node(t: AstTree, fixture_bytes: bytes, indent: int = 0) -> str:
    pad = "  " * indent
    name = t.kind_name()
    if t.kind == AST_IDENT:
        src = fixture_bytes[t.span_lo:t.span_hi].decode("latin-1")
        return f"{pad}{name} {src!r} ({t.span_lo}:{t.span_hi})"
    if t.kind == AST_INT_LIT:
        src = fixture_bytes[t.span_lo:t.span_hi].decode("latin-1")
        return f"{pad}{name} {src!r} ({t.span_lo}:{t.span_hi})"
    if t.kind == AST_BINOP:
        op_name = OP_NAMES.get(t.aux, f"op_{t.aux}")
        lines = [f"{pad}{name} {op_name} ({t.span_lo}:{t.span_hi})"]
        for child in t.children:
            lines.append(_fmt_node(child, fixture_bytes, indent + 1))
        return "\n".join(lines)
    if t.kind == AST_FN_DEF:
        is_pub = t.aux & 1
        params_len = t.aux >> 1
        pub_str = "pub " if is_pub else ""
        lines = [f"{pad}{name} {pub_str}(params={params_len}) ({t.span_lo}:{t.span_hi})"]
        for child in t.children:
            lines.append(_fmt_node(child, fixture_bytes, indent + 1))
        return "\n".join(lines)
    if t.kind == AST_CALL:
        lines = [f"{pad}{name} (args={t.aux}) ({t.span_lo}:{t.span_hi})"]
        for child in t.children:
            lines.append(_fmt_node(child, fixture_bytes, indent + 1))
        return "\n".join(lines)
    # Generic
    lines = [f"{pad}{name} (aux={t.aux}) ({t.span_lo}:{t.span_hi})"]
    for child in t.children:
        lines.append(_fmt_node(child, fixture_bytes, indent + 1))
    return "\n".join(lines)


def print_tree(label: str, t: AstTree, fixture_bytes: bytes) -> None:
    print(f"\n{label}:")
    print(_fmt_node(t, fixture_bytes))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class DiffResult:
    identical: bool
    first_divergence_path: list[str]
    got_node: Optional[AstTree]
    exp_node: Optional[AstTree]


def compare_trees(
    got: AstTree,
    exp: AstTree,
    fixture_bytes: bytes,
    path: list[str],
) -> DiffResult:
    """
    Recursively compare got vs exp.  Returns the first divergence found,
    or a DiffResult(identical=True) if the trees match.
    """
    # Check kind
    if got.kind != exp.kind:
        return DiffResult(False, path, got, exp)

    # For leaf nodes, compare span (which encodes the identifier/literal bytes).
    if got.kind in (AST_IDENT, AST_INT_LIT):
        if got.span_lo != exp.span_lo or got.span_hi != exp.span_hi:
            return DiffResult(False, path, got, exp)
        return DiffResult(True, [], None, None)

    # For binop, compare aux (op tag).
    if got.kind == AST_BINOP and got.aux != exp.aux:
        return DiffResult(False, path + ["aux/op"], got, exp)

    # For fn_def, compare aux (params_len * 2 + is_pub).
    if got.kind == AST_FN_DEF and got.aux != exp.aux:
        return DiffResult(False, path + ["aux/is_pub+params_len"], got, exp)

    # For call, compare aux (args count).
    if got.kind == AST_CALL and got.aux != exp.aux:
        return DiffResult(False, path + ["aux/args_count"], got, exp)

    # Check child count.
    if len(got.children) != len(exp.children):
        return DiffResult(
            False,
            path + [f"children_len({len(got.children)} vs {len(exp.children)})"],
            got, exp,
        )

    # Recurse.
    for i, (g_child, e_child) in enumerate(zip(got.children, exp.children)):
        child_path = path + [f"child[{i}]"]
        result = compare_trees(g_child, e_child, fixture_bytes, child_path)
        if not result.identical:
            return result

    return DiffResult(True, [], None, None)


# ---------------------------------------------------------------------------
# Main smoke runner
# ---------------------------------------------------------------------------

def run_smoke(
    lexer_so: pathlib.Path,
    parser_so: pathlib.Path,
    fixture_path: pathlib.Path,
) -> bool:
    """
    Load the .so files, run lex() then parse(), walk the AST, compare.
    Returns True on PASS.
    """
    for p in (lexer_so, parser_so, fixture_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            return False

    fixture_bytes = fixture_path.read_bytes()
    buf_len = len(fixture_bytes)
    print(f"Fixture: {fixture_path} ({buf_len} bytes)")

    # Load both shared libraries.  RTLD_LOCAL avoids symbol collisions between
    # the two .so files (each has its own copy of __mind_alloc / vec_new etc.).
    # Both link to the same libc malloc so heap pointers cross correctly.
    lexer_lib  = ctypes.CDLL(str(lexer_so),  mode=ctypes.RTLD_LOCAL)
    parser_lib = ctypes.CDLL(str(parser_so), mode=ctypes.RTLD_LOCAL)
    print(f"Loaded:  {lexer_so.name}")
    print(f"Loaded:  {parser_so.name}")

    # Step 1: tokenise the fixture.
    buf = ctypes.create_string_buffer(fixture_bytes)
    buf_addr = ctypes.cast(buf, ctypes.c_void_p).value
    assert buf_addr is not None

    lexer_lib.lex.restype  = ctypes.c_int64
    lexer_lib.lex.argtypes = [ctypes.c_int64, ctypes.c_int64]
    vec_handle = lexer_lib.lex(ctypes.c_int64(buf_addr), ctypes.c_int64(buf_len))
    print(f"\nlex() Vec handle: 0x{vec_handle:x}")
    if vec_handle == 0:
        print("ERROR: lex() returned null — alloc failure.")
        return False

    # Read Vec header to report token count.
    hdr = ctypes.cast(vec_handle, ctypes.POINTER(ctypes.c_int64))
    n_elems = int(hdr[1])
    n_tokens = n_elems // 3
    print(f"Token stream: {n_tokens} tokens ({n_elems} i64 elements)")

    # Step 2: parse the token stream.
    parser_lib.parse.restype  = ctypes.c_int64
    parser_lib.parse.argtypes = [ctypes.c_int64, ctypes.c_int64]
    ast_root_addr = parser_lib.parse(
        ctypes.c_int64(vec_handle), ctypes.c_int64(buf_addr)
    )
    print(f"parse() AST root: 0x{ast_root_addr:x}")
    if ast_root_addr == 0:
        print("ERROR: parse() returned null — alloc failure.")
        return False

    # Step 3: walk the AST tree.
    try:
        got_tree = walk_ast(ast_root_addr, fixture_bytes)
    except Exception as exc:
        print(f"ERROR walking AST: {exc}")
        return False

    print(f"AST root: kind={got_tree.kind_name()}, items={got_tree.aux}")

    # Step 4: build and compare against expected.
    exp_tree = build_expected(fixture_bytes)

    diff = compare_trees(got_tree, exp_tree, fixture_bytes, ["root"])

    if diff.identical:
        print("\nAST comparison vs EXPECTED.md: PASS")
        _print_pass_summary(got_tree, fixture_bytes)
        return True

    # Report first divergence.
    print("\nAST comparison vs EXPECTED.md: MISMATCH")
    print(f"First divergence at path: {' > '.join(diff.first_divergence_path)}")
    if diff.got_node is not None:
        print("Got:")
        print(_fmt_node(diff.got_node, fixture_bytes, indent=2))
    if diff.exp_node is not None:
        print("Expected:")
        print(_fmt_node(diff.exp_node, fixture_bytes, indent=2))

    print_tree("Got AST (full)", got_tree, fixture_bytes)
    print_tree("Expected AST (full)", exp_tree, fixture_bytes)
    return False


def _print_pass_summary(tree: AstTree, fixture_bytes: bytes) -> None:
    """Print a concise pre-order walk of the AST confirming key nodes."""
    print("\nPre-order walk (key nodes):")
    _walk_print(tree, fixture_bytes, 0)


def _walk_print(t: AstTree, fb: bytes, depth: int, max_depth: int = 8) -> None:
    if depth > max_depth:
        return
    pad = "  " * depth
    if t.kind == AST_IDENT:
        print(f"{pad}Ident {fb[t.span_lo:t.span_hi].decode('latin-1')!r}")
    elif t.kind == AST_BINOP:
        op = OP_NAMES.get(t.aux, f"op_{t.aux}")
        print(f"{pad}BinOp {op}")
    elif t.kind == AST_CALL:
        print(f"{pad}Call (args={t.aux})")
    elif t.kind == AST_FN_DEF:
        params_len = t.aux >> 1
        is_pub = t.aux & 1
        pub_str = "pub " if is_pub else ""
        print(f"{pad}FnDef {pub_str}params={params_len}")
    elif t.kind == AST_BLOCK:
        print(f"{pad}Block stmts={t.aux}")
    else:
        print(f"{pad}{t.kind_name()} aux={t.aux}")
    for child in t.children:
        _walk_print(child, fb, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = run_smoke(LEXER_SO_PATH, PARSER_SO_PATH, FIXTURE_PATH)
    if passed:
        print("\nRESULT: PASS — AST tree matches EXPECTED.md")
    else:
        print("\nRESULT: MISMATCH — see first divergence above")
    sys.exit(0 if passed else 1)
