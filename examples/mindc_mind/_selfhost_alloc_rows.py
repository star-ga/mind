"""Rows of the allocator-class smoke (self_host_native_alloc_bounded_smoke.py): the default-profile
programs that have no fixture file (generated_row) and the planted changes of its --mutant mode
(MUTANTS, and MUTANT_ROWS: the rows each change must turn red). Imported by that smoke; not a gate.
"""

import pathlib

# --- generated default-profile rows (no fixture file: a formatter would drop a CR, and a repeated literal is built here) ---

def _extract_fns(src: str, names) -> str:
    """Top-level `pub fn NAME(` ... first column-0 `}` for each name."""
    lines, out = src.splitlines(), []
    for name in names:
        hits = [i for i, row in enumerate(lines) if row.startswith(f"pub fn {name}(")]
        if len(hits) != 1:
            raise KeyError(f"{name}: {len(hits)} definitions")
        end = next(j for j in range(hits[0], len(lines)) if lines[j] == "}")
        out.append("\n".join(lines[hits[0]:end + 1]))
    return "\n".join(out) + "\n"


_WALK_AST = (
    "ast_kind", "ast_span_lo", "ast_span_hi", "ast_child0", "ast_child1", "ast_child2", "ast_aux",
    "ast_int_lit", "ast_ident", "ast_binop", "ast_call", "ast_fn_def", "ast_let", "ast_return",
    "ast_block", "ast_paren", "ast_struct_lit", "ast_neg", "ast_not", "ast_if", "ast_field",
    "ast_while", "ast_assign", "ast_break", "ast_continue", "ast_cast", "ast_array_lit",
    "ast_index", "ast_index_assign", "ast_enum_ctor", "ast_addr_of", "ast_deref",
    "ast_deref_assign", "ast_field_assign",
)
_WALK_SEL = (
    "nbas_is_free", "nbas_enqueue", "nbas_call_at", "nbas_node", "nbas_list", "nbas_block",
    "nbas_if", "nbas_expr", "nbas_stmt", "nbas_fn", "nbas_step_a", "nbas_step_b",
    "nbas_phase_a", "nbas_qlen", "nbas_phase_b", "nbas_zero", "nbas_walk", "nbas_phase_b_if",
)
# The driver's own image is BUMP: the free sits in the walked tree, not in a loop of this source.
# Names in the buffer: main / step1 / __mind_free. Dirt of word 1 covers the mark nbas_walk allocates;
# exit 33 means the walk zeroed that mark and found the free, 4 means the mark stayed queued, 1 means
# the arena handed back clean memory, 5 means the name lookup missed step1.
_WALK_DRIVER = (
    "fn wm_byte(buf: i64, i: i64) -> i64 {\n    return __mind_load_i64(buf + i) & 255;\n}\nfn wm_eq(buf: i64, a: i64, b: i64, n: i64) -> i64 {\n"
    "    let mut k: i64 = 0;\n    while k < n {\n        if wm_byte(buf, a + k) != wm_byte(buf, b + k) {\n            return 0;\n        }\n"
    "        k = k + 1;\n    }\n    return 1;\n}\nfn nb_last_def_idx(buf: i64, fns: i64, n_fns: i64, lo: i64, hi: i64) -> i64 {\n    let mut k: i64 = 0;\n"
    "    let mut found: i64 = 0 - 1;\n    while k < n_fns {\n        let name: i64 = ast_child0(__mind_load_i64(fns + k * 8));\n"
    "        let nlo: i64 = ast_span_lo(name);\n        let nhi: i64 = ast_span_hi(name);\n        if nhi - nlo == hi - lo {\n"
    "            if wm_eq(buf, nlo, lo, hi - lo) == 1 {\n                found = k;\n            }\n        }\n        k = k + 1;\n    }\n"
    "    return found;\n}\nfn wm_node(k: i64, c0: i64, c1: i64, c2: i64, aux: i64, lo: i64, hi: i64) -> i64 {\n    let n: i64 = __mind_alloc(64);\n"
    "    __mind_store_i64(n + 0, k);\n    __mind_store_i64(n + 8, lo);\n    __mind_store_i64(n + 16, hi);\n    __mind_store_i64(n + 24, c0);\n"
    "    __mind_store_i64(n + 32, c1);\n    __mind_store_i64(n + 40, c2);\n    __mind_store_i64(n + 48, aux);\n    __mind_store_i64(n + 56, 0);\n"
    "    return n;\n}\nfn wm_fill(addr: i64, n: i64, v: i64) -> i64 {\n    let mut i: i64 = 0;\n    while i < n {\n"
    "        __mind_store_i64(addr + i * 8, v);\n        i = i + 1;\n    }\n    return 0;\n}\nfn main() -> i64 {\n    let b: i64 = __mind_alloc(32);\n"
    "    __mind_store_i64(b + 0, 1852399981);\n    __mind_store_i64(b + 8, 212339094643);\n    __mind_store_i64(b + 16, 7376725140034117471);\n"
    "    __mind_store_i64(b + 24, 6645106);\n    let cond: i64 = wm_node(ast_int_lit(), 0, 0, 0, 0, 0, 0);\n"
    "    let step_id: i64 = wm_node(ast_ident(), 0, 0, 0, 0, 8, 13);\n    let call_step: i64 = wm_node(ast_call(), step_id, 0, 0, 0, 0, 0);\n"
    "    let one_step: i64 = __mind_alloc(8);\n    __mind_store_i64(one_step, call_step);\n"
    "    let wbody: i64 = wm_node(ast_block(), one_step, 0, 0, 1, 0, 0);\n    let wh: i64 = wm_node(ast_while(), cond, wbody, 0, 0, 0, 0);\n"
    "    let one_wh: i64 = __mind_alloc(8);\n    __mind_store_i64(one_wh, wh);\n    let main_body: i64 = wm_node(ast_block(), one_wh, 0, 0, 1, 0, 0);\n"
    "    let main_id: i64 = wm_node(ast_ident(), 0, 0, 0, 0, 0, 4);\n    let main_fn: i64 = wm_node(ast_fn_def(), main_id, 0, main_body, 0, 0, 0);\n"
    "    let free_id: i64 = wm_node(ast_ident(), 0, 0, 0, 0, 16, 27);\n    let call_free: i64 = wm_node(ast_call(), free_id, 0, 0, 0, 0, 0);\n"
    "    let ret: i64 = wm_node(ast_return(), call_free, 0, 0, 0, 0, 0);\n    let one_ret: i64 = __mind_alloc(8);\n    __mind_store_i64(one_ret, ret);\n"
    "    let step_body: i64 = wm_node(ast_block(), one_ret, 0, 0, 1, 0, 0);\n    let step_id2: i64 = wm_node(ast_ident(), 0, 0, 0, 0, 8, 13);\n"
    "    let step_fn: i64 = wm_node(ast_fn_def(), step_id2, 0, step_body, 0, 0, 0);\n    let ordered: i64 = __mind_alloc(16);\n"
    "    __mind_store_i64(ordered, main_fn);\n    __mind_store_i64(ordered + 8, step_fn);\n    let idx: i64 = nb_last_def_idx(b, ordered, 2, 8, 13);\n"
    "    if idx != 1 {\n        return 5;\n    }\n    let pad: i64 = __mind_alloc(8);\n    let filled: i64 = wm_fill(pad + 8, 200, 1);\n"
    "    let probe: i64 = __mind_alloc(8);\n    if __mind_load_i64(probe) != 1 {\n        return 1;\n    }\n    let got: i64 = nbas_walk(b, ordered, 2);\n"
    "    if got != 1 {\n        return 4;\n    }\n    return 33 + filled;\n}\n"
)


def _walk_mark_source(src_dir: pathlib.Path) -> bytes:
    ast = (src_dir / "nb_ast.mind").read_text()
    sel = (src_dir / "nb_alloc_select.mind").read_text()
    return (_extract_fns(ast, _WALK_AST) + _extract_fns(sel, _WALK_SEL) + _WALK_DRIVER).encode()


_CELL_ZERO_DRIVER = """
fn cz_dirty(addr: i64, n: i64) -> i64 {
    let mut k: i64 = 0;
    while k < n {
        __mind_store_i64(addr + k * 8, 90);
        k = k + 1;
    }
    return 0;
}
fn main() -> i64 {
    let a: i64 = __mind_alloc(8);
    let d1: i64 = cz_dirty(a + 8, 24);
    let raw: i64 = __mind_alloc(144);
    if __mind_load_i64(raw + 136) != 90 {
        return 1;
    }
    let d2: i64 = cz_dirty(raw + 144, 24);
    let c: i64 = nbwm_cell();
    if c < raw + 144 {
        return 2;
    }
    if c > raw + 160 {
        return 2;
    }
    if __mind_load_i64(c + 136) != 0 {
        return 3;
    }
    if __mind_load_i64(c + 128) != 0 {
        return 4;
    }
    return 33;
}
"""
# A store through an allocated pointer. The free is the value of the store, so only the
# dereference-assignment arm of the walk sees it. 1,100 x 1 MB.
_DEREF_ASSIGN_ROW = (
    b'fn main() -> i64 {\n    let cell: i64 = __mind_alloc(8);\n    let mut i: i64 = 0;\n'
    b'    let mut acc: i64 = 0;\n    while i < 1100 {\n        let p: i64 = __mind_alloc(1000000);\n'
    b'        __mind_store_i64(p, i);\n        acc = acc + __mind_load_i64(p) - i;\n        *cell = __mind_free(p);\n'
    b'        i = i + 1;\n    }\n    return acc + 33;\n'
    b'}\n'
)
_DEREF_ASSIGN_CALLEE_ROW = (
    b'fn main() -> i64 {\n    let cell: i64 = __mind_alloc(8);\n    let mut i: i64 = 0;\n'
    b'    let mut acc: i64 = 0;\n    while i < 1100 {\n        let p: i64 = __mind_alloc(1000000);\n'
    b'        __mind_store_i64(p, i);\n        acc = acc + __mind_load_i64(p) - i;\n        let z: i64 = step(cell, p);\n'
    b'        i = i + 1;\n    }\n    return acc + 33;\n'
    b'}\nfn step(cell: i64, p: i64) -> i64 {\n    *cell = __mind_free(p);\n'
    b'    return 0;\n}\n'
)
# A free that is the value of a store through a reference, in a while loop (1,100 x 1 MB). The native
# lowering refuses the prefix `&` (row licm_addr of native_lane_value_shapes_smoke.py): no image. When
# the program compiles, this REFUSED row turns red and must become a RECLAIM row (exit status 33).
_REF_STORE_ROW = (
    b'fn main() -> i64 {\n    let mut i: i64 = 0;\n    let mut x: i64 = 5;\n    let r: i64 = &x;\n'
    b'    while i < 1100 {\n        let p: i64 = __mind_alloc(1000000);\n        *r = __mind_free(p);\n'
    b'        i = i + 1;\n    }\n    return x + 33;\n}\n'
)
_BIG_FREE = (
    "    let mut i: i64 = 0;\n    let mut acc: i64 = 0;\n    while i < 1100 {\n"
    "        let p: i64 = __mind_alloc(1000000);\n        __mind_store_i64(p, i);\n"
    "        acc = acc + __mind_load_i64(p) - i;\n        let z: i64 = __mind_free(p);\n        i = i + 1;\n    }\n"
    "    return acc + 33;\n}\n"
)
_IDLE_FREE = (
    "    let mut q: i64 = 0;\n    while q != 0 {\n        let nx: i64 = __mind_load_i64(q);\n"
    "        let z: i64 = __mind_free(q);\n        q = nx;\n    }\n    return sink + 33;\n}\n"
)


def _zeros(n: int) -> str:
    return ", ".join(["0"] * n)


def _exhaust_array() -> bytes:
    return (
        "fn main() -> i64 {\n    let mut i: i64 = 0;\n    let mut sink: i64 = 0;\n    while i < 40000 {\n"
        f"        let a: [i64; 4096] = [{_zeros(4096)}];\n        sink = sink + a[0];\n        i = i + 1;\n    }}\n" + _IDLE_FREE
    ).encode()


def _exhaust_field() -> bytes:
    return (
        "struct R {\n    xs: [i64; 4096],\n}\nfn main() -> i64 {\n"
        f"    let r: R = R {{ xs: [{_zeros(4096)}] }};\n"
        "    let mut i: i64 = 0;\n    let mut sink: i64 = 0;\n    while i < 40000 {\n"
        "        let a: [i64; 4096] = r.xs;\n        sink = sink + a[0];\n        i = i + 1;\n    }\n" + _IDLE_FREE
    ).encode()


def _exhaust_struct() -> bytes:
    fields = ", ".join(f"f{i}: i64" for i in range(16))
    inits = ", ".join(f"f{i}: 0" for i in range(16))
    idle = _IDLE_FREE.replace("sink", "acc")
    return (
        f"struct S {{ {fields} }}\nfn main() -> i64 {{\n    let mut i: i64 = 0;\n    let mut acc: i64 = 0;\n"
        "    while i < 10000000 {\n"
        f"        let s: S = S {{ {inits} }};\n        acc = acc + s.f0;\n        i = i + 1;\n    }}\n" + idle
    ).encode()


def generated_row(name: str, src_dir: pathlib.Path) -> bytes:
    """Bytes of a default-profile row that has no fixture file. `src_dir` is the compiler source
    the row extracts (the mutant copy, when one is planted)."""
    if name == "cell_zero":
        return (_extract_fns((src_dir / "nb_branch_memo.mind").read_text(), ("nbwm_cell",)) + _CELL_ZERO_DRIVER).encode()
    if name == "arm_deref_assign":
        return _DEREF_ASSIGN_ROW
    if name == "arm_deref_assign_callee":
        return _DEREF_ASSIGN_CALLEE_ROW
    if name == "ref_store_free":
        return _REF_STORE_ROW
    if name == "realloc_blocked":
        return (
            "fn main() -> i64 {\n    let mut i: i64 = 0;\n    let mut acc: i64 = 0;\n    while i < 1100 {\n"
            "        let p: i64 = __mind_alloc(1000000);\n        __mind_store_i64(p, i);\n"
            "        let live: i64 = __mind_alloc(16);\n        __mind_store_i64(live, 1);\n"
            "        let q: i64 = __mind_realloc(p, 1004096);\n        acc = acc + __mind_load_i64(q) - i;\n"
            "        let z1: i64 = __mind_free(q);\n        let z2: i64 = __mind_free(live);\n        i = i + 1;\n"
            "    }\n    return acc + 33;\n}\n"
        ).encode()
    if name == "ctl_bumpy":
        return ("// mind: native_alloc = bumpy\nfn main() -> i64 {\n" + _BIG_FREE).encode()
    if name == "ctl_bumx":
        return ("// mind: native_alloc = bumx\nfn main() -> i64 {\n" + _BIG_FREE).encode()
    if name == "ctl_after_code":
        return (
            "fn helper() -> i64 {\n    return 1;\n}\n// mind: native_alloc = bump\nfn main() -> i64 {\n"
            "    let mut i: i64 = 0;\n    let mut acc: i64 = 0;\n    let h: i64 = helper();\n    while i < 1100 {\n"
            "        let p: i64 = __mind_alloc(1000000);\n        __mind_store_i64(p, i);\n"
            "        acc = acc + __mind_load_i64(p) - i;\n        let z: i64 = __mind_free(p);\n        i = i + 1;\n"
            "    }\n    return acc + h + 32;\n}\n"
        ).encode()
    if name == "ctl_crlf":
        return (
            "// mind: native_alloc = bump\r\nfn main() -> i64 {\n    let mut i: i64 = 0;\n    let mut acc: i64 = 0;\n"
            "    while i < 100 {\n        let p: i64 = __mind_alloc(1000);\n        __mind_store_i64(p, i);\n"
            "        acc = acc + __mind_load_i64(p) - i;\n        let z: i64 = __mind_free(p);\n        i = i + 1;\n"
            "    }\n    return acc + 33;\n}\n"
        ).encode()
    if name == "mutrec_nofree":
        return (
            "fn f(n: i64) -> i64 {\n    if n == 0 {\n        return 0;\n    }\n    return g(n);\n}\n"
            "fn g(n: i64) -> i64 {\n    return f(n - 1) + 1;\n}\nfn main() -> i64 {\n"
            "    let mut i: i64 = 0;\n    let mut a: i64 = 0;\n    while i < 3 {\n        a = a + f(4);\n        i = i + 1;\n"
            "    }\n    return a + 21;\n}\n"
        ).encode()
    if name == "exhaust_array":
        return _exhaust_array()
    if name == "exhaust_field":
        return _exhaust_field()
    if name == "exhaust_struct":
        return _exhaust_struct()
    if name == "walk_mark":
        return _walk_mark_source(src_dir)
    if name in ("bump_overshoot", "bump_exact_fill"):
        # the 1 GiB arena less the 16-byte block header; one byte more does not fit
        size = (1 << 30) - 16 + (name == "bump_overshoot")
        more = "    let q: i64 = __mind_alloc(8);\n    if q == 0 {\n        return 2;\n    }\n" if name == "bump_exact_fill" else ""
        return (f"fn main() -> i64 {{\n    let p: i64 = __mind_alloc({size});\n{more}"
                "    if p == 0 {\n        return 1;\n    }\n    return 33;\n}\n").encode()
    raise KeyError(name)


# --- planted changes of the allocator-class smoke (self_host_native_alloc_bounded_smoke.py --mutant) ---
# name -> (file, old text, new text): one planted change of the compiler sources. Each must turn a row red.
MUTANTS = {
    # the free of a RECLAIM image releases nothing
    "free_noop": ("main.mind", "            let e1: String = nb_emit_free_explicit(code, nb_slot_disp(a0));\n            return nb_store_rax(e1, nb_slot_disp(dst_id));",
                  "            let e1: String = nb_emit_free_xor(code);\n            return nb_store_rax(e1, nb_slot_disp(dst_id));"),
    # the walk stops after the callees queued by phase A: a callee two calls deep is missed
    "one_round": ("nb_alloc_select.mind", "    let mut r: i64 = 0;\n    while q < nbas_qlen(cell) {",
                  "    let mut r: i64 = 0;\n    let n0: i64 = nbas_qlen(cell);\n    while q < n0 {"),
    # the while condition is walked at depth 0
    "cond_depth0": ("nb_alloc_select.mind", "        if nbas_expr(cell, ast_child0(node), 1) == 1 {", "        if nbas_expr(cell, ast_child0(node), 0) == 1 {"),
    # the bare expression statement is not walked
    "expr_stmt_off": ("nb_alloc_select.mind", "    // A bare expression statement (`__mind_free(p);`, `f(x);`) and an `if`.\n    nbas_expr(cell, node, lp)\n}",
                      "    // A bare expression statement (`__mind_free(p);`, `f(x);`) and an `if`.\n    0\n}"),
    # the opt-out line is ignored
    "directive_ignored": ("nb_alloc_select.mind", "    if nbas_directive(buf, user_lo, src_len) == 1 {", "    if nbas_directive(buf, user_lo, src_len) == 7 {"),
    # the default layout keeps an overflow region: a full arena no longer exits 70
    "cap_removed": ("main.mind", "    if nb_layout_explicit() == 1 {\n        return nb_ovf_base();\n    }\n    nb_arena_addr()\n}", "    nb_ovf_base()\n}"),
    # the class is never RECLAIM / always RECLAIM
    "select_never": ("nb_alloc_select.mind", "    if nbas_walk(buf, ordered, n_fns) == 1 {", "    if nbas_walk(buf, ordered, n_fns) == 7 {"),
    "select_always": ("nb_alloc_select.mind", "    if ctx != 0 {\n        return 0;\n    }\n    if nbas_directive", "    if ctx != 0 {\n        return 0;\n    }\n    return nbas_flag_reclaim();\n    if nbas_directive"),
    # the alloc, realloc and free decisions and the runtime prefix ask for an explicit layout only
    "alloc_dec": ("main.mind", "    if nb_alloc_freelist(am) == 1 {\n        return nb_emit_alloc_explicit(code, bytes_disp);",
                  "    if nb_layout_explicit() == 1 {\n        return nb_emit_alloc_explicit(code, bytes_disp);"),
    "realloc_dec": ("main.mind", "        if nb_alloc_freelist(am) == 1 {\n            let e1: String = nb_emit_realloc_explicit",
                    "        if nb_layout_explicit() == 1 {\n            let e1: String = nb_emit_realloc_explicit"),
    "free_dec": ("main.mind", "        if nb_alloc_freelist(am) == 1 {\n            let a0: i64 = __mind_load_i64(ids + 0 * 8);\n            let e1: String = nb_emit_free_explicit",
                 "        if nb_layout_explicit() == 1 {\n            let a0: i64 = __mind_load_i64(ids + 0 * 8);\n            let e1: String = nb_emit_free_explicit"),
    "rt_prefix": ("main.mind", "pub fn nb_rt_prefix(am: i64) -> String {\n    if nb_alloc_freelist(am) == 1 {",
                  "pub fn nb_rt_prefix(am: i64) -> String {\n    if nb_layout_explicit() == 1 {"),
    # the while body is walked at depth 0
    "body_depth0": ("nb_alloc_select.mind", "        return nbas_block(cell, ast_child1(node), 1);", "        return nbas_block(cell, ast_child1(node), 0);"),
    # the enum constructor payload is not walked
    "enum_ctor_off": ("nb_alloc_select.mind", "        if ast_aux(node) == 1 {\n            return nbas_expr(cell, ast_child1(node), lp);",
                      "        if ast_aux(node) == 7 {\n            return nbas_expr(cell, ast_child1(node), lp);"),
    # the store through a pointer is not walked, so a free that is the stored value is missed
    "deref_off": ("nb_alloc_select.mind",
                  "    if k == ast_deref_assign() {\n        return nbas_expr(cell, ast_child2(node), lp);\n    }",
                  "    if k == ast_deref_assign() {\n        return 0;\n    }"),
    # the lowering cell stops zeroing its allocator class word
    "zero136": ("nb_branch_memo.mind", "    __mind_store_i64(c + 136, 0);\n", ""),
    # the queue mark never reads as already-queued, so a mutually recursive pair with no free is queued forever
    "queue_mark": ("nb_alloc_select.mind", "    if __mind_load_i64(mark + idx * 8) == 1 {\n",
                   "    if __mind_load_i64(mark + idx * 8) == 2 {\n"),
    # the 28-byte length check is skipped, so a longer near-miss still matches
    "dir_len": ("nb_alloc_select.mind", "    if hi - lo != 28 {\n", "    if hi - lo != hi - lo {\n"),
    # the last-word check never fails, so a 28-byte line with a different tail matches
    "dir_word": ("nb_alloc_select.mind",
                 "    if (__mind_load_i64(buf + lo + 24) & 4294967295) != 1886221666 {\n",
                 "    if (__mind_load_i64(buf + lo + 24) & 4294967295) != (__mind_load_i64(buf + lo + 24) & 4294967295) {\n"),
    # a carriage return stays in the line, so the CRLF spelling of the directive is not 28 bytes
    "dir_cr": ("nb_alloc_select.mind", "    if __mind_load_i8(buf + eol - 1) == 13 {\n",
               "    if __mind_load_i8(buf + eol - 1) == 7 {\n"),
    # the leading block does not stop at the first line of code, so a later column-0 directive counts
    "dir_block": ("nb_alloc_select.mind", "    if kind != 1 {\n", "    if kind == 2 {\n"),
    # the field-array copy, the array literal and the struct literal allocate with class 0
    "field_am0": ("main.mind",
                  "            return nb_materialize_field_array_ctx(s_recv, ptr_id, off, n, rep, __mind_load_i64(lcell + 120), __mind_load_i64(lcell + 136));\n",
                  "            return nb_materialize_field_array_ctx(s_recv, ptr_id, off, n, rep, __mind_load_i64(lcell + 120), 0);\n"),
    "array_am0": ("main.mind",
                 "        let aal1: String = nb_emit_alloc_am(am3, nb_slot_disp(amul_id), __mind_load_i64(lcell + 120), __mind_load_i64(lcell + 136));\n",
                 "        let aal1: String = nb_emit_alloc_am(am3, nb_slot_disp(amul_id), __mind_load_i64(lcell + 120), 0);\n"),
    "struct_am0": ("main.mind",
                  "        let al1: String = nb_emit_alloc_am(m3, nb_slot_disp(mul_id), __mind_load_i64(lcell + 120), __mind_load_i64(lcell + 136));\n",
                  "        let al1: String = nb_emit_alloc_am(m3, nb_slot_disp(mul_id), __mind_load_i64(lcell + 120), 0);\n"),
    # the walk does not zero its mark array
    "walk_unzeroed": ("nb_alloc_select.mind", "    let _zm: i64 = nbas_zero(mark, n_fns + 1);\n",
                     "    let _zm: i64 = mark - mark;\n"),
}


# The rows that each planted change must turn red. A red verdict of any other row does not kill the
# mutant: a row that fails for its own reason must not hide a mutant that survives.
MUTANT_ROWS = {
    "free_noop": {"arm_while_body", "arm_let"},
    "one_round": {"calls2", "calls3"},
    "cond_depth0": {"arm_while_cond", "arm_while_cond_only"},
    "expr_stmt_off": {"arm_expr_stmt"},
    "directive_ignored": {"ctl_directive", "ctl_both_directives", "selfcompile"},
    "cap_removed": {"exhaust_live"},
    "select_never": {"arm_while_body", "arm_let"},
    "select_always": {"ctl_no_free", "ctl_free_outside", "ctl_for_end_bound", "selfcompile"},
    "alloc_dec": {"arm_struct_lit", "arm_array_lit", "arm_index", "arm_enum_ctor"},
    "realloc_dec": {"realloc_blocked"},
    # a callee already marked queued is queued again: mutual recursion with no free grows the queue without bound
    "queue_mark": {"mutrec_nofree"},
    # the directive matches a longer near-miss, a wrong last word, a CR that was not stripped, or a line after code
    "dir_len": {"ctl_bumpy"},
    "dir_word": {"ctl_bumx"},
    "dir_cr": {"ctl_crlf"},
    "dir_block": {"ctl_after_code"},
    # the literal alloc site passes class 0, so a full arena stops with SIGILL instead of exit 70
    "field_am0": {"exhaust_field"},
    "array_am0": {"exhaust_array"},
    "struct_am0": {"exhaust_struct"},
    # the walk's mark array is left dirty: a word of 1 hides the callee that frees
    "walk_unzeroed": {"walk_mark"},
    "free_dec": {"arm_while_body", "arm_let"},
    "rt_prefix": {"arm_while_body", "arm_let"},
    "body_depth0": {"arm_while_body", "arm_let"},
    "enum_ctor_off": {"arm_enum_ctor"},
    "deref_off": {"arm_deref_assign", "arm_deref_assign_callee"},
    "zero136": {"cell_zero"},
}
assert set(MUTANT_ROWS) == set(MUTANTS)
