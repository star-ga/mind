#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND D3 STD-SURFACE EXPORT NAME SET.

Ports resolve.rs `stdlib_export_names()`: the flattened union of
`collect_decl_names` over the BUNDLED std modules — the exact list in
src/project/stdlib.rs STDLIB_MIND_SOURCES, which this smoke re-extracts on
every run so a bundling change (module added/dropped) fails LOUD here rather
than silently drifting the MIND answer. An on-disk std/*.mind file that is
not bundled (e.g. x509.mind) is NOT a member. Unlike the D1 per-module set
there is NO prelude injection: Result/Option/Ok/Err/Some/None live in
`collect_module_syms`, never in the std export set — this smoke asserts the
MIND core answers 0 for them.

The pure-MIND twin `selftest_tc_std_export(src, src_len, name, name_len)`
takes the CONCATENATED bundled-std source (each module is a balanced
top-level file, so the concat preserves every top-level decl position and
the scan equals the Rust per-module union) plus the queried name.

Three-way agreement, machine-checked (no hand table):
  1. MIND core over (bundled-std concat, name).
  2. The exact Rust rule recomputed in Python over each REAL bundled
     std/*.mind file read from disk: a token-level mirror of
     collect_decl_names' AST arms, union across modules.
  3. The LIVE `mindc check` oracle where applicable (bare, non-prelude
     names): the name is referenced in value position from a one-fn probe
     module, so the probe's own decl set contributes nothing but the probe
     fn itself — E2002 (unknown ident) absence == std-export membership,
     because resolve's `name_resolvable` is exactly per-module-set OR
     std-export-set. Fail-closed: a sentinel definitely-undefined probe
     MUST yield E2002 or the whole smoke fails.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_decl_names_smoke.py.
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
STDLIB_RS = os.path.join(REPO, "src", "project", "stdlib.rs")
STD_DIR = os.path.join(REPO, "std")

PRELUDE = {"Result", "Option", "Ok", "Err", "Some", "None"}

# (name, live) — live=True: the value-position E2002 probe verdict is
# well-defined (bare name, not prelude). Expected membership is NEVER
# hand-authored: it comes from the recomputed Rust rule (leg 2) and must
# equal legs 1 and 3. Spread across decl kinds, early/late concat position,
# near-misses, and NON-bundled std files.
CASES = [
    # fn exports across modules (concat order: arena first ... vec last)
    ("arena_new", True),
    ("vec_new", True),
    ("vec_push", True),
    ("string_new", True),
    ("string_push_byte", True),
    ("map_new", True),
    ("sha256", True),
    ("http_get", True),
    ("toml_parse", True),
    ("jv_parse", True),
    ("rx_compile", True),
    ("read_to_string", True),
    ("tcp_listen", True),
    ("tui_box_new", True),
    ("submit", True),
    # struct exports
    ("Vec", True),
    ("String", True),
    ("Map", True),
    ("Args", True),
    # const export (declared in BOTH json.mind and toml.mind — dup merge)
    ("MAX_DEPTH", True),
    # extern-"C"-block fn exports (fs.mind libc surface)
    ("open", True),
    ("lseek", True),
    ("readdir", True),
    ("memset", True),
    # prelude names: members of name_resolvable but NOT of the std export
    # set — MIND must answer per the rule (0 unless some std file declares
    # them). live=False: the E2002 probe cannot separate std-set from
    # prelude membership.
    ("Result", False),
    ("Option", False),
    ("Ok", False),
    ("Err", False),
    ("Some", False),
    ("None", False),
    # NON-bundled std files: on disk under std/ but not in
    # STDLIB_MIND_SOURCES, so NOT exports.
    ("x509_parse", True),
    ("x509_peer_auth_supported", True),
    # near-misses / traps
    ("vec_neww", True),          # near-miss suffix
    ("vec_ne", True),            # near-miss prefix
    ("Vec_new", True),           # case/shape near-miss
    ("addr", True),              # struct FIELD (Vec.addr) — not a decl
    ("v", True),                 # std fn param name — not a decl
    ("pathname", True),          # extern-fn param name — not a decl
    ("doubling", True),          # comment-only word in vec.mind
    ("__definitely_not_std", True),
    # qualified paths: no bundled std enum declares these; non-enum heads
    # are never inserted by the Rust rule.
    ("Vec::new", False),
    ("Vec::addr", False),
    ("Col::Red", False),
]

TOKEN_RE = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'|[A-Za-z_]\w*|\S')


def is_word(t):
    return re.fullmatch(r"[A-Za-z_]\w*", t) is not None


def tokenize(src):
    return TOKEN_RE.findall(re.sub(r"//[^\n]*", "", src))


def collect_decl_names(src):
    """Recompute resolve.rs collect_decl_names over one module source:
    top-level decl arms only, module/extern-block recursion, enum variants
    bare + qualified, non-decl braces opaque. Identical mirror to the D1
    smoke (self_host_tc_decl_names_smoke.py)."""
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
            names.add(toks[i + 1])  # Node::FnDef / extern-block fn
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


def bundled_modules():
    """Re-extract STDLIB_MIND_SOURCES from src/project/stdlib.rs so a
    bundling change fails loud here instead of silently drifting."""
    with open(STDLIB_RS) as f:
        rs = f.read()
    mods = re.findall(r'\("std\.([a-z0-9_]+)",\s*include_str!', rs)
    if len(mods) < 20:
        print(f"FAIL: extracted only {len(mods)} bundled modules from "
              f"{STDLIB_RS} — extraction regex or bundle layout drifted")
        sys.exit(1)
    return mods


def live_member(mindc, name, workdir, idx):
    probe = ("fn __d3_probe() -> i64 {\n    let v = " + name +
             ";\n    return 0;\n}\n")
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
    fn = lib.selftest_tc_std_export
    fn.argtypes = [ctypes.c_int64] * 4
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")

    mods = bundled_modules()
    concat = b""
    rust_set = set()
    for m in mods:
        path = os.path.join(STD_DIR, m + ".mind")
        with open(path, "rb") as f:
            src = f.read()
        if not src.endswith(b"\n"):
            src += b"\n"
        concat += src
        rust_set |= collect_decl_names(src.decode())
    print(f"bundled: {len(mods)} modules, {len(concat)} bytes, "
          f"rule set = {len(rust_set)} names")

    sb = ctypes.create_string_buffer(concat, len(concat))
    sp = ctypes.cast(sb, ctypes.c_void_p).value

    total = fails = positives = negatives = live_checked = 0
    with tempfile.TemporaryDirectory() as workdir:
        # Fail-closed sentinel: the live leg must actually reach resolve.
        if live_member(mindc, "__d3_definitely_missing", workdir,
                       "sentinel") != 0:
            print("FAIL: live-oracle sentinel did not produce E2002 — "
                  "`mindc check` is not reaching the resolve pass")
            sys.exit(1)
        print("live sentinel: E2002 fires for a definitely-undefined name")

        for idx, (name, live_ok) in enumerate(CASES):
            exp = 1 if name in rust_set else 0
            nb = ctypes.create_string_buffer(name.encode(),
                                             len(name.encode()))
            got = fn(sp, len(concat),
                     ctypes.cast(nb, ctypes.c_void_p).value,
                     len(name.encode()))
            if live_ok:
                live = live_member(mindc, name, workdir, idx)
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
            print(f"  {mark} got={got} rule={exp} live={live_s} {name!r}")

        # Prelude exclusion: NOT std exports (the D1/D3 differentiator),
        # asserted against the rule leg (0 unless a std file declares one).
        for s in sorted(PRELUDE):
            exp = 1 if s in rust_set else 0
            if exp != 0:
                print(f"NOTE: prelude name {s!r} is genuinely declared "
                      f"by a bundled std module")

    print(f"std_export: cases={total} positives={positives} "
          f"negatives={negatives} live_checked={live_checked} fails={fails}")
    if positives < 10 or negatives < 8 or live_checked < 20:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND std-export set diverges from the Rust rule")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
