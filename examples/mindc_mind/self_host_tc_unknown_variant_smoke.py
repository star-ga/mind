#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2008 unknown-enum-variant rule.

Ports the resolve.rs decision (`unknown_variant` + `collect_enum_variants`,
UNKNOWN_VARIANT_CODE "E2008"): a qualified `Enum::Variant` path whose HEAD
names a locally-declared enum but whose trailing segment is NOT one of that
enum's declared variants. Deferrals mirror the Rust source exactly: a bare
name (no "::"), a deeper `a::B::C` path, and a non-local head all return 0.
Duplicate enum declarations MERGE variant sets (BTreeMap or_default).

The pure-MIND twin `selftest_tc_unknown_variant(src, src_len, path, path_len)`
takes the FULL module source buffer (it builds the enum registry itself, via
the real self-host lexer) plus the queried path string — so the registry
construction, not just the final membership bit, is MIND-computed.

Three-way agreement, machine-checked (no hand table):
  1. MIND core over (source, path).
  2. The exact Rust rule recomputed in Python: enum decls parsed from the
     same source (brace/paren-depth entry scan mirroring
     collect_enum_variants), then split_once("::") / defer / membership.
  3. The LIVE `mindc check` oracle: the same source is checked and the
     presence of "E2008" for that path's usage is asserted equal.

Corpus: known/unknown variants in value AND call position, qualified
precision across enums sharing variant names, tuple + struct payload
variants, deeper-path and non-enum-head deferrals, duplicate-decl merge.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_self_host_only_call_smoke.py.
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")

ENV_SRC = """\
enum Color { Red, Green, Blue }
enum Shade { Red, Dark }
enum Shape { Circle(i64), Rect { w: i64, h: i64 }, Point }
enum Dup { A }
enum Dup { B }
"""

# (path, usage-template or None, note). usage None -> value position
# `let v = PATH;`; "call" -> call position `let v = PATH(1);`.
CASES = [
    ("Color::Red", None, "declared variant, value position"),
    ("Color::Rde", None, "typo'd variant, value position"),
    ("Color::Dark", None, "variant of a DIFFERENT enum (qualified precision)"),
    ("Shade::Red", None, "shared variant name resolves per-enum"),
    ("Shade::Green", None, "shared-name enum, undeclared variant"),
    ("Shape::Circle", "call", "tuple-payload variant, call position"),
    ("Shape::Circl", "call", "typo'd payload variant, call position"),
    ("Shape::Rect", None, "struct-payload variant name"),
    ("Shape::Point", None, "unit variant after payload variants"),
    ("Shape::W", None, "payload FIELD name is not a variant"),
    ("Shape::Bogus", None, "undeclared variant after payload skip"),
    ("Mode::On", None, "head is not a local enum (deferred)"),
    ("Color::Red::Deep", None, "deeper a::B::C path (deferred)"),
    ("Dup::A", None, "duplicate-decl merge: first decl's variant"),
    ("Dup::B", None, "duplicate-decl merge: second decl's variant"),
    ("Dup::C", None, "duplicate-decl merge: in neither decl"),
]


def collect_enums(src):
    """Mirror collect_enum_variants: enum name -> merged variant-name set."""
    enums = {}
    for m in re.finditer(r"\benum\s+(\w+)\s*\{", src):
        depth, i = 1, m.end()
        start = i
        while depth > 0 and i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        body = src[start : i - 1]
        variants = set()
        d = p = 0
        expect = True
        j = 0
        while j < len(body):
            c = body[j]
            if c == "{":
                d += 1
            elif c == "}":
                d -= 1
            elif c == "(":
                p += 1
            elif c == ")":
                p -= 1
            elif d == 0 and p == 0:
                if c == ",":
                    expect = True
                elif expect and (c.isalpha() or c == "_"):
                    k = j
                    while k < len(body) and (body[k].isalnum() or body[k] == "_"):
                        k += 1
                    variants.add(body[j:k])
                    expect = False
                    j = k
                    continue
            j += 1
        enums.setdefault(m.group(1), set()).update(variants)
    return enums


def rust_rule(path, enums):
    """The exact resolve.rs unknown_variant decision."""
    if "::" not in path:
        return 0
    head, _, rest = path.partition("::")
    if "::" in rest:
        return 0
    if head not in enums:
        return 0
    return 0 if rest in enums[head] else 1


def case_source(path, usage):
    use = f"{path}(1)" if usage == "call" else path
    return ENV_SRC + f"fn main() -> i64 {{\n    let v = {use};\n    return 0;\n}}\n"


def live_oracle(mindc, src, workdir):
    p = os.path.join(workdir, "case.mind")
    with open(p, "w") as f:
        f.write(src)
    r = subprocess.run([mindc, "check", p], capture_output=True, text=True)
    return 1 if "E2008" in (r.stdout + r.stderr) else 0


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
    fn = lib.selftest_tc_unknown_variant
    fn.argtypes = [ctypes.c_int64] * 4
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")
    total = fails = positives = negatives = 0
    with tempfile.TemporaryDirectory() as workdir:
        for path, usage, note in CASES:
            src = case_source(path, usage)
            enums = collect_enums(src)
            exp = rust_rule(path, enums)
            sb = ctypes.create_string_buffer(src.encode(), len(src.encode()))
            pb = ctypes.create_string_buffer(path.encode(), len(path.encode()))
            got = fn(
                ctypes.cast(sb, ctypes.c_void_p).value,
                len(src.encode()),
                ctypes.cast(pb, ctypes.c_void_p).value,
                len(path.encode()),
            )
            live = live_oracle(mindc, src, workdir)
            total += 1
            ok = got == exp == live
            if exp == 1:
                positives += 1
            else:
                negatives += 1
            if not ok:
                fails += 1
            mark = "ok " if ok else "DIFF"
            print(f"  {mark} got={got} rule={exp} live={live} {path!r} | {note}")

    print(
        f"unknown_variant: cases={total} positives={positives} "
        f"negatives={negatives} fails={fails}"
    )
    if positives < 1 or negatives < 1:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2008 core diverges from the Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
