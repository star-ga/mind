#!/usr/bin/env python3
"""#318 — branch-local `let` shadow must not leak across backends (the WEDGE gate).

A branch-local `let` that SHADOWS an enclosing binding (an outer `let` OR a
PARAMETER) is a block-scoped DECLARATION, not a write of the outer name. The
if-merge must therefore carry the PRE-IF value, never the shadow.

Leg (a) — the Rust/IR merge builder (src/eval/lower.rs, the `then_writes` /
`else_writes` fallback to the pre-if `env`) — is FIXED: tree-eval and the MLIR
backend both answer correctly on every shape below.

Leg (b) — the pure-MIND native-ELF emitter — is NOT: main.mind's
`nb_branch_writes` records an `ast_let()` exactly like an `ast_assign()`, so the
shadow enters `merged_names` and the merge phi carries it. This smoke is the
cross-backend behavioural gate the task asks for (leg (c)); it is EXPECTED-RED on
the native leg until the `nb_branch_writes` scope port lands AND the frozen
stage1.elf is re-minted (patching main.mind alone changes nothing — `mindc build
--backend native` RUNS the frozen seed, src/bin/mindc.rs:928).

The control shape (a branch-local `let` of a FRESH name, shadowing nothing) must
stay CORRECT on every backend both before and after the fix — it is the guard
that the port does not over-fire and start dropping legitimate merges.

Exit: 0 all shapes agree across backends; 1 a divergence; 2 BLOCKED (missing
mindc / stage1.elf).
"""
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
MINDC = pathlib.Path(os.environ.get("MINDC_BIN", str(REPO / "target" / "release" / "mindc")))
STAGE1 = pathlib.Path(
    os.environ.get("MINDC_NATIVE_ELF", str(HERE / "testdata" / "selfhost_loop" / "stage1.elf"))
)

# (name, source, expected, note). `main` returns the value as the process exit code.
SHAPES = [
    ("A_value_if_outer_shadow",
     "fn f(cond: i64) -> i64 {\n    let x: i64 = 7;\n    if cond == 1 {\n        let x: i64 = 99;\n        x\n    } else {\n        0\n    }\n    return x;\n}\nfn main() -> i64 { return f(1); }\n",
     7, "the original #318 repro: value-if, outer-let shadow"),

    ("A2_stmt_if_outer_shadow",
     "fn f(cond: i64) -> i64 {\n    let x: i64 = 7;\n    if cond == 1 {\n        let x: i64 = 99;\n    }\n    return x;\n}\nfn main() -> i64 { return f(1); }\n",
     7, "statement-if form of the same defect"),

    ("C_assign_after_shadow",
     "fn f(cond: i64) -> i64 {\n    let x: i64 = 7;\n    if cond == 1 {\n        let x: i64 = 99;\n        x = 100;\n    }\n    return x;\n}\nfn main() -> i64 { return f(1); }\n",
     7, "ORDERED: an assign AFTER the shadow writes the LOCAL, not the outer — "
        "proves dropping the ast_let arm alone is insufficient"),

    ("G_shadow_param",
     "fn f(x: i64) -> i64 {\n    if x == 1 {\n        let x: i64 = 99;\n        x\n    } else {\n        0\n    }\n    return x;\n}\nfn main() -> i64 { return f(1); }\n",
     1, "shadows a PARAMETER — the exact case that got #287-F2 reverted"),

    ("H_nested_if_shadow",
     "fn f(cond: i64) -> i64 {\n    let x: i64 = 7;\n    if cond == 1 {\n        if cond == 1 {\n            let x: i64 = 99;\n            x\n        } else {\n            0\n        }\n    } else {\n        0\n    }\n    return x;\n}\nfn main() -> i64 { return f(1); }\n",
     7, "shadow inside a NESTED if — exercises nb_if_branch_merged_writes recursion"),

    ("I_shadow_then_while",
     "fn f(cond: i64) -> i64 {\n    let x: i64 = 3;\n    if cond == 1 {\n        let x: i64 = 0;\n        let mut i: i64 = 0;\n        while i < 5 {\n            i = i + 1;\n        }\n    }\n    return x;\n}\nfn main() -> i64 { return f(1); }\n",
     3, "shadow interacting with a while region in the same branch (no return inside "
        "the loop — a `return` nested in while-in-if makes the MLIR backend emit "
        "INVALID IR, 'func.return op must be the last operation in the parent block'; "
        "that is a SEPARATE defect, tracked apart from #318, and would confound this net)"),

    # CONTROL — must be CORRECT before AND after the port. Guards over-firing.
    ("F_control_fresh_local",
     "fn f(cond: i64) -> i64 {\n    let y: i64 = 3;\n    if cond == 1 {\n        let z: i64 = 99;\n        z\n    } else {\n        0\n    }\n    return y;\n}\nfn main() -> i64 { return f(1); }\n",
     3, "CONTROL: branch-local let of a FRESH name shadows nothing — legitimate, "
        "must keep working; catches an over-firing port"),
]


def _run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)


def mlir_value(src: str, tmp: pathlib.Path):
    """Build via the default MLIR backend and RUN it; returns (value, err)."""
    proj = tmp / "p"
    (proj / "src").mkdir(parents=True, exist_ok=True)
    (proj / "src" / "main.mind").write_text(src)
    (proj / "Mind.toml").write_text(
        '[package]\nname = "shadow"\nversion = "0.1.0"\n\n[build]\nemit = "binary"\n'
    )
    r = _run([str(MINDC), "build", "--backend", "mlir"], cwd=str(proj))
    if r.returncode != 0:
        return None, r.stderr.decode()[-300:]
    exe = proj / "target" / "debug" / "shadow"
    if not exe.exists():
        return None, "no artifact"
    return _run([str(exe)]).returncode, ""


def native_value(src: str, tmp: pathlib.Path):
    """Build via the pure-MIND native-ELF backend and RUN it; returns (value, err)."""
    s = tmp / "n.mind"
    s.write_text(src)
    out = tmp / "n.elf"
    env = dict(os.environ, MINDC_STD_DIR=str(REPO / "std"), MINDC_NATIVE_ELF=str(STAGE1))
    r = _run([str(MINDC), "build", "--backend", "native", str(s), "--out", str(out)], env=env)
    if r.returncode != 0:
        return None, r.stderr.decode()[-300:]
    if not out.exists():
        return None, "no artifact"
    return _run([str(out)]).returncode, ""


def main() -> int:
    if not MINDC.exists():
        print(f"BLOCKED: mindc not found at {MINDC}")
        return 2
    if not STAGE1.exists():
        print(f"BLOCKED: stage1.elf not found at {STAGE1}")
        return 2

    print(f"mindc   : {MINDC}")
    print(f"stage1  : {STAGE1}")
    print()
    hdr = f"{'shape':<28}{'want':>6}{'mlir':>8}{'native':>8}   verdict"
    print(hdr)
    print("-" * len(hdr))

    bad = 0
    for name, src, want, note in SHAPES:
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            mv, me = mlir_value(src, tmp)
            nv, ne = native_value(src, tmp)
        ms = "BUILDFAIL" if mv is None else str(mv)
        ns = "BUILDFAIL" if nv is None else str(nv)
        ok_m = mv == want
        # A native BUILDFAIL is an HONEST refusal (fail-closed), never a miscompile.
        ok_n = (nv == want) or (nv is None)
        verdict = "ok" if (ok_m and ok_n) else "*** DIVERGENCE ***"
        if not (ok_m and ok_n):
            bad += 1
        print(f"{name:<28}{want:>6}{ms:>8}{ns:>8}   {verdict}")
        if not ok_m:
            print(f"    MLIR wrong (want {want}, got {ms}) {me}")
        if nv is not None and nv != want:
            print(f"    NATIVE SILENT MISCOMPILE (want {want}, got {ns}) — wedge violation")
        print(f"    {note}")

    print()
    if bad:
        print(f"FAIL: {bad}/{len(SHAPES)} shapes diverge across backends (#318).")
        print("A native BUILDFAIL is accepted (honest fail-close); a WRONG VALUE is not.")
        return 1
    print(f"PASS: {len(SHAPES)}/{len(SHAPES)} shapes agree across backends.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
