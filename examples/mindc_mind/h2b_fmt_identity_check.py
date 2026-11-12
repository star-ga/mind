#!/usr/bin/env python3
"""H2b layout-only format identity: --emit-ir and --emit-mic3 byte-identity.

Ruling Q1.1: a file may be formatted only when its diff is layout-only AND
`--emit-ir` + `--emit-mic3` under one mindc are byte-identical for the
pre-format and post-format bytes. This gate re-measures that proof for the
pinned fifteen-file formatted set (H2b's fourteen plus the seed batch's
format-only std/string.mind), against the committed base bytes.

Files that cannot produce IR under any available composition are NOT formatted
(restored to parent bytes and pinned in testdata/h2b_exclusion_pins.json);
this gate refuses to count a refusal / equal-error pair as identity.

Red mutant:
  --mutant-diverge  rewrite one literal in the current file before comparing,
                    so IR/mic3 must differ and the gate must FAIL
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "93238e8e"
# Compositions that cannot emit IR are excluded from the formatted set; listed
# here so a future attempt is measured explicitly rather than silently skipped.
IR_REFUSAL_EXCLUSIONS = {
    "examples/mindc_mind/source_note_driver.mind",
    "examples/mindc_mind/string_escape_driver.mind",
}

# These fourteen files are outside the formatted set. Q1.1 kept main.mind out of lane C.
# The seven seed-batch entries (S6b, S6a, #289) are semantic std edits or new test
# files made after the H2b base. The seed batch's one format-only file,
# std/string.mind (S6b b8f15059), is measured in the pinned set instead.
# The formatted set is the pinned manifest, not a git-diff walk.
PINNED_FORMATTED_LIST = "examples/mindc_mind/testdata/h2b_fmt_identity_files.txt"
SEMANTIC_EDIT_EXCLUSIONS = {
    "std/sha256.mind": (
        "seed batch: S6b b8f15059 format, then #289 8c94109e bounded scratch, "
        "sha256_checked and the trapping legacy entry (a semantic edit)"
    ),
    "std/toml.mind": (
        "seed batch S6b 64e105d6: the toml depth cap gets its own name (semantic edit)"
    ),
    "std/iouring.mind": (
        "seed batch S6a 697dff69: bounded completion waits (semantic edit)"
    ),
    "tests/sha256_allocation_kat.mind": ("new file from #289 8c94109e; no BASE bytes"),
    "tests/sha256_legacy_kat.mind": ("new file from #289 8c94109e; no BASE bytes"),
    "tests/sha256_memory_kat.mind": ("new file from #289 8c94109e; no BASE bytes"),
    "tests/sha256_repeated_kat.mind": ("new file from #289 8c94109e; no BASE bytes"),
    "tests/open_error_kat.mind": (
        "new file from main #290 (fcb44b97), taken by the main merge; it has no "
        "BASE bytes, so there is no pre/post-format identity to measure"
    ),
    "examples/mindc_mind/main.mind": (
        "C2fix4 99fa1ac2; REBLESS_C2_20261024 (private re-bless note); C2FIX5_20261024 (private re-bless note); "
        "IR 6a53b0d50df529024244b7af252167624c1a849c413ac43f2f4792eaecbbfa2c -> "
        "6d79644658c801952e042b851a5b6f0fa48769d050ed2edc399ff6170d1cc93a; "
        "mic3 48f2e466cebca4f6bfab82679e63c446b4e8591658b43e663a9265155c9f1b85 -> "
        "c5ad7f2e629e34f143717ab81b4be8514811259613449f13a6bf574251363e5c"
    ),
    # Edits after BASE that add or remove code or attributes (none is
    # layout-only). The std crypto/TLS files are main's #346 successors of the
    # branch's early join; std/ecdsa_p256.mind is that join itself.
    ".arch-mind/rules.mind": (
        "arch-mind rules, not a source module: main #351 59ac0915 adds the "
        "procgroup_sys OS-effect module row, 4687e3a4 the posix_sys row (semantic)"
    ),
    "std/async.mind": ("590dcfce private seed helpers prefixed: 7 renamed lines (semantic edit)"),
    "std/blas.mind": ("590dcfce private seed helpers prefixed: 3 renamed lines (semantic edit)"),
    "std/cli.mind": ("590dcfce private seed helpers prefixed: 12 renamed lines (semantic edit)"),
    "std/regex.mind": ("590dcfce private seed helpers prefixed: 4 renamed lines (semantic edit)"),
    "std/tui.mind": ("590dcfce private seed helpers prefixed: 54 renamed lines (semantic edit)"),
    "std/ecdsa_p256.mind": ("05d75b6d X.509/TLS join: +97/-1 code lines (semantic edit)"),
    "std/fs.mind": ("1c7b7c33 append mode for std.fs: +24 code lines (semantic edit)"),
    "std/x25519.mind": (
        "d55ec364 fmul copies through gf_copy16; gf_copy16_from_t removed, -10 code lines "
        "(semantic edit)"
    ),
    "std/posix.mind": (
        "4687e3a4 raw syscalls moved to std.posix_sys; #[protection] on the "
        "signal, fdopendir and readdir decisions (semantic edit)"
    ),
    "std/rsa_pss.mind": ("main #346 46a9c328: public EMSA-PSS check, +107/-47 (semantic edit)"),
    "std/tls13_finished.mind": ("main #346 46a9c328: +134 code lines (semantic edit)"),
    "std/tls13_handshake.mind": (
        "main #346 46a9c328: client handshake moved to std.tls13_client, "
        "+15/-276 (semantic edit)"
    ),
    "std/tls13_record.mind": (
        "main #346 46a9c328: #[protection] on tls13_record_open (semantic edit)"
    ),
    "std/x509.mind": (
        "main #346 46a9c328: path validation split into x509_* modules, +49/-700 "
        "(semantic edit)"
    ),
    # 0.10.3 train edits (lane commits, landed in the release commit 8db067a1).
    # The three std files change comment text, not layout: e9e0b363 rewords a
    # second line-start `// std/<x>.mind` comment line, which the seed
    # module-bound scan reads as a module bound.
    "std/chacha20_poly1305.mind": (
        "e9e0b363: the line-start `// std/sha256.mind` comment line is reworded so the "
        "seed module-bound scan reads one bound (comment text edit, not layout)"
    ),
    "std/mlkem768.mind": (
        "e9e0b363: the line-start `// std/hkdf.mind` comment line is reworded so the "
        "seed module-bound scan reads one bound (comment text edit, not layout)"
    ),
    "std/x25519mlkem768.mind": (
        "e9e0b363: the line-start `// std/mlkem768.mind` comment line is moved off the "
        "line start so the seed module-bound scan reads one bound (comment text edit)"
    ),
    "examples/detmath_kat/main.mind": (
        "675db451: the private detmath kernel copies are removed for `import std.detmath`, "
        "+77/-362 (semantic edit)"
    ),
    "std/detmath.mind": (
        "3f299672: reduce_arg subtracts each part of pi/2 once (pio2_1t removed) and seven "
        "sin/cos kernel coefficients get the right digits, +10/-12 (semantic edit)"
    ),
    "examples/mindc_mind/selfhost_driver.mind": (
        "3129845c: nb_report_unsupported reports the extern ABI refusal first, "
        "+4 code lines (semantic edit)"
    ),
    # Main #383 e3b01d11 lint-warning fixes: renamed bindings, `let mut` plus
    # reassignment for shadowed lets, an unused import dropped (code edits).
    "examples/autodiff_demo.mind": ("main #383 e3b01d11: snake_case gradient bindings"),
    "examples/bimap_currency/main.mind": ("main #384 1e1167b4: shadowed lets -> distinct names"),
    "examples/bimap_pairs/main.mind": ("main #383 e3b01d11: shadowed lets -> reassignment"),
    "examples/compliance/audit_report.mind": (
        "main #383 e3b01d11: unused std.tensor import dropped"
    ),
    "examples/fft_q16.mind": ("main #383 e3b01d11: snake_case loop bindings"),
    "examples/hello_stdlib.mind": ("main #383 e3b01d11: shadowed lets -> `let mut` + reassignment"),
    "examples/lorenz_q16.mind": ("main #383 e3b01d11: snake_case parameter bindings"),
    "examples/mindc_mind/native_emission_context.mind": (
        "C2fix4 99fa1ac2; REBLESS_C2_20261024 (private re-bless note); "
        "base emit refuses ndl_plan at line 30"
    ),
    "examples/mindc_mind/native_libc_binding.mind": (
        "C2fix4 99fa1ac2; REBLESS_C2_20261024 (private re-bless note); "
        "base emit refuses nb_cabi_free at line 1814"
    ),
    "examples/mindc_mind/native_libc_io.mind": (
        "C2fix4 99fa1ac2; REBLESS_C2_20261024 (private re-bless note); "
        "base emit refuses nb_cabi_build at line 306"
    ),
    "examples/mindc_mind/native_scoped_source_image.mind": (
        "C2fix4 99fa1ac2; REBLESS_C2_20261024 (private re-bless note); "
        "base emit refuses lex at line 19"
    ),
    "examples/mindc_mind/native_source_image.mind": (
        "C2fix4 99fa1ac2; REBLESS_C2_20261024 (private re-bless note); "
        "base emit refuses nsa_code_size at line 72"
    ),
}


def mindc() -> str:
    bin_path = os.environ.get("MINDC_BIN") or os.environ.get("MINDC")
    if not bin_path:
        raise SystemExit("h2b_fmt_identity: MINDC_BIN is required")
    return bin_path


PINNED_COUNT = 15


def walk_mind_files() -> set[str]:
    """The committed git-diff walk BASE..HEAD, minus both exclusion sets.

    Only paths that have BASE bytes: modified, renamed or type-changed. A file
    added after BASE (or deleted since) has no pre-format text, so it cannot be
    a formatting of BASE and there is no identity to measure.
    """
    out = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=MRT", BASE, "HEAD"], cwd=ROOT, text=True
    )
    return {
        p for p in out.splitlines()
        if p.endswith(".mind")
        and p not in SEMANTIC_EDIT_EXCLUSIONS
        and p not in IR_REFUSAL_EXCLUSIONS
    }


def changed_mind_files() -> list[str]:
    manifest = ROOT / PINNED_FORMATTED_LIST
    files = []
    for line in manifest.read_text().splitlines():
        path = line.strip()
        if not path or path.startswith("#"):
            continue
        if not path.endswith(".mind"):
            raise SystemExit(f"h2b_fmt_identity: pinned entry is not a .mind file: {path}")
        if path in SEMANTIC_EDIT_EXCLUSIONS or path in IR_REFUSAL_EXCLUSIONS:
            raise SystemExit(f"h2b_fmt_identity: pinned entry is excluded: {path}")
        files.append(path)
    if len(files) != PINNED_COUNT or len(set(files)) != PINNED_COUNT:
        raise SystemExit(
            f"h2b_fmt_identity: pinned list has {len(files)} entries "
            f"({len(set(files))} distinct), want {PINNED_COUNT} distinct"
        )
    # The walk is no longer the source of the set, but it must still agree: a
    # .mind file that existed at BASE and was edited since is either pinned or
    # explicitly excluded. Equality with the 15 pinned files keeps it non-vacuous.
    walk = walk_mind_files()
    if walk != set(files):
        raise SystemExit(
            "h2b_fmt_identity: pinned set != git-diff walk minus exclusions; "
            f"walk-only={sorted(walk - set(files))} pinned-only={sorted(set(files) - walk)}"
        )
    return files


def emit_pair(compiler: str, src: bytes, stem: pathlib.Path) -> tuple[bytes, bytes]:
    mind = stem.with_suffix(".mind")
    mind.write_bytes(src)
    ir = subprocess.run(
        [compiler, str(mind), "--emit-ir"],
        capture_output=True,
        timeout=120,
    )
    mic = stem.with_suffix(".mic3")
    if mic.exists():
        mic.unlink()
    m = subprocess.run(
        [compiler, str(mind), "--emit-mic3", str(mic)],
        capture_output=True,
        timeout=120,
    )
    if ir.returncode != 0 or m.returncode != 0 or not mic.exists():
        err = (ir.stderr + m.stderr)[:300]
        raise RuntimeError(f"emit refused rcs=({ir.returncode},{m.returncode}) {err!r}")
    return ir.stdout, mic.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant-diverge", action="store_true")
    args = ap.parse_args()
    compiler = mindc()
    files = [p for p in changed_mind_files() if p not in IR_REFUSAL_EXCLUSIONS]
    if not files:
        print("h2b_fmt_identity: FAIL — no formatted files measured (vacuous)")
        return 1
    asserted = 0
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h2b-id-") as td:
        tdir = pathlib.Path(td)
        for rel in files:
            shown = subprocess.run(
                ["git", "show", f"{BASE}:{rel}"], cwd=ROOT, capture_output=True
            )
            if shown.returncode != 0:
                failures.append(f"BASE_MISSING {rel} {shown.stderr[:200]!r}")
                continue
            old = shown.stdout
            if not (ROOT / rel).is_file():
                failures.append(f"FILE_MISSING {rel}")
                continue
            new = (ROOT / rel).read_bytes()
            if new == old:
                # A pinned file must carry a layout change, else it proves nothing.
                failures.append(f"NOT_FORMATTED {rel} is byte-identical to {BASE}")
                continue
            if args.mutant_diverge and rel == files[0]:
                # Semantic change: a comment is not enough — flip a stored
                # constant so the emitted IR/mic3 MUST move.
                for sample in (
                    b"__mind_store_i64(out, 0);",
                    b"return 1;",
                    b"let mut ok: i64 = 1;",
                ):
                    mutated = new.replace(sample, sample.replace(b"0;", b"1;").replace(b"return 1;", b"return 2;").replace(b"= 1;", b"= 2;"), 1)
                    if mutated != new:
                        new = mutated
                        break
                else:
                    # last resort: rewrite the first function body return
                    new = new.replace(b"return 0;", b"return 3;", 1)
            stem = tdir / pathlib.Path(rel).stem
            try:
                ir_a, mic_a = emit_pair(compiler, old, stem.with_name(stem.name + "-base"))
                ir_b, mic_b = emit_pair(compiler, new, stem.with_name(stem.name + "-head"))
            except RuntimeError as e:
                failures.append(f"EMIT_REFUSED {rel} {e}")
                continue
            if ir_a != ir_b:
                failures.append(
                    f"IR_DIVERGE {rel} base={hashlib.sha256(ir_a).hexdigest()[:12]} "
                    f"head={hashlib.sha256(ir_b).hexdigest()[:12]}"
                )
            else:
                asserted += 1
                print(f"[PASS] IR_IDENTICAL {rel}", flush=True)
            if mic_a != mic_b:
                failures.append(
                    f"MIC3_DIVERGE {rel} base={hashlib.sha256(mic_a).hexdigest()[:12]} "
                    f"head={hashlib.sha256(mic_b).hexdigest()[:12]}"
                )
            else:
                asserted += 1
                print(f"[PASS] MIC3_IDENTICAL {rel}", flush=True)
            print(
                f"IDENTITY {rel} IR_MIC3_IDENTICAL={ir_a == ir_b and mic_a == mic_b}",
                flush=True,
            )
    print(f"h2b_fmt_identity: files={len(files)} asserted={asserted}")
    if not failures and asserted != 2 * len(files):
        failures.append(f"ASSERTED_COUNT {asserted} != {2 * len(files)}")
    for f in failures:
        print(f"FAIL {f}")
    if failures:
        print("h2b_fmt_identity: FAIL")
        return 1
    print("h2b_fmt_identity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
