#!/usr/bin/env python3
"""Gate for the pure-MIND MIC3 v0x04 declared-prefix mirror.

Compiles examples/mind_mirror_v04/mic3_v04_prefix_mirror.mind with the frozen
pure-MIND native compiler ELF and RUNS the emitted artifact against every vector
in testdata/MANIFEST.tsv. Nothing in the measured leg is Rust, LLVM or MLIR: the
compiler is a static ELF emitted by the pure-MIND backend, and the decoder under
test is the ELF this script produces from the .mind source.

The vectors and their expected verdicts come from the reference codec (the
development oracle tests/v04_mirror_vector_dump.rs); this script never decides
what a vector should do.

Two independent checks per positive vector:
  1. the artifact's exit code, and
  2. the bytes it writes under `--dump`, compared HERE against the vector file's
     own leading bytes -- so the byte-equality claim is not taken on the
     program's word.

Usage:
  python3 examples/mind_mirror_v04/mirror_gate.py [--semantic]
  python3 examples/mind_mirror_v04/mirror_gate.py [--mutate <sed-expr>]

`--semantic` runs the separate module ValueId and function-metadata controls;
the default manifest selects the wire-canonical corpus in MANIFEST.tsv.
"""

import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "mic3_v04_prefix_mirror.mind"
TESTDATA = HERE / "testdata"
MANIFEST = ROOT / "examples/mindc_mind/testdata/stdlib_manifest.txt"
SEMANTIC_MANIFEST = TESTDATA / "SEMANTIC_MANIFEST.tsv"
COMPILER = ROOT / "examples/mindc_mind/testdata/selfhost_loop/stage1.elf"
FROZEN_SEED_COUNT = 21


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seed_modules():
    mods = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1] == "seed":
            mods.append(fields[0])
    return mods


# Pinned identity of the toolchain this gate speaks for. A gate that only
# PRINTS these attests to nothing: the identical output would appear after the
# compiler or the seed blob silently changed underneath it. Re-blessing is a
# deliberate edit of these two constants in the commit that changes them.
EXPECTED_COMPILER_SHA256 = (
    "377563045b8d626380bbc0ac4aa2e7994de3831e59569d0cefb6ec81bc83201b"
)
EXPECTED_STD_SEED_SHA256 = (
    "421d99c41cd6b0dcf77c39f67181b9846a2658437d72eb8b590d5c26089bffca"
)

# Every subprocess this gate starts is bounded, so a hang is a reported failure
# rather than a run that never returns.
COMPILE_TIMEOUT_S = 300
RUN_TIMEOUT_S = 60


class _TimedOut:
    """Stands in for a run that exceeded its bound, as a distinct exit code."""

    returncode = 124
    stdout = b""
    stderr = b""


def run_bounded(argv):
    try:
        return subprocess.run(argv, capture_output=True, timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _TimedOut()


def compile_native(src_bytes: bytes, out_path: pathlib.Path):
    """Compose the documented source image and shell to the frozen compiler.

    Identical composition to src/bin/mindc.rs::run_native_backend_bridge:
    [8B user_lo LE][8B src_len LE][std_blob ++ user_src] on stdin, ELF on stdout.
    """
    mods = seed_modules()
    if len(mods) != FROZEN_SEED_COUNT:
        sys.exit(f"seed module count {len(mods)} != {FROZEN_SEED_COUNT}")
    blob = b""
    for module in mods:
        blob += (ROOT / "std" / f"{module}.mind").read_bytes() + b"\n"
    combined = blob + src_bytes
    image = (
        len(blob).to_bytes(8, "little")
        + len(combined).to_bytes(8, "little")
        + combined
    )
    try:
        proc = subprocess.run(
            [str(COMPILER)], input=image, capture_output=True, timeout=COMPILE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"compiler exceeded {COMPILE_TIMEOUT_S}s\n")
        return None, sha(blob)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        return None, sha(blob)
    elf = proc.stdout
    if len(elf) < 256 or elf[:4] != b"\x7fELF":
        return None, sha(blob)
    out_path.write_bytes(elf)
    out_path.chmod(0o755)
    return elf, sha(blob)


def read_manifest(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, expected, size, digest, consumed, note = line.split("\t", 5)
        rows.append((name, int(expected), int(size), digest, int(consumed), note))
    return rows


def main():
    mutate = None
    semantic = "--semantic" in sys.argv[1:]
    if "--mutate" in sys.argv[1:]:
        index = sys.argv.index("--mutate")
        if index + 1 >= len(sys.argv):
            sys.exit("--mutate requires a sed expression")
        mutate = sys.argv[index + 1]

    src = SRC.read_bytes()
    if mutate:
        proc = subprocess.run(
            ["sed", mutate], input=src, capture_output=True, timeout=RUN_TIMEOUT_S
        )
        if proc.returncode != 0 or proc.stdout == src:
            sys.exit("mutation did not change the source")
        src = proc.stdout

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="mind-v04-mirror-"))
    # The compiled mirror lives in a scratch directory that is removed on every
    # exit from this function, so a run leaves nothing behind whether it passes,
    # fails a pinned digest, or raises.
    try:
        elf_path = tmpdir / "mirror.elf"
        elf, seed_digest = compile_native(src, elf_path)
        if elf is None:
            print("FAIL: the frozen pure-MIND compiler refused the mirror source")
            return 1

        compiler_digest = sha(COMPILER.read_bytes())
        print(f"compiler_elf_sha256   {compiler_digest}")
        print(f"std_seed_blob_sha256  {seed_digest}")
        print(f"mirror_source_sha256  {sha(src)}"
              + ("  (MUTATED)" if mutate else ""))
        print(f"mirror_artifact_sha256 {sha(elf)}  bytes={len(elf)}")
        print(f"harness_sha256        {sha(pathlib.Path(__file__).read_bytes())}")
        print("")

        pinned = []
        if compiler_digest != EXPECTED_COMPILER_SHA256:
            pinned.append(
                f"compiler ELF is {compiler_digest}, pinned {EXPECTED_COMPILER_SHA256}"
            )
        if seed_digest != EXPECTED_STD_SEED_SHA256:
            pinned.append(
                f"std seed blob is {seed_digest}, pinned {EXPECTED_STD_SEED_SHA256}"
            )
        if pinned:
            print(f"FAIL: this gate speaks for a different toolchain than it pins")
            for entry in pinned:
                print(f"  - {entry}")
            return 1

        manifest_path = SEMANTIC_MANIFEST if semantic else TESTDATA / "MANIFEST.tsv"
        fixture_dir = TESTDATA / "semantic" if semantic else TESTDATA
        rows = read_manifest(manifest_path)
        print(f"manifest              {manifest_path.name}")
        positives = [r for r in rows if r[0].startswith("pos_")]
        negatives = [r for r in rows if r[0].startswith("neg_")]
        if not positives or not negatives:
            sys.exit("VACUOUS: manifest lacks positives or negatives")

        failures = []
        dumped = 0
        for name, expected, size, digest, consumed, note in rows:
            path = fixture_dir / f"{name}.mic3"
            scratch = None
            if note.startswith("recipe="):
                # Rebuild a recipe-declared vector in scratch instead of shipping a
                # large permanent binary. The size and digest below still pin it, so
                # a wrong reconstruction fails rather than silently passing.
                spec = note.split()[0][len("recipe="):]
                kind, base, target = spec.split(":")
                if kind != "pad":
                    failures.append(f"{name}: unknown recipe kind {kind!r}")
                    continue
                seed_bytes = (fixture_dir / f"{base}.mic3").read_bytes()
                built = seed_bytes + b"\x00" * (int(target) - len(seed_bytes))
                scratch = tempfile.NamedTemporaryFile(suffix=".mic3", delete=False)
                scratch.write(built)
                scratch.close()
                path = pathlib.Path(scratch.name)
            # Scratch built from a recipe is released on EVERY exit from this
            # iteration, including the failure branches that skip the rest of it.
            try:
                data = path.read_bytes()
                if len(data) != size or sha(data) != digest:
                    failures.append(f"{name}: vector file does not match the manifest digest")
                    continue
                proc = run_bounded([str(elf_path), str(path)])
                got = proc.returncode
                status = "[PASS]" if got == expected else "[FAIL]"
                if got != expected:
                    failures.append(f"{name}: expected exit {expected}, got {got}")
                note_extra = ""
                if name.startswith("pos_"):
                    dump = run_bounded([str(elf_path), str(path), "--dump"])
                    emitted = dump.stdout
            # The expected length is published in the manifest by a SECOND
            # implementation (the oracle), not by the program under test and not
            # by the reference decoder -- the reference consumes the whole body,
            # so it cannot supply this subset boundary. Comparing against
            # data[:len(emitted)] would be self-lengthed: a mirror emitting a
            # SHORTER but correct prefix would satisfy it, and a truncating
            # mutant would survive. The generator carries the reference-derived
            # whole-body assertion that this leg cannot make.
                    if dump.returncode != expected:
                        failures.append(
                            f"{name}: --dump exited {dump.returncode}, expected "
                            f"{expected}; its output is not a completed re-emission"
                        )
                    elif consumed <= 0:
                        failures.append(
                            f"{name}: manifest carries no reference consumed length; the "
                            f"expected prefix length must not come from the program"
                        )
                    elif len(emitted) != consumed:
                        failures.append(
                            f"{name}: re-emitted {len(emitted)} bytes but the reference "
                            f"consumes exactly {consumed}"
                        )
                    elif emitted != data[:consumed]:
                        failures.append(
                            f"{name}: re-emitted bytes differ from the input's first "
                            f"{consumed} bytes"
                        )
                    else:
                        dumped += 1
                        print(f"[PASS] {name}: exact reference-length prefix bytes")
                        note_extra = f"  reemit={len(emitted)}B byte-identical"
                        if expected == 20 and len(emitted) >= len(data):
                            failures.append(
                                f"{name}: remainder-refused vector has no remainder"
                            )
                print(f"{status:9} {name:28} exit={got:<3} want={expected:<3}"
                      f" bytes={size}{note_extra}")
            finally:
                if scratch is not None:
                    path.unlink(missing_ok=True)

        print("")
        print(f"vectors={len(rows)}  positives={len(positives)}  negatives={len(negatives)}"
              f"  reemit_byte_compared={dumped}")
        if failures:
            print(f"FAIL ({len(failures)})")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("Complete: all vectors matched the reference-decided verdicts")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


sys.exit(main())
