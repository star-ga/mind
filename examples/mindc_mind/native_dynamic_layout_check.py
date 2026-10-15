#!/usr/bin/env python3
"""Diagnostic transport for the standalone MIND layout planner, not admission.

The executable algorithm and input/output wrapper are MIND. This ordinary
compiler test harness builds artifacts, supplies frozen binary fixtures and
checks results; it is neither a product runtime nor a theorem authority.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LIMIT = 2 * 1024 ** 3
CASES_SHA = "a151b212459dc5afa8cc4b835aa6f946609e81136ce40297a5e36dc347a886b9"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def check_geometry(request, words):
    """Independent range/accounting invariants, in addition to frozen words."""
    _, code, meta, relro, literals, arena, overflow, stack, guard, reserve = request
    assert words[0] == 1
    # (virtual start, mapped length, file start, bytes actually read from input)
    loads = [(words[1], 4096, 0, 4096),
             (words[3], words[4], words[2], code),
             (words[6], words[7], words[5], meta),
             (words[9], words[10], words[8], relro),
             (words[11], 4096, 0, 0),
             (words[15], words[16], 0, 0),
             (words[20], stack, 0, 0)]
    if literals:
        loads.append((words[13], words[14], words[12], literals))
    for va, size, off, supplied in loads:
        assert va % 4096 == off % 4096 == 0
        assert size > 0 and size % 4096 == 0 and 0 <= supplied <= size
        assert off + supplied <= words[27]
    intervals = sorted((va, va + size) for va, size, _, _ in loads)
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert words[15] + arena - 4096 == words[17]
    assert words[18] >= words[15] + arena + guard
    assert words[19] - words[18] == overflow
    assert words[20] - words[19] == guard
    assert words[21] == words[20] + stack == max(end for _, end in intervals)
    assert words[24] == sum(size for _, size, _, _ in loads)
    assert words[25] == words[24] + overflow
    assert words[26] == words[25] + words[22] + words[23] + reserve <= LIMIT
    assert words[28] == 13 + bool(literals)
    assert words[30] == 64 + 56 * words[28]
    assert words[30] + 256 <= words[31] and words[31] % 4 == 0
    assert words[31] + 52 <= words[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--seed-sha256", required=True)
    parser.add_argument("--mindc", type=Path, required=True)
    parser.add_argument("--mindc-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    seed, mindc = args.seed.resolve(), args.mindc.resolve()
    assert digest(seed.read_bytes()) == args.seed_sha256, "seed hash mismatch"
    assert digest(mindc.read_bytes()) == args.mindc_sha256, "compiler hash mismatch"
    case_file = HERE / "native_dynamic_layout_cases.tsv"
    assert digest(case_file.read_bytes()) == CASES_SHA, "frozen cases changed"
    library = (HERE / "native_dynamic_layout.mind").read_bytes()
    driver = (HERE / "native_dynamic_layout_driver.mind").read_bytes()
    source_hashes = {
        "native_dynamic_layout.mind": digest(library),
        "native_dynamic_layout_driver.mind": digest(driver),
        "native_dynamic_layout_cases.tsv": CASES_SHA,
        "native_dynamic_layout_check.py": digest(Path(__file__).read_bytes()),
    }
    (evidence / "SOURCE_HASHES.txt").write_text("".join(
        f"{value}  {key}\n" for key, value in source_hashes.items()))
    (evidence / "TOOLCHAIN.txt").write_text(
        f"{args.seed_sha256}  {seed}\n{args.mindc_sha256}  {mindc}\n")
    print("Frozen cases", CASES_SHA, flush=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("LD_") or key == "GLIBC_TUNABLES" or key == "MINDC_NATIVE_ELF":
            env.pop(key)
    env.update(LC_ALL="C", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    observations = []

    def run(label, argv, data=None, cwd=None, timeout=120):
        result = subprocess.run([str(x) for x in argv], input=data,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=cwd, env=env, timeout=timeout, preexec_fn=limited)
        stem = f"{len(observations):03d}-{label}"
        (evidence / f"{stem}.stdout").write_bytes(result.stdout)
        (evidence / f"{stem}.stderr").write_bytes(result.stderr)
        (evidence / f"{stem}.exit").write_text(str(result.returncode) + "\n")
        observations.append(f"{stem}\t{result.returncode}\t{digest(result.stdout)}\t"
                            f"{digest(result.stderr)}\t{repr([str(x) for x in argv])}\n")
        (evidence / "executions.tsv").write_text(
            "id\texit\tstdout_sha256\tstderr_sha256\targv\n" + "".join(observations))
        return result

    def native_build(name, body):
        source = body + b"\n" + driver
        (evidence / f"{name}.mind").write_bytes(source)
        packet = struct.pack("<6q", int.from_bytes(b"MINDLAY\x01", "little"),
                             268435456, 134217728, 4096, 0, len(source)) + source
        (evidence / f"{name}.packet").write_bytes(packet)
        result = run("build-" + name, [seed], data=packet, timeout=240)
        assert result.returncode == 0 and result.stderr == b"", (
            "native build failed", name, result.returncode, result.stderr)
        assert result.stdout[:4] == b"\x7fELF", ("native build refused", name)
        target = evidence / f"{name}.elf"
        target.write_bytes(result.stdout)
        target.chmod(0o700)
        print("Native build", name, digest(result.stdout), flush=True)
        return target

    native = native_build("planner", library)
    project = evidence / "mlir"
    project.mkdir()
    (project / "main.mind").write_bytes(library + b"\n" + driver)
    (project / "Mind.toml").write_text(
        '[package]\nname="dynamic_layout_check"\nversion="0.1.0"\n'
        'authors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
    mlir = project / "planner.elf"
    result = run("build-mlir", [mindc, "--allow-nondeterministic", "build", "--backend", "mlir",
                               "--target", "cpu", "--emit", "binary", "--optimize", "debug",
                               "--no-cache", "--out", mlir], cwd=project, timeout=300)
    assert result.returncode == 0 and mlir.is_file(), (
        "MLIR build failed", result.returncode, result.stderr)
    print("Fresh MLIR build", digest(mlir.read_bytes()), flush=True)

    cases = []
    for row in csv.DictReader(case_file.read_text().splitlines(), delimiter="\t"):
        request = tuple(map(int, row["request_i64"].split(",")))
        expected = tuple(map(int, row["expected_i64"].split(",")))
        assert len(request) == 10 and len(expected) == 32
        cases.append((row["id"], request, int(row["expected_exit"]),
                      int(row["output_delta"]), struct.pack("<32q", *expected)))
    assert len(cases) == 52
    assert digest(case_file.read_bytes()) == CASES_SHA, "cases changed after builds"
    for name, request, expected_exit, delta, output in cases:
        packet = struct.pack("<12q", delta, 0, *request)
        for leg, exe in (("native", native), ("mlir", mlir)):
            result = run(name + "-" + leg, [exe], data=packet)
            assert (result.returncode, result.stdout, result.stderr) == (
                expected_exit, output, b""), (name, leg, result.returncode,
                                            result.stdout.hex(), result.stderr)
            if expected_exit == 0:
                check_geometry(request, struct.unpack("<32q", result.stdout))
    # These stateful and repeated-call controls were declared in the contract.
    baseline = cases[0]
    for delta in (512, 0, 8, -8):
        for mode in (1, 2):
            for leg, exe in (("native", native), ("mlir", mlir)):
                packet = struct.pack("<12q", delta, mode, *baseline[1])
                result = run(f"state-{delta}-{mode}-{leg}", [exe], data=packet)
                expected = (1, bytes(256), b"") if mode == 1 else (0, baseline[4], b"")
                assert (result.returncode, result.stdout, result.stderr) == expected
    print("PASS 52 exact paired cases + 8 state/repeat controls per backend", flush=True)

    text = library.decode()
    mutations = [
        ("low-bound", "if low_end > 8388608 { return 4; }", "if low_end < 0 { return 4; }", "low_range_over_one"),
        ("literal-bound", "if literal_end > 16777216 { return 5; }", "if literal_end < 0 { return 5; }", "literal_over_one"),
        ("budget", "if reserved > 2147483648 { return 7; }", "if reserved < 0 { return 7; }", "budget_over_page"),
        ("alignment", "if overflow % 1048576 != 0 { return 3; }", "if overflow < 0 { return 3; }", "overflow_unaligned"),
        ("note-placement", "ndl_add(interp_off, 256)", "ndl_add(interp_off, 252)", "baseline"),
    ]
    mutant_bodies = []
    for name, old, new, case in mutations:
        assert text.count(old) == 1, ("mutation anchor changed", name)
        mutant_bodies.append((name, text.replace(old, new).encode(), case))
    clear = "    let _clear: i64 = ndl_zero(output);\n"
    assert text.count(clear) == 1
    alias_body = text.replace(clear, "").replace(
        "pub fn ndl_plan(request: i64, output: i64) -> i64 {\n",
        "pub fn ndl_plan(request: i64, output: i64) -> i64 {\n" + clear)
    mutant_bodies.append(("alias-capture", alias_body.encode(), "alias_same"))
    by_name = {case[0]: case for case in cases}
    for name, body, case_name in mutant_bodies:
        exe = native_build("mutant-" + name, body)
        _, request, expected_exit, delta, output = by_name[case_name]
        result = run("detect-" + name, [exe], data=struct.pack("<12q", delta, 0, *request))
        assert result.returncode in range(8) and result.stderr == b"", (
            "mutation did not produce a normal planner result", name, result.returncode)
        assert len(result.stdout) == 256, ("mutant did not return full result", name)
        assert (result.returncode, result.stdout) != (expected_exit, output), (
            "mutation escaped control", name)
        print("DETECTED normal wrong result", name, flush=True)
    for name, expected in source_hashes.items():
        assert digest((HERE / name).read_bytes()) == expected, ("source drift", name)
    # Preserve and execute the original writer gate, with its original fixtures.
    writer_dir = evidence / "writer-regression"
    writer_env = env.copy()
    writer_env.update(MIND_DYNAMIC_SEED=str(seed), MIND_DYNAMIC_SEED_SHA256=args.seed_sha256,
                      MIND_DYNAMIC_EVIDENCE=str(writer_dir))
    result = subprocess.run([sys.executable, str(HERE / "native_dynamic_writer_check.py")],
                            env=writer_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=300, preexec_fn=limited)
    (evidence / "writer-regression.stdout").write_bytes(result.stdout)
    (evidence / "writer-regression.stderr").write_bytes(result.stderr)
    (evidence / "writer-regression.exit").write_text(str(result.returncode) + "\n")
    assert result.returncode == 0, ("writer regression failed", result.stderr)
    print("PASS six source mutations and unchanged dynamic-writer regression", flush=True)
    (evidence / "RESULT.txt").write_text(
        "PASS standalone MIND layout component; no production C ABI admission or qualified pin.\n")


if __name__ == "__main__":
    main()
