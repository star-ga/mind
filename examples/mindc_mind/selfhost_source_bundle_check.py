#!/usr/bin/env python3
"""Captured compiler-source regression gate. Bootstrap orchestration only.

Default execution requires exact native and CPU compiler identities. Explicit
--unit-only runs filesystem/cache/view controls, never claims compiler parity.
"""

import argparse
import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import _selfhost_sources as sources
from _selfhost_so import build_captured_so
from self_host_loop_smoke import build_seed

MANIFEST = b'[package]\nname="bundle-fixture"\nversion="1.0.0"\n[build]\ntarget="cpu"\nemit="cdylib"\nentry="examples/mindc_mind/main.mind"\n'
COMPONENT = "examples/mindc_mind/a.mind"
OTHER = "examples/mindc_mind/b.mind"
REAL_COMPONENT = "examples/mindc_mind/native_dynamic_layout.mind"
PROBE_DRIVER = "examples/mindc_mind/bundle_probe_driver.mind"
DRIVER = b'''pub fn main() -> i64 {
    if ndl_add(40, 2) != 42 { return 41; }
    if ndl_add(9223372036854775807, 1) != 0 - 1 { return 42; }
    if ndl_add(0 - 1, 2) != 0 - 1 { return 43; }
    if ndl_round(4097, 4096) != 8192 { return 44; }
    0
}
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_fixture(root):
    for relative, data in (("Mind.toml", MANIFEST), (sources.MAIN, b"// main\r\n"),
                           (sources.DRIVER, b"// driver\n"), (COMPONENT, b"// alpha"),
                           (OTHER, b"")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in sources.NATIVE_STD_MODULES:
        path = root / "std" / (name + ".mind")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("// " + name + "\n").encode())


class BundleControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="source-bundle-")
        self.root = Path(self.tmp.name) / "repo"
        make_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_components(self):
        bundle = sources.capture_bundle(self.root)
        std = b"\n".join((self.root / "std" / (name + ".mind")).read_bytes()
                         for name in sources.NATIVE_STD_MODULES) + b"\n"
        main = (self.root / sources.MAIN).read_bytes()
        driver = (self.root / sources.DRIVER).read_bytes()
        expected = std + main + b"\n" + driver
        self.assertEqual(bundle.library_bytes, main)
        self.assertEqual(build_seed(bundle), (expected, struct.pack("<qq", len(std), len(expected)) + expected, len(std)))

    def test_order(self):
        a = sources.capture_bundle(self.root, (COMPONENT, OTHER))
        b = sources.capture_bundle(self.root, (OTHER, COMPONENT))
        main = (self.root / sources.MAIN).read_bytes()
        self.assertEqual(a.library_bytes, main + b"\n// alpha\n")
        self.assertEqual(b.library_bytes, main + b"\n\n// alpha")
        self.assertNotEqual(a.identity, b.identity)

    def test_after_capture(self):
        bundle = sources.capture_bundle(self.root, (COMPONENT,))
        expected = tuple(bundle.cpu_files())
        native = bundle.native_bytes
        for source in bundle.files:
            (self.root / source.path).unlink()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("source reread")):
            bundle.materialize_cpu(Path(self.tmp.name) / "cpu")
            self.assertEqual(bundle.native_bytes, native)
        for path, data in expected:
            self.assertEqual((Path(self.tmp.name) / "cpu" / path).read_bytes(), data)

    def test_same_metadata_drift(self):
        before = sources.capture_bundle(self.root, (COMPONENT,))
        path = self.root / COMPONENT
        stat = path.stat()
        path.write_bytes(b"// omega")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        after = sources.capture_bundle(self.root, (COMPONENT,))
        self.assertNotEqual(before.identity, after.identity)

    def test_driver_separation(self):
        before = sources.capture_bundle(self.root)
        (self.root / sources.DRIVER).write_bytes(b"// other driver")
        after = sources.capture_bundle(self.root)
        self.assertEqual(before.library_bytes, after.library_bytes)
        self.assertNotEqual(before.native_bytes, after.native_bytes)
        self.assertNotEqual(before.identity, after.identity)

    def test_std_capture(self):
        before = sources.capture_bundle(self.root)
        relative = "std/string.mind"
        (self.root / relative).write_bytes(b"// changed std")
        after = sources.capture_bundle(self.root)
        self.assertNotEqual(before.native_bytes, after.native_bytes)
        self.assertNotEqual(before.identity, after.identity)
        self.assertEqual(dict(before.cpu_files())[relative], b"// string\n")

    def test_missing(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            sources.capture_bundle(self.root, ("examples/mindc_mind/missing.mind",))

    def test_utf8(self):
        for relative in (COMPONENT, sources.MAIN, sources.DRIVER):
            old = (self.root / relative).read_bytes()
            (self.root / relative).write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF8"):
                sources.capture_bundle(self.root, (COMPONENT,))
            (self.root / relative).write_bytes(old)

    def test_duplicates(self):
        for paths in ((COMPONENT, COMPONENT), (sources.MAIN,), (sources.DRIVER,)):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sources.capture_bundle(self.root, paths)
        (self.root / OTHER).unlink()
        os.link(self.root / COMPONENT, self.root / OTHER)
        with self.assertRaisesRegex(ValueError, "aliased"):
            sources.capture_bundle(self.root, (COMPONENT, OTHER))

    def test_paths(self):
        for path in ("/tmp/a.mind", "../a.mind", "examples/mindc_mind/x/../a.mind",
                     "examples\\mindc_mind\\a.mind", "a\x00.mind", "std/string.mind",
                     "examples/mindc_mind/a.txt"):
            with self.subTest(path=repr(path)), self.assertRaises(ValueError):
                sources.capture_bundle(self.root, (path,))
        for extra in (b'\n[dependencies]\nother="path"\n',
                      b'\nsources=["elsewhere.mind"]\n'):
            (self.root / "Mind.toml").write_bytes(MANIFEST + extra)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                sources.capture_bundle(self.root)

    def test_symlinks(self):
        (self.root / OTHER).unlink()
        (self.root / OTHER).symlink_to(self.root / COMPONENT)
        with self.assertRaisesRegex(ValueError, "symlink"):
            sources.capture_bundle(self.root, (OTHER,))
        alias = self.root / "examples/mindc_mind/alias"
        alias.symlink_to(self.root / "examples/mindc_mind", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            sources.capture_bundle(self.root, ("examples/mindc_mind/alias/a.mind",))

    def test_spans_and_immutable_inputs(self):
        bundle = sources.capture_bundle(self.root, (COMPONENT, OTHER))
        for native, view in ((False, bundle.library_bytes), (True, bundle.native_bytes)):
            for span in bundle.spans(native=native):
                self.assertEqual(view[span.start:span.end], bundle.source(span.path).data)
                self.assertEqual(sha(view[span.start:span.end]), span.sha256)
        with self.assertRaises(ValueError):
            sources.CapturedSource("component", COMPONENT, bytearray(b"changed"))

    def fake_compiler(self):
        path = Path(self.tmp.name) / "compiler"
        path.write_text('#!/usr/bin/env python3\nimport pathlib,sys\np=pathlib.Path(next(x[6:] for x in sys.argv if x.startswith("--out=")))\np.write_bytes(b"\\x7fELF"+b"\\0"*60)\n')
        path.chmod(0o700)
        return path

    def test_cache_identity(self):
        compiler = self.fake_compiler()
        dependency = Path(self.tmp.name) / "mock-toolchain"
        dependency.write_bytes(b"version1")
        cache = Path(self.tmp.name) / "cache"
        bundle = sources.capture_bundle(self.root, (COMPONENT,))
        options = dict(dependency_files=(dependency,), reuse=True)
        result = build_captured_so(bundle, compiler, sha(compiler.read_bytes()), cache, **options)
        self.assertEqual(result, build_captured_so(bundle, compiler, sha(compiler.read_bytes()), cache, **options))
        result.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "artifact identity"):
            build_captured_so(bundle, compiler, sha(compiler.read_bytes()), cache, **options)
        dependency.write_bytes(b"version2")
        changed = build_captured_so(bundle, compiler, sha(compiler.read_bytes()), cache, **options)
        self.assertNotEqual(result, changed)
        (self.root / COMPONENT).write_bytes(b"// gamma")
        newer = build_captured_so(sources.capture_bundle(self.root, (COMPONENT,)), compiler,
                                 sha(compiler.read_bytes()), cache, **options)
        self.assertNotEqual(changed, newer)

    def test_compiler_mismatch(self):
        compiler = self.fake_compiler()
        with mock.patch("subprocess.run", side_effect=AssertionError("executed mismatched compiler")):
            with self.assertRaisesRegex(ValueError, "compiler hash"):
                build_captured_so(sources.capture_bundle(self.root), compiler, "0" * 64,
                                  Path(self.tmp.name) / "cache")


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_mutations(evidence):
    """Execute real modified helpers; only the intended assertion failure counts."""
    text = (HERE / "_selfhost_sources.py").read_text()
    join = 'return b"\\n".join(source.data for source in sources)'
    cases = (
        ("drop-component", join, 'return self.by_role("main")[0].data', "test_order", "AssertionError"),
        ("reread-original", "output.write(data)",
         'output.write((destination.parent / "repo" / relative).read_bytes())',
         "test_after_capture", "source reread"),
        ("driver-in-library", join,
         join + ' + b"\\n" + self.by_role("driver")[0].data',
         "test_empty_components", "AssertionError"),
    )
    evidence.mkdir(parents=True, exist_ok=True)
    for name, needle, replacement, target, diagnostic in cases:
        if text.count(needle) != 1:
            raise AssertionError(f"mutation anchor drift: {name}")
        directory = evidence / name
        directory.mkdir()
        for module in ("_selfhost_sources.py", "_selfhost_so.py",
                       "self_host_loop_smoke.py", "selfhost_source_bundle_check.py"):
            shutil.copyfile(HERE / module, directory / module)
        mutant = text.replace(needle, replacement)
        compile(mutant, "_selfhost_sources.py", "exec")
        (directory / "_selfhost_sources.py").write_text(mutant)
        (directory / "source.sha256").write_text(sha(mutant.encode()) + "\n")
        command = [sys.executable, "-m", "unittest", "-v",
                   "selfhost_source_bundle_check.BundleControls." + target]
        result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30,
                                preexec_fn=limits)
        (directory / "stdout").write_bytes(result.stdout)
        (directory / "stderr").write_bytes(result.stderr)
        (directory / "exit").write_text(str(result.returncode) + "\n")
        (directory / "command.txt").write_text("\n".join(command) + "\n")
        error = result.stderr.decode("utf-8", errors="replace")
        assert (result.returncode == 1 and "FAILED (failures=1)" in error
                and "ERROR:" not in error and diagnostic in error
                and "FAIL: " + target in error), (name, result.returncode, error)
        print("[PASS] actual orchestration mutation detected by assertion: " + name)


def real_build(args):
    ev = args.evidence.resolve()
    ev.mkdir(parents=True, exist_ok=True)
    for path, expected in ((args.seed, args.seed_sha256), (args.mindc, args.mindc_sha256)):
        if sha(path.read_bytes()) != expected:
            raise ValueError(f"compiler hash mismatch: {path}")
    original = sources.capture_bundle(REPO)
    std = b"\n".join((REPO / "std" / (name + ".mind")).read_bytes()
                     for name in sources.NATIVE_STD_MODULES) + b"\n"
    legacy = std + (REPO / sources.MAIN).read_bytes() + b"\n" + (REPO / sources.DRIVER).read_bytes()
    assert build_seed(original) == (legacy, struct.pack("<qq", len(std), len(legacy)) + legacy, len(std))
    captured = sources.capture_bundle(REPO, (REAL_COMPONENT,))
    probe_files = tuple(sources.CapturedSource("driver", PROBE_DRIVER, DRIVER)
                        if source.role == "driver" else source for source in captured.files)
    bundle = sources.SourceBundle(probe_files, captured.native_std_paths)
    (ev / "library.mind").write_bytes(bundle.library_bytes)
    (ev / "native.mind").write_bytes(bundle.native_bytes)
    (ev / "source.identity").write_text(bundle.identity + "\n")
    (ev / "views.sha256").write_text(sha(bundle.library_bytes) + "  library.mind\n" + sha(bundle.native_bytes) + "  native.mind\n")
    from self_host_native_cf_smoke import v2_header
    packet = v2_header() + bundle.stdin_image
    result = subprocess.run([str(args.seed)], input=packet, capture_output=True, timeout=300, preexec_fn=limits)
    for suffix, data in (("stdout", result.stdout), ("stderr", result.stderr), ("exit", (str(result.returncode) + "\n").encode())):
        (ev / ("native-build." + suffix)).write_bytes(data)
    assert result.returncode == 0 and result.stderr == b"" and result.stdout.startswith(b"\x7fELF"), "native component build failed"
    exe = ev / "native.elf"
    exe.write_bytes(result.stdout)
    exe.chmod(0o500)
    run = subprocess.run([str(exe)], capture_output=True, timeout=30, preexec_fn=limits)
    for suffix, data in (("stdout", run.stdout), ("stderr", run.stderr), ("exit", (str(run.returncode) + "\n").encode())):
        (ev / ("native-run." + suffix)).write_bytes(data)
    assert (run.returncode, run.stdout, run.stderr) == (0, b"", b""), "native component result differs"
    library = build_captured_so(bundle, args.mindc, args.mindc_sha256, ev / "cpu", timeout=300)
    lib = ctypes.CDLL(str(library))
    add = lib.ndl_add
    add.argtypes = [ctypes.c_int64, ctypes.c_int64]
    add.restype = ctypes.c_int64
    round_up = lib.ndl_round
    round_up.argtypes = [ctypes.c_int64, ctypes.c_int64]
    round_up.restype = ctypes.c_int64
    outcomes = [add(40, 2), add((1 << 63) - 1, 1), add(-1, 2), round_up(4097, 4096)]
    (ev / "cpu-outcomes.txt").write_text(repr(outcomes) + "\n")
    assert outcomes == [42, -1, -1, 8192], outcomes
    for path, expected in ((args.seed, args.seed_sha256), (args.mindc, args.mindc_sha256)):
        assert sha(path.read_bytes()) == expected, "compiler changed during gate"
    print("[PASS] actual enabled MIND component: native and captured CPU results agree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--seed-sha256")
    parser.add_argument("--mindc", type=Path)
    parser.add_argument("--mindc-sha256")
    args = parser.parse_args()
    if not args.unit_only and not all((args.evidence, args.seed, args.seed_sha256, args.mindc, args.mindc_sha256)):
        parser.error("full gate requires --evidence and both exact compiler paths/hashes")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BundleControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if args.evidence:
        run_mutations(args.evidence.resolve() / "mutations")
    else:
        with tempfile.TemporaryDirectory(prefix="source-bundle-mutations-") as temporary:
            run_mutations(Path(temporary))
    if args.unit_only:
        print("[PASS] captured bundle orchestration unit controls only; no native/CPU build claim")
    else:
        real_build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
