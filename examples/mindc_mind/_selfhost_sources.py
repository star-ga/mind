"""Immutable source views for internal compiler bootstrap orchestration.

This is not the product import resolver. The caller owns an immutable checkout
while capture runs; capture is not an atomic filesystem transaction. Later view
construction and materialization never read that checkout again.

The module also holds the direct-feed compile helpers of the native-lane smokes and start_child, the one
stdin start helper (its static rule is in _process_starts.py). The identity tool
is in _selfhost_identity.py, the runner of the string-rule acceptance rows
(tests/fixtures/native_lane_string_rows) in _selfhost_string_rows.py.
"""

from dataclasses import dataclass
import functools
import hashlib
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib


NATIVE_STD_MODULES = (
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
)
MAIN = "examples/mindc_mind/main.mind"
DRIVER = "examples/mindc_mind/selfhost_driver.mind"
# The compiler's part files, in this fixed order after main.mind wherever the
# compiler is composed: one list, read by every composer (native stream, the
# captured CPU build, the Rust oracle input). main.mind imports the parts, so a
# Mind.toml build of the checkout links them; a composed unit holds them inline.
COMPILER_COMPONENTS: tuple[str, ...] = tuple("examples/mindc_mind/" + name for name in (
    "nb_emit_core.mind", "nb_part_dispatch.mind", "nb_x86_vex.mind", "nb_x86_ymm_bank.mind", "nb_cpu_features.mind",
    "nb_measure.mind", "nb_shm.mind", "nbx_cuda_rows2.mind", "nbx_libc_rows.mind", "nb_exit.mind",
    "tc_std_surface_table.mind", "nb_x86_ymm_bank_checks.mind", "nb_x86_ymm_bank_walk.mind", "nb_x86_ymm_epi.mind", "nb_ast.mind",
    "nb_group_sum_enc.mind", "nb_group_bankw.mind", "nb_group_tail32.mind", "nb_group_fuse.mind", "nb_group_fuse32.mind", "nb_group_i32w.mind",
    "nb_group_terms.mind", "nb_group_epi.mind", "nb_group_sum_bank.mind", "nb_group_sum_acc.mind", "nb_opt_level.mind", "nb_opt_licm_facts.mind",
    "nb_prim_desc.mind", "nb_narrow_rebind.mind", "nb_opt_licm_scan.mind", "nb_opt_licm_check.mind", "nb_opt_licm.mind",
    "nb_intern_names.mind", "nb_intern_names2.mind", "nb_synth_sha.mind", "nb_synth_sha2.mind",
    "nb_float_lit.mind", "nb_float_lit2.mind", "nb_extern_abi.mind", "nbt_extern_scan.mind", "nbt_extern_keys.mind", "nb_lastdef.mind", "nb_alloc_select.mind", "nb_strtab_index.mind",
    "nb_lower_loop.mind", "nb_lex_scan.mind", "nb_stack_budget.mind", "nbx_bind.mind", "nb_print.mind", "nb_arr_fold.mind",
    "nb_branch_memo.mind", "nb_exit_carry.mind", "nb_value_if_memo.mind", "nbt_jump_table.mind", "nbt_fvc_names.mind",
    "nb_lane_keys.mind", "nb_lane_report.mind", "nb_lane_scan.mind", "nb_lane_str.mind", "nb_lane_str2.mind", "nb_lane_rules.mind",
    "nb_group_guard.mind", "nb_group_walk.mind"))
FROZEN_SEED = "examples/mindc_mind/testdata/selfhost_loop/stage1.elf"
LAYOUT_MAGIC = int.from_bytes(b"MINDLAY\x01", "little")
# Input magics of the stage driver (nb_lane_keys.mind): the 8 bytes before the v1 or v2 header. The host
# never sends one (its first word is user_lo or LAYOUT_MAGIC). stdlib_manifest_lint.py names the files that may
# hold these bytes.
SELF_MAGIC = b"MINDSLF\x01"  # self mode: the lane rules stay off (the compiler source is not a lane program)
KEY_MAGIC = b"MINDKEY\x01"   # the compiler prints its key table (id, a space, the name; one line per key)
HOOK_MAGIC = b"MINDHOOK"     # test input: the next 8 bytes (LE) are the word nb_lane_rules_run returns
BOUNDED_LAYOUT = (768 << 20, 384 << 20, 1 << 20)  # arena, stack, guard (bounded seed)
_REPO = pathlib.Path(__file__).resolve().parents[2]
_MAX_I64 = (1 << 63) - 1


def _field(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _path(value: str, *, component: bool = False) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError("unsupported source path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("source paths must be normalized and repository-relative")
    if component and (not value.startswith("examples/mindc_mind/") or not value.endswith(".mind")):
        raise ValueError("components must be compiler-local .mind paths")
    return value


@dataclass(frozen=True)
class CapturedSource:
    role: str
    path: str
    data: bytes

    def __post_init__(self):
        _path(self.path)
        if type(self.data) is not bytes:
            raise ValueError("captured source requires immutable bytes")
        self.data.decode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class SourceSpan:
    role: str
    path: str
    start: int
    end: int
    sha256: str


@dataclass(frozen=True)
class SourceBundle:
    files: tuple[CapturedSource, ...]
    native_std_paths: tuple[str, ...]

    def __post_init__(self):
        if type(self.files) is not tuple or type(self.native_std_paths) is not tuple:
            raise ValueError("source bundle requires immutable tuples")
        roles = tuple(source.role for source in self.files)
        expected = (("manifest",) + ("stdlib",) * roles.count("stdlib") + ("main",)
                    + ("component",) * roles.count("component") + ("driver",))
        if roles != expected:
            raise ValueError("invalid captured source roles/order")
        paths = tuple(source.path for source in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate captured source path")
        if self.files[0].path != "Mind.toml" or self.by_role("main")[0].path != MAIN:
            raise ValueError("unsupported captured manifest/main path")
        std_paths = tuple(source.path for source in self.by_role("stdlib"))
        if (not self.native_std_paths or len(set(self.native_std_paths)) != len(self.native_std_paths)
                or not set(self.native_std_paths).issubset(std_paths)):
            raise ValueError("invalid native standard-library order/set")
        if any(not path.startswith("std/") or not path.endswith(".mind") for path in std_paths):
            raise ValueError("unsupported standard-library path")
        for source in self.by_role("component") + self.by_role("driver"):
            _path(source.path, component=True)
        manifest = tomllib.loads(self.files[0].data.decode("utf-8"))
        if set(manifest) - {"package", "build", "exports", "mindcraft"}:
            raise ValueError("unsupported internal compiler manifest section")
        build = manifest.get("build", {})
        if not isinstance(build, dict) or set(build) - {"entry", "target", "emit", "optimize"}:
            raise ValueError("unsupported internal compiler manifest source/build option")
        if (build.get("entry") != MAIN or build.get("target") != "cpu"
                or build.get("emit") != "cdylib"):
            raise ValueError("unsupported internal compiler manifest entry/target/emit")

    def by_role(self, role: str) -> tuple[CapturedSource, ...]:
        return tuple(source for source in self.files if source.role == role)

    def source(self, path: str) -> CapturedSource:
        for source in self.files:
            if source.path == path:
                return source
        raise ValueError(f"source absent from captured bundle: {path}")

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(b"MIND_SELFHOST_SOURCE_BUNDLE_v1\0")
        digest.update(struct.pack("<Q", len(self.native_std_paths)))
        for path in self.native_std_paths:
            digest.update(_field(path.encode("utf-8")))
        digest.update(struct.pack("<Q", len(self.files)))
        for source in self.files:
            for value in (source.role.encode(), source.path.encode(), source.data):
                digest.update(_field(value))
        return digest.hexdigest()

    @property
    def library_bytes(self) -> bytes:
        sources = self.by_role("main") + self.by_role("component")
        return b"\n".join(source.data for source in sources)

    @property
    def std_prefix(self) -> bytes:
        return b"\n".join(self.source(path).data for path in self.native_std_paths) + b"\n"

    @property
    def user_lo(self) -> int:
        return len(self.std_prefix)

    @property
    def native_bytes(self) -> bytes:
        return self.std_prefix + self.library_bytes + b"\n" + self.by_role("driver")[0].data

    @property
    def stdin_image(self) -> bytes:
        native = self.native_bytes
        if len(native) > _MAX_I64 or self.user_lo > _MAX_I64:
            raise ValueError("source image length exceeds signed64-bit wire bounds")
        return struct.pack("<qq", self.user_lo, len(native)) + native

    def spans(self, *, native: bool = False) -> tuple[SourceSpan, ...]:
        sources = self.by_role("main") + self.by_role("component")
        if native:
            sources = tuple(self.source(path) for path in self.native_std_paths) + sources + self.by_role("driver")
        result = []
        start = 0
        for source in sources:
            end = start + len(source.data)
            result.append(SourceSpan(source.role, source.path, start, end, source.sha256))
            start = end + 1  # Separators belong to the view, not to either file.
        return tuple(result)

    def cpu_files(self) -> tuple[tuple[str, bytes], ...]:
        return tuple((source.path, source.data) for source in self.files
                     if source.role in ("manifest", "stdlib")) + ((MAIN, self.library_bytes),)

    def materialize_cpu(self, destination: pathlib.Path) -> pathlib.Path:
        destination = pathlib.Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or any(destination.iterdir()):
            raise ValueError("CPU materialization requires an empty owned directory")
        for relative, data in self.cpu_files():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(data)
            target.chmod(0o444)
        return destination / MAIN

    def verify_cpu_materialization(self, destination: pathlib.Path) -> None:
        # Reads owned build copies only, never the original source checkout.
        for relative, data in self.cpu_files():
            if (destination / relative).read_bytes() != data:
                raise ValueError(f"captured CPU input changed during build: {relative}")


def capture_bundle(repo: pathlib.Path, components: tuple[str, ...] = (), *,
                   driver: str = DRIVER,
                   native_std_modules: tuple[str, ...] = NATIVE_STD_MODULES) -> SourceBundle:
    repo = pathlib.Path(repo).resolve(strict=True)
    components = tuple(_path(path, component=True) for path in components)
    driver = _path(driver, component=True)
    native_std_paths = tuple(_path(f"std/{name}.mind") for name in native_std_modules)
    if len(set(native_std_paths)) != len(native_std_paths):
        raise ValueError("duplicate native std source")
    std_paths = tuple(sorted(path.relative_to(repo).as_posix() for path in (repo / "std").glob("*.mind")))
    if not set(native_std_paths).issubset(std_paths):
        raise ValueError("missing native standard-library source")
    requested = (("manifest", "Mind.toml"),) + tuple(("stdlib", path) for path in std_paths)
    requested += (("main", MAIN),) + tuple(("component", path) for path in components) + (("driver", driver),)
    paths = [path for _, path in requested]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate source path or reserved-source collision")
    files = []
    identities = set()
    for role, relative in requested:
        _path(relative)
        path = repo
        for part in relative.split("/"):
            path = path / part
            if path.is_symlink():
                raise ValueError(f"symlink source path is unsupported: {relative}")
        if not path.is_file():
            raise ValueError(f"missing or non-regular source: {relative}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise ValueError(f"aliased source file: {relative}")
        identities.add(identity)
        data = path.read_bytes()  # Exactly one content read per captured file.
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"source is not UTF8: {relative}") from error
        files.append(CapturedSource(role, relative, data))
    return SourceBundle(tuple(files), native_std_paths)


def compiler_library(main: bytes | None = None, repo: pathlib.Path = _REPO) -> bytes:
    """SourceBundle.library_bytes of main.mind (or `main` in its place) + COMPILER_COMPONENTS."""
    bundle = capture_bundle(pathlib.Path(repo), COMPILER_COMPONENTS)
    if main is not None:
        bundle = SourceBundle(tuple(CapturedSource(f.role, f.path, main) if f.role == "main" else f
                                    for f in bundle.files), bundle.native_std_paths)
    return bundle.library_bytes


def single_unit(library: bytes) -> bytes:
    """A composed unit as ONE translation unit for the Rust leg: drop each line `import <part>;`
    naming a COMPILER_COMPONENTS file, whose body the unit holds inline. A standalone compile
    refuses a local import (E2003); a Mind.toml build adds a project struct registry."""
    stems = b"|".join(re.escape(pathlib.PurePosixPath(p).stem).encode() for p in COMPILER_COMPONENTS)
    return re.sub(rb"(?m)^import (?:" + stems + rb");\n", b"", library)


def materialize_library(destination: pathlib.Path, library: bytes, repo: pathlib.Path = _REPO) -> pathlib.Path:
    """Rust-oracle input for a composed compiler unit: an owned Mind.toml project whose
    entry is `library`; the part imports resolve to no sibling, so it compiles as ONE unit.
    Run mindc with cwd=destination: outside a git tree the invocation dir is the root."""
    entry = pathlib.Path(destination) / MAIN
    entry.parent.mkdir(parents=True, exist_ok=True)
    (pathlib.Path(destination) / "Mind.toml").write_bytes((pathlib.Path(repo) / "Mind.toml").read_bytes())
    entry.write_bytes(library)
    return entry


def self_mode_image(image: bytes, compiler: pathlib.Path) -> bytes:
    """The ONE sender of MINDSLF: `image` (the stdin image of the COMPILER SOURCE) with the magic in front,
    when `compiler` carries the magic constant (a seed from before the advance that added it would read
    the magic as user_lo). Self-compile, mint, advance and reseed feed the compiler source through this.
    A user program fed directly to a compiler never gets the magic."""
    return SELF_MAGIC + image if SELF_MAGIC in pathlib.Path(compiler).read_bytes() else image


# Compiler source at run time (the compiler's parts and a driver): a payload that holds the first line of main.mind
# (every bundle holds it), or the whole text of a part of COMPILER_COMPONENTS or of a driver that calls compiler fns.
# The self-test proves that the mint bundle holds every part and that no other testdata program holds one.
COMPILER_SOURCE_MARKER = "// examples/mindc_mind/main.mind — Phase 6.5 Stage 5 APEX driver.".encode()
COMPILER_UNIT_DRIVERS = ("examples/mindc_mind/testdata/native_avx2/encoder_table_driver.mind",
                         "examples/mindc_mind/testdata/native_source_driver/note_fixture_tail.mind",
                         "examples/mindc_mind/testdata/native_string_handle/handle_driver.mind")


class RawCompilerSourceError(RuntimeError):
    """start_child refused a payload: compiler source without the self magic, for a program that may not get it."""


@functools.cache
def _compiler_texts(repo: pathlib.Path = _REPO) -> tuple[bytes, ...]:
    paths = (*COMPILER_COMPONENTS, *COMPILER_UNIT_DRIVERS)
    return (COMPILER_SOURCE_MARKER, *((pathlib.Path(repo) / path).read_bytes().strip() for path in paths))


def holds_compiler_source(data) -> bool:
    """True when `data` (bytes or text) holds a text of _compiler_texts: it is compiler source."""
    data = data.encode() if isinstance(data, str) else bytes(data or b"")
    return any(text in data for text in _compiler_texts())


def compiler_feed(image: bytes, compiler: pathlib.Path) -> bytes:
    """The stdin of `compiler` for `image`: compiler source in self mode (self_mode_image), a user program as it is."""
    return self_mode_image(image, compiler) if holds_compiler_source(image) else image


def raw_compiler_source_error(argv, payload, *, cwd=None, path=None, executable=None) -> str:
    """Why `payload` may not go to `argv` ('' when it may). Compiler source without SELF_MAGIC in front, in the payload
    or in an image file of argv, goes only to an old seed: the program that runs (argv[0] found as exec finds it, or
    `executable`) is a compiler image (it holds the layout magic) without the magic constant. A program that reads the
    magic, a wrapper or a shell in front of a compiler, and a program that can not be read get none (fail closed)."""
    items = [argv] if isinstance(argv, (str, bytes, os.PathLike)) else list(argv)
    images = [payload.encode() if isinstance(payload, str) else bytes(payload or b"")] + _argv_images(items, cwd)
    if not any(holds_compiler_source(data) and not data.startswith(SELF_MAGIC) for data in images):
        return ""
    name = executable or items[0]
    try:
        name = os.fsdecode(name)
        program = pathlib.Path(cwd or ".") / name if "/" in name else pathlib.Path(shutil.which(name, path=path))
        head = program.read_bytes()
    except (OSError, TypeError, ValueError) as error:
        return f"compiler source without the self magic for {name!r}, which can not be read ({error})"
    if SELF_MAGIC in head:
        return f"{name}: compiler source without the self magic: send self_mode_image(...)"
    if LAYOUT_MAGIC.to_bytes(8, "little") not in head:
        return f"{name}: compiler source without the self magic to a program that is no compiler image (a wrapper, a shell)"
    return ""


def _argv_images(items: list, cwd) -> list[bytes]:
    """The images in the files that argv[1:] names (under `cwd`): a driver may read its source image from a file. A
    text file (a source file of mindc) has no binary header and no magic, and is no image."""
    images = []
    for arg in items[1:]:
        try:
            file = pathlib.Path(cwd or ".") / os.fsdecode(arg)
            data = file.read_bytes() if file.is_file() else b""
        except (OSError, TypeError, ValueError):
            continue
        images += [data] if b"\0" in data[:16] or data.startswith(SELF_MAGIC) else []
    return images


def start_child(argv, payload=None, *, popen: bool = False, refusal_control: bool = False, **kwargs):
    """THE one start of a harness child that gets bytes on stdin or an image file on argv; a seed or candidate
    compiler reads its source only there. Any other process start in a tracked Python file is a row of
    PROCESS_STARTS (_process_starts.py). `payload` is the whole stdin (None: no bytes). Before the start,
    raw_compiler_source_error raises RawCompilerSourceError. popen=True: a started Popen that reads the payload from a
    file (for a caller that samples the child). refusal_control=True: the one raw send of compiler source, a control
    that the lane rules refuse it; a run that is not refused by a `lane.` key raises, so it never gives an image."""
    if "input" in kwargs or kwargs.get("stdin") not in (None, subprocess.DEVNULL):
        raise ValueError("start_child: stdin bytes go only through `payload`")
    program = argv if isinstance(argv, (str, bytes, os.PathLike)) else argv[0]
    error = raw_compiler_source_error(argv, payload, cwd=kwargs.get("cwd"), executable=kwargs.get("executable"),
                                      path=(kwargs.get("env") or os.environ).get("PATH"))
    if error and not refusal_control:
        raise RawCompilerSourceError(error)
    if not popen:
        result = subprocess.run(argv, input=payload, **kwargs)
        if refusal_control and (result.returncode == 0 or b"(lane." not in (result.stderr or b"")):
            raise RawCompilerSourceError(f"refusal control: {program} gave exit {result.returncode} with no lane key")
        return result
    if refusal_control:
        raise ValueError("start_child: a refusal control runs to its end (popen=False)")
    if payload is None:
        return subprocess.Popen(argv, **kwargs)
    with tempfile.TemporaryFile() as feed:  # the child keeps its own descriptor of the file
        feed.write(payload.encode() if isinstance(payload, str) else payload)
        feed.seek(0)
        return subprocess.Popen(argv, stdin=feed, **kwargs)


def core_limit_one():
    """The preexec_fn of an image run: the soft core size limit 1, the hard limit unchanged (soft 0 when the hard limit
    is 0). Never min(1, hard): RLIM_INFINITY is -1 on some Python versions, and that min gave an unlimited soft limit.
    At a soft limit of exactly 1 Linux aborts a dump to a core_pattern pipe, and it writes no core file under one page:
    a trap starts no dump handler, and the exit status (a trap is -4) does not change. qemu-user writes its own 1-byte
    core file of a trapped guest into its working directory, so an emulator child runs in a temporary directory.
    The child only sets it; core_limit_check proves the value on one real child."""
    import resource  # POSIX only: the images are Linux programs
    hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
    return lambda: resource.setrlimit(resource.RLIMIT_CORE, (0 if hard == 0 else 1, hard))


def core_limit_check(run_image) -> bool:
    """True when one real child, a script started by `run_image` (the image runner of _selfhost_identity.py, which
    applies core_limit_one), reads the soft core limit 1 (0 when the hard limit is 0); else False after a named FAIL
    line. A gate runs it before its first row: a wrong limit sends every trap to the dump handler of the host."""
    import resource
    want = 0 if resource.getrlimit(resource.RLIMIT_CORE)[1] == 0 else 1
    got = run_image(f"#!{sys.executable} -I\nimport resource\nprint(resource.getrlimit(resource.RLIMIT_CORE)[0])\n".encode(), 60)
    if got is None or got[0] != 0 or got[1].strip() != str(want).encode():
        print(f"FAIL  core limit start check: the child read the soft core limit {got}, not {want}")
        return False
    return True


def _refused(argv, payload, **kwargs) -> bool:
    try:
        start_child(argv, payload, **kwargs)
    except RawCompilerSourceError:
        return True
    except OSError:  # the program was started (and could not run): not refused
        return False
    return False


def _start_child_rows(repo: pathlib.Path) -> list[tuple[str, bool]]:
    """The run-time check of start_child on three small programs: one carries the magic constant, one does not (an
    old seed), one refuses by a lane key. A refused send must not start the program (it writes no file)."""
    library = compiler_library(repo=repo)
    image = struct.pack("<qq", 0, len(library)) + library
    rows = [("start_child: main.mind starts with the compiler-source marker",
             (repo / MAIN).read_bytes().startswith(COMPILER_SOURCE_MARKER)),
            ("start_child: the image of a bundle of main.mind alone holds the marker",
             COMPILER_SOURCE_MARKER in capture_bundle(repo).stdin_image)]
    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td)

        def program(name: str, body: bytes, mark: bytes) -> pathlib.Path:
            path = work / name
            path.write_bytes(b"#!/bin/sh\n# " + mark + b"\n" + body + b"\n")
            path.chmod(0o755)
            return path
        layout = LAYOUT_MAGIC.to_bytes(8, "little")
        candidate = SELF_MAGIC + b" " + layout  # a candidate holds both magics; an old seed the layout magic only
        reads = program("reads", b'cat > "$0.got"', candidate)
        old = program("old", b'cat > "$0.got"', layout)
        plain = program("plain", b"cat > /dev/null", b"")
        lane = program("lane", b'cat > /dev/null; echo "error[native-lane]: x (lane.string.result)" >&2; exit 3', candidate)
        got = work / "reads.got"
        rows.append(("start_child: compiler source without the magic is refused before the program starts",
                     _refused([reads], image, capture_output=True) and not got.exists()))
        sent = self_mode_image(image, reads)
        rows.append(("start_child: compiler_feed sends compiler source in self mode and a user image as it is",
                     compiler_feed(image, reads) == sent and compiler_feed(b"user", reads) == b"user"))
        rows.append(("start_child: the same source through self_mode_image reaches the program",
                     start_child([reads], sent, capture_output=True).returncode == 0 and got.read_bytes() == sent))
        (work / "packet").write_bytes(image)
        rows.append(("start_child: compiler source read back from a file is refused",
                     _refused([str(reads)], (work / "packet").read_bytes(), capture_output=True)))
        rows.append(("start_child: an image file of compiler source on argv is refused",
                     _refused([reads, work / "packet"], None, capture_output=True)))
        (work / "sent").write_bytes(sent)
        rows.append(("start_child: that image file through self_mode_image passes",
                     start_child([reads, work / "sent"], b"", capture_output=True).returncode == 0))
        (work / "source.mind").write_bytes(library)
        rows.append(("start_child: a text source file on argv is no image (mindc reads source files)",
                     start_child([reads, work / "source.mind"], b"", capture_output=True).returncode == 0))
        rows.append(("start_child: the magic later than byte 0 is no self mode",
                     _refused([reads], b"x" + sent, capture_output=True)))
        rows.append(("start_child: compiler source as text is refused",
                     _refused([reads], library.decode(), capture_output=True, text=True)))
        rows.append(("start_child: popen=True refuses compiler source without the magic", _refused([reads], image, popen=True)))
        got.unlink()
        child = start_child([reads], sent, popen=True)
        rows.append(("start_child: popen=True feeds the whole payload", child.wait(timeout=60) == 0 and got.read_bytes() == sent))
        rows.append(("start_child: a program without the magic constant (an old seed) gets the raw source",
                     start_child([old], image, capture_output=True).returncode == 0))
        rows.append(("start_child: an old seed named by a bare name is found on PATH, as exec finds it",
                     start_child(["old"], image, capture_output=True,
                                 env={"PATH": f"{work}{os.pathsep}{os.environ.get('PATH', '')}"}).returncode == 0))
        rows.append(("start_child: a program that is no compiler image gets no raw compiler source",
                     _refused([plain], image, capture_output=True)))
        rows.append(("start_child: a wrapper (env) in front of a program that reads the magic gets no raw source",
                     _refused(["env", str(reads)], image, capture_output=True)))
        rows.append(("start_child: a shell in front of an old seed gets no raw compiler source",
                     _refused(["sh", "-c", f'exec "{old}"'], image, capture_output=True)))
        rows.append(("start_child: executable= names the program that runs",
                     _refused([old], image, executable=str(reads), capture_output=True)))
        parts = b"\n".join((repo / part).read_bytes() for part in COMPILER_COMPONENTS[:2]) + b"\nfn main() -> i64 { 0 }\n"
        rows.append(("start_child: a unit of compiler parts (no main.mind, no driver) without the magic is refused",
                     _refused([reads], struct.pack("<qq", 0, len(parts)) + parts, capture_output=True)))
        rows.append(("start_child: each driver that calls compiler fns, alone, without the magic is refused",
                     all(_refused([reads], (repo / driver).read_bytes(), capture_output=True)
                         for driver in COMPILER_UNIT_DRIVERS)))
        rows.append(("start_child: a program that can not be read gets no raw compiler source",
                     _refused([work / "absent"], image, capture_output=True)))
        rows.append(("start_child: an old seed named by a path under cwd= is found there, as exec finds it",
                     not _refused(["./old"], image, capture_output=True, cwd=work)))
        rows.append(("start_child: an image file of compiler source on argv, named under cwd=, is refused",
                     _refused([reads, "packet"], b"", capture_output=True, cwd=work)))
        rows.append(("start_child: a compiler part without the newline at its end is compiler source",
                     holds_compiler_source((repo / COMPILER_COMPONENTS[0]).read_bytes().rstrip())))
        (work / "user.mind").write_bytes(b"fn main() -> i64 { 0 }\n")
        try:  # source.mind holds the compiler library; reads writes its stdin to reads.got
            fed = [direct_compile(work / n, reads, work / "out", repo).returncode == 0 and got.read_bytes()[:8] for n in ("source.mind", "user.mind")]
        except RawCompilerSourceError:
            fed = []
        rows.append(("start_child: direct_compile sends compiler source in self mode and a user program as it is",
                     len(fed) == 2 and fed[0] == SELF_MAGIC and fed[1] not in (SELF_MAGIC, False)))
        rows.append(("start_child: a user image passes", start_child([reads], b"user", capture_output=True).returncode == 0))
        for name, extra in (("stdin=PIPE", {"stdin": subprocess.PIPE}), ("input=", {"input": image})):
            try:
                start_child([reads], None, **extra)
                bypass = False
            except ValueError:
                bypass = True
            rows.append((f"start_child: stdin bytes through {name} are refused", bypass))
        rows.append(("start_child: a refusal control passes when a lane key refuses the source",
                     start_child([lane], image, capture_output=True, refusal_control=True).returncode == 3))
        rows.append(("start_child: a refusal control that the program does not refuse raises",
                     _refused([reads], image, capture_output=True, refusal_control=True)))
        capture, digest = work / "capture", hashlib.sha256(reads.read_bytes()).hexdigest()
        (capture / "v").mkdir(parents=True)
        (capture / "v" / "input.packet").write_bytes(b"user")
        rows.append(("start_child: bounded_native_builds builds nothing with a seed copy of another sha256",
                     not bounded_native_builds(capture, reads, "0" * 64, ["v/input.packet"])
                     and not (capture / "native" / "v").exists()))
        shutil.rmtree(capture / "native")
        rows.append(("start_child: bounded_native_builds feeds each packet to the seed copy and records its exit",
                     bounded_native_builds(capture, reads, digest, ["v/input.packet"])
                     and (capture / "native/v/native-build.exit").read_text() == "0\n"
                     and (capture / "native/compiler.elf.got").read_bytes() == b"user"))
        shutil.rmtree(capture / "native")
        (capture / "v" / "input.packet").write_bytes(image)
        try:  # the packet holds compiler source and the seed copy reads the magic: start_child refuses it
            bounded_native_builds(capture, reads, digest, ["v/input.packet"])
            raw = True
        except RawCompilerSourceError:
            raw = False
        rows.append(("start_child: bounded_native_builds sends compiler source only in self mode", not raw))
    testdata = sorted((repo / "examples/mindc_mind/testdata").glob("native_*/*.mind"))
    rows += [("start_child: the mint bundle holds the whole text of every compiler part",
              all(text in capture_bundle(repo, COMPILER_COMPONENTS).stdin_image
                  for text in _compiler_texts(repo)[:1 + len(COMPILER_COMPONENTS)])),
             ("start_child: of the testdata programs, only the drivers that call compiler fns hold compiler source",
              [p.relative_to(repo).as_posix() for p in testdata if holds_compiler_source(p.read_bytes())]
              == sorted(COMPILER_UNIT_DRIVERS))]
    return rows


def lane_key_table(compiler: pathlib.Path) -> bytes:
    """The key table a compiler prints on KEY_MAGIC (one `<id> <name>` line per key)."""
    r = start_child([str(compiler)], KEY_MAGIC, capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"{compiler} exited {r.returncode} on KEY_MAGIC: {r.stderr[-200:]!r}")
    return r.stdout


def _run_compiler(compiler: pathlib.Path, image: bytes, timeout: int = 900, *, refusal_control: bool = False) -> bytes:
    result = start_child([str(compiler)], image, capture_output=True, timeout=timeout, refusal_control=refusal_control)
    if result.returncode != 0 or result.stdout[:4] != b"\x7fELF":
        raise RuntimeError(f"{compiler} exited {result.returncode} with {len(result.stdout)} B: "
                           f"{result.stderr[-300:]!r}")
    return result.stdout


def _magic_control(run, with_magic: bytes, without_magic: bytes, hop2: bytes) -> None:
    """The mint control. `run(image)` returns the compiler's ELF or raises RuntimeError with its stderr.
    With the magic the image must be hop 2; without it the run must be refused and the refusal names a
    `lane.` key (a refusal of another kind, or an image, is a failure). A sender that stops sending the
    magic fails the first line."""
    if with_magic == without_magic:
        raise RuntimeError("the sender did not put the self magic in front of the compiler source")
    if run(with_magic) != hop2:
        raise RuntimeError("the compiler source with the self magic does not give hop 2 again")
    try:
        run(without_magic)
    except RuntimeError as e:
        if "(lane." not in str(e):
            raise RuntimeError(f"the compiler source without the self magic was refused, but not by a lane key: {e}")
        return
    raise RuntimeError("the compiler source without the self magic was not refused by the lane rules")


def _magic_control_selftest() -> bool:
    """The control must pass on the real shape and fail on each of its three breaks."""
    def make(refuse):
        def run(data):
            if data.startswith(SELF_MAGIC):
                return b"\x7fELFhop2"
            if refuse is None:
                return b"\x7fELFhop2"
            raise RuntimeError(refuse)
        return run
    def verdict(run, with_magic, without_magic):
        try:
            _magic_control(run, with_magic, without_magic, b"\x7fELFhop2")
            return True
        except RuntimeError:
            return False
    src, sent = b"source", SELF_MAGIC + b"source"
    lane = "exited 3: b'error[native-lane]: x (lane.string.result) in fn string_new'"
    return (verdict(make(lane), sent, src)                              # the real shape
            and not verdict(make(lane), src, src)                       # the helper sends no magic
            and not verdict(make(None), sent, src)                      # without the magic: an image
            and not verdict(make("exited 3: b'boom'"), sent, src))      # refused, not by a lane key


def mint_stage2(out: pathlib.Path, *, assert_fixed_point: bool = False, repo: pathlib.Path = _REPO,
                layout: tuple[int, int, int] = BOUNDED_LAYOUT) -> str:
    """Rehearsal compiler: frozen seed -> stage1 -> stage2 over the bundle WITH the
    parts (-> stage3 == stage2 when asserted). Never publishes: refuses testdata/."""
    out = pathlib.Path(out).resolve()
    if (pathlib.Path(repo).resolve() / "examples/mindc_mind/testdata") in out.parents:
        raise ValueError("refusing to write a rehearsal compiler under testdata/ (no seed moves)")
    if assert_fixed_point and not _magic_control_selftest():
        raise RuntimeError("the mint control did not catch its planted breaks")
    image = struct.pack("<qqqq", LAYOUT_MAGIC, *layout) + capture_bundle(repo, COMPILER_COMPONENTS).stdin_image
    with tempfile.TemporaryDirectory() as work:
        stages, compiler = [], pathlib.Path(repo) / FROZEN_SEED
        for k in (1, 2, 3) if assert_fixed_point else (1, 2):
            stages.append(_run_compiler(compiler, self_mode_image(image, compiler)))
            compiler = pathlib.Path(work) / f"stage{k}.elf"
            compiler.write_bytes(stages[-1])
            compiler.chmod(0o755)
        if assert_fixed_point and stages[2] != stages[1]:
            raise RuntimeError("stage2 != stage3: the composed compiler is not a fixed point")
        if assert_fixed_point:
            # The minted compiler must carry the magic, and the magic must change no image.
            if SELF_MAGIC not in stages[0]:
                raise RuntimeError("stage1 does not carry the self magic: nb_main_mode never compares it")
            # The positive control: WITH the magic the compiler source gives hop 2 again (stage1 on the sent
            # image == stages[1]); WITHOUT it the lane rules refuse it, by a `lane.` key.
            _magic_control(lambda data: _run_compiler(pathlib.Path(work) / "stage1.elf", data, refusal_control=data is image),
                           self_mode_image(image, pathlib.Path(work) / "stage1.elf"), image, stages[1])
    out.write_bytes(stages[1])
    out.chmod(0o755)
    return hashlib.sha256(stages[1]).hexdigest()


def direct_compile(program: pathlib.Path, compiler: pathlib.Path, out: pathlib.Path,
                   repo: pathlib.Path = _REPO) -> subprocess.CompletedProcess:
    """[8B user_lo][8B src_len][std prefix + program] on `compiler` (selfhost_driver.mind stdin protocol), through
    compiler_feed: a unit that holds compiler source goes in self mode; writes `out` only when the compiler emitted an ELF."""
    std = capture_bundle(repo).std_prefix
    source = std + pathlib.Path(program).read_bytes()
    return _keep_elf(start_child([str(compiler)], compiler_feed(struct.pack("<qq", len(std), len(source)) + source, compiler),
                                 capture_output=True, timeout=600), out)


def direct_compile_cli(program: str, compiler: str | None, out: str | None) -> int:
    """`--direct-compile` of the rehearsal smokes: one program on one compiler ELF; the exit code."""
    if not (compiler and out):
        print("BLOCKED: --direct-compile needs --compiler ELF and --out ELF")
        return 2
    r = direct_compile(pathlib.Path(program), pathlib.Path(compiler), pathlib.Path(out))
    sys.stderr.write(r.stderr.decode(errors="replace"))
    print(f"DIRECT rc={r.returncode} out={'written' if pathlib.Path(out).exists() else 'none'}")
    return r.returncode


def driver_compile(driver: str, compiler: pathlib.Path, out: pathlib.Path,
                   repo: pathlib.Path = _REPO) -> subprocess.CompletedProcess:
    """The minted unit (std + main.mind + COMPILER_COMPONENTS) with the repo-relative `driver`
    in place of selfhost_driver.mind, on `compiler`: a check driver calls compiler fns directly."""
    image = struct.pack("<qqqq", LAYOUT_MAGIC, *BOUNDED_LAYOUT) + capture_bundle(
        repo, COMPILER_COMPONENTS, driver=driver).stdin_image
    return _keep_elf(start_child([str(compiler)], self_mode_image(image, compiler),
                                 capture_output=True, timeout=1800), out)


def _keep_elf(result: subprocess.CompletedProcess, out: pathlib.Path) -> subprocess.CompletedProcess:
    if result.returncode == 0 and result.stdout[:4] == b"\x7fELF":
        pathlib.Path(out).write_bytes(result.stdout)
        pathlib.Path(out).chmod(0o755)
    return result



def bounded_native_builds(capture: pathlib.Path, seed: pathlib.Path, seed_sha256: str, packets: list[str]) -> bool:
    """The seed builds of a manual native check (the variadic plan and emit checks), each through start_child: a copy
    of `seed` (capture/native/compiler.elf, mode 500) builds each captured packet (a path under `capture`) into
    native.elf, native-build.stderr and native-build.exit of capture/native/<its directory>, in the bounds of the
    check: address space 2 GiB, no core file, 180 CPU seconds and 240 wall seconds (exit 124 at the wall limit, as
    timeout(1) gives; 128 + N for signal N, as a shell gives). False, and no build, when the sha256 of the copy is not
    `seed_sha256`."""
    import resource  # POSIX only: the manual native checks run on Linux

    def limits() -> None:
        for kind, value in ((resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_CORE, 0), (resource.RLIMIT_CPU, 180)):
            resource.setrlimit(kind, (value, value))
    native = capture / "native"
    native.mkdir()
    compiler = native / "compiler.elf"
    shutil.copyfile(seed, compiler)
    compiler.chmod(0o500)
    if hashlib.sha256(compiler.read_bytes()).hexdigest() != seed_sha256:
        return False
    for packet in packets:
        dest = native / pathlib.PurePosixPath(packet).parent
        dest.mkdir()
        with (dest / "native.elf").open("wb") as elf, (dest / "native-build.stderr").open("wb") as err:
            try:
                rc = start_child([str(compiler)], (capture / packet).read_bytes(), stdout=elf, stderr=err,
                                 timeout=240, preexec_fn=limits).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        (dest / "native-build.exit").write_text(f"{128 - rc if rc < 0 else rc}\n", encoding="utf-8")
    return True
