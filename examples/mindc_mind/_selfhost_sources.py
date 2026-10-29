"""Immutable source views for internal compiler bootstrap orchestration.

This is not the product import resolver. The caller owns an immutable checkout
while capture runs; capture is not an atomic filesystem transaction. Later view
construction and materialization never read that checkout again.
"""

from dataclasses import dataclass
import hashlib
import pathlib
import re
import struct
import tomllib


NATIVE_STD_MODULES = (
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
)
MAIN = "examples/mindc_mind/main.mind"
DRIVER = "examples/mindc_mind/selfhost_driver.mind"
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
