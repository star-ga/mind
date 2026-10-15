"""Shared self-host `.so` resolver for the examples/mindc_mind smokes.

Why this exists
---------------
`examples/mindc_mind/libmindc_mind.so` is a BUILD ARTIFACT — it is untracked and
gitignored (`examples/**/*.so`). Crucially, `cargo build` does NOT produce it;
only `mindc build --emit=cdylib` does. So an in-tree copy left over from an
older build goes STALE and silently false-fails the byte-identity smokes
(mic@3 flip, self-host loop, native-ELF, ...): the driver bytes drift from the
oracle even though nothing is actually wrong. Worse in the other direction, a
stale oracle that HAPPENS to agree with an equally stale expectation certifies
byte-identity for a compiler nobody is building any more.

`resolve_so()` gives a purely-local run the same guarantee CI has:

  1. `MINDC_SO` set          -> use it, AFTER verifying it is not older than the
                                compile inputs (stamp sidecar, else mtime).
  2. else, release `mindc`   -> BUILD THE SELF-HOST `.so` FRESH via
     binary present             `mindc build --release --emit=cdylib` into a
                                per-tree temp cache (rebuilt only when `mindc`,
                                `main.mind`, `selfhost_driver.mind`, `Mind.toml`
                                or any `std/*.mind` change).
  3. else                    -> the legacy in-tree path, tagged as such.

Provenance, and why it is enforced HERE
---------------------------------------
Every route returns a `ResolvedSo` — a `pathlib.Path` that CARRIES where it came
from (`FRESH_BUILD` / `ENV_VERIFIED` / `ENV_STALE` / `LEGACY_IN_TREE`). The
qualification used to be advisory ("callers that draw a CONCLUSION must qualify
it") and exactly ONE of ~47 importers did so, while the `MINDC_SO` route set no
marker at all — so pointing `MINDC_SO` at the legacy artifact laundered it as a
promised real oracle and sibling gates printed `ALL PASS` on bytes the self-host
loop gate, given the same file, refuses. A contract that every call site must
remember is not a contract.

So `resolve_so()` is STRICT BY DEFAULT: a non-fresh oracle is REFUSED
(`SystemExit`) and no importer has to opt in to that. A genuinely age-tolerant
caller opts out explicitly and NAMES its reason:

    so = resolve_so(allow_stale=True, reason="only reads the ELF header")

Escape hatch: `MINDC_SO_NOBUILD=1` still selects the legacy in-tree artifact
without building — but it is tagged `LEGACY_IN_TREE`, so it is refused unless
the caller also passes `allow_stale=True`. Gate:
`examples/mindc_mind/selfhost_so_provenance_smoke.py`.
"""

import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import struct

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
_LEGACY_SO = _HERE / "libmindc_mind.so"
_MINDC = _REPO / "target" / "release" / "mindc"

#: Provenance tags carried by every resolved path.
FRESH_BUILD = "fresh-build"  # emitted this run (or cached against an identical stamp)
ENV_VERIFIED = "env-verified"  # MINDC_SO, proven not older than the compile inputs
ENV_STALE = "env-stale"  # MINDC_SO, older than a compile input -> untrustworthy
LEGACY_IN_TREE = "legacy-in-tree"  # in-tree artifact of unknown age

#: The provenances a byte-identity verdict may rest on.
FRESH = frozenset({FRESH_BUILD, ENV_VERIFIED})

#: Back-compat view for callers that already qualified their verdict (kept so the
#: self-host loop gate's fallback wording keeps working). It is now DERIVED from
#: the provenance of the last resolution, not a second bookkeeping variable.
USED_LEGACY_FALLBACK = False

#: Provenance of the most recent `resolve_so()` result (None before the first).
PROVENANCE: str | None = None


class ResolvedSo(pathlib.Path):
    """A resolved `.so` path that carries its own provenance.

    A `Path` subclass so all ~47 existing call sites keep using it as a path
    (`str(so)`, `so.exists()`, `ctypes.CDLL(str(so))`) while a verdict-drawing
    caller can ask `so.is_fresh` instead of consulting a module global it has to
    know about. Class-level defaults keep derived paths (`so.parent / "x"`)
    from raising.
    """

    provenance: str = LEGACY_IN_TREE
    detail: str = "derived path — provenance not carried"

    def __init__(self, *args, provenance: str = LEGACY_IN_TREE, detail: str = ""):
        super().__init__(*args)
        self.provenance = provenance
        self.detail = detail

    @property
    def is_fresh(self) -> bool:
        """True only when a byte-identity verdict may rest on these bytes."""
        return self.provenance in FRESH


def _stamp_inputs() -> list[pathlib.Path]:
    """EVERY compile input that affects the emitted self-host `.so`.

    One list, two consumers (`stamp()` and `_newest_input_mtime_ns()`), so the
    freshness check can never police a narrower set than the cache key does.
    main.mind imports std.vec / std.map / std.string / std.io, so a std/ edit
    changes the emitted `.so`; the whole of std/ is stamped rather than the four
    current imports, so adding an import cannot silently narrow the fingerprint.
    """
    return [
        _MINDC,
        _HERE / "main.mind",
        _HERE / "selfhost_driver.mind",
        _REPO / "Mind.toml",
        *sorted((_REPO / "std").glob("*.mind")),
    ]


def stamp() -> str:
    """Fingerprint of every input that affects the emitted self-host `.so`."""
    parts = []
    for p in _stamp_inputs():
        try:
            st = p.stat()
            parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{p}:MISSING")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


#: Historical private name; `stamp()` is the public one (the gate calls it).
_stamp = stamp


def _newest_input_mtime_ns() -> tuple[int, pathlib.Path | None]:
    """(mtime_ns, path) of the most recently modified compile input."""
    newest, which = 0, None
    for p in _stamp_inputs():
        try:
            m = p.stat().st_mtime_ns
        except OSError:
            continue
        if m > newest:
            newest, which = m, p
    return newest, which


def _env_provenance(p: pathlib.Path) -> tuple[str, str]:
    """Classify a caller-supplied `MINDC_SO`: (provenance, human detail).

    Two independent ways to be fresh, because CI and a local build differ:
      * a `<so>.stamp` sidecar matching the current tree stamp — what
        `_build_fresh()` writes, and survives an mtime-rewriting checkout;
      * an mtime not older than every compile input — what CI's build-then-run
        step produces, with no sidecar at all.
    Anything else is `ENV_STALE`: it may predate the sources it claims to
    compile, and a byte verdict from it cannot tell drift from age.
    """
    sidecar = p.with_name(p.name + ".stamp")
    try:
        if sidecar.is_file() and sidecar.read_text().strip() == stamp():
            return ENV_VERIFIED, f"stamp sidecar {sidecar.name} matches the current tree"
    except OSError:
        pass
    newest, which = _newest_input_mtime_ns()
    try:
        so_m = p.stat().st_mtime_ns
    except OSError:
        return ENV_STALE, "could not stat the artifact"
    if so_m >= newest:
        return ENV_VERIFIED, "artifact is newer than every compile input"
    return (
        ENV_STALE,
        f"artifact mtime is OLDER than {which} "
        f"(artifact {so_m} < input {newest})",
    )


def _build_fresh() -> pathlib.Path | None:
    """Emit the self-host cdylib fresh (cached by input stamp). None on failure."""
    cache = pathlib.Path(tempfile.gettempdir()) / "mindc_mind_selfhost_cache"
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / "libmindc_mind.so"
    stampf = cache / "stamp"
    want = stamp()
    if so.exists() and stampf.is_file() and stampf.read_text() == want:
        return so
    proc = subprocess.run(
        [str(_MINDC), "build", "--release", "--emit=cdylib", f"--out={so}"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not so.exists():
        print(
            "WARN[_selfhost_so]: fresh `mindc build --emit=cdylib` failed "
            f"(rc={proc.returncode}) — falling back to legacy in-tree .so.\n"
            f"  stderr: {proc.stderr[-400:]}",
            file=sys.stderr,
        )
        return None
    stampf.write_text(want)
    # Sidecar next to the artifact so a caller that later passes this path
    # through MINDC_SO can prove its freshness without re-deriving it.
    try:
        so.with_name(so.name + ".stamp").write_text(want)
    except OSError:
        pass
    return so


def _refuse(so: ResolvedSo) -> None:
    """Fail CLOSED on an oracle that cannot support a byte-identity verdict."""
    how = (
        "unset MINDC_SO and let this resolver build one, or emit it yourself:\n"
        f"  cargo build --release --bin mindc --features "
        f"mlir-build,std-surface,cross-module-imports\n"
        f"  ./target/release/mindc build --release --emit=cdylib --out=/tmp/oracle.so\n"
        f"  MINDC_SO=/tmp/oracle.so <re-run this gate>"
    )
    raise SystemExit(
        f"FAIL[_selfhost_so]: refusing a NON-FRESH self-host oracle.\n"
        f"  path       : {so}\n"
        f"  provenance : {so.provenance}\n"
        f"  why        : {so.detail}\n"
        f"  A verdict drawn from these bytes cannot distinguish real drift from "
        f"an old artifact, and an agreement between two equally old artifacts "
        f"proves nothing about the current source. Build a fresh oracle: {how}\n"
        f"  (A caller that genuinely tolerates age must say so in code: "
        f"resolve_so(allow_stale=True, reason=...).)"
    )


def resolve_so(*, allow_stale: bool = False, reason: str = "") -> ResolvedSo:
    """Resolve the self-host `.so`, REFUSING a non-fresh oracle by default.

    `allow_stale=True` requires a non-empty `reason`: an opt-out from the
    freshness contract has to name itself, so it is reviewable and greppable
    rather than a silent default nobody notices.
    """
    global USED_LEGACY_FALLBACK, PROVENANCE
    if allow_stale and not reason.strip():
        raise SystemExit(
            "FAIL[_selfhost_so]: resolve_so(allow_stale=True) requires reason=... "
            "— an opt-out from the stale-oracle refusal must name why this caller "
            "does not draw a byte-identity verdict from the artifact."
        )

    env = os.environ.get("MINDC_SO")
    if env:
        # Fail CLOSED on an explicit-but-missing MINDC_SO. Setting it is a promise
        # that a real `.so` is there and the gate must run for real -- it is how
        # ci.yml, fast_keystone.sh and preflight.sh all invoke these smokes. Handing
        # the path back regardless left each of ~46 importers to decide, and eight of
        # them printed `SKIP ... not built` and returned 0: a broken wiring graded as
        # a green gate. `allow_stale` does NOT relax this: absence is not age.
        p = pathlib.Path(env)
        if not p.is_file():
            PROVENANCE = None
            raise SystemExit(
                f"FAIL[_selfhost_so]: MINDC_SO is set to {env!r} but no such file "
                f"exists. Refusing to skip -- with MINDC_SO set, a skip would grade "
                f"a broken .so wiring as a passing gate. Build it with "
                f"`mindc build --release --emit=cdylib --out={env}`, or unset "
                f"MINDC_SO to let this resolver build a fresh one."
            )
        prov, detail = _env_provenance(p)
        # The route that used to set NO marker at all. An explicit MINDC_SO is a
        # promise that a real oracle is there, but a promise is not a timestamp:
        # preflight itself pointed it at the legacy in-tree artifact, and the
        # unmarked path laundered those months-old bytes as a fresh oracle.
        so = ResolvedSo(p, provenance=prov, detail=detail)
    elif os.environ.get("MINDC_SO_NOBUILD") or not _MINDC.exists():
        # Second route to the legacy artifact (explicit escape hatch, or no
        # release mindc to build with). The REASON differs from the build-failure
        # route below, but the consequence is identical: these bytes may be
        # arbitrarily old.
        so = ResolvedSo(
            _LEGACY_SO,
            provenance=LEGACY_IN_TREE,
            detail=(
                "MINDC_SO_NOBUILD was set"
                if os.environ.get("MINDC_SO_NOBUILD")
                else f"no release mindc at {_MINDC} to build a fresh oracle with"
            ),
        )
    else:
        fresh = _build_fresh()
        if fresh is not None:
            so = ResolvedSo(
                fresh, provenance=FRESH_BUILD, detail="emitted from the current tree"
            )
        else:
            # Fell back to the legacy in-tree artifact. It WARNs above; the caller
            # still gets a path tagged for what it is.
            so = ResolvedSo(
                _LEGACY_SO,
                provenance=LEGACY_IN_TREE,
                detail="`mindc build --emit=cdylib` failed (see the WARN above)",
            )

    PROVENANCE = so.provenance
    USED_LEGACY_FALLBACK = not so.is_fresh
    if not so.is_fresh and not allow_stale:
        _refuse(so)
    return so


def resolve_mindc() -> str:
    """Resolve the `mindc` binary UNDER TEST. Fail-closed, never a PATH lookup.

    Why this exists (and why it must not consult PATH)
    --------------------------------------------------
    26 gates in this directory resolved the compiler as
    `os.environ.get("MINDC_BIN", "mindc")`. The default is a BARE NAME, so with
    no handle exported the gate ran whatever `mindc` the shell's PATH happened
    to point at — on a developer box, a symlink into a DIFFERENT checkout's
    `target/release/`. Measured: a pristine `git archive HEAD` copy with no
    `target/` at all, every handle unset, still had 21 compiler-dependent gates
    report `ALL PASS / asserted=1` — they were exercising another tree's binary,
    not the tree under test. A PATH lookup must never be able to select the
    subject under test.

    Order: `MINDC_BIN`, then `MINDC` (both already exported by ci.yml and
    preflight.sh), then this tree's `target/release/mindc`. A handle that is set
    but missing, and an unset handle with no built binary, are BOTH a hard exit —
    the gate cannot state anything about a compiler it never ran, and a SKIP here
    would grade a broken wiring as a green gate.
    """
    env = os.environ.get("MINDC_BIN") or os.environ.get("MINDC")
    p = pathlib.Path(env) if env else _MINDC
    if not p.is_file():
        raise SystemExit(
            f"FAIL[_selfhost_so]: mindc not found at {p}"
            + (" (from $MINDC_BIN/$MINDC)" if env else " (this tree's release build)")
            + ". Refusing to fall back to a PATH `mindc`: that would run another "
            "checkout's compiler and report a green gate about a tree that never "
            "built one. Build it with `cargo build --release --bin mindc`, or "
            "point $MINDC_BIN at the binary under test."
        )
    return str(p)


def build_captured_so(bundle, compiler: pathlib.Path, compiler_sha256: str,
                      cache_dir: pathlib.Path, *, timeout: int = 180,
                      dependency_files: tuple[pathlib.Path, ...] = (),
                      reuse: bool = False) -> ResolvedSo:
    """Build the captured library, never rereading its original source tree.

    Strict opt-in companion to resolve_so: no environment oracle override or
    mtime fallback. Fresh build is the default. Cache reuse additionally requires
    the caller's complete, pinned CPU toolchain file list; its exact bytes enter
    the key. This list is a caller obligation, not automatic dependency discovery
    or mathematical authority. The cache and toolchain are trusted owner inputs.
    Linux CPU bootstrap orchestration only; ordinary project builds are unchanged.
    """
    import fcntl
    from _selfhost_sources import SourceBundle

    if not isinstance(bundle, SourceBundle):
        raise ValueError("expected captured SourceBundle")
    if timeout <= 0:
        raise ValueError("CPU build timeout must be positive")
    if reuse and not dependency_files:
        raise ValueError("cache reuse requires an explicit pinned CPU dependency closure")
    compiler = pathlib.Path(compiler).resolve(strict=True)

    def file_hash(path):
        with pathlib.Path(path).open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()

    actual_compiler = file_hash(compiler)
    if actual_compiler != compiler_sha256:
        raise ValueError("captured CPU compiler hash mismatch")
    dependencies = tuple((pathlib.Path(path).resolve(strict=True), file_hash(path))
                         for path in dependency_files)
    identity = hashlib.sha256(b"MIND_CAPTURED_CPU_BUILD_v1\0")
    fields = (bundle.identity, actual_compiler, "build --release --emit=cdylib --out=library.so")
    for value in fields + tuple(str(path) + "\0" + digest for path, digest in dependencies):
        encoded = value.encode("utf-8")
        identity.update(struct.pack("<Q", len(encoded)) + encoded)
    key = identity.hexdigest()
    cache_dir = pathlib.Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact = cache_dir / (key + ".so")
    receipt = cache_dir / (key + ".sha256")
    with (cache_dir / (key + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if reuse and artifact.is_file() and receipt.is_file():
            expected = receipt.read_text().splitlines()
            if expected != [key, file_hash(artifact)]:
                raise ValueError("captured CPU cache artifact identity mismatch")
            return ResolvedSo(artifact, provenance=FRESH_BUILD,
                              detail="captured source/compiler/dependency key and artifact hash verified")
        work = pathlib.Path(tempfile.mkdtemp(prefix=key[:16] + "-", dir=cache_dir))
        source_root = work / "source"
        bundle.materialize_cpu(source_root)
        output = work / "library.so"
        env = dict(os.environ)
        env["MIND_STDLIB_PATH"] = str(source_root / "std")
        command = [str(compiler), "build", "--release", "--emit=cdylib", f"--out={output}"]
        try:
            result = subprocess.run(command, cwd=source_root, env=env, capture_output=True,
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            (work / "build.stdout").write_bytes(error.stdout or b"")
            (work / "build.stderr").write_bytes(error.stderr or b"")
            (work / "build.exit").write_text("124\n")
            (work / "command.txt").write_text("\n".join(command) + "\n")
            (work / "source.identity").write_text(bundle.identity + "\n")
            raise RuntimeError(f"captured CPU build timed out; evidence: {work}") from error
        (work / "build.stdout").write_bytes(result.stdout)
        (work / "build.stderr").write_bytes(result.stderr)
        (work / "build.exit").write_text(str(result.returncode) + "\n")
        (work / "command.txt").write_text("\n".join(command) + "\n")
        (work / "source.identity").write_text(bundle.identity + "\n")
        bundle.verify_cpu_materialization(source_root)
        if file_hash(compiler) != actual_compiler:
            raise ValueError("CPU compiler changed during captured build")
        if any(file_hash(path) != digest for path, digest in dependencies):
            raise ValueError("CPU toolchain dependency changed during captured build")
        if result.returncode != 0:
            raise RuntimeError(f"captured CPU build failed with exit {result.returncode}; evidence: {work}")
        if not output.is_file() or output.stat().st_size < 64:
            raise RuntimeError(f"captured CPU build produced no ELF library; evidence: {work}")
        with output.open("rb") as binary:
            if binary.read(4) != b"\x7fELF":
                raise RuntimeError(f"captured CPU build produced a non-ELF library; evidence: {work}")
        digest = file_hash(output)
        os.replace(output, artifact)
        staged_receipt = work / "artifact.sha256"
        staged_receipt.write_text(key + "\n" + digest + "\n")
        os.replace(staged_receipt, receipt)
        return ResolvedSo(artifact, provenance=FRESH_BUILD,
                          detail=f"built captured source closure {bundle.identity}; evidence: {work}")
