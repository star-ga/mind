"""Self-host LOOP reproduction and advancement gate.

The executable contract and operator run modes are documented in
self_host_loop_smoke.md beside this file. Keep this entrypoint small and
auditable: it builds the seeded stdin, runs the frozen pure-MIND chain, checks
the fresh oracle, and publishes only an agreed fixed point.
"""

import ctypes
import hashlib
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
_DEFAULT_SO = _HERE / "libmindc_mind.so"  # legacy in-tree path (fallback only)
# MINDC_SO (CI) verbatim; else build the self-host .so FRESH — never trust a
# stale in-tree libmindc_mind.so (a cargo build does not regenerate it).
sys.path.insert(0, str(_HERE))
from _selfhost_so import ResolvedSo, resolve_so  # noqa: E402
from _selfhost_sources import NATIVE_STD_MODULES, SourceBundle, capture_bundle  # noqa: E402

_SO: ResolvedSo | None = None


def so() -> ResolvedSo:
    """The self-host `.so` oracle, resolved ON FIRST USE and cached.

    Resolution is NOT done at import time. `resolve_so()` may build a fresh
    oracle and raises SystemExit on a missing or stale explicit oracle. Each
    mode resolves it only when needed; importing `build_seed` never resolves
    or builds an oracle. Existence/freshness guards after this call are
    defensive checks for callers that supply an already-resolved handle.
    """
    global _SO
    if _SO is None:
        _SO = resolve_so()
    return _SO

_STDLIB_MODULES = list(NATIVE_STD_MODULES)

_FROZEN = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
_FROZEN_MANIFEST = _HERE / "testdata" / "selfhost_loop" / "MANIFEST.txt"

RESEED_ENV = "MIND_SELFHOST_RESEED"
ADVANCE_ENV = "MIND_SELFHOST_ADVANCE"

#: The single replacement primitive every published file goes through. Named so
#: the "no cross-path atomicity" claim in `publish_fixture` is auditable at one
#: place, and so the rollback path can be exercised by a test without a
#: filesystem the harness cannot create.
_replace = os.replace


def requested_modes(argv: list[str], env) -> tuple[list[str], list[str]]:
    """(advance_sources, reseed_sources) — HOW each mode was requested.

    Returned as source lists rather than booleans so the conflict message can
    name the flag and the environment variable separately; an operator with
    `MIND_SELFHOST_RESEED=1` exported in their shell and `--advance` on the
    command line otherwise sees a refusal that does not say which two things
    collided.
    """
    advance = ["--advance"] if "--advance" in argv else []
    if env.get(ADVANCE_ENV) == "1":
        advance.append(f"{ADVANCE_ENV}=1")
    reseed = ["--reseed"] if "--reseed" in argv else []
    if env.get(RESEED_ENV) == "1":
        reseed.append(f"{RESEED_ENV}=1")
    return advance, reseed


def mode_conflict(advance: list[str], reseed: list[str]) -> str | None:
    """The refusal text when both publication modes are requested, else None.

    The two modes mint the frozen bootstrap from DIFFERENT seeds — --advance
    from the previous pure-MIND compiler, --reseed from Rust output — so
    "both" has no meaning and picking one silently would decide the trust
    chain for the operator. Checked before anything is read, run or written.
    """
    if not (advance and reseed):
        return None
    return (
        f"BLOCKED: --advance and --reseed are mutually exclusive, and both were "
        f"requested ({', '.join(advance)} and {', '.join(reseed)}). They mint the "
        f"frozen bootstrap from different seeds: --advance from the EXISTING "
        f"pure-MIND stage0 (trust chain continuous), --reseed from Rust output "
        f"(chain broken for that hop). Nothing has been read, run or written. "
        f"Pick one, and unset the environment form if it is not the one you meant."
    )


def build_seed(bundle: SourceBundle | None = None) -> tuple[bytes, bytes, int]:
    """Return (combined_source, stdin_image, user_lo). combined_source is the exact
    byte stream compiled; stdin_image is [8B user_lo][8B src_len][combined_source]."""
    if bundle is None:
        bundle = capture_bundle(_REPO, native_std_modules=tuple(_STDLIB_MODULES))
    return bundle.native_bytes, bundle.stdin_image, bundle.user_lo


def stage0_emit(combined: bytes, user_lo: int) -> bytes:
    """Rust `.so` (the DRIFT ORACLE / re-freeze seed) emits an ELF from the source."""
    lib = ctypes.CDLL(str(so()))
    lib.selftest_native_elf_u.restype = ctypes.c_int64
    lib.selftest_native_elf_u.argtypes = [ctypes.c_int64] * 3
    buf = ctypes.create_string_buffer(combined, len(combined))
    es = lib.selftest_native_elf_u(
        ctypes.cast(buf, ctypes.c_void_p).value, len(combined), user_lo
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    if not sh or rd(sh, 8) <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf_path: pathlib.Path, stdin_image: bytes) -> bytes:
    """Run the native ELF with the seeded stdin; return its stdout (the emitted ELF).
    Raises on non-zero exit or empty output (fail-closed)."""
    r = subprocess.run(
        [str(elf_path)], input=stdin_image, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=int(os.environ.get("MIND_SELFHOST_TIMEOUT_SECS", "120")),
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"{elf_path.name} exited {r.returncode} (signal/segfault?) — "
            f"stderr={r.stderr[:200]!r}"
        )
    if not r.stdout:
        raise RuntimeError(f"{elf_path.name} emitted no bytes")
    return r.stdout


class LoopFailure(RuntimeError):
    """A bootstrap-loop stage failed, carrying WHICH stage it was.

    The stage number is the whole point. Attributing a stage2 or stage3 failure
    to "the old seed refused the source" is false — by then the old seed has
    already compiled the source successfully — and a false attribution sends the
    reader to the wrong file. A `RuntimeError` subclass so the ordinary
    verification path's existing handler keeps catching it unchanged.
    """

    def __init__(self, stage: int, detail: str) -> None:
        super().__init__(f"stage{stage}: {detail}")
        self.stage = stage
        self.detail = detail


def is_static_elf(b: bytes) -> bool:
    return len(b) > 4096 and b[:4] == b"\x7fELF" and b[16:18] == b"\x02\x00"  # ET_EXEC


def _write_exec(dir_: pathlib.Path, name: str, data: bytes) -> pathlib.Path:
    p = dir_ / name
    p.write_bytes(data)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _derive(stage: int, exe: pathlib.Path, stdin_image: bytes) -> bytes:
    """Run one hop of the loop and screen its output, as a numbered stage.

    Every way this can go wrong — a non-zero exit, no output, a spawn error, a
    timeout — becomes a `LoopFailure` naming the stage. The ELF screen happens
    HERE, before the caller can write the bytes out and execute them: handing an
    unscreened image to `_write_exec` and then to `execve` is how a truncated or
    garbage emission became a spawn error attributed to the next stage.
    """
    try:
        out = run_elf(exe, stdin_image)
    except RuntimeError as e:
        raise LoopFailure(stage, str(e)) from e
    except subprocess.SubprocessError as e:  # includes TimeoutExpired
        raise LoopFailure(stage, f"{exe.name} did not complete: {e}") from e
    except OSError as e:
        raise LoopFailure(stage, f"could not execute {exe.name}: {e}") from e
    if not is_static_elf(out):
        raise LoopFailure(stage, f"emitted a non-ELF/short image ({len(out)}B)")
    return out


def run_loop_from_seed(seed_exe: pathlib.Path, stdin_image: bytes,
                       td: pathlib.Path) -> tuple[bytes, bytes, bytes]:
    """Given an executable seed ELF, produce (stage1, stage2, stage3):
    stage1 = seed(stdin); stage2 = stage1(stdin); stage3 = stage2(stdin).

    Raises `LoopFailure` naming the stage that failed. Each stage's output is
    screened as a static ELF before it is written and executed as the next
    stage's seed.
    """
    stage1 = _derive(1, seed_exe, stdin_image)
    p1 = _write_exec(td, "stage1.elf", stage1)
    stage2 = _derive(2, p1, stdin_image)
    p2 = _write_exec(td, "stage2.elf", stage2)
    stage3 = _derive(3, p2, stdin_image)
    return stage1, stage2, stage3


def manifest_text(stage1: bytes, *, published_by: str, receipt: dict[str, str]) -> str:
    """The MANIFEST.txt body: a human header, the provenance receipt, one row.

    The receipt is the reason the two-file publication is auditable at all. It
    records the sha256 of the ELF this manifest was written FOR, so a pair torn
    apart by a crash between the two replacements is detectable by comparing
    `stage1_sha256` against the file on disk — which is the honest answer to
    "these two paths cannot be replaced atomically together".

    Every field is DERIVED from the run's inputs and outputs. There is no
    timestamp: a timestamp cannot be re-derived, so it proves nothing and would
    make an otherwise reproducible file differ on every run.
    """
    h1 = hashlib.sha256(stage1).hexdigest()
    lines = [
        "# Frozen self-host bootstrap ELF (A6/RI-E1): the checked-in pure-MIND stage0",
        "# compiler that reproduces itself byte-identically (stage1==stage2==stage3)",
        "# and seeds the PRIMARY reproduction-independence loop. NOT skippable in CI.",
        f"# published-by\t{published_by}",
        "#",
        "# PROVENANCE RECEIPT — every value below is derived from this run's inputs",
        "# and outputs and is reproducible by re-running the same mode on the same",
        "# tree. No timestamp is recorded: it could not be re-derived and would only",
        "# make this file differ from itself.",
    ]
    lines += [f"# {k}\t{v}" for k, v in receipt.items()]
    lines += [
        "#",
        "# name\tsize_bytes\tsha256",
        f"stage1.elf\t{len(stage1)}\t{h1}",
    ]
    return "\n".join(lines) + "\n"


class PublicationTorn(RuntimeError):
    """Publication failed AND the rollback failed. Carries the recoverable copies.

    Distinct from an ordinary publication failure because the honest report is
    different: after this the two files are NOT known to be their original
    selves, and saying "unchanged" would be a lie. The preserved sibling copies
    are named so a human can finish by hand.
    """

    def __init__(self, detail: str, backups: dict[str, str]) -> None:
        super().__init__(detail)
        self.backups = backups


def _stage(parent: pathlib.Path, data: bytes, *, mode: int) -> pathlib.Path:
    """Write `data` to a UNIQUE, exclusively-created sibling of the live files.

    `mkstemp` is the point, not a detail. The previous fixed `<name>.advance-staging`
    was two bugs at once: a pre-existing symlink at that path would have been
    followed by `open(..., "wb")`, letting anything writable redirect the staged
    bytes (and, once replaced, the published fixture); and two concurrent runs
    would have shared the one staging file. `mkstemp` creates with O_CREAT|O_EXCL
    at a name nobody can predict, so it never follows a symlink and never
    collides. `mode` carries the live file's own permissions forward, so
    publication does not silently change them.
    """
    fd, name = tempfile.mkstemp(dir=str(parent), prefix=".advance-stage-")
    staged = pathlib.Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(staged, mode)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _discard(paths: list[pathlib.Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def publish_fixture(stage1: bytes, manifest: str) -> None:
    """Replace stage1.elf and MANIFEST.txt, with staging and rollback.

    This is NOT an atomic replacement of the pair and does not claim to be:
    POSIX offers no way to swap two independent paths in one step. What it does
    guarantee:

      * all three files this operation can ever need — the new ELF, the new
        manifest, and the ROLLBACK COPY of the old ELF — are written and fsynced
        to unique sibling files BEFORE either live path is touched. Rollback
        therefore needs no allocation: it is one `os.replace` of bytes already on
        disk. Writing the rollback copy only after the failure (the earlier
        shape) meant a full disk could fail the manifest replace AND then fail
        the rollback, stranding the new ELF beside the old manifest;
      * each individual replacement is atomic within one directory, so neither
        file is ever half-written;
      * if the manifest replacement fails, the ELF is restored from that
        pre-written copy and the failure is raised.

    Two residuals, both named rather than hidden:
      * a crash BETWEEN the two replacements is not prevented. It is made
        DETECTABLE, because the manifest's receipt carries the sha256 of the ELF
        it belongs to;
      * if the rollback replace ALSO fails, this raises `PublicationTorn` with
        the surviving copies, and the caller reports that exact state instead of
        claiming the old files are intact.
    """
    parent = _FROZEN.parent
    elf_exists = _FROZEN.exists()
    prior_elf = _FROZEN.read_bytes() if elf_exists else None
    elf_mode = _FROZEN.stat().st_mode & 0o7777 if elf_exists else 0o755
    man_mode = (
        _FROZEN_MANIFEST.stat().st_mode & 0o7777 if _FROZEN_MANIFEST.exists() else 0o644
    )

    staged: list[pathlib.Path] = []
    try:
        elf_tmp = _stage(parent, stage1, mode=elf_mode)
        staged.append(elf_tmp)
        man_tmp = _stage(parent, manifest.encode(), mode=man_mode)
        staged.append(man_tmp)
        rollback = None
        if prior_elf is not None:
            rollback = _stage(parent, prior_elf, mode=elf_mode)
            staged.append(rollback)
    except OSError:
        # Nothing live has been touched yet; leave no staging debris behind.
        _discard(staged)
        raise

    try:
        _replace(elf_tmp, _FROZEN)
    except OSError:
        _discard(staged)
        raise
    staged.remove(elf_tmp)

    try:
        _replace(man_tmp, _FROZEN_MANIFEST)
    except OSError as publish_error:
        try:
            if rollback is not None:
                _replace(rollback, _FROZEN)
                staged.remove(rollback)
            else:
                _FROZEN.unlink(missing_ok=True)
        except OSError as rollback_error:
            raise PublicationTorn(
                f"publishing the manifest failed ({publish_error}) and rolling the "
                f"ELF back also failed ({rollback_error})",
                {
                    "previous stage1.elf": str(rollback) if rollback else "(none)",
                    "unpublished MANIFEST.txt": str(man_tmp),
                },
            ) from publish_error
        _discard(staged)
        raise
    staged.remove(man_tmp)
    _discard(staged)


def do_advance(combined: bytes, stdin_image: bytes, user_lo: int) -> int:
    """ADVANCE: let the EXISTING frozen pure-MIND stage0 compile the CURRENT
    source, prove the result is a fixed point, corroborate it with a fresh Rust
    oracle, and only then publish it as the new frozen bootstrap.

    The difference from ordinary verification is exactly one assertion: the new
    fixed point MAY differ from the old frozen ELF. Everything else is stricter,
    not looser — publication requires the full loop AND a fresh oracle, and the
    oracle leg is not deferrable here.

    The difference from --reseed is the SEED. --reseed mints the bootstrap from
    Rust output; --advance mints it from the previous pure-MIND compiler, so the
    pure-MIND seed chain is never broken and the residual trusting-trust surface
    does not grow.
    """
    if not _FROZEN.exists():
        print(f"BLOCKED: --advance needs the existing frozen bootstrap seed at "
              f"{_FROZEN} — it is what compiles the new source. There is nothing "
              f"to advance FROM; this is the --reseed (legacy, Rust-seeded) case.")
        return 2
    old_seed = _FROZEN.read_bytes()
    h_old = hashlib.sha256(old_seed).hexdigest()
    print(f"  old seed = frozen pure-MIND stage0 ELF: {len(old_seed)}B sha256={h_old}")

    # The old seed runs FIRST, on the current source, from a PRIVATE SNAPSHOT of
    # the exact bytes just hashed — never from the live `_FROZEN` path. Hashing
    # one path and then executing it again is a time-of-check/time-of-use gap:
    # a concurrent advance (or anything else writing that path) would make the
    # recorded `advanced_from_seed_sha256` describe a file that is not what ran.
    # The snapshot lives in a 0700 temp dir and is mode 0500, so the bytes that
    # were hashed are the bytes that execute.
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        snapshot = _write_exec(tmp, "old-seed-snapshot.elf", old_seed)
        snapshot.chmod(0o500)
        try:
            stage1, stage2, stage3 = run_loop_from_seed(snapshot, stdin_image, tmp)
        except LoopFailure as e:
            if e.stage == 1:
                print(f"  FAIL  [ADVANCE] the old frozen seed refused the current "
                      f"source ({e.detail}).\n"
                      f"        Nothing was written. The old compiler must be able to "
                      f"compile the new source; if it cannot, the source uses a "
                      f"construct outside the bootstrap subset and that is the bug to "
                      f"fix — not the fixture.")
            else:
                # The old seed already compiled the source successfully, so
                # blaming it here would send the reader to the wrong file.
                print(f"  FAIL  [ADVANCE] bootstrap transition failed at stage{e.stage} "
                      f"({e.detail}).\n"
                      f"        The old seed accepted the source and produced stage1; "
                      f"the failure is in the compiler stage1 emitted. Nothing was "
                      f"written.")
            return 1

    h1 = hashlib.sha256(stage1).hexdigest()
    h2 = hashlib.sha256(stage2).hexdigest()
    h3 = hashlib.sha256(stage3).hexdigest()
    print(f"  stage1 (old frozen seed on CURRENT source): {len(stage1)}B sha256={h1}")
    print(f"  stage2 (stage1 run natively):               {len(stage2)}B sha256={h2}")
    print(f"  stage3 (stage2 run natively):               {len(stage3)}B sha256={h3}")

    for name, blob in (("stage2", stage2), ("stage3", stage3)):
        if not is_static_elf(blob):
            print(f"  FAIL  [ADVANCE] {name} is not a static ELF ({len(blob)}B) — "
                  f"nothing written.")
            return 1
    if not (stage1 == stage2 == stage3):
        print("  FAIL  [ADVANCE] the new source is NOT self-reproducing:")
        if stage1 != stage2:
            print(f"        stage1 != stage2 (sizes {len(stage1)} vs {len(stage2)})")
        if stage2 != stage3:
            print(f"        stage2 != stage3 (sizes {len(stage2)} vs {len(stage3)})")
        print("        Nothing written — a non-fixed-point compiler must never "
              "become the seed.")
        return 1
    print(f"  PASS  [ADVANCE] stage1 == stage2 == stage3 — fixed point "
          f"proved ({h1})")

    # ORACLE — required, never deferred. A new fixed point that only the old
    # compiler agrees with is one compiler's opinion; the independently built
    # Rust oracle on the SAME current source is the second witness.
    oracle = so()
    print(f"  oracle = {oracle.name} ({oracle.provenance})")
    if not oracle.exists():
        print(f"  BLOCKED  [ADVANCE] no Rust oracle at {oracle}. Advancing the "
              f"bootstrap on a single witness is exactly the unchecked step this "
              f"gate exists to refuse. Build one and re-run; nothing written.")
        return 2
    if not oracle.is_fresh:
        print(f"  BLOCKED  [ADVANCE] the oracle is {oracle.provenance}, NOT a fresh "
              f"build ({oracle.detail}). Bytes of unknown age cannot corroborate a "
              f"fixed point derived from the CURRENT source; nothing written.")
        return 2
    try:
        oracle_elf = stage0_emit(combined, user_lo)
    except OSError as e:
        print(f"  FAIL  [ADVANCE] the oracle .so could not be loaded ({e}) — "
              f"nothing written.")
        return 1
    if not is_static_elf(oracle_elf):
        print(f"  FAIL  [ADVANCE] the oracle emitted a non-ELF/empty image "
              f"({len(oracle_elf)}B) — nothing written.")
        return 1
    h_oracle = hashlib.sha256(oracle_elf).hexdigest()
    if oracle_elf != stage1:
        print(f"  FAIL  [ADVANCE] oracle output ({h_oracle}) != the new fixed point "
              f"({h1}). Two independent compilers disagree about the current "
              f"source, so neither may be frozen; nothing written.")
        return 1
    print(f"  PASS  [ADVANCE] fresh Rust oracle output == new fixed point "
          f"({h_oracle}) — independent agreement proved")

    # Nothing may be published over a fixture that changed under us: another
    # advance may have completed while this one was running, and overwriting it
    # would silently discard a seed someone else already proved. This closes the
    # window from capture to here; the remaining gap between this comparison and
    # the replace below is the residual, and it is stated in the report rather
    # than papered over with a lock file this gate would then have to own.
    live_now = _FROZEN.read_bytes() if _FROZEN.exists() else None
    if live_now != old_seed:
        what = "removed" if live_now is None else (
            f"changed to sha256={hashlib.sha256(live_now).hexdigest()}"
        )
        print(f"  FAIL  [ADVANCE] the frozen fixture {what} while this advance was "
              f"running (captured {h_old}). Refusing to publish over it — another "
              f"advance may have completed, and overwriting it would discard a seed "
              f"that was already proved. Nothing was written; re-run.")
        return 1

    receipt = {
        "advanced_from_seed_sha256": h_old,
        "combined_source_sha256": hashlib.sha256(combined).hexdigest(),
        "combined_source_bytes": str(len(combined)),
        "user_lo": str(user_lo),
        "stage1_sha256": h1,
        "stage2_sha256": h2,
        "stage3_sha256": h3,
        "oracle_sha256": h_oracle,
        "oracle_provenance": oracle.provenance,
    }
    try:
        publish_fixture(stage1, manifest_text(
            stage1, published_by="self_host_loop_smoke.py --advance", receipt=receipt
        ))
    except PublicationTorn as e:
        detail = "; ".join(f"{k}: {v}" for k, v in e.backups.items())
        print(f"  FAIL  [ADVANCE] {e}.\n"
              f"        The fixture is NOT known to be intact — this run cannot "
              f"claim the old files are unchanged. Recoverable copies were "
              f"preserved: {detail}. Restore them by hand and re-run.")
        return 1
    except OSError as e:
        print(f"  FAIL  [ADVANCE] every check passed but publication failed ({e}). "
              f"The previous fixture was rolled back from a copy staged before "
              f"anything was touched; nothing is half-advanced.")
        return 1

    moved = " (unchanged bytes)" if stage1 == old_seed else ""
    print(f"  ADVANCED  frozen bootstrap advanced by the OLD pure-MIND stage0:\n"
          f"            {h_old}\n         -> {h1}{moved} ({len(stage1)}B)\n"
          f"            corroborated by a {oracle.provenance} Rust oracle "
          f"({h_oracle})\n"
          f"            wrote {_FROZEN} and {_FROZEN_MANIFEST}")
    return 0


def do_reseed(combined: bytes, stdin_image: bytes, user_lo: int) -> int:
    """Run the legacy path in its focused support module."""
    from _selfhost_loop_reseed import do_reseed as reseed
    import sys as _sys
    return reseed(_sys.modules[__name__], combined, stdin_image, user_lo)



ORACLE_DEFER_ENV = "MIND_SELFHOST_LOOP_ORACLE_DEFERRED"


def oracle_unavailable(reason: str) -> int:
    """The ORACLE leg could not run: announce it as a SKIP and FAIL CLOSED.

    This gate has TWO legs and only the pair is the gate: PRIMARY proves the
    frozen pure-MIND stage0 still reproduces itself, ORACLE proves the CURRENT
    std/*.mind + main.mind source has not drifted away from that frozen ELF.
    Dropping ORACLE removes source-drift detection entirely.

    It used to be dropped by printing a `NOTE ... SKIPPED` line and returning 0.
    Measured with the oracle `.so` hidden (`MINDC_SO_NOBUILD=1`, MINDC_SO unset):
    `run_gate: PASS ... asserted=1` — the wedge's loop gate green with half of
    itself never executed, and its skip line invisible to BOTH skip detectors
    (`SKIP\b` does not match `SKIPPED`; nothing looked mid-line at all).

    So: a `SKIP` line the runner sees, plus a non-zero exit for a bare
    invocation. `MIND_SELFHOST_LOOP_ORACLE_DEFERRED=1` records a deliberate
    one-leg run for a box that cannot build the `.so`; it is NOT a way past CI,
    because ci.yml and preflight.sh both run this gate with `--min-asserted 2`
    and a one-leg run reports 1.
    """
    if os.environ.get(ORACLE_DEFER_ENV) == "1":
        print(f"  NOTE  [ORACLE] {reason} — deferred by {ORACLE_DEFER_ENV}=1. "
              f"The source-drift assertion did NOT run, so this run reports one "
              f"asserted leg; every `--min-asserted 2` call site still refuses it.")
        return 0
    print(f"  SKIP  [ORACLE] {reason} — the source-drift assertion did NOT run, "
          f"and half a gate is not this gate. Set MINDC_SO to a built self-host "
          f"`.so`, or build `mindc` so a fresh one can be emitted. To record a "
          f"deliberate one-leg run on a box that cannot, set "
          f"{ORACLE_DEFER_ENV}=1 (CI and preflight demand both legs regardless).")
    return 1


def main(argv: list[str] | None = None, env=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if env is None else env

    # Mode selection comes FIRST, before build_seed() reads a single source file
    # and before any oracle is resolved: a request the harness will refuse must
    # not have already run a compiler or touched the fixture.
    advance_req, reseed_req = requested_modes(argv, env)
    conflict = mode_conflict(advance_req, reseed_req)
    if conflict is not None:
        print(conflict)
        return 2

    combined, stdin_image, user_lo = build_seed()
    mode = "advance" if advance_req else ("reseed" if reseed_req else "verify")
    # The banner deliberately does NOT name the oracle: `so()` can shell out to
    # `mindc build --emit=cdylib` and exits on a non-fresh artifact, so printing
    # it here resolved (and possibly built) an oracle before --advance had even
    # checked that a seed exists to advance from. Each mode resolves it at the
    # point it actually needs it, and prints it there.
    print(f"[self-host loop] combined={len(combined)}B user_lo={user_lo} "
          f"seed={len(stdin_image)}B  mode={mode}")

    if advance_req:
        return do_advance(combined, stdin_image, user_lo)
    if reseed_req:
        return do_reseed(combined, stdin_image, user_lo)

    # ------------------------------------------------------------------
    # PRIMARY: reproduction-independence from the checked-in pure-MIND ELF.
    # Zero Rust / LLVM in this chain — only execve(self)/read/write/exit.
    # ------------------------------------------------------------------
    if not _FROZEN.exists():
        print(f"BLOCKED: frozen bootstrap seed {_FROZEN} not found — it is the "
              f"checked-in pure-MIND stage0. Re-freeze with --reseed (MINDC_SO set).")
        return 2
    frozen = _FROZEN.read_bytes()
    hf = hashlib.sha256(frozen).hexdigest()
    print(f"  seed = frozen pure-MIND stage0 ELF: {len(frozen)}B sha256={hf}")

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        try:
            stage1, stage2, stage3 = run_loop_from_seed(_FROZEN, stdin_image, tmp)
        except RuntimeError as e:
            print(f"  FAIL  {e}")
            return 1

    h1 = hashlib.sha256(stage1).hexdigest()
    h2 = hashlib.sha256(stage2).hexdigest()
    h3 = hashlib.sha256(stage3).hexdigest()
    print(f"  stage1 (frozen stage0 run natively): {len(stage1)}B sha256={h1}")
    print(f"  stage2 (stage1 run natively):        {len(stage2)}B sha256={h2}")
    print(f"  stage3 (stage2 run natively):        {len(stage3)}B sha256={h3}")

    if not (stage1 == stage2 == stage3):
        print("  FAIL  self-host loop NOT closed:")
        if stage1 != stage2:
            print(f"        stage1 != stage2 (sizes {len(stage1)} vs {len(stage2)})")
        if stage2 != stage3:
            print(f"        stage2 != stage3 (sizes {len(stage2)} vs {len(stage3)})")
        return 1
    if stage1 != frozen:
        print(f"  FAIL  stage1 ({h1}) != frozen seed ({hf}) — running the frozen "
              f"pure-MIND stage0 did NOT reproduce it; the bootstrap is not fixed.")
        return 1

    print(f"  PASS  [PRIMARY] stage1 == stage2 == stage3 == frozen stage0 "
          f"BYTE-IDENTICAL ({len(stage1)}B, sha256={h1}) — MIND reproduces its "
          f"compiler with ZERO Rust/LLVM in the chain (scalar subset, RI-E1).")

    # ------------------------------------------------------------------
    # ORACLE: Rust `.so` drift check (soft-skip if the .so is unavailable).
    # Catches std/*.mind or main.mind source drift where the frozen ELF was
    # not re-blessed. The PRIMARY path above does NOT depend on this.
    # ------------------------------------------------------------------
    oracle = so()
    print(f"  oracle = {oracle.name} ({oracle.provenance})")
    if not oracle.exists():
        return oracle_unavailable(f"Rust drift .so not present ({oracle})")
    try:
        so_stage1 = stage0_emit(combined, user_lo)
    except OSError as e:
        # With MINDC_SO set the operator/CI promised a real oracle .so, so a load
        # failure is a BROKEN gate, not an inapplicable one, and it names that
        # promise in the message rather than routing through the generic
        # unavailable path. Without the handle it is still not a pass: the
        # fall-through below announces a SKIP and exits non-zero. This smoke runs
        # BARE in ci.yml (no surrounding "SKIPped with MINDC_SO set" backstop like
        # the batch loops have), so run_gate.py's shared skip rule — which now
        # matches "SKIPPED" and mid-line reports too — is the whole backstop.
        if os.environ.get("MINDC_SO"):
            print(f"  FAIL  [ORACLE] MINDC_SO is set but the drift .so could not be "
                  f"loaded ({e}) — refusing to skip; the source-drift assertion did "
                  f"not run.")
            return 1
        return oracle_unavailable(f"could not load the drift .so ({e})")
    if not is_static_elf(so_stage1):
        print(f"  FAIL  [ORACLE] fresh .so emitted a non-ELF/empty image "
              f"({len(so_stage1)}B) — .so seed path is broken.")
        return 1
    hso = hashlib.sha256(so_stage1).hexdigest()
    if so_stage1 != frozen:
        if not so().is_fresh:
            # The resolver could not build a fresh oracle and fell back to the
            # legacy in-tree .so (it WARNs above). Those bytes cannot distinguish
            # real source drift from an artifact months old, so the normal advice
            # -- "--reseed" -- must NOT be given here: re-blessing the frozen
            # bootstrap from a stale oracle freezes the WRONG compiler, which is
            # the one outcome this gate exists to prevent.
            print(f"  FAIL  [ORACLE] output ({hso}) != frozen bootstrap ({hf}) — but "
                  f"this oracle is {so().provenance}, NOT a fresh build ({so().detail}). "
                  f"Stale bytes cannot tell real drift from an old "
                  f"artifact, so DO NOT advance the fixture on this evidence. "
                  f"Rebuild mindc with "
                  f"`--features mlir-build` so a fresh oracle can be emitted, then "
                  f"re-run; only then is a drift verdict trustworthy.")
        else:
            print(f"  FAIL  [ORACLE] fresh Rust .so output ({hso}) != frozen bootstrap "
                  f"({hf}) — std/main.mind SOURCE drifted. Advance the bootstrap in "
                  f"THIS change with `self_host_loop_smoke.py --advance` (MINDC_SO "
                  f"set): the OLD frozen compiler compiles the new source and the "
                  f"fresh oracle must agree, so the pure-MIND seed chain stays "
                  f"unbroken. `--reseed` re-mints the seed from RUST output and is "
                  f"only for a bootstrap that cannot be advanced at all.")
        return 1
    if not so().is_fresh:
        # The mirror of the FAIL case, and the more dangerous half: stale bytes
        # that HAPPEN to match an equally stale seed would otherwise print
        # "fresh Rust .so ... no source drift" and exit 0 while the current
        # source has genuinely drifted. The oracle leg cannot be satisfied by an
        # oracle we could not build, so it does not get to pass.
        print(f"  FAIL  [ORACLE] output matches the frozen bootstrap ({hso}), but "
              f"this oracle is {so().provenance}, NOT a fresh build ({so().detail}). "
              f"A stale oracle agreeing with an equally stale seed "
              f"proves nothing about current source. Rebuild mindc with "
              f"`--features mlir-build` and re-run.")
        return 1
    print(f"  PASS  [ORACLE] fresh Rust .so output == frozen bootstrap "
          f"({hso}) — no source drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
