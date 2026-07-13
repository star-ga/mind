"""Shared self-host `.so` resolver for the examples/mindc_mind smokes.

Why this exists
---------------
`examples/mindc_mind/libmindc_mind.so` is a BUILD ARTIFACT — it is untracked and
gitignored (`examples/**/*.so`). Crucially, `cargo build` does NOT produce it;
only `mindc build --emit=cdylib` does. So an in-tree copy left over from an
older build goes STALE and silently false-fails the byte-identity smokes
(mic@3 flip, self-host loop, native-ELF, ...): the driver bytes drift from the
oracle even though nothing is actually wrong. CI never hits this — it builds the
`.so` fresh each run and points `MINDC_SO` at `/tmp/libmindc_mind_self_host.so`.

`resolve_so()` gives a purely-local run the same guarantee CI has:

  1. `MINDC_SO` set          -> use it verbatim (the CI contract, unchanged).
  2. else, release `mindc`   -> BUILD THE SELF-HOST `.so` FRESH via
     binary present             `mindc build --release --emit=cdylib` into a
                                per-tree temp cache (rebuilt only when `mindc`,
                                `main.mind`, `selfhost_driver.mind` or `Mind.toml`
                                change), so no stale in-tree `.so` is ever trusted.
  3. else                    -> fall back to the legacy in-tree path so each
                                smoke's existing not-found (SKIP/BLOCKED)
                                fail-closed handling fires exactly as before.

Escape hatch: `MINDC_SO_NOBUILD=1` forces the legacy in-tree default (no build) —
e.g. to point a smoke at whatever `.so` happens to be next to it on purpose.
"""

import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
_LEGACY_SO = _HERE / "libmindc_mind.so"
_MINDC = _REPO / "target" / "release" / "mindc"


def _stamp() -> str:
    """Fingerprint of every input that affects the emitted self-host `.so`."""
    parts = []
    for p in (
        _MINDC,
        _HERE / "main.mind",
        _HERE / "selfhost_driver.mind",
        _REPO / "Mind.toml",
    ):
        try:
            st = p.stat()
            parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{p}:MISSING")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _build_fresh() -> pathlib.Path | None:
    """Emit the self-host cdylib fresh (cached by input stamp). None on failure."""
    cache = pathlib.Path(tempfile.gettempdir()) / "mindc_mind_selfhost_cache"
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / "libmindc_mind.so"
    stampf = cache / "stamp"
    want = _stamp()
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
    return so


def resolve_so() -> pathlib.Path:
    """Resolve the self-host `.so` path (see module docstring for the order)."""
    env = os.environ.get("MINDC_SO")
    if env:
        return pathlib.Path(env)
    if os.environ.get("MINDC_SO_NOBUILD") or not _MINDC.exists():
        return _LEGACY_SO
    fresh = _build_fresh()
    return fresh if fresh is not None else _LEGACY_SO
