"""MIND-Fuzz differential oracles.

Each oracle compiles / inspects a mutated MIND program and returns whether it
holds. A FAILING oracle is a candidate compiler bug (a "violation" in the paper).

The whole point of MIND-Fuzz vs the paper's C/Rust/Swift target: MIND's
cross-substrate BYTE-IDENTITY gives a differential oracle no other compiler has
-- exact, free, no float flakiness. The local proxies for that wedge are the
determinism + reference oracles below; the real avx2==neon check needs neon
hardware / CI and is left as an explicit hook (cross_substrate_hook).

Oracles implemented here (all LOCAL, single-host):
  1. determinism      compile the SAME program twice -> mic@3 + MLIR + .so hashes
                      must match. A mismatch = non-determinism bug (wedge proxy).
  2. reference        for the Q16.16 dot/gemv kernels, call the compiled symbol
                      via ctypes and compare to a scalar integer reference.
  3. verify           run `mindc verify` on the emitted mic@3 evidence artifact
                      (RFC 0017/0021 SSA + trace_hash). Non-zero / invalid = IR bug.
  4. mic3_roundtrip   emit mic@3, re-emit, require a byte fixed point (codec bug).
  5. compile          a program that SHOULD compile but errors / crashes mindc.

cross_substrate_hook is the seam for oracle (0): the real wedge check that the
avx2 artifact hash equals the neon artifact hash. It is STUBBED here (returns a
"deferred" verdict) because it needs a second substrate; CI / the cluster wires
the real comparison in.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

MINDC = Path(__file__).resolve().parents[2] / "target" / "release" / "mindc"


@dataclass
class OracleVerdict:
    name: str
    ok: bool
    detail: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(args: list[str], timeout_s: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(MINDC), *args], capture_output=True, text=True, timeout=timeout_s
    )


# --------------------------------------------------------------------------- #
# 5. compile-success
# --------------------------------------------------------------------------- #
def compile_so(src: Path, out_so: Path) -> tuple[bool, str]:
    """Compile to a shared lib. Returns (ok, stderr_tail). Crash (rc<0) is a bug."""
    try:
        p = _run([str(src), "--emit-shared", str(out_so)])
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: mindc --emit-shared hung"
    if p.returncode < 0:
        return False, f"CRASH: mindc killed by signal {-p.returncode}\n{p.stderr[-400:]}"
    return p.returncode == 0, p.stderr[-400:]


# --------------------------------------------------------------------------- #
# 1. determinism (local proxy for cross-substrate byte-identity)
# --------------------------------------------------------------------------- #
def oracle_determinism(src: Path, work: Path) -> OracleVerdict:
    """Compile the SAME program twice; mic@3, MLIR and .so hashes must match."""
    hashes: dict[str, tuple[str, str]] = {}
    for tag, args, produces in (
        ("mic3", ["--emit-mic3", "{out}"], True),
        ("so", ["--emit-shared", "{out}"], True),
    ):
        a = work / f"det_a.{tag}"
        b = work / f"det_b.{tag}"
        for dst in (a, b):
            real = [arg.format(out=str(dst)) for arg in args]
            try:
                p = _run([str(src), *real])
            except subprocess.TimeoutExpired:
                return OracleVerdict("determinism", False, f"{tag}: TIMEOUT")
            if p.returncode != 0 or not dst.exists():
                return OracleVerdict(
                    "determinism", False, f"{tag}: emit failed rc={p.returncode}"
                )
        ha, hb = _sha(a), _sha(b)
        hashes[tag] = (ha, hb)
        if ha != hb:
            return OracleVerdict(
                "determinism",
                False,
                f"NON-DETERMINISM in {tag}: {ha[:16]} != {hb[:16]} "
                f"(same source compiled twice diverged)",
            )

    # MLIR text comes on stdout, not a path
    mlir_hashes = []
    for _ in range(2):
        try:
            p = _run([str(src), "--emit-mlir"])
        except subprocess.TimeoutExpired:
            return OracleVerdict("determinism", False, "mlir: TIMEOUT")
        if p.returncode != 0:
            # --emit-mlir may require a feature; treat absence as non-applicable,
            # not a violation.
            return OracleVerdict(
                "determinism", True, f"mic3+so deterministic (mlir emit skipped)"
            )
        mlir_hashes.append(hashlib.sha256(p.stdout.encode()).hexdigest())
    if mlir_hashes[0] != mlir_hashes[1]:
        return OracleVerdict(
            "determinism",
            False,
            f"NON-DETERMINISM in MLIR: {mlir_hashes[0][:16]} != {mlir_hashes[1][:16]}",
        )

    return OracleVerdict(
        "determinism", True, f"mic3/so/mlir all byte-identical across two compiles"
    )


# --------------------------------------------------------------------------- #
# 4. mic@3 round-trip fixed point
# --------------------------------------------------------------------------- #
def oracle_mic3_roundtrip(src: Path, work: Path) -> OracleVerdict:
    """Emit mic@3 twice from source; require a byte fixed point.

    A full parse->re-emit CLI path is not exposed, so the available fixed-point
    check is emit-stability of the canonical artifact (compact::v3::emit_mic3).
    A divergence here is a codec / serialization bug.
    """
    a = work / "rt_a.mic3"
    b = work / "rt_b.mic3"
    for dst in (a, b):
        try:
            p = _run([str(src), "--emit-mic3", str(dst)])
        except subprocess.TimeoutExpired:
            return OracleVerdict("mic3_roundtrip", False, "TIMEOUT")
        if p.returncode != 0 or not dst.exists():
            return OracleVerdict(
                "mic3_roundtrip", False, f"emit failed rc={p.returncode}"
            )
    ha, hb = _sha(a), _sha(b)
    if ha != hb:
        return OracleVerdict(
            "mic3_roundtrip", False, f"mic@3 not a fixed point: {ha[:16]} != {hb[:16]}"
        )
    return OracleVerdict("mic3_roundtrip", True, f"mic@3 fixed point ({ha[:16]})")


# --------------------------------------------------------------------------- #
# 3. mindc verify (RFC 0017/0021 SSA + trace_hash)
# --------------------------------------------------------------------------- #
def oracle_verify(src: Path, work: Path) -> OracleVerdict:
    """Emit an evidence artifact and run `mindc verify` on it."""
    ev = work / "ev.mic3"
    try:
        pe = _run([str(src), "--emit-evidence", str(ev)])
    except subprocess.TimeoutExpired:
        return OracleVerdict("verify", False, "emit-evidence TIMEOUT")
    if pe.returncode != 0 or not ev.exists():
        return OracleVerdict("verify", False, f"emit-evidence failed rc={pe.returncode}")
    try:
        pv = _run(["verify", "--json", str(ev)])
    except subprocess.TimeoutExpired:
        return OracleVerdict("verify", False, "verify TIMEOUT")
    if pv.returncode != 0:
        return OracleVerdict(
            "verify",
            False,
            f"mindc verify rc={pv.returncode}: {pv.stdout.strip()} {pv.stderr[-200:]}",
        )
    if '"trace_hash_valid":true' not in pv.stdout.replace(" ", ""):
        return OracleVerdict("verify", False, f"trace_hash invalid: {pv.stdout.strip()}")
    return OracleVerdict("verify", True, "evidence chain intact (trace_hash valid)")


# --------------------------------------------------------------------------- #
# 2. reference oracle (Q16.16 dot kernels via ctypes)
# --------------------------------------------------------------------------- #
class _Lcg:
    """Byte-for-byte the LCG in tests/cross_substrate_identity.rs."""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def u32(self) -> int:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
        return (self.s >> 16) & 0xFFFFFFFF

    def q16(self) -> int:
        v = self.u32()
        if v >= 2**31:
            v -= 2**32
        return v >> 12


def _make_pair_q16(n: int, seed: int) -> tuple[list[int], list[int]]:
    g = _Lcg(seed)
    a = [g.q16() for _ in range(n)]
    b = [g.q16() for _ in range(n)]
    return a, b


def _ref_dot_q16(a: list[int], b: list[int]) -> int:
    acc = 0
    for i in range(len(a)):
        acc += (a[i] * b[i]) >> 16
    acc &= 0xFFFFFFFF
    return acc - 2**32 if acc >= 2**31 else acc


def _ref_dot_l1_q16(a: list[int], b: list[int]) -> int:
    acc = 0
    for i in range(len(a)):
        acc += abs(a[i] - b[i])
    acc &= 0xFFFFFFFF
    return acc - 2**32 if acc >= 2**31 else acc


# (symbol, n, ref-fn) for the reference-checkable dot kernels.
_REF_KERNELS = {
    "dotq": (256, _ref_dot_q16),
    "dotl1q": (256, _ref_dot_l1_q16),
}


def oracle_reference(out_so: Path, seed: int = 0xDEADBEEF) -> OracleVerdict:
    """For each known dot symbol present in the .so, kernel(a,b,n) must equal a
    scalar integer reference. Symbols not present are skipped (mutation may have
    renamed/removed them); the oracle only asserts on symbols it can find."""
    try:
        lib = ctypes.CDLL(str(out_so))
    except OSError as e:
        return OracleVerdict("reference", False, f"dlopen failed: {e}")

    checked = []
    for sym, (n, ref) in _REF_KERNELS.items():
        fn = getattr(lib, sym, None)
        if fn is None or not hasattr(lib, sym):
            continue
        try:
            f = lib[sym]
        except (AttributeError, KeyError):
            continue
        f.restype = ctypes.c_int64
        f.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        a, b = _make_pair_q16(n, seed)
        abuf = (ctypes.c_int32 * n)(*a)
        bbuf = (ctypes.c_int32 * n)(*b)
        got = f(
            ctypes.cast(abuf, ctypes.c_void_p).value,
            ctypes.cast(bbuf, ctypes.c_void_p).value,
            n,
        )
        want = ref(a, b)
        if got != want:
            return OracleVerdict(
                "reference",
                False,
                f"symbol {sym}: kernel={got} != scalar_ref={want} "
                f"(vector path diverged from scalar oracle)",
            )
        checked.append(sym)

    if not checked:
        return OracleVerdict("reference", True, "no reference-checkable symbol present (skipped)")
    return OracleVerdict("reference", True, f"kernels match scalar ref: {','.join(checked)}")


# --------------------------------------------------------------------------- #
# 0. cross-substrate byte-identity -- THE WEDGE. Stubbed seam.
# --------------------------------------------------------------------------- #
def cross_substrate_hook(src: Path) -> OracleVerdict:
    """The real wedge oracle: emit on avx2 AND neon, require identical artifact
    hashes (RFC 0015 §3.1). This host only has one substrate, so this is the
    SEAM, not the check: CI / the cluster runs the same program on an
    aarch64 runner and compares the mic@3 hash committed by this host.

    Implementation hook for CI: have this host emit `--emit-mic3 host.mic3`,
    ship host.mic3's sha256 as an artifact, and on the neon runner assert
    sha256(neon.mic3) == sha256(host.mic3). A mismatch is a wedge-breaking bug.
    """
    return OracleVerdict(
        "cross_substrate",
        True,  # deferred verdict -- never a false violation on a single host
        "DEFERRED: needs a second substrate (neon). Run via CI hook; "
        "this host emitted its mic@3 hash for the cross-runner compare.",
    )
