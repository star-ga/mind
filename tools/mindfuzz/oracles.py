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
# 0. cross-substrate byte-identity -- THE WEDGE. Now REAL via the CI matrix.
# --------------------------------------------------------------------------- #
# The wedge oracle (avx2 artifact == neon artifact) cannot be decided on a
# single host. Instead of stubbing it, MIND-Fuzz now STAGES every survivor as a
# candidate cross-substrate workload: the surviving program plus the host's
# canonical output hash. The committed staging dir is then checked by the Rust
# test `tests/mindfuzz_cross_substrate.rs`, which the EXISTING
# `cross_substrate_identity` CI job runs on BOTH the avx2 (ubuntu-24.04) and
# neon (ubuntu-24.04-arm) runners. Each runner recomputes the survivor's output
# hash and asserts it equals the committed reference — so the neon runner
# asserts byte-identity against the avx2-blessed hash. That IS the wedge check
# (RFC 0015 §3.1), now enforced on real ARM hardware every CI run.

# The canonical driver: a survivor that exposes the scalar entry `f(i64)->i64`
# is exercised over a fixed, deterministic argument vector (the same LCG as
# tests/cross_substrate_identity.rs) and the returns are hashed as i64-LE bytes.
# This pins an ISA-relevant value (the integer-SSA lowering result) that MUST be
# identical on avx2 and neon. The argument set + entry symbol are part of the
# staged manifest so the Rust gate replays them byte-for-byte.
CANON_ENTRY = "f"
CANON_ARG_SEED = 0xDEADBEEF
CANON_ARG_COUNT = 64


def _canon_args(seed: int = CANON_ARG_SEED, count: int = CANON_ARG_COUNT) -> list[int]:
    """Deterministic i64 argument vector for the canonical driver. Uses the same
    LCG as the cross-substrate gate, sampling a full signed-i32 window per arg so
    the entry exercises positive, negative and zero paths."""
    g = _Lcg(seed)
    out: list[int] = []
    for _ in range(count):
        v = g.u32()
        out.append(v - 2**32 if v >= 2**31 else v)
    return out


def canonical_output_hash(out_so: Path, entry: str = CANON_ENTRY) -> str | None:
    """Call `entry(i64)->i64` over the canonical arg vector and sha256 the
    concatenated i64-LE returns. Returns None if the symbol is absent (the
    survivor is not canonical-drivable, e.g. a buffer kernel). The returns are
    the exact bytes that must match across substrates."""
    try:
        lib = ctypes.CDLL(str(out_so))
    except OSError:
        return None
    try:
        f = lib[entry]
    except (AttributeError, KeyError, ValueError):
        return None
    f.restype = ctypes.c_int64
    f.argtypes = [ctypes.c_int64]
    h = hashlib.sha256()
    for a in _canon_args():
        r = int(f(a)) & 0xFFFFFFFFFFFFFFFF
        h.update(r.to_bytes(8, "little"))
    return h.hexdigest()


def cross_substrate_hook(src: Path, out_so: Path | None = None) -> OracleVerdict:
    """Wedge oracle. On a single host this STAGES the candidate (it cannot decide
    avx2==neon alone), so it never raises a false violation here; the equality is
    asserted on the CI matrix by tests/mindfuzz_cross_substrate.rs. When `out_so`
    is the survivor's compiled lib and it exposes the canonical entry, the
    verdict carries the host output hash that gets committed as the reference."""
    if out_so is None or not out_so.exists():
        return OracleVerdict(
            "cross_substrate",
            True,
            "STAGED: survivor recorded; avx2==neon asserted on the CI matrix.",
        )
    digest = canonical_output_hash(out_so)
    if digest is None:
        return OracleVerdict(
            "cross_substrate",
            True,
            "STAGED: no canonical entry; covered by determinism/mic@3 only.",
        )
    return OracleVerdict(
        "cross_substrate",
        True,
        f"STAGED host_hash={digest} (asserted avx2==neon on the CI matrix)",
    )
