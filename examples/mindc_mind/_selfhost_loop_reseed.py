"""Legacy Rust-seeded re-freeze implementation for the LOOP harness.

The public entrypoint stays in self_host_loop_smoke.py; this support module
keeps its legacy publication path focused and independently readable.
"""
from __future__ import annotations

import hashlib
import pathlib
import tempfile


def do_reseed(harness, combined: bytes, stdin_image: bytes, user_lo: int) -> int:
    """Run the legacy reseed path against the caller's patched harness module."""
    so = harness.so
    stage0_emit = harness.stage0_emit
    is_static_elf = harness.is_static_elf
    _write_exec = harness._write_exec
    _derive = harness._derive
    LoopFailure = harness.LoopFailure
    PublicationTorn = harness.PublicationTorn
    publish_fixture = harness.publish_fixture
    manifest_text = harness.manifest_text
    _FROZEN = harness._FROZEN
    _FROZEN_MANIFEST = harness._FROZEN_MANIFEST
    __file__ = harness.__file__
    """RESEED / re-mint — LEGACY, Rust-seeded. Needs MINDC_SO (fail-closed).

    Seeds stage1 with the Rust `.so`, confirms the loop closes, and re-freezes
    testdata/selfhost_loop/{stage1.elf,MANIFEST.txt}. It is the ONLY mode that
    uses the `.so` as a SEED, which is precisely what makes it legacy: the
    frozen bootstrap it mints descends from Rust output, not from the previous
    pure-MIND compiler, so the pure-MIND seed chain is broken for that hop and
    the residual trusting-trust surface grows by one Rust artifact.

    `--advance` is the ordinary answer to source drift and is preferred
    everywhere; keep this path for a bootstrap that genuinely cannot be
    advanced (the old seed cannot compile the new source at all), and say so in
    the change that uses it.
    """
    so_ = so()
    if not so_.exists():
        print(f"BLOCKED: --reseed needs the Rust seed .so; {so_} not found "
              f"(set MINDC_SO to a driver-capable libmindc_mind.so).")
        return 2
    # `so_.exists()` is NOT sufficient: the legacy in-tree artifact ALWAYS exists,
    # so on a box whose mindc lacks `mlir-build` (it cannot emit a cdylib at all)
    # this path would silently mint the frozen bootstrap from a months-old .so --
    # freezing the WRONG compiler, which is the one catastrophe this gate exists
    # to prevent. A legacy re-mint still requires a fresh independent oracle.
    # Defence in depth: `resolve_so()` is STRICT by default and refuses a
    # non-fresh oracle before this file does any work, so this branch is now
    # reachable only if this smoke ever opts out with allow_stale=True. It
    # reads provenance off the handle, so there is one source of truth.
    if not so_.is_fresh:
        print(f"BLOCKED: --reseed refuses a {so_.provenance} oracle. {so_} is "
              f"NOT a fresh build ({so_.detail}), so its bytes may be "
              f"arbitrarily old. Seeding the "
              f"frozen bootstrap from it would freeze whatever compiler that "
              f"artifact came from. Build a real oracle first:\n"
              f"  cargo build --release --bin mindc --features "
              f"mlir-build,std-surface,cross-module-imports\n"
              f"  <that mindc> build --release --emit=cdylib --out=/tmp/oracle.so\n"
              f"  MINDC_SO=/tmp/oracle.so python3 {__file__} --reseed\n"
              f"(For ordinary source drift use --advance instead: it seeds from the "
              f"EXISTING pure-MIND stage0 rather than from Rust output.)")
        return 2
    try:
        stage1 = stage0_emit(combined, user_lo)
    except OSError as e:
        print(f"  FAIL  --reseed: the oracle could not be loaded ({e}) — nothing written.")
        return 1
    if not is_static_elf(stage1):
        print(f"  FAIL  .so emitted a non-ELF/empty image ({len(stage1)}B) — "
              f"nb_trace_hash may have failed closed, or the driver entry is missing.")
        return 1
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        p1 = _write_exec(tmp, "stage1.elf", stage1)
        try:
            stage2 = _derive(2, p1, stdin_image)
            p2 = _write_exec(tmp, "stage2.elf", stage2)
            stage3 = _derive(3, p2, stdin_image)
        except (LoopFailure, OSError) as e:
            print(f"  FAIL  --reseed: {e} — nothing written.")
            return 1
    if not (stage1 == stage2 == stage3):
        print("  FAIL  --reseed: fresh .so loop did NOT close (stage1/2/3 differ) — "
              "the source is not self-reproducing; do not freeze.")
        return 1
    h1 = hashlib.sha256(stage1).hexdigest()
    old_seed = _FROZEN.read_bytes() if _FROZEN.exists() else b""
    receipt = {
        "reseeded_from_rust_oracle_sha256": h1,
        "replaced_seed_sha256": (
            hashlib.sha256(old_seed).hexdigest() if old_seed else "none-present"
        ),
        "combined_source_sha256": hashlib.sha256(combined).hexdigest(),
        "combined_source_bytes": str(len(combined)),
        "user_lo": str(user_lo),
        "stage1_sha256": h1,
        "stage2_sha256": hashlib.sha256(stage2).hexdigest(),
        "stage3_sha256": hashlib.sha256(stage3).hexdigest(),
        "oracle_provenance": so_.provenance,
        # The bootstrap policy (docs/INDEPENDENCE_ROADMAP.md, "Bootstrap
        # seed-advance policy") admits a reseed only as a disclosed exception
        # whose artifact says, verbatim, what the docstring above says.
        "seed_chain": "RUST-SEEDED (legacy) — the pure-MIND seed chain is broken "
                      "for that hop: it does not descend from the previous "
                      "pure-MIND compiler",
    }
    try:
        publish_fixture(stage1, manifest_text(
            stage1,
            published_by="self_host_loop_smoke.py --reseed (LEGACY, Rust-seeded)",
            receipt=receipt,
        ))
    except PublicationTorn as e:
        detail = "; ".join(f"{k}: {v}" for k, v in e.backups.items())
        print(f"  FAIL  --reseed: {e}. The fixture is NOT known to be intact; "
              f"recoverable copies were preserved: {detail}.")
        return 1
    except OSError as e:
        print(f"  FAIL  --reseed: publication failed ({e}). The previous fixture "
              f"was rolled back from a pre-staged copy; nothing is half-written.")
        return 1
    print(f"  RESEEDED  frozen bootstrap stage1.elf re-minted from RUST output "
          f"(legacy path): {len(stage1)}B sha256={h1}\n"
          f"            the pure-MIND seed chain is broken for that hop — disclose "
          f"this reseed in the change that commits it\n"
          f"            wrote {_FROZEN} and {_FROZEN_MANIFEST}")
    return 0
