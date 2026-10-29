#!/usr/bin/env python3
"""Focused admission, mode, oracle, and receipt controls for --advance."""
from _self_host_loop_advance_support import *

def case_advance_publishes_a_new_fixed_point() -> None:
    """A fixed point DISTINCT from the old seed is published, seeded by the old
    seed, and the manifest carries the full receipt."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda combined, user_lo: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)

        check("advance: publishes on full agreement", rc, 0)
        check("advance: the new point differs from the old seed",
              NEW_POINT != OLD_SEED, True)
        check("advance: stage1.elf now holds the new fixed point",
              sb.elf_path.read_bytes(), NEW_POINT)
        check("advance: published ELF is executable",
              bool(sb.elf_path.stat().st_mode & 0o111), True)

        # The old seed must be the FIRST thing invoked, and it is asserted by
        # CONTENT: the harness executes a private snapshot, so a path check would
        # pass on any file that happened to sit there. This is what makes the
        # recorded advanced_from_seed_sha256 describe the thing that actually ran.
        check("advance: the bytes that ran first are the captured old seed",
              log[:1], [OLD_SEED])
        check("advance: the snapshot was not the live fixture path",
              sb.elf_path.read_bytes() == NEW_POINT, True)
        check("advance: three stages were derived", len(log), 3)

        text = sb.manifest_path.read_text()
        for key, value in (
            ("advanced_from_seed_sha256", sha(OLD_SEED)),
            ("combined_source_sha256", sha(SRC)),
            ("stage1_sha256", sha(NEW_POINT)),
            ("stage2_sha256", sha(NEW_POINT)),
            ("stage3_sha256", sha(NEW_POINT)),
            ("oracle_sha256", sha(NEW_POINT)),
        ):
            check(f"advance receipt records {key}", f"# {key}\t{value}" in text, True)
        check("advance receipt names the publishing mode",
              "published-by\tself_host_loop_smoke.py --advance" in text, True)
        check("advance receipt records the row for the new ELF",
              f"stage1.elf\t{len(NEW_POINT)}\t{sha(NEW_POINT)}" in text, True)


# ── refusals: every one must leave BOTH files untouched ────────────────────

def case_advance_refused_when_the_old_seed_rejects_the_source() -> None:
    """REAL process boundary: an executable seed that exits non-zero.

    `run_elf` is NOT stubbed here — a real subprocess is spawned, exits 3, and
    the refusal travels through `run_elf`'s non-zero handling into `do_advance`.
    """
    with Sandbox() as sb:
        sb.elf_path.write_text("#!/bin/sh\necho 'refused' >&2\nexit 3\n")
        sb.elf_path.chmod(0o755)
        sb.before_elf = sb.elf_path.read_bytes()
        sb.patch(so=lambda: StubOracle(), stage0_emit=lambda c, u: NEW_POINT)
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: old-seed refusal fails", rc, 1)
        check("advance: old-seed refusal leaves the fixture untouched",
              sb.unchanged(), True)


def case_advance_refused_on_stage_mismatch() -> None:
    """stage1 != stage2 is not a fixed point; a non-reproducing compiler must
    never become the seed."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, OTHER, OTHER), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: stage mismatch fails", rc, 1)
        check("advance: stage mismatch leaves the fixture untouched",
              sb.unchanged(), True)


def case_advance_refused_on_non_elf_stage() -> None:
    """A later stage that is not a static ELF is refused (the first stage is
    already screened inside run_loop_from_seed)."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, b"not an elf", b"not an elf"), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: non-ELF stage fails", rc, 1)
        check("advance: non-ELF stage leaves the fixture untouched",
              sb.unchanged(), True)


def case_advance_refused_when_the_oracle_is_missing() -> None:
    """One witness is not corroboration: with no oracle the advance is blocked."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(present=False),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: missing oracle is blocked", rc, 2)
        check("advance: missing oracle leaves the fixture untouched",
              sb.unchanged(), True)


def case_real_resolver_refuses_an_explicitly_missing_oracle_before_load() -> None:
    """The real resolver must reject a missing MINDC_SO before ctypes is used.

    The normal advance cases use a stub oracle to stay compiler-free. This case
    deliberately uses the production `so()`/`resolve_so()` path and only stubs
    the old-seed loop, so a missing explicit handle cannot be mistaken for a
    synthetic oracle result.
    """
    with Sandbox() as sb:
        log: list[bytes] = []
        loaded: list[object] = []
        prior_so = loop._SO
        resolver_globals = loop.resolve_so.__globals__
        prior_provenance = resolver_globals["PROVENANCE"]
        prior_legacy_fallback = resolver_globals["USED_LEGACY_FALLBACK"]
        prior_env = os.environ.get("MINDC_SO")
        real_cdll = loop.ctypes.CDLL
        missing = sb.dir / "missing-oracle.so"

        def forbidden_load(*args, **kwargs):
            loaded.append(args[0] if args else None)
            raise AssertionError("ctypes must not load a resolver-refused oracle")

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            _SO=None,
        )
        loop.ctypes.CDLL = forbidden_load
        os.environ["MINDC_SO"] = str(missing)
        caught: BaseException | None = None
        try:
            loop.do_advance(SRC, IMAGE, USER_LO)
        except BaseException as exc:  # strict resolver uses SystemExit
            caught = exc
        finally:
            loop.ctypes.CDLL = real_cdll
            loop._SO = prior_so
            resolver_globals["PROVENANCE"] = prior_provenance
            resolver_globals["USED_LEGACY_FALLBACK"] = prior_legacy_fallback
            if prior_env is None:
                os.environ.pop("MINDC_SO", None)
            else:
                os.environ["MINDC_SO"] = prior_env

        check("real resolver: missing MINDC_SO is refused", isinstance(caught, SystemExit), True)
        check("real resolver: missing path is named",
              "MINDC_SO" in str(caught), True)
        check("real resolver: ctypes was never loaded", loaded, [])
        check("real resolver: missing-oracle fixture is unchanged", sb.unchanged(), True)


def case_real_resolver_refuses_a_deliberately_old_oracle_before_load() -> None:
    """A real explicit artifact older than the compile inputs is refused."""
    with Sandbox() as sb:
        log: list[bytes] = []
        loaded: list[object] = []
        prior_so = loop._SO
        resolver_globals = loop.resolve_so.__globals__
        prior_provenance = resolver_globals["PROVENANCE"]
        prior_legacy_fallback = resolver_globals["USED_LEGACY_FALLBACK"]
        prior_env = os.environ.get("MINDC_SO")
        real_cdll = loop.ctypes.CDLL
        old_oracle = sb.dir / "old-oracle.so"
        old_oracle.write_bytes(b"deliberately old oracle")
        os.utime(old_oracle, ns=(1, 1))

        def forbidden_load(*args, **kwargs):
            loaded.append(args[0] if args else None)
            raise AssertionError("ctypes must not load a stale resolver oracle")

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            _SO=None,
        )
        loop.ctypes.CDLL = forbidden_load
        os.environ["MINDC_SO"] = str(old_oracle)
        caught: BaseException | None = None
        try:
            loop.do_advance(SRC, IMAGE, USER_LO)
        except BaseException as exc:  # strict resolver uses SystemExit
            caught = exc
        finally:
            loop.ctypes.CDLL = real_cdll
            loop._SO = prior_so
            resolver_globals["PROVENANCE"] = prior_provenance
            resolver_globals["USED_LEGACY_FALLBACK"] = prior_legacy_fallback
            if prior_env is None:
                os.environ.pop("MINDC_SO", None)
            else:
                os.environ["MINDC_SO"] = prior_env

        check("real resolver: old MINDC_SO is refused", isinstance(caught, SystemExit), True)
        check("real resolver: stale provenance is named",
              "NON-FRESH" in str(caught), True)
        check("real resolver: stale oracle was never loaded", loaded, [])
        check("real resolver: stale-oracle fixture is unchanged", sb.unchanged(), True)


def case_advance_refused_on_a_stale_oracle() -> None:
    """Bytes of unknown age cannot corroborate a fixed point derived from the
    CURRENT source, even when they would agree."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(provenance="legacy-in-tree"),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: stale oracle is blocked", rc, 2)
        check("advance: stale oracle leaves the fixture untouched",
              sb.unchanged(), True)


def case_advance_refused_when_the_oracle_disagrees() -> None:
    """Two independent compilers disagreeing about the current source means
    neither may be frozen."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: OTHER,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: oracle mismatch fails", rc, 1)
        check("advance: oracle mismatch leaves the fixture untouched",
              sb.unchanged(), True)


def case_advance_oracle_leg_is_not_deferrable() -> None:
    """The deferral env var records a one-leg VERIFICATION run. It must not buy
    a publication with the corroborating witness switched off."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(present=False),
        )
        prior = os.environ.get(loop.ORACLE_DEFER_ENV)
        os.environ[loop.ORACLE_DEFER_ENV] = "1"
        try:
            rc = loop.do_advance(SRC, IMAGE, USER_LO)
        finally:
            if prior is None:
                os.environ.pop(loop.ORACLE_DEFER_ENV, None)
            else:
                os.environ[loop.ORACLE_DEFER_ENV] = prior
        check("advance: oracle deferral does not buy a publication", rc, 2)
        check("advance: deferred-oracle attempt leaves the fixture untouched",
              sb.unchanged(), True)


def case_advance_rolls_back_when_publication_fails_part_way() -> None:
    """The ELF is replaced, the manifest replacement then fails, and the ELF is
    rolled back to exactly its previous bytes.

    Two independent paths cannot be swapped atomically, so this is the case that
    decides whether a torn publication is repaired or left on disk.
    """
    with Sandbox() as sb:
        log: list[bytes] = []
        calls: list[int] = []
        real_replace = os.replace

        def flaky(src, dst):
            calls.append(1)
            if len(calls) == 2:  # the manifest replacement
                raise OSError(28, "No space left on device (planted)")
            return real_replace(src, dst)

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
            _replace=flaky,
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: publication failure fails the run", rc, 1)
        check("advance: the ELF replacement was actually attempted",
              len(calls) >= 2, True)
        check("advance: publication failure rolls the ELF back",
              sb.elf_path.read_bytes(), sb.before_elf)
        check("advance: publication failure leaves the manifest untouched",
              sb.manifest_path.read_bytes(), sb.before_manifest)
        check("advance: no staging file is left behind",
              sorted(p.name for p in sb.dir.iterdir()),
              ["MANIFEST.txt", "stage1.elf"])


def case_advance_blocked_when_there_is_no_seed_to_advance_from() -> None:
    """--advance is defined by the seed it starts from; without one it is the
    --reseed case and must say so rather than mint anything."""
    with Sandbox() as sb:
        sb.elf_path.unlink()
        tripped: list[str] = []
        sb.patch(
            run_elf=lambda *a, **k: tripped.append("run_elf") or b"",
            so=lambda: tripped.append("so") or StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("advance: missing seed is blocked", rc, 2)
        check("advance: missing seed runs nothing", tripped, [])
        check("advance: missing seed does not create an ELF",
              sb.elf_path.exists(), False)
        check("advance: missing seed leaves the manifest untouched",
              sb.manifest_path.read_bytes(), sb.before_manifest)


# ── mode selection ─────────────────────────────────────────────────────────

def case_advance_and_reseed_together_are_refused_before_any_work() -> None:
    """All four request shapes are refused, and the refusal happens before the
    source is read, before any process runs and before any oracle is resolved."""
    shapes = (
        ("flag+flag", ["--advance", "--reseed"], {}),
        ("flag+env", ["--advance"], {loop.RESEED_ENV: "1"}),
        ("env+flag", ["--reseed"], {loop.ADVANCE_ENV: "1"}),
        ("env+env", [], {loop.ADVANCE_ENV: "1", loop.RESEED_ENV: "1"}),
    )
    for label, argv, env in shapes:
        with Sandbox() as sb:
            tripped: list[str] = []
            sb.patch(
                build_seed=lambda: tripped.append("build_seed") or (SRC, IMAGE, USER_LO),
                run_elf=lambda *a, **k: tripped.append("run_elf") or b"",
                stage0_emit=lambda *a, **k: tripped.append("stage0_emit") or b"",
                so=lambda: tripped.append("so") or StubOracle(),
            )
            rc = loop.main(argv, env)
            check(f"conflict({label}): blocked", rc, 2)
            check(f"conflict({label}): nothing read, run or resolved", tripped, [])
            check(f"conflict({label}): fixture untouched", sb.unchanged(), True)


def case_a_single_mode_request_is_not_a_conflict() -> None:
    """The conflict gate must not refuse the ordinary single-mode requests it
    exists to disambiguate — otherwise it would be a mode killer, not a gate."""
    cases = (
        ("advance flag", ["--advance"], {}, ["--advance"], []),
        ("reseed flag", ["--reseed"], {}, [], ["--reseed"]),
        ("advance env", [], {loop.ADVANCE_ENV: "1"}, [f"{loop.ADVANCE_ENV}=1"], []),
        ("reseed env", [], {loop.RESEED_ENV: "1"}, [], [f"{loop.RESEED_ENV}=1"]),
        ("neither", [], {}, [], []),
    )
    for label, argv, env, want_adv, want_res in cases:
        adv, res = loop.requested_modes(argv, env)
        check(f"modes({label}): advance sources", adv, want_adv)
        check(f"modes({label}): reseed sources", res, want_res)
        check(f"modes({label}): no conflict", loop.mode_conflict(adv, res), None)


# ── the ordinary mode must not have been weakened ──────────────────────────

def case_verify_mode_still_requires_the_frozen_fixed_point() -> None:
    """Ordinary verification keeps asserting stage1 == the frozen seed. --advance
    is the ONLY mode where a new fixed point is acceptable."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            build_seed=lambda: (SRC, IMAGE, USER_LO),
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.main([], {})
        check("verify: a new fixed point is still a failure", rc, 1)
        check("verify: failure leaves the fixture untouched", sb.unchanged(), True)


def case_verify_mode_still_requires_a_fresh_oracle() -> None:
    """A stale oracle that agrees with an equally stale seed still fails."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            build_seed=lambda: (SRC, IMAGE, USER_LO),
            run_elf=stub_loop((OLD_SEED, OLD_SEED, OLD_SEED), log),
            stage0_emit=lambda c, u: OLD_SEED,
            so=lambda: StubOracle(provenance="legacy-in-tree"),
        )
        rc = loop.main([], {})
        check("verify: a stale agreeing oracle still fails", rc, 1)
        check("verify: stale-oracle failure leaves the fixture untouched",
              sb.unchanged(), True)


def case_verify_mode_passes_on_an_unchanged_fixed_point() -> None:
    """Positive control for the ordinary mode: the unmodified path still passes,
    so the failures above are the assertions firing and not the mode broken."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            build_seed=lambda: (SRC, IMAGE, USER_LO),
            run_elf=stub_loop((OLD_SEED, OLD_SEED, OLD_SEED), log),
            stage0_emit=lambda c, u: OLD_SEED,
            so=lambda: StubOracle(),
        )
        rc = loop.main([], {})
        check("verify: unchanged fixed point + fresh oracle passes", rc, 0)
        check("verify: a passing run publishes nothing", sb.unchanged(), True)


# ── the receipt ────────────────────────────────────────────────────────────

def case_receipt_is_reproducible_and_carries_no_timestamp() -> None:
    """Same inputs, same bytes — the receipt is derived, never observed.

    A timestamp cannot be re-derived, so it could not be checked by anyone
    later; it would also make the file differ from itself on every run.
    """
    receipt = {"advanced_from_seed_sha256": sha(OLD_SEED), "stage1_sha256": sha(NEW_POINT)}
    first = loop.manifest_text(NEW_POINT, published_by="x --advance", receipt=receipt)
    second = loop.manifest_text(NEW_POINT, published_by="x --advance", receipt=receipt)
    check("receipt: identical inputs give identical bytes", first, second)
    check("receipt: records the ELF it belongs to",
          f"stage1.elf\t{len(NEW_POINT)}\t{sha(NEW_POINT)}" in first, True)
    # A torn pair is detectable precisely because the manifest names the ELF's
    # hash; assert that hash is the published one and not a placeholder.
    check("receipt: the recorded hash is the new point's",
          sha(NEW_POINT) in first, True)
    # Scan the RECEIPT ROWS only (`# <key>\t<value>`), not the prose header —
    # the header explains why there is no timestamp, and a naive substring
    # search over the whole file matches that explanation instead of a field.
    rows = [ln[2:].split("\t", 1)[0] for ln in first.splitlines()
            if ln.startswith("# ") and "\t" in ln]
    check("receipt: rows were actually found (the scan is not vacuous)",
          "stage1_sha256" in rows, True)
    clockish = [k for k in rows
                if any(w in k.lower() for w in ("time", "date", "unix", "clock"))]
    check("receipt: no wall-clock field is emitted", clockish, [])


def case_import_does_not_resolve_or_build_the_oracle() -> None:
    """Importing the harness must be a pure operation.

    `resolve_so()` may shell out to `mindc build --emit=cdylib` and exits the
    process on a non-fresh oracle. Doing that at import time made a plain
    `import self_host_loop_smoke` — which a sibling gate and this test both do —
    able to launch a compiler build nobody asked for. Run in a CLEAN subprocess
    so this file's own import cannot mask the regression.
    """
    probe = (
        "import sys; sys.path.insert(0, %r)\n"
        "import self_host_loop_smoke as m\n"
        "print('SO=', m._SO)\n" % str(SMOKE_DIR)
    )
    r = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )
    check("import: exits cleanly", r.returncode, 0)
    check("import: oracle is not resolved", "SO= None" in r.stdout, True)
    check("import: no build was attempted",
          "mindc build" in (r.stdout + r.stderr), False)


# ── two-hop advance: argument admission (T0.3 bootstrap policy) ────────────

def case_hops_argument_is_refused_before_any_work() -> None:
    """`--hops` qualifies --advance and accepts exactly 1 or 2. Every other
    shape is refused before the source is read, any process runs, or an oracle
    is resolved — a malformed policy request must not have run a compiler.

    `--hops` with --reseed is refused rather than ignored: a hop count names a
    PURE-MIND derivation, and --reseed has none, so silently accepting it would
    let a Rust-seeded re-mint wear a pure-MIND flag."""
    shapes = (
        ("hops without advance", ["--hops", "2"], {}),
        ("hops with reseed flag", ["--reseed", "--hops", "2"], {}),
        ("hops with reseed env", ["--hops", "2"], {loop.RESEED_ENV: "1"}),
        ("three hops", ["--advance", "--hops", "3"], {}),
        ("zero hops", ["--advance", "--hops", "0"], {}),
        ("non-numeric", ["--advance", "--hops", "two"], {}),
        ("missing value", ["--advance", "--hops"], {}),
        ("value is a flag", ["--hops", "--advance"], {}),
        ("duplicate", ["--advance", "--hops", "2", "--hops", "2"], {}),
        ("equals form, bad value", ["--advance", "--hops=9"], {}),
        ("equals form, empty", ["--advance", "--hops="], {}),
    )
    for label, argv, env in shapes:
        with Sandbox() as sb:
            tripped: list[str] = []
            sb.patch(
                build_seed=lambda: tripped.append("build_seed") or (SRC, IMAGE, USER_LO),
                run_elf=lambda *a, **k: tripped.append("run_elf") or b"",
                stage0_emit=lambda *a, **k: tripped.append("stage0_emit") or b"",
                so=lambda: tripped.append("so") or StubOracle(),
            )
            rc, text = captured(loop.main, argv, env)
            check(f"hops({label}): blocked", rc, 2)
            check(f"hops({label}): nothing read, run or resolved", tripped, [])
            check(f"hops({label}): fixture untouched", sb.unchanged(), True)
            check(f"hops({label}): the refusal names --hops", "--hops" in text, True)


def case_hops_forms_reach_the_two_hop_advance() -> None:
    """Every accepted spelling reaches the two-hop mode and publishes stage2:
    space and equals forms, and the environment form of --advance qualified by
    the --hops flag. Asserted by the published bytes, not by a dispatch flag."""
    forms = (
        ("space form", ["--advance", "--hops", "2"], {}),
        ("equals form", ["--advance", "--hops=2"], {}),
        ("env advance + flag", ["--hops", "2"], {loop.ADVANCE_ENV: "1"}),
    )
    for label, argv, env in forms:
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(
                build_seed=lambda: (SRC, IMAGE, USER_LO),
                run_elf=stub_loop((INTERMEDIATE, NEW_POINT, NEW_POINT), log),
                stage0_emit=lambda c, u: NEW_POINT,
                so=lambda: StubOracle(),
            )
            rc, _ = captured(loop.main, argv, env)
            check(f"hops form({label}): publishes", rc, 0)
            check(f"hops form({label}): the fixture holds stage2",
                  sb.elf_path.read_bytes(), NEW_POINT)


def case_one_hop_still_refuses_a_codegen_change_and_names_two_hops() -> None:
    """The default (one-hop) --advance is NOT weakened: stage1 != stage2 is
    refused exactly as before, with --hops 1 spelled out or not. When the
    shape is a legitimate two-hop one (stage2 == stage3) the refusal must name
    `--advance --hops 2` as the next step — never steer to --reseed."""
    for label, argv in (("implicit", ["--advance"]), ("explicit", ["--advance", "--hops", "1"])):
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(
                build_seed=lambda: (SRC, IMAGE, USER_LO),
                run_elf=stub_loop((INTERMEDIATE, NEW_POINT, NEW_POINT), log),
                stage0_emit=lambda c, u: NEW_POINT,
                so=lambda: StubOracle(),
            )
            rc, text = captured(loop.main, argv, {})
            check(f"one hop({label}): stage1 != stage2 is refused", rc, 1)
            check(f"one hop({label}): fixture untouched", sb.unchanged(), True)
            check(f"one hop({label}): the refusal names the two-hop advance",
                  "--advance --hops 2" in text, True)
            check(f"one hop({label}): the refusal does not steer to --reseed",
                  "--reseed" in text, False)


def case_bootstrap_policy_is_written_where_the_gate_lifecycle_lives() -> None:
    """The policy the code enforces is recorded in the roadmap's Gate lifecycle
    section, names the command that implements it, and quotes the reseed
    disclosure verbatim from the legacy module it cites. The operator contract
    no longer prescribes --reseed for source drift.

    Positive controls first: each file is read and its anchor heading exists,
    so an absent phrase means the text is missing, not that nothing was read."""
    roadmap = (REPO / "docs" / "INDEPENDENCE_ROADMAP.md").read_text()
    start = roadmap.find("## Gate lifecycle")
    check("policy: the Gate lifecycle section exists", start >= 0, True)
    end = roadmap.find("\n## ", start + 1)
    section = roadmap[start:end if end > start else len(roadmap)]
    phrase = "the pure-MIND seed chain is broken for that hop"
    legacy = (SMOKE_DIR / "_selfhost_loop_reseed.py").read_text()
    check("policy: the cited phrase is verbatim in _selfhost_loop_reseed.py",
          phrase in legacy, True)
    for needle in ("Bootstrap seed-advance policy", "--advance --hops 2",
                   "stage2 == stage3", phrase):
        check(f"policy: Gate lifecycle states {needle!r}", needle in section, True)
    contract = (SMOKE_DIR / "self_host_loop_smoke.md").read_text()
    check("policy: the operator contract was read", "FAIL-CLOSED" in contract, True)
    check("policy: the operator contract documents --hops 2",
          "--advance --hops 2" in contract, True)
    check("policy: the operator contract no longer prescribes --reseed for drift",
          "re-freeze with --reseed in the same change" in contract, False)
