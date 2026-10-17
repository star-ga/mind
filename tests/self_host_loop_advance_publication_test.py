#!/usr/bin/env python3
"""Focused staging, publication, failure, and dispatch controls for --advance."""
from _self_host_loop_advance_support import *

def case_staging_ignores_an_occupied_or_symlinked_conventional_name() -> None:
    """The old fixed `<name>.advance-staging` path is inert.

    It was two bugs: a symlink there would have been followed by an ordinary
    `open(..., "wb")`, redirecting the staged bytes anywhere the link pointed;
    and two concurrent runs shared the one name. Exclusive unique siblings fix
    both. Planted here as BOTH a symlink and a regular file, so publication has
    to ignore the name entirely rather than merely handle one shape.
    """
    with Sandbox() as sb:
        canary = sb.dir / "canary-outside-the-fixture"
        canary.write_bytes(b"canary must not be written through a staging symlink")
        before_canary = canary.read_bytes()
        link = sb.dir / "stage1.elf.advance-staging"
        os.symlink(canary, link)
        squatter = sb.dir / "MANIFEST.txt.advance-staging"
        squatter.write_bytes(b"an occupant at the old conventional name")
        before_squatter = squatter.read_bytes()

        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)

        check("staging: publication still succeeds", rc, 0)
        check("staging: the symlink target was not written",
              canary.read_bytes(), before_canary)
        check("staging: the symlink itself is untouched",
              link.is_symlink() and os.readlink(link) == str(canary), True)
        check("staging: the occupied conventional name is untouched",
              squatter.read_bytes(), before_squatter)
        check("staging: the fixture really was published",
              sb.elf_path.read_bytes(), NEW_POINT)
        check("staging: no staging debris remains",
              [q.name for q in sb.dir.iterdir() if q.name.startswith(".advance-stage-")],
              [])


def case_publication_preserves_the_existing_file_modes() -> None:
    """Publishing must not silently re-permission the fixture."""
    with Sandbox() as sb:
        sb.elf_path.chmod(0o750)
        sb.manifest_path.chmod(0o640)
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("modes: published", rc, 0)
        check("modes: ELF mode preserved", sb.elf_path.stat().st_mode & 0o7777, 0o750)
        check("modes: manifest mode preserved",
              sb.manifest_path.stat().st_mode & 0o7777, 0o640)


def case_the_executed_seed_is_a_snapshot_not_the_live_path() -> None:
    """The bytes that were HASHED are the bytes that RAN.

    Hashing `_FROZEN` and then executing `_FROZEN` again is a
    time-of-check/time-of-use gap: a write landing in between makes the recorded
    `advanced_from_seed_sha256` describe a file that never ran. This case opens
    that window deliberately — the live fixture is replaced immediately before
    the first execution — so a harness that runs a private snapshot still
    executes the captured bytes, while one that re-opens the live path executes
    the impostor. The hook is on `run_elf`, which BOTH shapes call, so the case
    cannot be dodged by simply not creating a snapshot.
    """
    with Sandbox() as sb:
        log: list[bytes] = []

        def swap_the_live_fixture(n: int) -> None:
            if n == 1:
                sb.elf_path.write_bytes(OTHER)

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log,
                              before_call=swap_the_live_fixture),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("snapshot: the executed bytes are the CAPTURED old seed",
              log[:1], [OLD_SEED])
        check("snapshot: the impostor was never executed", OTHER in log, False)
        # And because the live fixture did change, publication is refused too.
        check("snapshot: publication refused over the changed fixture", rc, 1)
        check("snapshot: the changed fixture was not overwritten",
              sb.elf_path.read_bytes(), OTHER)


def case_advance_refuses_to_overwrite_a_fixture_that_changed_mid_run() -> None:
    """A concurrent advance that completed while this one ran must not be
    silently discarded."""
    with Sandbox() as sb:
        log: list[bytes] = []

        def meddle(n: int) -> None:
            if n == 1:  # someone else publishes while we are deriving stages
                sb.elf_path.write_bytes(OTHER)

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log, on_call=meddle),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("changed seed: refused", rc, 1)
        check("changed seed: the other advance was not overwritten",
              sb.elf_path.read_bytes(), OTHER)
        check("changed seed: the manifest was not published",
              sb.manifest_path.read_bytes(), sb.before_manifest)
        check("changed seed: the snapshot still ran the CAPTURED bytes",
              log[:1], [OLD_SEED])


def case_rollback_needs_no_new_allocation_after_the_failure() -> None:
    """The rollback copy is written BEFORE either live file is touched.

    If it were written only after the manifest replace failed, a full disk could
    fail the publish AND the rollback, stranding the new ELF beside the old
    manifest. Asserted by counting staging writes: none may happen after the
    failure.
    """
    with Sandbox() as sb:
        log: list[bytes] = []
        replaces: list[int] = []
        stages_at_failure: list[int] = []
        stage_calls: list[int] = []
        real_replace = os.replace
        real_stage = loop._stage

        def counting_stage(parent, data, *, mode):
            stage_calls.append(1)
            return real_stage(parent, data, mode=mode)

        def flaky(src, dst):
            replaces.append(1)
            if len(replaces) == 2:
                stages_at_failure.append(len(stage_calls))
                raise OSError(28, "No space left on device (planted)")
            return real_replace(src, dst)

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
            _stage=counting_stage,
            _replace=flaky,
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("rollback: publication failure fails the run", rc, 1)
        check("rollback: three files were staged up front", stages_at_failure, [3])
        check("rollback: no staging write happened after the failure",
              len(stage_calls), 3)
        check("rollback: the ELF is back to its previous bytes",
              sb.elf_path.read_bytes(), sb.before_elf)
        check("rollback: the manifest is untouched",
              sb.manifest_path.read_bytes(), sb.before_manifest)
        check("rollback: no staging debris remains",
              [q.name for q in sb.dir.iterdir() if q.name.startswith(".advance-stage-")],
              [])


def case_a_failed_rollback_is_reported_as_torn_with_backups_kept() -> None:
    """When the rollback ITSELF fails, the harness must not claim the old files
    are unchanged — it must say so and name the surviving copies."""
    with Sandbox() as sb:
        log: list[bytes] = []
        replaces: list[int] = []
        real_replace = os.replace

        def flaky(src, dst):
            replaces.append(1)
            if len(replaces) >= 2:  # the manifest publish AND the rollback
                raise OSError(28, "No space left on device (planted)")
            return real_replace(src, dst)

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
            _replace=flaky,
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("torn: the run fails", rc, 1)
        check("torn: the rollback was attempted", len(replaces) >= 3, True)
        # The honest state: the ELF really is the new one and the manifest is old.
        check("torn: the ELF is the unpublished-new one",
              sb.elf_path.read_bytes(), NEW_POINT)
        check("torn: the manifest is still the old one",
              sb.manifest_path.read_bytes(), sb.before_manifest)
        survivors = [q for q in sb.dir.iterdir() if q.name.startswith(".advance-stage-")]
        check("torn: recoverable copies were preserved", len(survivors) >= 1, True)
        check("torn: the old ELF is recoverable from one of them",
              any(q.read_bytes() == sb.before_elf for q in survivors), True)


def case_staging_failure_leaves_no_debris_and_touches_nothing() -> None:
    """A staging error is discovered while both live files are still intact."""
    with Sandbox() as sb:
        log: list[bytes] = []
        calls: list[int] = []
        real_stage = loop._stage

        def failing_stage(parent, data, *, mode):
            calls.append(1)
            if len(calls) == 2:  # the manifest's staging file
                raise OSError(28, "No space left on device (planted)")
            return real_stage(parent, data, mode=mode)

        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
            _stage=failing_stage,
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("staging failure: the run fails", rc, 1)
        check("staging failure: fixture untouched", sb.unchanged(), True)
        check("staging failure: no debris remains",
              [q.name for q in sb.dir.iterdir() if q.name.startswith(".advance-stage-")],
              [])


def case_a_later_stage_failure_is_not_blamed_on_the_old_seed() -> None:
    """stage2/stage3 failures must not be reported as "the old seed refused the
    source" — by then the old seed has already compiled it successfully."""
    import io
    import contextlib

    for stage, chain in (
        (2, (NEW_POINT, b"not an elf", NEW_POINT)),
        (3, (NEW_POINT, NEW_POINT, b"not an elf")),
    ):
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(
                run_elf=stub_loop(chain, log),
                stage0_emit=lambda c, u: NEW_POINT,
                so=lambda: StubOracle(),
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = loop.do_advance(SRC, IMAGE, USER_LO)
            out = buf.getvalue()
            check(f"attribution(stage{stage}): fails", rc, 1)
            check(f"attribution(stage{stage}): names the stage",
                  f"stage{stage}" in out, True)
            check(f"attribution(stage{stage}): does NOT blame the old seed",
                  "old frozen seed refused" in out, False)
            check(f"attribution(stage{stage}): fixture untouched", sb.unchanged(), True)


def case_a_non_elf_stage2_is_refused_before_it_is_executed() -> None:
    """The screen happens before the bytes are written out and run: a garbage
    stage2 must never reach execve as stage3's seed."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, b"not an elf", NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.do_advance(SRC, IMAGE, USER_LO)
        check("non-ELF stage2: fails", rc, 1)
        check("non-ELF stage2: stage3 was never executed", len(log), 2)
        check("non-ELF stage2: fixture untouched", sb.unchanged(), True)


def case_stage_spawn_and_timeout_failures_are_structured_and_harmless() -> None:
    """A spawn error and a timeout both become a numbered stage failure, a
    non-zero verdict, and an untouched fixture."""
    for label, boom in (
        ("timeout", subprocess.TimeoutExpired(cmd="stage2.elf", timeout=120)),
        ("spawn", OSError(8, "Exec format error (planted)")),
    ):
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(
                run_elf=stub_loop((NEW_POINT, boom, NEW_POINT), log),
                stage0_emit=lambda c, u: NEW_POINT,
                so=lambda: StubOracle(),
            )
            rc = loop.do_advance(SRC, IMAGE, USER_LO)
            check(f"stage {label}: structured non-zero verdict", rc, 1)
            check(f"stage {label}: fixture untouched", sb.unchanged(), True)


def case_reseed_later_stage_failures_are_structured_and_harmless() -> None:
    """Legacy --reseed must identify stage2/3 failures and publish nothing.

    These controls use a stub fresh oracle and a synthetic stage1, but exercise
    the reseed path's real `_derive` calls. The malformed output, spawn error,
    and timeout cases all represent failures after the Rust seed emitted stage1.
    """
    import contextlib
    import io

    cases = (
        ("stage2-non-ELF", 2, (b"not an elf",)),
        ("stage2-spawn", 2,
         (OSError(8, "Exec format error (planted)"),)),
        ("stage2-timeout", 2,
         (subprocess.TimeoutExpired(cmd="stage1.elf", timeout=120),)),
        ("stage3-non-ELF", 3, (NEW_POINT, b"not an elf")),
    )
    for label, stage, chain in cases:
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(
                run_elf=stub_loop(chain, log),
                stage0_emit=lambda c, u: NEW_POINT,
                so=lambda: StubOracle(),
            )
            output = io.StringIO()
            raised: BaseException | None = None
            with contextlib.redirect_stdout(output):
                try:
                    rc = loop.do_reseed(SRC, IMAGE, USER_LO)
                except BaseException as exc:  # regression: raw OSError/traceback
                    raised = exc
                    rc = None
            text = output.getvalue()
            check(f"reseed({label}): returns a failure", rc, 1)
            check(f"reseed({label}): does not raise", raised, None)
            check(f"reseed({label}): names stage{stage}", f"stage{stage}:" in text, True)
            check(f"reseed({label}): fixture untouched", sb.unchanged(), True)


def case_preflight_self_host_branch_fails_closed_for_exit_statuses() -> None:
    """The actual preflight loop branch must aggregate BLOCKED as a failure.

    Extracting only that committed branch keeps this control independent of
    Cargo and compiler artifacts. A synthetic `python3` executable supplies
    each gate exit status. The real status dispatch calls a minimal aggregate
    `bad()` control; its log is redirected into this test's private directory.
    """
    source = PREFLIGHT.read_text(encoding="utf-8")
    start_marker = "  loop_rc=0; python3 scripts/run_gate.py --min-asserted 2 "
    end_marker = "\n  fi\n\n  step \"bench gate"
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len("\n  fi")
    branch = source[start:end]

    for fake_rc, want_fail in ((0, 0), (1, 1), (2, 1)):
        with tempfile.TemporaryDirectory(prefix="preflight-control-") as td:
            root = pathlib.Path(td)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                "echo \"synthetic runner rc=$FAKE_RC\"\n"
                "if [ \"$FAKE_RC\" = 2 ]; then echo 'BLOCKED synthetic'; fi\n"
                "if [ \"$FAKE_RC\" = 1 ]; then echo 'FAIL synthetic'; fi\n"
                "exit \"$FAKE_RC\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            script = root / "preflight-loop-snippet.sh"
            script.write_text(
                "#!/bin/sh\n"
                "fail=0\n"
                "bad() { fail=1; }\n"
                + branch.replace("/tmp/preflight-loop.out", str(root / "loop.out"))
                + "\nprintf 'aggregate_fail=%s\\n' \"$fail\"\n",
                encoding="utf-8",
            )
            script.chmod(0o755)
            env = dict(os.environ)
            env.update({"PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                        "FAKE_RC": str(fake_rc)})
            result = subprocess.run(
                ["bash", str(script)], cwd=str(REPO), env=env,
                capture_output=True, text=True, timeout=30,
            )
            check(f"preflight synthetic exit{fake_rc}: snippet runs",
                  result.returncode, 0)
            check(f"preflight synthetic exit{fake_rc}: aggregate bad count",
                  f"aggregate_fail={want_fail}" in result.stdout, True)


def case_derive_labels_the_stage_it_failed_on() -> None:
    """`LoopFailure` carries the stage number, which is what the attribution
    above is built on. Asserted directly so a silent renumbering is caught."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(run_elf=stub_loop((NEW_POINT, b"garbage", NEW_POINT), log))
        seed = sb.dir / "seed.elf"
        seed.write_bytes(OLD_SEED)
        seed.chmod(0o755)
        try:
            loop.run_loop_from_seed(seed, IMAGE, sb.dir)
            failed = None
        except loop.LoopFailure as e:
            failed = e.stage
        check("derive: the failing stage is identified", failed, 2)
        check("derive: LoopFailure is a RuntimeError (verify path still catches it)",
              issubclass(loop.LoopFailure, RuntimeError), True)


# ── dispatch through the REAL main(), not do_advance directly ──────────────

def case_main_advance_does_not_resolve_the_oracle_before_admission() -> None:
    """Order through the real entry point: mode gate, then seed admission, and
    only then the oracle.

    Resolving early is not free — `so()` can shell out to `mindc build
    --emit=cdylib` — so an --advance with no seed to advance from must refuse
    without ever building one.
    """
    with Sandbox() as sb:
        sb.elf_path.unlink()
        resolved: list[str] = []
        sb.patch(
            build_seed=lambda: (SRC, IMAGE, USER_LO),
            so=lambda: resolved.append("so") or StubOracle(),
            run_elf=lambda *a, **k: b"",
        )
        rc = loop.main(["--advance"], {})
        check("main(--advance): blocked with no seed", rc, 2)
        check("main(--advance): the oracle was never resolved", resolved, [])


def case_main_advance_publishes_through_the_real_dispatch() -> None:
    """The orchestration a mocked `do_advance` call cannot see: argv -> mode ->
    build_seed -> do_advance -> published fixture."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            build_seed=lambda: (SRC, IMAGE, USER_LO),
            run_elf=stub_loop((NEW_POINT, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc = loop.main(["--advance"], {})
        check("main(--advance): publishes", rc, 0)
        check("main(--advance): the fixture holds the new point",
              sb.elf_path.read_bytes(), NEW_POINT)
        check("main(--advance): env form dispatches identically",
              loop.requested_modes([], {loop.ADVANCE_ENV: "1"})[0],
              [f"{loop.ADVANCE_ENV}=1"])


def case_main_reseed_dispatches_to_the_legacy_path() -> None:
    """--reseed still reaches its own mode and is not silently re-routed."""
    with Sandbox() as sb:
        seen: list[str] = []
        sb.patch(
            build_seed=lambda: (SRC, IMAGE, USER_LO),
            do_reseed=lambda c, i, u: seen.append("reseed") or 0,
            do_advance=lambda c, i, u: seen.append("advance") or 0,
        )
        check("main(--reseed): dispatches to reseed", loop.main(["--reseed"], {}), 0)
        check("main(--reseed): reached the legacy path", seen, ["reseed"])
        seen.clear()
        check("main(env reseed): dispatches to reseed",
              loop.main([], {loop.RESEED_ENV: "1"}), 0)
        check("main(env reseed): reached the legacy path", seen, ["reseed"])


# ── two-hop advance: what is published, and what is refused ────────────────

def case_two_hop_publishes_stage2_once_stage2_equals_stage3() -> None:
    """ACCEPT. The old seed compiles the new source into an intermediate
    compiler (stage1, old codegen); stage1 compiles the same source into the new
    fixed point; stage2 == stage3 and the fresh oracle agrees with stage2. The
    PUBLISHED image is stage2 — never the intermediate — and the receipt says
    which stage that is, without a `stage1_sha256` row a reader could mistake
    for the published hash."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            build_seed=lambda: (SRC, IMAGE, USER_LO),
            run_elf=stub_loop((INTERMEDIATE, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc, out = captured(loop.main, ["--advance", "--hops", "2"], {})
        check("two hop: publishes on stage2 == stage3 + oracle agreement", rc, 0)
        check("two hop: the fixture holds stage2, not the intermediate",
              sb.elf_path.read_bytes(), NEW_POINT)
        check("two hop: published ELF is executable",
              bool(sb.elf_path.stat().st_mode & 0o111), True)
        # The chain is asserted by CONTENT: the old seed ran first, then the
        # intermediate it emitted, then stage2. Nothing Rust-emitted was run.
        check("two hop: executed old seed -> intermediate -> stage2",
              log, [OLD_SEED, INTERMEDIATE, NEW_POINT])
        text = sb.manifest_path.read_text()
        rows = receipt_rows(text)
        for key, value in (
            ("advance_hops", "2"),
            ("advanced_from_seed_sha256", sha(OLD_SEED)),
            ("combined_source_sha256", sha(SRC)),
            ("intermediate_stage1_sha256", sha(INTERMEDIATE)),
            ("stage2_sha256", sha(NEW_POINT)),
            ("stage3_sha256", sha(NEW_POINT)),
            ("published_stage", "stage2"),
            ("oracle_sha256", sha(NEW_POINT)),
            ("oracle_provenance", "fresh-build"),
        ):
            check(f"two hop receipt: {key}", rows.get(key), value)
        check("two hop receipt: no ambiguous stage1_sha256 row",
              "stage1_sha256" in rows, False)
        check("two hop receipt: names the pure-MIND chain",
              rows.get("seed_chain", "").startswith("PURE-MIND two-hop"), True)
        check("two hop receipt: names the publishing mode",
              rows.get("published-by"), "self_host_loop_smoke.py --advance --hops 2")
        check("two hop receipt: the pinned row is stage2's",
              f"stage1.elf\t{len(NEW_POINT)}\t{sha(NEW_POINT)}" in text, True)
        check("two hop: the report names the intermediate hash",
              sha(INTERMEDIATE) in out, True)


def case_two_hop_refused_unless_stage2_equals_stage3() -> None:
    """REFUSE. stage2 != stage3 means the compiler stage1 emitted does not
    reproduce itself — a miscompile or nondeterminism, not a codegen change.
    A third hop is never the answer; both stage1 shapes are refused."""
    for label, chain in (
        ("stage1 != stage2", (INTERMEDIATE, NEW_POINT, OTHER)),
        ("stage1 == stage2", (NEW_POINT, NEW_POINT, OTHER)),
    ):
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(
                run_elf=stub_loop(chain, log),
                stage0_emit=lambda c, u: NEW_POINT,
                so=lambda: StubOracle(),
            )
            rc, out = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=2)
            check(f"two hop refuse({label}): stage2 != stage3 fails", rc, 1)
            check(f"two hop refuse({label}): fixture untouched", sb.unchanged(), True)
            check(f"two hop refuse({label}): all three stages were derived", len(log), 3)
            check(f"two hop refuse({label}): the report names stage2 != stage3",
                  "stage2 != stage3" in out, True)


def case_two_hop_oracle_must_corroborate_the_published_stage2() -> None:
    """An oracle that agrees with the INTERMEDIATE (old-codegen) stage1 but not
    with stage2 corroborates the wrong image: refused, nothing written."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((INTERMEDIATE, NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: INTERMEDIATE,
            so=lambda: StubOracle(),
        )
        rc, _ = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=2)
        check("two hop: oracle == intermediate != stage2 is refused", rc, 1)
        check("two hop: oracle mismatch leaves the fixture untouched",
              sb.unchanged(), True)


def case_two_hop_stays_strict_on_the_shared_refusals() -> None:
    """The shared one-hop refusals still bind with --hops 2: no seed to advance
    from, a missing or stale oracle, and the oracle deferral variable."""
    with Sandbox() as sb:
        sb.elf_path.unlink()
        tripped: list[str] = []
        sb.patch(
            run_elf=lambda *a, **k: tripped.append("run_elf") or b"",
            so=lambda: tripped.append("so") or StubOracle(),
        )
        rc, _ = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=2)
        check("two hop: missing seed is blocked", rc, 2)
        check("two hop: missing seed runs nothing", tripped, [])
    for label, oracle in (
        ("missing oracle", StubOracle(present=False)),
        ("stale oracle", StubOracle(provenance="legacy-in-tree")),
    ):
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(
                run_elf=stub_loop((INTERMEDIATE, NEW_POINT, NEW_POINT), log),
                stage0_emit=lambda c, u: NEW_POINT,
                so=lambda o=oracle: o,
            )
            prior = os.environ.get(loop.ORACLE_DEFER_ENV)
            os.environ[loop.ORACLE_DEFER_ENV] = "1"
            try:
                rc, _ = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=2)
            finally:
                if prior is None:
                    os.environ.pop(loop.ORACLE_DEFER_ENV, None)
                else:
                    os.environ[loop.ORACLE_DEFER_ENV] = prior
            check(f"two hop: {label} is blocked even with the deferral set", rc, 2)
            check(f"two hop: {label} leaves the fixture untouched", sb.unchanged(), True)


def case_two_hop_old_seed_refusal_is_a_real_process_failure() -> None:
    """REAL process boundary with --hops 2: an old seed that exits non-zero is
    refused through run_elf's own handling, and nothing is written."""
    with Sandbox() as sb:
        sb.elf_path.write_text("#!/bin/sh\necho 'refused' >&2\nexit 3\n")
        sb.elf_path.chmod(0o755)
        sb.before_elf = sb.elf_path.read_bytes()
        sb.patch(so=lambda: StubOracle(), stage0_emit=lambda c, u: NEW_POINT)
        rc, out = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=2)
        check("two hop: old-seed refusal fails", rc, 1)
        check("two hop: old-seed refusal is attributed to the old seed",
              "old frozen seed refused" in out, True)
        check("two hop: old-seed refusal leaves the fixture untouched",
              sb.unchanged(), True)


def case_reseed_receipt_records_the_broken_seed_chain_verbatim() -> None:
    """The bootstrap policy allows --reseed only as a disclosed exception whose
    receipt records "the pure-MIND seed chain is broken for that hop", verbatim
    from the legacy module's own contract. A Rust-seeded re-mint that does not
    say so in the artifact it publishes is the silent reseed the policy bans."""
    with Sandbox() as sb:
        log: list[bytes] = []
        sb.patch(
            run_elf=stub_loop((NEW_POINT, NEW_POINT), log),
            stage0_emit=lambda c, u: NEW_POINT,
            so=lambda: StubOracle(),
        )
        rc, out = captured(loop.do_reseed, SRC, IMAGE, USER_LO)
        check("reseed receipt: the stub re-mint publishes", rc, 0)
        check("reseed receipt: the fixture holds the re-minted ELF",
              sb.elf_path.read_bytes(), NEW_POINT)
        rows = receipt_rows(sb.manifest_path.read_text())
        check("reseed receipt: a seed_chain row exists", "seed_chain" in rows, True)
        check("reseed receipt: records the broken pure-MIND chain verbatim",
              "the pure-MIND seed chain is broken for that hop"
              in rows.get("seed_chain", ""), True)
        check("reseed report: says the chain is broken",
              "the pure-MIND seed chain is broken for that hop" in out, True)
