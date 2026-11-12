#!/usr/bin/env python3
"""Runner for the self-host LOOP advancement harness controls.

The stable command remains python3 tests/self_host_loop_advance_test.py.
Cases live in focused modules so each file stays readable and below the
repository's 800-line harness ceiling. The seed-chain controls (no-op refusal,
computed `seed_chain` row, --verify-chain) live here.
"""
from __future__ import annotations

import pathlib
import shlex
import subprocess
import sys
import tempfile

import self_host_loop_advance_contract_test as contract
import self_host_loop_advance_publication_test as publication
from _self_host_loop_advance_support import (
    FAILURES, IMAGE, INTERMEDIATE, NEW_POINT, OLD_SEED, SRC, USER_LO, Sandbox,
    StubOracle, captured, check, loop, receipt_rows, sha, stub_loop,
)
from _selfhost_loop_reseed import (ChainBlocked, git_receipt_history, parse_receipt,
                                   walk_seed_chain)

ROOT = loop.SEED_CHAIN_ROOT
FROM = "advanced_from_seed_sha256"
NOOP = "BLOCKED: --advance would republish the current seed unchanged; nothing written."


def receipt(seed: bytes, **rows: str) -> str:
    return loop.manifest_text(seed, published_by="synthetic", receipt=rows)


#: Every in-process case walks THIS history, never the repository's own refs:
#: one synthetic receipt that publishes the sandbox's OLD_SEED from the root.
#: (case_repository_seed_chain_verifies runs the real verifier in a subprocess.)
ROOT_HOP = ("synthetic-root-hop", receipt(OLD_SEED, **{FROM: ROOT}))


def case_noop_advance_is_refused_and_writes_nothing() -> None:
    """A-B1: an advance whose fixed point IS the current seed publishes nothing."""
    for label, hops, stages in (("one hop", 1, (OLD_SEED,) * 3),
                                ("two hops", 2, (INTERMEDIATE, OLD_SEED, OLD_SEED))):
        with Sandbox() as sb:
            log: list[bytes] = []
            sb.patch(run_elf=stub_loop(stages, log),
                     stage0_emit=lambda c, u: OLD_SEED, so=lambda: StubOracle())
            rc, out = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=hops)
            check(f"no-op advance ({label}): refused with rc 2", rc, 2)
            check(f"no-op advance ({label}): names the refusal", NOOP in out, True)
            check(f"no-op advance ({label}): the stages really ran", len(log), 3)
            check(f"no-op advance ({label}): MANIFEST and ELF bytes unchanged",
                  sb.unchanged(), True)


def case_advance_writes_a_computed_seed_chain_row() -> None:
    """A-B3 positive control: the row is derived from the refs and ends in the
    published sha, for one and for two hops, and re-derives from the receipt."""
    for label, hops, stages, tail in (
        ("one hop", 1, (NEW_POINT,) * 3, [sha(NEW_POINT)]),
        ("two hops", 2, (INTERMEDIATE, NEW_POINT, NEW_POINT),
         [f"{sha(INTERMEDIATE)}:intermediate", sha(NEW_POINT)]),
    ):
        with Sandbox() as sb:
            sb.patch(run_elf=stub_loop(stages, []),
                     stage0_emit=lambda c, u: NEW_POINT, so=lambda: StubOracle())
            rc, _ = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=hops)
            rows = receipt_rows(sb.manifest_path.read_text())
            want = " -> ".join([ROOT, sha(OLD_SEED), *tail])
            check(f"chain row ({label}): advance published", rc, 0)
            check(f"chain row ({label}): computed seed_chain row", rows.get("seed_chain"), want)
            check(f"chain row ({label}): ends in the published sha",
                  rows.get("seed_chain", "").endswith(sha(NEW_POINT)), True)
            links, _, problem = walk_seed_chain(rows.get(FROM), tail, [ROOT_HOP])
            check(f"chain row ({label}): the walk re-derives the row",
                  (" -> ".join(links), problem), (want, None))


def case_advance_blocked_when_the_old_seed_has_no_chain() -> None:
    """Unresolved provenance refuses before executing any compiler or oracle."""
    for hops in (1, 2):
        for label, history, diagnostic in (
            ("unchained seed", lambda: [],
             f"{sha(OLD_SEED)} is published by no receipt"),
            ("unreadable history", _blocked, "the clone is shallow"),
        ):
            with Sandbox() as sb:
                log: list[bytes] = []
                oracle_calls: list[str] = []

                def oracle():
                    oracle_calls.append("resolved")
                    return StubOracle()

                sb.patch(_receipt_history=history,
                         run_elf=stub_loop((NEW_POINT,) * 3, log),
                         stage0_emit=lambda c, u: NEW_POINT, so=oracle)
                rc, out = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=hops)
                name = f"{label} ({hops} hop)"
                check(f"{name}: refused with rc 2", rc, 2)
                check(f"{name}: names provenance failure", diagnostic in out, True)
                check(f"{name}: no seed or generated compiler executed", log, [])
                check(f"{name}: no oracle resolved", oracle_calls, [])
                check(f"{name}: fixture untouched", sb.unchanged(), True)


def case_provenance_refusal_precedes_real_seed_execution() -> None:
    """A real harmless seed writes a marker only after its lineage is admitted."""
    for hops in (1, 2):
        with Sandbox() as sb:
            marker = sb.dir / "seed-executed"
            sb.elf_path.write_text(
                "#!/bin/sh\nprintf executed > " + shlex.quote(str(marker)) + "\nexit 3\n"
            )
            sb.elf_path.chmod(0o755)
            sb.before_elf = sb.elf_path.read_bytes()
            seed_receipt = receipt(sb.before_elf, **{FROM: ROOT})
            # The exact same seed bytes serve every history case in this group.
            for history_kind in ("admitted", "absent", "unreadable"):
                marker.unlink(missing_ok=True)
                if history_kind == "admitted":
                    history = lambda: [("marker-seed", seed_receipt)]
                elif history_kind == "absent":
                    history = lambda: []
                else:
                    history = _blocked
                sb.patch(_receipt_history=history)
                # run_elf is deliberately real: exit3 cannot publish or reach an oracle.
                rc, out = captured(loop.do_advance, SRC, IMAGE, USER_LO, hops=hops)
                admitted = history_kind == "admitted"
                name = f"real seed ({hops} hop, {history_kind})"
                check(f"{name}: checked result", rc, 1 if admitted else 2)
                check(f"{name}: execution marker iff admitted", marker.exists(), admitted)
                if admitted:
                    check(f"{name}: exact marker bytes", marker.read_bytes(), b"executed")
                    check(f"{name}: reached actual process refusal",
                          "old frozen seed refused" in out, True)
                check(f"{name}: fixture untouched", sb.unchanged(), True)


def _verify(sb: Sandbox, head: str, history) -> tuple[int, str]:
    sb.elf_path.write_bytes(NEW_POINT)
    sb.manifest_path.write_text(head)
    sb.patch(_receipt_history=history)
    return captured(loop.main, ["--verify-chain"], {})


def case_verify_chain_passes_and_fails_closed() -> None:
    """A-B2: --verify-chain on synthetic receipts, one leg per refusal."""
    good = " -> ".join([ROOT, sha(OLD_SEED), sha(NEW_POINT)])
    head = receipt(NEW_POINT, **{FROM: sha(OLD_SEED), "seed_chain": good})
    hist = lambda: [("head", head), ROOT_HOP]  # noqa: E731
    with Sandbox() as sb:
        rc, out = _verify(sb, head, hist)
        check("verify-chain: intact chain passes", rc, 0)
        check("verify-chain: one PASS [CHAIN] line per link",
              out.count("PASS  [CHAIN] "), 3)
    cases = (
        ("dangling advanced_from", 1, receipt(NEW_POINT, **{FROM: "ab" * 32}), hist),
        ("altered seed_chain link", 1,
         receipt(NEW_POINT, **{FROM: sha(OLD_SEED), "seed_chain": good.replace(
             sha(OLD_SEED), sha(OTHER_LINK))}), hist),
        ("stage1.elf row != the ELF", 1,
         head.replace(f"stage1.elf\t{len(NEW_POINT)}\t", f"stage1.elf\t{len(NEW_POINT) + 1}\t"),
         hist),
        ("one historical receipt", 2, head, lambda: [ROOT_HOP]),
        ("history unreadable", 2, head, _blocked),
    )
    for label, want, text, history in cases:
        with Sandbox() as sb:
            rc, out = _verify(sb, text, history)
            check(f"verify-chain: {label} -> rc {want}", rc, want)
            check(f"verify-chain: {label} prints no chain PASS",
                  "PASS  [CHAIN] " in out, False)


def case_verify_chain_walks_a_two_hop_receipt() -> None:
    """A two-hop receipt in history adds its intermediate as a link: the shape of
    the real root -> intermediate -> seed hop the head receipt descends from."""
    mid = sha(OTHER_LINK)
    two_hop = ("two-hop", receipt(OLD_SEED, **{"advance_hops": "2", FROM: ROOT,
                                             "intermediate_stage1_sha256": mid}))
    head = receipt(NEW_POINT, **{FROM: sha(OLD_SEED)})
    want = " -> ".join([ROOT, f"{mid}:intermediate", sha(OLD_SEED), sha(NEW_POINT)])
    with Sandbox() as sb:
        rc, out = _verify(sb, head, lambda: [("head", head), two_hop])
        check("two-hop history: verify-chain passes", rc, 0)
        check("two-hop history: the derived chain names the intermediate",
              f"derived seed_chain\t{want}\n" in out, True)
        check("two-hop history: one PASS [CHAIN] line per link",
              out.count("PASS  [CHAIN] "), 4)


OTHER_LINK = b"not-a-seed"


def _blocked():
    raise ChainBlocked("the clone is shallow; fetch the full history and tags")


def case_git_history_is_refused_on_a_shallow_clone() -> None:
    """The real git leg: a full clone yields every receipt, --depth 1 is refused."""
    path = "examples/mindc_mind/testdata/selfhost_loop/MANIFEST.txt"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"]
    with tempfile.TemporaryDirectory() as td:
        src, shallow = pathlib.Path(td) / "src", pathlib.Path(td) / "shallow"
        (src / path).parent.mkdir(parents=True)
        subprocess.run([*git, "init", "-q", str(src)], check=True)
        for body in (ROOT_HOP[1], receipt(NEW_POINT, **{FROM: sha(OLD_SEED)})):
            (src / path).write_text(body)
            subprocess.run([*git, "-C", str(src), "add", "-A"], check=True)
            subprocess.run([*git, "-C", str(src), "commit", "-qm", "r"], check=True)
        check("git history: a full clone yields both receipts",
              len(git_receipt_history(src)), 2)
        subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{src}", str(shallow)],
                       check=True)
        try:
            git_receipt_history(shallow)
            refused = ""
        except ChainBlocked as e:
            refused = str(e)
        check("git history: a shallow clone is refused", "shallow" in refused, True)


REPO = pathlib.Path(__file__).resolve().parents[1]
REPO_CHAIN = ["scripts/run_gate.py", "--min-asserted", "3",
              "examples/mindc_mind/self_host_loop_smoke.py", "--verify-chain"]


def case_repository_seed_chain_verifies() -> None:
    """A-B2 wiring: the real --verify-chain on THIS repository's refs, under the
    gate runner with floor 3. Nested here, never a second CI argv of the
    compiler-dependent loop smoke (that argv would pass the vacuity sweep with no
    compiler and count as loop-gate wiring while running neither leg). Needs full
    history and tags: a shallow clone BLOCKs (rc 2) and fails this case."""
    proc = subprocess.run([sys.executable, *REPO_CHAIN], cwd=REPO, capture_output=True,
                          text=True, timeout=600, check=False)
    lines = (proc.stdout + proc.stderr).splitlines()
    why = next((ln.strip() for ln in lines if "BLOCKED" in ln or "FAIL" in ln), "")
    links = [ln.split()[2] for ln in lines if ln.strip().startswith("PASS  [CHAIN] ")]
    _, published, _ = parse_receipt(
        (REPO / "examples/mindc_mind/testdata/selfhost_loop/MANIFEST.txt").read_text())
    check("repository chain: run_gate --min-asserted 3 --verify-chain passes",
          (proc.returncode, why), (0, ""))
    check("repository chain: at least 3 links", len(links) >= 3, True)
    check("repository chain: first link is the chain root", links[:1], [ROOT])
    check("repository chain: last link is the committed stage1.elf sha",
          links[-1:], [published])


def main() -> int:
    loop._receipt_history = lambda: [ROOT_HOP]
    cases = [value for name, value in globals().items()
             if name.startswith("case_") and callable(value)]
    for module in (contract, publication):
        cases.extend(
            value for name, value in vars(module).items()
            if name.startswith("case_") and callable(value)
        )
    for fn in sorted(cases, key=lambda function: function.__name__):
        fn()
    if FAILURES:
        print()
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print()
    print("RESULT: PASS — --advance publishes only on full agreement, "
          "--advance --hops 2 publishes stage2 only once stage2 == stage3 and "
          "the oracle agrees with it, a no-op advance is refused, every receipt "
          "carries a seed_chain row derived from the refs, --verify-chain fails "
          "closed on a dangling, altered or shallow chain and passes on this "
          "repository's refs, --reseed records the "
          "broken seed chain, every ordinary refusal preserves the fixture, failed "
          "rollback is reported with recoverable copies, and ordinary verification "
          "is unweakened. (Harness controls: synthetic images, stub oracle — not "
          "compiler evidence.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
