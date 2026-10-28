"""Seed-provenance support for the LOOP harness: the legacy Rust-seeded
re-freeze, and the seed chain every receipt is checked against.

The public entrypoint stays in self_host_loop_smoke.py; this support module
keeps its publication and provenance paths focused and independently readable
(and the entrypoint under the 800-line ceiling).
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import tempfile


def manifest_text(stage1: bytes, *, published_by: str, receipt: dict[str, str]) -> str:
    """The MANIFEST.txt body: a human header, the provenance receipt, one row.

    The receipt is the reason the two-file publication is auditable at all. It
    records the sha256 of the ELF this manifest was written FOR, so a pair torn
    apart by a crash between the two replacements is detectable by comparing
    the `stage1.elf` row (the fixture's file name, whatever stage produced its
    bytes) against the file on disk — which is the honest answer to "these two
    paths cannot be replaced atomically together".

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


# ── seed chain (A1-B2/B3): every link is read back from a receipt on a ref ──
#: Root of the pure-MIND seed lineage, ruled in FABLE_DECISIONS A1 (private
#: ruling). Every computed `seed_chain` row starts here.
SEED_CHAIN_ROOT = "abb0622b558bf5e482cc538ca8adc852ac6e6f211fdf24232b90ef75ec6b2d00"
#: Receipt history BEFORE the root, named in the A1 review. --verify-chain
#: prints each as a measured PRE-ROOT line, counted apart; never a chain link.
PRE_ROOT_HISTORY = (
    ("3cdbc1a8b8001f035213ac6bf4f62e07691fb21a", "advances from a seed no receipt publishes"),
    ("f173f00bc8b92bd5521033ed4213de6e07a6bfec", "advances from the seed it publishes"),
)
_RECEIPT = "examples/mindc_mind/testdata/selfhost_loop/MANIFEST.txt"  # repo-relative
_FROM = "advanced_from_seed_sha256"


class ChainBlocked(RuntimeError):
    """The receipt history cannot be read in full, so no chain verdict exists."""


def parse_receipt(text: str) -> tuple[dict[str, str], str | None, str | None]:
    """(receipt rows, published sha256, published size) of a MANIFEST body."""
    rows: dict[str, str] = {}
    sha = size = None
    for line in text.splitlines():
        if line.startswith("# ") and "\t" in line:
            key, value = line[2:].split("\t", 1)
            rows[key] = value
        elif line.startswith("stage1.elf\t") and line.count("\t") == 2:
            _, size, sha = line.split("\t")
    return rows, sha, size


def git_receipt_history(repo: pathlib.Path) -> list[tuple[str, str]]:
    """Every (commit, MANIFEST body) on ANY ref incl. tags, newest first.

    All refs, because a squash landing leaves the intermediate receipts on the
    evidence tag only. A shallow clone would cut the chain short, so it is
    refused rather than walked.
    """
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=300)

    if shutil.which("git") is None:
        raise ChainBlocked("git is not available")
    shallow = git("rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0:
        raise ChainBlocked(f"not a git checkout ({shallow.stderr.strip()[:160]})")
    if shallow.stdout.strip() != "false":
        raise ChainBlocked("the clone is shallow; fetch the full history and tags")
    log = git("log", "--all", "--format=%H", "--", _RECEIPT)
    if log.returncode != 0:
        raise ChainBlocked(f"git log exited {log.returncode}")
    bodies = [(c, git("show", f"{c}:{_RECEIPT}")) for c in log.stdout.split()]
    return [(c, r.stdout) for c, r in bodies if r.returncode == 0]  # rc!=0: a deletion


def _links(rows: dict[str, str], sha: str) -> list[str]:
    """The links one receipt adds: a two-hop intermediate, then its seed."""
    if rows.get("advance_hops") == "2":
        return [f"{rows.get('intermediate_stage1_sha256')}:intermediate", sha]
    return [sha]


def walk_seed_chain(frm: str | None, tail: list[str], history: list[tuple[str, str]]
                    ) -> tuple[list[str], list[str], str | None]:
    """Follow `advanced_from_seed_sha256` from `frm` back to SEED_CHAIN_ROOT.

    Returns (links ending in `tail`, receipt commits walked oldest first,
    problem or None). A seed that no receipt on any ref publishes is returned
    as a `<sha>:dangling` link; a self-advance never resolves a link.
    """
    index: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for commit, text in history:
        rows, sha, _ = parse_receipt(text)
        if sha and rows.get(_FROM) not in (None, sha):
            index.setdefault(sha, []).append((commit, rows))
    links, commits, seen = list(tail), [], set(tail)
    while frm != SEED_CHAIN_ROOT:
        if frm is None:
            return links, commits, f"a receipt in the chain names no {_FROM}"
        if frm in seen:
            return links, commits, f"the chain loops back to {frm}"
        found = index.get(frm, [])
        if not found:
            return [f"{frm}:dangling", *links], commits, \
                f"{frm} is published by no receipt on any ref (dangling)"
        if len({(r.get(_FROM), r.get("advance_hops"), r.get("intermediate_stage1_sha256"))
                for _, r in found}) > 1:
            return links, commits, f"{len(found)} receipts publish {frm} with different provenance"
        commit, rows = found[-1]
        if rows.get("advance_hops") == "2" and not rows.get("intermediate_stage1_sha256"):
            return links, commits, f"the two-hop receipt {commit} names no intermediate"
        links[:0] = _links(rows, frm)
        commits.insert(0, commit)
        seen.add(frm)
        frm = rows.get(_FROM)
    return [SEED_CHAIN_ROOT, *links], commits, None


def old_seed_chain(harness, h_old: str) -> tuple[list[str], str | None]:
    """(chain ending at the old seed, problem or None) for an --advance receipt.

    The committed MANIFEST is walked as one receipt beside every historical one,
    so the row a new receipt carries is derived from the refs, never typed.
    """
    try:
        path = harness._FROZEN_MANIFEST
        live = [("(committed MANIFEST)", path.read_text())] if path.exists() else []
        links, _, problem = walk_seed_chain(h_old, [], live + harness._receipt_history())
    except ChainBlocked as e:
        return [], str(e)
    return links, problem


def verify_chain(harness) -> int:
    """--verify-chain: re-derive the seed chain from git alone (no build) and
    check it against the committed fixture. Contract: self_host_loop_smoke.md."""
    _FROZEN, _FROZEN_MANIFEST = harness._FROZEN, harness._FROZEN_MANIFEST
    try:
        history = harness._receipt_history()
    except ChainBlocked as e:
        print(f"BLOCKED: --verify-chain cannot read the receipt history: {e}. "
              f"Nothing verified.")
        return 2
    receipts = [c for c, text in history if _FROM in parse_receipt(text)[0]]
    if len(receipts) < 2:
        print(f"BLOCKED: --verify-chain found {len(receipts)} historical receipt(s) on "
              f"the refs; a chain needs at least 2. Nothing verified.")
        return 2
    published = {parse_receipt(text)[1] for _, text in history}
    bodies = dict(history)
    for commit, why in PRE_ROOT_HISTORY:
        if commit not in bodies:
            print(f"  PRE-ROOT {commit} {why}: commit not on any ref here")
            continue
        rows, sha, _ = parse_receipt(bodies[commit])
        frm = rows.get(_FROM)
        kind = ("self-advance" if frm == sha else
                "resolves" if frm in published else "dangling: no receipt publishes it")
        print(f"  PRE-ROOT {commit} {why}: publishes {sha} from {frm} ({kind})")
    print(f"  pre-root receipts listed: {len(PRE_ROOT_HISTORY)} (history before "
          f"{SEED_CHAIN_ROOT[:8]}, not chain links)")
    rows, sha, size = parse_receipt(_FROZEN_MANIFEST.read_text())
    if sha is None:
        print(f"  FAIL  [CHAIN] {_FROZEN_MANIFEST} has no stage1.elf row")
        return 1
    if rows.get("advance_hops") == "2" and not rows.get("intermediate_stage1_sha256"):
        print("  FAIL  [CHAIN] the head receipt is two-hop but names no intermediate")
        return 1
    links, commits, problem = walk_seed_chain(rows.get(_FROM), _links(rows, sha), history)
    derived = " -> ".join(links)
    print(f"  derived seed_chain\t{derived}\n  receipts walked: {' '.join(commits)} "
          f"+ the committed MANIFEST")
    if problem is not None:
        print(f"  FAIL  [CHAIN] {problem}")
        return 1
    elf = _FROZEN.read_bytes() if _FROZEN.exists() else b""
    h_elf = hashlib.sha256(elf).hexdigest()
    if (h_elf, str(len(elf))) != (sha, size):
        print(f"  FAIL  [MANIFEST] stage1.elf is {len(elf)}B sha256={h_elf}; the committed "
              f"row says {size}B sha256={sha}")
        return 1
    claimed = rows.get("seed_chain")
    if claimed is not None and claimed != derived:
        print(f"  FAIL  [CHAIN-ROW] the head receipt's seed_chain row\n        {claimed}\n"
              f"        != the chain derived from the refs\n        {derived}")
        return 1
    for i, link in enumerate(links):
        nxt = links[i + 1] if i + 1 < len(links) else f"stage1.elf ({size}B)"
        print(f"  PASS  [CHAIN] {link} -> {nxt}")
    print(f"  PASS  [MANIFEST] stage1.elf sha256 and size equal the committed row ({sha})")
    print("  PASS  [CHAIN-ROW] " + ("the head receipt's seed_chain row is byte-equal to "
          "the derived chain" if claimed is not None else "the head receipt has no "
          "seed_chain row; the derived chain above is the asserted one"))
    return 0


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
