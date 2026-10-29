# Self-host LOOP gate contract

The default source bytes are now captured through the internal
[compiler source bundle](selfhost_source_bundle.md). With no extra components,
the library, native source and stdin image remain byte-identical to the
historical construction. Explicit component bundles have a separate captured
CPU-oracle API; they do not alter the normal seed/advance/reseed contract.

Self-host LOOP gate (Rust-independence #14, roadmap A7 + RI-E1) — the PERMANENT proof
that MIND reproduces its own compiler with Rust+LLVM out of the loop, for the scalar-i64
subset that the pure-MIND compiler (main.mind) is written in.

RI-E1 — reproduction-independence via a checked-in pure-MIND bootstrap stage0
-----------------------------------------------------------------------------
This is the standard GCC/rustc "checked-in stage0" bootstrap model: the frozen
`testdata/selfhost_loop/stage1.elf` IS the bootstrap compiler, and the Rust `.so`
is DEMOTED from seed to a re-freeze / drift oracle.

  * PRIMARY (always, no Rust in the chain): seed stage1 by RUNNING the frozen
    pure-MIND ELF on the seeded stdin — `stage1 = run_elf(FROZEN, stdin)` — then
    stage2 = run_elf(stage1), stage3 = run_elf(stage2). Assert
    stage1 == stage2 == stage3 == frozen. The ONLY syscalls in this chain are
    execve(self)/read/write/exit — zero rustc, zero LLVM, zero clang, zero .so.
  * ORACLE (drift check, when the Rust `.so` is present): assert the FRESH `.so`
    output stage0_emit(combined, user_lo) == frozen. This catches std/*.mind or
    main.mind SOURCE drift where the frozen ELF was not re-blessed. The primary
    reproduction path never depends on the `.so` being buildable, but an
    UNBUILDABLE `.so` is not a pass: the leg announces a `SKIP` line and exits
    non-zero (see oracle_unavailable), because a green gate with its drift half
    silently dropped is the vacuous pass this corpus exists to refuse.
  * ADVANCE (--advance / MIND_SELFHOST_ADVANCE=1): the deliberate path for a
    source change. Seeds stage1 by running the EXISTING frozen pure-MIND ELF on
    the CURRENT combined source, derives stage2/stage3 from it, and requires
    stage1 == stage2 == stage3 — a NEW fixed point, which unlike ordinary
    verification is ALLOWED to differ from the old frozen ELF. It additionally
    requires a present, FRESH, current-source Rust `.so` oracle emitting those
    same bytes. Only when every one of those checks passes does it publish
    testdata/selfhost_loop/{stage1.elf,MANIFEST.txt}. The old compiler is what
    compiles the new source, so the trust chain is continuous.
  * TWO-HOP ADVANCE (--advance --hops 2): the bootstrap policy's answer to a
    source change that alters CODEGEN (docs/INDEPENDENCE_ROADMAP.md, "Bootstrap
    seed-advance policy"). stage1 = old seed on the current source still carries
    the OLD codegen, so stage1 != stage2 is expected and is NOT refused; stage1
    is the unpublished intermediate. stage2 is published, and only when
    stage2 == stage3 AND the fresh oracle emits exactly stage2. Every stage still
    descends from the old pure-MIND seed. The receipt records advance_hops 2,
    intermediate_stage1_sha256, published_stage stage2 and the computed
    seed_chain row (SEED CHAIN below), and carries no stage1_sha256 row (that key means "published hash" in a
    one-hop receipt). --hops accepts 1 or 2 only: stage1 and stage2 compile the
    same source, so stage2 != stage3 is a miscompile or nondeterminism, never a
    reason for a third hop.
  * RESEED (--reseed / MIND_SELFHOST_RESEED=1) — LEGACY, Rust-seeded: the ONLY
    mode that uses the `.so` as the SEED. It mints the frozen bootstrap from
    Rust output rather than from the previous pure-MIND compiler, so it breaks
    the pure-MIND seed chain for that hop. Prefer --advance for ordinary source
    drift; keep --reseed only for a bootstrap that genuinely cannot be advanced
    (e.g. the old seed cannot compile the new source at all). Under the bootstrap
    policy it is a DISCLOSED, REVIEWED exception: the change that commits it says
    so, and its receipt's seed_chain row records "the pure-MIND seed chain is
    broken for that hop".
  * --advance and --reseed are MUTUALLY EXCLUSIVE, in flag and environment form
    alike, and the conflict is refused before anything is read, run or written.
  * VERIFY-CHAIN (--verify-chain): read-only and build-free; see SEED CHAIN.

SEED CHAIN (the receipt row and its offline verifier):
  Every --advance receipt (one or two hops) ends with a COMPUTED row
      # seed_chain<TAB><link> -> <link> -> ...
  where a link is `<sha256>` or `<sha256>:intermediate` (a two-hop stage1 that
  was never published). The first link is the chain root SEED_CHAIN_ROOT
  (abb0622b558bf5e482cc538ca8adc852ac6e6f211fdf24232b90ef75ec6b2d00), the last
  is the published sha. The row is never typed: one function walks
  `git log --all -- testdata/selfhost_loop/MANIFEST.txt` (every ref and tag; a
  squash landing leaves intermediate receipts on the evidence tag only), reads
  each receipt with `git show`, and follows advanced_from_seed_sha256 (plus
  intermediate_stage1_sha256 for two-hop receipts) back to the root. A seed no
  receipt on any ref publishes is a `<sha>:dangling` link. --advance derives the
  chain up to the old seed BEFORE running anything and refuses (BLOCKED exit 2)
  when it does not resolve. --verify-chain re-derives the chain with the same
  function and asserts: (i) every link resolves; (ii) sha256 and size of
  stage1.elf equal the committed MANIFEST row; (iii) a seed_chain row in the head
  receipt is byte-equal to the derived chain; (iv) without such a row (receipts
  older than this contract) the derived chain is printed and asserted. It prints
  one `PASS  [CHAIN] <link> -> <next>` line per link. History before the root
  (the receipts named in PRE_ROOT_HISTORY) is printed as measured PRE-ROOT lines,
  counted apart, never a chain link and never a PASS. CI runs it with full
  history and tags (`fetch-depth: 0`) under --min-asserted 3, from one case of
  tests/self_host_loop_advance_test.py (the "Bootstrap advancement harness
  controls" step), never as a second CI argv of this smoke: this smoke is
  compiler-dependent, so a build-free argv of it would pass the vacuity sweep
  with no compiler and count as LOOP-gate wiring while running neither leg.

HONEST FRAMING (no overclaim):
  This proves REPRODUCTION-independence — the seed chain is now Rust-free and the
  `.so` is only an oracle. It does NOT claim "mindc builds from scratch with zero
  Rust" nor "LLVM dropped". Two residuals remain, orthogonal and stated plainly:
    (i)  the FIRST frozen stage1.elf was originally minted by the Rust `.so`
         (chicken-and-egg; residual trusting-trust, universal to every bootstrapped
         toolchain — gcc/rustc included);
    (ii) a HARNESS-FREE standalone mindc (its own file-IO/argv/CLI, no Python
         driver) is NOT delivered here — that is the separate C8 + argv/CLI track.

The seeded source is  [8B user_lo LE][8B src_len LE][ 21 std/*.mind ++ main.mind ++
selfhost_driver.mind ]  on stdin (fd 0); the ELF is written to stdout (fd 1). main.mind
is NOT modified — the driver is a separate appended shim, so the mic@1 fixed-point and
mic@3-flip gates are untouched.

FAIL-CLOSED (never skips when asked to run):
  * frozen bootstrap fixture missing                 -> BLOCKED exit 2  (it is the seed/oracle now)
  * running the frozen ELF exits non-zero / emits nothing -> FAIL exit 1
  * stage1 != stage2 or stage2 != stage3             -> FAIL exit 1
  * stage1 (from frozen) != frozen fixture           -> FAIL exit 1  (should be impossible;
        run_elf(frozen) reproduces frozen by construction)
  * .so present AND fresh .so output != frozen        -> FAIL exit 1  (source drifted;
        advance in the same change with --advance, or --advance --hops 2 when
        the change alters codegen; never a silent --reseed)
  * .so unavailable (drift oracle cannot run)        -> SKIP line + exit 1, unless
        MIND_SELFHOST_LOOP_ORACLE_DEFERRED=1 records a deliberate one-leg run
        (which still reports asserted=1, so a --min-asserted 2 caller refuses it)
  --advance and --reseed requested together     -> BLOCKED exit 2  (before any read/run/write)
  --hops without --advance, with --reseed, repeated, without a value, or not
        1 or 2                                     -> BLOCKED exit 2  (before any read/run/write)
  --advance only:
  * frozen bootstrap seed missing                     -> BLOCKED exit 2  (nothing to advance FROM)
  * old seed refuses the current source               -> FAIL exit 1     (nothing written)
  * stage1 != stage2 or stage2 != stage3              -> FAIL exit 1     (nothing written;
        when stage2 == stage3 the refusal names --advance --hops 2)
  * a stage is not a static ELF                       -> FAIL exit 1     (nothing written)
  * oracle .so missing / not a fresh build            -> BLOCKED exit 2  (nothing written)
  * oracle output != the new fixed point              -> FAIL exit 1     (nothing written)
  * the fixed point IS the current seed (no-op)       -> BLOCKED exit 2  (nothing written)
  * the old seed's chain does not resolve (dangling,
        shallow clone, no git)                        -> BLOCKED exit 2  (before the oracle; nothing written)
  --advance --hops 2: the refusals above, except stage1 != stage2, plus
  * stage2 != stage3                                  -> FAIL exit 1     (nothing written)
  * oracle output != stage2 (even if == stage1)       -> FAIL exit 1     (nothing written)
  * publication fails part-way                        -> FAIL exit 1     (old files rolled back)
  The oracle leg of --advance is NOT deferrable: MIND_SELFHOST_LOOP_ORACLE_DEFERRED
  records a deliberate one-leg VERIFICATION run and has no effect on publication.
  --verify-chain only:
  * git unavailable, shallow clone, or fewer than 2
        historical receipts on the refs               -> BLOCKED exit 2  (never a pass)
  * a dangling link, a looping or ambiguous chain      -> FAIL exit 1
  * stage1.elf sha256/size != the committed row        -> FAIL exit 1
  * head seed_chain row != the derived chain           -> FAIL exit 1
  * combined with --advance, --reseed or --hops        -> BLOCKED exit 2
  --reseed only (LEGACY):
  * MINDC_SO unset/missing                            -> BLOCKED exit 2  (needs the seed .so)
  * .so emits an empty / non-ELF image                -> FAIL exit 1

Run:
  python3 examples/mindc_mind/self_host_loop_smoke.py                       # PRIMARY + oracle(if .so)
  MINDC_SO=/path/to/libmindc_mind.so python3 .../self_host_loop_smoke.py    # + .so drift oracle
  MINDC_SO=/path/to/libmindc_mind.so python3 .../self_host_loop_smoke.py --advance  # advance the seed
  MINDC_SO=/path/to/libmindc_mind.so python3 .../self_host_loop_smoke.py --advance --hops 2  # codegen change
  MINDC_SO=/path/to/libmindc_mind.so python3 .../self_host_loop_smoke.py --reseed   # LEGACY re-mint
  python3 .../self_host_loop_smoke.py --verify-chain   # offline seed-chain check (full clone)
