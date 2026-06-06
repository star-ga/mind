# MIND-Fuzz

An LLM-mutation differential-testing harness for the MIND compiler.

MIND-Fuzz **drives** `mindc` (it never modifies any compiler source under
`src/`). It implements the technique from recent research on LLM-assisted
compiler testing (arXiv:2501.00655, *Finding Missed Code Size Optimizations in
Compilers using LLMs*, §2): start from a small **seed** program, hand an
off-the-shelf LLM a predetermined list of **mutation instructions**, let it
incrementally rewrite the program, and run a **differential test at every
mutation step**. Iterate until the program no longer compiles or an oracle
trips a violation; violations are saved and minimized.

## The MIND-native insight

The paper's targets (C/Rust/Swift) use *code size* as the differential signal.
MIND has a sharper oracle no other compiler has: **cross-substrate
byte-identity**. The same source must compile to a byte-identical artifact on
x86 (avx2) and ARM (neon). That is an exact, free, flake-proof differential
oracle. This first slice runs the LOCAL proxies for that wedge and leaves a
clear seam for the real cross-substrate check (which needs a second substrate /
CI).

## Oracles

| # | Oracle | Status | What a failure means |
|---|--------|--------|----------------------|
| 0 | **cross-substrate byte-identity** | **STUBBED (seam)** | avx2 artifact hash != neon artifact hash = wedge-breaking miscompile. Needs an aarch64 runner; see `cross_substrate_hook`. |
| 1 | **determinism** (local wedge proxy) | LIVE | Compile the SAME program twice; `mic@3` + `.so` + MLIR text hashes must match. A mismatch = non-determinism in codegen. |
| 2 | **reference oracle** | LIVE | For the Q16.16 `__mind_blas_*` dot kernels, call the compiled symbol via `ctypes` and compare to a scalar integer reference (the same LCG + scalar oracle as `tests/cross_substrate_identity.rs`). Mismatch = vector path diverged from scalar. |
| 3 | **mindc verify** | LIVE | Emit a mic@3 evidence artifact, run `mindc verify` (RFC 0017 SSA + RFC 0021 trace_hash). Non-zero exit / `trace_hash_valid:false` = IR / evidence bug. |
| 4 | **mic@3 round-trip** | LIVE | Emit mic@3 twice; require a byte fixed point. Divergence = codec / serialization bug. |
| 5 | **compile-success** | LIVE | A program that should compile but errors or crashes `mindc` is a bug (and is the loop's stop condition per the paper). |

For a `[semantics]` mutation (one allowed to change what the program computes)
the **reference** oracle is advisory and skipped; the determinism / verify /
mic@3 / compile oracles never depend on the computed value, so they assert
regardless. `[preserve]` mutations must keep observable semantics and the
reference oracle asserts in full.

## Files

```
tools/mindfuzz/
  seeds/dot_q16.mind       Q16.16 dot/L1/gemv kernels (lifted from
                           tests/cross_substrate_identity.rs) -- reference-checkable
  seeds/scalar_arith.mind  trivial integer fn (paper's Listing 1 shape) -- SSA stress
  mutations.txt            predetermined mutation instructions (paper Table 1,
                           MIND-adapted: no pointers/unions/structs)
  mutate.py                LLM mutation engine (claude CLI) + deterministic
                           template fallback
  oracles.py               the five local oracles + the cross-substrate seam
  fuzz_loop.py             the loop: seed -> mutate(counter) -> compile -> oracles
                           -> save + minimize on violation
  violations/              saved + minimized violations land here
```

## Determinism of the loop

The loop is replay-stable (autorun.py-style): the mutation instruction at step
*i* is `instructions[i % N]` chosen by a **counter, never random**, and no
clock / RNG sits in any decision path. The only nondeterminism is the LLM
itself; `--no-llm` forces the template mutator for a fully reproducible run.

## How to run

Build `mindc` first (the harness uses `target/release/mindc`):

```bash
cargo build --release --no-default-features \
  --features mlir-build,std-surface,cross-module-imports --bin mindc
```

Then, from `tools/mindfuzz/`:

```bash
# real LLM-mutation run (uses the local `claude` CLI)
python3 fuzz_loop.py --seed seeds/dot_q16.mind --iters 6

# fully deterministic run (template mutator, no LLM)
python3 fuzz_loop.py --seed seeds/scalar_arith.mind --iters 8 --no-llm

# SANITY: prove the oracles actually catch a deliberately-broken program
python3 fuzz_loop.py --inject-fault
```

The LLM step calls `env -u ANTHROPIC_API_KEY claude -p "<prompt>"`. If the LLM
call fails, times out, or returns something that does not look like a MIND
program, the loop falls back to the deterministic template mutator (recorded as
`[template-fallback]` in the step log) so the loop is always demonstrable.

## Cross-substrate CI / cluster seam

`oracles.cross_substrate_hook` is the seam for oracle 0 (the wedge). On a single
host it returns a `DEFERRED` verdict (never a false violation). The real check,
wired in CI / the cluster:

1. this host emits `--emit-mic3 host.mic3` and publishes `sha256(host.mic3)`;
2. an `ubuntu-24.04-arm` (neon) runner compiles the SAME mutated program and
   computes `sha256(neon.mic3)`;
3. assert `sha256(neon.mic3) == sha256(host.mic3)` (RFC 0015 §3.1). A mismatch
   is a wedge-breaking miscompile -- the highest-severity violation MIND-Fuzz
   can find.

This reuses the existing `cross_substrate_identity` CI job's two-runner shape.
