# Frontier Scientific Computation — proposed roadmap extension

Status: PROPOSED / NOT IMPLEMENTED. This roadmap addition does not alter release gates, accepted compiler pins, scientific statuses, or production claims.

## Objective

Build progressively qualified, long-horizon autonomous scientific computation capable of reproducing difficult published symbolic results with independent checks, strict provenance and measured resources. The public report about Claude's nine-loop scattering-amplitude calculation is motivation, **not** evidence that STARGA reproduces it or that the original result is independently verified here.

## Exact symbolic computation compiler profile

Investigate arbitrary-precision integer/rational interfaces, canonical symbolic DAG representation, bounded expression simplification, deterministic rewrite evaluation, C ABI/stdlib support, streaming output, compile-time/memory profiles and safe resource refusal. Test native/MLIR/evaluator parity and reproducible codegen. Benchmark against matched baselines. Keep full-language native independence and current release gates primary.

## Acceptance and sequencing

1. Preserve existing release-blocking obligations and repository ownership. Do not divert current compiler/native-app qualification.
2. Freeze a small, independently known symbolic benchmark and exact reference before implementation.
3. Require positive, negative, mutated and interrupted/restarted execution cases, including false-positive rejection.
4. Require independent checking via a distinct implementation or mathematically checkable certificate; solver output alone is not proof.
5. Record exact source/toolchain/dependency SHAs, inputs, assumptions, outputs, CPU/memory/wall-time/cost and immutable provenance.
6. Report measured baseline comparisons and limitations. Scale complexity only after earlier tiers pass.
7. Keep this track separate from RH scientific research; RH remains paused and no RH status changes are authorized.

## Coordination

MIND-Lab owns scientific campaign admission; MIND compiler owns language/backend; MIND-CI owns execution qualification; MIND-Parser owns extraction; MIND-Mem owns persistent derivation context; NAESTRO owns orchestration; independent verification remains separate from agent consensus. Do not duplicate the authoritative keep/discard or proof decision.
