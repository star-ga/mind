# Roadmap

This roadmap outlines upcoming milestones for the MIND language, runtime, and tooling.

## Shipped (v0.10.0)

- ✅ **Enum tagged-union payloads construct, match, and RUN** (v0.10.0) – an
  enum with a payload variant is a uniform `[tag, …fields]` heap record;
  `match` reads the tag and binds each field positionally. Single- and
  multi-field **i64** payloads (`Tri::T(a, b, c)`) and **f64** payloads
  (`Option<f64>`, `Result<f64>`, mixed `(i64, f64)`) all lower to running code,
  verified end-to-end (`.so` + dlopen). f64 fields round-trip through the i64
  slot as raw bits (`arith.bitcast`), bit-exactly. The boxed layout fixed an
  Option/Result-shaped-enum **segfault** (fieldless and payload variants now
  share one layout). Unsupported shapes (non-i64/f64 fields, nested sub-patterns)
  fail **loud** on the emit path — never a silent miscompile. Keystone 7/7 +
  cross-substrate 8/8 stay byte-identical throughout.
- ✅ **Integer-determinism cluster** (v0.10.0) – `INT_MIN / -1`, division by
  zero (`x / 0 = 0`), oversized shifts, and condition truthiness (`if c` tests
  non-zero, not the low bit) all lower deterministically; the narrow-int call
  ABI (i32/u32 across call boundaries) and struct narrow-field ABI are sound.
- ✅ **Tensor operations tier 1: in-function + return** (v0.10.0) – dense tensor
  literals (`[1.0, 2.0, 3.0]`), elementwise ops, and tensor-returning functions
  now compile to ELF via the bufferization preset (commits 9f807bd, cb43c90).
  **Scope note:** inter-function tensor-arg calls and deterministic intrinsics
  (`zeros`, `ones`, `matmul`, `softmax`, `randn`, `transpose`) follow in Phase 11.
  f32 tensor results are reproducible within a single substrate (x86 or ARM);
  int/Q16 results are byte-identical across substrates (cross_substrate gate 12/12).
- ✅ **Native-ELF self-host fixed-point closed** (v0.10.0) – the pure-MIND
  front-end now emits the NATIVE x86-64/ELF of the entire seeded module (21 stdlib
  modules + main.mind, 1 055 777 B) byte-identically against the Rust reference —
  all three self-host gates pass: mic@1 IR-text bootstrap fixed point, mic@3
  canonical-binary-IR flip, and the native-ELF fixed point. This is the core of
  Rust-independence.
- ✅ **Self-computed PT\_NOTE trace-hash + `src/native` deleted** (2026-07-01) –
  the pure-MIND front-end self-computes the `ir_trace_hash` PT\_NOTE byte-identically
  at full main.mind scale (~1.5 MB combined stdlib+main.mind source), zero Rust
  oracle bytes fed anywhere in the loop (the no-feed rung in
  `self_host_native_elf_smoke.py`). This closed the last oracle tie in the
  native-ELF self-host (#17), which made the Rust `src/native` backend (2441
  lines) fully redundant — deleted (Phase 1.4 / #15), after freezing its current
  output as permanent test fixtures
  (`examples/mindc_mind/testdata/native_elf_oracle/`). Full Rust-independence
  for the native-ELF path is complete.
- ✅ **Scalar IEEE-754 float64 on the strict deterministic path** (2026-07-02) –
  scalar `f64` (and `f32`) arithmetic now compiles and runs through
  `arith.mulf` / `arith.addf` with **no `fmuladd`-contraction, no `fastmath`
  flag, and no reassociation**, in fixed source order. A loop-carried `f64`
  computation — an `f64` Lorenz–Euler integrator — runs **bit-exact against a
  reference** and is **run-to-run bit-identical**. Because scalar `+ − × ÷ √`
  are correctly-rounded IEEE-754 operations (identical on x86-SSE2 and
  ARM-NEON), cross-ISA bit-identity follows on any conforming FPU; this is now
  verified byte-identical on real ARM64 (NEON) hardware (2026-07-05), where the
  `cross_substrate` gate's scalar-`f64` canaries reproduced the x86_64 references
  byte-for-byte. The vector-reduction, transcendental, and GPU float tiers remain on
  the roadmap (see Phase 11 / Phase 13.6).

## Shipped (v0.7.1)

- ✅ **Cross-substrate Q16.16 bit-identity** – identical Q16.16 source compiles
  to byte-identical artifacts across x86 and ARM (keystone gate). This is the
  proven wedge; no vendor toolchain offers it.
- ✅ **Emitted evidence chain** – each artifact carries an embedded evidence
  chain, `trace_hash = SHA-256` of the canonical `mic@3` bytes. Cryptographic
  Ed25519 *signing* of the chain is the next milestone (see Roadmap), so the
  chain is emitted/embedded but not yet signed.
- ✅ **Self-host fixed-point — canonical binary IR (`mic@3`)** – the pure-MIND
  `mindc` front-end reproduces the **canonical `mic@3` binary IR** of its own
  ~15k-line source **byte-for-byte** against the Rust reference
  (`selftest_mic3_module_nfn(main.mind) == mindc --emit-mic3 main.mind`). This
  goes beyond the earlier `mic@1` MLIR-text fixed-point: the Rust front-end is now
  decorative at the canonical-binary-IR layer (the layer the evidence chain's
  `trace_hash` anchors on). CI-enforced by `mic3_flip_smoke.py`.
- ✅ **Autodiff Engine** – Reverse-mode AD for the Core v1 tensor ops, single-output `main` entry point; non-Core-v1 ops error rather than emit a silent zero gradient.
- ✅ **FFI Stabilization** – ABI frozen, C header generation.
- ✅ **Phase 10.5: Governance Logic** – enum, struct, if/else, while,
  const, bitwise/boolean ops, struct literals, enum-variant `::`
  access shipped in v0.2.5–v0.2.10.
- ✅ **Docs Migration** – Documentation published to `mindlang.dev/docs/*`.
- ✅ **Link-Check Automation** – CI enforces documentation link health.

## Roadmap

- ✅ **Full Rust-independent native-ELF pipeline** (2026-07-01) – the native-ELF
  self-host fixed point is closed and the pure-MIND front-end self-computes its
  own `ir_trace_hash` PT\_NOTE byte-identically at full scale; the Rust
  `src/native` backend (Phase 15) has been deleted. See the README for what
  remains on the broader Rust-independence effort (MLIR/mindc itself is still
  Rust — this closes the native-ELF track specifically).
- **Ed25519-signed evidence chain** – cryptographic signing of the
  already-emitted evidence chain.
- **GPU / accelerator backends** – the open-source `mindc` compiler in this repo
  emits for the **CPU**. GPU and accelerator execution (CUDA, Metal, ROCm,
  WebGPU, and the broader chip-target set) ships today in the commercial
  `mind-runtime`, available to consumers under a commercial license. What is on
  the roadmap is **bit-identical determinism** across those substrates.
- **Deterministic distributed runtime** – the commercial runtime ships NCCL/Gloo
  collectives, RingAllReduce, and pipeline parallelism today; making them
  bit-identical by fixed reduction order is the active work (see Phase 13.6).
- **Deployment & Serving** – HTTP/gRPC inference, dynamic batching, metrics.
- **Package Manager** – PubGrub resolver, SLSA provenance, SBOM, sparse registry.

## In Progress

- 🚧 **Phase 11: Complete Tensor System** – inter-function tensor-arg calls
  (cross-function tensor ABI), deterministic intrinsics (`zeros`, `ones`,
  `matmul`, `softmax`, `randn`, `transpose`), and full cross-substrate bit-identity
  for int/Q16/f32 tensor results. The float-determinism tiers, stated explicitly:
  - **Scalar `f64`/`f32`** — ✅ done. Runs on the strict path, run-to-run
    bit-identical (fixed source order, no FMA-contraction, no reassociation);
    verified byte-identical across x86_64 (AVX2) + ARM64 (NEON) on real hardware
    by the `cross_substrate` gate.
  - **Vector `f32`/`f64` reductions** — 🚧 roadmap. A documented ~1e-4 relative
    tolerance today, not bit-identity → canonical reduction trees /
    superaccumulators (see Phase 13.6).
  - **Transcendentals** (`sin`, `exp`, …) — 🚧 roadmap → a vendored
    correctly-rounded libm (not host libm).
  - **GPU float** — 🚧 roadmap → fixed-tree / Ozaki-scheme reductions; Metal and
    WebGPU have no hardware `f64`.
- 🚧 **Phase 10.6: Surface Syntax & Library Output** – tuple types,
  references, generics, slices, fixed-size arrays, struct literals,
  indexed/field assignment, multi-line arithmetic, RFC 0002 C-ABI export
  (D1/D3/D5 shipped in 0.2.6–0.2.8; D2 codegen pending).
- 🚧 **Improved Diagnostics** – Structured error messages with fix-it hints.

## Planned

- **Formal Verification** – Proof-carrying IR passes for safety-critical deployments.
- **Ecosystem Integrations** – Official bindings for Python, Swift, and WebAssembly targets.
- **Package Registry** – Public `mindpkg` registry with curated models.

## Genesis Ecosystem Audit — verified findings (2026-05-29)

Findings below were hand-reviewed against current code before listing. Open-core boundary findings are handled separately (per-project runtime-protection libraries are by design).

**Shipped 2026-05-29 (verified + CI-green):**
- Cross-substrate Q16.16 bit-identity CI gate on x86_64 (avx2) + ARM (neon) — #307.
- Format Check + Documentation CI restored to green; README/STATUS metadata corrected.
- Out-of-box `pip install` fallback (native → pytorch) for the skill router; route-server `top_k` clamped to [1, 64].

**Open (verified, prioritized):**
- **#306 — `std` heap out-of-bounds store — migration LANDED, behavior-verified (v0.7.1 gate: keystone re-bless remaining).** Every genuine single-byte store in `string` / `sha256` / `toml` now uses the byte-width-correct `__mind_store_i8` intrinsic (i64-aligned struct/array stores left intact; `tui` had no byte-store sites), closing the cross-substrate bit-identity landmine at the source. Behavior preservation is verified: SHA-256 FIPS 180-4 conformance (3/3, incl. the two-block-padding vector where the OOB would manifest) and the std-surface string/toml lowering suites all pass against a real `--emit-shared` ELF. Remaining for the tag: the bootstrap keystone byte-identity re-bless of `examples/mindc_mind/libmindc_mind.so` — the oracle is a real, post-migration ELF, but the `mindc build --emit=cdylib` reproduction path currently emits a launcher stub in the open toolchain (the ELF-capable path is `mindc --emit-shared`). See [`docs/byte-store-migration.md`](byte-store-migration.md).
- **Real cross-substrate bit-identity gate downstream** — the router's CI bit-identity check is sentinel-mode only; build the native encoder in CI and assert a real cross-architecture hash equality.
- **Determinism / CALF benchmark** — the inference pipeline's determinism harness is currently a placeholder; implement it so the CALF claim is falsifiable, and add a CI workflow.
- **RFC 0021 step 5 (#309)** — demote the legacy v2 IR; converge on one canonical IR (mic@1/mic@3) with attached provenance, closing the two-IR gap.
- **Recall-claim honesty** — scope any "byte-identical across substrates" wording to the audit-hash chain (Q16.16), not floating-point ranking scores.
- **Cross-ecosystem version-alignment discipline (#301)** — per-release sweep so `Cargo.toml` / `Mind.toml` / README / CHANGELOG versions never drift.
- **Keystone test fail-closed** — make `phase_g_keystone_bootstrap` fail (not pass) on a stub artifact when `MIND_BENCH_REQUIRE=1`, so a stub-environment re-bless cannot bake a fake byte-identity oracle.

**Prior-art rules locked (genesis research):** treat `trace_hash` as a type-system invariant, not a prose convention; hash canonical mic@3 *binary* bytes (re-anchored from mic@1 text on 2026-05-31 after a collision audit — mic@1 text can drop function-body semantics); close the two-IR gap before adding more evidence features; adopt `SOURCE_DATE_EPOCH` + a self-host fixed-point CI gate; consumer-side evidence verification is mandatory (generation without verification is theater); encode the substrate identity in the hash (SLSA L3); GPU substrates must avoid tensor cores for bit-identity claims (not IEEE-754 compliant); RVV kernels must use ordered reductions; pin each repo's toolchain via `rust-toolchain.toml` so the format gate cannot drift.

## Contributing

Roadmap items are tracked as GitHub issues with the `roadmap` label. Proposals should follow the RFC template in [`docs/rfcs`](rfcs/).

---

# Extended 2025–2026 Roadmap

The following phases extend the existing roadmap and define the next development milestones for MIND as it moves toward full v1.0 readiness, broader ecosystem adoption, and enterprise deployment.

---

## Phase 10 — Developer SDK & Examples

### Goals
Establish a developer-friendly ecosystem that showcases real-world usage of MIND across training, inference, MLIR workflows, and edge targets.

### Deliverables
- Add `examples/` directory with:
  - CNN classifier (training + inference)
  - Tiny edge model (optimized for <200 KB footprint)
  - Autodiff demonstration (reverse-mode)
  - MLIR/LLVM pipeline example (`mind → mlir → opt → ll → bin`)
- Improve CLI ergonomics and error messaging
- Provide initial Python bindings (`mindlang`)
- Add minimal model zoo with reproducible builds
- Publish example-focused documentation

---

## Phase 10.5 — Systems Programming Foundation

### Goals
Extend the MIND compiler to support governance logic alongside tensor computation. These features enable execution boundary kernels, access control policies, and deterministic decision logic — all compiled with the same verification guarantees as tensor programs.

### Motivation
The `policy.mind` execution boundary kernel (see `examples/policy.mind`) demonstrates the use case: fail-closed access control with enum-based typing, struct-based requests, and bitwise-packed confirmation codes. MIND becomes a *verifiable behavior language* — expressing certified decisions alongside tensor computation.

### Design Decision
**Tiers 1–2 ship. Tier 3 deferred to Rust FFI.** MIND owns governance logic (typed decisions, policy rules). Rust owns byte-level implementation (string matching, memory) via FFI. This preserves MIND's identity as a certified tensor + governance language without scope-creeping into general systems programming. (Internal review: unanimous on Option A.)

### Deliverables

**Tier 1 — Control Flow & Constants (2–3 days):**
- `if` / `else` expressions (already in EBNF grammar, not yet in parser)
- `while` loops (already in EBNF grammar, not yet in parser)
- `const` declarations (compile-time constants)
- Comparison operators in non-tensor context (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- Bitwise operators (`|`, `&`, `<<`, `>>`, `^`)
- Boolean operators (`||`, `&&`, `!`)

**Tier 2 — Algebraic Data Types (3–5 days):**
- `struct` with typed fields (already in EBNF grammar, not yet in parser)
- `enum` with unit variants and explicit discriminants
- Struct literal construction (`Effect { tag: ..., code: ... }`)
- Field access (`req.target.path`)
- Enum-to-integer casting (`code as u32`)
- `u32` integer type (for packed codes and bitfields)

**Deferred — Byte-Level Operations (use Rust FFI):**
- `&[u8]` byte slices, `b"..."` literals, `u8` type, `mut` bindings
- Byte-level string matching (`starts_with`, `contains_ascii_ci`) implemented as Rust FFI helpers
- Rationale: low-level memory types are redundant with Rust and would blur MIND's certified-AI niche

### Benchmark Impact
All new features are gated behind new keyword/token checks that fail immediately for existing tensor programs. Estimated regression on existing Criterion benchmarks: **<1%**. New parser arms compile to jump tables (O(1) dispatch).

### Spec Status
- `if`, `while`, `struct`, `match`, `for` are already defined in the EBNF grammar (`grammar-syntax.ebnf`)
- `enum`, `const` require grammar additions (see `mind-spec` future-extensions)

---

## Phase 10.6 — Library Output & C ABI (mindc 0.2.6 / 0.3.0)

### Goals
Turn `mindc` from an executable-only compiler into a library-emitting one,
so MIND code can be consumed by C, Python, Rust, and other runtimes via a
stable C ABI. This is the foundation Phase-2 native inference paths
(mind-nerve, MindLLM, rfn-mind) compile against.

### Motivation
Today: `Mind.toml`'s `[exports]` section is parsed but silently dropped.
`Node::Export` is parsed and discarded. The .so produced has no AOT
symbols for user `pub fn` — runtimes embed source and re-parse on dlopen.
The mind-runtime embedded parser has drifted from mindc's surface parser;
shipping `.mic` IR instead of source is the canonical path forward.

### Compile-speed invariant (non-negotiable)
The IP moat is cross-substrate bit-identity + the embedded evidence chain; the
1.8–15.5 µs frontend latency is a **protected compile-speed budget** that defends
it, not the moat itself (speed is a budget, not the wedge — see
`.bench-baseline-2026-06-01-correctness.txt`). Every Phase 10.6 feature
is gated **module-level only** — no statement-level cfg, no runtime
dispatch. Each item below ships a dedicated sub-benchmark that does not
move the headline numbers. CI gate: one-sided **+10% regression** per size
(a speedup of any size always passes; a larger regression is a documented
worth-it decision, not an auto-block).

### Deliverables — mindc 0.2.6 (library-emit foundation)

1. **`Node::Export` → IR `FnExport` variant.** Lower the parsed
   `export { foo, bar }` into `IRModule.exports: HashSet<FnId>` and verify
   the named functions exist. Gate: `feature = "ffi-c-user"`, parse-time
   early-out when no `export` keyword present.
2. **C wrapper codegen for `pub fn`.** For each `IRModule.exports` entry,
   emit `mind_fn_{name}_invoke(inputs, in_count, outputs, out_count) -> int`
   using the existing `MindIO` calling convention. Static FFI stubs stay
   for backward compat.
3. **`Mind.toml [exports]` honored end-to-end.** `ProjectManifest.exports`
   already parses; wire the consumer in the build pipeline to prepend
   manifest-declared symbols into the IR export list before codegen.
4. **Document `mic@1` IR consumption pattern.** Canonical parse happens at
   compile time; runtimes consume `.mic` via `ir::load()`. Embedded
   runtime parser is deprecated. mind-nerve, MindLLM, rfn-mind switch from
   shipping source to shipping `.mic` artifacts.
5. **`--profile=<default|systems|embedded>` CLI flag.** Wire the
   ProfileTag from `src/cache/fingerprint.rs` through the build pipeline
   so the same `Mind.toml` produces three distinct artifacts.

### Deliverables — mindc 0.3.0 (cdylib era + protection actions)

6. **`mind build --lib` cdylib emit.** Output `.so`/`.dylib`/`.dll`
   directly with symbol visibility limited to exported user wrappers +
   standard FFI stubs. Entry point not invoked; library is dlopen'd.
7. **AOT native codegen with symbol versioning.** LLVM lowering via
   `inkwell`; emit object files with `mind_fn_{name}_v1` symbols.
   Runtimes call `dlsym()` instead of embedded parser. Gates: existing
   `mlir-build` feature + new `aot-exports` feature.
8. **`Mind.toml [protection]` action transforms.** Three tiers documented
   in a new RFC (`docs/rfcs/0002-protection.md`): `obfuscate_strings`
   (compiler-side string table), `anti_debug` (runtime canary hook),
   `anti_tamper` (binary hash check). Replaces hand-rolled post-processing
   in downstream commercial build pipelines with a first-class compiler
   directive.
9. **Cross-directory imports (`use crate::foo::bar`).** Module-qualified
   resolver populates a module table during project load; type checker
   resolves identifiers across module boundaries. Gate:
   `feature = "cross-module-imports"`.

### Deprecated in this phase

- `Mind.toml [targets.cpu] sources / c_sources / headers` — decorative
  fields with no consumer. MIND's compilation target is IR → MLIR →
  LLVM IR, not mixed C+MIND. C interop comes via `[package.ffi]` in a
  separate RFC if needed.

### Sequencing

Items 1–5 ship in mindc 0.2.6 (~2 weeks). Items 6–9 ship in mindc 0.3.0
(~6 weeks). Item 8's `[protection]` action transforms move build-time
hardening into the compiler proper, replacing external post-processing
scripts.

### Anti-regression discipline

- Every PR touching parser / type_checker / lowering / codegen runs
  `cargo bench --bench compiler` and reports baseline-vs-current.
- New features ship as separate sub-benchmarks (e.g.,
  `export_list_1item`), never as deltas to the headline numbers.
- Gated features verify their cfg cost is <0.5% on headline benchmarks.
- `CacheKey` extends for every feature that changes compilation behavior.

### Acceptance gates

- mind-nerve compiles with `pub fn preselect_pre_tokenized` exported and
  callable as `mind_fn_preselect_pre_tokenized_invoke()`.
- mind-nerve's `Mind.toml [exports] c_abi = [...]` works end-to-end.
- mind-nerve build flow documents `mind compile --emit-ir` and the
  runtime loads `.mic` instead of `.mind`.
- MindLLM, rfn-mind, mind-mem repeat the same pattern.
- Cross-arch bit-identity gate (mind-nerve task #57) unblocked by 0.3.0.
- p95 ≤ 30 ms on 4-core CPU — already met on the Phase 1 PyTorch path via
  the `mind-nerve-routed` UNIX-socket daemon (23 ms p95 after warmup,
  closed in 0.1.0-alpha.7). The 0.3.0 AOT path tightens this further by
  replacing the embedded-parser dlopen path.

---

## Phase 11 — Benchmarks & Cloud Compiler Prototype

### Goals
Demonstrate performance, determinism, and flexibility. Introduce cloud-assisted builds.

### Deliverables
- Official benchmark suite:
  - matmul (CPU)
  - conv2d
  - small CNN
  - compile-time comparison (MIND vs PyTorch 2.0 AOT)
  - memory footprint analysis
- Publish results under `docs/benchmarks/*`
- Add deterministic build mode documentation
- Prototype cloud compiler endpoint:
  - `mind build --remote`
  - REST API for compilation requests
- Regression suite to ensure deterministic builds and MLIR→LLVM stability
- TurboQuant benchmark module (`bench/turboquant.mind`):
  - Validate that PolarQuant rotation + Lloyd-Max codebook + QJL residual correction
    can be expressed and compiled as pure MIND (requires Phase 10.5 struct/enum)
  - Benchmark: quantize/dequantize 1M KV vectors at 3-bit, measure throughput vs Python
  - This is the first real-world mind-inference kernel that exercises struct, enum,
    const, bitwise ops, and tensor ops together — ideal Phase 10.5 integration test
  - Grounded in recent KV-cache quantization research.

---

## Phase 12 — Enterprise Runtime & Edge Deployments

### Goals  
Enable enterprise and safety-critical adoption through stable runtime features, reproducibility, and mature edge support.

### Deliverables
- Finalize enterprise runtime:
  - feature-gated modules
  - deterministic execution
  - stable configuration API
- Edge deployment pipeline:
  - ARM Cortex-M support
  - RISC-V minimal backend
  - QEMU demo environment
- Deployment guidelines for:
  - cloud + on-prem
  - hybrid architectures
  - reproducible inference workloads
- Runtime safety documentation (Rust-based guarantees)

---

## Phase 13 — Neuroscience & Brain-Computer Interfaces

### Goals
Establish MIND as the premier language for real-time neural signal processing and brain-computer interface (BCI) applications. Leverage MIND's native tensor operations, static shape inference, and high-performance compilation for ultra-low-latency neural decoding in next-generation BCI systems (Neuralink-style implants, research BMIs, clinical neurotechnology).

### Deliverables

**Near-term (Phase 13.1):**
- Add `mind::neuro` standard library module:
  - Specialized tensor types for neural time-series (Channel × Time × Batch)
  - Common signal processing filters: bandpass, notch, ICA decomposition
  - Feature extraction primitives: CSP (Common Spatial Patterns), wavelet transforms, spectral power
- IO support for neuroscience data formats:
  - EDF/EDF+ (European Data Format) for clinical EEG
  - GDF (General Data Format) for BCI research
  - Optional integration with existing Rust neuro crates
- Example implementations:
  - Spike detection algorithms for invasive recordings
  - Motor intent decoding from ECoG/multi-unit activity
  - Basic P300 and SSVEP classifiers for non-invasive BCI

**Medium-term (Phase 13.2):**
- Optimized neural decoders:
  - RNN/LSTM templates for sequence-to-intent decoding
  - Transformer-based models for multi-modal neural data
  - Integration with autodiff for on-device decoder adaptation
- Tutorials and examples:
  - BCI Competition datasets (motor imagery, P300)
  - OpenNeuro integration for reproducible research
  - Real-time decoding pipeline examples
- Performance benchmarks:
  - Sub-millisecond inference for invasive BCI
  - Comparison with existing frameworks (MNE-Python, FieldTrip)

**Long-term (Phase 13.3):**
- Closed-loop BCI support:
  - Real-time feedback mechanisms
  - Online learning and decoder adaptation
  - Safety-critical execution guarantees for medical devices
- Research collaborations:
  - Partner with neuroscience labs for validation
  - Contribute to open BCI standards (IEEE, INCF)
  - Support for neuromorphic hardware targets

### Integration with Existing Features

- **Tensors**: Multi-channel neural data naturally maps to MIND's tensor model (channels, time, trials, batches)
- **Autodiff**: Enable gradient-based decoder optimization directly on implanted devices
- **MLIR/LLVM**: Generate highly optimized code for real-time constraints (<1ms latency)
- **Edge Runtime**: Deploy to resource-constrained BCI hardware (ARM Cortex-M, RISC-V)
- **Determinism**: Critical for reproducible neuroscience and FDA-regulated medical devices

---

## Phase 13.5 — Observer-Dependent Cognition (ODC) Language Primitives

### Goals
Introduce language-level support for observation-dependent computation. MIND already has `axis` for tensor operations — extend this to cognitive/governance contexts where the result of a computation depends on which observation basis is chosen.

**Spec:** `specs/observer-dependent-cognition.md`

### Deliverables
- **`@axis` annotation** — declare which observation basis a function/block operates in
  ```mind
  @axis("semantic")
  fn recall(query: Tensor[1, 768]) -> Tensor[K, 768] { ... }
  
  @axis("governance", invariants=[1,2,5,7])
  fn verify(txn: Transaction) -> Evidence { ... }
  ```
- **Axis contracts** — compile-time verification that composed computations use compatible observation bases
- **Determinism certificates** — compiler emits proof that a code path is:
  - `axis_independent` — fully deterministic, same result regardless of observation basis
  - `axis_dependent` — result varies with declared axis (must be explicitly declared)
  - **Generalization — property certificates (proof-carrying artifact).**
    The determinism certificate is the first instance of a broader class:
    a machine-checkable witness, emitted at compile time and travelling in
    the bundle next to the evidence chain, that a small trusted checker
    (re-run by `mindc verify` / the loader) can accept or reject. Where the
    evidence chain proves *provenance* ("this artifact came from this trace,
    bit-identically"), a property certificate proves a *property* ("this
    code path stays within its declared Q16.16 bound / halts / is
    axis_independent"). This is tractable here precisely because the forward
    is deterministic integer arithmetic, not floating-point — decidable
    refinement / bounded-SMT over Q16.16 integers, checked by a *small*
    kernel, not an imported prover. Direction only — adopt the lesson of the
    proof-assistant program (small checkable kernel; properties carried in
    the artifact) without adopting a dependent-type rewrite or a Lean/Coq
    dependency. Downstream consumer + falsifiable-promotion criteria are
    tracked in `rfn-mind/docs/roadmap.md` → Research watch → *Proof-carrying
    artifact*; the certificate-generation machinery is owned here (compiler)
    because that is where the evidence chain lives.
- **`observe` keyword** — explicit measurement operator for superposition states
  ```mind
  let candidates = speculate(draft_model, context)  // superposition
  let result = observe(candidates, axis="verifier_a")  // collapse
  ```
- **Multi-axis composition** — compose results from different observation bases with explicit fusion strategy (RRF, voting, weighted)

### Integration
- MIND compiler validates `@axis` annotations at type-check time
- MLIR lowering preserves axis metadata for runtime evidence chain
- 512-mind governance modules use `@axis` to declare invariant coverage

---

## Phase 13.6 — Deterministic Distributed Primitives

### Goals

Today, distributed AI runtimes ship TCP transport, NCCL/Gloo backends,
RingAllReduce, pipeline parallelism, and fault tolerance — production-grade but
**non-deterministic by reduction order**. Different runs of the same
sharded model produce slightly different logits because IEEE-754
floating-point reductions are not associative under parallel scheduling.

For audit, regulatory, and replay-required workloads this is a blocker.
Phase 13.6 closes the gap with **language-level primitives** that emit a
deterministic TP+PP plan from a single `.mind` source — sharding stays,
the reduction race goes away.

The driving use case is the BitNet b1.58 ternary-weight LLM described in
[`bitnet-mind-governance/docs/parallel_pipeline.md`](https://github.com/star-ga/bitnet-mind-governance/blob/main/docs/parallel_pipeline.md):
3-way TP+PP across consumer GPUs with bit-identical replay across runs
*and* across hardware vendors (verified by per-stage merkle roots).

### Surface (proposed in `.mind`)

```mind
use mind.distributed.shard
use mind.distributed.allreduce
use mind.distributed.allgather
use mind.distributed.pipeline
```

### New invariants enforced by `mindc`

| Invariant | Effect |
|-----------|--------|
| `deterministic_all_reduce` | refuses to emit if any cross-shard reduction is not lexicographic-order |
| `reduction_order_lexicographic` | extends the existing local-matmul reduction-order rule to cross-device gathers |
| `gather_order_lexicographic` | shard contributions concatenate in fixed shard-id order, never timestamp-arrival order |
| `evidence_chain_continuous` | every pipeline stage must verify the previous stage's evidence emit before processing — bypassing the chain refuses to compile |

### Compiler-side IR (✅ scaffolded in `mindc` — Apr 2026)

The [`mind/src/distributed/`](../src/distributed/) module ships the
IR-layer primitives now. 31 unit tests pass on `cargo test --lib distributed::`.
Exposed types:

* `ShardSpec` / `ShardLayout` — tensor sharding spec with replicated /
  split / split-2D variants and an even-divisibility check at typecheck
  time.
* `AllReduceOp` / `ReductionOrder::Lexicographic` — lexicographic
  shard-ID schedule baked into the op at compile time.
* `AllGatherOp` / `GatherOrder` — same discipline applied to all-gather.
* `PipelineGraph` / `PipelineStage` / `StageBoundary` — `[pipeline_stage(N)]`
  attribute lowering, with the evidence-chain-continuous invariant
  attached to every transition.
* `DistributedInvariant` / `InvariantViolation` — typed enforcement of
  the four invariants above.

### Engineering scope to reach proof-of-concept

Six to eight weeks of focused work, scoped for paper-validation:

1. **`mind.distributed.shard` + `allreduce` + `allgather`** (~3 weeks)
   - Direct CUDA P2P for proof-of-concept (no NCCL needed initially)
   - Or shared-memory IPC for single-machine multi-process validation
   - Lexicographic-order all-reduce + all-gather primitives
2. **`mind.distributed.pipeline` with `[pipeline_stage(N)]`** (~2 weeks)
   - Send/recv between stages; evidence-chain continuity verification
3. **Port `bitnet.mind` to use the new primitives** (~1 week)
   - Re-shard along attention heads (TP), split layer groups (PP),
     combine for 3-way memory split
4. **Cross-shard determinism harness** (~1 week)
   - Capture per-stage merkle roots, compare to single-device run
   - Validate across (x86 + 3× consumer GPU) and (ARM + 3× different
     GPU) for cross-hardware bit-identicality

### Hypothesis the work proves

> **H4**: Cross-shard determinism is achievable on TP+PP at the cost of
> < 5% throughput overhead vs. non-deterministic FP16 baseline.

Two measurable claims (governance overhead < 1% on local BitNet from
H1, and cross-shard determinism overhead < 5% from H4) — together this
is the shape of an MLSys / OSDI methods paper.

### Speed-preservation discipline

The 1.8–15.5 µs frontend latency is a **protected compile-speed budget** (the IP
moat is cross-substrate bit-identity + the embedded evidence chain; speed is a
budget, not the wedge). Distributed primitives must **never** widen it. Same rules as the language-profiles plan:

| Risk | What it looks like | Forbidden in mindc |
|------|--------------------|--------------------|
| Tax non-distributed code | Single-device compiles pay an analysis pass for collectives that aren't there | Distributed analysis only runs if at least one `mind.distributed.*` symbol is imported. Zero-import cost stays at 0 ns. |
| Linear shard-count blowup | Adding shards reruns shape inference per shard | `ShardSpec` is one struct attached to an SSA value; per-shard shape is computed once via `local_shape()`. World size doesn't enter typecheck cost. |
| Runtime collective dispatch in the binary | Binary contains a switch over collective backend / order | Reduction order is **compile-time-fixed** to lexicographic. The dispatch table never exists. |
| Per-statement invariant evaluation | Every op gets re-checked for `deterministic_all_reduce` | Invariant is a single flag on the op; verifier reads it in O(1). |

**Expected per-profile latency under Phase 13.6:**

| Profile + scenario | Expected frontend latency | vs. baseline |
|--------------------|---------------------------|--------------|
| `default`, no `mind.distributed.*` import | 1.8 – 15.5 µs | identical |
| `default`, with TP+PP collectives | ≤ 1.10 × baseline | bounded by O(num_collectives) shard-spec checks |
| `systems`, with TP collectives | ~0.9 – 3.3 µs | still faster than `default` baseline |

**Hard refuses (these stay out permanently):**

- Statement-level `#[cfg(distributed)]`. Module-level imports only.
- Runtime collective-order detection. Order is compile-time constant.
- Cross-profile distributed linking. A `default` shard cannot be linked
  into a `systems`-built peer (their stdlibs differ). Linker enforces
  via the MIC binary header.

The cache key in [`mind/src/cache/`](../src/cache/) already incorporates
`ProfileTag`; Phase 13.6 extends it with `WorldSize` so cross-shard
re-compilation never hits a stale entry.

### Dependencies

- mindc 0.3 primitives (`Vec`, struct fields, recursive functions)
  needed for the example modules to compile. Same blocker as the
  Language Profiles roadmap entry.

---

## Phase 14 — Full-Stack AI Framework

### Vision

MIND is evolving from a single-stack language focused on AI computation into a comprehensive full-stack AI framework that supports end-to-end machine learning. This stack will cover everything from building and training models to deploying them at scale in cloud and edge environments, providing a unified and cohesive experience for machine learning engineers, data scientists, and developers.

### Goals

- **End-to-End AI Development**: From research and prototyping to large-scale deployment
- **Seamless Integration**: Work naturally with popular data engineering and AI tools
- **Optimized Performance**: Efficient execution across CPU, GPU, and cloud environments
- **Ease of Use**: Well-integrated tools, APIs, and deployment frameworks

### Key Areas of Full-Stack Development

#### 1. Frontend Layer

**Language Enhancements:**
- Extend the language to improve developer experience through enhanced IDE support
- Advanced syntax highlighting and semantic code analysis
- Improved debugging tools with step-through tensor inspection

**Interactive Notebooks:**
- Browser-based interactive notebooks for writing, testing, and visualizing MIND code
- Jupyter-style cells with native MIND kernel support
- Real-time tensor visualization and model architecture diagrams

**Visualization Tools:**
- Integrated tools for visualizing tensor data and intermediate representations
- Model architecture visualization with automatic graph generation
- Training progress dashboards with loss curves, metrics, and resource utilization

#### 2. Backend Layer

**Compiler Optimization:**
- Extend the MIND Compiler to support advanced optimizations:
  - Automatic model pruning and quantization
  - Tensor fusion and operator scheduling
  - Enhanced support for TPUs, GPUs, and specialized AI accelerators

**Distributed Execution:**
- Distributed execution framework for training on clusters and multi-GPU setups
- Data parallelism and model parallelism strategies
- Gradient synchronization with NCCL-style collectives

#### 3. Middleware Layer

**API Layer:**
- REST and gRPC APIs for MIND-based model inference and serving
- OpenAPI/Swagger documentation generation
- Authentication and rate limiting for production deployments

**Data Integration:**
- Data engineering integrations for preprocessing and managing data pipelines:
  - Apache Kafka for streaming data
  - Apache Spark for batch processing
  - Native connectors for cloud storage (S3, GCS, Azure Blob)

**Model Serving:**
- Scalable, real-time model serving APIs for production deployment
- Batching and request queuing for throughput optimization
- A/B testing and canary deployment support

#### 4. Deployment Layer

**Containerization:**
- Pre-configured Docker images for training and serving MIND models
- Multi-stage builds optimized for minimal production footprint
- NVIDIA CUDA and ROCm base images for GPU workloads

**CI/CD Pipelines:**
- Continuous integration and deployment pipelines for automatic testing, training, and deployment
- GitHub Actions and GitLab CI templates
- Model versioning and artifact management

**Kubernetes Integration:**
- Helm charts for deploying MIND-based applications
- Horizontal pod autoscaling based on inference load
- Integration with Kubernetes operators for ML workflows (Kubeflow, MLflow)

#### 5. Ecosystem Support

**Pre-trained Models:**
- Library of pre-trained models for standard use cases:
  - Image classification (ResNet, EfficientNet variants)
  - Natural language processing (transformers, embeddings)
  - Time-series forecasting and anomaly detection

**Framework Interoperability:**
- Import/export compatibility with other frameworks:
  - ONNX model import/export
  - TensorFlow SavedModel conversion
  - PyTorch model migration tools

#### 6. Candidate Primitives — to be EVALUATED (not committed)

Model-primitive leads parked here for a future scoping pass. Each is a
*candidate* to be measured against the IFR (see TRIZ-Driven Direction) and the
wedge gates (cross-substrate byte-identity, run-to-run determinism) before any
commitment. Nothing here is planned work yet.

**Deterministic Kolmogorov–Arnold Network (KAN) layer** — *evaluate (scoped)*.
- **What:** a KAN layer (learnable univariate functions on edges instead of
  fixed node activations) implemented as a MIND-native kernel. Mathematically
  it is the Kolmogorov–Arnold superposition form: `f = Σq Φq(Σp ψqp(xp))` —
  spline compositions over a *fixed* dataflow graph.
- **Why it fits the wedge (the reason it's worth evaluating):** on a *fixed*
  grid with a *division-free* basis the layer reduces to exactly the primitives
  MIND already proves bit-identical — a per-edge univariate map (a Horner chain
  of Q16.16 `mul(>>16)` + `add`, structurally the existing dtype-branched
  elementwise lowering) feeding two pinned associative Q16.16 reductions (the
  existing `emit_vec_dot_q16` machinery, frozen by its scalar C oracle and the
  avx2 == neon canaries). No new determinism primitive is introduced, and a new
  kernel enters with no new ABI (a `__mind_blas_kan_*_q16_v` extern + one
  cross-substrate canary). A *deterministic* KAN — bit-identical across
  x86/ARM/GPU with a signed trace — is something no incumbent ships. Splines
  also sit inside the compiler-integrated-autodiff surface (US Provisional
  63/947,737): `d/dx` and `d/dcoeff` of the basis are the same division-free
  primitives, so the backward pass stays in-family.
- **Wedge-safe SCOPE (load-bearing — outside this it fights the wedge):**
  - *Inference on a FIXED / UNIFORM grid only.* Grid adaptivity during training
    is data-dependent and breaks training determinism; non-uniform grids
    (Cox–de Boor) need runtime knot-difference **division**, which is lossy and
    round-divergent across avx2/neon — **forbid** it in the deterministic tier.
  - *Division-free basis.* First-kind **Chebyshev** via the recurrence
    `T_{k+1} = 2·x·T_k − T_{k-1}` (coefficients exactly 2 and −1 → zero
    compile-time rounding, unlike Legendre/Jacobi rationals) with a fixed-point
    **clamp to [-1,1]** (exact integer min/max — `T_n` grows exponentially in
    degree outside the interval); **or** a uniform-grid B-spline pre-expanded to
    piecewise-cubic Horner (co-equal — wins on Q16.16 coefficient dynamic range
    and local features, costs one deterministic interval-select).
  - *Drop the SiLU base term.* `x·sigmoid(x)` needs `exp` — MIND has no
    deterministic Q16.16 transcendental; keep the basis only, or add a pinned
    polynomial approximation that itself passes a canary.
  - **Honest port note:** published Chebyshev-KAN code typically normalizes
    inputs with `tanh` and evaluates `T_n` via `cos(n·arccos x)` — i.e. three
    transcendentals. MIND's version is a determinism-hardened *re-derivation*
    (clamp instead of `tanh`; forced recurrence/Clenshaw instead of `cos·arccos`),
    **not** a drop-in port. So "no dynamic control flow" is only exact for the
    global-recurrence basis; a B-spline basis carries one deterministic
    interval-select branch.
- **Falsifiable go/no-go (first experiment — the smallest thing that kills it
  if it's wrong):** land a Q16.16 Chebyshev/Horner edge-eval as a
  `cross_substrate_identity` canary and require avx2 == neon plus the 15-canary
  run-to-run gate. *Status:* a standalone Q16.16 Chebyshev-recurrence eval
  already passes run-to-run determinism + byte-exact vs an independent
  fixed-point reference on x86; the remaining gate is the avx2 == neon canary on
  real aarch64.
- **Then measure (commit gates, not assumptions):** (2) approximation quality
  vs the fixed-point quantum (1/65536) on a real task — fixed-point KANs trade
  dynamic range for determinism, and the cost must be characterized, including
  how quantization error accumulates under two-layer composition; (3)
  autodiff-through-basis cost under the strict-FP / fixed-point tiers.
- **Explicitly out of scope of this entry:** this is *function approximation*,
  orthogonal to the execution-determinism / trace-anchor experiments — do not
  conflate the two threads. Grounded in recent basis-function-design research;
  superposition-theorem lineage (Kolmogorov / Arnold / Lorentz / Sprecher).

**Data Pipeline Connectors:**
- Seamless integration with data lakes and cloud platforms
- Connectors for popular databases (PostgreSQL, MongoDB, ClickHouse)
- Integration with feature stores (Feast, Tecton)

### Phased Rollout

**Phase 14.1 — Frontend Language and Developer Tools:**
- Enhanced language features for easier model development
- Full IDE support with syntax highlighting and auto-completion
- Integration of interactive notebooks for AI experimentation

**Phase 14.2 — Compiler Optimizations and Backend Support:**
- Extension of MIND's compiler to support advanced tensor optimizations
- Development of runtime support for distributed training
- Full support for cloud-based, multi-GPU, and distributed computing environments

**Phase 14.3 — Middleware Development:**
- Creation of MIND's API layer for inference and model serving
- Integration with data engineering frameworks (Kafka, Spark, etc.)
- Development of real-time serving APIs for production-ready AI models

**Phase 14.4 — Full-Stack Deployment and Ecosystem Integration:**
- Docker images and Kubernetes deployment support for MIND models
- Pre-trained models and ecosystem tools for end-users
- Full integration with CI/CD pipelines and cloud-native deployments

---

## Summary

These extended phases (10–14) guide MIND into the next major development era:
- richer SDK and examples,
- systems programming foundation (enum, struct, control flow, byte slices),
- formal benchmarks,
- early cloud compiler,
- enterprise-grade runtime,
- robust edge interfaces,
- and comprehensive full-stack AI framework capabilities.

Together they represent the roadmap toward **MIND v1.0** and production-ready adoption across multiple industries, enabling end-to-end machine learning from research to deployment.

---

## TRIZ-Driven Direction

This section captures the *criterion* against which roadmap items are evaluated, separate from the items themselves. It is direction-setting, not feature-generating. Every entry above is auditable against the Ideal Final Result below; every future entry should be too.

### Ideal Final Result (IFR)

> A developer writes governance and tensor logic at its natural level of abstraction. The compiler produces code that is provably safe under declared invariants, runs at the speed of unverified code, and requires no runtime checks the developer did not declare.

The IFR is *unreachable*. That is the point. It defines the gradient along which every roadmap decision should move, and it makes regression visible: any change that pushes the system away from this state is structurally suspect, regardless of how attractive it looks in isolation.

### Laws of System Development Applied to MIND

Five trajectories that govern how engineered systems evolve. Each suggests where to invest and where *not* to.

| Law | What it means for MIND | Status |
|---|---|---|
| **Uneven subsystem development** | One subsystem lags and gates the others | Frontend is fast (≤15.5 µs frontend stages). Tooling, IDE integration, diagnostics lag. Investing further compiler optimization yields less than investing tooling. |
| **Mono → bi → poly system** | Single → multi → many, with shared semantics | MIND is bi-system today (multiple targets via `mindc`). Poly-system goal: unified invariant semantics across MIND-language consumers, with cross-project contract verification. |
| **Increasing controllability** | Rigid → parameterized → automatic | `[invariant]` annotations are parameterization. Next stage: invariant *inference* under bounded conditions, not only assertion. |
| **Transition to micro-level** | Coarse mechanisms → fine-grained mechanisms | **Anti-pattern for MIND.** Statement-level configuration or runtime dispatch would break the compile-speed property. The micro-level transition is forbidden at the language frontend; if forced, it migrates to the runtime layer instead. |
| **Rhythm coordination** | Synchronization across subsystems | Q16.16 fixed-point already coordinates numerical rhythm across backends. Next stage: coordinating compile-time invariants with runtime evidence chains so consumers receive matched compile-time and runtime guarantees. |

### Separation Principles for Conflict Resolution

When two desirable properties conflict (e.g. *more invariants* vs *no compile-speed regression*), the design discipline is to separate the conflict rather than compromise:

- **Separation in time** — the property holds at one phase of the pipeline, not another (e.g. invariant verified at compile time, not re-checked at runtime).
- **Separation in space** — the property holds in one module or scope, not another (e.g. governance invariants on `governance/` paths, not on tensor kernels).
- **Separation in condition** — the property holds under one configuration, not another (e.g. debug-mode trace, release-mode silent).
- **Separation in scale** — the property holds at one granularity, not another (e.g. module-level feature gates, never statement-level).

This is the same separation discipline used by `[invariant]` conflict resolution in MIND-language consumers (see consumer-side governance roadmaps). It is *not* a 40-principles invocation. It is the small subset of TRIZ that has direct mechanical use here; the rest is left as catalog reference, not method.

### Anti-Patterns Made Explicit

- **Statement-level `cfg` or runtime dispatch in the frontend.** Violates compile-speed preservation. Any feature requiring it is rejected at design time, not after benchmarking.
- **Adding invariants without a separation strategy.** New invariants must declare *which separation principle* makes them coexist with existing ones. If none applies, the conflict is real and a redesign is required.
- **Driving design from the analogy.** TRIZ is a checklist for direction, not a generator. Features come from product needs and consumer pull; TRIZ filters them, it does not invent them.

### Runtime Layer (Mind-Runtime)

The frontend IFR above governs the compiler. The runtime layer has a *different* IFR with different optima — including one law that is explicitly inverted from the frontend rule. Both layers live in this repository and must stay coherent under both criteria.

**Runtime IFR.** Compiled MIND programs execute with bit-exact determinism across backends — realized today for deterministic-integer / Q16.16 lowering on CPU substrates (x86-`avx2` == ARM-`neon`) and for scalar IEEE-754 float on the strict path; cross-substrate vector-float and GPU determinism are the roadmap target — record verifiable evidence of every governance-relevant decision, and enforce declared invariants at zero overhead on the legitimate execution path. Failures degrade into sealed evidence rather than silent corruption.

**Laws applied at the runtime layer.**

| Law | Runtime application | Status |
|---|---|---|
| Uneven subsystem development | Runtime backends evolve unevenly (CPU, GPU, accelerator). Slowest backend gates portability claims. | Invest in the slowest backend, not the fastest. |
| Mono → bi → poly | Runtime is poly today (multiple backends). | Next stage: unified evidence chain across backends so a program's evidence trail is backend-independent. |
| Increasing controllability | Runtime policy is largely compile-time-fixed. | Next stage: runtime-tunable severity tiers within a compile-time-bounded envelope (the envelope is immutable; the position inside it is configurable). |
| Transition to micro-level | **Inverted from frontend.** At runtime, instruction-level scheduling, branch hints, vectorization are appropriate and often required. The forbidden boundary is *crossing* a compile-time invariant, not *implementing* it efficiently. | Pursue micro-level wherever the legitimate path demands it; never let it weaken an invariant. |
| Rhythm coordination | Runtime emits evidence; compile-time consumers verify invariants against it. | The cycle closes only when both timescales agree on the schema and meaning of each evidence record. |

**Separation principles at runtime.**

- **Time** — compile-time verification vs runtime check vs runtime evidence record. Three phases, three different cost/value trade-offs. A property proven at compile time is *not* re-verified at runtime; that path has been replaced with evidence emission, not redundancy.
- **Space** — hot path vs cold path vs governance audit path. Hot path admits no overhead beyond the IFR; cold path admits diagnostic instrumentation; audit path admits full evidence-chain materialisation.
- **Condition** — release vs debug vs forensic mode. The same program produces different evidence detail under different modes; the compile-time invariants are unchanged.
- **Scale** — per-instruction, per-basic-block, per-module, per-program. Each scale has its own legitimate optimisation surface; mixing scales is a code smell.

**Runtime anti-patterns.**

- **Re-verifying at runtime what was proven at compile time.** Wasted overhead, no added safety, violates the IFR.
- **Silent failure paths.** Any failure that does not produce a sealed evidence record breaks the chain and the contract with governance consumers.
- **Backend-specific semantics leaking into program behaviour.** Determinism is a per-program property; backend differences must be invisible above the FFI boundary.
- **Performance regressions justified by future features.** The IFR demands zero overhead on the legitimate path *now*; deferred-cost arguments are rejected at design time.

---

## Phase 15 — Self-Hosting & AI-Era Language Efficiency

The frontend moat and the invariant/evidence/determinism core only
compound if the language can host itself and is optimised for the
metric that actually matters now: an AI writes the code, a compiler
must prove it, and the result must be reproducible without re-running
it. Phase 15 makes both explicit and tracked.

### Guiding principle (holds for every Phase 15+ feature)

**AI-era efficiency = minimise tokens-to-a-trusted-reproducible
artifact, never at the cost of the µs frontend.** Concretely the
product to minimise is:

```
(tokens to express correct intent)
  × P(the generator got it wrong)
  × (cost to detect it is wrong)
  × (cost to reproduce / audit the result)
```

C optimises only the first factor's runtime; dynamic languages
optimise only token count. MIND optimises the whole product. Any
feature that improves one factor by regressing the frontend envelope
(`.bench-baseline` one-sided **+10% regression** gate on the three headline
benches) triggers a worth-it review at design time — the same anti-regression
discipline already in force (a speedup never trips it).

### Deliverables — self-hosting bootstrap

1. **Stage-0 → Stage-1 → Stage-2 bootstrap.** Rust `mindc` (stage-0)
   compiles a MIND-sourced `mindc.mind` (stage-1); stage-1 compiles
   itself (stage-2). `stage1 == stage2` byte-identical is the
   self-hosting acceptance gate. Rust leaves the build path on pass.
2. **Native-ELF backend (NORMATIVE self-host path — `src/native`).** The
   pure-MIND front-end emits native ELF directly; the native-ELF fixed point
   is closed as of v0.10.0 (1 055 777 B, byte-identical). MLIR stays a
   downstream-interchange / exotic-chip-reach concern; the self-host bootstrap
   path uses the native-ELF backend. `IRModule` is the fixed point shared by
   both stages — it is not redesigned for self-hosting. Remaining: wire the
   pure-MIND SHA-256 to the `ir_trace_hash` PT\_NOTE emit, then delete
   `src/native` Rust backend.
3. **Pure-MIND std surface.** Growable `Vec`, `String` operations,
   order-deterministic `Map`, evidence-emitting file I/O. This is the
   long pole: a lexer/parser/symbol-table cannot be written without
   it. Scoped in **RFC 0005** (`docs/rfcs/0005-pure-mind-std-surface.md`)
   — five compiler intrinsics, everything else pure MIND, six-phase
   adoption ending in a self-hosted-lexer smoke test.
4. **Cross-module imports** (`use crate::x::y`) — already roadmap
   item 9 of Phase 10.6; restated here as a hard self-hosting
   prerequisite (a compiler is many files).

### Deliverables — pure-MIND dogfooding (Stage 4)

4b. **Pure-MIND MCP server over `mic@3` + evidence chain.** Reimplement the
   MCP tool-serving surface (starting with mind-mem) in pure MIND — no Python —
   using MIND's own protocols and formats end-to-end: `mic@3` canonical binary
   as the wire transport and the MAP epilogue's signed evidence chain
   (`trace_hash = sha256(emit_mic3(ir))`), so every tool call and response is
   byte-canonical and cryptographically tamper-evident — the property no
   JSON-RPC MCP has. A thin JSON-RPC↔`mic@3` bridge at the client edge keeps
   standard-MCP interop. Blocked-by the pure-MIND IO/JSON/HTTP stack (TLS
   client, HTTP semantics, evidence-chain HTTP layer) and the deterministic-IO
   substrate. **Ships behind FULL security review** (`mind-security-reviewer`:
   wire-format attacks, evidence-chain forgery, loader DoS, atomic ordering)
   **plus features hardening** — fail-loud validation at every boundary, no
   surface lands un-audited.

4c. **`mic@4` — autoresearch/alg-inv-evolved successor wire format.** Once
   `mic@3` is stable, turn the autoresearch + alg-inv (AB-MCTS) engine on the
   canonical IR encoding itself — fitness = **smaller + faster emit/parse**,
   HARD-GATED by every invariant `mic@3` guarantees: deterministic byte-canonical
   output, the emit→parse→emit fixed point, cross-substrate byte-identity, and
   `trace_hash` stability (the evidence chain must survive). Target a denser
   opcode / string-table layout and a faster codec than `mic@3` with zero loss
   of determinism or provenance. Ships behind a version bump (`MIC4_VERSION`), a
   `mic@3`↔`mic@4` migration, a full round-trip proof, and security review — a
   real speed/size win only if it beats `mic@3` on a measured corpus AND stays
   byte-identical (no fake wins).

### Deliverables — AI-era efficiency surface

5. **Token-efficiency as a first-class design constraint.** Every
   surface-syntax decision is evaluated by "how many tokens does a
   generator spend to express this *correctly*?" Terse-by-default;
   ceremony is a regression. Tracked as a review checklist item, not
   aesthetics.
6. **AI-era efficiency benchmark.** A published harness measuring the
   four-factor product above on a fixed task suite, reported per
   release alongside the frontend µs benches. The yardstick is
   defined by this metric, not SPECint.
7. **Deterministic concurrency model (RFC pending).** Structured,
   replayable, evidence-carrying parallelism. MIND today is
   deterministic-sequential; AI systems are inherently parallel, so
   this is the largest single gap between "efficient language" and
   "efficient language for AI systems." Prerequisite for any
   concurrent MIND program. Highest-priority new RFC.
8. **Index-refinement types (RFC pending).** Presburger-decidable
   refinements of the form `{x: T | lo <= x < hi}` over integer
   indices — SMT-free, so frontend-safe. Encodes tensor-shape bounds
   and governance ranges as types. (Flux-style; full SMT refinement
   is explicitly rejected as moat-incompatible.)
9. **Tensor layout + uniqueness annotations (RFC pending).**
   Type-level memory-layout (row/col-major, alignment) and
   Futhark-style uniqueness for in-place mutation proofs — the
   missing primitives for provably-aliasing-free GPU kernels. O(1)
   parse cost, single-bit IR flag.

### Sequencing

Items 3 + 4 are the long pole and gate everything else. 1 + 2 are
mechanical once 3 + 4 land (the backend mirrors the existing
`src/mlir/lowering.rs` structural-fold pattern). 5 is continuous
discipline from now. 6–9 are independent RFCs; 7 (deterministic
concurrency) is the highest priority because it gates both concurrent
MIND programs and the AI-era efficiency claim.

### Acceptance gates

- `stage1 == stage2` byte-identical; Rust removed from the documented
  build path.
- A non-trivial MIND program (target: the lexer/parser of `mindc`
  itself) compiles and runs through the self-hosted backend with
  cross-arch bit-identical output.
- The AI-era efficiency benchmark is published and tracked per
  release; no frontend headline-bench regression beyond the one-sided
  +10% gate (speedups always welcome) across all Phase 15 work.

---

## Phase 16 — CLI Agent Harness Stack

> **Full design: [`docs/rfcs/0013-cli-agent-harness-stack.md`](rfcs/0013-cli-agent-harness-stack.md)** (Draft). This section is the roadmap surface; per-tier design — including the Tier 2 TLS scope decision (pure-MIND vs named-C-crypto FFI) — lives in the RFC.

### Goals

Bring the MIND stdlib + runtime to the point where a Rust-class CLI
agent harness (claw-code / OpenClaw / claude-code class — streaming
LLM client, MCP protocol, tool execution, terminal UI) can be built
in pure MIND end-to-end. This is a credibility target like the
mindc self-host: a non-tensor system that exercises general-purpose
capability the way Rust+tokio+reqwest+ratatui does today.

### Motivation

The v0.7.0 stdlib already ships the boring half (string/vec/map,
json/toml/regex, fs/process, cross-platform binary releases). What's
missing is the *network and concurrency* half — and a CLI agent is
the cleanest external dogfood target for that work. Each piece below
also unlocks downstream consumers independently: TLS unlocks any
secure I/O, async I/O unlocks mind-inference's serving layer, MCP
client unlocks tool-using MIND programs, TUI primitives unlock
operator-facing MIND tools.

### Design Decisions

**Pure-MIND, not FFI.** The whole point of this phase is to exercise
MIND's general-purpose surface. FFI-wrapping `libcurl`/`tokio`/`rustls`
would satisfy the demo but skip the language work. The acceptance bar
is the same as Phase 15 self-host: byte-identical builds, no
`unsafe` outside named FFI boundaries (libc syscalls only).

**Sequencing is dependency-driven.** TLS gates HTTP-as-anything-real;
async I/O gates streaming; both gate the agent. CLI parser, TUI, and
MCP impl are independent and parallelizable once those land.

### Current Status (verified against HEAD, v0.7.0)

| Capability                | Status                                  | Gap |
|---------------------------|-----------------------------------------|-----|
| CLI parser                | `std.string`/`std.vec`/`std.map` shipped; no `std.cli` | Build `std.cli` (arg parsing, subcommands, env binding) |
| JSON / TOML / regex       | `std.json`/`std.toml`/`std.regex` shipped (RFC 0005 Phase 2) | None |
| FS / process              | `std.fs`/`std.process` shipped          | None |
| Cross-platform binary     | Linux musl + macOS universal + Windows MSVC shipped (#225 closed) | None |
| `std.io`                  | Bare `stdin`/`stdout`/`stderr` only     | No ANSI / no TTY detection / no color |
| `std.net`                 | Raw POSIX TCP/UDP only (IPv4/IPv6)      | **No HTTP, no TLS, no HTTPS** |
| Async I/O                 | RFC 0011 Phase A shipped (sync `ReplayScheduler` only) | Phases B–F deferred; real async I/O is the long pole |
| Streaming (SSE / chunked) | n/a                                     | Requires async + HTTP framing first |
| MCP protocol              | Not in tree                             | Build from scratch in MIND |
| TUI / ANSI / coloring     | Not in tree                             | Build `std.tui` (sequences + termios) |

### Deliverables

**Tier 1 — CLI surface & terminal control (parallelizable, ~1–2 weeks):**
- `std.cli` — flag parsing, subcommand dispatch, env-var binding, help generation; deterministic argv ordering
- `std.io` extensions — TTY detection (`isatty`), ANSI escape sequences (color, cursor, clear), `winsize` ioctl wrapper
- `std.tui` — termios raw-mode toggle, key-event reader, line editor primitive, frame buffer (minimal — no full ratatui clone)

**Tier 2 — TLS layer (the long pole, ~6–10 weeks):**
- Decision gate: pure-MIND TLS 1.3 vs FFI to a minimal crypto core (`ring` / `BoringSSL`). Pure-MIND fits the "exercise the language" goal but is a multi-quarter project; FFI to a named C crypto lib is a defensible compromise if scoped as the *only* FFI exception
- `std.tls` — TLS 1.3 client (server-auth only at first), ALPN, SNI, session resumption
- Acceptance: connects to a public HTTPS endpoint, verifies cert against system trust store, performs round-trip with byte-identical handshake transcript across runs given the same `ReplayScheduler` seed

**Tier 3 — HTTP client (~2–3 weeks after TLS):**
- `std.http` — HTTP/1.1 request builder, response parser, header handling, **chunked transfer-encoding decoder** (SSE rides on this)
- `std.http.sse` — `text/event-stream` parser sitting on chunked transfer; deterministic event emission
- Acceptance: streams a chunked response (e.g., an Anthropic/OpenAI-compatible `chat/completions` with `stream:true`) end-to-end without buffering the full body

**Tier 4 — Async I/O (RFC 0011 Phases B–D, the actual blocker behind Tier 2/3):**
- Phase B — real executor (work-stealing or fixed-pool), still under the `Scheduler` first-class-value model from Phase A
- Phase C — async I/O primitives (epoll/kqueue/IOCP), `Future<T>` allocated via `GenRef<T>` per RFC 0010 §3.3
- Phase D — `Sender`/`Receiver` channels, `select`/`race`, structured task supervision
- Acceptance: a `ReplayScheduler` run of the agent reproduces byte-identical transcript across machines

**Tier 5 — MCP protocol (~3–4 weeks, can start after Tier 1):**
- `std.mcp.client` — JSON-RPC framing, capability negotiation, tool listing, tool invocation
- `std.mcp.server` — same surface from the server side, for MIND programs that *expose* MCP tools (mind-mem / mind-nerve become candidates)
- Acceptance: a MIND CLI calls a stdio MCP server (claudeai-flavored), invokes a tool, and parses results — round-trip in pure MIND

**Tier 6 — Demo target: `mindcraft-agent`:**
- A claw-code-class CLI: streaming LLM client (HTTPS + SSE), MCP client, tool execution (`std.process` sandbox + `std.fs` scoped to a workdir), terminal UI, `--replay` flag using `ReplayScheduler` for deterministic playback
- Acceptance: byte-identical transcript across machines under `ReplayScheduler`; loads a remote LLM endpoint and an MCP server; works on Linux/macOS/Windows from the shipped cross-platform binaries

### Sequencing

Tier 1 + RFC 0011 Phase B start in parallel — both independent. Tier
2 (TLS) is the load-bearing long pole: nothing in Tier 3 ships
without it. Tier 4 Phases C/D are pre-req for the *deterministic*
streaming claim in Tier 3 acceptance. Tier 5 (MCP) can start as soon
as Tier 1 is in, but its acceptance test needs Tiers 2+3 for a real
HTTPS-backed MCP. Tier 6 is the integration sprint after all five
land.

Realistic earliest end-to-end: **mid-to-late 2027**, gated by the
Phase 15 self-host work landing first and the TLS scope decision.

### Acceptance Gates

- Every Tier deliverable ships with conformance tests in `tests/`
  and a worked example in `examples/`
- The mindcraft-agent demo runs on Linux musl, macOS universal, and
  Windows MSVC from a single source tree, with byte-identical
  `ReplayScheduler` traces verified in CI across all three OSes
- No FFI outside libc syscalls *unless* the Tier 2 TLS decision
  explicitly approves a named C crypto library, in which case that
  library is the only exception, named and version-pinned in the RFC
- Frontend µs benchmarks: no regression beyond +10% (one-sided) across all
  Phase 16 work (same gate as Phase 15)

### Out of Scope

- Full ratatui-class TUI framework (a minimal one is in Tier 1; rich
  widgets are a separate later effort)
- HTTP server (this phase is client-only; server is mind-inference's
  concern in Phase 12)
- WebSocket (not required for the SSE-based MCP/LLM demo; future RFC)
- HTTP/2 / HTTP/3 (HTTP/1.1 is sufficient for SSE; defer)
- Generic certificate-issuance / ACME / let's-encrypt automation

---

## AGI Integration

This repo is part of the broader STARGA AGI stack. The downstream
orchestrator's roadmap pins:
- Gap 3: Rule induction → .mind programs (compile + verify)
- Gap 8: Reasoning primitives (hypothesis, evidence, inference_step)
- Phase 10.5 is prerequisite for all AGI .mind modules
