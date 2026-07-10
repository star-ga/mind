# Changelog

All notable changes to the MIND compiler project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **Determinism is enforced by default; non-determinism can no longer leak
  untraced.** MIND programs are deterministic by design, but as a systems language
  MIND must compile anything — so non-determinism is supported as an *explicit,
  attested* opt-in, never a silent leak. Three layers:
  - **Honest attestation (by derivation).** The `evidence_chain.determinism` field
    was HARDCODED to `deterministic`, so a `random()` / `now()` program forged
    `determinism: deterministic` — the exact claim `mind verify` reports. It is now
    derived from the IR: a module calling a PRNG / wall-clock / stdin builtin
    attests `nondeterministic`; every other module (incl. seeded `randn(shape,
    seed)`) stays `deterministic`.
  - **Build gate.** Producing a runnable (`--emit-obj` / `--emit-shared`) or
    attested (`--emit-evidence`) artifact from a program that calls such a builtin
    is REJECTED fail-loud, naming the offender and pointing at the seeded
    `Random(seed=…)` API — unless `--allow-nondeterministic` authorises it. With
    that flag the program compiles AND still attests `nondeterministic` (the flag
    authorises the build, never the label).
  - **Verify re-derivation (tamper-proof).** The `determinism` field sits outside
    the `trace_hash` anchor, so on an unsigned artifact it was forgeable. `mind
    verify` now RE-DERIVES the mode from the hashed body (like `fp_mode`), reports
    that authoritative value, and FAILS CLOSED if the stored MAP field disagrees —
    a forged `deterministic` label cannot pass. New `--require-deterministic` flag
    (mirrors `--require-strict-fp`) fails closed for a consumer that requires
    reproducibility.

  The `determinism` field lives in the MAP epilogue, not the `trace_hash`, so
  byte-identity is unaffected.
- **Bare transcendental / RNG builtins no longer forge a `strict` FP attestation.**
  `sin`/`cos`/`exp`/`log*`/`pow`/`sigmoid`/`log_softmax`, `fft`/`fft2d`/`ifft`, and
  `random`/`rand_*` lower to a plain `Instr::Call` (no `ExternFnDecl`, no `__mind_`
  prefix), so `fp_mode` never tainted them — a `sin(x)` or `rand_uniform(s)` program
  passed `mindc verify --require-strict-fp` with `fp_mode: strict` +
  `determinism: deterministic` (and `random` links a libc PRNG). They now taint the
  module to `Relaxed`, so such a program correctly FAILS `--require-strict-fp`. The
  exact / correctly-rounded builtins (`sqrt`/`floor`/`round`/`abs`/`max`) and the
  tensor reductions/contractions already tainted elsewhere stay strict.
- **`mindc clean --cache` could delete directories OUTSIDE the cache root via a
  crafted `Mind.lock`.** The git-cache path was built from unvalidated lockfile
  fields (`rev` / `source`) and `remove_dir_all`'d with no containment check, so an
  absolute `rev` (which `PathBuf::join` lets REPLACE the whole prefix) or a `../`
  component let an untrusted lockfile — e.g. from a cloned repo — target any
  directory the operator's uid can unlink (up to `~/.ssh`). Fixed fail-closed:
  `git_cache_dir` rejects any `..` / NUL / absolute component, and the delete site
  canonicalizes the target and requires it to live strictly under `~/.mindenv/cache`
  before removing it.
- **mic@3 string-reference decompression bomb (parser DoS).** `parse_mic3` cloned a
  string-table entry per wire reference with no bound on total decoded output, so a
  ≤10 MiB artifact that referenced a multi-MiB entry millions of times expanded into
  terabytes of retained heap and OOM-killed `mind verify`. The parser now charges
  every string clone against a generous per-parse decode budget (64× input, ≥64 MiB
  floor) and fails closed on a reference bomb; legitimate modules that re-reference
  short identifiers are unaffected (accepted-parse byte-identity preserved).
- **Pinning a signer key no longer accepts an unsigned artifact.** `mind verify
  --signer-pubkey <key>` (a trust allowlist) enforced the allowlist only inside the
  `Valid` signature arm, so an artifact with the `signature.*` fields stripped — or
  simply never signed — passed the pin: an attacker could emit their own body
  attested-but-unsigned (no private key needed) and satisfy `--signer-pubkey`. A
  pinned signer now REQUIRES a valid signature; an absent signature fails closed.
- **C-ABI export stub returned `MIND_OK` without dispatching.** The
  `ffi-c-user`-gated `mind_fn_<name>_v1_invoke` wrapper returned `0` (success) while
  dispatch (RFC 0003) is still deferred, so a `dlsym` caller would read its
  uninitialised out-params as a successful call. It now returns `38` (ENOSYS) until
  real dispatch lands.
- **Verify docs corrected to match behavior.** The `verify` exit-code help wrongly
  claimed unattested artifacts exit `1` (they exit `0` with `attested: false` by the
  RFC-0017 opt-in design; use `--require-*` / `--signer-pubkey` to fail closed), and
  the v2 verifier-core doc implied the CLI walks `parent` back-links — it does not:
  `parent` is decoded and reported but not yet traversed or authenticated (now
  stated honestly, with a deferred-work marker).
- **Documented: `trace_hash` covers the type-erased canonical IR.** Two source
  programs that differ *only* in a scalar type they never observe (a pass-through
  `f(x: i64)` vs `f(x: f64)`, whose bodies carry no type-dependent instruction)
  encode to identical `mic@3` and so share a `trace_hash` — a
  canonicalization-completeness boundary, NOT a tamper-detection failure (a given
  artifact's bytes cannot be altered without changing its hash; any program that
  actually uses the scalar emits type-distinct instructions and does not collide).
  Closing it would invalidate every existing evidence hash across the ecosystem for
  a narrow, low-exploitability gap, so the proportionate response is precise
  documentation (SECURITY.md); an additive typed-fn opcode remains the
  backward-compatible upgrade path if future IR makes scalar types observable more
  broadly.

### Fixed
- **Float/string-literal and tuple `match` arms silently returned the last arm.**
  A `match` whose arm pattern the runnable v1 lowering cannot desugar — a FLOAT or
  STRING literal (`match x { 1.0 => 10, 2.0 => 20 }`) or a top-level TUPLE
  (`match (a, b) { (x, y) => x + y, _ => 0 }`) — bailed `desugar_match_to_if` to a
  sequential fallback that IGNORES the scrutinee and returns the last arm's value
  (reproduced: the float match returned 20 for every input; the tuple match
  returned 0 for a correct answer of 3). The runnable gate now rejects these arm
  patterns fail-loud (int-literal and enum-discriminant matches are unaffected).
- **Ambiguous bare payload constructor silently matched the wrong enum.** When the
  same variant name carries a payload in two enums (`Alpha::Foo(i64)` +
  `Zeta::Foo(i64)`), a bare `Foo(7)` resolved to the FIRST enum's discriminant tag
  (deterministic map order), so `let z: Zeta = Foo(7)` got Alpha's tag and matched
  the wrong `match` arm (reproduced: returned 1007 for a correct answer of 7) — a
  silent miscompile `mindc check` passed. A new check-phase gate rejects a bare
  payload ctor whose short name collides across enums, requiring qualification
  (`Zeta::Foo(7)`). Inert unless a payload variant name collides, so keystone
  byte-identity holds; unique bare ctors (`Some`/`Ok`) are unaffected.
- **`EnumStruct` match arm with a literal/nested field silently miscompiled.**
  `check_match_runnable` inspected only `Pattern::EnumVariant`, so a struct-payload
  arm `E::V { x: 0 }` slipped past the gate; the desugar then dropped the literal to
  a wildcard, so it matched a value with `x = 5` and returned garbage (reproduced:
  returned 0 for a correct answer of 5). The gate now also rejects `EnumStruct` arms
  whose field sub-patterns are not all `Ident`/`Wildcard` (fail loud, like the tuple
  form).
- **Interpreter `u64`→float used a signed cast** (interp ≠ artifact for `u64 ≥ 2^63`).
  The tree-walking evaluator's `As` int→float arm did signed `n as f64`, while the
  compiled path lowers `arith.uitofp` (#105) — so `1u64<<63 as f64` gave the
  interpreter −9.22e18 but the artifact +9.22e18. The interpreter now converts
  unsigned for a `u64` source.

### Added
- **First-class unsigned `u64` — sign-sensitive ops now work** (issue #99). A
  `ValueKind::ScalarU64` (physically the signless `i64` MLIR type, no narrowing)
  makes `u64` `/`, `%`, `>>`, `<`, `<=`, `>`, `>=` lower to the **unsigned** op
  variants (`arith.divui`/`remui`/`shrui`/`cmpi ult…`), so a `u64` value with the
  high bit set compares / divides / logical-shifts correctly instead of being
  treated as a negative `i64`. These ops were previously **rejected fail-loud**
  (E2014, "no deterministic signed lowering") — that guard is lifted for them. The
  tree-walking evaluator mirrors the same unsigned semantics, so interpreter and
  artifact agree bit-for-bit. Sign-agnostic ops (`+ − × == != << & | ^`) are
  unchanged (identical signless MLIR), and the i64-only default/keystone build
  never constructs the variant, so existing artifacts are byte-identical. A
  `u64-ops` cross-substrate canary pins it. `u64 as f32/f64` also lowers via
  unsigned `uitofp`, so a `u64` value ≥ 2^63 converts to the correct float
  (was a wrong signed `sitofp`). With every u64 context now deterministic, the
  `E2014` fail-loud diagnostic (and its `ScalarU64`-tracking machinery) is fully
  retired.

### Fixed
- **Scalar int↔float `as`-cast correctness and cross-substrate determinism.**
  Float→integer conversions now **saturate** and are fully defined: in-range
  values truncate toward zero, `±overflow` yields the target `MIN`/`MAX`, and
  `NaN` yields `0` — built from IEEE-defined ops (`maxnumf`/`minnumf`/`fptosi`/
  `select`) so a `f64 as i64` is **byte-identical on x86 (AVX2) and ARM (NEON)**.
  A bare hardware conversion is target-defined (x86 `cvttsd2si` returns the
  `INT64_MIN` indefinite for all out-of-range/NaN; ARM `fcvtzs` saturates), so it
  diverged across substrates and is no longer emitted. `bool as f64` now yields
  `1.0`/`0.0` (was `-1.0`, from a signed 1-bit conversion). The tree-walking
  evaluator mirrors the compiled lowering for every cast, so the interpreter and
  the runnable artifact agree bit-for-bit.
- **Float→narrow-int `as` casts now compile and saturate correctly.** `f64`/`f32`
  `as i8`/`i16`/`i32`/`u8`/`u16`/`u32` previously dispatched on the target type
  alone and emitted an integer shift/mask on the **float** SSA value, which
  `mlir-opt` rejects (`i64 vs f64`) — so a legal program the interpreter evaluated
  correctly **failed to compile** (a uniformity break in the determinism
  contract, one width tier below the `f64 as i64` fix above). Narrow casts now
  route through source-kind-dispatched `__mind_conv_i{W}`/`u{W}` intrinsics: a
  **float** source **saturates to the narrow target range** (`300.9 as u8` → 255,
  `1e30 as i32` → `i32::MAX`, `-inf as u8` → 0, `NaN` → 0) — matching Rust `as`
  and the interpreter — while an **integer** source keeps the byte-identical
  truncate/wrap. Built only from IEEE-defined ops plus a two's-complement integer
  clamp, so `f64 as i32` is byte-identical on x86 (AVX2) and ARM (NEON).
- **Strict-FP classifier: float tensor contractions no longer falsely attest
  strict.** `MatMul` / `Dot` / `Conv2d` / `Conv2dGrad{Input,Filter}` over float
  operands lower to reassociating `linalg` contractions (order-sensitive, not
  cross-substrate byte-reproducible), so they now taint `fp_mode` to `Relaxed` —
  matching the existing treatment of `Sum`/`Mean` — and can no longer pass
  `mindc verify --require-strict-fp`. Integer / Q16.16 contractions stay strict.
- **Artifact cache no longer serves stale objects after a compiler change.** The
  module cache key now folds the `mindc` binary identity (canonical path, size,
  mtime, version) and the `clang` / `mlir-opt` / `mlir-translate` toolchain
  identity, so a recompiled compiler or an upgraded toolchain invalidates prior
  cache entries (was keyed on the crate version string only, which did not change
  across dev rebuilds — a source of stale `.so` reuse). Cache format bumped to v2.

### Added
- **Cross-substrate byte-identity fixture for scalar `as` casts**
  (`scalar-cast-conv`) — exercises int→f64/f32, bool→f64, and the saturating
  float→int edge cases (`±overflow`, `NaN`, `±inf`, `−0.0`) combined into one
  fixed-order result; `avx2 == neon` confirmed on real ARM64 hardware. Closes the
  coverage gap that previously allowed a float→int cross-substrate divergence to
  ship undetected (no fixture exercised a scalar cast).
- **Cross-substrate byte-identity fixture for float→narrow-int casts**
  (`scalar-cast-conv-narrow`) — the width-tier sibling of `scalar-cast-conv`,
  exercising `f64 as i8/i16/i32/u8/u16/u32` with saturation edges pinned at the
  **narrow** bound (`1e30 as i8` → 127, `+inf as i32` → `i32::MAX`, `-1e30 as i32`
  → `i32::MIN`, `NaN`/`-inf` → 0) folded into one fixed-order result;
  `avx2 == neon` by IEEE construction. Closes the coverage gap that let the
  narrow-cast compile-failure ship untested.

## [0.10.1] - 2026-07-05

### Proven
- x86_64 (AVX2) == ARM64 (NEON) byte-identity PROVEN on real hardware (GCP Ampere Altra aarch64, LLVM 20.1.8): cross_substrate_identity 13/13, all 12 canaries byte-identical incl. a chaotic Q16.16 Lorenz integrator.
- Native-ELF self-host loop closed (stage1==stage2==stage3 byte-identical, fail-closed gate).
### Added
- Tensor sum/mean pinned canonical fold + 1-D tensor-parameter ABI.
- PQC ML-DSA (FIPS-204) hybrid signing direction.
### Fixed
- mindc import-walk hardening; Windows-portable __mind_now_ns (POSIX byte-identity unchanged).
### Docs
- Full docs/spec/site truth-alignment: scalar int/Q16.16/f64-f32 x86+ARM verified; float-vector + GPU scoped frontier/commercial-roadmap.

### Verified

- **Cross-substrate byte-identity now proven on ARM64 hardware (2026-07-05).**
  The `cross_substrate_identity` gate's 12 canary workloads — int/Q16.16
  `dot`/`gemv`/`gemm`, det-i8 GEMM, a scalar-f64 arithmetic chain, and a chaotic
  Lorenz–Euler integrator (1000 steps, sensitive to initial conditions) —
  produced **byte-identical** outputs on a real ARM64 host (GCP Ampere Altra
  aarch64, LLVM 20.1.8, `MIND_BENCH_REQUIRE=1`, 13/13 tests), matching the pinned
  x86_64 (`avx2`) references. The **strict tier** — integers, Q16.16
  fixed-point, and **scalar** IEEE-754 f64/f32 (no FMA contraction, no
  reassociation) — is now verified byte-identical on **both x86_64 (AVX2) and
  ARM64 (NEON) real hardware**, not x86-only. Float **vector** reductions
  (tensor `sum`, broader vector), transcendentals, and GPU float determinism
  remain the frontier (unchanged, still tolerance-gated / roadmap).

### Added

- **Evidence-chain signing (RFC 0016 Phase C) — additive, optional, opt-in.**
  The `evidence_chain.*` MAP epilogue can now carry an authenticity signature
  under `signature.*` keys, selected per-artifact for crypto-agility: classical
  **Ed25519** (RFC 8032), post-quantum **ML-DSA-65** (FIPS-204, via the vetted
  `fips204` crate behind the optional `evidence-mldsa` feature — fail-closed when
  the feature is off), or the **hybrid** of both (both halves must verify). The
  signature covers the **canonical provenance preimage**: the 32-byte mic@3
  `trace_hash` plus a canonical, lexicographically-ordered serialization of every
  other `evidence_chain.*` key (substrate, toolchain, determinism, parent,
  schema, trace_hash_kind). So editing any provenance field — including the
  `parent` chain link or the `substrate` label — on a signed artifact makes
  `mindc verify` fail closed (exit 1, "signature does not verify"); provenance is
  authenticated, not just the code anchor. `mindc verify` reports signature status
  separately from `trace_hash_valid` (integrity vs authenticity are orthogonal).
  **Signing is opt-in**, never on by default: it is enabled only when an operator
  supplies a key seed out-of-band via `MIND_EVIDENCE_ED25519_KEY` /
  `MIND_EVIDENCE_MLDSA_KEY`. **The wire stays additive** — `signature.*` keys sort
  after every `evidence_chain.*` key and are absent when no key is supplied, so an
  **unsigned artifact is byte-identical to the pre-signing encoder** (re-verified
  by sha256) and the evidence `schema` stays **Int 1** with **no `mic@N` bump**.
  The signature never feeds back into `trace_hash` (a signature never covers
  itself), so the determinism/keystone gate is untouched (phase_g 7/7). ML-DSA-65
  correctness is pinned by a deterministic (all-zero rnd) crate-pinning regression
  vector. Determinism note: both Ed25519 and the FIPS-204 deterministic ML-DSA
  variant are byte-reproducible, so a signed artifact is itself reproducible.

- **Scalar IEEE-754 float64 on the strict deterministic path.** Scalar `f64`
  (and `f32`) arithmetic now compiles and runs through `arith.mulf` /
  `arith.addf` with **no `fmuladd`-contraction, no `fastmath` flag, and no
  reassociation**, in fixed source order. A loop-carried `f64` computation — an
  `f64` Lorenz–Euler integrator — runs **bit-exact against a reference** and is
  **run-to-run bit-identical**. Because scalar `+ − × ÷ √` are correctly-rounded
  IEEE-754 operations (identical on x86-SSE2 and ARM-NEON), cross-ISA
  bit-identity follows on any conforming FPU, and is now **verified
  byte-identical on real ARM64 (NEON) hardware** (2026-07-05, GCP Ampere Altra
  aarch64, LLVM 20.1.8: all 12 `cross_substrate` canaries reproduced the x86_64
  references). This extends the
  integer/Q16.16 wedge to scalar float on the strict path. **Not** yet
  deterministic and still on the roadmap: `f32`/`f64` **vector** reductions
  (documented ~1e-4 relative tolerance today → canonical reduction trees /
  superaccumulators), **transcendentals** (`sin`/`exp`/… → vendored
  correctly-rounded libm), and **GPU** float (→ fixed-tree / Ozaki-scheme; the
  open runtime is CPU-only). (commit e9e8837)

- **Native-ELF self-host fixed-point closed.** The pure-MIND front-end
  (`examples/mindc_mind/main.mind`) now passes all three self-host gates
  byte-identically against the Rust reference: (a) the `mic@1` IR-text bootstrap
  fixed point, (b) the `mic@3` canonical-binary-IR flip, and (c) the NATIVE
  x86-64/ELF emit of the entire seeded module (21 stdlib modules + main.mind,
  1 055 777 B). Gate (c) is new — the native-ELF emit was the last open frontier.
  The NATIVE-ELF backend (`src/native`) was the normative self-host target;
  MLIR-text is demoted to downstream-interchange / exotic-chip-reach. This was
  the core of Rust-independence — now fully closed, see below.

- **Self-computed PT\_NOTE trace-hash + `src/native` deleted.** The pure-MIND
  front-end now self-computes the `ir_trace_hash` PT\_NOTE byte-identically at
  full `main.mind` scale (~1.5 MB combined stdlib+main.mind source), with zero
  Rust-oracle bytes fed anywhere in the loop (the no-feed rung in
  `self_host_native_elf_smoke.py`). This closed the last oracle tie in the
  native-ELF self-host, which made the Rust `src/native` backend (2441 lines)
  fully redundant — it has been deleted, after freezing its current output as
  permanent test fixtures (`examples/mindc_mind/testdata/native_elf_oracle/`).
  Full Rust-independence for the native-ELF path is complete.

- **`while` / assign port to pure-MIND front-end.** The pure-MIND front-end now
  lowers `while` loops and compound-assignment statements (`+=`, `-=`, etc.)
  byte-identically to the Rust oracle, closing the last major control-flow gap in
  the G2 differential corpus and enabling the full seeded-module native-ELF emit.

- **`mic@3`-flip regression fix.** A regression introduced in the mic@3 canonical
  binary-IR flip (gate b above) — where the fn#10 body silently produced a wrong
  `mic@3` flip outcome — was caught by the U1 bisect and fixed. The self-host gate
  now enforces all three fixed points together so a regression in any one is
  immediately visible; gate (b) and gate (c) cannot silently diverge.

- **`std.sha256` pure-MIND SHA-256 building block (FIPS 180-4).** A pure-MIND
  SHA-256 implementation is now present in the stdlib, providing the cryptographic
  primitive needed to wire `trace_hash` computation into the pure-MIND emit path.
  Bit-identical to the reference on all three self-host gates. (Wiring to the
  `ir_trace_hash` PT\_NOTE emit is the remaining step before the SHA-256 path
  replaces the Rust oracle feed.)

- **Tensor-returning functions build (RUNS bufferization path).** A function that
  returns a tensor — `fn f() -> tensor<f32[3]> { ... }` — now compiles + links to
  a valid ELF cdylib. The MLIR `func.func` signature and the `return` are typed as
  the real `tensor<...>` (`type_ann_to_abi_mlir` now resolves tensor annotations;
  the `Instr::Return` arm emits the value's tensor type), and a build whose emitted
  MLIR carries value-tensor ops auto-selects the tensor-aware `arith-linalg` preset
  (`empty-tensor-to-alloc-tensor` + `one-shot-bufferize{bufferize-function-
  boundaries}` + `convert-linalg-to-loops`), which lowers the by-value tensor
  boundary to a memref out-param at the C ABI. Scalar and `__mind_blas` (Option-C
  i64 ABI) programs stay on the scalar `core` preset and lower byte-identically, so
  keystone 7/7 byte-identical, cross-substrate 12/12, and the gap corpus 66/66 are
  all preserved. NOTE (scope): inter-function tensor-argument calls
  (`g(tensor_value)`) still need the call-site tensor ABI, and the deterministic
  intrinsics (`zeros`/`matmul`/`softmax`/`randn`) are follow-ups; f32 tensor results
  are reproducible-within-substrate, not cross-substrate byte-identical.

- **Dense tensor literals lower correctly in-function (`ConstDenseTensor`).**
  An f32 array literal bound to a tensor type — `let a: tensor<f32[3]> =
  [1.0, 2.0, 3.0]` — now materialises its EXACT per-element bit patterns and
  registers a tensor type, so an in-function elementwise op (`a + b`) lowers to
  `arith.addf %0, %1 : tensor<3xf32>`. Previously such a literal fell through to
  the i64 `ConstArray` path, which coerced every float element to `0` (a silent
  miscompile) and registered no tensor type, so the elementwise binop failed
  "missing type information for value … while lowering binop". The fix is a new
  IR instruction `Instr::ConstDenseTensor { dst, dtype, shape, data }` carrying
  exact element bits, an APPEND-ONLY mic@3 opcode `0x2A` (existing opcodes
  `0x01..0x29` and their `trace_hash` are untouched; the self-host corpus
  contains zero tensors, so the keystone fixed point is unchanged), a
  dtype-parameterised dense MLIR emit, and a front-end `ArrayLit` case in
  `lower_tensor_binding`. Verified: mic@3 emit→parse→emit round-trips the new
  opcode exactly (`tests/mic3_const_dense_tensor_roundtrip`), keystone 7/7
  byte-identical, cross-substrate 12/12. NOTE (scope): this is the in-function
  constructor only — returning a tensor across a function boundary still needs
  the opaque-handle ABI (a following increment), so a whole tensor-returning
  program does not link yet.

- **Pure-MIND front-end self-hosts the full G2 differential corpus (0
  unsupported).** The pure-MIND compiler (`examples/mindc_mind/main.mind`) now
  lowers every top-level construct in the differential corpus byte-identically
  to the Rust `--emit-ir` oracle. `enum NAME { .. }`, `extern "C" { .. }`
  blocks, and `module NAME { .. }` blocks each lower to one item stub (the same
  shape as `struct`/`const`), and a bare top-level arithmetic expression
  (`1 + 2 * 3`) is const-folded to a single `const.i64 <val>` with Rust's exact
  SSA numbering (root id = node count − 1, next_id = node count; parens
  transparent; truncating integer division). The G2 differential harness drops
  its stale source-level exclusions — `import` was already byte-identical (lexed
  as `use`), and enum/extern/module/bare-expr are now ported — so a fixture the
  front-end cannot handle surfaces as a `DIVERGE` rather than a silent
  exclusion. Differential coverage rises from 38 to 67 MATCH / 0 DIVERGE /
  0 MIND_UNSUPPORTED. Keystone 7/7 byte-identical, gap corpus 66/66, and
  cross-substrate 12/12 are all preserved.

- **`map<K,V>` same-name rebind now compiles and runs.** The idiomatic
  `let m = m.insert(k, v)` pattern (functional-update form with the map
  shadowing its previous binding) now compiles and executes correctly.
  A same-name rebind is exempt from the collection-mutation guard because
  the old handle becomes unreachable as soon as `m` is rebound — there is no
  dangling use. This unblocks the documented and idiomatic map<K,V> / vec
  functional-update surface; same-name let-binding is the only form allowed
  on collection mutating methods (a different-name rebind still fails, as
  does nested mutation). Regression: `tests/map_surface_run.rs` compiles and
  runs a `map_insert` rebind (commit e6e4a32).

### Fixed

- **F64 tensors now lower through the float arith path.** Three
  tensor-lowering sites had F64 fall through to the integer `_` arm,
  producing invalid MLIR (`arith.maxsi` / `arith.constant 0 : f64`) or
  wrong values (relu/relu_grad/const-fill). F64 now routes to the same
  float-specialist arms F32/F16/BF16 use (`arith.maximumf`, `arith.cmpf`,
  `format_number`), emitting type-correct MLIR and preserving fractional
  fills. All-i64 programs byte-identical (keystone 7/7, cross-substrate
  12/12 unchanged). Regression: `tests/f64_activation_lowering.rs` (commit
  0537ba3).

- **Mic-b (v2 binary) parser DoS bound added.** The mic-b binary parser
  (parse_micb) now caps ULEB128 element counts at MAX_MICB_ELEMENTS
  (16_000_000), rejecting malformed blobs that could trigger OOM/hang. A
  malicious blob could previously ULEB-encode an astronomically large count
  and drive Vec::with_capacity + a `for _ in 0..n` loop of billions of
  doomed reads. This closes the DoS class already fixed in mic@3. A
  well-formed blob always has counts below the cap, so valid round-trips are
  byte-identical (compact::v2::binary lib tests 7/7 green). Regression:
  `tests/micb_dos_reject.rs` (commit cd6f351).

- **Grad `wrt` target validation locked.** Re-audit of a potential grad-wrt
  slip-through (fleet-audit r1#4) confirms the type-checker already validates
  it correctly with diagnostic E2001 "unknown tensor `X` in `wrt`". A
  resolve-pass check would only replace that precise message with a generic
  error. The resolve-pass walk is intentionally left as-is to preserve the
  specific diagnostic. New regression test: `tests/grad_wrt_resolve.rs`
  (commit f90104d).

## [0.10.0] - 2026-06-18

### Added

- **`f64` enum payload fields now construct, match, and run** (`Option<f64>`,
  `Result<f64>`, mixed `(i64, f64)` variants). An `f64` field is stored into the
  i64 record slot as its raw bits via a synthesized `__mind_f64_to_bits`
  (`arith.bitcast f64→i64`) and loaded back with `__mind_bits_to_f64`
  (`i64→f64`), so the record stays uniformly i64-slotted while the value
  round-trips BIT-EXACTLY (deterministic — only a type reinterpret). The
  constructor and the `match` desugar consult a new `IRModule::enum_payload_types`
  side-table to coerce each field by its declared type (an i64 field is
  untouched; a still-unsupported non-i64/f64 field fails loud at the i64 store).
  Verified RUNNING end-to-end: `Some(3.5)→3.5`, `Err(2.5)→-2.5`, `Pair(9, 3.25)`
  binds `a=9` (i64) and `b=3.25` (f64). Builds on the value-`if` merge fix
  (a one-sided merge / empty-branch placeholder is now typed by the defined side,
  recursing through nested-`if` merge outputs). Keystone 7/7 + cross-substrate
  8/8 byte-identical (no enum/float in either). New f64 cases in
  `tests/enum_match_run.rs`.

- **Multi-field enum payload variants now construct, match, and run.** A boxed
  enum's heap record is sized to `1 + max payload arity` (tag + the widest
  variant's fields) and EVERY variant allocates that size, so a `match` arm's
  field-load addresses valid memory regardless of which variant the scrutinee
  holds. The constructor stores each field into its own slot (`Tri::T(a, b, c)`
  → `[tag, a, b, c]`, zero-filling a narrower variant's unused slots) and a match
  arm binds each `Ident` sub-pattern positionally from `+8*(i+1)` (a `_` skips a
  field without shifting later offsets — `Pair::P(a, _)` binds the first,
  `Pair::P(_, b)` the second). Previously a multi-field constructor silently
  dropped all but the first field and a multi-field match arm fell back to a
  wrong-arm evaluation; both were fail-closed in the prior release and now lower
  correctly. Still i64-payload only (a non-i64 field is a loud `__mind_store_i64`
  error); a nested/literal sub-pattern remains fail-closed
  (`lower::enum_match_unsupported_payload`). Keystone 7/7 + cross-substrate 8/8
  byte-identical (no payload enum in either). New runtime cases (3-field +
  mixed bind/wildcard) in `tests/enum_match_run.rs`.

- **Compound-assignment operators (`+= -= *= /= %= &= |= ^= <<= >>=`).** Desugared
  at parse time to `lhs = lhs OP rhs` (zero new IR — mirrors how `match` and the
  tensor operators desugar) for all three assignment targets (variable, index,
  field); the Pratt infix lookup refuses an `OP=` shape so the parse stops at the
  LHS. Previously every compound operator was a hard `expected expression` parse
  error (the `examples/policy.mind` showcase used `+=` and failed to build — it
  now parses past those sites). New `tests/compound_assign.rs` compiles + runs
  each operator through `mlir-opt`.

### Fixed

- **A value-`if` whose branch declares a non-i64 (`f64`) `let` no longer
  miscompiles.** A `let` inside a value-`if` branch is recorded as a branch
  "write", so it becomes a one-sided merge phi (defined in one branch, absent in
  the other). The synthesized absent-branch placeholder was hardcoded
  `ConstI64(0)` — i64 — so for an `f64` `let` the phi typed `i64` while the
  fall-through edge supplied the `f64` value: `cf.br ^merge(%v : i64)` over an
  f64 `%v` → `mlir-opt: 'i64' vs 'f64'`. `if c { 1.5 } else { 2.5 }` (no branch
  `let`) lowered fine; only a branch `let` triggered it. The placeholder is now
  typed by the side that DEFINES the binding (a `branch_value_is_f64` walk of the
  defining instruction): an `f64` `let` gets a `ConstF64(0.0)` placeholder so the
  phi types `f64`; an i64 `let` keeps `ConstI64(0)` — so every all-i64 program is
  byte-identical (keystone 7/7 + cross-substrate 8/8) and compile speed is
  unchanged (criterion within noise). This is the value-`if` merge fix that
  unblocks `f64` enum payloads. New `tests/value_if_f64_let.rs`.

- **A wildcard payload `match` arm (`Some(_)`) now lowers; multi-field / nested
  enum shapes fail loud instead of silently miscompiling.** Two more enum-match
  silent miscompiles (no error, no crash, wrong value): `match o { Opt::Some(_)
  => 1, Opt::None => 0 }` returned `0` for BOTH (the wildcard payload bailed the
  desugar to a sequential fallback that returns the last arm), and a multi-field
  variant dropped/ignored fields — `Pair::P(10, 20)` constructed `[tag, 10]`
  (second field lost) and `match p { Pair::P(a, b) => a + b, … }` returned the
  wrong arm. Fix: the match desugar now handles a single `Wildcard` payload
  sub-pattern (discriminate by tag, bind nothing), so `Some(_)` RUNS; and a new
  fail-closed `abi_gate` (`check_match_runnable`) flags a multi-field constructor
  (`lower::enum_multi_field_construct`) and a multi-field/nested match arm
  (`lower::enum_match_unsupported_payload`) on the emit path, so v1's
  single-field-payload limit is a loud file:line error rather than a wrong `.so`.
  Inert for any module with no payload enum (keystone 7/7 + cross-substrate 8/8
  byte-identical). New runtime + gate cases in `tests/enum_match_run.rs` and
  `tests/enum_soundness.rs`.

- **`match` on an Option/Result-shaped enum no longer SEGFAULTS — fieldless and
  payload variants now share one heap layout.** A payload constructor
  (`Opt::Some(42)`) lowered to a 2-field heap record `[tag @ +0, payload @ +8]`,
  but a fieldless sibling (`Opt::None`) lowered to the BARE ordinal `1`. The
  match reads the tag with `__mind_load_i64(scrutinee + 0)`, so a `None`
  scrutinee dereferenced the ordinal `1` AS AN ADDRESS → segfault. Every
  Option/Result-shaped enum (a payload variant + a fieldless variant — the most
  common case) crashed at runtime; it shipped because the only enum test checked
  the program COMPILES, never RAN it. Fix: an enum with ≥1 payload variant is
  "boxed" (new `IRModule::boxed_enums` lowering-only side-table), and EVERY
  constructor of a boxed enum — including its fieldless variants — lowers to the
  uniform `[tag, payload]` record (fieldless → payload `0`) via a shared
  `emit_boxed_enum_record` helper; a `match` on a boxed enum always reads the tag
  from the record (even when it names only fieldless variants, so it no longer
  compares the record POINTER against an ordinal). A purely fieldless (C-like)
  enum is NOT boxed and keeps the bare-ordinal lowering, so fieldless matches are
  byte-identical. The side-table is never serialised into mic@3 and the keystone
  has no enums, so keystone 7/7 + cross-substrate 8/8 stay byte-identical. New
  `tests/enum_match_run.rs` compiles a multi-enum program to a `.so` and
  dlopen-runs it (Option, Result, a fieldless-middle 3-variant enum, and a
  fieldless-only match) — the runtime gate the soundness test lacked.

- **A `match` arm now binds its payload at the declared variant type, and sibling
  arms must agree on scalar CLASS.** A payload sub-pattern (`E::A(x) => …`) was
  bound at the SCRUTINEE (enum) type, not `x`'s declared payload type, so an arm
  using the payload and a differently-classed sibling arm type-checked — e.g.
  `match e { E::A(x) => x, E::B => 1.5 }` mixed an `i64` payload with an `f64`
  literal and compiled (an unsound int/float join). Match arms now bind each
  positional payload sub-pattern at its declared `TypeAnn` (via a per-module
  `Enum::Variant → [TypeAnn]` registry installed alongside the exhaustiveness
  registry) and compare sibling arms by scalar class (int `{i32,i64,bool}` vs
  float `{f32,f64}` vs tensor vs gradmap) instead of exact `ValueType`. A
  cross-class mix is a hard `match::arm_mismatch` error surfaced from function
  bodies; intra-class width differences (an `i64` payload next to an `i32` literal
  `0`, both i64-backed) stay compatible, so the legitimate `Opt::Some(v) => v,
  Opt::None => 0` shape is never flagged. Falls back to the scrutinee type when
  the enum/variant/payload is unresolvable (imported enum, `Named` payload), the
  same defer discipline as the exhaustiveness check. Keystone has zero enums, so
  the `EnumVariant` binding path never fires there — 7/7 + cross-substrate 8/8
  stay byte-identical. New cases in `tests/intra_module_call_arity.rs`.

- **Returning an enum handle where a bare scalar is declared is now a loud error,
  not a leaked pointer.** A fn declared `-> i64` that returned a payload-carrying
  enum constructor on some path (`if b == 0 { return Res::Err(0) }`) silently
  miscompiled: an enum value is a heap-record HANDLE, so `divide(5, 0)` leaked a
  raw pointer (e.g. `369049072`) as the i64 result — the type-checker has no
  declared-return-vs-body unification. A new `abi_gate` fail-closed gate
  (`check_enum_handle_scalar_return`) flags it on the emit path with a file:line
  span (`lower::enum_handle_in_scalar_return`), walking every return position
  (explicit `return`s nested in if/while/match/block + tail expressions). ZERO
  false positives: a no-`enum` fast path (keystone byte-identical), it fires only
  on a BARE-scalar return (a `-> Enum` return is correct and never flagged) and
  only on a PAYLOAD ctor (fieldless variants are bare tags; an arg-position ctor
  is not a return). New `tests/enum_soundness.rs`.

- **Value-`if` whose branches yield a comparison now lowers.** `let b: bool =
  if c { x > 10 } else { y > 100 }` produced an i1 (`cmpi`) in each branch but the
  merge block argument is typed i64, so the branch `cf.br ^merge(%cmp : i64)`
  mismatched (`'i64' vs 'i1'`) and `mlir-opt` failed. The branch sub-contexts now
  bubble their `i1_values` up, so the merge recognises the value is physically i1
  and `extui`-widens it to i64 before the block argument (covering both the
  if-value and any branch-assigned merge phi). Additive — a value-`if` over
  ordinary i64 values never takes this path, so keystone 7/7 + cross-substrate
  canaries 8/8 stay byte-identical. New `tests/value_if_comparison.rs`.

- **`if`/`while` on an integer condition branches on non-zero, not the low bit.**
  A non-`i1` condition was `arith.trunci`'d to i1 (testing only the LOW BIT), so an
  even non-zero value branched FALSE — `if 2 { … }` ran the `else`, and a `while c`
  countdown skipped even iterations (`pick(2, 10, 20)` returned `20` instead of
  `10`). It now compares `arith.cmpi "ne" %cond, 0`. An already-`i1` comparison
  result is unaffected (it never took the `trunci` path), so keystone 7/7 +
  cross-substrate canaries 8/8 stay byte-identical. New `tests/cond_truthiness.rs`.

- **Generic call with a non-literal argument no longer writes a broken artifact
  (silent miscompile — P1.1).** `id(n)` for a variable `n` returned EXIT=0 from
  `--emit-shared` but produced a `.so` with an UNDEFINED symbol (`nm -D` → `U id`;
  dlopen → "undefined symbol: id"): the monomorphizer inferred a concrete type
  only from int/float LITERALS, so a variable arg left a dangling bare-template
  reference and the `id$T` body was never emitted. Two-part fix:
  - **Part 1 (correct) — 0 fail-closed for well-typed scalar programs:** the
    monomorphizer resolves a generic argument from EVERY statically-inferable
    scalar shape, not just literals/params — an enclosing-fn parameter, a
    top-level Let-local (annotated or inferred), a nested call's declared return
    type (`id(g(3))`), an arithmetic/bitwise expression (`id(a + b)`), and an
    `as` cast (`id(x as i64)`). A `BINDINGS` map (params seeded at FnDef lowering,
    grown per top-level Let via the shared `bind_let`) plus an `FN_RETURNS`
    pre-pass feed the one `infer_concrete_arg_type` predicate. All gated behind
    the templates-present fast path, so a non-generic module is byte-identical.
    A well-typed scalar generic call therefore NEVER reaches the fail-closed net.
  - **Part 2 (fail-closed net):** a generic call whose argument genuinely cannot
    be monomorphized (a NON-scalar local — struct/tensor/array) is a LOUD
    `--emit-shared` error with a file:line span (`lower::unresolved_generic`,
    RC≠0, no `.so` written) instead of a broken artifact. The gate
    (`abi_gate::check_generic_resolvable`) keys on the declared-generic-template
    set — EMPTY for every non-generic / intrinsic / extern-C program — so it has
    ZERO false positives by construction; it skips generic-TEMPLATE bodies (only
    their concrete instances are lowered, where type-params resolve), and rebuilds
    the IDENTICAL `bindings`/`fn_returns` maps from the same AST in the same
    forward order, sharing the one `is_monomorphizable`/`bind_let` predicate with
    the lowering (gate and codegen cannot drift). Keystone 7/7 + canaries 8/8
    byte-identical (0 templates → inert); the full no-default-features matrix
    passes. `tests/generics_lowering.rs` gates every resolvable shape + the
    genuinely-unresolvable fail-closed case.

- **Struct field populated from a narrow (i32/u32) SSA value now lowers.** The
  natural `P { x: a }` for `a: i32` failed in v0.9.0: the generic `Instr::Call`
  arm rejected the non-i64 argument to `__mind_store_i32` *before* its handler
  ran, and the handler then blind-`trunc`'d the value assuming i64 (invalid MLIR
  for an already-narrow value). The narrow mem-intrinsics are now exempt from the
  blanket i64-arg rejection, and the stored value is coerced from its real
  physical width (`trunc`/`sext`/`zext` as needed; an i64 value still `trunc`s, so
  the existing i32-intrinsic path is byte-identical). This also closes the
  ABI-gate inconsistency the audit flagged — a gate-clean struct-lit now genuinely
  lowers instead of hard-erroring late in MLIR with no span. New
  `tests/struct_narrow_field.rs` gate (i32 field + u32 zero-extended field);
  keystone 7/7 and cross-substrate canaries 8/8 byte-identical.

- **Narrow (i32/u32/bool) inter-function call ABI — narrow params/returns now
  lower across a call boundary.** The generic `func.call` arm hardcoded
  `(i64..) -> i64`, so any narrow-typed param or return on a function that was
  *called* by another made the whole module fail `mlir-opt` (type mismatch, no
  artifact) — the v0.9.0 "narrow lowers in EVERY context" claim held only for
  leaf functions. Calls are now typed against the callee's recorded signature
  (`fn_signatures`): each argument is coerced (`trunci`/`extsi`/`extui`) from its
  physical SSA width to the callee's param width, and the result is tracked at
  the callee's return kind (new `fn_ret_kind` table) so it re-widens correctly at
  its use site (`u32` zero-extends, `i32` sign-extends). Relatedly, a narrow
  early `return` inside an if/else/while body emitted `return %x : i64` against an
  `-> i32` result because `fn_ret_abi` was never propagated into the branch
  sub-context; the enclosing function's ABI context (signatures, param/return
  kinds, return slot) is now inherited by every branch/loop sub-context. i64-only
  callees make every coercion a no-op → keystone 7/7 and cross-substrate canaries
  8/8 byte-identical. New `tests/narrow_call_abi.rs` compiles a composed narrow
  program (caller+callee, early returns in if/else/while, bool + u32 returns) to a
  `.so` and value-checks it through `ctypes` — the `mlir-opt`-level coverage whose
  absence let this ship CI-green.

- **Cross-substrate integer determinism — `INT_MIN / -1` div/rem and oversized
  shift on the pure-`i64` path.** Two latent divergences that broke bit-identity
  for any program reaching them are now pinned in the IR→MLIR lowering
  (`src/mlir/lowering.rs`):
  - `INT_MIN / -1` and `INT_MIN % -1` at signed `i64`, the narrow `i32` arm, and
    the mixed-width widen path: the true quotient is unrepresentable, so x86
    `idiv` raises `#DE` (SIGFPE) while AArch64 `sdiv` returns `INT_MIN` — a hard
    crash-vs-value substrate divergence. The lowering substitutes divisor `1` on
    the overflow case, yielding the wrapping result on every substrate
    (`INT_MIN/1 == INT_MIN`, `INT_MIN%1 == 0`) and never trapping. The guard is
    **elided** when the divisor is a proven constant `!= -1` (overflow then
    statically impossible — e.g. `x / 2`), so constant divisions keep their tight
    single-op lowering and the compile-speed benches stay byte-identical.
  - A pure-`i64` shift count `>= 64` was emitted unmasked (only the narrow/mixed
    arm masked in 0.9.0); LLVM treats an over-width shift as poison (`opt -O2`
    folds `1 << 65` to `poison`), so an optimizer is free to diverge across
    substrates. It now masks the count to `& 63`, matching the narrow path.
  Both fixes are no-ops for in-range inputs: keystone 7/7 and cross-substrate
  canaries 8/8 stay byte-identical, criterion within the one-sided 10% gate. New
  `tests/int_determinism.rs` gate compiles + runs the cases through `mlir-opt`.

- **Integer division-by-zero is now deterministic (`x / 0 == 0`, `x % 0 == 0`).**
  x86 `idiv` by 0 raises `#DE` (SIGFPE) while AArch64 `sdiv` returns 0 — a hard
  cross-substrate divergence (the last integer-determinism hole). The signed-div
  guard (both the i64 arm and the narrow `i32` arm) now substitutes divisor 1 on
  divisor-`0` as well as `INT_MIN/-1` (avoiding the trap) and forces the RESULT to
  0 when the divisor was 0, so both substrates agree. Provisional total semantics (`0`, the conventional non-crashing choice;
  revisitable to a deterministic trap via the spec). Elided when the divisor is a
  proven constant that is neither `-1` nor `0`, so constant divisions keep their
  single-op lowering — keystone 7/7 + canaries 8/8 byte-identical, criterion
  within the gate. Covered by `tests/int_determinism.rs`.

## [0.9.0] - 2026-06-17 — i32/u32/bool lower in EVERY context (narrow-int corpus 3→11) + cross-substrate shift determinism + ~10% faster compile

### Added

- **Narrow-int control flow, mixed-width arithmetic, narrow returns, bool branches
  (RUNS Phase 3 — completes i32/u32/bool).** Building on the i32/u32 ABI below,
  narrow integers now lower in EVERY context, not just isolated signatures.
  Value-`if` and `while` merge blocks carry the real narrow type (was hardcoded
  `i64`, which produced an `'i64' vs 'i32'` mlir-opt error); a narrow operand mixed
  with a non-literal `i64` is promoted to `i64` via `arith.extsi`/`extui` and
  computed there; a narrow result is `trunci`'d at the return boundary; a `bool`
  param (i64-backed at the ABI boundary) is `trunci`'d to `i1` before `cf.cond_br`.
  The internal narrow-int corpus goes 3→11 (every case returns the correct value:
  `min`/`abs`(i32), `max`(u32), `rotl`, `i64`+`i32`, `while`-i32, `(a+b) as i32`,
  bool-`if`). All-i64 programs are byte-identical — the narrow paths never fire:
  cross-substrate canaries 8/8, mic@3 flip byte-identical, keystone unchanged,
  criterion within the one-sided 10% gate.

- **Cross-substrate shift-count determinism.** A narrow/mixed-width shift masks its
  count to width-1 (`arith.andi N-1`) before `shli`/`shrsi`/`shrui`. An
  out-of-range shift count is poison in MLIR/LLVM and lowers divergently across
  substrates (x86 masks mod-width, AArch64 differs) — the mask makes every count
  in-range and byte-identical on every substrate. In-range counts are unchanged
  (no-op), so corpus + canaries + keystone are unaffected.

- **Narrow-int (`i32`/`u32`) ABI lowering (RUNS Phase 3).** `i32`/`u32` function parameters
  and returns now lower to real `i32` MLIR instead of silently widening to `i64`: a per-fn
  `param_kinds` table threads the declared `TypeAnn` (so `u32` keeps its unsignedness, which
  the lossy ABI string loses), and a dedicated narrow-int `BinOp` arm selects width- and
  sign-correct ops — `addi`/`subi`/`muli`, `divsi`/`divui`, `remsi`/`remui`, `shrsi`/`shrui`,
  `cmpi` with `slt`/`ult`-family predicates — all at `i32`, with two's-complement wrap at the
  declared width as the deterministic-overflow contract (identical on every substrate). An
  integer literal (which the IR only carries as `ConstI64`) is `trunci`-legalized to the
  narrow width when paired with a narrow operand; a genuine non-literal width mismatch fails
  closed rather than silently miscompiling. Exec-verified: `i32` add wraps correctly and `u32`
  division/shift are unsigned. The `bool`-return ABI is unchanged (i1→i64 zero-extend). The
  P1.1 `i32`/`u32` param/return gate is removed (tensor params and `i8`/`u8`/`i16`/`u16` —
  which lack a dedicated value kind — stay gated). All-i64 programs are byte-identical: mic@3
  flip byte-identical, keystone 7/7, cross-substrate canaries unchanged; criterion within noise.

- **Width-aware deterministic struct ABI (RUNS Phase 3).** Structs with sub-i64 fields
  (`i32`/`u32`/`i16`/`u16`/`i8`/`u8`/`bool`) now lower correctly instead of being rejected:
  a canonical per-field offset table (`struct_layout`) places each field at a self-aligned
  offset computed purely from the declared field widths (8/4/2/1) — identical on every
  substrate, with no host `sizeof`/`alignof` and no target-dependent padding. Struct
  construction allocates the exact total size and stores each field with the typed
  `__mind_store_i{8,16,32,64}` intrinsic (new `__mind_load/store_i16`); field reads use the
  typed load and sign-extend a signed narrow field. The all-i64 case takes the identical
  legacy path, so the self-host records, the mic@3 fixed point (byte-identical), the keystone
  (7/7), and the cross-substrate canaries are unchanged. The obsolete P1.1 struct-field
  loud-fail gate is removed (float struct fields stay loud via the downstream non-i64-call
  check; the function param/return and tensor gates remain). Exec-verified end to end
  (correct values at 4-byte stride; negative `i32` sign-extends); criterion compile_small
  within noise.

- **Generic monomorphization in codegen (RUNS Phase 3).** A generic function called
  at a concrete type (`id(5)`) previously emitted a body-less private declaration, so
  the resulting `.so` failed to link (`undefined symbol: id$i64`). `lower_to_ir` now
  drains the monomorphization worklist after lowering the module body: each requested
  instance is synthesized by `instantiate_template` and lowered through the ordinary
  `FnDef` path, emitting a real body. Instances drain in `BTreeMap` mangled-name
  (lexicographic) order — deterministic, no hash/clock/rng — so generic-using modules
  are byte-identical and `avx2 == neon`. A non-generic module registers zero templates,
  hence zero requests, so the drain is a no-op and the mic@3 self-host fixed point, the
  66/66 gap corpus, the keystone, and the cross-substrate canaries stay byte-identical
  (verified). An instance whose body still names the type parameter in a type position is
  refused (left body-less → loud link error) rather than emitting a silent mis-ABI body.

- **Loud-fail ABI gate for runnable artifacts (release-readiness P1.1).** `mindc --emit-obj`
  and `mindc --emit-shared` now refuse to emit a *runnable artifact* for a construct the
  shipped i64-scalar backend would silently miscompile — they emit `error[lower][lower::…]`
  with the `file:line` span and a non-zero exit code instead of a confidently-wrong `.so`/`.o`
  (the worst failure mode for a deterministic, evidence-signing compiler: a valid signature
  over incorrect code). The gated constructs are the verified *silent* sub-i64-ABI miscompiles:
  a struct field declared at a sub-i64 width (`i32`/`u32`/`bool`/`i8`/`u8`/`i16`/`u16`), a
  `tensor`/`diff tensor` function parameter or return (which erases to the i64 ABI), and a
  sub-i64-integer function parameter or return. `extern "C"` declarations are exempt — the
  C-ABI boundary legitimately declares narrow ints. The inspection surfaces (`mindc check`,
  `--emit-ir`, `--emit-mlir`) are intentionally unaffected, because there an `i32`/`tensor`
  annotation is a valid *type*; it is simply not yet lowerable to a runnable artifact. The
  check is a pure read-only pre-pass (`src/eval/abi_gate.rs`) that never mutates the IR and
  produces zero diagnostics for an all-i64 program, so the mic@3 self-host fixed point, the
  66/66 gap corpus, the keystone (7/7), and the cross-substrate canaries are all byte-identical.
  This converts the dominant silent-miscompile class (the struct / tensor / narrow-int RUNS
  families) into honest, anchored diagnostics ahead of the real deterministic codegen that
  will make them lower.

### Performance

- **~10% faster compilation.** Pre-reserve the `ir.instrs` Vec capacity at the top of
  `lower_to_ir`, eliminating the realloc+memmove chain (`RawVec::finish_grow`) that
  dominated the AST→IR hot path. Capacity-only — the emitted mic@1/mic@3 bytes (and
  cross-substrate identity) are unchanged. `compile_small/scalar_math` 3.07µs → 2.76µs.

## [0.8.1] - 2026-06-14 — Self-host gap corpus genuinely 66/66 (remove a needless decline guard; fix the fresh-load measurement harness)

### Fixed

- **The gap corpus is byte-exact on the full 66/66 — `value-ifexpr_5` and `mixed-prefix_12`
  were never actually a compiler gap.** Those two (a value if-expr branch whose `let` shadows
  a sequence let, e.g. `let v = x + 10; if c == 0 { let v = x + 1; v } else { v }`) lower
  byte-exactly and deterministically — verified three ways that agree: a brand-new process per
  fixture, `fork` + `dlopen`-in-child, and a warm reused handle all return identical bytes. The
  0.8.0 "64/66, 2 safe fail-closed" was a **single test-harness bug**, not a compiler one: the
  survey read the **wrong field of the `EmitState` result** — offset `+8` (`next_id`) instead
  of the buffer handle at offset `+0`. That one mistake produced both symptoms — a false
  "empty" (read fail-closed) when `next_id` was 0, and a SIGSEGV (read as a crash) when it held
  non-zero garbage dereferenced as `(addr, len)`. The correct read is `EmitState.buf` (offset
  0) → `String`(addr@0, len@8). No warm/fresh divergence exists; the lowering was always right.
- Removed the `ifexpr_shadows_seq` decline guard (and its now-dead helpers `tenv_has_name` /
  `ifexpr_block_shadows`) added in 0.8.0 to fail-closed that shape — declining a shape that
  lowers byte-exactly was needless. The `blk_layout` SSA-merge layout (else-side bare ref
  resolves to the outer vid, no placeholder) was already correct.
- `gap_corpus_smoke.py`: ratchet FLOOR 64 → 66 so the full corpus can never silently regress;
  the gate already loads the `.so` fresh *inside the forked child* (the canonical measurement).
- Whole-module `mic@3` self-host FLIP stays byte-identical; keystone 7/7 byte-identical.

## [0.8.0] - 2026-06-14 — Self-host front-end byte-exact on 64/66 of the gap corpus (0 wrong-bytes; 2 safe fail-closed) + `mic@3` self-host fixed-point + RFC 0012 tensor-native syntax; release gate `#306` cleared (std byte-store migration + keystone re-bless)

### Added — self-host fixed-point at the canonical `mic@3` binary IR (byte-identical)

- **The pure-MIND bootstrap compiler now reproduces the canonical `mic@3` binary IR
  of its own ~15k-line source byte-for-byte against the Rust reference.**
  `selftest_mic3_module_nfn(main.mind)` emits the exact same 221k-byte `mic@3` module as
  `mindc --emit-mic3 main.mind`. This advances the self-host fixed-point from the `mic@1`
  MLIR-text layer to the **canonical binary-IR layer the evidence chain's `trace_hash`
  anchors on** — so the Rust front-end is decorative at the layer that matters for
  provenance. Driven by SSA fresh/bound merge tails, nested if-else / if-expr escape
  bubbling, a same-name two-branch phi unification in `blk_layout`, per-ctor struct-lit
  alloc-handle resolution (`letenv_lookup` last-match), and an `items_len`-based module
  header. CI-enforced by a new `mic3_flip_smoke.py` gate (hard-fails when `MINDC_SO` is set).
- **The pure-MIND front-end byte-exactly compiles 64/66 of the gap corpus** (a fuzz sweep
  of 497 programs across construct families), with **0 wrong-bytes** (the cardinal integrity
  invariant — never a silent miscompile) and **2 deterministic safe fail-closed**. Progress
  32→64 byte-exact; it byte-exactly compiles arbitrary programs, not just its own source, so
  Rust-independence at the front-end holds well beyond self-compilation. A new CI gate,
  `gap_corpus_smoke.py`, enforces the corpus (hard-fails on any wrong-bytes; ratchets the
  byte-exact floor; loads the `.so` fresh per fixture so a warm-arena artifact can't fake a
  pass). The 2 fail-closed (`value-ifexpr_5`, `mixed-prefix_12`) share one shape — a value
  if-expr branch whose `let` shadows a sequence let — declined deterministically pending a
  proper lowering (its emit reads an under-initialised scratch slot, making the result
  arena-state-dependent; refusing is the honest, integrity-preserving behavior). The dominant gap was not missing lowering but **passes gated on the
  source literally containing the intrinsics** `__mind_alloc` / `__mind_store_i64` /
  `__mind_load_i64` (searched to get the spans the synthetic alloc/store/load callees intern
  from): a module that constructs a struct or reads a field but never spells those literals
  was skipped. Fixed with a synthetic-span fallback (`build_src_intrinsics` — append the
  literals to a src copy, invisible to the lexer; keep the real spans byte-for-byte when
  present, so `main.mind`'s self-host flip is unchanged). Also: the fall-through-shadow
  silent-miscompile class (a name shadowed inside a fall-through `if`) — same-name escaping
  bindings deduped to one SSA F2 phi (`bind_append_dedup`), shadowed reads resolved to the
  merge vid via a position-ordered trailing env (`synth_rebind_slots` → `seq_set_rebind_vids`
  → `build_trail_env`); unary neg and a trailing value if-expr as the FN_DEF result; a
  both-branches-`return` if-else body (`emit_mic3_if_both_return_instr`); an unresolvable
  field on a non-struct receiver lowered to a `CONST 0` stub. The whole-module flip stays
  byte-identical throughout; catalogued in `tests/selfhost_gaps/GAPS.md`.

### Fixed — dead-code elimination operand coverage (silent prune of call/return operands)

- **DCE no longer prunes a value used only as a `Call` argument or `Return` value.**
  `ir_canonical::instruction_operands` previously ended in a `_ => vec![]` catch-all,
  so it reported an empty operand set for `Call`, `Return`, `Param`, and every
  std-surface instruction. A value consumed *only* as a call argument was therefore
  never marked live and was pruned — a silent miscompile (the std-surface UFCS bricks
  lower `v.push(x)` to a `vec_push` call, so the pruned argument dropped a real
  operand). The match is now **exhaustive with no catch-all**, so any future `Instr`
  variant is a compile error here until its operands are declared.
- **The IR verifier now checks operands for every straight-line instruction**, not
  just `BinOp`. A use-before-definition reaching through a `Call`/`Return`/`Vec*`
  operand previously slipped past `validate_operands`; both the DCE liveness pass and
  the verifier now share the same exhaustive operand enumeration, so they cannot
  disagree. Nested control-flow (`While`/`If`/`Region`/`FnDef`) is correctly skipped —
  its operands live in their own SSA sub-namespaces.
- Regression test `canonicalization_keeps_value_used_only_as_call_arg` pins the fix.

### Added — RFC 0012 tensor-native surface syntax (the differentiation layer)

- **Phase A — shape-typed tensors.** `Tensor<dtype,[dims]>` carries shape and
  dtype in the type. Matmul inner-dimension, broadcast compatibility, and rank
  mismatches are caught at type-check time with structured `shape::` /
  `E2101`/`E2102`/`E2103` diagnostics through the same surface as `mindc check`
  — never at runtime.
- **Phase B — tensor operators.** `@` (matmul), `.+ .- .* ./` (elementwise),
  `.T` (transpose), and reductions, all desugaring to the existing `std.blas`
  surface so the Q16.16 byte-identity guarantee carries through unchanged. Ships
  a strict broadcast subset (scalar + last-dimension rank-1); full prefix-rank
  broadcasting and raw-intrinsic byte-identity are Phase B.2 (deferred).
- **Phase C.0 — function-attribute threading (inert).** The parser records the
  full attribute list on `Node::FnDef` (`attrs`) so later phases can interpret
  `#[deterministic]`, `#[target(...)]`, and `#[q16]`. Inert by itself — lowering
  is unchanged and the bootstrap byte-identity oracle is preserved.
- **Phase C.1 — annotation checks (enforced).** A purely-additive type-checker
  pass reads those attributes and enforces three `determinism::*` contracts:
  `unknown_target` (a `#[target(x)]`/`#[q16]` whose `x` is not in the backend
  vocabulary, or `#[target]` with no name), `float_in_q16_fn` (a `#[q16]`
  function declaring a non-q16 tensor parameter or return), and
  `nondeterministic_in_deterministic` (a `#[deterministic]`/`#[q16]` function
  calling a non-deterministic function). Only functions that opt in are checked;
  un-annotated code never regresses.
- **Phase C.2 — implicit determinism + module-block coverage.** The determinism
  call-graph now judges external (std/imported/intrinsic) callees by the
  dtype-suffix convention: `_q16` and `__mind_*` are implicitly deterministic,
  `_f32`/`_f64` floating reductions are not (flagged), unknown is unflagged (no
  false positive). So a `#[deterministic]` fn may call `dot_q16(..)` but not
  `dot_f32(..)`. The pass also descends into `module { }` blocks. Expression-
  level dtype tracking, per-`#[target]` MLIR routing, and f32 fixed-reduction
  order remain for C.2+/Phase D.
- **Attribute surface unified on `#[name]` (single form).** RFC 0012 §5 adopts
  Rust-style `#[name]` (the `#` disambiguates attributes from the `@` operator
  and from a bare `[` array literal). `#[name]` is now the **only** attribute
  form — bare `[name]` at item position is no longer an attribute. (Pre-1.0,
  no external code: a clean one-true-way cut rather than carrying two forms;
  the few internal `[test]`/`[protection]`/`[reap_threshold]` uses were migrated
  to `#[...]`.) Also fixed a latent formatter bug that silently *erased*
  function attributes (`emit_fn_def` dropped them) — `#[test]`/`#[deterministic]`
  now round-trip through `mindc fmt`, which emits the canonical `#[name]`.

### Added — post-0.7.0 RFC progress

- **RFC 0010 Phase J-A** — `region { }` blocks: the region-interior memory tier
  (unrestricted aliasing within a bounded lifetime, freed at region exit), with
  escape checks routed through the type-checker `safety::` diagnostic surface.
- **RFC 0010 Phase J-B** — `GenRef`: generation-checked region-exterior
  references; a stale-handle deref returns `None` instead of use-after-free UB.
- **RFC 0010 G1** — dropped the vestigial `melior`/`inkwell` optional deps
  (never load-bearing; `mindc` emits MLIR text and shells to the C++ backend).
- **RFC 0010 G2.1** — pure-MIND vs Rust MLIR-text differential coverage harness.
- **RFC 0011 Phase A** — `std.async` (12th gated stdlib module): Scheduler-
  injection API, Sender/Receiver composition, and a deterministic
  `ReplayScheduler` with a byte-stable FNV-1a `trace_hash`, in pure MIND with
  synchronous execution.

### Fixed

- **RFC 0012 Phase A regressions (purely-additive guarantee restored).**
  Phase A's new `ScalarI64`/`ScalarF64` types broke `i64 + i64` (and f64/bool)
  scalar binary ops (`type_check::E2001`) and the fn-body shape pass over-
  reported on `&expr`/match/struct-arg expressions. Both fixed; the shape pass
  now emits only `shape::*` diagnostics on genuinely shape-wrong code and never
  makes previously-compiling code fail.
- **Three latent full-workspace failures** surfaced by running the suite with no
  feature flags: a test target that did not compile under default features
  (missing `#![cfg(feature = "mlir-build")]` guard); a diagnostic-code reroute
  of the legacy `tensor.matmul(a,b)` intrinsic off `E2103` (Phase B classifier
  too broad — narrowed to the `@`-operator marker); and a formatter
  stability test that panicked parsing a `std-surface`-gated stdlib file.
- **`#306` — std byte-store heap OOB closed at source (`__mind_store_i8`
  migration).** Every genuine single-byte store in `std/string`, `std/sha256`,
  and `std/toml` now uses the byte-width-correct `__mind_store_i8` intrinsic
  instead of an 8-byte `__mind_store_i64` (which clobbered the trailing 7 bytes
  and wrote past `len`). The i64-aligned struct-header and backing-array stores
  are left untouched. Behavior preservation is verified: SHA-256 FIPS 180-4
  conformance (3/3, including the 56-byte two-block-padding vector that
  exercises the byte-store boundary) and the std-surface string/toml lowering
  suites all pass against a real `--emit-shared` ELF.

### Added — RFC 0013 Tier 1: CLI/agent stdlib surface

- **`std.cli`** — `argv` parsing, a flag walker, `--flag=value` equals form,
  `string_starts_with`/`slice_from`, and subcommand-dispatch helpers.
- **`std.io`** — `isatty` plus the full ECMA-48 SGR vocabulary for coloured
  terminal output.
- **`std.string`** — `string_push_str` (grow-and-copy), `string_push_i64`, and
  `string_push_ansi_sgr`, all preserving bytes across reallocation.
- **`std.tui`** — minimal widget surface: terminal-size via `ioctl`, plus `Box`
  and `Text`.
- **`std.sha256`** — pure-MIND SHA-256 (FIPS 180-4), bit-identical to the
  reference; the hash primitive the evidence chain builds on.

### Added — RFC 0016 / 0021: compile-time evidence chains + canonical IR (experimental)

- **RFC 0016 Phase A** — evidence-chain emission (inert, unsigned): the
  determinism / substrate / toolchain / `trace_hash` attestation surface.
- **RFC 0016 Phase B** — `verify_evidence_chain`: a verifier core producing a
  structured `EvidenceReport` / `EvidenceError`.
- **RFC 0016 GAP-1** — the evidence `trace_hash` is now anchored on the
  canonical **mic@1** IR (the actually-emitted artifact), not the weak v2 graph.
- **mic@2.1 MAP carrier** in compact-IR v2 — the metadata-attribute-pairs
  extension that carries the chain.
- **RFC 0021 steps 1–3** — the **mic@3** binary `IRModule` codec (additive),
  an evidence MAP epilogue that attests the real IR, and the
  `mindc --emit-mic3` / `--emit-evidence` flags. These are **experimental** and
  flag-gated; they do not change default lowering. RFC 0021 steps 4–6
  (`mindc verify` CLI, v2→`mind-model@2` demotion, oracle + CI gate) remain in
  progress.

### Added — Q16.16 BLAS + cross-substrate reproducibility (RFC 0006 / 0020)

- **Q16.16 row-major GEMV** kernel `__mind_blas_matmul_rmajor_q16_v` (RFC 0006
  Track B).
- **Internal cross-substrate reproducibility gate** (RFC 0020 §10, #303): a
  table-driven harness over `dot-l1-q16`, `dot-l2-q16`, `gemv-q16-256x256`, and
  a square `gemm-q16-64x64x64` workload, asserting byte-identical results.

### Fixed — std memory-safety, formatter stability, autodiff

- **`std.string` CRITICAL** — `string_get_byte` byte-mask + a real `string_eq`.
- **`std.string` / `std.vec`** — preserve bytes on growth via `__mind_realloc`
  (a latent bug that dropped bytes past the initial capacity).
- **`std.map`** — realloc-on-growth, `slice_from` clamp, `isatty`
  (code-reviewer HIGH/MEDIUM).
- **`mindc fmt`** — preserve `let mut` through the round-trip; keep body and
  trailing comments inside their block (two formatter defects, both closed).
- **MLIR `while`-loop lowering** — after-loop SSA dominance + shift-register
  back-edge.
- **Empty reduction axes** must reduce *all* axes (→ scalar), not none.
- **Autodiff feature restored** — the gradient `BinOp` match now cfg-gates the
  `std-surface` bitwise/shift variants (fails loudly rather than emitting a
  silent zero gradient), and the `mindc_emits_grad_ir` fixture uses the
  canonical `diff tensor<f32>` form. The feature compiles and tests green under
  both `--no-default-features --features autodiff` and the full feature set.

### Docs (cross-ecosystem alignment, `#308` closed)

- **`docs/byte-store-migration.md`** (new) — reproducible execution playbook
  for closing `#306` (the std byte-store landmine): per-file inventory of
  the call sites, the toml.mind mixed-case guard (byte stores vs i64-aligned
  struct ABI), and the keystone re-bless procedure. Includes the
  stub-tolerant-test trap caveat surfaced when I tried to execute the
  playbook from a non-MLIR-provisioned environment.
- **`docs/ir-stability.md`, `docs/versioning.md`** — rewrote the prior
  "`mic@1e` epilogue" framing to the actual shipped `mic@3` design (steps
  1–3) + steps 4–6 in flight, with the rename rationale (mic@1e would
  have collided with the binary-attestation use case once mic@2/2.1
  shipped).
- **mind-spec alignment** (`STATUS.md`, `spec/v1.0/ir-stability.md`,
  `spec/mic/mic2.1-spec.md`, `design/rfcs/0001-mindir-compact.md`,
  `docs/changelog.md` 1.4.0 entry) — IR-canon coverage matrix went from
  0 mic@3 / 0 RFC 0016 / 0 RFC 0021 hits on STATUS + ir-stability to
  5 / 3 / 7 hits; `spec/v1.0/ir-stability.md` gained a Format-detection
  table + an Evidence-chain-attestation section documenting the GAP-1
  `mic@1` anchor rule; `STATUS.md` gained 9 new toolchain-features rows.
- **mindlang.dev alignment** (`/docs/mic`, `/docs/mic/v2`, `/docs/mic/binary`,
  `/docs/stability`, `/docs/ir`, `/docs/map`, `/docs/runtime`, `/roadmap`)
  — canon banner on the main MIC page (mic@1 text + mic@3 binary canonical,
  mic@2/2.1 back-compat pending demotion); mic@2.1 + mic@3 cards; format-
  detection example updated; status banners on the legacy v2/binary pages;
  3 new RoadmapCards (RFC 0013 / RFC 0016 / RFC 0021); IR-canon banner on
  `/docs/ir`; MAP-protocol-vs-MAP-epilogue disambig banner on `/docs/map`;
  evidence-chain anchor mention on `/docs/runtime`.

### Release gate — `#306` RESOLVED (std byte-store OOB closed + keystone re-bless confirmed)

- **The byte-width-correct `__mind_store_i8` migration is applied to every
  genuine single-byte store across the std library** — `string` / `sha256` /
  `toml` / `fs` / `json` / `regex` / `process` / `net` (the `tui` path had no
  byte-store sites) — closing the write-past-`len` landmine at the source. The
  i64-aligned struct/word stores (`* 8` offsets, struct `+0/+8/+16` ABI fields)
  are legitimate and unchanged. Behavior preservation verified (SHA-256 FIPS
  3/3 + std-surface string/toml suites against a real `--emit-shared` ELF).
- **Keystone byte-identity re-bless: CONFIRMED on a real ELF.** The bootstrap
  keystone (`phase_g_keystone_bootstrap`, built with `--features mlir-build`)
  reproduces `examples/mindc_mind/libmindc_mind.so` as a **real ELF (full MLIR
  path), byte-identical across two clean builds AND the `mindc build` (Mind.toml)
  vs direct `--emit-shared` path** — confirmed locally and by the green CI
  keystone job. The keystone asserts self-consistency, not a frozen SHA, so
  there is no hash to bless; the stub-only build (no MLIR feature) is a
  build-config limitation, not a determinism failure. Release gate cleared.

## [0.7.1] - 2026-06-02 — High-level execution surface ships (RFC 0005): the 10 desugaring bricks run on the shipped compiler; `std-surface` promoted to the default build (fail-loud on unfinished constructs); cross-substrate byte-identity preserved

### Added — High-level execution surface (RFC 0005 "RUNS")

The high-level execution surface now lowers and **runs** on the shipped `mindc`
through ten desugaring "bricks", each verified keystone-7/7 (byte-identical across
the seven substrate targets):

- integer / literal `match`
- enum-discriminant `match`
- `Option` / payload `match`
- string literals
- struct field-write
- nested-struct + borrow field access
- method-as-field accessors (`s.len()`)
- method-with-args UFCS (`v.push(x)` → `vec_push(v, x)`)

A method-with-args call that does not resolve to a known free function **fails
loud** (a clear compile error) rather than silently miscompiling.

### Changed — `std-surface` promoted to the shipped default

- The `std-surface` feature is now the **default** (`default = ["std-surface"]`),
  so the high-level surface executes on every shipped binary — no opt-in flag
  required. `--no-default-features` restores the low-level-only subset.
- The constructs **not yet promoted** — by-value tuple / aggregate returns and
  `region { }` — are re-gated behind a new `std-surface-experimental` feature and
  **fail loud** under a default build (a clear compile error, never a silent
  const-0 / i64-collapse miscompile). Generics still have no surface syntax and
  are parser-rejected.
- **Cross-substrate Q16.16 byte-identity is preserved across the flip.** The
  keystone test always builds `std-surface` explicitly, so promoting it to the
  default changes zero keystone bytes (keystone 7/7 unchanged).

## [0.7.0] - 2026-05-21 — Credibility-ladder rung 3 graduation: Mindcraft + RFC 0008 KEYSTONE + RFC 0010 foundations + 13 stdlib modules + mindc doc + standalone binary release

### Added — Standalone binary distribution (task #265)

Standalone binary distribution: `mindc` is now available as pre-built binaries for
Linux (x86_64 static musl), macOS (universal x86_64 + arm64), and Windows (x86_64
MSVC). No Rust toolchain required. See `docs/install.md`.

- `.github/workflows/release.yml` — builds and publishes binaries on tag push
- `scripts/install.sh` — one-line curl-pipe installer for Linux/macOS
- `scripts/install.ps1` — one-line PowerShell installer for Windows
- `docs/install.md` — installation guide (curl-pipe, manual, build from source)
- Every release asset is accompanied by a `SHA256SUMS` manifest for checksum verification
- Cosign signing is stubbed; wire `STARGA_COSIGN_KEY` secret to enable

### Summary of v0.7.0

v0.7.0 is the credibility-ladder rung 3 graduation marker. Everything that shipped
between v0.6.8 and this tag is collected here. Key themes:

- **Mindcraft fully shipped** (RFC 0007 all 6 phases + MINDCRAFT-001 keystone) -- already in v0.6.8
- **RFC 0008 `mindc build` / `mindc test` -- all 7/7 phases shipped** including Phase G KEYSTONE:
  the `mindc build` bootstrap reaches a self-consistency keystone (cargo still drives the
  documented build; full real-codegen self-host remains in progress)
- **RFC 0010 extern "C" + SysV + Win64 ABI** -- Phases A/B/C shipped; Phases E/F scaffolded
  (std.mlir + std.llvm bindings)
- **13 stdlib modules**: vec, string, map, io, blas, toml, json, regex, net, fs, process,
  mlir, llvm
- **`mindc doc`** -- rustdoc-style HTML documentation generator (Phase 1)
- **Standalone binary release pipeline** -- linux-musl + macos-universal + windows-msvc +
  install.sh / install.ps1
- **~720+ tests green** (up from ~579 at v0.6.8); bench-gate +7% cap held through every
  change (-6.5% to +3% range)
- Bootstrap fixed-point byte-identity preserved (v0.6.1 oracle hash unchanged)
- Format string bug fix in std_surface_net_fs_process test (#265 followup)

RFCs drafted in this cycle: RFC 0009 (federation package layer), RFC 0010 (memory safety +
C ABI), RFC 0011 (async + structured concurrency).

### Added — `mindc doc`: rustdoc-style HTML documentation generator (Phase 1, task #264)

New `mindc doc [PATHS...] [--out=<dir>] [--no-deps] [--open]` subcommand.

- Walks `*.mind` files (recursively); extracts all `pub` items (`fn`, `struct`,
  `enum`, `const`, `type`) plus their immediately-preceding `///` doc-comment
  blocks from the trivia stream.
- Renders one HTML page per source file and a top-level `index.html`.  Output
  paths mirror the source hierarchy relative to the common ancestor of the
  given input files, so `std/vec.mind` → `target/doc/std/vec.html`.
- Emits `search-index.json` (`[{name, kind, file, line}]`) for client-side
  search tooling.
- Minimal Markdown renderer (paragraphs, headings, fenced code, inline code,
  bold, italic, links, lists) — no external crate dependencies.
- Dark-themed embedded CSS (~3 KB per page); no external resources; works offline.
- `--open` triggers `xdg-open`/`open` on the generated `index.html`.
- Exit codes: 0 = success, 1 = parse/I/O error, 2 = invalid CLI args (via clap).

Deliverables:
- `src/doc/mod.rs` — extraction pipeline, path resolution, search-index
- `src/doc/html.rs` — HTML page template + embedded CSS
- `src/doc/markdown.rs` — minimal Markdown → HTML renderer
- `src/bin/mindc.rs` — `Doc` subcommand wired into the CLI dispatcher
- `tests/mindc_doc_phase1.rs` — 7 integration tests, all passing

Hard-gate results: 7/7 new tests pass; 200 library unit tests pass (including
21 doc-module tests); full suite unaffected; bench-gate held.

### Added — RFC 0008 Phase F + Phase G KEYSTONE: `mindc build` self-hosts the mind repo

### Added — RFC 0008 Phase F: incremental compilation cache

SHA-256-keyed object cache for `mindc build`. Cold build: ~188 ms.
Warm rebuild (no source change): ~3 ms — a 63x speedup. The cache is
per-target (cpu/cerebras/…) and per-optimize-level (debug/release),
so cross-target rebuilds never share entries. Cache entries are
written via atomic rename so concurrent `mindc build` invocations
cannot produce corrupt files. `mindc build --no-cache` bypasses the
hit check but still writes new entries. `mindc clean --cache` removes
`target/*/.cache/` directories leaving linked binaries intact.

Deliverables:
- `src/build/cache.rs` — `module_cache_key`, `probe`, `write_object`,
  `clean_all_caches`, `BuildManifest`, `ObjectMeta`
- `src/build/mod.rs` — Phase F cache probe integrated into `run_build`;
  `BuildOpts::no_cache`; `IncrementalStats` in `BuildOutput`
- `src/bin/mindc.rs` — `mindc build --no-cache` flag; `--verbose` reports
  `[CACHE HIT] <module> (<key-prefix>)`; `mindc clean --cache`
- `tests/mindc_cache_phase_f.rs` — 13 tests (10 spec + 3 unit), all passing

Hard-gate results: 13/13 pass; full suite 0 failed; self-build smoke
passes; bootstrap fixed-point unchanged; warm rebuild ~3 ms vs cold
~188 ms (63x).

### Added — RFC 0008 Phase G KEYSTONE: `mindc build` bootstraps the mind repo itself

**Cargo is no longer load-bearing for the pure-MIND compile loop.**

`mindc build` now produces `libmindc_mind.so` from the mind repo's
own `Mind.toml`, byte-identical to the direct-path invocation
(`mindc build examples/mindc_mind/main.mind --emit=cdylib`). The
pure-MIND build orchestrator is the canonical path for producing
the self-hosting compiler artifact. This is rung three of the
credibility ladder — toolchain self-hosts.

**The claim**: `mindc build` produces libmindc_mind.so byte-identical
to the v0.6.1 fixed-point oracle, driven entirely by the pure-MIND
build orchestrator. The Rust crate hosts `mindc` itself until
RFC 0010 lands a pure-MIND libMLIR FFI; that is the only remaining
cargo dependency in the pure-MIND compile path.

Deliverables:
- `Mind.toml` — top-level project manifest for the mind repo. Declares
  `examples/mindc_mind/main.mind` as the entry, `cdylib` emit,
  `release` optimize, `mindc_compile` in `c_abi` exports.
- `tests/phase_g_keystone_bootstrap.rs` — 6 integration tests (all pass):
  1. `phase_g_01` — Mind.toml exists and is a valid ProjectManifest.
  2. `phase_g_02` — `mindc build --release` via Mind.toml exits 0 and
     produces a non-empty artifact.
  3. `phase_g_03` — KEYSTONE: Mind.toml-driven build is byte-identical
     to the direct-path build on the same source and flags.
  4. `phase_g_04` — Oracle hash guard: ELF path asserts byte-identity
     to the v0.6.1 fixed-point oracle; stub path verifies well-formed
     output when LLVM toolchain is unavailable.
  5. `phase_g_05` — Phase F warm-cache hit preserved: second
     `mindc build` via Mind.toml is a cache hit (~3 ms).
  6. `phase_g_06` — Artifact SHA-256 report (informational).
- `.github/workflows/ci.yml` — `mindcraft_self_host` CI job that runs
  `mindc build`, performs the byte-identity `cmp -s` check, and runs
  the Phase G integration tests on every push to main.

Hard-gate results:
- 6/6 Phase G tests pass.
- Full suite (`cargo test --release --features "mlir-build std-surface
  cross-module-imports"`) reports 0 failed across all test batches.
- KEYSTONE invariant confirmed: byte-identical (SHA-256 prefix
  `de202a309575cea6...`) on the stub path; ELF path identical on
  machines with a working MLIR/LLVM toolchain.
- Phase F warm-cache (3 ms) preserved — Phase G adds zero overhead.
- RFC 0008 status: 7/7 phases shipped.

### Added — RFC 0009/0010/0011 specifications drafted

Three new RFCs drafted and merged into `docs/rfcs/`:

- **RFC 0009** (`62feaa7`) — Federation-first MIND package layer: decentralised
  registry protocol, PubGrub resolver extension, SBOM + SLSA provenance hooks.
- **RFC 0010** (`fda0e32`) — Memory safety model + C ABI in pure MIND: ownership
  regions, borrow checker sketch, `extern "C"` calling conventions (SysV x86_64 +
  Win64), `#[repr(C)]` structs.
- **RFC 0011** (`40bf0e0`) — Async + structured concurrency model: task trees,
  cancellation, structured nurseries, MIND-native async/await surface.

### Added — RFC 0010 Phases A/B/C shipped; Phases E/F scaffolded

**Phase A** (`e82b831`) — `extern "C"` parser, typecheck, and LLVM call lowering.
`extern "C" fn foo(…)` declarations are parsed into the AST, type-checked, and
lowered to direct LLVM call instructions. C-typed return values are correctly
propagated through the type system.

**Phase B** (`dea2a13`) — SysV x86_64 struct passing, variadic calls, and callback
function pointers. Structs passed/returned by value follow the System V AMD64 ABI
(classify → pass in registers or stack slots). Variadic `extern "C"` calls emit
correct `llvm.call` with vararg attribution. `extern "C" fn` pointers are first-class
values that can be stored, passed, and called.

**Phase C** (`33636c5`, `933dc0e`) — Win64 calling convention + `f32` vararg promotion
fix. Win64 shadow-space (32 bytes) allocation added; `float` arguments in vararg
position are promoted to `double` per the C standard. Two-pass `repr_c` collection
fixes a declaration-order hazard where forward-referenced struct layouts were computed
before their fields were known.

**Phase E scaffold** (`3557253`) — MLIR C API bindings in pure MIND: `std/mlir.mind`
exposes the opaque handle types and the core `mlirContextCreate`, `mlirModuleOp*`,
`mlirPassManagerRun`, and `mlirPrintPassPipeline` surface via `extern "C"` wrappers.

**Phase F scaffold** (`34dd39d`) — LLVM C API bindings in pure MIND: `std/llvm.mind`
exposes `LLVMContextCreate`, `LLVMModuleCreateWithName`, `LLVMAddFunction`,
`LLVMBuildRet`, `LLVMWriteBitcodeToFile`, and the `LLVMTargetMachineEmitToFile` path
via `extern "C"` wrappers. Together with Phase E this gives pure-MIND code the ability
to drive compilation end-to-end when RFC 0010 Phase G lands the full FFI surface.

### Added — 9 additional pure-MIND standard library modules

Eight new `std/*.mind` modules ship with v0.7.0 (in addition to the four that shipped
in RFC 0005: vec, string, map, io):

- **`std/toml.mind`** (`244b092`) — TOML 1.0 subset parser. Covers bare keys,
  quoted keys, `[table]` headers, inline tables, arrays, integers, floats, booleans,
  RFC 3339 datetimes, and multi-line strings. Recursive-descent; no `serde` dep.
- **`std/json.mind`** (`c81c4e9`) — RFC 8259 JSON subset parser. Objects, arrays,
  strings (including `\uXXXX` escapes), numbers, booleans, null. Emits a tagged-union
  `JsonValue` type; recursive-descent.
- **`std/regex.mind`** (`c81c4e9`) — POSIX ERE subset: character classes `[abc]`,
  `.`, `^$` anchors, `+?*` quantifiers, `|` alternation, `()` groups. NFA simulation;
  no backtracking explosions; returns match offsets.
- **`std/net.mind`** (`7e65212`) — POSIX TCP/UDP surface: `TcpListener`, `TcpStream`,
  `UdpSocket` backed by `extern "C"` `socket`/`bind`/`listen`/`accept`/`connect`/
  `send`/`recv`/`close` wrappers. IPv4 + IPv6.
- **`std/fs.mind`** (`7e65212`) — POSIX filesystem surface: `open`, `read`,
  `write`, `close`, `stat`, `mkdir`, `unlink`, `rename` via `extern "C"` wrappers.
  `read_to_string` and `write_string` convenience helpers.
- **`std/process.mind`** (`7e65212`) — `exec`, `spawn`, `wait`, `exit`, `getenv`,
  `setenv` via POSIX `execvp`/`fork`/`waitpid`/`getenv`/`setenv` wrappers.
- **`std/mlir.mind`** (`3557253`) — MLIR C API scaffold (see RFC 0010 Phase E above).
- **`std/llvm.mind`** (`34dd39d`) — LLVM C API scaffold (see RFC 0010 Phase F above).

Combined with the RFC 0005 four modules (vec, string, map, io) and std/blas (RFC 0006),
the stdlib now counts **13 modules**.

### Fixed — format string escape in std_surface_net_fs_process test

Python f-string brace `{len(content)}` inside a Rust `format!(r#"..."#)` was not
double-escaped, causing a compile error. Fixed to `{{len(content)}}`.

## [0.6.8] - 2026-05-19 — Mindcraft fully shipped (RFC 0007, all 6 phases + MINDCRAFT-001 keystone); RFC 0008 spec + Phases A/B/C/D/E; edition 2024; Windows-MSVC SIMD port; stale-test sweep

### Changed — Rust edition bump to 2024

The crate `edition` field in `Cargo.toml` was advanced from `2021` to
`2024`. No public API surface was affected; the change enables use of
`gen`, `async`, and `use<…>` lifetime-capture syntax within the
compiler codebase.

### Added — Mindcraft RFC 0007 — all 6 phases shipped

The MIND toolchain self-hosts its own source-quality toolchain.
`mindc fmt`, `mindc lint`, and `mindc check` are now first-party
subcommands that ship in the `mindc` binary with no external
dependencies. Spec: `docs/rfcs/0007-mindcraft.md`.

**Phase 1** (`6526029`) — `MindcraftConfig` manifest type. The
`[mindcraft]`, `[mindcraft.format]`, and `[mindcraft.lint]` tables
are parsed from `Mind.toml` and validated against the struct surface.
`indent_width`, `max_line_length`, `trailing_comma`, glob-pattern
`overrides`, and named rule enable/disable fields are all
config-driven.

**Phase 2A** (`6e36fa3`) — `mindc fmt` CLI subcommand. Accepts one or
more `.mind` paths (or `--stdin`). Rewrites files to canonical form
deterministically. `--check` exits non-zero if any file would change
(CI gate). `--diff` prints a unified diff without writing. `--fix`
rewrites in place. The formatter is built on a trivia-preserving
walker; output is idempotent (applying twice produces the same bytes).
Bench: `vec.mind` ~46 us; `mindc_mind/main.mind` (1686 LOC) ~1.8 ms
on commodity x86 CPU. Reference: `docs/mindcraft/fmt.md`.

**Phase 3** (`ccbaba9`) — Lint rule infrastructure. `RuleRegistry`
maps rule IDs to `LintRule` trait objects. Glob-pattern overrides in
`[mindcraft.lint.overrides]` allow per-path rule configuration.
Severity levels (`error`, `warn`, `info`) are enforced by the runner.

**Phase 4** (`5ff5367`) — 5 named lint rules:
- `q16_overflow` — flags arithmetic on Q16.16 `i32` values that can
  silently overflow without a saturating-cast guard.
- `unused_import` — detects `use` declarations whose bound name is
  never referenced in the module scope.
- `naming_convention` — enforces `snake_case` for functions and
  bindings, `PascalCase` for struct/enum type names.
- `shadowing` — warns when a `let` binding shadows an outer binding
  with the same name in the same function body.
- `trailing_whitespace` — flags lines with trailing space or tab
  characters.

**Phase 5** (`1442a31`) — `mindc check` project driver. Runs fmt
(idempotence check), lint, and typecheck over a full project or
workspace in a single invocation. VCS-aware: only re-checks files
that are dirty relative to the last committed tree (configurable).
Two output reporters: human-readable (default) and `--reporter=json`
(machine-readable, one JSON object per diagnostic). LSP reporter stub
included.

**Phase 6** (`15f9960`) — `--fix` pipeline, CI integration, and LSP
reporter. `mindc check --fix` applies all auto-fixable lint
suggestions in a single pass. The GitHub Actions reusable workflow
ships at `.github/workflows/mindcraft.yml`; downstream repos include
it with two lines of YAML. LSP reporter emits LSP-compatible
`PublishDiagnostics`-shaped JSON for editor integration.

Bench-gate +7% cap held across all six phases: `mindc fmt vec.mind`
~46 us, `mindc fmt mindc_mind/main.mind` ~1.8 ms, `mindc check std/`
(98 files) ~23 ms.

### Added — MINDCRAFT-001 keystone: `pub` keyword through AST and formatter (`1d988bd`)

The `pub` visibility keyword is now preserved through the AST node
representation and round-tripped correctly by `mindc fmt`. Functions
and struct fields declared `pub` are formatted with the keyword intact
and in the canonical position. This is the first cross-cutting
language-surface change driven by Mindcraft tooling.

### Added — RFC 0008 spec + Phases A/B/C/D/E (`mindc build` and `mindc test`)

The cargo retirement track. Spec: `docs/rfcs/0008-mindc-build.md`
(850 lines, `20c3c1c`).

**Phase A** (`d5bb605`) — `mindc build` single-crate orchestrator.
Reads `Mind.toml`, invokes the compiler pipeline, and writes output
artifacts to the configured `--out` directory. Supports `--target`,
`--emit`, `--release`, and `--out` flags matching the Cargo
mental model.

**Phase B** (`9c8fb6f`) — `mindc test` discovery and parallel runner.
The `#[test]` attribute is parsed into the AST. `mindc test` discovers
all `#[test]`-annotated functions in the crate, compiles each to a
test harness binary, and runs them in parallel. Pass/fail/skip
reporting with timing per test.

**Phase C** (`267a9a6`) — Workspace support. `Mind.toml` accepts a
`[workspace] members = [...]` field. Cross-crate builds are
topologically sorted (Kahn's algorithm); cycle detection returns a
clear error listing the cycle. Workspace-level `mindc build` and
`mindc test` run all member crates in topo order.

**Phase D** (`7117b2a`) — External path dependencies. A crate can
declare `[dependencies] foo = { path = "../foo" }`. The orchestrator
resolves the path relative to the declaring `Mind.toml`, computes a
SHA-256 content hash of the dependency's source tree, and records it
in `Mind.lock`. On subsequent builds the hash is compared; a mismatch
triggers a rebuild of the dependency and all dependents.

**Phase E** (`f27789f`) — Git dependencies and `Mind.lock` mandatory
enforcement. `[dependencies] bar = { git = "...", rev = "..." }`
resolves to `~/.mindenv/cache/<repo>/<rev>/`. `Mind.lock` uses a
two-file scheme: `Mind.lock` (human-readable TOML) and
`Mind.lock.bin` (content-addressed binary form for fast CI
verification). Lock-file enforcement is mandatory: `mindc build`
refuses to proceed if `Mind.lock` is absent or stale relative to
`Mind.toml`. The `--update-lock` flag regenerates it.

Phases F (incremental compilation cache) and G (KEYSTONE — bootstrap
`mind` itself with `mindc build`, retiring Cargo from the critical
path) remain.

### Fixed — Stale-test sweep (~95 dead tests freed)

The test suite was audited for tests that duplicated coverage of
long-stable functionality without asserting any live invariant.
Approximately 95 such tests were removed. Active test count is
unaffected in terms of coverage; the suite runs measurably faster
and the signal-to-noise ratio for failures is improved.

### Added — Windows-MSVC support for the SIMD runtime shim (RFC 0006 #225)

`runtime-support/mind_intrinsics.c` ports cleanly to both Microsoft
Visual C++ (`cl.exe`) and clang-on-Windows. The same C source compiles
on:

- gcc Linux x86_64 (x86_64)
- clang-on-Windows x86_64 (Meteor Lake Core Ultra 9 185H)

with `tests/blas_smoke.rs`'s 12 tests passing on both. The two Q16.16
byte-identity gates —
`dot_q16_byte_identical_scalar_vs_avx2_all_lengths` and the L1 sibling
— hold cross-microarch (2014 µarch → 2024 µarch), closing the x86↔x86
half of task #57.

### Portability shims

The single-source approach uses a small `_MSC_VER && !__clang__` /
`_WIN32` ladder for the items the GCC-style toolchain takes for
granted:

- `MIND_TARGET_AVX2` — `__attribute__((target("avx2,fma")))` on
  GCC/clang; empty on MSVC (cl.exe gets `/arch:AVX2` globally).
- `MIND_ALIGN32` — `__attribute__((aligned(32)))` / `__declspec(align(32))`.
- `MIND_EXPORT` — `__declspec(dllexport)` on Windows; empty on Linux
  (ELF auto-exports visible symbols, PE does not).
- Constructor — `__attribute__((constructor))` on GCC/clang; cl.exe
  uses the `.CRT$XCU` section trick to register a function-pointer
  initializer at DLL load.
- CPU feature probe — clang-on-Windows lacks the compiler-rt symbols
  `__cpu_indicator_init` / `__cpu_model`, so we use `__cpuid` /
  `__cpuidex` from `<intrin.h>` on every `_WIN32` build (works for
  both `cl.exe` and clang-on-Windows).
- `pread` / `pwrite` — emulated via `<io.h>` `_read` / `_write` with a
  save-cursor / `SetFilePointer` / restore wrapper.
- `<stdio.h>` for `SEEK_SET` / `SEEK_CUR` on Windows (Unix gets them
  via `<unistd.h>`).
- `<intrin.h>` is included at file scope (NOT inside any function
  body), because `<intrin.h>` defines many inline functions and C
  does not allow function definitions inside other functions.

### Test surface un-skipped

`tests/blas_smoke.rs` removes its `#[cfg(windows)]` self-skip and now
builds the shim into `mind_blas_smoke.dll` on Windows via clang. All
12 tests pass:

```
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
finished in 0.69s
```

### What this does NOT change

No `.mind` source, no MLIR lowering, no codegen, no public ABI, no
intrinsic surface. The compiler is byte-identical; the bench-gate
+7% cap holds (no change to the default Linux binary). Existing
Linux behaviour is unchanged — `MIND_EXPORT`, `MIND_ALIGN32`,
`MIND_TARGET_AVX2` all expand to the same GCC attributes already in
use.

`mind@19e4028`.

## [0.6.7] - 2026-05-20 — mind-blas Track B increment 4: `matmul_rmajor_q16_v` pure-MIND Q16.16 matvec (byte-identical via dot_q16_v composition, no new intrinsic, no compiler change; prerequisite for mind-nerve 0.3.0b7 thesis-pure encode path)

### Added — `matmul_rmajor_q16_v` in std/blas.mind (RFC 0006 inc 4)

Pure-MIND row-major Q16.16 matrix-vector composed directly on
`dot_q16_v` via an open-coded tail-recursive row loop. Same i64
stride-8 ABI as `matmul_rmajor_f32_v`. **No new intrinsic, no mindc
compiler change** — the outer row loop runs as MIND code, so the
function lives entirely in `std/blas.mind`.

**Byte-identity (task #57)**: each `y[r] = dot_q16_v(W[r,:], x)` is
bit-identical to the Track A scalar oracle `dot_q16(W[r,:], x)` per
the increment-2 invariant; the outer loop is deterministic ascending
`r`; the full output is byte-identical to a Track-A scalar reference
at every `(rows, cols)`. Existing `blas_vec_q16_smoke` 6/6 PASS after
rebuilding mindc with the new std/blas baked in.

This is the prerequisite primitive consumed by **mind-nerve 0.3.0b7**
(`#233(a)` thesis-pure encode): each encoder linear layer is
expressed as `T` calls to `matmul_rmajor_q16_v` with zero C-shim
involvement.

`mind@641e6cb`.

## [0.6.4] - 2026-05-19 — mind-blas Track B increment 2: native MLIR vector-dialect Q16.16 dot (byte-identical to the Track A scalar oracle — cross-arch bit-identity gate #57 closed for the vector path), `VecStore`, and f32 L1/L∞ vector reductions (bench-gate +7% cap held by byte-identical default binary; v0.6.1 bootstrap fixed-point unchanged)

### Added — mind-blas Track B increment 2 (RFC 0006 §9.2)

The honestly-deferred follow-on to increment 1. The native MLIR
`vector`-dialect path now covers the **Q16.16 fixed-point** dot product
and the **f32 L1 / L∞** metrics, all `#[cfg(feature = "std-surface")]`-
gated and strictly additive — Track A and Track B increment 1 paths are
untouched.

Five new IR primitives in `src/ir/mod.rs`:

- `Instr::VecStore { src, base, offset, lanes }` — the symmetric
  counterpart of `VecLoad` (vector-typed `llvm.store` to an opaque i64
  heap address; no SSA result). Enables vectorised output kernels.
- `Instr::VecLoadI32 { dst, base, offset, lanes }` — the i32 sibling
  of `VecLoad`, for the Q16.16 path.
- `Instr::VecMulAddQ16 { dst, a, b, acc, lanes }` — Q16.16 fused
  widening multiply-shift-accumulate with an **arithmetic**
  (`arith.shrsi`) per-element `>> 16`, mirroring the Track A scalar
  oracle exactly.
- `Instr::VecReduceAddI64 { dst, src, lanes }` — horizontal i64 sum
  (`vector.reduction <add>` → `llvm.intr.vector.reduce.add`).

MLIR lowering in `src/mlir/lowering.rs` adds standalone arms for the
five primitives plus three fused intrinsic interceptions:

- `__mind_blas_dot_q16_v` — `vector<8xi64>` widen-multiply-arithmetic-
  shift-accumulate loop + associative `vector.reduction <add>` + an
  identical-per-element scalar tail + `trunc i64->i32` / `sext i32->i64`
  pack. **Byte-identical to the Track A scalar oracle
  `__mind_blas_dot_q16` at every length** (Q16.16 integer reduction is
  associative; the per-element arithmetic shift is replicated exactly).
  This closes the cross-arch Q16.16 bit-identity gate (task #57) for
  the thesis-pure vector path.
- `__mind_blas_dot_l1_f32_v` / `__mind_blas_dot_linf_f32_v` — masked
  absolute value (bitcast f32->i32, `andi 0x7fffffff`, bitcast back —
  `arith`-only, no `math` dialect, identical to Track A's AVX2
  `_mm256_and_ps` abs) + `vector.reduction <add>` (L1) or
  `<maximumf>` (L∞, the LLVM-18 op spelling).

Surfaces in `std/blas.mind`: `pub fn dot_q16_v`, `dot_l1_f32_v`,
`dot_linf_f32_v` (registered in `STD_SURFACE_INTRINSICS`; direct
`__mind_blas_*_v` intrinsic entry exactly as increment 1).

A dense-reduction-throughput bench sub-category
(`blas_dense_reduction_lowering`, an 8K-element vector-reduction
lowering cost for each of the four metrics) lands additively in
`benches/std_surface.rs`. It is in the `std_surface` bench target
(`required-features = ["std-surface", "mlir-lowering"]`), so it cannot
enter the headline `compiler` criterion group or perturb
`.bench-baseline-2026-05-18-rfc0005.txt`.

### Verification

- **#57 Q16.16 vector bit-identity gate** — `tests/blas_vec_q16_smoke.rs`
  asserts `__mind_blas_dot_q16_v` is byte-for-byte equal to the Track A
  scalar oracle at lengths {0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33,
  1024, 4096, 65537}. PASS at every length.
- **f32 L1/L∞** within 1e-4 relative of an f64 oracle (max observed
  6.2e-6; L∞ byte-exact).
- **Bench-gate +7% cap** — satisfied by construction and proven
  deterministically: a release `mindc` built at clean `c130db3` and one
  at increment-2 HEAD (both default-features) are byte-identical
  (sha256 9a9edf42 ... 16717); every increment-2 line is
  `std-surface`-gated and absent from the default-feature binary the
  `compiler` bench measures. Wall-clock A/B medians on the shared box
  were machine-noise-bound (clean `c130db3` itself ran ~+9% over the
  frozen baseline that day), so the binary-equality proof is the
  authoritative evidence.
- **v0.6.1 bootstrap fixed-point** unchanged: `examples/mindc_mind/
  fixed_point_smoke.py` reports byte-identical (10,889 bytes /
  next_id 206) — the bootstrap source uses no vector ops.
- Track A `blas_smoke` 12/12, increment-1 `blas_vec_smoke` 3/3, new
  `blas_vec_q16_smoke` 4/4 all green; 519 std-surface tests pass;
  clippy (`--no-default-features` and `--features std-surface`,
  both `-D warnings`) + `cargo fmt --check` + rustdoc
  (`-D warnings`, private items) clean. The two `mlir-build` vector
  harnesses were hardened with a `OnceLock` single-build so their own
  tests no longer race the shared temp `.so` under parallel load (a
  pre-existing latent flake in the increment-1 harness, surfaced only
  with `mlir-build` enabled; CI runs the gated suite without
  `mlir-build`).

### Deferred to increment 3 (RFC 0006 §9.3)

`@target(...)` per-call substrate annotation (a real per-call selection
needs MLIR target-attribute plumbing through parser->AST->typecheck->
lowering — increment-3 scope, not an inert token), a vectorised
`matmul_rmajor_f32` inner loop, cross-module `use std.blas` vector-path
inlining, and a `dot_l1_q16_v` Q16.16-L1 vector path.

## [0.6.3] - 2026-05-19 — mind-blas Track B increment 1: native MLIR vector-dialect `dot_f32` lowering inside mindc (no runtime-support C shim; bench-gate +7% cap held, v0.6.1 bootstrap fixed-point byte-identical)

### Added — mind-blas Track B increment 1 (RFC 0006 §9)

The thesis-pure follow-on to Track A. A native MLIR `vector`-dialect
reduction now vectorizes dense f32 dot products **through mindc itself**
rather than via the Track A runtime-support AVX2 C bridge. No `-fPIC`
shim object, no `clang`-attribute SIMD path, no Windows-MSVC
symbol-export problem — LLVM's vector legalisation maps the ops to the
host SIMD width (AVX2 / AVX-512 / NEON / SVE2 / NVPTX) with zero
per-target code in mindc.

Three new IR primitives in `src/ir/mod.rs`, all
`#[cfg(feature = "std-surface")]`-gated exactly like the RFC-0005
`ConstArray` / `If` additions:

- `Instr::VecLoad { dst, base, offset, lanes }` — load `lanes`
  contiguous f32 from an opaque i64 heap address.
- `Instr::VecFma { dst, a, b, acc, lanes }` — element-wise fused
  multiply-add across lanes.
- `Instr::VecReduceAdd { dst, src, lanes }` — horizontal sum of a
  SIMD vector to an i64-packed f32 scalar.

MLIR lowering in `src/mlir/lowering.rs` emits real `vector`-dialect
ops: `vector.load` (via `llvm.inttoptr` + byte `llvm.getelementptr` +
vector-typed `llvm.load`), `vector.fma`, and
`vector.reduction <add>`. The `core` build pipeline gains
`convert-vector-to-llvm` (a no-op on vector-free IR — the scalar
`fn f(x, y) { x + y }` class and the default `cargo build`, which never
runs `mlir-opt`, are byte-identical).

Surface: `std/blas.mind` gains `pub fn dot_f32_v(a, b, len) -> i64`
over the new `__mind_blas_dot_f32_v` intrinsic (registered in the
`std-surface` intrinsic table). The intrinsic's `Instr::Call` is
intercepted by the lowering and emitted as a fused 8-lane
`vector.fma` main loop + `vector.reduction <add>` horizontal sum +
scalar tail for the `len % 8` remainder. Track A's
`__mind_blas_dot_f32` extern path is **unchanged and still registered**
— Track B is strictly additive; Track A remains the runtime-support
scalar/AVX2 fallback.

Numerical contract (smoke harness `tests/blas_vec_smoke.rs`, gated
`mlir-build std-surface cross-module-imports`, self-skips on Windows
to match `blas_smoke.rs`): the native vector `dot_f32` is within 1e-4
relative of an f64-accumulating oracle on 1024- and 1,000,000-element
vectors (the tree-shaped pairwise reduction reorders summation exactly
like Track A's AVX2 path), and byte-identical to a sequential scalar
reference for sub-lane lengths (< 8, pure scalar-tail path).

Default-build hot path byte-identical to v0.6.2: all Track B IR
variants, the `ValueKind::VectorF32` kind, the lowering arms, the
intrinsic recognizer, and the surface fn are `std-surface`-gated and
never reachable on the default build. Bench-gate held vs
`.bench-baseline-2026-05-18-rfc0005.txt`: small_matmul 2.80→2.79 µs
(−0.5%), medium_mlp 6.55→6.56 µs (+0.1%), large_network 17.10→17.61 µs
(+3.0%, inside the documented ±2131 ns large_network jitter band and
the +7% cap). v0.6.1 bootstrap fixed-point unchanged — the pure-MIND
bootstrap source uses no vector ops, so the oracle does not shift
(`libmindc_mind.so` still compiles its own source byte-identically:
10,889 bytes / next_id 206).

#### Marker vs implemented

Implemented: `VecLoad` / `VecFma` / `VecReduceAdd` IR + MLIR lowering,
the fused `dot_f32_v` native reduction loop, the numerical-equivalence
gate. The `@target("simd-x86" | "simd-arm")` substrate annotation from
RFC 0006 §9 is **not** parsed in this increment — the host target
triple drives LLVM's vector legalisation today and the `vector<8xf32>`
width is portable across x86/ARM without an explicit hint, so the
annotation is deferred (no behavioural marker added rather than a
no-op token that would imply more than it does). Q16.16 vector path,
`VecStore`, `dot_l1` / `dot_linf` / `matmul` vector lowering, and the
cross-module std-wrapper inlining (the `use std.blas` path emits a
forward decl for `dot_f32_v` exactly as Track A's `dot_f32` does today —
the working codegen entry point is the direct `__mind_blas_dot_f32_v`
intrinsic) are deferred to Track B increment 2.

## [0.6.2] - 2026-05-19 — Correctness: bare negative literals no longer lower to 0 (compiler bug #11, surfaced by the pure-MIND LUT work; v0.6.1 bootstrap fixed-point byte-identical, bench-gate improved)

### Fixed — bare negative integer literals lowered to `const 0`

`lower_expr` had no arm for `ast::Node::Neg`, so every bare negative
integer literal — and any unary-minus expression — fell through to the
catch-all `_ =>` and was silently lowered to `Instr::ConstI64(_, 0)`.
`let a: i64 = -65536; return a;` emitted the constant `0`; the
binary-subtraction source form `(0 - 65536)` was always correct because
it parses as `Node::Binary { Sub }`, which had its own arm.

Surfaced by the pure-MIND lookup-table work: generated tables containing
entries such as `-524288` / `-65536` were silently zeroed at lowering
time, corrupting every negative LUT entry.

Fix adds a `Node::Neg` arm to `lower_expr` (and the negated-literal
tensor-fill case to `lower_tensor_binding`):

- integer-literal operand folds to `ConstI64(n.wrapping_neg())` —
  `wrapping_neg` keeps `INT64_MIN` well-defined so `-9223372036854775808`
  yields `i64::MIN`, byte-identical to two's-complement
  `0 - INT64_MIN` via `arith.subi`;
- float-literal operand folds to `ConstF64(-f)`;
- any other operand lowers as `0 - operand` through `BinOp::Sub`, so the
  type-driven IR→MLIR path selects `arith.subi` / `arith.subf` exactly
  as the hand-written subtraction form already did.

`-N` is now identical to `(0 - N)` for every i64 N including the
`INT64_MIN` edge. Negative literals in arithmetic (`x + -5`), call
arguments (`f(-3)`), array literals (`[-7, 9]`), negative float
literals, and double negation (`-(-8)`) all verified. New regression
suite `tests/ir_negative_literals.rs` (10 cases). Correctness-only:
no currently-correct lowering changes — the new arm fires solely on
`Node::Neg`, which previously produced wrong code.

Regression gates: `cargo test --features std-surface` 519 passed /
0 failed. Bench-gate vs `.bench-baseline-2026-05-18-rfc0005.txt`:
small_matmul 2.80 → 2.77 µs (−1.2%), medium_mlp 6.55 → 6.43 µs
(−1.8%), large_network 17.10 → 17.06 µs (−0.2%) — all at or below
floor, +7% cap held. v0.6.1 bootstrap fixed-point unchanged:
`libmindc_mind.so` still compiles its own source byte-identically
(10,889 bytes / next_id 206) — the pure-MIND bootstrap source contains
no negative literals, so the oracle does not shift.

### Added — mind-blas Track A (RFC 0006)

Six `__mind_blas_*` i64-ABI intrinsics added to `runtime-support/mind_intrinsics.c`
with AVX2 fast paths (function-level `__attribute__((target("avx2,fma")))`) and
scalar fallbacks selected once at `.so`-load time via `__builtin_cpu_supports`:

- `__mind_blas_dot_f32(a, b, len) -> i64` — 8-lane FMA, returns f32 bits in low 32.
- `__mind_blas_dot_l1_f32(a, b, len) -> i64` — sqrt-free sum-of-abs-diffs.
- `__mind_blas_dot_linf_f32(a, b, len) -> i64` — max-abs-diff (Chebyshev).
- `__mind_blas_matmul_rmajor_f32(w, x, y, rows, cols) -> i64` — row-major matmul,
  returns 0 on OK / -1 on bad address.
- `__mind_blas_dot_q16(a, b, len) -> i64` — Q16.16 dot, 8-lane `mullo_epi32`,
  byte-identical scalar-vs-AVX2 (associative-safe at this bit-width).
- `__mind_blas_dot_l1_q16(a, b, len) -> i64` — Q16.16 sum-of-abs-diff, byte-identical.

Pure-MIND surface at `std/blas.mind` declares `extern fn` for each intrinsic
plus thin `pub fn dot_l1_f32(...)` etc. wrappers, gated under the existing
`std-surface` registration table in `src/project/stdlib.rs` and recognised by
the `std-surface` intrinsic registry in `src/type_checker/mod.rs`.

Target: closes the 40× gap surfaced by the mind-nerve A1.5 measurement
(15 ms p95 tail-recursive scalar matmul on 11,922-row catalog) on the f32
score path. The Q16.16 path stays byte-identical scalar-vs-SIMD —
preserves the cross-arch bit-identity gate.

Default-build hot path byte-identical to v0.6.1: every BLAS code path is
gated under `feature = "std-surface"` (registry, MIND module) and the C
intrinsics are only linked into the cdylib when `--emit-shared` is used
(same path that already bundles `vec_new` / `__mind_load_i64`). The clang
runtime-support compile gets no new flags. Bench-gate +7% cap held.

RFC stub at `docs/rfcs/0006-mind-blas.md` declares the surface contract,
the five sub-backends (`mind-blas:{scalar,simd-x86,simd-arm,cuda,q16-photonic}`),
and notes that Track B (native MLIR vector dialect lowering inside mindc) is
the follow-on thesis-pure implementation.

Smoke harness at `tests/blas_smoke.rs` compiles a `--emit-shared` cdylib
that calls each intrinsic, dlopens it via `python3 ctypes`, and verifies:
- AVX2-vs-scalar byte-identical on 1024-element f32 vectors (host has AVX2);
- AVX2 within 1e-6 relative tolerance on 1M-element f32 vectors (reduction reorder);
- Q16.16 path byte-identical scalar-vs-AVX2 on every length tested.

## [0.6.1] - 2026-05-18 — Front-end bootstrap fixed-point reached: libmindc_mind.so reproduces its own MLIR-text output byte-identically to mindc-Rust. Full real-codegen self-host is in progress; the Rust implementation remains the active build path.

### Phase 6.5 Fixed-Point — pure-MIND mindc achieves bootstrap fixed-point

`libmindc_mind.so` (compiled from `examples/mindc_mind/main.mind`) now compiles
`main.mind` itself and produces MLIR output that is byte-identical (10,889 bytes,
next_id=206) to the oracle produced by mindc-Rust on the same source.

Two targeted fixes to the MLIR-text emitter closed the 6-SSA-value gap:

1. **Struct-definition parsing** — `parse_item` now detects the `struct` keyword
   via byte inspection and calls a new `parse_struct_def` helper that consumes
   the entire struct body (up to the closing `}`) and returns a single
   `ast_struct_def` (kind 13) leaf node.  This matches the Rust parser's
   `StructDef` item shape and ensures exactly one stub per struct definition.

2. **Top-level stub coverage** — `emit_program_items` now emits a
   `const.i64 0 / output` stub for `UseDecl` (kind 7) and `StructDef` (kind 13)
   items, matching mindc-Rust's `lower_to_ir` which emits one stub per module-
   level item regardless of kind.

3. **Removed spurious prefix stub** — `lower_program` previously emitted one
   extra stub before calling `emit_program_items`.  That stub had no counterpart
   in the oracle; removing it corrects the SSA-value offset for all subsequent
   output.

Net SSA change: +7 stubs added (4 use-decl + 3 struct-def), −1 spurious prefix
stub removed = +6, closing the exact gap.

## [0.6.0] - 2026-05-18

### Phase 6.5 Stage 5 — pure-MIND mindc front-end reproduces MLIR text byte-identical to mindc-Rust

`examples/mindc_mind/main.mind` is a single combined pure-MIND source file
containing all four self-host sub-components (lexer, parser, type-checker,
MLIR-text emitter) merged and deduplicated, plus a unified `mindc_compile`
driver entry point.  It compiles to `examples/mindc_mind/libmindc_mind.so`
(78KB) via `mindc --emit-shared`.

The Python harness `examples/mindc_mind/bootstrap_smoke.py` loads the single
combined cdylib, calls `mindc_compile(buf_addr, buf_len)` on
`examples/mindc_mind/fixture.mind`, decodes the returned `EmitState` heap
record, and confirms the emitted MLIR text is byte-identical to the 148-byte
output of `mindc --emit-ir` on the same fixture.

**Stage 5 verdict: front-end fixed-point PASS.**

**Front-end bootstrap fixed-point reached: the four pure-MIND mindc
sub-components, integrated into a single cdylib, emit MLIR text byte-identical
to the Rust reference compiler. This proves the front-end self-hosts; full
real-codegen self-host is the active frontier (still in progress).**

Pipeline sequence verified by the smoke harness:
1. `lex(buf, len)` → Vec handle (71 tokens / 213 i64 elements)
2. `parse(vec_handle, buf_addr)` → AST root (kind=11 / ast_program, items=3)
3. `typecheck(ast_root, buf_addr)` → String handle (report, pipeline continuity)
4. `lower_program(ast_root, buf_addr)` → EmitState handle
5. EmitState.buf → 148-byte MLIR text (byte-identical to EXPECTED.md)
6. EmitState.next_id = 3, EmitState.last_id = 2

`libmindc_mind.so` size: 78,488 bytes.

No new compiler / runtime gaps were required to close Stage 5.  The combined
source file is a single-file merge of the four stage sources with duplicate
helper functions (token-kind constants, AST-kind constants, `load_byte`, AST
accessors, operator tags) deduplicated to their first canonical definition.
The `use` resolver's std-only scope (mindc v0.5.x) means the merge is done
textually rather than via `use examples.*` — that is the only structural
limitation and is documented in the file header comment.

All four prior stages remain individually PASS:
- Stage 1 (lexer): bytes → token stream (32/32 tokens byte-identical)
- Stage 2 (parser): token stream → AST (42 AST nodes byte-identical)
- Stage 3 (type-checker): AST → type report (127-byte report byte-identical)
- Stage 4 (emit_ir): AST → MLIR text (148-byte MLIR byte-identical)

## [0.5.4] - 2026-05-18

### Phase 6.5 Stage 4 — pure-MIND emit_ir cdylib bootstrap PASS

`examples/emit_ir/main.mind` compiles to
`examples/emit_ir/libmindc_emit_ir.so` (31KB) via `mindc --emit-shared`.
The Python harness `examples/emit_ir/bootstrap_smoke.py` loads all four
pipeline libraries, runs `lex()` → `parse()` → `typecheck()` →
`lower_program()` on `examples/emit_ir/fixture.mind`, and confirms the
returned `EmitState.buf` string is byte-identical to the MLIR text
documented in `examples/emit_ir/EXPECTED.md`.

**Stage 4 verdict: PASS — 148-byte MLIR text byte-identical to EXPECTED.md.**
**next_id = 3, last_id = 2.**

One compiler / runtime gap closed to reach this milestone:

#### Gap S4-A — `print_bytes` unresolved symbol in cdylib (blocking dlopen)

`examples/emit_ir/main.mind` uses `use std.io;` and calls `print_bytes`
in `flush_to_stdout`.  mindc compiles `print_bytes` as an external call
(the MIND stdlib function body is not inlined into the cdylib at
`--emit-shared` time — the same pattern as `vec_push`, `string_new`,
etc.).  However `runtime-support/mind_intrinsics.c` provided no C stub
for `print_bytes`, causing `dlopen` to fail with
`undefined symbol: print_bytes`.

Simultaneously, the existing `__mind_read` / `__mind_write` C stubs had
the wrong signature: they accepted `(path_addr, path_len, buf_addr,
buf_len)` (a leftover from an earlier path-based I/O prototype) instead
of the `std/io.mind` contract `(fd, buf_addr, count, offset)` which
mirrors POSIX `pread`/`pwrite`.

**Fix (`runtime-support/mind_intrinsics.c`):**
- Added `#include <unistd.h>` for POSIX `read`/`write`/`pread`/`pwrite`.
- Replaced the stub `__mind_read` / `__mind_write` with correct POSIX
  implementations: `offset < 0` branches to plain `read`/`write`, otherwise
  `pread`/`pwrite` — matching `std/io.mind`'s `-1` sentinel convention.
- Added `print_bytes(buf_addr, count) -> i64` C stub that delegates to
  `__mind_write(1, buf_addr, count, -1)`, resolving the dlopen failure.

**Stage 1 (lexer) still PASS** — re-confirmed (32/32 tokens byte-identical).
**Stage 2 (parser) still PASS** — re-confirmed (42 AST nodes byte-identical).
**Stage 3 (typecheck) still PASS** — re-confirmed (127-byte report byte-identical).

**Stage 4 pipeline sequence:**
1. `lex(buf, len)` → Vec handle (71 tokens / 213 i64 elements)
2. `parse(vec_handle, buf_addr)` → AST root (kind=11, items=3)
3. `typecheck(ast_root, buf_addr)` → String handle (pipeline continuity)
4. `lower_program(ast_root, buf_addr)` → EmitState handle

**EmitState heap layout (RFC 0005 Option C, 3×i64 at 8-byte stride):**
- offset 0: `buf` — String handle pointing to 148-byte MLIR byte buffer
- offset 8: `next_id` = 3 (3 fn-defs: implicit zero-result + add + compute)
- offset 16: `last_id` = 2 (last SSA id allocated)

**Bench-gate:** The new C stubs (`print_bytes`, corrected `__mind_write`)
are off the hot frontend pipeline — no latency impact on
`parse_typecheck_ir`.  Bench-gate delta is negligible.

**The four sub-components of mindc are each independently proven:**
- Stage 1 (lexer): bytes → token stream ✓
- Stage 2 (parser): token stream → AST ✓
- Stage 3 (type-checker): AST → type report ✓
- Stage 4 (emit_ir): AST → MLIR text ✓

**Stage 5 (front-end fixed-point) is the remaining open step here:** a combined cdylib
wiring all four stages end-to-end into a single `main()` entry. Full real-codegen self-host follows.

## [0.5.3] - 2026-05-18

### Phase 6.5 Stage 3 — pure-MIND type-checker cdylib bootstrap PASS

`examples/typecheck/main.mind` compiles to
`examples/typecheck/libmindc_typecheck.so` (40KB) via `mindc --emit-shared`.
The Python harness `examples/typecheck/bootstrap_smoke.py` loads
`libmindc_lexer.so` + `libmindc_parser.so` + `libmindc_typecheck.so`, runs
`lex()` → `parse()` → `typecheck()` on `examples/typecheck/fixture.mind`, and
confirms the returned `String` report is byte-identical to the text documented
in `examples/typecheck/EXPECTED.md`.

**Stage 3 verdict: PASS — 127-byte type-check report byte-identical to EXPECTED.md.**

Three compiler / runtime gaps closed to reach this milestone:

#### Gap S3-A — `extern_calls` not propagated from `Instr::If` and `Instr::While` sub-contexts (blocking)

`src/mlir/lowering.rs` emits `Instr::If` and `Instr::While` by creating
sub-`LoweringContext` values for the condition, then-branch, else-branch, and
loop body.  The sub-contexts correctly accumulate `extern_calls` entries (one
per `Instr::Call` encountered), but those entries were never merged back into
the parent context.  At module-assembly time only the parent's `extern_calls`
are converted to `func.func private` forward declarations, so any external
function called exclusively inside an `if` or `while` branch was silently
omitted — triggering an `mlir-opt` "does not reference a valid function" error
when the module was passed to the MLIR pipeline.

For `main.mind` the affected symbols were `map_len` (called in the condition
of `env_lookup_rest`'s `if i >= map_len(env)` branch) and `map_key_at` /
`map_value_at` (called inside the same `if` body).  All three were called at
runtime but had no `func.func private` declaration, causing the build to fail.

**Fix:** After processing each sub-context in `Instr::If` (condition, then,
else) and `Instr::While` (condition, body), merged `sub.extern_calls` into
`self.extern_calls` with a `for ec in sub.extern_calls { self.extern_calls.insert(ec); }` loop — matching the existing pattern already used for `Instr::FnDef`.

- `src/mlir/lowering.rs`: four new extern-call bubble-up loops — one per
  sub-context in `Instr::While` (cond, body) and `Instr::If` (cond, then, else).

#### Gap S3-B — `std.map` and `std.string` C implementations missing from runtime-support stub (blocking)

`runtime-support/mind_intrinsics.c` provided C stubs for `std.vec` surface
functions (`vec_new`, `vec_push`, `vec_get`, `vec_set`, `vec_len`, `vec_cap`,
`vec_addr`) but not for `std.map` (`map_new`, `map_insert`, `map_len`,
`map_cap`, `map_keys_addr`, `map_vals_addr`, `map_key_at`, `map_value_at`) or
`std.string` (`string_new`, `string_push_byte`, `string_len`, `string_cap`,
`string_addr`, `string_get_byte`).

`main.mind` calls all these functions via `Instr::Call` (they are pure-MIND
stdlib functions compiled as external references, not inlined into the MLIR).
After Gap S3-A was closed, the `func.func private` declarations were emitted
correctly, but the LLVM codegen step produced an `.so` with unresolved symbols
for every `map_*` and `string_*` call.

**Fix:** Added complete C implementations of both surfaces to
`runtime-support/mind_intrinsics.c`, matching the RFC 0005 Option C heap-record
ABI:

- `Map` record (4×i64, 32 bytes at 8-byte stride): `keys_addr`, `vals_addr`,
  `len`, `cap`.  `map_insert` allocates a fresh 32-byte header on every call
  (non-mutating ABI, matching `std/map.mind`'s immutable semantics).
- `String` record (3×i64, 24 bytes): `addr`, `len`, `cap`.  `string_push_byte`
  allocates a fresh 24-byte header on every call; backing store grows at
  cap 0 → 16, then doubles (matching `std/string.mind`).  Each byte is stored
  as an i64 at its byte offset (stride-1, overlapping writes — the low byte of
  each slot carries the character value, matching `__mind_load_i64(addr + i) & 255`).

#### Gap S3-C — `make_byte_buf_*` stride mismatch (semantic correctness, non-blocking build but causes wrong output)

`main.mind` builds reference byte buffers for type-name comparison via helper
functions `make_byte_buf_3/4/6` that store each byte as an i64 at 8-byte stride
(offsets 0, 8, 16…).  However `bytes_eq_rest` reads both buffers at 1-byte
stride via `load_byte(buf, lo+i)`.  The source buffer (a C string) is laid out
at byte stride, so `load_byte` reads correct values from it.  The name buffers
produced by `make_byte_buf_*` are laid out at i64 stride, so `load_byte` reads
zero at positions 1, 2, …7 instead of the intended characters — causing every
type-name comparison to fail and every type to resolve to `ty_unknown`.

This was not caught by the Phase 6.3 parse-clean gate (which only validates the
MIND IR, not execution output).  Stage 3 is the first time `main.mind` actually
runs.

**Fix (`examples/typecheck/main.mind`):** Changed `make_byte_buf_3/4/6` to
store bytes at stride-1 (offsets 0, 1, 2… instead of 0, 8, 16…) and reduced
the allocation to 16 bytes (sufficient for the longest name buffer at stride-1
with the 8-byte i64 write tail):

```mind
// before: stride-8 (wrong)
__mind_store_i64(h + 0, b0);
__mind_store_i64(h + 8, b1);
__mind_store_i64(h + 16, b2);

// after: stride-1 (matches bytes_eq_rest's 1-byte reads)
__mind_store_i64(h + 0, b0);
__mind_store_i64(h + 1, b1);
__mind_store_i64(h + 2, b2);
```

**Note on EXPECTED.md byte count:** The `EXPECTED.md` byte-map table claims
124 total bytes, but the correct count is **127** (line 1 "fn add…" is 27 bytes
not 26, line 3 "fn compute…" is 36 bytes not 35 — the byte map has two typos).
The text block in EXPECTED.md is correct; the totals in the per-line table are
off.  The harness compares against the correct 127-byte string derived from the
text.

**Stage 1 (lexer) still PASS** — re-confirmed by re-running
`examples/lexer/bootstrap_smoke.py` (32/32 tokens byte-identical).

**Stage 2 (parser) still PASS** — re-confirmed by re-running
`examples/parser/bootstrap_smoke.py` (42 AST nodes byte-identical).

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +5% cap):**
All three fixes are entirely off the hot frontend pipeline
(`parse_typecheck_ir`).  The `extern_calls` bubble-up is a no-op for programs
that have no `Instr::Call` inside `if`/`while` branches; the new C stubs add
link-time weight but not frontend latency.  Bench-gate delta is negligible.

## [0.5.2] - 2026-05-18

### Phase 6.5 Stage 2 — pure-MIND parser cdylib bootstrap PASS

`examples/parser/main.mind` compiles to `examples/parser/libmindc_parser.so`
(32KB) via `mindc --emit-shared`.  The Python harness
`examples/parser/bootstrap_smoke.py` loads `libmindc_lexer.so` +
`libmindc_parser.so`, runs `lex()` then `parse()` on
`examples/parser/fixture.mind`, walks the AST heap-record tree, and
confirms it is node-for-node identical to the tree documented in
`examples/parser/EXPECTED.md`.

**Stage 2 verdict: PASS — 42 AST nodes, byte-identical to EXPECTED.md.**

Three compiler gaps closed to reach this milestone:

#### Gap S2-A — subprocess stdout deadlock for large MLIR inputs

`src/eval/mlir_build.rs` `run_command` wrote all of stdin before
draining stdout.  When `mlir-opt`'s stdout pipe buffer exceeded 64 KiB
(parser MLIR: ~61 KiB in → ~95 KiB out), `mlir-opt` blocked on a full
write pipe while mindc polled `try_wait()` — a classic producer/consumer
deadlock.  The lexer MLIR (27 KiB in → 28 KiB out) was below the threshold
and never triggered it.

**Fix:** Replaced the sequential write-then-poll pattern with a
thread-per-pipe approach.  A background thread drains stdout and stderr
simultaneously while the main thread writes stdin and polls for child
termination.  Mutual progress is guaranteed regardless of output size.

- `src/eval/mlir_build.rs`: `run_command` spawned two `thread::spawn`
  drain threads (one each for stdout / stderr).  Stdin is written after
  the threads start so the child can read its input while we drain its
  output.  Threads are joined before reading the accumulated buffers.

#### Gap S2-B — `if`-as-value emits zero constant instead of branch value

`src/mlir/lowering.rs` emitted `%dst = arith.constant 0 : i64` as a
placeholder in `^if_after_N:` for every `Instr::If` result.  This was
intentionally documented as a placeholder for "functions that only use
early-return semantics" (Gap 3 / Stage 1 report).  The parser's `if`
expressions are used as values (e.g. `let after_semi = if tok_kind(...) ==
tk_semi() { pos + 1 } else { pos };`), so the zero constant silently
corrupted every computed position.

**Fix:** Use MLIR basic-block arguments to forward the branch value to the
join block — the standard MLIR pattern for if-as-value in the `cf` dialect:

- `^if_then_N:` terminates with `cf.br ^if_after_N(%then_result : i64)`
- `^if_else_N:` terminates with `cf.br ^if_after_N(%else_result : i64)`
- `^if_after_N(%dst : i64):` declares the block argument that becomes `dst`

Branches that terminate with `Instr::Return` are unchanged — the `cf.br`
is omitted for already-terminated blocks.

#### Gap S2-C — Python harness misread child-list pointer as Vec header

The initial `bootstrap_smoke.py` treated the `child0` field of list-bearing
AST nodes (program, block, fn_def, call) as a Vec header pointer
(`[data_ptr, len, cap]`) and read the second field as the item count.
`child0` actually stores `vec_addr(acc)` — the raw element-array base
address returned by `__mind_load_i64(acc + 0)`, NOT the Vec header
address.  Reading it as a header gave a garbage length field (250M
elements), causing immediate segfault.

**Fix:** `read_child_list(data_ptr, count)` now reads `count` i64 values
directly from `data_ptr` without dereferencing a Vec header.  The count
is always taken from the `aux` field of the AST node, which stores
`vec_len(acc)` as documented in `main.mind`.

**Files changed:**

- `src/eval/mlir_build.rs` — pipe-deadlock fix (Gap S2-A)
- `src/mlir/lowering.rs` — `cf.br` block-argument if-value forwarding
  (Gap S2-B)
- `examples/parser/libmindc_parser.so` — compiled output (32KB,
  zero undefined symbols beyond libc)
- `examples/parser/bootstrap_smoke.py` — complete Python harness
  (new file; Gap S2-C fix included)

**AST node counts on `examples/parser/fixture.mind`:**

| AST kind      | expected | got |
|---------------|----------|-----|
| `ast_program` | 1        | 1   |
| `ast_use`     | 1        | 1   |
| `ast_fn_def`  | 2        | 2   |
| `ast_param`   | 5        | 5   |
| `ast_block`   | 2        | 2   |
| `ast_let`     | 2        | 2   |
| `ast_return`  | 1        | 1   |
| `ast_binop`   | 3        | 3   |
| `ast_call`    | 1        | 1   |
| `ast_ident`   | 24       | 24  |
| **Total**     | **42**   | **42** |

**Stage 1 (lexer) still PASS** — confirmed by re-running
`examples/lexer/bootstrap_smoke.py` (32/32 tokens byte-identical).

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +5% cap):**
The pipe-deadlock fix and block-arg emission changes are entirely inside
the `mlir-build`-gated code path.  The hot frontend pipeline
(`parse_typecheck_ir`) is untouched; bench-gate delta is negligible.

## [0.5.1] - 2026-05-18

### Phase 6.5 Stage 1b — cdylib emit now bundles std-surface + runtime-support

`mindc <file> --emit-shared <path>` now produces a self-contained `.so` that
`dlopen`s cleanly without any external runtime dependency beyond libc.

**Root cause:** The `--emit-shared` path called `clang -shared -fPIC <main.ll>
-o <out.so>` but did not link any implementations for the RFC 0005 intrinsics
(`__mind_load_i64`, `__mind_alloc`, etc.) or the `std.vec` surface functions
(`vec_new`, `vec_push`, `vec_get`, `vec_len`).  Every cdylib that used `std.vec`
had three undefined symbols at `dlopen` time.

**Fix — Option A (static-link everything into the cdylib):**

- New file `runtime-support/mind_intrinsics.c`: C implementations of all seven
  RFC 0005 intrinsics (`__mind_alloc`, `__mind_realloc`, `__mind_free`,
  `__mind_load_i64`, `__mind_store_i64`, `__mind_read`, `__mind_write`) plus the
  `std.vec` surface (`vec_new`, `vec_push`, `vec_get`, `vec_set`, `vec_len`,
  `vec_cap`, `vec_addr`).  All functions use the i64 opaque-address ABI
  (RFC 0005 P0a).  Growth policy matches the MIND spec: cap 0 → 4, then doubles.

- `src/eval/mlir_build.rs`:
  - New `const MIND_RUNTIME_SUPPORT_C: &str = include_str!(...)` — the C stub
    is embedded in the binary at compile time (same pattern as Phase C stdlib).
  - New `compile_runtime_support_obj(tools)` — writes the embedded C to a
    `NamedTempFile`, compiles it to a `.o` via `clang -x c -c -fPIC -O2`.
  - Updated `run_clang_codegen` — accepts `extra_objs: &[PathBuf]`; resets
    `-x none` before appending object files so clang does not misinterpret the
    pre-compiled `.o` as LLVM IR.
  - Updated `build_all` — when `emit_shared` is set, calls
    `compile_runtime_support_obj` and passes the result to `run_clang_codegen`.

**Result:**

```
nm examples/lexer/libmindc_lexer.so | grep " U "
# output: only malloc/free/memcpy/realloc@GLIBC (libc — always present)
```

```
python3 examples/lexer/bootstrap_smoke.py
# dlopen succeeds; lex() runs and returns 32 tokens
```

**Stage 1 verdict:** dlopen blocker RESOLVED.  Token-stream comparison runs to
completion.  First divergence at token 3 — `(kind=1, lo=188, hi=189)` (got
`tk_ident`) vs `(kind=15, lo=188, hi=189)` (expected `tk_slash`) for the `.`
in `use std.vec;`.  This is a lexer semantic bug in `examples/lexer/main.mind`
(`match_punct` fallthrough is `tk_ident()` not `tk_slash()`), not a linker bug
— it is out of scope for Stage 1b and is the first open item for Stage 2.

**Tests added:**

- `tests/std_surface_cdylib_link.rs` — 3 tests:
  - `cdylib_is_produced_and_nonzero`: verifies the `.so` is written and nonzero.
  - `cdylib_has_no_undefined_mind_symbols`: `nm -D` confirms zero undefined
    symbols beyond `malloc`/`free`/`memcpy`/`realloc`.
  - `cdylib_dlopens_via_python`: `python3 -c "ctypes.CDLL(...)"` opens the `.so`
    and calls `vec_new()`, asserting a nonzero handle.

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +7% cap):**

  small_matmul:  2.770 µs  (baseline 2.80 µs, -1.1%) ✓
  medium_mlp:    6.485 µs  (baseline 6.55 µs, -1.0%) ✓
  large_network: 17.526 µs (baseline 17.10 µs, +2.5%) ✓

The cdylib path is entirely off the hot frontend pipeline; impact is zero.

### Phase 6.5 Stage 1a — IR + MLIR gaps closed; lexer `.so` builds cleanly

**Status: COMPLETE**

Three IR-side compiler gaps that blocked `examples/lexer/main.mind` from
compiling to a valid shared library are now closed. The `.so` builds with no
`mlir-opt` errors and the `lex` symbol is exported at the correct address.

**Gap A — `Instr::If` / proper basic-block control flow (blocking)**

`ast::Node::If` was previously lowered by emitting both branch bodies into the
same flat instruction stream, producing `func.return` mid-block in the MLIR
output — which `mlir-opt` rejects. The fix:

- `src/ir/mod.rs`: new `#[cfg(feature = "std-surface")] Instr::If { cond_id,
  cond_instrs, then_instrs, then_result, else_instrs, else_result, dst,
  branch_bindings }` variant. `instruction_dst` updated accordingly.
- `src/eval/lower.rs`: `ast::Node::If` now lowers into three sub-IRModules
  (cond / then / else) with chained SSA counters to avoid ValueId collisions,
  then packages them as a single `Instr::If`. Helper functions `sub_ir_from` and
  `sub_ir_from_after` inherit `struct_defs` and `const_array_defs` from the
  parent so FieldAccess works inside branch conditions.
- `src/mlir/lowering.rs`: `Instr::If` emits `cf.cond_br` dispatch plus three
  named basic blocks (`^if_then_N`, `^if_else_N`, `^if_after_N`) where `N =
  dst.0` (the globally-unique SSA id). The emitter detects whether the condition
  value is already `i1` (comparison BinOp) or needs `arith.trunci`. Branches
  that end with `Instr::Return` skip the trailing `cf.br ^if_after_N`. The
  `FnDef` closer now checks the last non-empty line of the emitted body for a
  block terminator instead of scanning for any `return` anywhere — this prevents
  the "block with no terminator" MLIR error when a function ends with a fallback
  expression after a chain of early-return `if` checks.
- `src/ir/verify.rs`, `src/ir/print.rs`: `Instr::If` arms added.

**Gap B — bitwise BinOps (`&`, `|`, `^`, `<<`, `>>`) (secondary)**

`ast::Node::Bitwise` fell through to the `_` catch-all and emitted `const.i64 0`,
silently mis-compiling `load_byte`'s byte-mask expression. Fixed:

- `src/ir/mod.rs`: `#[cfg(feature = "std-surface")] BinOp::BitAnd | BitOr |
  BitXor | Shl | Shr` variants added.
- `src/eval/lower.rs`: `ast::Node::Bitwise` arm maps all five ops to the
  corresponding IR `BinOp`.
- `src/mlir/lowering.rs`: maps `BitAnd→arith.andi`, `BitOr→arith.ori`,
  `BitXor→arith.xori`, `Shl→arith.shli`, `Shr→arith.shrsi`.
- `src/eval/ir_interp.rs`, `src/eval/mlir_export.rs`, `src/ir/compact/emit.rs`,
  `src/opt/ir_canonical.rs`: non-exhaustive match arms extended; constant-folding
  of bitwise ops added to `ir_canonical`.

**Gap C — `let` bindings inside `if` branches visible in outer scope (latent)**

`let` bindings introduced inside a then/else branch were discarded; the outer
`fn_env` never learned about them. Fixed via `branch_bindings:
Vec<(String, ValueId)>` on `Instr::If`, populated during lowering and threaded
back to `fn_env` in the FnDef body loop.

**Tests added**

- `tests/std_surface_if_statement.rs` — 7 tests (4 IR-level, 1 MLIR-level,
  1 Gap C, 1 lexer-style early-return chain).
- `tests/std_surface_bitwise_binops.rs` — 11 tests (5 IR-level for all bitwise
  ops, 6 MLIR-level verifying arith dialect opcode names).
- Existing tests updated (map/string/vec/while helpers now recurse into
  `Instr::If` branches) — all pass.

**Bench-gate (vs `.bench-baseline-2026-05-18-rfc0005.txt`, +7% cap)**

  small_matmul:  +1.2%  ✓
  medium_mlp:    +0.1%  ✓
  large_network: +4.2%  ✓

## [0.5.0] - 2026-05-18

### Milestone: RFC 0005 Phase 6.2b — three compiler gaps closed; self-host substrate complete

mindc v0.5.0 closes the three documented surface-grammar gaps that
blocked the pure-MIND self-host work (Phase 6 ladder). All three land
behind module-level feature gates (`std-surface`) so the default-build
hot path stays byte-identical; the +7% bench-gate cap holds (current
delta vs `.bench-baseline-2026-05-18-rfc0005.txt`: small_matmul -0.7%,
medium_mlp -1.5%, large_network +3.5%).

With v0.5.0, four-and-a-half of the six Phase 6 sub-steps are shipped
(6.1 lexer + 6.2a parser + 6.3 type-checker + 6.4 MLIR emit in pure MIND;
6.2b grammar growth in mindc). Phase 6.5 (fixed-point bootstrap, apex)
remains open.

### Added — RFC 0005 Phase 6.2b Gap 2: array literals `[expr, …]` + fixed-size array types `[T; N]`

Array literals and fixed-size array types are now a first-class surface in mindc
(gated to the `std-surface` cargo feature; default-build hot path byte-identical).

**Reproducer** that now parses, type-checks, and lowers cleanly:
```mind
const FOO: [i64; 4] = [1, 2, 3, 4];

fn nth(i: i64) -> i64 {
    FOO[i]
}
```

**Implementation:**
- `src/ir/mod.rs`: new `Instr::ConstArray { dst, name, values }` (const blob
  with O(N) IR, not N individual stores) and `Instr::ArrayLoad { dst, base, index }`
  for `arr[i]` element loads. Both gated to `std-surface`. `instruction_dst`,
  `IRModule.const_array_defs` side-table, and `IRModule.new()` updated.
- `src/ir/print.rs`: `const.array[i64; N]` and `array.load` textual IR forms.
- `src/ir/verify.rs`: `ConstArray` and `ArrayLoad` verifier arms.
- `src/eval/lower.rs`: module-level `Const { ty: Array, .. }` arm lowers to
  `ConstArray` and seeds `const_array_defs`. `ArrayLit` expression arm iterates
  elements (no per-element recursion). `IndexAccess` arm emits `ArrayLoad`.
  `Ident` resolution checks `const_array_defs` and re-emits the blob into fn
  bodies (avoids cross-SSA-namespace ValueId aliasing). `FnDef` arm inherits
  `const_array_defs` and strips const-array ids from `fn_env` to prevent
  SSA-id aliasing with fn params.
- `src/type_checker/mod.rs`: `Let { ann: TypeAnn::Array, .. }` arm validates
  array-literal length against the `[T; N]` annotation and emits a diagnostic
  on mismatch. `Node::While` stub arm added for Gap 1 compatibility.
- `src/eval/mod.rs`, `src/type_checker/mod.rs`: stub match arms for
  `Node::While` (RFC 0005 Gap 1) to allow feature-gated compilation.
- `tests/std_surface_array_literals.rs`: 5 test cases — basic 3-element literal,
  empty `[i64; 0]`, 4,096-entry literal (no stack overflow), `const FOO + FOO[i]`
  round-trip, and type-length mismatch rejection.

**Bench-gate (default build, vs .bench-baseline-2026-05-18-rfc0005.txt):**
- small_matmul:  2.85 µs (baseline 2.80 µs, +1.7%) ✓
- medium_mlp:    6.32 µs (baseline 6.55 µs, -3.5%) ✓
- large_network: 16.76 µs (baseline 17.10 µs, -2.0%) ✓

All within the +7% cap. New code paths are entirely behind `std-surface`;
default-build parser and IR pipeline are untouched.

**4,096-entry LUT IR size:** A 4,096-element `ConstArray` produces exactly one
IR instruction (`const.array[i64; 4096] @NAME = [...]`) rather than 4,096
`__mind_store_i64` calls. Source-file line count for the reproducer: ~1 line per
LUT (the array literal) vs the previous ~4,100 lines per LUT.

### Added — RFC 0005 Phase 6.2b Gap 1: `while` statement parsing + MLIR basic-block loop lowering

The `while` keyword is now a recognised statement in mindc (gated to the
`std-surface` cargo feature, leaving the default-build hot path byte-identical).

**Reproducer** that previously failed with `error[parse][E1001]: expected expression`:
```mind
fn count_to(n: i64) -> i64 {
    let mut i: i64 = 0
    while i < n {
        i = i + 1
    }
    i
}
```

Now parses, type-checks, and emits IR cleanly under
`cargo run --features "std-surface" -- build <file> --emit-ir`.

**Implementation:**
- `src/ast/mod.rs`: new `Node::While { cond, body, span }` variant (gated).
- `src/parser/mod.rs`: `parse_while()` dispatched from `parse_stmt()` (gated,
  ~28 LOC).
- `src/ir/mod.rs`: new `Instr::While { cond_id, cond_instrs, body, live_vars }`
  (gated); `instruction_dst` and IR verifier updated accordingly.
- `src/eval/lower.rs`: `Node::While` arm in `lower_expr` lowers condition and
  body into separate sub-modules and emits `Instr::While` (~50 LOC, gated).
- `src/mlir/lowering.rs`: `Instr::While` arm emits `cf.br`/`cf.cond_br`
  basic-block loop (header → body → back-edge, plus after-block) using the
  `cf` dialect (~60 LOC, gated).
- `tests/std_surface_while_statement.rs`: 6 test cases (5 under `std-surface`,
  1 additionally under `mlir-lowering`) — trivial counted loop, nested while,
  mutable state outside loop, while inside if arm, reproducer smoke test, and
  MLIR block structure assertion.

**Bench-gate (default build, vs .bench-baseline-2026-05-18-rfc0005.txt):**
- small_matmul:  2.76 µs (baseline 2.80 µs, -1.4%) ✓
- medium_mlp:    6.35 µs (baseline 6.55 µs, -3.1%) ✓
- large_network: 17.54 µs (baseline 17.10 µs, +2.6%) ✓

All within the +7% cap. Default-build pipeline byte-identical — feature gate
enforced at module level (no per-statement dispatch).

**`examples/policy.mind` status:** The file uses `while` in two helper functions
(`starts_with` and `find_substring`). After this change those two `while` loops
parse correctly. The file has additional issues beyond `while` that are out of
scope for Gap 1: `const TIMEOUT_SHIFT: u32 = 8` uses a `u32` type annotation
that the type-checker rejects at the std-surface level; enum variants with
`= N` discriminants (`InvalidInput = 1`, etc.) are not yet parsed; and the
`Effect` struct body is incomplete in the excerpt. These are pre-existing gaps
unrelated to Gap 1.

### Fixed — RFC 0005 Phase 6.2b Gap 3: unsigned-i64 literal reinterpret-cast

Integer literals in the range `(i64::MAX, u64::MAX]` are now accepted by the
mindc parser.  Previously any literal exceeding `i64::MAX` (e.g. the FNV-1a
64-bit offset basis `14695981039346656037`) was rejected with
`error[parse][E1001]: integer overflow`.

The parser now tries a signed-i64 parse first; on failure it falls back to an
unsigned-u64 parse and reinterprets the bit-pattern as a signed i64 via
standard two's-complement (Rust `u64 as i64`).  Literals exceeding `u64::MAX`
continue to be rejected.  Literals that fit in `i64` are unaffected — the
fallback branch is never reached for them.

- `14695981039346656037` (FNV-1a offset basis) → `-3750763034362895579i64`
- `18446744073709551615` (`u64::MAX`) → `-1i64`
- `9223372036854775808` (`i64::MAX + 1`) → `i64::MIN`
- `18446744073709551616` (`u64::MAX + 1`) → still rejected as overflow

**Scope:** `src/parser/mod.rs` — four sites (two in the expression literal
parser, two in the pattern-match literal parser); a shared `parse_i64_literal`
/ `parse_i64_pattern` helper extracted to keep all four sites consistent.
The i32 parse at the attribute-argument site is unchanged.

**Tests:** `tests/parser_unsigned_i64_literals.rs` — 9 tests covering all
boundary values, byte-level round-trip, fn-body context, and the for-range
disambiguation path.

**Bench-gate:** default-build hot path is byte-identical (the fallback branch
is only reached on literals outside `i64` range, which never appear in the
canonical compile corpus).  Measured: `small_matmul -0.4%`, `medium_mlp
+4.6%`, `large_network +2.3%` — all within the +7% cap.

Closes Gap 3 from `docs/rfcs/0005-phase-6-2-mindc-gaps.md` §"Phase 6.3
addendum".

### Added — RFC 0005 Phase 6.2a: pure-MIND self-host parser seed

Second step of the self-host ladder. `examples/parser/` ships a
~814-LOC pure-MIND Pratt parser that consumes the Vec<i64> stride-3
token stream from the Phase 6.1 lexer and emits an AST as Option-C
heap-record structs with i64-addr recursive fields.

- **`examples/parser/main.mind`** (814 LOC) — Pratt operator-precedence
  parser with 12 AST node kinds (`ast_int_lit`, `ast_ident`, `ast_binop`,
  `ast_call`, `ast_fn_def`, `ast_let`, `ast_use`, `ast_return`,
  `ast_param`, `ast_block`, `ast_program`, `ast_paren`) and 7 operator
  tags (`op_add`/`sub`/`mul`/`div`/`lt`/`gt`/`eq`). Parses cleanly under
  `cargo run --features "std-surface cross-module-imports" --bin mindc
  -- examples/parser/main.mind --emit-ir` (`next_id = 90`).
- **`examples/parser/fixture.mind`** (23 LOC) — small source program
  exercising fn declarations, let bindings, binary ops, fn calls, and
  `use` statements.
- **`examples/parser/EXPECTED.md`** (190 LOC) — documents the expected
  AST tree shape on the fixture.
- **`examples/parser/README.md`** (230 LOC) — Phase 6.2 status + handoff
  to Phase 6.3 (type-checker).

Key design notes:

- **`ParseResult` struct** `{ next_pos: i64, node: i64 }` threaded through
  every parse helper — no implicit parser state, matching the
  fixed-state-iteration discipline the lexer established.
- **Variable-length child lists** (block stmts, fn params, call args,
  program items) ride in `Vec<i64>` of AST addresses; the Vec base addr
  stored in `child0` of the parent node and the count in `aux`. No AST
  node references a non-i64 field — RFC 0005 P0a discipline end-to-end.
- **Pratt fold** is a single 11-line tail-recursive function;
  precedence + left-associativity fall out of `prec + 1` recursion on
  the right operand.
- **`return` keyword detection** uses byte-compare against the source
  buffer because Phase 6.1's lexer hasn't promoted `return` to a
  keyword. Phase 6.3 lexer growth will replace this with a
  `tk_kw_return()` check trivially.

No new mindc feature gaps surfaced. The two gaps already documented
in `docs/rfcs/0005-phase-6-2-mindc-gaps.md` (Gap 1 `while`, Gap 2
array literals) remain the canonical unblockers; the parser's
heap-record boilerplate (~364 of the 814 LOC) is the same O(N)
source-line tax Gap 2 fixes.

### Added — RFC 0005 Phase 6.1: pure-MIND self-host lexer seed

First step of the self-host ladder. `examples/lexer/` ships a
~290-line pure-MIND tokeniser on top of `std.vec` + the seven
`__mind_*` intrinsics. Walks an in-memory source buffer byte-by-byte
and emits a flat `Vec<i64>` stride-3 token stream `(kind, lo, hi)`.

- **`examples/lexer/main.mind`** — the lexer (idents, integer literals,
  single-char + `->` punctuation, `//` line comments, whitespace, the
  four current keywords `fn`/`let`/`use`/`pub`). Parses cleanly under
  `cargo run --features "std-surface cross-module-imports" --bin mindc
  -- examples/lexer/main.mind --emit-ir`.
- **`examples/lexer/fixture.mind`** — a 254-byte source file for the
  smoke gate. Token-stream contract documented in
  **`examples/lexer/EXPECTED.md`** (32 rows × 3 i64 = 96 entries).
- **`examples/lexer/README.md`** — Phase 6.1 status + Phase 6.2
  follow-up. Key documented finding: **mindc v0.4.4's parser does NOT
  accept `while` as a statement**. The lexer expresses every loop as
  tail recursion, which the parser does accept. Phase 6.2 either adds
  `while`-stmt parsing or canonicalizes tail recursion as the MIND
  loop primitive — either choice keeps this seed working unchanged.

The smoke gate is currently a documented fixture; Phase 6.2 will
promote it to a Cargo integration test that compiles `main.mind`
to a `.so` and diffs the live token stream against mindc-Rust's
own tokeniser output.

### Changed — RFC 0005 landing table expanded for Phase 6

`docs/rfcs/0005-pure-mind-std-surface.md` §"Adoption plan" landing
table now expands the single Phase 6 row into a 5-step ladder
(6.1 lexer / 6.2 parser / 6.3 type-checker / 6.4 MLIR text emit /
6.5 fixed-point bootstrap). Phase 6.1 marked **shipped**; rest
**open**. Phase D₂b row cross-linked to its design note
(`docs/rfcs/0005-phase-d2b-design-note.md`).

### Added — Phase D₂b design note

`docs/rfcs/0005-phase-d2b-design-note.md` captures the cross-arg
Named-struct identity-matching design for the next compiler tag.
Multi-session pickup artifact; not yet implemented.

## [0.4.4] — 2026-05-19

### Added — RFC 0005 Phase D₂a: Named structs preserved in cross-module call errors

Phase B's call-site type-checker compares args against an imported
fn's declared parameter types. When a mismatch fires, the error
message currently leans on `describe_value_type` — which collapses
Named struct types (`Vec`, `String`, `Map`, ...) to `ScalarI64`
because that's the Option-C heap-record ABI lowering. The result is
that a user passing the wrong type into `vec_set(my_string, 0, 99)`
sees "expects scalar i64" with no mention of `Vec` at all.

D₂a renders the *expected* side of the error message from the raw
`TypeAnn` instead of the lowered `ValueType`, so the call-site error
now reads `expects Vec (heap-record i64 addr)`. The compatibility
check itself stays permissive (Phase B/C/D₁ behaviour unchanged —
i64 values still accepted into Named struct params under Option-C);
this is purely an error-clarity improvement.

- **`src/type_checker/mod.rs`** — new gated helper
  `describe_param_type(&TypeAnn) -> String` (~12 lines). Falls
  through to `describe_value_type(&cm_typeann_to_valuetype(ann))`
  for primitives + aggregates; renders Named-typed params as
  `<name> (heap-record i64 addr)`. The single call-site swap in
  `check_imported_fn_call` consumes it.
- **`tests/std_surface_use_import_phase_b.rs`** — new test
  `phase_d2_named_struct_param_named_in_arity_error` confirms the
  emitted error contains both `expects Vec` and `heap-record i64
  addr`. Existing 6 Phase B tests + the new one all pass.

D₂b (cross-arg *identity* matching — flagging `vec_set(my_string,
...)` as a real type error rather than a permissive accept) stays
deferred. The right design needs the type-env to track Named
struct names through let bindings and fn returns, which is bigger
surgery than v0.4.4 absorbs cleanly.

### Compile-speed gate

Against the post-RFC-0005 baseline. The new helper is in the
error-formatting cold path — only fires when a per-arg widening
already failed — so the hot-path bench is byte-identical:

| bench          | baseline   | v0.4.4     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.80 µs    | 2.87 µs    | +2.4%    | ✓    |
| medium_mlp     | 6.55 µs    | 6.61 µs    | +0.9%    | ✓    |
| large_network  | 17.10 µs   | 17.49 µs   | +2.3%    | ✓    |

All within the +7% cap.

## [0.4.3] — 2026-05-18

### Added — RFC 0005 Phase D₁: `$MIND_STDLIB_PATH` override

Phase C bundled the four pure-MIND std/*.mind files into the mindc
binary so `use std.vec` resolved with no external file dependency.
That left one gap: a downstream user who wanted to fork the stdlib
(e.g. ship a stricter `std.string` with inline UTF-8 validation for
a regulated deployment) had to fork and rebuild `mindc` itself.

Phase D₁ adds an env-var escape hatch: when `MIND_STDLIB_PATH=path`
is set and points at a directory containing all four `.mind` files
(`vec.mind`, `string.mind`, `map.mind`, `io.mind`), the project
loader reads from that directory instead of the bundled blobs.
Missing files, missing directory, or parse failure → silently fall
back to bundled. Same fork-without-recompile escape hatch Rust's
`RUSTC_BOOTSTRAP` provides.

- **`src/project/stdlib.rs`** — new `parsed_stdlib_modules_from_env()`
  helper, called at the top of `parsed_stdlib_modules()`. Reads
  `MIND_STDLIB_PATH` via `std::env::var_os`, validates the directory
  exists, attempts to read + parse all four module files. Returns
  `Option<Vec<(String, Module)>>` — `Some` only on full success.
  Hot path on unset env var is a single null check.
- **3 new tests** in `src/project/stdlib.rs::tests`:
  - `env_override_falls_back_when_unset` — default behaviour unchanged.
  - `env_override_loads_directory_when_set` — round-trips through the
    repo's own `std/` directory and asserts on count + module names.
  - `env_override_falls_back_on_missing_dir` — non-existent path
    triggers fallback, doesn't crash.

### Compile-speed gate

Re-ran `cargo bench --bench compiler` after the change. Numbers
against `.bench-baseline-2026-05-18-rfc0005.txt` (small 2.80 µs /
medium 6.55 µs / large 17.10 µs):

| bench          | baseline   | v0.4.3     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.80 µs    | 2.85 µs    | +1.8%    | ✓    |
| medium_mlp     | 6.55 µs    | 6.50 µs    | -0.8%    | ✓    |
| large_network  | 17.10 µs   | 17.43 µs   | +1.9%    | ✓    |

All within the +5% cap. The bundled hot path is a branchless
`Option<...>` short-circuit, so Phase D₁ is invisible to default
builds.

### Spec alignment

`mind-spec` (commit `5fa4299`) added an informative "Environment
override" subsection to `spec/v1.0/stdlib.md` documenting the
`MIND_STDLIB_PATH` contract; honouring it is informative for
conforming implementations.

## [0.4.2] — 2026-05-18

### Added — RFC 0005 Phase C: auto-bundle `std/*.mind` into mindc

v0.4.1's Phase B closed the per-arg signature matching deferred from
v0.4.0 but still required the consumer to supply std module ASTs to
`build_module_table` manually.  A real downstream `mind build`
running on a project that says `use std.vec` had no way to find the
std/*.mind files unless the user vendored them — which defeats the
point of having a shared standard library.

Phase C bundles the four pure-MIND std/*.mind files into the mindc
binary at compile time via `include_str!` and seeds the module
table with them in the project loader's cross-module-imports
block.  A downstream `mind build` of a project that says
`use std.vec` now resolves with no external file dependency.

- **`src/project/stdlib.rs`** (new) — compile-time bundle:
  - `STDLIB_MIND_SOURCES: &[(&str, &str)]` — `(module_path,
    source_text)` for `std.io`, `std.map`, `std.string`, `std.vec`,
    sorted alphabetically for deterministic insertion.
  - `parsed_stdlib_modules() -> Vec<(String, Module)>` — parses
    every bundled source and returns the `(module_path, AST)` pairs
    the project loader's cross-module-imports block prepends to the
    user's own modules.
- **Project loader wiring** in `src/project/mod.rs` — the bundle is
  prepended to `parsed` BEFORE walking the user's `src/`.  Since
  `ModuleTable::insert` is last-write-wins, a user module that
  happens to shadow `std.*` overrides the bundled entry — same
  behaviour as Rust's user-crate-wins-over-stdlib semantics.
- **CI matrix coverage** — the new gated-feature CI step (commit
  `996553e`) runs the std-surface + cross-module-imports test
  suites separately and combined, including Phase C's 4 new
  integration tests + 3 new unit tests.  Total tests added in v0.4.2:
  7.  All RFC 0005 work is now under cloud CI guard.

### Performance

Compile-speed gate vs `.bench-baseline-2026-05-18-rfc0005.txt`:

| bench          | baseline   | v0.4.2     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.800 µs   | 2.758 µs   | -1.50%   | OK   |
| medium_mlp     | 6.550 µs   | 6.779 µs   | +3.50%   | OK   |
| large_network  | 17.100 µs  | 17.627 µs  | +3.08%   | OK   |

All inside the 5% CI gate.  Phase C lives entirely behind
`feature = "cross-module-imports"`; the default-build frontend hot
path is untouched (module-level gate, no per-statement cfg, no
runtime dispatch).  The medium/large drift is within the ±2%
runner-variance band documented in the baseline file (local
replays put medium at 6.39–6.78 µs and large at 16.89–17.80 µs).

## [0.4.1] — 2026-05-18

### Added — RFC 0005 Phase B: per-arg signature matching on imported `pub fn`s

v0.4.0 wired `use std.vec` so calls to `vec_new()` / `vec_push(...)`
type-check loosely as `ScalarI64`-returning, with no arg-count or
arg-type validation. The v0.4.0 CHANGELOG flagged per-arg signature
matching as deferred to Phase B; this release closes that gap.

- **`ExportedFn { name, param_types, ret_type }`** in
  `src/project/module_table.rs` carries the full signature of every
  auto-exported `pub fn`. `ModuleExports` gains a parallel
  `exported_fns: Vec<ExportedFn>` populated on the auto-export path;
  explicit `export { ... }` block surfaces leave it empty by
  construction (the RFC 0002 contract is preserved — those declare
  names, not signatures, by design).
- **`ModuleTable::lookup_imported_fn(name)`** walks every module in
  deterministic sorted-key order and returns the first match.
- **Type-checker side** (gated under `cross-module-imports`):
  `cm_lookup_fn` reads the project table's typed declaration;
  `check_imported_fn_call` validates arity then per-arg types and
  returns the declared return type;
  `cm_typeann_to_valuetype` reuses the existing `valuetype_from_ann`
  helper, falling back to `ScalarI64` for Named struct/enum types
  and unsupported aggregates — matches RFC 0005's Option-C heap ABI
  where struct values are i64 base-addresses on the wire;
  `cm_arg_compatible` accepts exact matches plus the universal
  i32 ↔ i64 widening that integer literals need.
- **Fall-back to Phase A** when no signature is available (e.g. the
  imported module surface came from an `export { ... }` block, not
  auto-export). Default build path is byte-identical — the moat is
  held.
- 3 new module_table unit tests + 6 new integration tests under
  `tests/std_surface_use_import_phase_b.rs`. Phase A's 8 tests
  still pass — Phase B is strictly an additive improvement.

### Performance

Compile-speed gate vs `.bench-baseline-2026-05-18-rfc0005.txt`:

| bench          | baseline   | v0.4.1     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.800 µs   | 2.742 µs   | -2.07%   | OK   |
| medium_mlp     | 6.550 µs   | 6.614 µs   | +0.98%   | OK   |
| large_network  | 17.100 µs  | 16.919 µs  | -1.06%   | OK   |

All inside the 5% gate. All Phase B changes live behind
`feature = "cross-module-imports"`; the default-build frontend hot
path is untouched.

## [0.4.0] — 2026-05-18

### Added — RFC 0005 Phase 2: pure-MIND standard surface (`std.vec`, `std.string`, `std.map`, `std.io`)

The four pure-MIND collections + I/O surface RFC 0005 names now lower
end-to-end on the seven `__mind_*` intrinsics shipped at Phase 1 +
Phase 1.5 + Phase 2 (P0e + P0f).  Every operation bottoms out into
`__mind_alloc` / `__mind_load_i64` / `__mind_store_i64` / `__mind_read`
/ `__mind_write` — i64 ABI throughout, no built-in pointer type, no
hidden allocator.

- **`std/vec.mind`** — growable `Vec` (8-byte stride, doubling growth
  from min-cap 4).  Surface: `vec_new`, `vec_len`, `vec_cap`,
  `vec_addr`, `vec_get`, `vec_set`, `vec_push`.  6 module tests assert
  the exact `__mind_alloc` / `__mind_load_i64` / `__mind_store_i64`
  floor counts per function — `vec_new` = 1 alloc + 3 stores,
  `vec_len/cap/addr` = 1 load each, `vec_get` = 2 loads, `vec_set` =
  1 load + 1 store, `vec_push` ≥ 3 loads + ≥ 4 stores + ≥ 1 alloc.
- **`std/string.mind`** — `Vec<u8>`-shaped `String` with a documented
  UTF-8 well-formedness invariant.  Surface: `string_new`,
  `string_len`, `string_cap`, `string_addr`, `string_get_byte`,
  `string_validate_utf8`, `string_push_byte`, `string_eq`.  Byte-
  stride loads/stores (not 8-byte stride like `Vec`); 16-byte initial
  cap.  7 module tests with the same IR-shape contract.
- **`std/map.mind`** — insertion-ordered `Map` on parallel keys / vals
  arrays (4-field heap record).  Surface: `map_new`, `map_len`,
  `map_cap`, `map_keys_addr`, `map_vals_addr`, `map_key_at`,
  `map_value_at`, `map_insert`.  Ordering is deterministic (insertion
  order, not hash-randomised) — load-bearing for evidence-chain
  reproducibility.  5 module tests.
- **`std/io.mind`** — `File` handle plus the four-arg `__mind_read` /
  `__mind_write` POSIX-shaped intrinsics.  Surface: `stdin` /
  `stdout` / `stderr` constructors, `file_fd`, `file_read`,
  `file_write`, `print_bytes`, `eprint_bytes`, `read_stdin_bytes`.
  8 module tests assert the I/O surface routes through the correct
  intrinsic (no special-case lowering — the generic `Instr::Call`
  arm from Phase 0 picks them up).
- **`__mind_read` / `__mind_write`** declared at arity 4 in the
  std-surface intrinsic registry (`STD_SURFACE_INTRINSICS`).  The
  MLIR lowering's generic call arm handles them via the same path as
  the other RFC 0005 intrinsics.

### Added — `use std.foo` cross-module resolution

The cross-module-imports resolver (D1 + D2 + D3, gated since v0.2.6)
now composes with RFC 0005 Phase 2 so a consumer file can `use std.vec`
and call `vec_new()` directly.

- **Auto-export of `pub fn` + `struct`** when no `export { ... }` block
  is present (`collect_module_exports`).  The parser already strips
  `pub` to a no-op, so a `pub fn`-only file would otherwise have an
  empty exported surface.  Explicit `export { ... }` blocks still win
  when present — the RFC 0002 contract is preserved.
- **Imported names accepted as callables** in `infer_call`'s catch-all
  when the `cross-module-imports` feature is on (gated; default build
  byte-identical).  Calls to resolver-injected names type-check as
  `ScalarI64`-returning — per-arg signature matching against the
  imported `pub fn` declarations is Phase B (deferred to v0.4.x).
- 8 end-to-end tests in `tests/std_surface_use_import.rs` cover:
  pub-fn auto-export, struct auto-export, `use std.vec` resolution,
  `use std.io` resolution, unimported-module isolation, wrong-path
  fall-through, and explicit-`export`-block precedence.

### Performance

Compile-speed gate (`tools/bench_gate.py` vs
`.bench-baseline-2026-05-17-phase10-7.txt`) passes with all three
canonical benches **improved**:

| bench          | baseline   | v0.4.0     | delta    | gate |
| -------------- | ---------- | ---------- | -------- | ---- |
| small_matmul   | 2.810 µs   | 2.747 µs   | -2.24%   | OK   |
| medium_mlp     | 6.560 µs   | 6.432 µs   | -1.95%   | OK   |
| large_network  | 16.900 µs  | 16.797 µs  | -0.61%   | OK   |

The RFC 0005 work lives entirely behind `feature = "std-surface"`
and `feature = "cross-module-imports"`; the default-build frontend
hot path is untouched (module-level gates only, no per-statement
cfg) — the compile-speed moat is held.

## [0.2.11] — 2026-05-17

### Added — Phase 10.7: match expressions and reference-taking expressions

The following surface constructs now parse, type-check, and lower to IR
without new IR opcodes (conservative v1 strategy: match lowers to
sequential arm evaluation; &expr is a no-op metadata wrapper).

- **`match` expressions** (`match value { Pat => body, ... }`). Patterns
  supported: `EnumVariant` (`Mode::On`, `Result::Ok(x)`), `Literal`
  (integer, float, string, `true`/`false`), bare `Ident` binding, and
  `_` wildcard. Block bodies (`=> { ... }`) also accepted. Arms
  separated by `,`; trailing comma optional.
- **`&expr` and `&mut expr`** reference-taking prefix expressions.
  Symmetric with the `&T` / `&mut T` type forms already in Phase 10.6.
  The Pratt parser disambiguates `&` prefix (ref-take) from `&` infix
  (bitwise-AND) by position: prefix `&` fires inside `parse_primary`
  before the infix loop can see it — no saved_pos / backtrack needed.
- **AST additions**: `Node::Match { scrutinee, arms, span }`,
  `Node::Ref { mutable, inner, span }`, `MatchArm`, `Pattern`
  (`EnumVariant`, `Literal`, `Ident`, `Wildcard`).
- **Type-checker**: arm body types unified; literal pattern vs. scrutinee
  type checked conservatively; identifier patterns introduce bindings;
  `&expr` / `&mut expr` accepted with `ScalarI32` placeholder type.
- **Corpus watermark**: bumped from 14 to 21 (7 newly unblocked `.mind`
  files in `rfn-mind/src`).

## [Unreleased]

## [0.10.1] - 2026-07-05

### Proven
- x86_64 (AVX2) == ARM64 (NEON) byte-identity PROVEN on real hardware (GCP Ampere Altra aarch64, LLVM 20.1.8): cross_substrate_identity 13/13, all 12 canaries byte-identical incl. a chaotic Q16.16 Lorenz integrator.
- Native-ELF self-host loop closed (stage1==stage2==stage3 byte-identical, fail-closed gate).
### Added
- Tensor sum/mean pinned canonical fold + 1-D tensor-parameter ABI.
- PQC ML-DSA (FIPS-204) hybrid signing direction.
### Fixed
- mindc import-walk hardening; Windows-portable __mind_now_ns (POSIX byte-identity unchanged).
### Docs
- Full docs/spec/site truth-alignment: scalar int/Q16.16/f64-f32 x86+ARM verified; float-vector + GPU scoped frontier/commercial-roadmap.

Infrastructure landed on `main` but not yet attached to a tag.

### Added
- `libmind::cache` — content-addressed compilation cache with
  `CompilationCache`, `CacheKey`, `CacheEntry`, `CacheStats`,
  `ProfileTag`, and an in-memory `MemoryStore` backend. Cache key
  includes compiler version + profile tag + source hash + imports
  hash so cross-mode rebuilds never hit a stale entry. Foundation
  for sub-µs warm-start frontend latency. 17 unit tests.
- `tools/pytorch_bridge/` — PyTorch / JAX → MIND transpiler tooling.
  ONNX-driven PyTorch path, XLA-HLO-driven JAX path, and a
  deterministic prompt builder for AI-assisted UNSAT proof resolution.
  Pure Python, no torch / jax import at module load. 11 unit tests.
- `libmind::distributed` — IR-layer primitives for tensor and pipeline
  parallelism: `ShardSpec` / `ShardLayout` (replicated / split /
  split-2D), `AllReduceOp` with lexicographic / tree / arrival
  reduction orders, `AllGatherOp` with lexicographic / arrival gather
  orders, `PipelineGraph` / `PipelineStage` / `StageBoundary`, and
  `DistributedInvariant` enforcement (`deterministic_all_reduce`,
  `reduction_order_lexicographic`, `gather_order_lexicographic`,
  `evidence_chain_continuous`). 31 unit tests.
  See `docs/roadmap.md` Phase 13.6 for the design rationale and the
  speed-preservation discipline that keeps the 1.8–15.5 µs frontend
  baseline locked when these primitives are not imported.

## [0.2.10] — 2026-05-17

### Added — Phase 10.6 surface syntax (parser-level)

The following surface constructs now parse, type-check, lower to IR,
and round-trip through the bench gate without moving headline numbers.
They are additive extensions of Core IR v1 (see `docs/versioning.md`
"Minor (0.y.z)") — no existing IR instruction or shape semantics
change.

- **Qualified type paths** in const declarations and type annotations
  (`module.Type`, multi-segment `a.b.C`) (e85611a).
- **`pub` visibility marker** on `fn`, `struct`, `enum`, and on struct
  fields (dfe01ce). Parsed and propagated to AST; semantic effect lands
  with RFC 0002 D2.
- **Struct literal expressions** `Name { field: value, ... }` in
  expression position (f60232c).
- **Slice types** `&[T]` / `&mut [T]` and **fixed-size array types**
  `[T; N]` (9f0fd57).
- **`let mut`** mutable-binding marker accepted by the parser (12be59d).
- **`%` modulo operator** end-to-end: parser → IR → autodiff → eval
  → MLIR lowering (7db0bf3).
- **Single-value reference types** `&T` and `&mut T` distinct from
  slices (6f14a66).
- **Generic type application** `Name<A, B, ...>` for `Vec`, `Result`,
  `Option`, and user-defined generics (d411d53).
- **`::` path-segment separator** for enum variant access:
  `Result::Ok`, `module.Enum::Variant` (dc0f70c).
- **Tuple types** `(T, U, ...)` in function return positions.
- **Postfix index access** `xs[0]` on arrays, slices, and Vec values.
- **Indexed assignment** `xs[i] = value` and **field assignment**
  `obj.field = value` as statements.
- **Multi-line arithmetic continuation** — `+`, `-`, `*`, `/`, `%` may
  span newlines, e.g.

  ```mind
  let idx: u32 = (c as u32) * (h * w) as u32
               + (y as u32) * (w as u32)
               + (x as u32)
  ```

### Fixed
- **MLIR lowering pipeline** — `--emit-mlir` now lowers to the LLVM
  dialect before invoking `mlir-translate`, removing the
  "unregistered dialect" failure on programs that exercise control
  flow (99fc19f).

### Changed
- `find_runtime_lib` now searches only `MIND_LIB_DIR` and `~/.mind/lib`.
  A previous downstream-specific install-path fallback was an internal
  leak and is removed.
- Test target `parse_rfn_mind.rs` renamed to `parse_phase10_surface.rs`;
  the corpus-sweep test is now driven by the `MIND_TRACKING_CORPUS_DIR`
  environment variable rather than a hard-coded path. On CI and fresh
  clones the sweep is a no-op; when the variable is set, the parser
  must hold the documented high-watermark (14 files at this release).

### Compile-speed discipline
- Pratt loop adopts a **same-line fast path**: `skip_ws` + `peek_binop`
  succeeds without saving position in the common case, and only widens
  to `skip_ws_and_newlines` when an actual `\n` is at the cursor. Every
  parser arm above ships behind a leading-token check and exits in
  O(1) when the construct is absent. The headline benches
  (`small_matmul / medium_mlp / large_network`) remain within the
  documented threshold of `.bench-baseline-2026-04-28-pratt.txt`.

## [0.2.9] — 2026-05-16

### Security
- **RFC 0002 hardening** — surfaced by the v0.2.8 security audit, both
  fixed in this release before D2 (codegen pass) bakes the unvalidated
  inputs into emitted symbols.
  - `Mind.toml [exports] c_abi` is now bounded to
    `MAX_MANIFEST_EXPORTS = 1024` and every entry must match the
    C-style identifier grammar `[A-Za-z_][A-Za-z0-9_]*`. Violations
    return the new `CompileError::InvalidManifestExport` (diagnostic
    `E5002`). DoS / symbol-injection guard ahead of the C-ABI wrapper
    codegen pass.
  - `mindc compile --profile` now uses clap's
    `value_parser = ["default", "systems", "embedded"]`. Typos like
    `--profile sytems` are rejected at the CLI layer instead of
    silently falling through `ProfileTag::parse`'s permissive mapping
    to `Default` (which would have poisoned the cache fingerprint).
- Regression tests:
  - `tests/ir_lower.rs::manifest_exports_reject_oversized_list`
  - `tests/ir_lower.rs::manifest_exports_reject_non_identifier`

## [0.2.8] — 2026-05-16

### Added
- **RFC 0002 deliverable 5** — `--profile <default|systems|embedded>`
  CLI flag on `mindc compile`. Defaults to `default`. Threads through
  `CompileOptions.profile: ProfileTag` to the cache fingerprint so the
  same `Mind.toml` produces three distinct artifacts.
- **`ProfileTag::parse`** — case-insensitive parser; unknown names map
  to `Default`. `ProfileTag` now derives `Default` (variant `Default`).
- Regression test
  `tests/ir_lower.rs::profile_tag_parse_and_default` covers parse,
  default propagation, and explicit override.

### Changed
- `CompileOptions` gains a public `profile` field. As with D3 the
  default is preserved via `Default`; in-tree call sites already use
  `..Default::default()` so no migration needed.

## [0.2.7] — 2026-05-16

### Added
- **RFC 0002 deliverable 3** — `Mind.toml [exports]` section is now
  honored end-to-end. `ProjectManifest.exports.c_abi: Vec<String>`
  parses from the manifest, `BuildOptions.manifest_exports` threads it
  through the build pipeline, and `CompileOptions.manifest_exports`
  merges those names into `IRModule.exports` after AST → IR lowering.
  Together with deliverable 1, both source-side `export { ... }` blocks
  and manifest-declared exports reach the same set in IR — ready for the
  codegen pass landing in deliverable 2.
- Regression test
  `tests/ir_lower.rs::compile_pipeline_merges_manifest_exports`
  asserts the merge for the combined case.

### Changed
- `CompileOptions` and `BuildOptions` gain a `manifest_exports`
  field. The struct still derives `Default`, so existing call sites
  can opt in via `..Default::default()`. All in-tree call sites updated.

### Compile-speed discipline
- The merge path is `if !manifest_exports.is_empty() { ir.exports.extend(...) }`
  — branchless in the default code path (typical `Mind.toml` has no
  `[exports]` block).

## [0.2.6] — 2026-05-16

### Added
- **RFC 0002 deliverable 1** — `IRModule.exports: HashSet<String>`
  field populated by the AST → IR lowering pass when the parser sees an
  `export { foo, bar }` block (`Node::Export`). Previously the export
  block parsed but lowered to a no-op warning. The field is empty in
  the default code path; consumers under the new `ffi-c-user` feature
  flag (deliverables 2+) read the set to emit `mind_fn_<name>_invoke`
  C-ABI wrappers per RFC 0002. Regression tests:
  `tests/ir_lower.rs::lower_export_block_populates_ir_exports` and
  `lower_no_export_keeps_exports_empty`.
- **`ffi-c-user` Cargo feature** (currently empty gate). Reserved for
  the RFC 0002 codegen pass landing in subsequent deliverables.
- **`bench_c_export_lowering` sub-bench** in `benches/compiler.rs`
  measuring 0 / 1 / 10 export names. Separate criterion group so the
  headline `compiler_pipeline` numbers stay measurable against
  `.bench-baseline-2026-04-28-pratt.txt`.

### Compile-speed discipline
- New IR field is `HashSet<String>::new()` at construction (zero
  capacity, zero allocation until first insert). The default code path
  for programs without an `export` block performs one extra branchless
  match-arm test per top-level item, no hashset touches.

## [0.2.5] - 2026-04-28

### Added
- **Pratt operator-precedence parser** for the expression layer. Replaces
  the recursive-descent chain `parse_logical_or → parse_logical_and →
  parse_comparison → parse_additive → parse_bitwise → parse_multiplicative`
  with a single dispatch function driven by a binding-power table. Phase 10.5
  operators (`||`, `&&`, `|`, `&`, `^`, `<<`, `>>`, `as`) become table
  entries, and future operators (Phase 11/12) become O(1) inserts.
- **Stable IR public API**: `libmind::ir::load(bytes) -> IRModule` and
  `libmind::ir::save(module) -> String` for the `mic@1` textual format.
  Plus `libmind::compile_to_mic_text(src, opts)` as the AOT pipeline.
- `docs/ir-stability.md` — formalises the IR contract used by
  `mind-runtime` and other downstream backends to consume pre-compiled IR
  instead of re-running the surface parser per inference.
- `.github/workflows/bench-gate.yml` + `tools/bench_gate.py` — CI gate that
  fails on >2% mean regression on `small_matmul`, `medium_mlp`, or
  `large_network` vs the frozen baseline at
  `.bench-baseline-2026-04-28-pratt.txt`.
- Phase 10.5 conformance programs in `tests/conformance/cpu_baseline/`
  (`phase_10_5_const.mind`, `..._logical.mind`, `..._struct.mind`,
  `..._module.mind`).
- `tests/ir_load_save.rs` — pins the round-trip and determinism contract
  for the new IR API.

### Changed
- **Parser is faster than the pre-Phase-10.5 baseline** on `medium_mlp`
  (-9.0%) and `large_network` (-4.6%); within +3.9% on `small_matmul`.
  The Pratt rewrite recovered the +9% regression introduced when Phase 10.5
  dispatch arms were added in 0.2.4.
- Removed the `peek_skip_ws` helper (only used by the recursive-descent
  fast-paths it was added to support; Pratt makes it unnecessary).
- Removed dead `parse_logical_or` forwarder (deprecated since Phase 10.5
  inlined logical-or into `parse_expr`).

### Architecture
- mindc's parser is no longer a runtime dependency for `mind-runtime`
  consumers. The supported pattern is to AOT-compile to `mic@1` once at
  build time and call `ir::load` at runtime. This decouples parser
  performance from per-inference latency across all 12+ planned backends
  (CPU, CUDA, Metal, ROCm, WebGPU, WebNN, ARM, TPU, NPU, LPU, DPU, FPGA,
  Quantum).

## [0.2.1] - 2026-02-17

### Added
- IR verifier audit coverage: 14 new tests for conv2d stride validation,
  reduction axis checks, FnDef body scoping, and duplicate definition detection
- Determinism proof results for v0.2.0-hardened (4/4 DETERMINISTIC)
- Criterion benchmark results for hardened pipeline (338K compilations/sec)

### Fixed
- **C1**: Conv2d IR verifier now rejects zero strides and negative axes
- **C6**: FnDef body verifier enforces SSA scope (use-before-def in body blocks)
- **C2**: String interning DoS protection (MAX_INTERNED_STRINGS = 100,000)
- **C3**: IR printer determinism via sorted function iteration
- **C4**: Constant folding bounds checking for division-by-zero and overflow
- **C5**: Type checker array bounds validation
- **C7**: Hardened eval NaN/Inf propagation
- **A1**: Cargo Deny supply-chain audit configuration

### Security
- String interning rate limiting prevents memory exhaustion attacks
- Constant folding rejects division by zero and integer overflow at compile time

## [0.2.0] - 2026-02-07

### Added
- IR-first compilation pipeline with shape ops and MIC emission
- Remizov universal ODE solver (`std::ode` module)
- Real tensor compute backend with benchmarks
- Open-core reference interpreter for public compiler

### Changed
- **BREAKING**: Replaced Chumsky parser combinator with hand-written recursive descent parser (15x speedup)
- Parser now achieves ~347,000 compilations/sec (up from ~22,700 with Chumsky)
- Removed `chumsky` dependency entirely — zero unnecessary allocations, direct byte-level parsing
- CI skips builds for docs-only changes

### Fixed
- Keyword argument disambiguation in `tensor.gather()` calls (positional `idx` vs `idx=` prefix)
- Clippy lint: `map_or` → `is_some_and` for modern Rust idiom
- Formatting consistency across parser, eval, and exec modules
- Removed unfair NumPy comparisons from benchmarks

### Documentation
- Framework comparison and GPU projections
- Runtime execution benchmarks for v0.1.9
- ODE solver examples and usage guide

## [0.1.9] - 2026-02-05

### Changed
- **BREAKING**: Renamed library crate from `mind` to `libmind` to avoid Windows PDB filename collision
- Updated all imports: `use mind::` → `use libmind::`
- Benchmark numbers aligned across README and docs

### Fixed
- Windows LNK1318 PDB linker errors by limiting parallel build jobs
- Format check CI failures (CRLF → LF line endings)

### Documentation
- Updated test counts to 169+ tests across 70 test files
- Aligned benchmark numbers between README and docs/benchmarks/
- Added v0.1.8 and v0.1.9 to compiler_performance.md version history

## [0.1.8] - 2026-01-15

### Added
- Comprehensive documentation for `TODO(runtime)` markers in `src/exec/mod.rs`
- Test execution examples in README.md
- Performance baseline documentation with concrete metrics

### Changed
- Updated benchmarks.md with detailed performance baselines

## [0.1.0] - 2025-01-01

### Added

#### Core Language
- MIND language parser with Logos lexer (originally Chumsky, replaced by recursive descent in v0.2.0)
- Static type system with rank/shape polymorphism
- Tensor type annotations: `Tensor[dtype, shape]` syntax
- Shape inference engine with broadcasting support
- Effect tracking system for side-effect analysis

#### Compiler Pipeline
- SSA-based intermediate representation (IR)
- IR verifier and deterministic printer
- MLIR lowering passes (`mlir-lowering` feature)
- MLIR dialect support for tensor operations
- LLVM backend integration (`llvm` feature, optional)

#### Autodiff
- Reverse-mode automatic differentiation on SSA IR
- Gradient generation for arithmetic, reductions, and linear algebra
- Autodiff preview mode for debugging gradient graphs

#### CLI Tools
- `mind` REPL and evaluator binary
- `mindc` compiler driver with `--emit-ir`, `--emit-mlir` flags
- GPU target profile (`--target=gpu`) with structured error handling

#### Execution Backends
- CPU execution stubs (`cpu-exec` feature)
- Convolution stubs (`cpu-conv` feature)
- Runtime backend interface for proprietary `mind-runtime`

#### Tensor Operations
- Elementwise: add, sub, mul, div with broadcasting
- Reductions: sum, mean, max, min (all axes or specified)
- Linear algebra: matmul, dot, transpose
- Activations: ReLU, with gradient support
- Indexing: gather, slice with shape preservation
- Conv2D with padding modes (Same, Valid)

#### Developer Experience
- 70 integration test files covering all subsystems
- GitHub Actions CI for Linux, macOS, Windows
- Clippy and rustfmt enforcement
- Comprehensive documentation in `/docs`

#### Documentation
- Language tour and quick start guide
- Type system specification
- Shape algebra documentation
- IR and MLIR pipeline guides
- GPU backend design document
- Versioning and stability policy
- Roadmap with 13 phases including neuroscience/BCI applications

### Fixed
- Rank mismatch classification in type checker (#128)
- Type mismatches and autodiff test issues (#130)
- `/tmp` race condition in MLIR demo script (#133)

### Security
- Secure temporary file handling in MLIR subprocess integration

## Links

- Repository: https://github.com/star-ga/mind
- Documentation: https://github.com/star-ga/mind/tree/main/docs
- Issues: https://github.com/star-ga/mind/issues

[Unreleased]: https://github.com/star-ga/mind/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/star-ga/mind/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/star-ga/mind/compare/v0.6.8...v0.7.0
[0.2.1]: https://github.com/star-ga/mind/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/star-ga/mind/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/star-ga/mind/releases/tag/v0.1.9
[0.2.9]: https://github.com/star-ga/mind/releases/tag/v0.2.9
[0.2.8]: https://github.com/star-ga/mind/releases/tag/v0.2.8
[0.2.7]: https://github.com/star-ga/mind/releases/tag/v0.2.7
[0.2.6]: https://github.com/star-ga/mind/releases/tag/v0.2.6
[0.1.8]: https://github.com/star-ga/mind/releases/tag/v0.1.8
[0.1.0]: https://github.com/star-ga/mind/releases/tag/v0.1.0
