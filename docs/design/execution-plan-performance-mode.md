# Design: PerformanceMode + ExecutionPlan + ExecutionProvider

| Field | Value |
|---|---|
| Status | **Draft / design-only** (no code, no `cargo`) |
| Authors | STARGA Inc. |
| Created | 2026-06-03 |
| Depends | RFC 0021 (canonical IR unification, mic@3), RFC 0016 (evidence chains), RFC 0015/0020 (cross-substrate identity), RFC 0014 (per-substrate lowering) |
| Touches | **public** `mind` crate (PerformanceMode, provider trait shapes); **private** `mind-runtime` (ExecutionPlan builder, concrete providers, CUDA-Graph replay) |
| Convergence | Independent cross-review and STARGA grounding agree on the layering below; this doc fixes the load-bearing decisions so impl cannot drift |

---

## 0. Scope and the one rule everything else serves

This design adds a **performance contract** to MIND **without touching the two
things that make MIND defensible**: the µs-scale deterministic frontend, and the
cross-substrate byte-identity of the compiled artifact. It does so by drawing a
hard line:

- **Compile time** (`src/pipeline.rs::compile_source`) produces the canonical
  `IRModule` and its mic@3 serialization. This is the wedge. It stays semantic,
  deterministic, and substrate-invariant. **Nothing in this design adds an
  instruction-selection or scheduling pass to this path.**
- **Run time** (`mind-runtime`) consumes the *verified* mic@3 artifact and builds
  a derived, cacheable **ExecutionPlan**: fusion groups, reduction pinning,
  stream/capture boundaries, provider assignment, memory placement. This is where
  speed lives, and it is **outside the `trace_hash` preimage** by construction.

> **The load-bearing invariant (do not violate):** the ExecutionPlan is *derived
> from* mic@3, *after* MIC verification, and is *never* an input to
> `ir_trace_hash`. `trace_hash = SHA-256(emit_mic3(ir))` today
> (`src/ir/evidence.rs:72-74`); the ExecutionPlan must not appear in those bytes.
> A scheduling decision can change *how fast* an artifact runs but can **never**
> change *what bytes the compiler signed* — that is the entire reason this can be
> bolted on without re-opening the determinism story.

### What this is NOT

- **Not a second IR.** Per RFC 0021 §2/§3 — *"attach, do not fork."* The
  `IRModule` (mic@1 text / mic@3 binary) remains the sole canonical compiled-artifact
  IR. The ExecutionPlan is a *runtime execution product*, the same way an
  ONNX Runtime session's partitioned/compiled graph is a product of the model, not
  a new model format.
- **Not "Region IR."** `Instr::Region` already exists as a first-class IR
  instruction (opcode `0x24`, std-surface; `src/ir/mod.rs:596`,
  `src/ir/compact/v3/mod.rs:92`). The new runtime structure is **`ExecutionPlan`**,
  never "Region IR" — overloading "Region" would collide with a shipped IR concept
  and re-introduce exactly the two-IR confusion RFC 0021 closed.
- **Not a CUDA-Graphs-as-IR move.** CUDA Graphs / capture-replay are a *runtime
  replay layer* the provider may use to amortize launch cost (`§4.4`); they are an
  implementation detail of `run_partition`, not a serialized artifact and not part
  of any hash.

---

## 1. Layered pipeline

```text
                        ┌─────────────────────────── COMPILE TIME (public mind crate) ───────────────────────────┐
                        │                                                                                          │
  source ──▶ parse ──▶ AST ──▶ type_check ──▶ lower_to_ir ──▶ verify_module ──▶ canonicalize ──▶ verify_module    │
  (.mind)    parser.rs        eval::lower_to_ir       IRModule        ir::verify_module   opt::ir_canonical        │
                        │                                  │                                                       │
                        │                                  ▼                                                       │
                        │                          ┌──────────────┐   PerformanceMode threads through here:       │
                        │                          │  IRModule     │   CompileOptions.perf_mode  (pipeline.rs:67) │
                        │                          │ (canonical)   │   reaches the cache-key fingerprint           │
                        │                          └──────┬───────┘   (CacheKey, cache/entry.rs:12)               │
                        │                                 │                                                        │
                        │             emit_mic3(ir) ◀─────┘   src/ir/compact/v3/emit.rs                            │
                        │                  │                                                                       │
                        │                  ▼                                                                       │
                        │         ┌────────────────────────────────┐                                              │
                        │         │ mic@3 binary IRModule body      │  ← trace_hash = SHA-256(this) ───┐           │
                        │         │  + optional MAP evidence epilogue│     evidence.rs:72-74            │           │
                        │         │    evidence_chain.{trace_hash,   │     (mode recorded as a MAP key  │           │
                        │         │     substrate, toolchain,        │      OUTSIDE the body preimage)  │           │
                        │         │     determinism, schema, mode}   │                                 │           │
                        │         └────────────────┬────────────────┘                                 │           │
                        └──────────────────────────┼──────────────────────────────────────────────────┼──────────┘
                                                   │  (artifact on disk / wire)                        │
   ════════════════════════════════════════════════╪═══════════════════════════════════════════════════╪══════════
                                                   ▼                                                   │
                        ┌────────────────────────── RUN TIME (private mind-runtime) ───────────────────┼──────────┐
                        │                                                                              │           │
   load(bytes) ──▶ parse_mic3 ──▶ VERIFY ───────────────────────────────────────────────────────────┘           │
                        │   (1) magic/version/size/depth guards (parse.rs)                                        │
                        │   (2) recompute SHA-256(body) == evidence_chain.trace_hash  (mic3_evidence_report)      │
                        │   (3) artifact PerformanceMode must be loadable by this runtime's mode policy ◀── GATE   │
                        │                                  │                                                       │
                        │                                  ▼   build AFTER verify, OUTSIDE trace_hash preimage     │
                        │                          ┌──────────────────────────────────────────────┐               │
                        │                          │           ExecutionPlan(mode, caps)           │               │
                        │                          │  fused groups · pinned reductions · overlap   │               │
                        │                          │  cache reuse · stream/capture boundaries ·    │               │
                        │                          │  provider assignment · memory placement ·     │               │
                        │                          │  attestation hooks                            │               │
                        │                          └──────────────┬───────────────────────────────┘               │
                        │                                         │                                                │
                        │   for each ExecutionProvider:           ▼                                                │
                        │     caps = get_capability(ir, mode) ─▶ partitions                                        │
                        │     compile_partition(part) ─▶ CompiledPartition  (cached, keyed by mode+caps+hash)      │
                        │                                         │                                                │
                        │                                         ▼                                                │
                        │     run_partition(compiled, inputs) ─▶ outputs  (+ optional CUDA-Graph replay)           │
                        │                                         │                                                │
                        │                                         ▼                                                │
                        │             per-request evidence: {trace_hash, mode, substrate, plan_digest, result_hash}│
                        └──────────────────────────────────────────────────────────────────────────────────────┘
```

Three independent boundaries are visible in the diagram and are the spine of §3:

1. The **compile/run line** — `emit_mic3` is the last compile-time step; everything
   below is runtime and pays no µs-path cost.
2. The **trace_hash preimage line** — only the mic@3 *body* is hashed; mode lives in
   the MAP epilogue (already excluded from the body, `v3/mod.rs:38-51`), and the
   ExecutionPlan never enters the hash at all.
3. The **op→provider line** — the runtime no longer dispatches one opcode string at
   a time; it asks each provider which sub-DAG it can claim, then compiles and runs
   *partitions*.

---

## 2. Signatures (Rust)

### 2.1 PerformanceMode — public `mind` crate, threaded through `CompileOptions`

```rust
// src/perf.rs (new, public mind crate)

/// The performance/semantics contract an artifact is built and run under.
///
/// This is a *load-bearing* tag: it is part of the cache key, is stamped into the
/// mic@3 MAP epilogue (`evidence_chain.mode`), and gates whether a runtime is
/// allowed to load an artifact (§3.2). Lower-determinism modes MUST NOT be able to
/// masquerade as higher ones, and a higher-determinism artifact MUST NOT silently
/// run on a runtime that can only honor a lower one.
///
/// Ordering is *capability lattice*, not "speed": `BitExactQ16 ⊑ Audited ⊑
/// DeterministicFloat ⊑ FastFloat` in terms of how much reordering freedom the
/// runtime is granted. It is deliberately NOT `Ord` — comparisons go through an
/// explicit `permits()` policy so the lattice direction can never be read off an
/// accidental `<`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PerformanceMode {
    /// Maximum throughput. The runtime may reassociate float reductions, fuse
    /// freely, and pick the fastest provider kernel. Results are NOT guaranteed
    /// bit-reproducible across substrates and carry NO "audited"/"bit-exact"
    /// wording in any evidence envelope.
    FastFloat,
    /// Float math with a fixed, documented reduction/fusion order so a single
    /// substrate is run-to-run reproducible. Still NOT cross-substrate bit-exact
    /// (float rounding differs per ISA); see RFC 0021 §3.5.
    DeterministicFloat,
    /// `DeterministicFloat` plus a complete per-request evidence envelope
    /// (trace_hash + plan_digest + result_hash). The audit trail is the product;
    /// performance is secondary.
    Audited,
    /// The wedge mode: Q16.16 fixed-point end-to-end. Pinned reductions, no
    /// reassociation, cross-substrate byte-identical results (RFC 0015/0020).
    /// This is the only mode whose result hash is asserted equal across x86 /
    /// ARM / GPU / wafer.
    #[default]
    BitExactQ16,
}

impl PerformanceMode {
    /// Stable token used in the cache-key fingerprint and the MAP `mode` value.
    /// MUST be stable forever once shipped (it is a wire/cache contract).
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMode::FastFloat          => "fast-float",
            PerformanceMode::DeterministicFloat => "det-float",
            PerformanceMode::Audited            => "audited",
            PerformanceMode::BitExactQ16        => "bitexact-q16",
        }
    }

    /// Policy gate (§3.2): may a runtime configured for `self` load an artifact
    /// built for `artifact`? A runtime MUST refuse to *upgrade* an artifact's
    /// guarantees and MUST refuse to run a stronger artifact it cannot honor.
    ///
    /// Rule: exact-match by default. A runtime that explicitly opts into
    /// "downgrade execution" (e.g. run a BitExactQ16 artifact under FastFloat for
    /// a throughput preview) MUST relabel the result envelope to the *weaker*
    /// mode and strip audited/bit-exact wording — never the converse.
    pub fn permits_load(self, artifact: PerformanceMode) -> bool {
        self == artifact
    }
}
```

`CompileOptions` gains one field (everything else in `src/pipeline.rs:67-84`
unchanged):

```rust
#[derive(Debug, Default, Clone)]
pub struct CompileOptions {
    pub func: Option<String>,
    pub enable_autodiff: bool,
    pub target: BackendTarget,
    pub manifest_exports: Vec<String>,
    pub profile: crate::cache::ProfileTag,
    /// NEW. The performance/semantics contract this artifact is built under.
    /// Reaches the cache-key fingerprint (so the same source under two modes
    /// yields two distinct artifacts) and is stamped into the mic@3 MAP epilogue
    /// when `--emit-evidence` is used. Default = `BitExactQ16` (wedge-safe).
    pub perf_mode: crate::perf::PerformanceMode,
}
```

`CacheKey` gains the mode so a `FastFloat` build can never be served from a
`BitExactQ16` cache entry (mirrors how `profile` already shards the key,
`src/cache/entry.rs:12-41`):

```rust
pub struct CacheKey {
    pub compiler_version: String,
    pub profile: ProfileTag,
    pub perf_mode: PerformanceMode,   // NEW — render(): "...:{perf_mode.as_str()}:..."
    pub source_hash: String,
    pub imports_hash: String,
}
```

> **Note on the trace_hash:** `perf_mode` does **not** enter `emit_mic3(ir)` and
> therefore does **not** change `trace_hash`. It is recorded as a MAP key
> (`evidence_chain.mode`), which already lives in the epilogue *after* the hashed
> body (`v3/mod.rs:38-51`). Two artifacts that differ only in mode share a body and
> a `trace_hash` but carry different MAP `mode` values and different cache keys —
> which is correct: the *computation* is identical; the *contract under which it is
> permitted to run* differs.

### 2.2 ExecutionProvider — provider contract (replaces op-level `run_op`)

Replaces the per-opcode `GPUBackend::run_op(&Instr, …)`
(`src/runtime/gpu.rs:20-25`) and `MindRuntime::run_op(&str, …)`
(`src/runtime_interface.rs:33-38`). The interface declares its trait surface in the
**public** `mind` crate (so `mind`, `mind-runtime`, `mind-nerve`, `mind-mem` share
one vocabulary); concrete providers live in **private** `mind-runtime`.

Prior art: ONNX Runtime execution providers' `GetCapability` partitioning — the
provider is *asked which subgraph it can claim*, rather than the runtime pushing one
node at a time.

```rust
// src/runtime/provider.rs (new, public mind crate — trait + plain data only)

use crate::ir::IRModule;
use crate::perf::PerformanceMode;
use crate::runtime::types::{DeviceKind, RuntimeError, TensorHandle};
use crate::runtime_interface::TensorDesc;

/// Static description of what a provider can do. Used by the ExecutionPlan
/// builder to assign partitions and by the cache to key compiled artifacts.
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub device: DeviceKind,
    /// Modes this provider can honor. A provider that cannot produce
    /// cross-substrate Q16.16 results MUST NOT advertise `BitExactQ16`.
    pub supported_modes: Vec<PerformanceMode>,
    /// Opcode discriminants this provider can execute (by mic@3 opcode byte).
    pub supported_ops: Vec<u8>,
    /// Whether the provider supports capture/replay (e.g. CUDA Graphs) — a hint
    /// the plan builder uses to set capture boundaries, never a semantic change.
    pub supports_capture_replay: bool,
    /// Stable id for cache keys + evidence (e.g. "cuda-sm90", "cpu-avx2").
    pub provider_id: &'static str,
}

/// A contiguous sub-DAG of the verified IR that a provider has claimed.
/// Carries indices into the module's instruction stream; it is NOT a new IR —
/// it borrows/references the canonical `IRModule`.
#[derive(Debug, Clone)]
pub struct Partition {
    pub provider_id: &'static str,
    /// Indices (pre-order, into the flattened IR incl. Region/If/While bodies)
    /// of the instructions this partition owns. Disjoint across providers;
    /// the union over a plan covers every executable instruction exactly once.
    pub instr_path: Vec<InstrPath>,
}

/// A path to an instruction inside nested control-flow regions, e.g.
/// `[Top(4), Region(2), If(0, Then(1))]`. Keeps partitions precise without
/// flattening control flow away.
pub type InstrPath = smallvec::SmallVec<[u32; 4]>;

/// A provider-specific compiled object for one partition (opaque to mind).
pub struct CompiledPartition {
    pub provider_id: &'static str,
    pub mode: PerformanceMode,
    /// Hash of (partition bytes + mode + provider_id) — the compile cache key.
    pub compile_key: [u8; 32],
    /// Provider-private payload (kernels, CUDA graph handle, …). `Box<dyn Any>`
    /// in the trait object; concrete type lives in mind-runtime.
    pub artifact: Box<dyn std::any::Any + Send + Sync>,
}

/// The provider contract. `&str`-op dispatch is gone; providers reason over
/// IR sub-DAGs and modes.
pub trait ExecutionProvider: Send + Sync {
    /// Static capabilities (cheap, called during plan construction).
    fn capabilities(&self) -> &ProviderCapabilities;

    /// ONNX-Runtime-style claim: given the verified module and the requested
    /// mode, return the maximal set of partitions this provider can execute
    /// under that mode. MUST return empty if `mode` is not in
    /// `capabilities().supported_modes`. Pure/read-only over `ir`.
    fn get_capability(&self, ir: &IRModule, mode: PerformanceMode) -> Vec<Partition>;

    /// Compile one claimed partition into a runnable object. Deterministic for a
    /// given (partition, mode) so `compile_key` is a sound cache key.
    fn compile_partition(
        &self,
        ir: &IRModule,
        partition: &Partition,
        mode: PerformanceMode,
    ) -> Result<CompiledPartition, RuntimeError>;

    /// Execute a compiled partition on device-resident handles. May use
    /// capture/replay internally; that is invisible to the caller and to the
    /// hash. In `BitExactQ16` mode the result MUST be byte-identical to every
    /// other compliant provider (RFC 0015/0020).
    fn run_partition(
        &self,
        compiled: &CompiledPartition,
        inputs: &[TensorHandle],
        outputs: &[TensorHandle],
    ) -> Result<(), RuntimeError>;

    fn allocate(&self, desc: &TensorDesc) -> Result<TensorHandle, RuntimeError>;
    fn synchronize(&self) -> Result<(), RuntimeError>;
}
```

**Legacy shim (kept, not deleted).** `run_op` becomes a thin wrapper that builds a
single-op plan fragment, so existing callers and the `NoOpRuntime`
(`runtime_interface.rs:58-68`) keep compiling:

```rust
/// Back-compat: execute exactly one IR instruction by wrapping it in a one-node
/// partition. New code should build an ExecutionPlan over the whole module.
fn run_op_legacy(
    provider: &dyn ExecutionProvider,
    ir: &IRModule,
    instr_path: InstrPath,
    mode: PerformanceMode,
    inputs: &[TensorHandle],
    outputs: &[TensorHandle],
) -> Result<(), RuntimeError> {
    let part = Partition { provider_id: provider.capabilities().provider_id,
                           instr_path: vec![instr_path] };
    let compiled = provider.compile_partition(ir, &part, mode)?;
    provider.run_partition(&compiled, inputs, outputs)
}
```

### 2.3 ExecutionPlan — runtime-side, derived, off-hash

```rust
// mind-runtime crate (PRIVATE) — the builder + concrete fields live here.
// `mind` may host a minimal `ExecutionPlanShape` for testing, but the optimizer
// is private.

/// A cacheable execution product derived from a *verified* mic@3 IRModule.
///
/// INVARIANTS (enforced by construction, asserted in tests):
///  1. Built only AFTER `mic3_evidence_report` verifies the artifact.
///  2. NEVER fed into `ir_trace_hash` / `emit_mic3`. `plan_digest` below is a
///     SEPARATE digest for caching/telemetry; it is NOT the artifact trace_hash
///     and never appears in the mic@3 body.
///  3. Pure function of (IRModule, mode, available providers). Same inputs ⇒
///     same plan ⇒ same `plan_digest` (so plans are cacheable).
pub struct ExecutionPlan {
    /// The artifact this plan executes. `trace_hash` is the IR-body hash; the
    /// plan does not alter it.
    pub trace_hash: [u8; 32],
    pub mode: PerformanceMode,

    /// Fused instruction groups (elementwise chains, conv+relu, …). Fusion is a
    /// runtime scheduling choice; in BitExactQ16 it may not reassociate.
    pub fused_groups: Vec<FusedGroup>,
    /// Reductions pinned to a fixed accumulation order/tree. REQUIRED non-empty
    /// coverage of every reduction when `mode == BitExactQ16`.
    pub pinned_reductions: Vec<PinnedReduction>,
    /// Buffers reusable across overlapping live ranges (memory-overlap cache).
    pub overlap_reuse: Vec<BufferReuse>,
    /// Stream / capture-replay boundaries (e.g. CUDA Graph capture spans).
    pub capture_spans: Vec<CaptureSpan>,
    /// Which provider owns each partition (from get_capability).
    pub assignments: Vec<(Partition, ProviderId)>,
    /// Device placement per buffer.
    pub placement: Vec<(BufferId, DeviceKind)>,
    /// Attestation hooks: where per-request evidence is captured (result hashes,
    /// substrate id) so the runtime can emit a per-request envelope (§3 / §1).
    pub attestation_hooks: Vec<AttestationHook>,

    /// Digest of the plan itself (for plan-cache lookup + telemetry). DISTINCT
    /// from `trace_hash`. Documented as "plan_digest" everywhere to prevent
    /// anyone confusing it with the artifact hash.
    pub plan_digest: [u8; 32],
}
```

---

## 3. Orthogonality — why the three pillars stay independent

The whole point is that **fast frontend**, **determinism**, and **runtime speed** are
three axes that this design keeps from interfering. Each pillar has a concrete
mechanism, not a hand-wave.

### 3.1 Fast frontend stays fast (the µs compile path is untouched)
The ExecutionPlan, partitioning, fusion analysis, and provider compilation are
**100% runtime-side**. `compile_source` (`pipeline.rs:223-321`) gains exactly one
plain-data field read (`perf_mode` → cache key → MAP key) and **zero new passes**.
There is no instruction selection, no scheduling, no cost model on the compile path.
The frozen criterion baselines and the `compile-timings` waterfall
(`pipeline.rs:176-218`) are unaffected because nothing in the hot loop changed shape.
A `FastFloat` build and a `BitExactQ16` build run the *same* frontend at the *same*
speed and emit the *same kind* of mic@3 artifact — they differ only in a tag.

### 3.2 Determinism is structurally protected (ExecutionPlan is off the hash)
`trace_hash = SHA-256(emit_mic3(ir))` (`evidence.rs:72-74`), and `emit_mic3` encodes
**only the IRModule body** (`v3/mod.rs:18-95`). The ExecutionPlan is derived *after*
that hash is computed and verified, and is **never** an argument to `emit_mic3` or
`ir_trace_hash`. Therefore:
- No fusion choice, reduction-tree pin, stream boundary, provider assignment, or
  CUDA-Graph capture can change `trace_hash`. Two runtimes that schedule the same
  artifact completely differently still verify against the *same* signed hash.
- `PerformanceMode` rides in the MAP epilogue (`evidence_chain.mode`), which already
  sits *after* the hashed body — so even the mode tag cannot perturb the IR-body
  hash. The cross-substrate gate (RFC 0015/0020, RFC 0021 §3.5) compares
  `trace_hash` (IR identity) and, for `BitExactQ16`, the lowered *result* hash;
  neither is a function of the plan.
- The mode gate (`permits_load`, §2.1) closes the labeling hole the task calls out:
  a `BitExactQ16` artifact **cannot** load on a `FastFloat`-only runtime (the load
  is refused), and a `FastFloat` result **cannot** inherit `Audited`/`BitExactQ16`
  wording (a downgrade execution must relabel the envelope to the weaker mode). The
  artifact's mode is committed in the MAP; the runtime's mode is policy; they must
  match or the run is refused/relabeled — never silently upgraded.

### 3.3 Runtime is where speed lives (provider layer, fully private-side)
All throughput work — fusion, pinned reductions, buffer-overlap reuse, capture/replay,
provider kernel selection — lives in the ExecutionPlan + `ExecutionProvider`
implementations in `mind-runtime`. The public crate only declares the *contract*. This
means STARGA can ship aggressive GPU/wafer optimizations privately without changing a
single byte of the public artifact format or the determinism gate. CUDA Graphs are a
replay optimization inside `run_partition`; they never become an IR or a hash input.

> The three axes meet at exactly one object — the verified mic@3 artifact — and touch
> nothing of each other's machinery: the frontend produces it, the determinism gate
> hashes it, the runtime schedules around it.

---

## 4. Prioritized implementation order (public-repo-first)

Each step is independently testable and additive; the keystone bootstrap and mic@1/mic@3
byte-output are never regressed.

| # | Step | Crate | Gate / operator buy-in |
|---|---|---|---|
| **1** | **`PerformanceMode` enum + `CompileOptions.perf_mode`** (`src/perf.rs`, `pipeline.rs`). Default `BitExactQ16`. Pure additive field. | **public `mind`** | None — lands now. **First concrete step.** |
| **2** | Thread `perf_mode` into `CacheKey` + `render()` (`cache/entry.rs`, `cache/fingerprint.rs` style). Test: two modes ⇒ two cache entries, no cross-serve. | **public `mind`** | None |
| **3** | Stamp `evidence_chain.mode` into the mic@3 MAP epilogue on `--emit-evidence` (`v3/evidence.rs`); assert IR-body bytes + `trace_hash` byte-identical with/without the mode key. | **public `mind`** | None |
| **4** | `mindc --perf-mode <…>` CLI flag (clap) + `Mind.toml [build] perf_mode`. Strict validation at the clap layer; unknown ⇒ error. | **public `mind`** | None |
| **5** | `permits_load` gate wired into the loader: a runtime refuses to load an artifact whose MAP `mode` it cannot honor; downgrade-execution path relabels the envelope. | **public `mind`** loader hooks; policy honored by **private** runtime | Light — interface only |
| **6** | **`ExecutionProvider` trait + `Partition`/`ProviderCapabilities`/`CompiledPartition`** in `src/runtime/provider.rs`; keep `run_op` as `run_op_legacy` shim. | **public `mind`** (trait + data) | None for the trait shapes |
| **7** | **`ExecutionPlan` builder + concrete providers (CPU/CUDA/wafer)**, fusion, pinned reductions, overlap reuse, capture spans, per-request evidence. | **PRIVATE `mind-runtime`** | **Operator buy-in required** |
| **8** | Adopt the shared `ExecutionProvider` vocabulary in `mind-nerve` / `mind-mem` (they consume mic@3 today; align capability descriptors). | cross-repo | Coordinated, after #7 |

Steps **1–6 live entirely in the public `mind` crate** and can land without touching
the private runtime. Step **1 is the first concrete commit** and is a trivially-additive field.

---

## 5. Public `mind` vs private `mind-runtime` — explicit split

| Concern | Public `mind` | Private `mind-runtime` |
|---|---|---|
| `PerformanceMode` enum + `as_str`/`permits_load` | ✅ source of truth | consumes |
| `CompileOptions.perf_mode`, `CacheKey.perf_mode`, MAP `mode` key | ✅ | consumes |
| `ExecutionProvider` **trait**, `Partition`, `ProviderCapabilities`, `CompiledPartition` (shapes) | ✅ contract only | implements |
| `run_op` legacy shim | ✅ | — |
| **`ExecutionPlan` builder** (fusion, pinned reductions, overlap reuse, capture spans, placement, assignment) | ✗ | ✅ **private** |
| **Concrete providers** (CPU-AVX2, CUDA-sm90, Cerebras/wafer), kernel libs, CUDA-Graph capture/replay | ✗ (only the `NoOpRuntime`-style stub) | ✅ **private** |
| Per-request attestation emission (result hashes, substrate id) | hook *interface* | ✅ implementation |
| Cross-substrate Q16.16 result-hash gate harness | gate spec / `oracle.rs` (RFC 0021 §3.5, #307) | the per-substrate runners |

**Rule:** anything that is a *contract* (enum, trait, data shape, hash anchor, cache
key, MAP key) is **public** and committed in `mind`; anything that is *how fast it
runs* (the plan optimizer, kernels, capture/replay, real providers) is **private** in
`mind-runtime` and needs operator buy-in (step #7). The split is exactly the existing
RFC 0021 / `runtime_interface.rs` boundary — *"the open-core repo only defines this
interface; production backends live in the private `mind-runtime` repository"*
(`runtime_interface.rs:20-22`) — extended from `MindRuntime` to `ExecutionProvider`.

---

## 6. Open questions (flagged, not resolved here)

1. **`InstrPath` representation under nested regions.** Partitioning must address
   instructions inside `Region`/`If`/`While` bodies (`v3/mod.rs:80-95`). The
   `SmallVec` path is a proposal; the exact encoding must match whatever pre-order
   the cross-substrate gate uses so partition coverage is verifiable.
2. **Plan-cache key vs determinism.** `plan_digest` must be a pure function of
   `(IRModule, mode, provider set)` and explicitly *not* of wall-clock/device-probe
   nondeterminism, or plan caching itself becomes a nondeterminism source. Needs a
   "no environment in the plan key" lint.
3. **Downgrade-execution UX.** When a `BitExactQ16` artifact is run under `FastFloat`
   for a throughput preview, the relabeled envelope must be unmistakable to a
   verifier. Wording/format is a security-review item (mind-security-reviewer).
4. **`MindRuntime` vs `ExecutionProvider` coexistence.** Decide whether
   `MindRuntime::run_op(&str)` is deprecated in place via the shim or eventually
   removed; either way the shim is the migration bridge.
