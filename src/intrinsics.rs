// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! The intrinsic registry: the ONE table naming every `__mind_*` primitive the
//! compiler will emit, its i64 arity, and whether calling it can make the
//! program's result depend on something other than its declared inputs.
//!
//! ## Why the determinism verdict lives HERE, beside the arity
//!
//! It used to live in two hand-maintained parallel arrays — one in
//! `crate::ir::evidence` (the IR-side classifier that decides what the
//! RFC 0016 evidence chain ATTESTS) and one in [`crate::type_checker`] (the
//! AST-side classifier behind the `#[deterministic]` call-graph check) — each
//! carrying a comment asking the next author to keep it in sync with the other.
//! They drifted identically: the whole `__mind_nerve_rt_*` host surface (clock /
//! stdin / file / entropy / getenv), registered for emission in this very table,
//! was added to the registry and to NEITHER classifier. A three-line program
//! whose `main` returned `__mind_nerve_rt_monotonic_ns()` therefore compiled
//! with the determinism-by-default gate silent, emitted an evidence artifact
//! attesting `determinism: deterministic`, and PASSED
//! `mindc verify --require-deterministic` — the one consumer gate documented as
//! fail-closed. An attestation that can be made to lie is worse than none.
//!
//! Re-deriving the label from the hashed mic@3 body (the Risk-2 fix in
//! `mindc verify`) does not help when the classifier the derivation runs is
//! itself incomplete. So the classification is now data on the registry row,
//! with two structural consequences, both deliberate:
//!
//! 1. Adding a registry entry is a COMPILE ERROR until its [`Det`] is supplied.
//!    There is no default and no blanket `__mind_*` rule ahead of the table, so
//!    a new intrinsic cannot be silently admitted as pure.
//! 2. The `classification_covers_every_runtime_support_intrinsic` test reads
//!    `runtime-support/mind_intrinsics.c` and FAILS until every `__mind_*`
//!    symbol the C runtime exports is explicitly classified here — including
//!    those NOT in the registry, so REGISTERING one later cannot re-open the
//!    hole this module exists to close.

/// Whether a callee's result is a function of the program's declared inputs.
///
/// The evidence chain's `determinism` field is derived from this: a module that
/// reaches one [`Det::World`] callee is attested `nondeterministic`, and the
/// determinism-by-default build gate refuses to produce a runnable or attested
/// artifact for it unless `--allow-nondeterministic` authorises the build (the
/// flag authorises the build, never the label).
///
/// The line is drawn at *can two runs of this program, on the same declared
/// inputs, disagree?* — NOT at *does this touch the OS*. Over-tainting is as
/// dishonest as under-tainting: taint `__mind_alloc` or `__mind_read` and every
/// std-surface program — mindc's own self-hosted build included — is attested
/// `nondeterministic`, which tells a consumer nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Det {
    /// Pure with respect to declared inputs: same program, same inputs, same
    /// result. Effects (writes, frees, aborts) are classified here too — they
    /// are observable, but they do not make the RESULT vary.
    Pure,
    /// Reads the world through a channel the artifact does not name: the wall
    /// or monotonic clock, OS entropy, the environment block, the stdin stream,
    /// host CPU-feature state.
    World,
}

// RFC 0005 Phase 1 + 1.5 — pure-MIND standard surface intrinsics.
//
// The primitives the std surface (`Vec`, `String`, `Map`, `io`)
// is allowed to bottom out into. All take and return `i64` only (no
// `Ptr` type — see RFC 0005 P0a; an address is a 64-bit integer).
// The pair (`__mind_load_i64`, `__mind_store_i64`) was added at Phase
// 1.5 to resolve P0c — without scalar load/store at address, `vec.push`
// cannot write the new value into the `__mind_alloc`-returned backing
// store. Lowered by the gated Phase-0 `Instr::Call` arm in
// `src/mlir/lowering.rs` to `func.call @__mind_*(%a..) : (i64..) -> i64`,
// with a matching `func.func private` declaration emitted once per
// distinct callee in sorted order. Default builds compile out the
// recogniser entirely.
pub(crate) const STD_SURFACE_INTRINSICS: &[(&str, usize, Det)] = &[
    ("__mind_alloc", 1, Det::Pure),
    ("__mind_blas_dot_f32", 3, Det::Pure),
    // RFC 0006 Track B (increment 1): native MLIR vector-dialect
    // `dot_f32`. Same i64 ABI / arity (3) as the Track A scalar bridge;
    // the difference is purely in lowering — the `Instr::Call` for this
    // name emits a `vector`-dialect reduction loop, not a `func.call` to
    // the runtime-support C shim. Track A's `__mind_blas_dot_f32` stays
    // registered and is the unchanged scalar/AVX2 fallback.
    ("__mind_blas_dot_f32_v", 3, Det::Pure),
    // "int-dot" tier: native MLIR vector-dialect int16 dot product. Same i64
    // ABI / arity (3) as the other vector dots. Inputs are i16 row-major;
    // byte-identical to the scalar oracle `(i32) sum_k ((i32)a[k]*(i32)b[k])`
    // for ALL int16 inputs (i64-lane accumulate, no shift, no saturation, no
    // early narrow). The widen-multiply-accumulate loop is the AVX2 vpmaddwd
    // idiom at -march=x86-64-v3 — the fast deterministic int GEMM tier.
    ("__mind_blas_dot_i16_v", 3, Det::Pure),
    ("__mind_blas_dot_l1_f32", 3, Det::Pure),
    // RFC 0006 Track B (increment 2): native MLIR vector-dialect f32 L1
    // (sum-of-abs) reduction. Same i64 ABI / arity (3) as the Track A
    // scalar bridge; lowering interception emits an abs-diff + add
    // reduction loop. Track A's `__mind_blas_dot_l1_f32` is unchanged.
    ("__mind_blas_dot_l1_f32_v", 3, Det::Pure),
    ("__mind_blas_dot_l1_q16", 3, Det::Pure),
    // RFC 0006 Track B (increment 3): native MLIR vector-dialect Q16.16 L1
    // (Manhattan, sum-of-abs) reduction. Byte-identical to the Track A
    // scalar oracle `__mind_blas_dot_l1_q16` at every length (task #57
    // cross-arch bit-identity gate); closes the Q16.16 vector-path metric
    // parity deferred in increment 2. Track A's `__mind_blas_dot_l1_q16`
    // is unchanged.
    ("__mind_blas_dot_l1_q16_v", 3, Det::Pure),
    ("__mind_blas_dot_linf_f32", 3, Det::Pure),
    // RFC 0006 Track B (increment 2): native MLIR vector-dialect f32 L∞
    // (max-of-abs) reduction. Track A's `__mind_blas_dot_linf_f32` is
    // unchanged.
    ("__mind_blas_dot_linf_f32_v", 3, Det::Pure),
    ("__mind_blas_dot_q16", 3, Det::Pure),
    // RFC 0006 Track B (increment 2): native MLIR vector-dialect Q16.16
    // dot product. Byte-identical to the Track A scalar oracle
    // `__mind_blas_dot_q16` at every length (task #57 cross-arch
    // bit-identity gate). Track A's `__mind_blas_dot_q16` is unchanged.
    ("__mind_blas_dot_q16_v", 3, Det::Pure),
    ("__mind_blas_matmul_rmajor_f32", 5, Det::Pure),
    // RFC 0006 Track B (increment 3b): native MLIR vector-dialect row-major
    // f32 matmul.  Outer scf.for over rows, inner vectorised dot_f32_v
    // (8-lane FMA + scalar tail) inlined per row, stores to caller-allocated
    // y buffer, returns 0.  Same arity (5) and i64 ABI as Track A.
    ("__mind_blas_matmul_rmajor_f32_v", 5, Det::Pure),
    // "int-dot" tier: native MLIR vector-dialect row-major int16 matmul.
    // Outer scf.for over rows, inner int16 dot reduction from emit_vec_dot_i16
    // (sext i16->i64, i64-lane accumulate, vector.reduction <add>, scalar
    // tail, trunc) inlined per row, stores i32 to the caller-allocated y
    // buffer, returns 0. Same arity (5) and i64 ABI as the f32/q16 matmuls.
    // Byte-identical to the scalar oracle applied per row, for all int16
    // inputs. Track B vector-dialect only — no Track A i16 matmul extern.
    ("__mind_blas_matmul_rmajor_i16_v", 5, Det::Pure),
    // RFC 0006 Track B (increment 4): native MLIR vector-dialect row-major
    // Q16.16 matmul.  Outer scf.for over rows, inner Q16.16 dot reduction
    // from emit_vec_dot_q16 (widen i32→i64, >> 16, i64-lane accumulate,
    // vector.reduction <add>, scalar tail, trunc+extsi) inlined per row,
    // stores i32 to caller-allocated y buffer, returns 0.  Byte-identical
    // to the scalar oracle __mind_blas_dot_q16 applied per row (cross-arch
    // bit-identity gate, task #57).  Track B vector-dialect only — there is
    // no Track A q16 matmul extern; the per-row oracle is __mind_blas_dot_q16.
    ("__mind_blas_matmul_rmajor_q16_v", 5, Det::Pure),
    // "det.igemm" tier: fused int8 GEMM. A is M×K row-major int8 (1 byte), B
    // is K×N row-major int8, C is M×N row-major INT32 caller-allocated; arity 6
    // (a, b, c, m, k, n), i64 ABI, returns 0. Same BLIS-blocked register-tiled
    // kernel as the Q16 path with i8→i32 sign-extension during the pack and NO
    // >> 16 shift (int8 is integer, not fixed-point). The C-tile accumulates
    // i64; the i64→i32 truncation happens once at the store — byte-identical to
    // the per-element scalar int32 oracle (i32) Σ_k (i32)A[i,k]*(i32)B[k,j] for
    // all shapes. The same MLIR lowers to vpmaddwd (AVX2) / SDOT (aarch64),
    // both yielding the identical exact int32 sum.
    ("__mind_blas_matmul_mm_i8_v", 6, Det::Pure),
    // Multithreaded fused int8 GEMM. Same ABI (arity 6: a, b, c, m, k, n; i64;
    // returns 0) and byte-for-byte output as __mind_blas_matmul_mm_i8_v,
    // parallelised over contiguous owner-computes M-row bands with raw POSIX
    // threads. Output is independent of the thread count (no cross-thread
    // reduction), so cross-substrate bit-identity holds.
    ("__mind_blas_matmul_mm_i8_mt_v", 6, Det::Pure),
    // RFC 0006 Track B: fused outer-product Q16.16 GEMM. A is M×K row-major,
    // B is K×N row-major (un-transposed), C is M×N row-major caller-allocated;
    // arity 6 (a, b, c, m, k, n), i64 ABI, returns 0. Register-tiled
    // outer-product microkernel (no horizontal reduction) — byte-identical to
    // the per-element scalar oracle Σ_k (A[i,k]*B[k,j])>>16 for all shapes.
    ("__mind_blas_matmul_mm_q16_v", 6, Det::Pure),
    // Multithreaded fused outer-product Q16.16 GEMM. Same ABI (arity 6:
    // a, b, c, m, k, n; i64; returns 0) and byte-for-byte output as
    // __mind_blas_matmul_mm_q16_v, parallelised over contiguous owner-computes
    // M-row bands with raw POSIX threads. Output is independent of the thread
    // count (no cross-thread reduction), so cross-substrate bit-identity holds.
    ("__mind_blas_matmul_mm_q16_mt_v", 6, Det::Pure),
    // Multithreaded fused Q16.16 GEMV — the routing SCORE kernel
    // (`scores[M] = catalog[M×K] · query[K]`, catalog + query packed-i32, scores
    // i32; arity 5: catalog, query, scores, m, k; i64 ABI, returns 0). Same
    // owner-computes MT band wrapper as `__mind_blas_matmul_mm_q16_mt_v` with N
    // pinned to 1, but each band vectorises the K reduction (exact i64 accumulate
    // + horizontal reduce) instead of the N-columns tile — streaming the catalog
    // once (memory-bound). Byte-for-byte identical to the per-row scalar dot oracle
    // `Σ_k (catalog[i,k]*query[k])>>16` and independent of the thread count, so
    // cross-substrate bit-identity holds.
    ("__mind_blas_gemv_q16_mt", 5, Det::Pure),
    // mind-nerve C-ABI runtime surface (RFC 0006 pattern, i64 ABI). These nine
    // symbols are the native encoder's Q16.16 BLAS + LUT-handle bridge, defined
    // in mind-nerve's `runtime/blas_shims_i64.c` + `runtime/lut_cache.c` and
    // resolved at .so link time (the C objects are compiled+linked via the
    // manifest `[build].native_sources` list). Registering them here — same as
    // any `__mind_blas_*` extern — makes mindc's MLIR backend emit a plain
    // arity-checked `func.call @__mind_nerve_*` (NOT the E2024 self-host-only
    // advisory + runtime-JIT fallback), so the kernel/LUT modules that call them
    // compile NATIVELY. All args + result are i64 (opaque heap addresses / Q16.16
    // scalars in the low 32 bits); the C side is byte-identical scalar-vs-AVX2
    // (task #57 cross-arch gate). The four `_lut_*_h` accessors are 0-arity cached
    // table-handle getters; the five `_blas_*` are Q16.16 dot / score / GEMM /
    // attention contractions with the un-transposed row-major operand layout.
    ("__mind_nerve_blas_attnv_q16_i64", 6, Det::Pure),
    ("__mind_nerve_blas_dot_q16_i64", 3, Det::Pure),
    ("__mind_nerve_blas_matmul_q16_i64", 6, Det::Pure),
    ("__mind_nerve_blas_matmul_score_q16_i64", 5, Det::Pure),
    ("__mind_nerve_blas_qkt_q16_i64", 6, Det::Pure),
    ("__mind_nerve_lut_exp_h", 0, Det::Pure),
    ("__mind_nerve_lut_recip_h", 0, Det::Pure),
    ("__mind_nerve_lut_rsqrt_h", 0, Det::Pure),
    ("__mind_nerve_lut_tanh_h", 0, Det::Pure),
    // mind-nerve host/runtime FFI surface (src/runtime_ffi.mind): the envelope
    // CLI's clock / stdio / file / entropy / exit primitives, defined in the
    // mind-nerve runtime C shim and statically linked via
    // `[targets.*].native_sources`. Registered here — same as the
    // `__mind_nerve_blas_*` / `__mind_nerve_lut_*` surface — so the MLIR backend
    // emits a plain arity-checked `func.call @__mind_nerve_rt_*` instead of the
    // E2024 self-host-only advisory + runtime-JIT fallback. All args + result are
    // i64 (opaque heap addresses / byte counts / status codes); arities match the
    // `extern fn` decls in runtime_ffi.mind.
    // DETERMINISM (F5): this block is where the two hand-maintained classifier
    // arrays drifted — the clock / entropy / stdin / getenv rows below were
    // registered for EMISSION here and classified by neither, so a program whose
    // `main` returned `__mind_nerve_rt_monotonic_ns()` was attested
    // `deterministic` and passed `mindc verify --require-deterministic`.
    // `Det::World` is the honest verdict for the four channels the artifact
    // cannot name; the file reads and the write/exit effects stay `Det::Pure`
    // (the two `deferred:` blocks below say exactly why the line falls there).
    ("__mind_nerve_rt_exit", 1, Det::Pure),
    ("__mind_nerve_rt_file_size", 2, Det::Pure),
    ("__mind_nerve_rt_getenv", 4, Det::World),
    ("__mind_nerve_rt_monotonic_ns", 0, Det::World),
    ("__mind_nerve_rt_os_entropy", 2, Det::World),
    ("__mind_nerve_rt_read_file", 4, Det::Pure),
    ("__mind_nerve_rt_read_stdin", 2, Det::World),
    ("__mind_nerve_rt_write_stderr", 2, Det::Pure),
    ("__mind_nerve_rt_write_stdout", 2, Det::Pure),
    // Phase 17.3 — `f64` bit-cast surface. These three same-width coercions let
    // an `f64` aggregate be built on the existing i64 heap: `__mind_f64_to_bits`
    // reinterprets an `f64` as its i64 bit pattern for storage, `__mind_bits_to_f64`
    // reinterprets a loaded i64 back to `f64`, and `__mind_conv_f64` is the
    // pure-marker identity used by the enum-payload / declared-type coercion path.
    // All arity 1. The MLIR backend already lowers them (an `arith.bitcast`, or a
    // pass-through marker); registering them here lets std / user source name them
    // through the arity-checked cross-backend surface instead of the type checker
    // rejecting the call. All three are classified strict in `src/ir/fp_mode.rs`
    // (STRICT_FLOAT_INTRINSICS / dtype-recovery), so the FP attestation stays honest.
    ("__mind_bits_to_f64", 1, Det::Pure),
    ("__mind_conv_f64", 1, Det::Pure),
    ("__mind_f64_to_bits", 1, Det::Pure),
    ("__mind_free", 1, Det::Pure),
    ("__mind_load_i64", 1, Det::Pure),
    // RFC 0005 Phase 1.6 (task #306) — single-byte load/store. The
    // (`__mind_store_i64(base + i, b)` writes one byte / `__mind_load_i64(base + i) & 255`
    // reads one byte) convention used by `std.string` / `std.sha256` / `std.toml`
    // / `std.tui` clobbers 7 bytes per store and can read past the buffer. The
    // store form is currently masked by a 7-byte backing-store pad in runtime-support
    // (commit `cc5a513`), but the garbage past `len` is a cross-substrate
    // bit-identity landmine (NEON / RVV may not have the same pad). These two
    // intrinsics provide a proper one-byte ABI; `load_i8` zero-extends to i64 so
    // call sites preserve the `& 255` mask semantics during migration.
    ("__mind_load_i8", 1, Det::Pure),
    ("__mind_load_i32", 1, Det::Pure),
    ("__mind_load_i16", 1, Det::Pure),
    // `Det::Pure` is the NAME-level row only. `__mind_read`'s real verdict is
    // decided per CALL SITE from its `fd` argument — see `fd_dependent_read_arg`
    // / `read_fd_is_world` below: a read of an inherited standard stream, or of a
    // descriptor the call site cannot prove, is `World`.
    ("__mind_read", 4, Det::Pure),
    ("__mind_realloc", 2, Det::Pure),
    ("__mind_store_i64", 2, Det::Pure),
    ("__mind_store_i8", 2, Det::Pure),
    ("__mind_store_i32", 2, Det::Pure),
    ("__mind_store_i16", 2, Det::Pure),
    ("__mind_write", 4, Det::Pure),
    // `c.byte()` — the byte (low 8 bits) of a char/int receiver. The method-call
    // type-check validates it as a 1-arg call `byte(recv)`; lowering desugars it
    // to `recv & 0xFF` (see the MethodCall arm in eval/lower.rs). mind-flow's
    // lexer relies on it (`'0'.byte()`).
    ("byte", 1, Det::Pure),
];

// deferred: the raw-memory intrinsics `__mind_load_i{8,16,32,64}` /
// `__mind_store_i{8,16,32,64}` are classified `Det::Pure`, so a source-level
// `__mind_load_i64(arbitrary_addr)` reading uninitialized/OOB memory is attested
// `deterministic`. This is a DELIBERATE non-taint, not an oversight — reason:
// (1) every compiler-GENERATED use lowers only against a `__mind_alloc`-returned
// arena base with a prior deterministic store (RFC 0005 P0c: `vec.push`/
// struct-field/`std.string`/`std.sha256` bottom out here), so the load result IS
// a pure function of program inputs; (2) tainting the names would attest EVERY
// std-surface program (Vec/String/Map/sha256) as `nondeterministic` — a
// false-positive that breaks the honesty invariant in the OTHER direction
// (over-tainting is as dishonest as under-tainting). The residual risk is a
// hand-written source calling these `__`-internal intrinsics on an
// attacker-chosen address; that is a MEMORY-SAFETY (OOB/uninit-read) violation
// for the bounds/SSA layer to catch, not a name-match the determinism classifier
// can soundly distinguish from a legitimate arena load (both are syntactically
// `Call { name: "__mind_load_i64", args: [addr] }`). Upgrade path: if
// `__mind_load/store` ever become a public source API, gate them behind a
// pointer-provenance analysis that proves the address is arena-relative and
// in-range, and taint only the un-provable case — NOT a blanket name match.
// (The argument carried a third reason while the verdict lived in two arrays:
// that tainting here would contradict the AST-side classifier and diverge the
// layers. It is moot now that both layers read this one row.)

// deferred: FILE reads — `__mind_read` (name-level row; its per-call-site
// descriptor rule is below), `__mind_nerve_rt_read_file`,
// `__mind_nerve_rt_file_size` — are classified `Det::Pure` while the STDIN read
// `__mind_nerve_rt_read_stdin` is `Det::World`. The line is drawn where the
// classifier can name the input: a file read takes a path/fd the program itself
// supplies, so the content is in principle a DECLARED input the evidence chain
// could bind (content-hash it as an input link, RFC 0016 §4); a stdin stream is
// unnamed and unaddressable, which is why the pre-existing `read_line` /
// `read_input` source-level builtins were already tainted and why this keeps
// them so. The consequence of the other choice decided it: `__mind_read` is what
// `examples/mindc_mind/main.mind` (the self-hosted compiler) and `std.fs` /
// `std.io` use to read source files, so tainting it attests mindc's own output —
// and every file-processing program — as `nondeterministic`, the over-taint
// failure mode described on `Det`. Residual risk, stated plainly: a program that
// reads a file whose content changes between runs is attested `deterministic`
// today. Upgrade path: hash declared file inputs into the evidence chain as
// input links and taint only reads that cannot be bound to one.
//
// UPDATE — the STDIN half of that line is no longer left to the name. `__mind_read`
// is the same symbol for both channels, so `Det::Pure` on the row let
// `__mind_read(0, …)` (a raw stdin read, and what `std.io`'s
// `read_stdin_bytes` / `file_read(stdin(), …)` lower to) attest
// `determinism: deterministic`. The descriptor is an IR operand, so the verdict is
// now taken at the CALL SITE — `fd_dependent_read_arg` / `read_fd_is_world` here,
// the proof that `args[0]` is a known constant in `crate::ir::evidence`. What is
// still deferred is only the FILE half quoted above: a read from a proven
// non-standard descriptor stays `Pure`.

/// Non-registry `__mind_*` symbols exported by `runtime-support/mind_intrinsics.c`
/// that read the world. None is in [`STD_SURFACE_INTRINSICS`], so a source-level
/// call is today stopped on the artifact-producing paths by the
/// unregistered-intrinsic diagnostic (E2024) before determinism is ever consulted
/// — but that is an INCIDENTAL guard, not a determinism defence: it says nothing
/// about the clock, and it disappears the moment the symbol is registered.
/// Classifying them here means registering one later moves it into the table with
/// its verdict already argued, instead of silently opening a hole.
const UNREGISTERED_WORLD_INTRINSICS: &[&str] = &[
    // `__mind_blas_get_use_avx2` / `_set_use_avx2` read (and the setter returns)
    // the process-global host CPU-feature flag. A program that branches on it
    // produces substrate-dependent output — precisely what the cross-substrate
    // byte-identity invariant forbids — so both are `World`, not just the getter.
    "__mind_blas_get_use_avx2",
    "__mind_blas_set_use_avx2",
    // Wall-clock nanoseconds since the epoch. `runtime-support/mind_intrinsics.c`
    // documents it as "EXPLICITLY NON-DETERMINISTIC, by design", and `std/time.mind`
    // repeats the warning over its `now_ns()` wrapper; the classifier now says the
    // same thing.
    "__mind_now_ns",
];

/// Non-registry `__mind_*` symbols that are pure with respect to declared inputs.
/// Enumerated rather than covered by a blanket `__mind_*` rule: the blanket rule
/// is exactly what admitted the `__mind_nerve_rt_*` clock and entropy calls as
/// deterministic (see the module docs).
const UNREGISTERED_PURE_INTRINSICS: &[&str] = &[
    // Native-ELF self-host-only surface (`src/type_checker/resolve.rs`): the OS
    // argument vector and `open`. argv and a path are declared program inputs,
    // classified with the same reasoning as the file-read `deferred:` above.
    "__mind_argc",
    "__mind_argv",
    "__mind_open",
    // Aborts, bounds traps, allocator, generation-checked handles, region
    // bookkeeping, zeroed-vector helper: effects and arena addresses, not
    // varying results — the `__mind_load/store` `deferred:` above carries the
    // full argument for the allocator family.
    "__mind_assert_fail",
    "__mind_calloc",
    "__mind_gen_alloc",
    "__mind_gen_deref",
    "__mind_gen_free",
    "__mind_oob_check",
    "__mind_region_enter",
    "__mind_region_exit",
    "__mind_region_track",
    "__mind_vec_zeroed",
];

/// Bare non-deterministic SOURCE-level builtin names — PRNG draws that read
/// hidden generator state, and wall-clock / stdin reads. A compiled module that
/// calls one is genuinely non-deterministic (its output is not a pure function of
/// its inputs), so its evidence chain MUST honestly declare `nondeterministic`
/// rather than forge `deterministic` (the claim `mindc verify` reports). This is
/// the determinism wedge's honesty invariant: the attestation can never lie.
///
/// These are names, not registry rows, because they are surface spellings rather
/// than `__mind_*` primitives; they are matched on the bare name or the last
/// dotted / `::`-qualified segment, so `std.rand.random` and `rng::rand_uniform`
/// resolve to the same verdict. The legitimate, DETERMINISTIC randomness API is
/// the SEEDED counter-based form (`randn(shape, seed)`, `Random(seed=…)`,
/// Philox/Threefry) — those are pure functions of `(seed, index)` and are NOT in
/// this list. `randn` appears here as the BARE/unseeded draw; a seeded call is
/// resolved to its explicit generator, not this implicit builtin.
pub(crate) const NONDETERMINISTIC_BUILTINS: &[&str] = &[
    // PRNG draws.
    "rand",
    "rand_bytes",
    "rand_int",
    "rand_normal",
    "rand_range",
    "rand_seed",
    "rand_uniform",
    "randn",
    "random",
    "shuffle",
    // Wall-clock / nondeterministic environment reads. `now_ns` is `std.time`'s
    // wrapper over `__mind_now_ns`; without it, an imported `time.now_ns()` whose
    // body lives in another module reaches neither classifier.
    "monotonic_now",
    "now",
    "now_ns",
    "read_input",
    "read_line",
    "system_time",
    "time_now",
];

/// The determinism classification of an intrinsic, or `None` when the name is
/// not one the compiler knows about (a user function, a std surface call, an
/// unrecognised extern).
pub(crate) fn intrinsic_determinism(name: &str) -> Option<Det> {
    if let Some((_, _, det)) = STD_SURFACE_INTRINSICS.iter().find(|(n, _, _)| *n == name) {
        return Some(*det);
    }
    if UNREGISTERED_WORLD_INTRINSICS.contains(&name) {
        return Some(Det::World);
    }
    if UNREGISTERED_PURE_INTRINSICS.contains(&name) {
        return Some(Det::Pure);
    }
    None
}

/// Whether `callee` names something that can make the program's result vary
/// between runs on the same declared inputs — a registry row classified
/// [`Det::World`], or a bare non-deterministic builtin (matched on the bare name
/// or the last dotted / `::`-qualified segment, so `std.rand.random` and
/// `rng::rand_uniform` are caught too).
///
/// The SINGLE classifier: `ir::evidence` (what the evidence chain attests, and
/// what the determinism-by-default build gate and `mindc verify
/// --require-deterministic` re-derive) and `type_checker` (the `#[deterministic]`
/// call-graph check) both call this, so the two layers cannot drift apart again.
pub(crate) fn callee_is_nondeterministic(callee: &str) -> bool {
    let tail = callee.rsplit(['.', ':']).next().unwrap_or(callee);
    if intrinsic_determinism(callee) == Some(Det::World)
        || intrinsic_determinism(tail) == Some(Det::World)
    {
        return true;
    }
    NONDETERMINISTIC_BUILTINS.contains(&tail) || NONDETERMINISTIC_BUILTINS.contains(&callee)
}

// ---------------------------------------------------------------------------
// CALL-SITE classification: intrinsics whose determinism depends on an ARGUMENT
// ---------------------------------------------------------------------------
//
// `callee_is_nondeterministic` above answers by NAME. For `__mind_read` the name
// is not enough: the SAME symbol reads a file the program opened (a declared
// input) and reads the stdin stream (world input). Classifying it by name alone
// was a hole in the attestation — `fn main() -> i64 { __mind_read(0, buf, 1, -1) }`
// (or, identically, `std.io`'s `file_read(stdin(), …)`) piped its stdin byte into
// the process exit code and STILL emitted an artifact attesting
// `determinism: deterministic` that passed `mindc verify --require-deterministic`.
//
// The descriptor is statically visible at the call site — it is `args[0]` of the
// `Instr::Call` — so the verdict is decided there, by the IR-side classifier in
// `crate::ir::evidence`, using the two predicates below. This module owns the
// POLICY (which intrinsic, which argument, which descriptors are world); the IR
// module owns the ANALYSIS (proving what `args[0]` holds).

/// The POSIX standard-stream descriptors. These three are supplied by whoever
/// INVOKES the artifact — the artifact neither names nor opens them — so their
/// contents are not a declared input and two runs can disagree. `0` is the case
/// the finding names (stdin); `1`/`2` are the same class, and reading them is
/// world-input for the identical reason (a caller can point stdout/stderr at any
/// file or pipe, `mindc … 1<>file`), so all three are classified together rather
/// than leaving two trivially-equivalent spellings of the same forge open.
const INHERITED_STREAM_FDS: &[i64] = &[0, 1, 2];

/// For an intrinsic whose determinism depends on WHICH descriptor it reads, the
/// argument position holding that descriptor; `None` for every other callee.
///
/// Matched on the bare name or the last dotted / `::`-qualified segment, exactly
/// as [`callee_is_nondeterministic`] does, so a module-qualified spelling cannot
/// walk past the check.
pub(crate) fn fd_dependent_read_arg(callee: &str) -> Option<usize> {
    let tail = callee.rsplit(['.', ':']).next().unwrap_or(callee);
    // `__mind_read(fd, buf_addr, count, offset)` — RFC 0005 Phase 2, `std/io.mind`.
    // `__mind_write` is NOT here: writing is an observable EFFECT, not a channel
    // that makes the RESULT vary (the line `Det` draws).
    (tail == "__mind_read").then_some(0)
}

/// Whether a read from the descriptor `fd` is a world-input.
///
/// `fd` is the value the call site was PROVEN to pass. An UNPROVEN descriptor is
/// not this function's business — the IR-side classifier fails closed on it
/// (an unprovable descriptor can be `0`, so it may not be attested
/// deterministic); this predicate only decides the proven case.
///
/// deferred: a proven non-standard descriptor (`__mind_read(3, …)`) is admitted
/// as deterministic here, and it is NOT strictly a declared input — a hardcoded
/// `3` names an inherited slot in the process descriptor table just as `0` does,
/// it is merely one no runtime in this tree ever hands a program. Admitting it
/// keeps the line where the file-read `deferred:` block above draws it (a
/// descriptor the program itself supplies is in principle bindable as an input
/// link) instead of collapsing `__mind_read` to unconditional `World`, which is
/// the over-taint failure mode `Det` documents. Upgrade path: descriptor
/// PROVENANCE — prove the descriptor is the result of a `__mind_open` on a path
/// that the evidence chain content-hashes as an input link (RFC 0016 §4), and
/// admit only that case, tainting every other proven constant too.
pub(crate) fn read_fd_is_world(fd: i64) -> bool {
    INHERITED_STREAM_FDS.contains(&fd)
}

/// The declared i64 arity of a std-surface intrinsic, or `None` if `name` is not
/// in the cross-backend table.
#[cfg(feature = "std-surface")]
pub(crate) fn std_surface_intrinsic_arity(name: &str) -> Option<usize> {
    STD_SURFACE_INTRINSICS
        .iter()
        .find_map(|(n, arity, _)| (*n == name).then_some(*arity))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `__mind_*` symbol `runtime-support/mind_intrinsics.c` EXPORTS, in
    /// the order the C file declares them.
    fn exported_runtime_support_intrinsics() -> Vec<String> {
        const C_SRC: &str = include_str!("../runtime-support/mind_intrinsics.c");
        let mut names = Vec::new();
        for line in C_SRC.lines() {
            let line = line.trim_start();
            // `#  define MIND_EXPORT ...` starts with `#`, so macro definitions
            // are skipped and only definition sites are scanned.
            if !line.starts_with("MIND_EXPORT") {
                continue;
            }
            let Some(at) = line.find("__mind_") else {
                continue;
            };
            let rest = &line[at..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            // Only a call-shaped `__mind_x(` is a definition site.
            if rest[end..].starts_with('(') {
                names.push(rest[..end].to_string());
            }
        }
        names.sort();
        names.dedup();
        names
    }

    /// THE structural gate this module exists for: a new `__mind_*` intrinsic in
    /// the C runtime fails this test until someone classifies it, instead of
    /// being silently admitted as deterministic by a blanket `__mind_*` rule.
    #[test]
    fn classification_covers_every_runtime_support_intrinsic() {
        let exported = exported_runtime_support_intrinsics();
        // A scan that finds nothing must not pass vacuously.
        assert!(
            exported.len() >= 25,
            "runtime-support scan found only {} intrinsics — the scan is broken, \
             not the runtime: {exported:?}",
            exported.len()
        );
        let unclassified: Vec<&String> = exported
            .iter()
            .filter(|n| intrinsic_determinism(n).is_none())
            .collect();
        assert!(
            unclassified.is_empty(),
            "these `__mind_*` intrinsics are exported by runtime-support but not \
             classified deterministic or non-deterministic: {unclassified:?} — add \
             each to STD_SURFACE_INTRINSICS (with its Det) or to \
             UNREGISTERED_{{WORLD,PURE}}_INTRINSICS. An unclassified intrinsic that \
             reads the clock/entropy would be attested `deterministic`."
        );
    }

    /// The classifier must actually READ the registry row, not a parallel array.
    #[test]
    fn classifier_agrees_with_every_registry_row() {
        for (name, _, det) in STD_SURFACE_INTRINSICS {
            assert_eq!(
                callee_is_nondeterministic(name),
                *det == Det::World,
                "`{name}` is classified {det:?} in the registry but the classifier disagrees"
            );
        }
    }

    /// The F5 regression: the registered mind-nerve host surface (clock, entropy,
    /// stdin, getenv) and the wall clock must all report non-deterministic. Each
    /// of these was attested `deterministic` — and passed `mindc verify
    /// --require-deterministic` — before the registry carried the verdict.
    #[test]
    fn world_reading_intrinsics_are_nondeterministic() {
        for name in [
            "__mind_nerve_rt_monotonic_ns",
            "__mind_nerve_rt_os_entropy",
            "__mind_nerve_rt_read_stdin",
            "__mind_nerve_rt_getenv",
            "__mind_now_ns",
            "__mind_blas_get_use_avx2",
            "now_ns",
        ] {
            assert!(
                callee_is_nondeterministic(name),
                "`{name}` reads the world but is classified deterministic"
            );
        }
    }

    /// The pure side of the same invariant: the allocator / raw-memory / BLAS
    /// surface must NOT be tainted, or every std-surface program is attested
    /// `nondeterministic` (the over-taint failure mode).
    #[test]
    fn pure_intrinsics_are_not_tainted() {
        for name in [
            "__mind_alloc",
            "__mind_load_i64",
            "__mind_store_i64",
            "__mind_read",
            "__mind_write",
            "__mind_blas_dot_q16",
            "__mind_nerve_rt_write_stdout",
            "__mind_nerve_blas_dot_q16_i64",
        ] {
            assert!(
                !callee_is_nondeterministic(name),
                "`{name}` is pure w.r.t. declared inputs but is tainted"
            );
        }
    }

    #[test]
    fn registry_and_side_lists_do_not_overlap_or_repeat() {
        let mut seen = std::collections::BTreeSet::new();
        for (name, _, _) in STD_SURFACE_INTRINSICS {
            assert!(seen.insert(*name), "duplicate registry entry `{name}`");
        }
        for name in UNREGISTERED_WORLD_INTRINSICS
            .iter()
            .chain(UNREGISTERED_PURE_INTRINSICS)
        {
            assert!(
                seen.insert(*name),
                "`{name}` is in a side list AND the registry — the registry row wins, \
                 so the side entry is dead and can only mislead"
            );
        }
    }

    /// A qualified path must resolve to the same verdict as the bare name, or a
    /// module-qualified clock read walks straight past the classifier.
    #[test]
    fn qualified_paths_resolve_to_the_same_verdict() {
        assert!(callee_is_nondeterministic("std.rand.random"));
        assert!(callee_is_nondeterministic("rng::rand_uniform"));
        assert!(callee_is_nondeterministic("time.now_ns"));
        assert!(callee_is_nondeterministic(
            "nerve::__mind_nerve_rt_os_entropy"
        ));
        assert!(!callee_is_nondeterministic("std.blas.dot_q16"));
    }
}
