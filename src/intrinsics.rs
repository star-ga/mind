//! The intrinsic registry: the ONE table naming every `__mind_*` primitive the
//! compiler will emit, its i64 arity, and whether calling it can make the
//! program's result depend on something other than its declared inputs. Rows
//! with a canonical contract additionally own the logical spelling, result
//! policy, effect, offset policy, and backend profiles used for admission.
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
pub(crate) use crate::intrinsic_contract::{
    FIVE_I64, FOUR_I64, IntrinsicContract, IntrinsicEffect, IntrinsicImplementation,
    IntrinsicOffsetPolicy, IntrinsicProfile, IntrinsicResult, IntrinsicSpec, IntrinsicValueType,
    NO_I64, ONE_I64, SIX_I64, THREE_I64, TWO_I64, YMM_READ32_WRITE32, YMM_READ64_WRITE32,
    intrinsic_contract, intrinsic_supports_profile,
};
#[rustfmt::skip] // data table: the R-01b bank rows are one row per line (module ceiling)
pub(crate) const STD_SURFACE_INTRINSICS: &[IntrinsicSpec] = &[
    IntrinsicSpec::legacy("__mind_assert_fail", 1, Det::Pure), // Nonreturning runtime trap.
    IntrinsicSpec::frozen_value("__mind_argc", "argc", NO_I64, IntrinsicEffect::Argc, 10),
    IntrinsicSpec::frozen_value("__mind_argv", "argv", ONE_I64, IntrinsicEffect::Argv, 11),
    IntrinsicSpec::frozen_value(
        "__mind_alloc",
        "alloc",
        ONE_I64,
        IntrinsicEffect::ArenaAlloc,
        1,
    ),
    IntrinsicSpec::legacy("__mind_blas_dot_f32", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_f32_v", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_i16_v", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_l1_f32", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_l1_f32_v", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_l1_q16", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_l1_q16_v", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_linf_f32", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_linf_f32_v", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_q16", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_dot_q16_v", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_rmajor_f32", 5, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_rmajor_f32_v", 5, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_rmajor_i16_v", 5, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_rmajor_q16_v", 5, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_mm_i8_v", 6, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_mm_i8_mt_v", 6, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_mm_q16_v", 6, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_matmul_mm_q16_mt_v", 6, Det::Pure),
    IntrinsicSpec::legacy("__mind_blas_gemv_q16_mt", 5, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_blas_attnv_q16_i64", 6, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_blas_dot_q16_i64", 3, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_blas_matmul_q16_i64", 6, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_blas_matmul_score_q16_i64", 5, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_blas_qkt_q16_i64", 6, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_lut_exp_h", 0, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_lut_recip_h", 0, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_lut_rsqrt_h", 0, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_lut_tanh_h", 0, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_rt_exit", 1, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_rt_file_size", 2, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_rt_getenv", 4, Det::World),
    IntrinsicSpec::legacy("__mind_nerve_rt_monotonic_ns", 0, Det::World),
    IntrinsicSpec::legacy("__mind_nerve_rt_os_entropy", 2, Det::World),
    IntrinsicSpec::legacy("__mind_nerve_rt_read_file", 4, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_rt_read_stdin", 2, Det::World),
    IntrinsicSpec::legacy("__mind_nerve_rt_write_stderr", 2, Det::Pure),
    IntrinsicSpec::legacy("__mind_nerve_rt_write_stdout", 2, Det::Pure),
    IntrinsicSpec::legacy("__mind_bits_to_f64", 1, Det::Pure),
    IntrinsicSpec::legacy("__mind_conv_f64", 1, Det::Pure),
    IntrinsicSpec::legacy("__mind_f64_to_bits", 1, Det::Pure),
    IntrinsicSpec::legacy("__mind_free", 1, Det::Pure),
    IntrinsicSpec::frozen_value(
        "__mind_load_i64",
        "load_i64",
        ONE_I64,
        IntrinsicEffect::MemoryRead { width_bytes: 8 },
        3,
    ),
    IntrinsicSpec::frozen_value(
        "__mind_load_i8",
        "load8",
        ONE_I64,
        IntrinsicEffect::MemoryRead { width_bytes: 1 },
        5,
    ),
    IntrinsicSpec::legacy("__mind_load_i32", 1, Det::Pure),
    IntrinsicSpec::legacy("__mind_load_i16", 1, Det::Pure),
    IntrinsicSpec::legacy("__mind_string_eq", 2, Det::Pure),
    IntrinsicSpec::frozen_value(
        "__mind_open",
        "open",
        ONE_I64,
        IntrinsicEffect::OpenReadOnly,
        12,
    ),
    IntrinsicSpec::frozen_io("__mind_read", "read", FOUR_I64, IntrinsicEffect::FdRead, 9),
    IntrinsicSpec::legacy("__mind_realloc", 2, Det::Pure),
    IntrinsicSpec::frozen_discard(
        "__mind_store_i64",
        "store_i64",
        TWO_I64,
        IntrinsicEffect::MemoryWrite { width_bytes: 8 },
        2,
    ),
    IntrinsicSpec::frozen_discard(
        "__mind_store_i8",
        "store8",
        TWO_I64,
        IntrinsicEffect::MemoryWrite { width_bytes: 1 },
        4,
    ),
    IntrinsicSpec::legacy("__mind_store_i32", 2, Det::Pure),
    IntrinsicSpec::frozen_discard(
        "__mind_store_i16",
        "store_i16",
        TWO_I64,
        IntrinsicEffect::MemoryWrite { width_bytes: 2 },
        13,
    ),
    // runtime-support/mind_intrinsics.c gives it a STRONG definition that the
    // `--emit-shared` path links into every cdylib, and `Instr::Call` lowers it as a
    // plain `func.call` (it is not in `classify_intrinsic`'s inline set) exactly like
    // `__mind_alloc`/`__mind_free`. Unregistered, the E2024 advisory fired and
    // `pipeline.rs` turned that WARNING into a hard compile abort, so
    // `mindc std/vec.mind --emit-shared` exited 1 and emitted no .so at all.
    IntrinsicSpec::legacy("__mind_vec_zeroed", 1, Det::Pure),
    IntrinsicSpec::frozen_io(
        "__mind_write",
        "write",
        FOUR_I64,
        IntrinsicEffect::FdWrite,
        8,
    ),
    // Raw filesystem calls observe descriptor/process/filesystem state.
    // Keep this state visible to determinism admission and evidence labels.
    IntrinsicSpec::legacy("__mind_sys_openat", 4, Det::World),
    IntrinsicSpec::legacy("__mind_sys_close", 1, Det::World),
    IntrinsicSpec::legacy("__mind_sys_fstat", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_newfstatat", 4, Det::World),
    IntrinsicSpec::legacy("__mind_sys_fsync", 1, Det::World),
    IntrinsicSpec::legacy("__mind_sys_flock", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_fcntl", 3, Det::World),
    IntrinsicSpec::legacy("__mind_sys_lseek", 3, Det::World),
    IntrinsicSpec::legacy("__mind_sys_unlinkat", 3, Det::World),
    IntrinsicSpec::legacy("__mind_sys_renameat", 4, Det::World),
    IntrinsicSpec::legacy("__mind_sys_umask", 1, Det::World),
    IntrinsicSpec::legacy("__mind_sys_getuid", 0, Det::World),
    IntrinsicSpec::legacy("__mind_sys_fork", 0, Det::World),
    IntrinsicSpec::legacy("__mind_sys_pipe2", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_dup2", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_close_range", 3, Det::World),
    IntrinsicSpec::legacy("__mind_sys_execveat", 5, Det::World),
    IntrinsicSpec::legacy("__mind_sys_wait4", 4, Det::World),
    IntrinsicSpec::legacy("__mind_sys_waitid", 5, Det::World),
    IntrinsicSpec::legacy("__mind_sys_kill", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_setpgid", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_getpgid", 1, Det::World),
    IntrinsicSpec::legacy("__mind_sys_prctl", 5, Det::World),
    IntrinsicSpec::legacy("__mind_sys_getrlimit", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_setrlimit", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_rt_sigaction", 4, Det::World),
    IntrinsicSpec::legacy("__mind_sys_nanosleep", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_clock_gettime", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_memfd_create", 2, Det::World),
    IntrinsicSpec::legacy("__mind_sys_fchdir", 1, Det::World),
    IntrinsicSpec::legacy("__mind_sys_getdents64", 3, Det::World),
    IntrinsicSpec::legacy("__mind_sys_exit_group", 1, Det::World),
    // Reserved per-lane row blocks (kind ledger: examples/mindc_mind/nb_part_dispatch.mind).
    // --- rows: ymm (Layer A + R-02) ---
    // R-01 Layer A + R-02: x86-64 AVX2 ymm rows, lowered only by a native
    // compiler carrying `nbymm001` (kinds 60-66). Each operand is an address of
    // a 32-byte vector. MLIR has no lowering; registered rows carry no E2024.
    IntrinsicSpec::frozen_native_value(
        "__mind_vpaddd_i32x8",
        "vpaddd_i32x8",
        TWO_I64,
        YMM_READ64_WRITE32,
        60,
    ),
    IntrinsicSpec::frozen_native_value(
        "__mind_vpsubd_i32x8",
        "vpsubd_i32x8",
        TWO_I64,
        YMM_READ64_WRITE32,
        61,
    ),
    IntrinsicSpec::frozen_native_value(
        "__mind_vpaddb_i8x32",
        "vpaddb_i8x32",
        TWO_I64,
        YMM_READ64_WRITE32,
        62,
    ),
    IntrinsicSpec::frozen_native_value(
        "__mind_vpshufb_i8x32",
        "vpshufb_i8x32",
        THREE_I64,
        YMM_READ64_WRITE32,
        63,
    ),
    IntrinsicSpec::frozen_native_value(
        "__mind_vmovdqu_i32x8",
        "vmovdqu_i32x8",
        TWO_I64,
        YMM_READ32_WRITE32,
        64,
    ),
    IntrinsicSpec::frozen_native_value(
        "__mind_vpaddq_i64x4",
        "vpaddq_i64x4",
        TWO_I64,
        YMM_READ64_WRITE32,
        65,
    ),
    IntrinsicSpec::frozen_native_value(
        "__mind_vpsubq_i64x4",
        "vpsubq_i64x4",
        TWO_I64,
        YMM_READ64_WRITE32,
        66,
    ),
    // --- end rows: ymm ---
    // --- rows: cpu_features ---
    // R-C: host CPU-feature bits (0 SSE2, 1 AVX+OSXSAVE+XGETBV, 2 AVX2,
    // 3 BMI2, 4 AVX-VNNI, 5 RDTSCP). A host-state read, so World: a program
    // reaching it is attested nondeterministic. Kind 67, mark `nbcpuf01`.
    IntrinsicSpec {
        physical_name: "__mind_cpu_features_v1",
        arity: 0,
        det: Det::World,
        contract: Some(IntrinsicContract {
            logical_name: "cpu_features_v1",
            parameters: NO_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: IntrinsicImplementation::FrozenNative { seed_id: 67 },
        }),
    },
    // --- end rows: cpu_features ---
    // --- rows: measure (M-A3) ---
    // Kinds 68, 69, sys 93 (examples/mindc_mind/nb_measure.mind). Each reads host
    // state, so `Det::World`: a program reaching one is attested `nondeterministic`
    // and `mindc verify --require-deterministic` refuses it. Native-only (x86-64
    // Linux); the Rust/MLIR leg refuses the names. Struct literals because the
    // `frozen_native_*` constructors are `Det::Pure`.
    IntrinsicSpec {
        physical_name: "__mind_rdtscp_v1",
        arity: 0,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "rdtscp_v1",
            parameters: NO_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 68,
            },
        }),
    },
    IntrinsicSpec {
        physical_name: "__mind_lfence_v1",
        arity: 0,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "lfence_v1",
            parameters: NO_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 69,
            },
        }),
    },
    IntrinsicSpec {
        physical_name: "__mind_sys_perf_event_open",
        arity: 5,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "sys_perf_event_open",
            parameters: crate::intrinsic_contract::FIVE_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 93,
            },
        }),
    },
    // --- end rows: measure ---
    // --- rows: shm (A5) ---
    // Kinds 84-86, sys 90-92 (examples/mindc_mind/nb_shm.mind), native-only. Det::World: memory
    // another process writes, host mappings and affinity. Struct literals: frozen_native_* is Pure.
    IntrinsicSpec {
        physical_name: "__mind_cpu_relax",
        arity: 0,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "cpu_relax",
            parameters: NO_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 84,
            },
        }),
    },
    IntrinsicSpec {
        physical_name: "__mind_atomic_load_acquire_i64",
        arity: 1,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "atomic_load_acquire_i64",
            parameters: ONE_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 85,
            },
        }),
    },
    IntrinsicSpec {
        physical_name: "__mind_atomic_store_release_i64",
        arity: 2,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "atomic_store_release_i64",
            parameters: TWO_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::MemoryWrite { width_bytes: 8 },
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 86,
            },
        }),
    },
    IntrinsicSpec {
        physical_name: "__mind_sys_mmap",
        arity: 6,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "sys_mmap",
            parameters: crate::intrinsic_contract::SIX_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 90,
            },
        }),
    },
    IntrinsicSpec {
        physical_name: "__mind_sys_munmap",
        arity: 2,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "sys_munmap",
            parameters: TWO_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 91,
            },
        }),
    },
    IntrinsicSpec {
        physical_name: "__mind_sys_sched_setaffinity",
        arity: 3,
        det: Det::World,
        contract: Some(crate::intrinsic_contract::IntrinsicContract {
            logical_name: "sys_sched_setaffinity",
            parameters: crate::intrinsic_contract::THREE_I64,
            result: IntrinsicResult::I64,
            effect: IntrinsicEffect::HostState,
            offset_policy: crate::intrinsic_contract::IntrinsicOffsetPolicy::Any,
            profiles: &[IntrinsicProfile::FrozenNative],
            implementation: crate::intrinsic_contract::IntrinsicImplementation::FrozenNative {
                seed_id: 92,
            },
        }),
    },
    // --- end rows: shm ---
    IntrinsicSpec::frozen_native_value("__mind_group_bank_i64", "group_bank_i64", FIVE_I64, IntrinsicEffect::MemoryWrite { width_bytes: 8 }, 87),
    IntrinsicSpec::frozen_native_value("__mind_group_acc32_i64", "group_acc32_i64", SIX_I64, IntrinsicEffect::MemoryWrite { width_bytes: 8 }, 88),
    // ymm bank rows 70-83 and 94-99: literal ymm, base and disp. Result is discard-only.
    IntrinsicSpec::frozen_native_only("__mind_ymm_vmovdqu_ld", "ymm_vmovdqu_ld", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 32, writes: 0 }, IntrinsicOffsetPolicy::Any, 70),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vmovdqu_st", "ymm_vmovdqu_st", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 32 }, IntrinsicOffsetPolicy::Any, 71),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpaddd_i32x8_m", "ymm_vpaddd_i32x8_m", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 32, writes: 0 }, IntrinsicOffsetPolicy::Any, 72),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpsubd_i32x8_m", "ymm_vpsubd_i32x8_m", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 32, writes: 0 }, IntrinsicOffsetPolicy::Any, 73),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpaddb_i8x32_m", "ymm_vpaddb_i8x32_m", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 32, writes: 0 }, IntrinsicOffsetPolicy::Any, 74),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpshufb_i8x32_m", "ymm_vpshufb_i8x32_m", FOUR_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 32, writes: 0 }, IntrinsicOffsetPolicy::Any, 75),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpaddd_i32x8_r", "ymm_vpaddd_i32x8_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 76),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpsubd_i32x8_r", "ymm_vpsubd_i32x8_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 77),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpaddb_i8x32_r", "ymm_vpaddb_i8x32_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 78),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpshufb_i8x32_r", "ymm_vpshufb_i8x32_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 79),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpaddq_i64x4_m", "ymm_vpaddq_i64x4_m", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 32, writes: 0 }, IntrinsicOffsetPolicy::Any, 80),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpsubq_i64x4_m", "ymm_vpsubq_i64x4_m", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 32, writes: 0 }, IntrinsicOffsetPolicy::Any, 81),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpaddq_i64x4_r", "ymm_vpaddq_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 82),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpsubq_i64x4_r", "ymm_vpsubq_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 83),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpmuludq_i64x4_r", "ymm_vpmuludq_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 94),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpand_i64x4_r", "ymm_vpand_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 95),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpor_i64x4_r", "ymm_vpor_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 96),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpcmpgtq_i64x4_r", "ymm_vpcmpgtq_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 97),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpsrlq_i64x4_r", "ymm_vpsrlq_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 98),
    IntrinsicSpec::frozen_native_only("__mind_ymm_vpsllq_i64x4_r", "ymm_vpsllq_i64x4_r", THREE_I64, IntrinsicResult::DiscardOnly, IntrinsicEffect::YmmMemory { reads: 0, writes: 0 }, IntrinsicOffsetPolicy::Any, 99),
    // --- end rows: ymm_bank ---
    // `c.byte()` — the byte (low 8 bits) of a char/int receiver. The method-call
    // type-check validates it as a 1-arg call `byte(recv)`; lowering desugars it
    // to `recv & 0xFF` (see the MethodCall arm in eval/lower.rs). mind-flow's
    // lexer relies on it (`'0'.byte()`).
    IntrinsicSpec::legacy("byte", 1, Det::Pure),
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
/// Non-registry symbols that read the world: `__mind_*` exports of the runtime
/// shim (E2024 stops a source call today only by accident, so the verdict is
/// argued here before one is registered), the libc/POSIX symbols the shipped std
/// calls, and the run-time loader (spec "Extern bindings").
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
    // ---- libc / POSIX symbols the shipped std declares and CALLS ----------
    //
    // These are World by the enum's own definition: each reads a channel the
    // artifact does not name. They are listed because leaving them UNCLASSIFIED
    // was measured to be a determinism REGRESSION, not merely a labelling gap.
    //
    // Before the shipping veto was split from the attestation label, an unclassified extern blocked
    // artifact emission, so these were caught by accident. After the split the hard gate consults
    // only genuine World rows, and a program calling `clock_gettime` EMITTED with exit 0 — a
    // wall-clock read passing a gate whose own doc scopes it to "PRNG / wall-clock / stdin".
    // Classifying them restores the block for real nondeterminism without re-blocking `memset`,
    // which is the whole point of the split.
    // Each row names the unnamed channel it reads:
    "clock_gettime",     // wall / monotonic clock
    "getenv",            // process environment
    "getpid",            // OS-assigned process identity
    "getcwd",            // process working directory
    "isatty",            // terminal state of a descriptor
    "sched_getaffinity", // host CPU topology
    "syscall",           // arbitrary kernel entry — unbounded by construction
    "mmap",              // kernel-chosen address, and file bytes when fd is real
    "getrandom",         // kernel CSPRNG bytes (std.entropy)
    "getaddrinfo",       // hosts file, resolver config, network (std.dns)
    "dlopen",            // the run-time loader reads host state: which files exist where
    "dlsym",
    "dlclose",
    "dlerror",
    // Descriptor / socket reads: the bytes come from outside the artifact. This
    // is the discriminator the classification pass named -- `read(fd, buf, n)`
    // writes through a caller pointer exactly as `memset` does, and the two
    // differ in the SOURCE of the written bytes, never in the write itself.
    "read",
    "recv",
    "recvfrom",
    "readdir",
    "accept",
    "connect",
    // `stat`/`lstat` fill a buffer that CONTAINS A TIMESTAMP: std/fs.mind's
    // `filestat_mtime` reads `st_mtim` back out of it (fs.mind:214). A row
    // claiming these read only "the filesystem" would omit the clock.
    "stat",
    "lstat",
    // Kernel-assigned identifiers handed back through an out-buffer. The dead
    // return is a trap: `let _ = getsockname(...)` at std/net.mind:105-117 still
    // yields a kernel-chosen ephemeral port via the buffer.
    "getsockname",
    "socket",
    "bind",
    "listen",
    "ioctl",   // device state
    "fork",    // returns a pid, and diverges the process
    "waitpid", // child scheduling outcome
];
/// Non-registry `__mind_*` symbols that are pure with respect to declared inputs.
/// Enumerated rather than covered by a blanket `__mind_*` rule: the blanket rule
/// is exactly what admitted the `__mind_nerve_rt_*` clock and entropy calls as
/// deterministic (see the module docs).
const UNREGISTERED_PURE_INTRINSICS: &[&str] = &[
    // `__mind_argc` and `__mind_argv` have exact contract rows above in both
    // profiles: the MLIR path calls the runtime-support pair, which keeps the
    // native semantics (argv[i] read without a bounds check).
    // Aborts, bounds traps, allocator, generation-checked handles, region
    // bookkeeping: effects and arena addresses, not varying results — the `__mind_load/store` `deferred:` above carries the
    // full argument for the allocator family.
    "__mind_calloc",
    "__mind_gen_alloc",
    "__mind_gen_deref",
    "__mind_gen_free",
    "__mind_oob_check",
    "__mind_region_enter",
    "__mind_region_exit",
    "__mind_region_track",
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
    // `std.procgroup`'s world readers, for the same reason: an imported call's
    // body (`fork`, `getenv`, `clock_gettime`, `waitpid`) is not in the caller's
    // IR. `procgroup_spawn` forks and reads the environment, `procgroup_wait_all`
    // reads the clock for its deadline and reaps whatever finished, and
    // `procgroup_task_index` reads the environment. The other procgroup calls
    // read the group's own memory and stay pure.
    "procgroup_spawn",
    "procgroup_task_index",
    "procgroup_wait_all",
    // `std.procgroup_sys`, the effect module those three reach their OS calls
    // through: each straight-line wrapper whose one OS call is a World row
    // above (`fork`, `waitpid`, `read`, `clock_gettime`, `getenv`). The child
    // descriptor walk, the pollfd records, the deadline arithmetic and the
    // environment save and restore live in `std.procgroup`. close, kill, poll,
    // pipe, dup2, exec, setenv and unsetenv call no World row and stay pure,
    // like `procgroup_new`.
    "procgroup_sys_clock",
    "procgroup_sys_env_read",
    "procgroup_sys_fork",
    "procgroup_sys_read",
    "procgroup_sys_wait",
    "read_input",
    "read_line",
    "system_time",
    "time_now",
    // `std.dns` / `std.entropy` are deliberately absent. Matching the bare name
    // refused a user function spelled the same way. A call is a world read only
    // when it resolves to that module, by its qualifier or, bare, by an import
    // of that module and no definition of the name (`crate::std_qualified`).
];
/// The determinism classification of an intrinsic, or `None` when the name is
/// not one the compiler knows about (a user function, a std surface call, an
/// unrecognised extern).
pub(crate) fn intrinsic_determinism(name: &str) -> Option<Det> {
    if let Some(spec) = STD_SURFACE_INTRINSICS
        .iter()
        .find(|spec| spec.physical_name == name)
    {
        return Some(spec.det);
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
/// The declared i64 arity of a registered intrinsic in ANY profile, or `None`
/// if `name` is not in the table. The type checker's arity/i64 check and the
/// IR call-arity verifier key on it, so a FrozenNative-only row type-checks
/// (and reaches `--emit-mic3`), and E2024 fires only for a `__mind_*` name
/// registered in no profile (`lookup is None`).
pub(crate) fn registered_intrinsic_arity(name: &str) -> Option<usize> {
    STD_SURFACE_INTRINSICS
        .iter()
        .find(|spec| spec.physical_name == name)
        .map(|spec| spec.arity)
}
/// The declared i64 arity of a Rust/MLIR-selectable std-surface intrinsic, or
/// `None` if `name` is native-only or not in the cross-backend table.
#[cfg(feature = "std-surface")]
pub(crate) fn std_surface_intrinsic_arity(name: &str) -> Option<usize> {
    STD_SURFACE_INTRINSICS.iter().find_map(|spec| {
        if spec.physical_name != name {
            return None;
        }
        let selectable = spec
            .contract
            .is_none_or(|contract| contract.profiles.contains(&IntrinsicProfile::RustMlir));
        selectable.then_some(spec.arity)
    })
}
#[cfg(test)]
mod tests;
