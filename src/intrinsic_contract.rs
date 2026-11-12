// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

//! Contract metadata shared by the intrinsic registry and canonical types.

use crate::intrinsics::Det;

/// The native result-use policy visible to canonical metadata. Stores retain
/// their historical i64 wire return for compatibility, but the FrozenNative
/// emitter does not define a usable synthetic destination for them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IntrinsicResult {
    I64,
    DiscardOnly,
}

/// A backend profile whose ABI is described by a contract row. FrozenNative
/// provenance is kept distinct from the existing Rust/MLIR rows; a profile is
/// present only when that emitter's support is established by the registry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IntrinsicProfile {
    FrozenNative,
    RustMlir,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IntrinsicEffect {
    Argc,
    Argv,
    OpenReadOnly,
    ArenaAlloc,
    MemoryRead {
        width_bytes: u8,
    },
    MemoryWrite {
        width_bytes: u8,
    },
    FdRead,
    FdWrite,
    /// 32-byte ymm memory operands read and written per call (rows land in the
    /// reserved registry blocks; row metadata, no consumer).
    #[allow(dead_code)]
    YmmMemory {
        reads: u8,
        writes: u8,
    },
    /// Reads host state (CPUID, the TSC, memory another process writes): Det::World rows.
    #[allow(dead_code)]
    HostState,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IntrinsicOffsetPolicy {
    Any,
    MustBeMinusOneIgnored,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IntrinsicValueType {
    I64,
}

/// The proven emitter implementation attached to a contract.  The seed ID is
/// provenance, not a public wire or CLI selector; later admission code can
/// require the exact frozen implementation before assigning native capability.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IntrinsicImplementation {
    FrozenNative { seed_id: u8 },
}

/// One row in the intrinsic registry.  Legacy rows retain their historical
/// arity/determinism behaviour, while the ten FrozenNative rows additionally
/// carry exact logical spelling, ABI result, effect, and profile metadata.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct IntrinsicSpec {
    pub(crate) physical_name: &'static str,
    pub(crate) arity: usize,
    pub(crate) det: Det,
    pub(crate) contract: Option<IntrinsicContract>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct IntrinsicContract {
    pub(crate) logical_name: &'static str,
    pub(crate) parameters: &'static [IntrinsicValueType],
    pub(crate) result: IntrinsicResult,
    pub(crate) effect: IntrinsicEffect,
    pub(crate) offset_policy: IntrinsicOffsetPolicy,
    pub(crate) profiles: &'static [IntrinsicProfile],
    pub(crate) implementation: IntrinsicImplementation,
}

const FROZEN_NATIVE_ONLY: &[IntrinsicProfile] = &[IntrinsicProfile::FrozenNative];
const FROZEN_AND_RUST_MLIR: &[IntrinsicProfile] =
    &[IntrinsicProfile::FrozenNative, IntrinsicProfile::RustMlir];

impl IntrinsicSpec {
    pub(crate) const fn legacy(physical_name: &'static str, arity: usize, det: Det) -> Self {
        Self {
            physical_name,
            arity,
            det,
            contract: None,
        }
    }

    pub(crate) const fn frozen(
        physical_name: &'static str,
        logical_name: &'static str,
        parameters: &'static [IntrinsicValueType],
        result: IntrinsicResult,
        effect: IntrinsicEffect,
        offset_policy: IntrinsicOffsetPolicy,
        seed_id: u8,
    ) -> Self {
        Self {
            physical_name,
            arity: parameters.len(),
            det: Det::Pure,
            contract: Some(IntrinsicContract {
                logical_name,
                parameters,
                result,
                effect,
                offset_policy,
                profiles: FROZEN_AND_RUST_MLIR,
                implementation: IntrinsicImplementation::FrozenNative { seed_id },
            }),
        }
    }

    pub(crate) const fn frozen_native_only(
        physical_name: &'static str,
        logical_name: &'static str,
        parameters: &'static [IntrinsicValueType],
        result: IntrinsicResult,
        effect: IntrinsicEffect,
        offset_policy: IntrinsicOffsetPolicy,
        seed_id: u8,
    ) -> Self {
        Self {
            physical_name,
            arity: parameters.len(),
            det: Det::Pure,
            contract: Some(IntrinsicContract {
                logical_name,
                parameters,
                result,
                effect,
                offset_policy,
                profiles: FROZEN_NATIVE_ONLY,
                implementation: IntrinsicImplementation::FrozenNative { seed_id },
            }),
        }
    }

    pub(crate) const fn frozen_value(
        physical_name: &'static str,
        logical_name: &'static str,
        parameters: &'static [IntrinsicValueType],
        effect: IntrinsicEffect,
        seed_id: u8,
    ) -> Self {
        Self::frozen(
            physical_name,
            logical_name,
            parameters,
            IntrinsicResult::I64,
            effect,
            IntrinsicOffsetPolicy::Any,
            seed_id,
        )
    }

    pub(crate) const fn frozen_native_value(
        physical_name: &'static str,
        logical_name: &'static str,
        parameters: &'static [IntrinsicValueType],
        effect: IntrinsicEffect,
        seed_id: u8,
    ) -> Self {
        Self::frozen_native_only(
            physical_name,
            logical_name,
            parameters,
            IntrinsicResult::I64,
            effect,
            IntrinsicOffsetPolicy::Any,
            seed_id,
        )
    }

    pub(crate) const fn frozen_discard(
        physical_name: &'static str,
        logical_name: &'static str,
        parameters: &'static [IntrinsicValueType],
        effect: IntrinsicEffect,
        seed_id: u8,
    ) -> Self {
        Self::frozen(
            physical_name,
            logical_name,
            parameters,
            IntrinsicResult::DiscardOnly,
            effect,
            IntrinsicOffsetPolicy::Any,
            seed_id,
        )
    }

    pub(crate) const fn frozen_io(
        physical_name: &'static str,
        logical_name: &'static str,
        parameters: &'static [IntrinsicValueType],
        effect: IntrinsicEffect,
        seed_id: u8,
    ) -> Self {
        Self::frozen(
            physical_name,
            logical_name,
            parameters,
            IntrinsicResult::I64,
            effect,
            IntrinsicOffsetPolicy::MustBeMinusOneIgnored,
            seed_id,
        )
    }
}

pub(crate) const NO_I64: &[IntrinsicValueType] = &[];
pub(crate) const ONE_I64: &[IntrinsicValueType] = &[IntrinsicValueType::I64];
pub(crate) const TWO_I64: &[IntrinsicValueType] =
    &[IntrinsicValueType::I64, IntrinsicValueType::I64];
// Reserved for the vpshufb / bank / sys rows (3 args); no row uses it yet.
#[allow(dead_code)]
pub(crate) const THREE_I64: &[IntrinsicValueType] = &[IntrinsicValueType::I64; 3];
pub(crate) const FOUR_I64: &[IntrinsicValueType] = &[
    IntrinsicValueType::I64,
    IntrinsicValueType::I64,
    IntrinsicValueType::I64,
    IntrinsicValueType::I64,
];
// Five i64 args. group_bank_i64 uses this row.
pub(crate) const FIVE_I64: &[IntrinsicValueType] = &[IntrinsicValueType::I64; 5];
// Six i64 args. sys_mmap and group_acc32_i64 use this row.
pub(crate) const SIX_I64: &[IntrinsicValueType] = &[IntrinsicValueType::I64; 6];

/// The ymm row effects: two 32-byte vectors read and one written (a vmovdqu
/// copy reads one).
pub(crate) const YMM_READ64_WRITE32: IntrinsicEffect = IntrinsicEffect::YmmMemory {
    reads: 64,
    writes: 32,
};
pub(crate) const YMM_READ32_WRITE32: IntrinsicEffect = IntrinsicEffect::YmmMemory {
    reads: 32,
    writes: 32,
};

/// Look up a canonical intrinsic declaration by its logical identity name.
/// The owner is checked by the canonical-types layer; only exact contract rows
/// are exposed here.
pub(crate) fn intrinsic_contract(logical_name: &str) -> Option<&'static IntrinsicContract> {
    crate::intrinsics::STD_SURFACE_INTRINSICS
        .iter()
        .find_map(|spec| {
            spec.contract
                .as_ref()
                .filter(|contract| contract.logical_name == logical_name)
        })
}

pub(crate) fn intrinsic_supports_profile(logical_name: &str, profile: IntrinsicProfile) -> bool {
    intrinsic_contract(logical_name).is_some_and(|contract| contract.profiles.contains(&profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsics::STD_SURFACE_INTRINSICS;
    #[cfg(feature = "std-surface")]
    use crate::intrinsics::std_surface_intrinsic_arity;

    #[test]
    fn frozen_native_contract_registry_has_exact_first_cut_surface() {
        let mut rows: Vec<&IntrinsicContract> = STD_SURFACE_INTRINSICS
            .iter()
            .filter_map(|spec| spec.contract.as_ref())
            .collect();
        rows.sort_by_key(|contract| contract.logical_name);
        assert_eq!(
            rows.iter()
                .map(|contract| contract.logical_name)
                .collect::<Vec<_>>(),
            vec![
                "alloc",
                "argc",
                "argv",
                "atomic_load_acquire_i64",
                "atomic_store_release_i64",
                "cpu_features_v1",
                "cpu_relax",
                "group_acc32_i64",
                "group_bank_i64",
                "lfence_v1",
                "load8",
                "load_i64",
                "open",
                "rdtscp_v1",
                "read",
                "store8",
                "store_i16",
                "store_i64",
                "sys_mmap",
                "sys_munmap",
                "sys_perf_event_open",
                "sys_sched_setaffinity",
                "vmovdqu_i32x8",
                "vpaddb_i8x32",
                "vpaddd_i32x8",
                "vpaddq_i64x4",
                "vpshufb_i8x32",
                "vpsubd_i32x8",
                "vpsubq_i64x4",
                "write",
                "ymm_vmovdqu_ld",
                "ymm_vmovdqu_st",
                "ymm_vpaddb_i8x32_m",
                "ymm_vpaddb_i8x32_r",
                "ymm_vpaddd_i32x8_m",
                "ymm_vpaddd_i32x8_r",
                "ymm_vpaddq_i64x4_m",
                "ymm_vpaddq_i64x4_r",
                "ymm_vpand_i64x4_r",
                "ymm_vpcmpgtq_i64x4_r",
                "ymm_vpmuludq_i64x4_r",
                "ymm_vpor_i64x4_r",
                "ymm_vpshufb_i8x32_m",
                "ymm_vpshufb_i8x32_r",
                "ymm_vpsllq_i64x4_r",
                "ymm_vpsrlq_i64x4_r",
                "ymm_vpsubd_i32x8_m",
                "ymm_vpsubd_i32x8_r",
                "ymm_vpsubq_i64x4_m",
                "ymm_vpsubq_i64x4_r",
            ]
        );
        for (name, seed_id) in [
            ("argc", 10),
            ("argv", 11),
            ("open", 12),
            ("alloc", 1),
            ("load_i64", 3),
            ("store_i64", 2),
            ("load8", 5),
            ("store8", 4),
            ("store_i16", 13),
            ("read", 9),
            ("write", 8),
            ("vpaddd_i32x8", 60),
            ("vpsubd_i32x8", 61),
            ("vpaddb_i8x32", 62),
            ("vpshufb_i8x32", 63),
            ("vmovdqu_i32x8", 64),
            ("vpaddq_i64x4", 65),
            ("vpsubq_i64x4", 66),
            ("cpu_features_v1", 67),
            ("rdtscp_v1", 68),
            ("lfence_v1", 69),
            ("cpu_relax", 84),
            ("group_bank_i64", 87),
            ("group_acc32_i64", 88),
            ("atomic_load_acquire_i64", 85),
            ("atomic_store_release_i64", 86),
            ("sys_mmap", 90),
            ("sys_munmap", 91),
            ("sys_sched_setaffinity", 92),
            ("sys_perf_event_open", 93),
        ] {
            assert!(matches!(
                intrinsic_contract(name).expect("contract row").implementation,
                IntrinsicImplementation::FrozenNative { seed_id: id } if id == seed_id
            ));
        }
        for (name, effect) in [
            ("alloc", IntrinsicEffect::ArenaAlloc),
            ("argc", IntrinsicEffect::Argc),
            ("argv", IntrinsicEffect::Argv),
            ("load8", IntrinsicEffect::MemoryRead { width_bytes: 1 }),
            ("load_i64", IntrinsicEffect::MemoryRead { width_bytes: 8 }),
            ("open", IntrinsicEffect::OpenReadOnly),
            ("read", IntrinsicEffect::FdRead),
            ("store8", IntrinsicEffect::MemoryWrite { width_bytes: 1 }),
            ("store_i16", IntrinsicEffect::MemoryWrite { width_bytes: 2 }),
            ("store_i64", IntrinsicEffect::MemoryWrite { width_bytes: 8 }),
            ("write", IntrinsicEffect::FdWrite),
            (
                "vmovdqu_i32x8",
                IntrinsicEffect::YmmMemory {
                    reads: 32,
                    writes: 32,
                },
            ),
            (
                "vpshufb_i8x32",
                IntrinsicEffect::YmmMemory {
                    reads: 64,
                    writes: 32,
                },
            ),
            (
                "vpaddq_i64x4",
                IntrinsicEffect::YmmMemory {
                    reads: 64,
                    writes: 32,
                },
            ),
            ("cpu_features_v1", IntrinsicEffect::HostState),
            ("cpu_relax", IntrinsicEffect::HostState),
            ("atomic_load_acquire_i64", IntrinsicEffect::HostState),
            (
                "atomic_store_release_i64",
                IntrinsicEffect::MemoryWrite { width_bytes: 8 },
            ),
            (
                "group_bank_i64",
                IntrinsicEffect::MemoryWrite { width_bytes: 8 },
            ),
            (
                "group_acc32_i64",
                IntrinsicEffect::MemoryWrite { width_bytes: 8 },
            ),
            ("sys_mmap", IntrinsicEffect::HostState),
            ("sys_munmap", IntrinsicEffect::HostState),
            ("sys_sched_setaffinity", IntrinsicEffect::HostState),
            ("rdtscp_v1", IntrinsicEffect::HostState),
            ("lfence_v1", IntrinsicEffect::HostState),
            ("sys_perf_event_open", IntrinsicEffect::HostState),
        ] {
            assert_eq!(
                intrinsic_contract(name).expect("contract row").effect,
                effect
            );
        }
        for name in ["read", "write"] {
            assert_eq!(
                intrinsic_contract(name)
                    .expect("I/O contract")
                    .offset_policy,
                IntrinsicOffsetPolicy::MustBeMinusOneIgnored
            );
        }
        for name in [
            "alloc",
            "argc",
            "argv",
            "load8",
            "load_i64",
            "open",
            "store8",
            "store_i16",
            "store_i64",
        ] {
            assert_eq!(
                intrinsic_contract(name)
                    .expect("non-I/O contract")
                    .offset_policy,
                IntrinsicOffsetPolicy::Any
            );
        }
        assert!(intrinsic_supports_profile(
            "load_i64",
            IntrinsicProfile::RustMlir
        ));
        assert!(intrinsic_supports_profile(
            "load_i64",
            IntrinsicProfile::FrozenNative
        ));
        assert!(intrinsic_supports_profile(
            "argc",
            IntrinsicProfile::FrozenNative
        ));
        assert!(intrinsic_supports_profile(
            "argc",
            IntrinsicProfile::RustMlir
        ));
        #[cfg(feature = "std-surface")]
        {
            assert_eq!(std_surface_intrinsic_arity("__mind_argc"), Some(0));
            assert_eq!(std_surface_intrinsic_arity("__mind_argv"), Some(1));
            assert_eq!(std_surface_intrinsic_arity("__mind_load_i8"), Some(1));
        }
        // Seed ids 60-67 are native-only: registered (any-profile arity), never
        // Rust/MLIR-selectable, and the CPU-feature read is a World row.
        for (physical, arity) in [
            ("__mind_vpaddd_i32x8", 2),
            ("__mind_vpsubd_i32x8", 2),
            ("__mind_vpaddb_i8x32", 2),
            ("__mind_vpshufb_i8x32", 3),
            ("__mind_vmovdqu_i32x8", 2),
            ("__mind_vpaddq_i64x4", 2),
            ("__mind_vpsubq_i64x4", 2),
            ("__mind_cpu_features_v1", 0),
            ("__mind_group_bank_i64", 5),
            ("__mind_group_acc32_i64", 6),
        ] {
            let spec = STD_SURFACE_INTRINSICS
                .iter()
                .find(|spec| spec.physical_name == physical)
                .expect("native-only row");
            let contract = spec.contract.expect("contract");
            assert_eq!(contract.profiles, FROZEN_NATIVE_ONLY, "{physical}");
            assert_eq!(contract.result, IntrinsicResult::I64, "{physical}");
            assert_eq!(
                crate::intrinsics::registered_intrinsic_arity(physical),
                Some(arity)
            );
            #[cfg(feature = "std-surface")]
            assert_eq!(std_surface_intrinsic_arity(physical), None, "{physical}");
            let world = physical == "__mind_cpu_features_v1";
            assert_eq!(spec.det == Det::World, world, "{physical}");
        }
        assert_eq!(
            crate::intrinsics::registered_intrinsic_arity("__mind_vpadd_i32x8"),
            None
        );

        let read = intrinsic_contract("read").expect("read contract");
        assert_eq!(read.parameters.len(), 4);
        assert_eq!(read.result, IntrinsicResult::I64);
        assert_eq!(read.effect, IntrinsicEffect::FdRead);
        assert_eq!(
            read.offset_policy,
            IntrinsicOffsetPolicy::MustBeMinusOneIgnored
        );
        // A5 shm rows: native-only, raw i64 results, Det::World on the registry row.
        for (name, arity) in [
            ("cpu_relax", 0),
            ("atomic_load_acquire_i64", 1),
            ("atomic_store_release_i64", 2),
            ("sys_mmap", 6),
            ("sys_munmap", 2),
            ("sys_sched_setaffinity", 3),
        ] {
            let contract = intrinsic_contract(name).expect("shm contract");
            assert_eq!(contract.parameters.len(), arity);
            assert_eq!(contract.result, IntrinsicResult::I64);
            assert_eq!(contract.offset_policy, IntrinsicOffsetPolicy::Any);
            assert!(intrinsic_supports_profile(
                name,
                IntrinsicProfile::FrozenNative
            ));
            assert!(!intrinsic_supports_profile(
                name,
                IntrinsicProfile::RustMlir
            ));
            let spec = STD_SURFACE_INTRINSICS
                .iter()
                .find(|spec| spec.contract.is_some_and(|c| c.logical_name == name))
                .expect("shm registry row");
            assert_eq!((spec.arity, spec.det), (arity, Det::World));
            #[cfg(feature = "std-surface")]
            assert_eq!(std_surface_intrinsic_arity(spec.physical_name), None);
        }
        for name in ["store_i64", "store8", "store_i16"] {
            let contract = intrinsic_contract(name).expect("store contract");
            assert_eq!(contract.result, IntrinsicResult::DiscardOnly);
            assert!(matches!(
                contract.effect,
                IntrinsicEffect::MemoryWrite { .. }
            ));
        }
    }
}
