// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Fixed-arity `extern "C"` admission against the closed `nbx_row` table.
//!
//! The table is the same rows `examples/mindc_mind/main.mind` binds through
//! a static-layout dynamic image. A declaration outside the table, with a
//! different arity or width, a varargs marker, or a callconv other than C/SysV
//! is refused BY NAME so the diagnostic names the symbol that cannot be bound.
//! Types are the ABI codes of `nbx_code`: 1 i32, 2 u32, 3 i64, 4 *const u8,
//! 5 *mut u8, 6 *mut i32, 7 void (a missing result type), 8 *const *const u8,
//! 9 *mut *mut u8, 10 *mut i64.
//! Results refuse u32, *const u8, *const *const u8, *mut *mut u8 and *mut i64,
//! matching `nbx_row_sig`. The libc rows live here; the CUDA driver rows
//! (libcuda.so.1, unversioned) live in `admit_extern_cuda`.

use super::types::{Check, Functions, LaneProfileRejection, Signature, Ty, reject};
use crate::ast::{CallConv, ExternFn, TypeAnn};

/// ABI width class of one declared parameter or result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AbiTy {
    I32,
    U32,
    I64,
    PtrConstU8,
    PtrMutU8,
    PtrMutI32,
    /// A declaration with no result type. Parameter position stays refused.
    Void,
    /// `*const *const u8`. Any other pointer-to-pointer shape stays refused
    /// except the mutable pair below.
    PtrConstPtrConstU8,
    /// `*mut *mut u8`: an out-param slot a driver entry writes a handle into,
    /// and the `void**` kernel-argument array of `cuLaunchKernel`.
    PtrMutPtrMutU8,
    /// `*mut i64`: a mutable device-pointer out-param (`CUdeviceptr *`).
    PtrMutI64,
}

impl AbiTy {
    /// `nbx_code` value: 1 i32, 2 u32, 3 i64, 4 *const u8, 5 *mut u8, 6 *mut i32,
    /// 7 void, 8 *const *const u8, 9 *mut *mut u8, 10 *mut i64.
    #[allow(dead_code)]
    pub(super) fn code(self) -> i64 {
        match self {
            AbiTy::I32 => 1,
            AbiTy::U32 => 2,
            AbiTy::I64 => 3,
            AbiTy::PtrConstU8 => 4,
            AbiTy::PtrMutU8 => 5,
            AbiTy::PtrMutI32 => 6,
            AbiTy::Void => 7,
            AbiTy::PtrConstPtrConstU8 => 8,
            AbiTy::PtrMutPtrMutU8 => 9,
            AbiTy::PtrMutI64 => 10,
        }
    }
    pub(super) fn to_ty(self) -> Ty {
        match self {
            AbiTy::I32 => Ty::I32,
            AbiTy::U32 => Ty::U32,
            AbiTy::I64 => Ty::I64,
            AbiTy::Void => Ty::Unit,
            AbiTy::PtrConstU8
            | AbiTy::PtrMutU8
            | AbiTy::PtrMutI32
            | AbiTy::PtrConstPtrConstU8
            | AbiTy::PtrMutPtrMutU8
            | AbiTy::PtrMutI64 => Ty::Ptr,
        }
    }
}

/// One bindable libc symbol, glibc version, and source signature.
pub(super) struct Row {
    pub(super) name: &'static str,
    pub(super) params: &'static [AbiTy],
    pub(super) result: AbiTy,
    #[allow(dead_code)]
    pub(super) version: &'static str,
}

/// The closed table. Mirrors `nbx_row` in `examples/mindc_mind/main.mind`.
/// Rows after the `// L2b: extern table rows` anchor use the same width codes.
pub(super) const TABLE: &[Row] = &[
    Row {
        name: "close",
        params: &[AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "fstat",
        params: &[AbiTy::I32, AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "GLIBC_2.33",
    },
    Row {
        name: "getuid",
        params: &[],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "__errno_location",
        params: &[],
        result: AbiTy::PtrMutI32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "clock_gettime",
        params: &[AbiTy::I32, AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "GLIBC_2.17",
    },
    Row {
        name: "close_range",
        params: &[AbiTy::U32, AbiTy::U32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.34",
    },
    Row {
        name: "closedir",
        params: &[AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "dup2",
        params: &[AbiTy::I32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "fchdir",
        params: &[AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "fdopendir",
        params: &[AbiTy::I32],
        result: AbiTy::PtrMutU8,
        version: "GLIBC_2.4",
    },
    Row {
        name: "flock",
        params: &[AbiTy::I32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "fork",
        params: &[],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "fstatat",
        params: &[AbiTy::I32, AbiTy::PtrConstU8, AbiTy::PtrMutU8, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.33",
    },
    Row {
        name: "fsync",
        params: &[AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "getpgid",
        params: &[AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "getrlimit",
        params: &[AbiTy::I32, AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "kill",
        params: &[AbiTy::I32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "lseek",
        params: &[AbiTy::I32, AbiTy::I64, AbiTy::I32],
        result: AbiTy::I64,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "memfd_create",
        params: &[AbiTy::PtrConstU8, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.27",
    },
    Row {
        name: "nanosleep",
        params: &[AbiTy::PtrConstU8, AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "pipe2",
        params: &[AbiTy::PtrMutI32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.9",
    },
    Row {
        name: "readdir",
        params: &[AbiTy::PtrMutU8],
        result: AbiTy::PtrMutU8,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "renameat",
        params: &[AbiTy::I32, AbiTy::PtrConstU8, AbiTy::I32, AbiTy::PtrConstU8],
        result: AbiTy::I32,
        version: "GLIBC_2.4",
    },
    Row {
        name: "setpgid",
        params: &[AbiTy::I32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "setrlimit",
        params: &[AbiTy::I32, AbiTy::PtrConstU8],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "signal",
        params: &[AbiTy::I32, AbiTy::PtrConstU8],
        result: AbiTy::PtrMutU8,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "umask",
        params: &[AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "unlinkat",
        params: &[AbiTy::I32, AbiTy::PtrConstU8, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.4",
    },
    Row {
        name: "waitid",
        params: &[AbiTy::I32, AbiTy::I32, AbiTy::PtrMutU8, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "waitpid",
        params: &[AbiTy::I32, AbiTy::PtrMutI32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "_exit",
        params: &[AbiTy::I32],
        result: AbiTy::Void,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "fexecve",
        params: &[
            AbiTy::I32,
            AbiTy::PtrConstPtrConstU8,
            AbiTy::PtrConstPtrConstU8,
        ],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    // Socket, address, fill and entropy rows. `socklen_t` and the getrandom
    // flags are spelled `i32`, as std's net, dns and entropy blocks declare them.
    Row {
        name: "socket",
        params: &[AbiTy::I32, AbiTy::I32, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "connect",
        params: &[AbiTy::I32, AbiTy::PtrConstU8, AbiTy::I32],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "send",
        params: &[AbiTy::I32, AbiTy::PtrConstU8, AbiTy::I64, AbiTy::I32],
        result: AbiTy::I64,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "recv",
        params: &[AbiTy::I32, AbiTy::PtrMutU8, AbiTy::I64, AbiTy::I32],
        result: AbiTy::I64,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "inet_pton",
        params: &[AbiTy::I32, AbiTy::PtrConstU8, AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "memset",
        params: &[AbiTy::PtrMutU8, AbiTy::I32, AbiTy::I64],
        result: AbiTy::PtrMutU8,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "setsockopt",
        params: &[
            AbiTy::I32,
            AbiTy::I32,
            AbiTy::I32,
            AbiTy::PtrConstU8,
            AbiTy::I32,
        ],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "getaddrinfo",
        params: &[
            AbiTy::PtrConstU8,
            AbiTy::PtrConstU8,
            AbiTy::PtrConstU8,
            AbiTy::PtrMutU8,
        ],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "getrandom",
        params: &[AbiTy::PtrMutU8, AbiTy::I64, AbiTy::I32],
        result: AbiTy::I64,
        version: "GLIBC_2.25",
    },
];

/// libc rows past the CUDA block: `nbx_row_part` indices 87.. in
/// `examples/mindc_mind/nb_part_dispatch.mind` (std.dns needs both).
pub(super) const TABLE_TAIL: &[Row] = &[
    Row {
        name: "freeaddrinfo",
        params: &[AbiTy::PtrMutU8],
        result: AbiTy::Void,
        version: "GLIBC_2.2.5",
    },
    Row {
        name: "uname",
        params: &[AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "GLIBC_2.2.5",
    },
];

/// Map a source `TypeAnn` onto the ABI width class of the table.
pub(super) fn abi_type(ann: &TypeAnn) -> Check<AbiTy> {
    match ann {
        TypeAnn::ScalarI32 => Ok(AbiTy::I32),
        TypeAnn::ScalarU32 => Ok(AbiTy::U32),
        TypeAnn::ScalarI64 => Ok(AbiTy::I64),
        TypeAnn::RawPtr { mutable, pointee } => {
            if let TypeAnn::RawPtr {
                mutable: inner_mut,
                pointee: inner,
            } = pointee.as_ref()
            {
                let inner_u8 = matches!(inner.as_ref(), TypeAnn::Named(n) if n == "u8");
                if !*mutable && !*inner_mut && inner_u8 {
                    return Ok(AbiTy::PtrConstPtrConstU8);
                }
                if *mutable && *inner_mut && inner_u8 {
                    return Ok(AbiTy::PtrMutPtrMutU8);
                }
                return reject("lane.extern.type");
            }
            let pointee_u8 = matches!(pointee.as_ref(), TypeAnn::Named(n) if n == "u8");
            let pointee_i32 = matches!(pointee.as_ref(), TypeAnn::ScalarI32);
            let pointee_i64 = matches!(pointee.as_ref(), TypeAnn::ScalarI64);
            match (pointee_u8, pointee_i32, pointee_i64, *mutable) {
                (true, _, _, false) => Ok(AbiTy::PtrConstU8),
                (true, _, _, true) => Ok(AbiTy::PtrMutU8),
                (_, true, _, true) => Ok(AbiTy::PtrMutI32),
                (_, _, true, true) => Ok(AbiTy::PtrMutI64),
                _ => reject("lane.extern.type"),
            }
        }
        _ => reject("lane.extern.type"),
    }
}

/// Result widths refuse u32, *const u8, *const *const u8, *mut *mut u8 and
/// *mut i64, matching `nbx_row_sig`. Void is the absence of a result type,
/// not a written type.
fn result_ty(ann: &TypeAnn) -> Check<AbiTy> {
    let ty = abi_type(ann)?;
    match ty {
        AbiTy::U32
        | AbiTy::PtrConstU8
        | AbiTy::PtrConstPtrConstU8
        | AbiTy::PtrMutPtrMutU8
        | AbiTy::PtrMutI64 => reject("lane.extern.result_type"),
        _ => Ok(ty),
    }
}

fn named(name: &str, construct: &'static str) -> LaneProfileRejection {
    LaneProfileRejection::new(construct).in_function(name)
}

/// Validate one declaration against the table. Returns its lane signature.
/// The arity cap is 11, the widest row (`nbx_row_sig` and `nbx_row_sig_hi`
/// refuse anything above it); SysV arguments 6..10 travel on the stack.
pub(super) fn admit_extern_fn(f: &ExternFn, callconv: CallConv) -> Check<(String, Signature)> {
    if !matches!(callconv, CallConv::C | CallConv::SysV) {
        return Err(named(&f.name, "lane.extern.callconv"));
    }
    if f.is_varargs {
        return Err(named(&f.name, "lane.extern.varargs"));
    }
    if f.params.len() > 11 {
        return Err(named(&f.name, "lane.extern.arity"));
    }
    let row = TABLE
        .iter()
        .chain(super::admit_extern_cuda::ROWS.iter())
        .chain(TABLE_TAIL.iter())
        .find(|r| r.name == f.name)
        .ok_or_else(|| named(&f.name, "lane.extern.unknown_symbol"))?;
    if f.params.len() != row.params.len() {
        return Err(named(&f.name, "lane.extern.arity"));
    }
    let mut params = Vec::with_capacity(row.params.len());
    for (p, want) in f.params.iter().zip(row.params) {
        let got = abi_type(&p.ty).map_err(|r| named(&f.name, r.construct))?;
        if got != *want {
            return Err(named(&f.name, "lane.extern.signature_mismatch"));
        }
        params.push(got.to_ty());
    }
    let result = match f.ret_type.as_ref() {
        Some(ann) => {
            let got = result_ty(ann).map_err(|r| named(&f.name, r.construct))?;
            if got != row.result {
                return Err(named(&f.name, "lane.extern.signature_mismatch"));
            }
            got.to_ty()
        }
        None => {
            if row.result != AbiTy::Void {
                return Err(named(&f.name, "lane.extern.signature_mismatch"));
            }
            AbiTy::Void.to_ty()
        }
    };
    Ok((f.name.clone(), Signature { params, result }))
}

/// Admit every top-level `extern "C"` block of a user module into the
/// extern tier. A name already claimed by a user function or by another
/// extern is `lane.call.ambiguous`, matching L4's cross-tier rule. A repeat
/// declaration of an admitted extern (std modules each declare what they call)
/// is the same symbol: both passed `admit_extern_fn` against the one row, so
/// their signatures are equal; a different signature refused there already.
pub(super) fn collect_externs(items: &[crate::ast::Node], user: &Functions) -> Check<Functions> {
    let mut externs = Functions::new();
    for item in items {
        let (callconv, fns) = match item {
            crate::ast::Node::ExternBlock { callconv, fns, .. } => (*callconv, fns),
            _ => continue,
        };
        for f in fns {
            let (name, sig) = admit_extern_fn(f, callconv)?;
            if user.contains_key(&name) {
                return Err(named(&name, "lane.call.ambiguous"));
            }
            externs.entry(name).or_insert(sig);
        }
    }
    Ok(externs)
}

#[cfg(test)]
mod abi_ty_tests {
    use super::super::types::word;
    use super::*;

    /// The ABI codes 1..=10 of `nbx_code`, and the Ty each source width lowers to.
    #[test]
    fn abi_ty_codes_and_width_classes_are_the_table() {
        assert_eq!(AbiTy::I32.code(), 1);
        assert_eq!(AbiTy::U32.code(), 2);
        assert_eq!(AbiTy::I64.code(), 3);
        assert_eq!(AbiTy::PtrConstU8.code(), 4);
        assert_eq!(AbiTy::PtrMutU8.code(), 5);
        assert_eq!(AbiTy::PtrMutI32.code(), 6);
        assert_eq!(AbiTy::Void.code(), 7);
        assert_eq!(AbiTy::PtrConstPtrConstU8.code(), 8);
        assert_eq!(AbiTy::PtrMutPtrMutU8.code(), 9);
        assert_eq!(AbiTy::PtrMutI64.code(), 10);
        assert_eq!(AbiTy::I32.to_ty(), Ty::I32);
        assert_eq!(AbiTy::U32.to_ty(), Ty::U32);
        assert_eq!(AbiTy::I64.to_ty(), Ty::I64);
        assert_eq!(AbiTy::Void.to_ty(), Ty::Unit);
        assert_eq!(AbiTy::PtrConstU8.to_ty(), Ty::Ptr);
        assert_eq!(AbiTy::PtrMutU8.to_ty(), Ty::Ptr);
        assert_eq!(AbiTy::PtrMutI32.to_ty(), Ty::Ptr);
        assert_eq!(AbiTy::PtrConstPtrConstU8.to_ty(), Ty::Ptr);
        assert_eq!(AbiTy::PtrMutPtrMutU8.to_ty(), Ty::Ptr);
        assert_eq!(AbiTy::PtrMutI64.to_ty(), Ty::Ptr);
    }

    fn ptr(mutable: bool, pointee: TypeAnn) -> TypeAnn {
        TypeAnn::RawPtr {
            mutable,
            pointee: Box::new(pointee),
        }
    }

    /// Pointer-to-pointer has exactly two parameter codes: 8 is only
    /// `*const *const u8` and 9 is only `*mut *mut u8` (the out-param and
    /// `void**` shapes of the driver rows). Every other pointer-to-pointer
    /// shape, and every pointer-to-pointer result, stays out of range.
    #[test]
    fn pointer_to_pointer_is_one_parameter_code_and_refuses_the_rest() {
        let pp = ptr(false, ptr(false, TypeAnn::Named("u8".to_string())));
        assert_eq!(abi_type(&pp).unwrap(), AbiTy::PtrConstPtrConstU8);
        assert!(result_ty(&pp).is_err());
        let mm = ptr(true, ptr(true, TypeAnn::Named("u8".to_string())));
        assert_eq!(abi_type(&mm).unwrap(), AbiTy::PtrMutPtrMutU8);
        assert!(result_ty(&mm).is_err());
        let refused = [
            ptr(true, ptr(false, TypeAnn::Named("u8".to_string()))),
            ptr(false, ptr(true, TypeAnn::Named("u8".to_string()))),
            ptr(true, ptr(true, TypeAnn::ScalarI32)),
            ptr(false, ptr(false, TypeAnn::ScalarI32)),
            ptr(false, ptr(false, TypeAnn::ScalarI64)),
            ptr(
                false,
                ptr(false, ptr(false, TypeAnn::Named("u8".to_string()))),
            ),
        ];
        for ann in &refused {
            assert!(abi_type(ann).is_err(), "{ann:?}");
        }
    }

    /// Code 10 is only `*mut i64` (the mutable device-pointer out-param).
    /// `*mut i32` is code 6 (`AbiTy::PtrMutI32`), an admitted parameter, so it
    /// is not in the refused list. A const pointee, every other scalar pointee
    /// and a result position stay out of range.
    #[test]
    fn pointer_to_i64_is_one_mutable_parameter_code_and_refuses_the_rest() {
        let pm = ptr(true, TypeAnn::ScalarI64);
        assert_eq!(abi_type(&pm).unwrap(), AbiTy::PtrMutI64);
        assert_eq!(AbiTy::PtrMutI64.code(), 10);
        assert!(result_ty(&pm).is_err());
        let admitted = ptr(true, TypeAnn::ScalarI32);
        assert_eq!(abi_type(&admitted).unwrap(), AbiTy::PtrMutI32);
        assert_eq!(AbiTy::PtrMutI32.code(), 6);
        for ann in [
            ptr(false, TypeAnn::ScalarI64),
            ptr(true, TypeAnn::ScalarU32),
            ptr(false, TypeAnn::ScalarI32),
            ptr(true, ptr(true, TypeAnn::ScalarI64)),
        ] {
            assert!(abi_type(&ann).is_err(), "{ann:?}");
        }
    }

    /// u32 is a parameter width, never a result width; i64 is both.
    #[test]
    fn u32_is_param_only_and_i64_is_both() {
        assert_eq!(AbiTy::U32.to_ty(), Ty::U32);
        assert!(result_ty(&TypeAnn::ScalarU32).is_err());
        assert!(result_ty(&TypeAnn::ScalarI64).is_ok());
        assert!(result_ty(&TypeAnn::ScalarI32).is_ok());
    }

    /// Word erasure: i32/u32/ptr all travel in 64-bit IR slots.
    #[test]
    fn word_erasure_maps_narrow_and_ptr_to_i64() {
        assert_eq!(word(Ty::I32), Ty::I64);
        assert_eq!(word(Ty::U32), Ty::I64);
        assert_eq!(word(Ty::Ptr), Ty::I64);
        assert_eq!(word(Ty::I64), Ty::I64);
        assert_eq!(word(Ty::F64), Ty::F64);
    }

    /// `nbx_row` literals and this table name the same symbol, version and
    /// width at each index: the 41 glibc rows here, then the driver rows of
    /// `admit_extern_cuda` (13 in `nbx_row`, the rest in the `nbcu_row` part,
    /// a name over 24 bytes as six `nbcu_long` words), both signature words
    /// included.
    #[test]
    fn rust_table_matches_nbx_row_literals() {
        /// Length, then `n` little-endian 8-byte words of the name.
        fn pack(s: &str, n: usize) -> Vec<i64> {
            let b = s.as_bytes();
            let mut out = vec![b.len() as i64];
            for k in 0..n {
                let mut w = 0i64;
                for (i, c) in b.iter().skip(8 * k).take(8).enumerate() {
                    w |= i64::from(*c) << (8 * i);
                }
                out.push(w);
            }
            out
        }
        fn signature(row: &Row) -> i64 {
            let mut packed = row.result.code();
            let mut shift = 256i64;
            for param in row.params.iter().take(6) {
                packed += param.code() * shift;
                shift *= 256;
            }
            packed
        }
        fn signature_hi(row: &Row) -> i64 {
            let mut packed = 0i64;
            let mut shift = 256i64;
            for param in row.params.iter().skip(6) {
                packed += param.code() * shift;
                shift *= 256;
            }
            packed
        }
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/mindc_mind/");
        let text = std::fs::read_to_string(format!("{dir}main.mind")).unwrap();
        let start = text.find("pub fn nbx_row(").expect("nbx_row");
        let body = &text[start..];
        let end = body.find("\npub fn nbx_code(").expect("nbx_code");
        let part = std::fs::read_to_string(format!("{dir}nbx_cuda_rows2.mind")).unwrap();
        let part = &part[part.find("pub fn nbcu_row(").expect("nbcu_row")..];
        let part = &part[..part.find("\n}\n").expect("nbcu_row end")];
        // libc rows past the CUDA block (TABLE_TAIL): nbx_row_part, 87..
        let tail = std::fs::read_to_string(format!("{dir}nb_part_dispatch.mind")).unwrap();
        let tail = &tail[tail.find("pub fn nbx_row_part(").expect("nbx_row_part")..];
        let tail = &tail[..tail.find("\n}\n").expect("nbx_row_part end")];
        let part = format!("{part}\n{tail}");
        let part = part.as_str();
        let mut got = Vec::new();
        let mut lines = body[..end].lines().chain(part.lines()).map(str::trim);
        while let Some(head) = lines.next() {
            if !head.starts_with("if i == ") {
                continue;
            }
            // `mindc fmt` puts the writer call on the line after `if i == N {`.
            let line = if head.ends_with('{') {
                format!("{head} {}", lines.next().unwrap_or(""))
            } else {
                head.to_string()
            };
            let idx: i64 = line
                .split("==")
                .nth(1)
                .unwrap()
                .split('{')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            // The part writes a name over 24 bytes with nbcu_long (six name
            // words), every other row with nbx_put.
            let call = ["nbcu_long(", "nbx_put("]
                .into_iter()
                .find(|c| line.contains(&format!("{c}out, ")))
                .unwrap_or_else(|| panic!("no record writer: {line}"));
            let inside = line.split(call).nth(1).unwrap();
            let inside = inside.split(')').next().unwrap();
            let mut rec = vec![idx];
            rec.extend(
                inside
                    .split(',')
                    .skip(1)
                    .map(|s| s.trim().parse::<i64>().unwrap()),
            );
            got.push(rec);
        }
        let all: Vec<&Row> = TABLE
            .iter()
            .chain(super::super::admit_extern_cuda::ROWS.iter())
            .chain(TABLE_TAIL.iter())
            .collect();
        assert_eq!(got.len(), all.len());
        for (i, row) in all.iter().enumerate() {
            let mut want = vec![i as i64];
            if row.name.len() > 24 {
                assert!(
                    row.name.len() <= 48 && row.version.is_empty(),
                    "{}",
                    row.name
                );
                want.extend(pack(row.name, 6));
            } else {
                want.extend(pack(row.name, 3));
                let tail = i >= TABLE.len() + super::super::admit_extern_cuda::ROWS.len();
                let unversioned = i < 54 || tail || row.version.is_empty();
                assert!(unversioned, "{} is a part row", row.name);
                want.extend(pack(row.version, 2));
            }
            want.extend([signature(row), signature_hi(row), row.params.len() as i64]);
            assert_eq!(got[i], want, "{}", row.name);
        }
    }
}
