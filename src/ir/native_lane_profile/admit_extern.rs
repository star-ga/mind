// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Typing of `extern "C"` declarations for the native-lane check.
//! Types are the ABI codes of `nbx_code`: 1 i32, 2 u32, 3 i64, 4 *const u8,
//! 5 *mut u8, 6 *mut i32, 7 void (a missing result type), 8 *const *const u8,
//! 9 *mut *mut u8, 10 *mut i64.

use super::types::{Check, Functions, LaneProfileRejection, Signature, Ty};
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
    /// `*const *const u8`.
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

/// Map a source `TypeAnn` onto its ABI width class.
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
            }
            let pointee_u8 = matches!(pointee.as_ref(), TypeAnn::Named(n) if n == "u8");
            let pointee_i32 = matches!(pointee.as_ref(), TypeAnn::ScalarI32);
            let pointee_i64 = matches!(pointee.as_ref(), TypeAnn::ScalarI64);
            match (pointee_u8, pointee_i32, pointee_i64, *mutable) {
                (true, _, _, false) => Ok(AbiTy::PtrConstU8),
                (true, _, _, true) => Ok(AbiTy::PtrMutU8),
                (_, true, _, true) => Ok(AbiTy::PtrMutI32),
                (_, _, true, true) => Ok(AbiTy::PtrMutI64),
                _ => Ok(AbiTy::PtrMutU8),
            }
        }
        _ => Ok(AbiTy::I64),
    }
}

fn named(name: &str, construct: &'static str) -> LaneProfileRejection {
    LaneProfileRejection::new(construct).in_function(name)
}

/// Type one declaration by its ABI classes. Returns its lane signature.
pub(super) fn admit_extern_fn(f: &ExternFn, _callconv: CallConv) -> Check<(String, Signature)> {
    let mut params = Vec::with_capacity(f.params.len());
    for p in &f.params {
        let got = abi_type(&p.ty).map_err(|r| named(&f.name, r.construct))?;
        params.push(got.to_ty());
    }
    let result = match f.ret_type.as_ref() {
        Some(ann) => {
            let got = abi_type(ann).map_err(|r| named(&f.name, r.construct))?;
            got.to_ty()
        }
        None => AbiTy::Void.to_ty(),
    };
    Ok((f.name.clone(), Signature { params, result }))
}

/// Admit every top-level `extern "C"` block of a user module into the
/// extern tier. A name already claimed by a user function is
/// `lane.call.ambiguous`, matching L4's cross-tier rule. A repeat declaration
/// of one extern (std modules each declare what they call) keeps the first
/// signature; whether the declarations of one symbol agree is the compiler's
/// decision.
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

    /// Pointer-to-pointer has exactly two codes: 8 is only
    /// `*const *const u8` and 9 is only `*mut *mut u8` (the out-param and
    /// `void**` shapes of the driver rows). Every other pointer-to-pointer
    /// shape passes through as a pointer word; the compiler's rows judge it.
    #[test]
    fn pointer_to_pointer_is_one_parameter_code_and_passes_the_rest() {
        let pp = ptr(false, ptr(false, TypeAnn::Named("u8".to_string())));
        assert_eq!(abi_type(&pp).unwrap(), AbiTy::PtrConstPtrConstU8);
        let mm = ptr(true, ptr(true, TypeAnn::Named("u8".to_string())));
        assert_eq!(abi_type(&mm).unwrap(), AbiTy::PtrMutPtrMutU8);
        let passed = [
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
        for ann in &passed {
            assert_eq!(abi_type(ann).unwrap().to_ty(), Ty::Ptr, "{ann:?}");
        }
    }

    /// Code 10 is only `*mut i64` (the mutable device-pointer out-param).
    /// `*mut i32` is code 6 (`AbiTy::PtrMutI32`). A const pointee and every
    /// other scalar pointee pass through as a pointer word.
    #[test]
    fn pointer_to_i64_is_one_mutable_parameter_code_and_passes_the_rest() {
        let pm = ptr(true, TypeAnn::ScalarI64);
        assert_eq!(abi_type(&pm).unwrap(), AbiTy::PtrMutI64);
        assert_eq!(AbiTy::PtrMutI64.code(), 10);
        let admitted = ptr(true, TypeAnn::ScalarI32);
        assert_eq!(abi_type(&admitted).unwrap(), AbiTy::PtrMutI32);
        assert_eq!(AbiTy::PtrMutI32.code(), 6);
        for ann in [
            ptr(false, TypeAnn::ScalarI64),
            ptr(true, TypeAnn::ScalarU32),
            ptr(false, TypeAnn::ScalarI32),
            ptr(true, ptr(true, TypeAnn::ScalarI64)),
        ] {
            assert_eq!(abi_type(&ann).unwrap().to_ty(), Ty::Ptr, "{ann:?}");
        }
    }

    /// Every integer width is a parameter and a result width. A float, a bool,
    /// or a type outside the classes passes as the i64 word: the host has no
    /// class for it, and the compiler's rows judge.
    #[test]
    fn every_width_passes_as_param_and_result() {
        assert_eq!(AbiTy::U32.to_ty(), Ty::U32);
        assert_eq!(abi_type(&TypeAnn::ScalarU32).unwrap(), AbiTy::U32);
        assert_eq!(abi_type(&TypeAnn::ScalarI64).unwrap(), AbiTy::I64);
        assert_eq!(abi_type(&TypeAnn::ScalarI32).unwrap(), AbiTy::I32);
        assert_eq!(abi_type(&TypeAnn::ScalarF64).unwrap(), AbiTy::I64);
        assert_eq!(abi_type(&TypeAnn::ScalarF32).unwrap(), AbiTy::I64);
        assert_eq!(abi_type(&TypeAnn::ScalarBool).unwrap(), AbiTy::I64);
        assert_eq!(
            abi_type(&TypeAnn::Named("Foo".to_string())).unwrap(),
            AbiTy::I64
        );
        assert_eq!(
            abi_type(&TypeAnn::Array {
                element: Box::new(TypeAnn::Named("u8".to_string())),
                length: 16,
            })
            .unwrap(),
            AbiTy::I64
        );
    }

    fn param(name: &str, ty: TypeAnn) -> crate::ast::Param {
        crate::ast::Param {
            name: name.to_string(),
            ty,
            span: crate::ast::Span::new(0, 0),
        }
    }

    fn decl(name: &str, params: Vec<TypeAnn>, ret: Option<TypeAnn>) -> ExternFn {
        ExternFn {
            is_unsafe: true,
            name: name.to_string(),
            params: params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| param(&format!("p{i}"), ty))
                .collect(),
            ret_type: ret,
            is_varargs: false,
            span: crate::ast::Span::new(0, 0),
        }
    }

    fn block(fns: Vec<ExternFn>) -> crate::ast::Node {
        crate::ast::Node::ExternBlock {
            callconv: CallConv::C,
            binding: crate::ast::ExternBinding::Base,
            fns,
            span: crate::ast::Span::new(0, 0),
        }
    }

    /// Two declarations that disagree are not judged here: the first is kept,
    /// a call of it passes through, and the compiler's rows name the conflict.
    #[test]
    fn differing_declarations_are_left_to_the_compiler() {
        let items = [block(vec![
            decl("close", vec![TypeAnn::ScalarI64], Some(TypeAnn::ScalarI32)),
            decl("close", vec![TypeAnn::ScalarI32], Some(TypeAnn::ScalarI32)),
        ])];
        let got = collect_externs(&items, &Functions::new()).unwrap();
        let sig = got.get("close").unwrap();
        assert_eq!(sig.params, vec![Ty::I64]);
        assert_eq!(sig.result, Ty::I32);
    }

    /// A repeat with the same types keeps that signature.
    #[test]
    fn an_equal_repeat_keeps_the_signature() {
        let items = [block(vec![
            decl("close", vec![TypeAnn::ScalarI32], Some(TypeAnn::ScalarI32)),
            decl("close", vec![TypeAnn::ScalarI32], Some(TypeAnn::ScalarI32)),
        ])];
        let got = collect_externs(&items, &Functions::new()).unwrap();
        let sig = got.get("close").unwrap();
        assert_eq!(sig.params, vec![Ty::I32]);
        assert_eq!(sig.result, Ty::I32);
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
}
