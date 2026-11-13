// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

//! Focused registry-backed intrinsic contract controls.

use super::canonical_types::*;
use crate::ir::{IRModule, Instr, ValueId, verify_canonical_metadata};

fn empty_bundle() -> CanonicalModuleTypes {
    CanonicalModuleTypes::new(
        crate::types::SchemaRegistryBuilder::default()
            .finish()
            .expect("empty schema registry"),
    )
}

fn intrinsic(name: &str, kind: FunctionKind, signature: FunctionSignature) {
    let mut bundle = empty_bundle();
    let result = bundle.add_declaration(FunctionDeclaration::new(
        FunctionIdentity::new("__mind_intrinsic", name),
        kind,
        signature,
    ));
    assert!(
        result.is_ok(),
        "expected valid intrinsic {name}: {result:?}"
    );
}

fn i64s(count: usize) -> Vec<SemanticType> {
    (0..count)
        .map(|_| SemanticType::Scalar(ScalarType::I64))
        .collect()
}

#[test]
fn frozen_native_contracts_accept_exact_value_and_discard_signatures() {
    for (name, arity, result) in [
        ("argc", 0, Some(SemanticType::Scalar(ScalarType::I64))),
        ("argv", 1, Some(SemanticType::Scalar(ScalarType::I64))),
        ("open", 1, Some(SemanticType::Scalar(ScalarType::I64))),
        ("alloc", 1, Some(SemanticType::Scalar(ScalarType::I64))),
        ("load_i64", 1, Some(SemanticType::Scalar(ScalarType::I64))),
        ("load8", 1, Some(SemanticType::Scalar(ScalarType::I64))),
        ("read", 4, Some(SemanticType::Scalar(ScalarType::I64))),
        ("write", 4, Some(SemanticType::Scalar(ScalarType::I64))),
    ] {
        intrinsic(
            name,
            FunctionKind::Intrinsic,
            FunctionSignature::new(i64s(arity), result),
        );
    }
    for name in ["store_i64", "store8", "store_i16"] {
        intrinsic(
            name,
            FunctionKind::Intrinsic,
            FunctionSignature::new(i64s(2), Some(SemanticType::Scalar(ScalarType::I64))),
        );
    }
}

#[test]
fn fabricated_or_malformed_intrinsic_declarations_refuse_closed() {
    let cases = [
        ("evil", 0, None),
        (
            "__mind_load_i64",
            1,
            Some(SemanticType::Scalar(ScalarType::I64)),
        ),
        ("load_i64", 0, Some(SemanticType::Scalar(ScalarType::I64))),
        ("load_i64", 1, Some(SemanticType::Scalar(ScalarType::I32))),
        ("store_i64", 2, Some(SemanticType::Scalar(ScalarType::I32))),
    ];
    for (name, arity, result) in cases {
        let mut bundle = empty_bundle();
        let error = bundle
            .add_declaration(FunctionDeclaration::new(
                FunctionIdentity::new("__mind_intrinsic", name),
                FunctionKind::Intrinsic,
                FunctionSignature::new(i64s(arity), result),
            ))
            .expect_err("malformed intrinsic was admitted");
        assert!(
            matches!(
                error,
                SchemaError::UnknownIntrinsic { .. }
                    | SchemaError::IntrinsicSignatureMismatch { .. }
            ),
            "unexpected malformed intrinsic error: {error:?}"
        );
    }

    let mut bundle = empty_bundle();
    let error = bundle
        .add_declaration(FunctionDeclaration::new(
            FunctionIdentity::new("__mind_intrinsic", "load_i64"),
            FunctionKind::Local,
            FunctionSignature::new(i64s(1), Some(SemanticType::Scalar(ScalarType::I64))),
        ))
        .expect_err("reserved intrinsic owner admitted as a local");
    assert!(matches!(
        error,
        SchemaError::InvalidIntrinsicIdentity { .. }
    ));
}

#[test]
fn native_contract_profile_is_explicit_and_rust_mlir_is_refused() {
    // A native-only row: the native profile admits it, Rust/MLIR refuses it.
    let identity = FunctionIdentity::new("__mind_intrinsic", "cpu_relax");
    assert!(
        validate_intrinsic_profile(&identity, crate::intrinsics::IntrinsicProfile::FrozenNative)
            .is_ok()
    );
    let error =
        validate_intrinsic_profile(&identity, crate::intrinsics::IntrinsicProfile::RustMlir)
            .expect_err("unimplemented Rust/MLIR profile admitted");
    assert!(matches!(
        error,
        SchemaError::IntrinsicProfileMismatch { .. }
    ));

    // argc/argv carry both profiles: the MLIR path calls the runtime pair.
    for name in ["argc", "argv"] {
        let identity = FunctionIdentity::new("__mind_intrinsic", name);
        for profile in [
            crate::intrinsics::IntrinsicProfile::FrozenNative,
            crate::intrinsics::IntrinsicProfile::RustMlir,
        ] {
            assert!(
                validate_intrinsic_profile(&identity, profile).is_ok(),
                "{name}"
            );
        }
    }

    let generic = FunctionIdentity::new("__mind_intrinsic", "load_i64").with_type_args(["T"]);
    let error =
        validate_intrinsic_profile(&generic, crate::intrinsics::IntrinsicProfile::FrozenNative)
            .expect_err("generic intrinsic identity granted native capability");
    assert!(matches!(
        error,
        SchemaError::UnsupportedFunctionTypeArguments { .. }
    ));
}

#[test]
fn canonical_verifier_accepts_only_an_exact_intrinsic_contract() {
    let mut bundle = empty_bundle();
    let intrinsic = FunctionIdentity::new("__mind_intrinsic", "load_i64");
    let i64_ty = SemanticType::Scalar(ScalarType::I64);
    bundle
        .add_declaration(FunctionDeclaration::new(
            intrinsic.clone(),
            FunctionKind::Intrinsic,
            FunctionSignature::new(vec![i64_ty.clone()], Some(i64_ty.clone())),
        ))
        .expect("reserved intrinsic declaration");
    for value in [ValueId(1), ValueId(0)] {
        bundle
            .set_module_value_type(value, i64_ty.clone())
            .expect("intrinsic value type");
    }
    let mut module = IRModule::new();
    module.instrs.push(Instr::ConstI64(ValueId(1), 1));
    module.instrs.push(Instr::Call {
        dst: ValueId(0),
        name: "load_i64".to_string(),
        args: vec![ValueId(1)],
        resolved_callee: Some(Box::new(intrinsic)),
    });
    module.canonical_types = Some(Box::new(bundle));
    assert_eq!(verify_canonical_metadata(&module), Ok(()));
}
