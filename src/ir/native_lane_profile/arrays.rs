// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Fixed i64 array reads: `let a = [1, 2, 3];` (or `let a: [i64; 3] = ...`)
//! and `a[i]` with an i64 index. An element is an integer constant expression
//! (`[0 - 1, 2 * 3]`), folded once by `crate::opt::fold::const_int`. The
//! marked compiler emits the reads, the runtime bounds trap (exit 77) and the
//! constant-index refusal; these checks only admit that shape. Every other use of an array (an element store, the
//! array as an operand, argument, return value or assignment, a non-constant
//! element, an array-typed parameter) refuses with a named kind.

use super::types::{Check, LaneProfileRejection, Ty, equal, reject};
use crate::ast::{Literal, Node, TypeAnn};
use crate::ir::{ArrayType, ValueId};
use crate::types::{DType, ScalarType, SemanticType};
use std::collections::{BTreeMap, BTreeSet};

/// Env key of an array binding. `[]` cannot begin an identifier, so the key
/// never collides with a scalar binding, and it scopes with the env.
fn key(name: &str) -> String {
    format!("[]{name}")
}

/// An element the compiler bakes as a constant: an integer constant
/// expression that folds by the spec 3.2 rule (`crate::opt::fold::const_int`).
/// A division by zero or an overflow does not fold and refuses.
fn int_literal(node: &Node) -> bool {
    crate::opt::fold::const_int(node).is_some()
}

/// `let name [: [i64; N]] = [lit, ...];` binds `name` as an i64 array and
/// hides a scalar of the same name.
pub(super) fn bind(
    name: &str,
    ann: Option<&TypeAnn>,
    elements: &[Node],
    env: &mut BTreeMap<String, Ty>,
) -> Check<()> {
    if elements.is_empty() || !elements.iter().all(int_literal) {
        return reject("lane.source.array_element");
    }
    match ann {
        None => {}
        Some(TypeAnn::Array { element, length })
            if matches!(element.as_ref(), TypeAnn::ScalarI64)
                && usize::try_from(*length).ok() == Some(elements.len()) => {}
        // A tensor annotation names the tensor construct, not an array mismatch.
        Some(
            TypeAnn::Tensor { .. } | TypeAnn::DiffTensor { .. } | TypeAnn::SparseTensor { .. },
        ) => {
            return reject("lane.type.tensor");
        }
        Some(_) => return reject("lane.type.array"),
    }
    env.remove(name);
    env.insert(key(name), Ty::I64);
    Ok(())
}

/// A scalar binding named `name` hides an outer array of that name.
pub(super) fn shadow(name: &str, env: &mut BTreeMap<String, Ty>) {
    env.remove(&key(name));
}

/// The refusal for a name with no scalar binding: an array used as a value,
/// or an unbound name.
pub(super) fn not_scalar(name: &str, env: &BTreeMap<String, Ty>) -> LaneProfileRejection {
    if env.contains_key(&key(name)) {
        LaneProfileRejection::new("lane.source.array_value")
    } else {
        LaneProfileRejection::new("lane.source.unbound")
    }
}

/// `base[index]`: an element read from an array binding, typed i64.
pub(super) fn read(base: &Node, index: Ty, env: &BTreeMap<String, Ty>) -> Check<Ty> {
    match base {
        Node::Lit(Literal::Ident(name), _)
            if !env.contains_key(name) && env.contains_key(&key(name)) =>
        {
            equal(index, Ty::I64)?;
            Ok(Ty::I64)
        }
        _ => reject("lane.source.index"),
    }
}

/// Aggregate metadata may describe only the i64 arrays the walk admitted.
pub(super) fn metadata(
    arrays: &BTreeSet<ValueId>,
    value_types: &BTreeMap<ValueId, ArrayType>,
) -> Check<()> {
    for (id, ty) in value_types {
        if !arrays.contains(id) || ty.elem_dtype != DType::I64 {
            return reject("lane.ir.aggregate_metadata");
        }
    }
    Ok(())
}

/// A canonical semantic type naming an admitted i64 array.
pub(super) fn semantic(arrays: &BTreeSet<ValueId>, id: &ValueId, ty: &SemanticType) -> bool {
    matches!(ty, SemanticType::FixedArray { element, .. }
        if matches!(element.as_ref(), SemanticType::Scalar(ScalarType::I64)))
        && arrays.contains(id)
}
