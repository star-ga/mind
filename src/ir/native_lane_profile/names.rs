// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Construct names for the lane's catch-all refusals. A refusal names what it
//! refused (`lane.type.tensor`, `lane.source.trait`), so a diagnostic says which
//! construct is outside the profile instead of only "unsupported".

use super::types::{Check, LaneProfileRejection, Ty, scalar};
use crate::ast::{Node, TypeAnn};
use crate::ir::Instr;

/// [`scalar`], with its generic type refusal replaced by the annotation's name.
pub(super) fn scalar_named(ann: &TypeAnn) -> Check<Ty> {
    scalar(ann).map_err(|r| match r.construct {
        "lane.type.unsupported" => LaneProfileRejection::new(type_ann(ann)),
        _ => r,
    })
}

/// Integer spellings other than `i64`: the lane computes in full-width i64 only.
const INT_WIDTHS: &[&str] = &[
    "i8", "u8", "i16", "u16", "i32", "u32", "u64", "isize", "usize",
];

fn type_ann(ann: &TypeAnn) -> &'static str {
    match ann {
        TypeAnn::Tensor { .. } | TypeAnn::DiffTensor { .. } | TypeAnn::SparseTensor { .. } => {
            "lane.type.tensor"
        }
        TypeAnn::Array { .. } => "lane.type.array",
        TypeAnn::Slice { .. } => "lane.type.slice",
        TypeAnn::Tuple { .. } => "lane.type.tuple",
        TypeAnn::Ref { .. } => "lane.type.reference",
        TypeAnn::RawPtr { .. } => "lane.type.raw_pointer",
        TypeAnn::FnPtr { .. } => "lane.type.fn_pointer",
        TypeAnn::Generic { .. } => "lane.type.generic",
        TypeAnn::Named(name) if INT_WIDTHS.contains(&name.as_str()) => "lane.type.int_width",
        TypeAnn::Named(_) => "lane.type.named",
        TypeAnn::ScalarI32 | TypeAnn::ScalarU32 => "lane.type.int_width",
        TypeAnn::ScalarI64 | TypeAnn::ScalarBool | TypeAnn::ScalarF32 | TypeAnn::ScalarF64 => {
            "lane.type.unsupported"
        }
    }
}

/// The name of a source construct the lane does not admit.
pub(super) fn source(node: &Node) -> &'static str {
    match node {
        Node::CallGrad { .. }
        | Node::CallTensorSum { .. }
        | Node::CallTensorMean { .. }
        | Node::CallReshape { .. }
        | Node::CallExpandDims { .. }
        | Node::CallSqueeze { .. }
        | Node::CallTranspose { .. }
        | Node::CallIndex { .. }
        | Node::CallSlice { .. }
        | Node::CallSliceStride { .. }
        | Node::CallGather { .. }
        | Node::CallDot { .. }
        | Node::CallMatMul { .. }
        | Node::TensorMatmul { .. }
        | Node::TensorElemwise { .. }
        | Node::CallTensorRelu { .. }
        | Node::CallTensorRand { .. }
        | Node::CallTensorConv2d { .. } => "lane.source.tensor",
        Node::Tuple { .. } | Node::LetTuple { .. } => "lane.source.tuple",
        Node::ArrayLit { .. } => "lane.source.array_literal",
        Node::IndexAccess { .. } | Node::SliceRange { .. } => "lane.source.index",
        Node::IndexAssign { .. } => "lane.source.index_store",
        Node::MapLit { .. } => "lane.source.map",
        Node::SetLit { .. } => "lane.source.set",
        Node::ForEach { .. } => "lane.source.for_each",
        Node::MethodCall { .. } => "lane.source.method_call",
        Node::StructDef { .. }
        | Node::StructLit { .. }
        | Node::FieldAccess { .. }
        | Node::FieldAssign { .. } => "lane.source.struct",
        Node::EnumDef { .. } => "lane.source.enum",
        Node::Match { .. } => "lane.source.match",
        Node::TraitDef { .. } | Node::ImplBlock { .. } => "lane.source.trait",
        Node::Closure(..) => "lane.source.closure",
        Node::Ref { .. } => "lane.source.reference",
        Node::Region { .. } => "lane.source.region",
        Node::Try { .. } => "lane.source.try",
        Node::Assert { .. } => "lane.source.assert",
        Node::TypeAlias { .. } => "lane.source.type_alias",
        Node::ExternConst { .. } => "lane.source.extern_const",
        Node::Const { .. } | Node::Export { .. } | Node::Import { .. } => "lane.source.nested_item",
        _ => "lane.source.unsupported",
    }
}

/// The name of an IR instruction the lane does not admit.
pub(super) fn instr(instr: &Instr) -> &'static str {
    match instr {
        Instr::ConstArray { .. } | Instr::ArrayLoad { .. } => "lane.ir.array",
        Instr::ArrayStore { .. } => "lane.ir.array_store",
        Instr::Region { .. } => "lane.ir.region",
        Instr::SparseAttr { .. } => "lane.ir.sparse_attr",
        Instr::VecLoad { .. }
        | Instr::VecFma { .. }
        | Instr::VecReduceAdd { .. }
        | Instr::VecStore { .. }
        | Instr::VecLoadI32 { .. }
        | Instr::VecMulAddQ16 { .. }
        | Instr::VecReduceAddI64 { .. } => "lane.ir.simd",
        Instr::ConstTensor(..)
        | Instr::ConstDenseTensor { .. }
        | Instr::Sum { .. }
        | Instr::Mean { .. }
        | Instr::Relu { .. }
        | Instr::ReluGrad { .. }
        | Instr::Reshape { .. }
        | Instr::ExpandDims { .. }
        | Instr::Squeeze { .. }
        | Instr::Transpose { .. }
        | Instr::Dot { .. }
        | Instr::MatMul { .. }
        | Instr::Conv2d { .. }
        | Instr::Conv2dGradInput { .. }
        | Instr::Conv2dGradFilter { .. }
        | Instr::Index { .. }
        | Instr::Slice { .. }
        | Instr::Gather { .. } => "lane.ir.tensor",
        _ => "lane.ir.unsupported",
    }
}
