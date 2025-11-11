#![allow(dead_code, unused_variables, unused_imports)]
use super::ShapeDim;
use super::TensorType;

/// Left-biased, minimal unifier for Phase 1.
/// - Known==Known => Known
/// - Known <-> Sym => Known
/// - Sym <-> Sym  => left symbol
/// - Different rank => left-biased "UNK" dims
pub fn unify(a: &TensorType, b: &TensorType) -> TensorType {
    let len = a.shape.len().max(b.shape.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let x = a.shape.get(i).cloned().unwrap_or(ShapeDim::Sym("UNK"));
        let y = b.shape.get(i).cloned().unwrap_or(ShapeDim::Sym("UNK"));
        let r = match (x, y) {
            (ShapeDim::Known(m), ShapeDim::Known(n)) if m == n => ShapeDim::Known(m),
            (ShapeDim::Known(_), ShapeDim::Known(_)) => ShapeDim::Sym("UNK"),
            (ShapeDim::Known(m), ShapeDim::Sym(_)) => ShapeDim::Known(m),
            (ShapeDim::Sym(_), ShapeDim::Known(n)) => ShapeDim::Known(n),
            (ShapeDim::Sym(s), ShapeDim::Sym(_)) => ShapeDim::Sym(s),
        };
        out.push(r);
    }
    TensorType {
        dtype: a.dtype.clone(),
        shape: out,
    }
}
