#![allow(dead_code, unused_variables, unused_imports)]

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
