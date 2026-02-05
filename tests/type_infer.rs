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

use libmind::types::infer::unify;
use libmind::types::DType;
use libmind::types::ShapeDim;
use libmind::types::TensorType;

#[test]
fn unify_known_and_sym() {
    let a = TensorType::new(DType::F32, vec![ShapeDim::Known(32), ShapeDim::Sym("B")]);
    let b = TensorType::new(DType::F32, vec![ShapeDim::Known(32), ShapeDim::Known(8)]);
    let u = unify(&a, &b);
    assert_eq!(u.shape, vec![ShapeDim::Known(32), ShapeDim::Known(8)]);
}

#[test]
fn unify_rank_mismatch_left_biased() {
    let a = TensorType::new(DType::F32, vec![ShapeDim::Known(4), ShapeDim::Sym("N")]);
    let b = TensorType::new(DType::F32, vec![ShapeDim::Known(4)]);
    let u = unify(&a, &b);
    assert_eq!(u.shape[0], ShapeDim::Known(4));
}
