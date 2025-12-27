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

use mind::stdlib::tensor::Tensor;

#[test]
fn tensor_shape_and_reshape() {
    let t = Tensor::<f32>::zeros(&[2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    let r = t.reshape(&[3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
}

#[test]
fn tensor_ops_placeholders_compile() {
    let t = Tensor::<f32>::ones(&[2, 3]);
    let _ = t.sum();
    let _ = t.mean();
}
