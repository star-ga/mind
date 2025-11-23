// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
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
