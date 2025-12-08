// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use mind::shapes::*;
use mind::types::{ConvPadding, ShapeDim};

fn kd(n: usize) -> ShapeDim {
    ShapeDim::Known(n)
}

#[test]
fn broadcast_examples() {
    let out = broadcast_shapes(&[kd(3), kd(1)], &[kd(1), kd(4)]).unwrap();
    assert_eq!(out, vec![kd(3), kd(4)]);

    let scalar = broadcast_shapes(&[], &[kd(2), kd(5)]).unwrap();
    assert_eq!(scalar, vec![kd(2), kd(5)]);

    let err = broadcast_shapes(&[kd(3)], &[kd(2), kd(2)]).unwrap_err();
    assert!(matches!(err, ShapeError::BroadcastIncompatible { .. }));
}

#[test]
fn reductions_with_axes() {
    let shape = vec![kd(2), kd(3), kd(4)];
    let reduced = reduce_shape(&shape, &[1], false).unwrap();
    assert_eq!(reduced, vec![kd(2), kd(4)]);

    let kept = reduce_shape(&shape, &[0, 2], true).unwrap();
    assert_eq!(kept, vec![kd(1), kd(3), kd(1)]);

    let err = reduce_shape(&shape, &[3], false).unwrap_err();
    assert!(matches!(err, ShapeError::AxisOutOfRange { .. }));
}

#[test]
fn reshape_checks() {
    let old = vec![kd(2), kd(3)];
    let new = vec![kd(3), kd(2)];
    let reshaped = reshape_shape(&old, &new).unwrap();
    assert_eq!(reshaped, new);

    let err = reshape_shape(&old, &[kd(2), kd(2)]).unwrap_err();
    assert!(matches!(err, ShapeError::ElementCountMismatch { .. }));
}

#[test]
fn transpose_and_axis_ops() {
    let shape = vec![kd(1), kd(2), kd(3)];
    let transposed = transpose_shape(&shape, &[2, 0, 1]).unwrap();
    assert_eq!(transposed, vec![kd(3), kd(1), kd(2)]);

    let expanded = expand_dims_shape(&shape, -1).unwrap();
    assert_eq!(expanded, vec![kd(1), kd(2), kd(3), kd(1)]);

    let squeezed = squeeze_shape(&expanded, &[]).unwrap();
    assert_eq!(squeezed, vec![kd(2), kd(3)]);
}

#[test]
fn indexing_and_slicing() {
    let shape = vec![kd(4), kd(5), kd(6)];
    let indexed = index_shape(&shape, 1).unwrap();
    assert_eq!(indexed, vec![kd(4), kd(6)]);

    let slice = slice_shape(&shape, 0, 1, 3).unwrap();
    assert_eq!(slice, vec![kd(2), kd(5), kd(6)]);

    let strided = slice_stride_shape(&shape, 2, 0, 6, 2).unwrap();
    assert_eq!(strided, vec![kd(4), kd(5), kd(3)]);
}

#[test]
fn gather_shapes() {
    let shape = vec![kd(2), kd(3), kd(4)];
    let idx = vec![kd(5), kd(6)];
    let gathered = gather_shape(&shape, 1, &idx).unwrap();
    assert_eq!(gathered, vec![kd(2), kd(5), kd(6), kd(4)]);
}

#[test]
fn conv2d_shapes() {
    let input = vec![kd(1), kd(8), kd(8), kd(3)];
    let filter = vec![kd(3), kd(3), kd(3), kd(4)];
    let (h, w) = conv2d_shape(&input, &filter, 1, 1, ConvPadding::Valid).unwrap();
    assert_eq!(h, kd(6));
    assert_eq!(w, kd(6));

    let same = conv2d_shape(&input, &filter, 2, 2, ConvPadding::Same).unwrap();
    assert_eq!(same.0, kd(4));
    assert_eq!(same.1, kd(4));
}
