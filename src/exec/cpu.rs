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

//! CPU execution stubs. Real implementations live in the proprietary `mind-runtime` backend.

use crate::eval::value::TensorVal;
use crate::types::ShapeDim;

#[derive(Debug)]
pub enum ExecError {
    Unsupported(String),
    Shape(String),
    Type(String),
    Math(String),
}

type R<T> = Result<T, ExecError>;

fn runtime_stub<T>() -> R<T> {
    Err(ExecError::Unsupported(
        "CPU execution is provided by the proprietary MIND runtime backend".into(),
    ))
}

fn _shape_usize(_shape: &[ShapeDim]) -> Option<Vec<usize>> {
    // TODO(runtime): implement shape materialization in the proprietary backend.
    None
}

pub fn exec_add(_lhs: &TensorVal, _rhs: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_sub(_lhs: &TensorVal, _rhs: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_mul(_lhs: &TensorVal, _rhs: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_div(_lhs: &TensorVal, _rhs: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_add_scalar(_t: &TensorVal, _scalar: f32) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_sub_scalar(_t: &TensorVal, _scalar: f32) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_scalar_sub(_scalar: f32, _t: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_mul_scalar(_t: &TensorVal, _scalar: f32) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_div_scalar(_t: &TensorVal, _scalar: f32, _tensor_on_left: bool) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_sum_all(_t: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_mean_all(_t: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn relu_inplace(_buf: &mut [f32]) {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    unimplemented!("In-place ReLU is provided by the proprietary MIND runtime backend");
}

pub fn exec_relu(_t: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_matmul(_lhs: &TensorVal, _rhs: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}

pub fn exec_dot(_lhs: &TensorVal, _rhs: &TensorVal) -> R<TensorVal> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    runtime_stub()
}
