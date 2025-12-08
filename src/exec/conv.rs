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

use crate::eval::value::TensorVal;
use crate::types::ConvPadding;
use crate::types::ShapeDim;

use super::cpu::ExecError;

fn _shape_as_usize(_shape: &[ShapeDim]) -> Result<Vec<usize>, ExecError> {
    // TODO(runtime): shape handling is implemented in the proprietary runtime backend.
    Err(ExecError::Unsupported(
        "Convolution shape materialization is provided by the proprietary MIND runtime".into(),
    ))
}

pub fn exec_conv2d(
    _input: &TensorVal,
    _weights: &TensorVal,
    _stride_h: usize,
    _stride_w: usize,
    _padding: ConvPadding,
) -> Result<TensorVal, ExecError> {
    // TODO(runtime): implemented in proprietary `mind-runtime` backend.
    Err(ExecError::Unsupported(
        "Conv2d execution is provided by the proprietary MIND runtime backend".into(),
    ))
}
