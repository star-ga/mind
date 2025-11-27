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

use std::marker::PhantomData;

/// Minimal placeholder tensor for Phase 1 (no data buffer).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    _t: PhantomData<T>,
}

impl<T> Tensor<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            _t: PhantomData,
        }
    }
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            _t: PhantomData,
        }
    }
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        Self {
            shape: new_shape.to_vec(),
            _t: PhantomData,
        }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn sum(&self) -> f64 {
        0.0
    } // placeholder
    pub fn mean(&self) -> f64 {
        0.0
    } // placeholder
}
