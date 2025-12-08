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

//! Static automatic differentiation for the public MIND IR.
//!
//! The autodiff pipeline builds gradient IR at compile time without relying on
//! any runtime tape. The entry point is [`differentiate_function`], which
//! consumes a deterministic [`IRModule`](crate::ir::IRModule) and produces a
//! gradient IR module plus metadata describing the gradients that were
//! computed.
//!
//! The implementation is intentionally deterministic and only depends on the
//! public IR and evaluator code. No hooks into the private runtime are
//! required.

mod engine;
mod rules;

pub use engine::{
    differentiate_function, differentiate_with_options, AutodiffError, GradientOptions,
    GradientResult,
};
