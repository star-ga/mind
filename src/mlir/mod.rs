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

//! MLIR lowering pipeline for the public MIND IR.
//!
//! The lowering stage is intentionally textual and deterministic to
//! support debugging, demos, and future backend integration without
//! pulling in private runtime crates. The entry points are
//! [`lower_ir_to_mlir`] and [`compile_ir_to_mlir_text`].

/// Cache-blocking knobs for the fused Q16.16 GEMM macro-kernel (autotuner
/// surface). Public so a sweep harness can read/override the chosen point.
pub mod gemm_tuning;

mod lowering;

#[cfg(feature = "ffi-c-user")]
pub mod c_export;

pub use lowering::{
    MlirLowerError, MlirModule, compile_ir_to_mlir_text, lower_ir_to_mlir,
    lower_ir_to_mlir_with_entry,
};
