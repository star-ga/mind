// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Execution backend stubs for the MIND runtime.
//!
//! # Architecture
//!
//! This module contains stub implementations that define the execution backend
//! interface. Real implementations are provided by the proprietary `mind-runtime`
//! backend (see <https://github.com/star-ga/mind-runtime>).
//!
//! # TODO(runtime) Markers
//!
//! The `TODO(runtime)` comments in this module are **architectural boundary markers**,
//! not missing features. They serve as documentation for where the proprietary runtime
//! hooks into the open-core compiler. Each stub:
//!
//! - Returns `ExecError::Unsupported` to indicate runtime dependency
//! - Documents the expected interface for the proprietary backend
//! - Enables the open-core compiler to type-check and IR-lower without runtime
//!
//! ## Stub Categories (18 total)
//!
//! | Module   | Count | Operations                                          |
//! |----------|-------|-----------------------------------------------------|
//! | `cpu.rs` | 16    | Arithmetic, reductions, activations, linear algebra |
//! | `conv.rs`| 2     | Shape materialization, Conv2D execution             |
//!
//! These stubs are intentionally minimal and stable. The runtime backend provides
//! optimized implementations for CPU (AVX2/AVX-512), GPU (CUDA/ROCm), and accelerators.

#[cfg(feature = "cpu-exec")]
pub mod cpu;

#[cfg(feature = "cpu-conv")]
pub mod conv;

#[cfg(feature = "cpu-exec")]
pub fn simd_chunks_mut(data: &mut [f32]) -> impl Iterator<Item = &mut [f32]> + '_ {
    const CHUNK: usize = 1024;
    data.chunks_mut(CHUNK)
}

#[cfg(not(feature = "cpu-exec"))]
mod cpu_disabled {
    #[allow(dead_code)]
    pub struct CpuExecUnavailable;
}
