// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

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
