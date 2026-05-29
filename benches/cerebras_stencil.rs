// Copyright 2025-2026 STARGA Inc.
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

//! Compile-time cost benchmark for `mind.cerebras.stencil_tile` emission.
//!
//! This benchmark measures how long the MIND compiler surface takes to
//! construct and lower a `stencil_tile` op node for representative fabric
//! region sizes. It is purely a *compile-time* (IR construction + MLIR text
//! emission) measurement — runtime utilisation on actual WSE-3 hardware is
//! the partnership ask and requires physical access.
//!
//! All measurements are in the `compiler_pipeline` criterion group so they
//! appear alongside the headline `small_matmul / medium_mlp / large_network`
//! benches and are subject to the same 5% gate in `bench-gate.yml`.
//!
//! # Configurations
//!
//! | Variant | Fabric region | Notes                              |
//! |---------|---------------|------------------------------------|
//! | tiny    | 32 × 32       | Minimal representative tile         |
//! | medium  | 128 × 128     | Typical working sub-region          |
//! | large   | 256 × 256     | `FabricRegion::default()` in runtime|
//! | wafer   | 750 × 750     | Approximate WSE-3 working block     |

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use libmind::ops::cerebras::{StencilElemType, StencilTileOp};

/// Configurations exercised by the benchmark.
const CONFIGS: &[(&str, u32, u32)] = &[
    ("tiny_32x32", 32, 32),
    ("medium_128x128", 128, 128),
    ("large_256x256", 256, 256),
    ("wafer_750x750", 750, 750),
];

/// Benchmark group: stencil_tile op construction + MLIR text emission.
///
/// Placed in the `compiler_pipeline` group so it is gated by the baseline
/// file `.bench-baseline-2026-05-17-phase10-6.txt` at the 5% threshold.
fn bench_stencil_tile_emit(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiler_pipeline");

    for &(name, rows, cols) in CONFIGS {
        group.bench_with_input(
            BenchmarkId::new("stencil_tile_emit_q16_16", name),
            &(rows, cols),
            |b, &(r, c)| {
                b.iter(|| {
                    let op = StencilTileOp::new(
                        black_box("%stencil_out"),
                        black_box(r),
                        black_box(c),
                        black_box(StencilElemType::Q16_16),
                        black_box("laplacian_5pt"),
                    )
                    .expect("bench config must be valid");
                    // Emit MLIR text — this is the "compilation" cost the
                    // Cerebras pitch is measuring.
                    black_box(op.to_mlir_text())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_stencil_tile_emit);
criterion_main!(benches);
