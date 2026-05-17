//! Cerebras backend target — first-class surface tests.

use libmind::BackendTarget;
use libmind::{compile_source, CompileError, CompileOptions};

#[test]
fn cerebras_target_displays_as_cerebras() {
    assert_eq!(format!("{}", BackendTarget::Cerebras), "cerebras");
}

#[test]
fn cerebras_target_is_distinct_from_gpu() {
    assert_ne!(BackendTarget::Cerebras, BackendTarget::Gpu);
    assert_ne!(BackendTarget::Cerebras, BackendTarget::Cpu);
}

#[test]
fn compile_source_marks_cerebras_unavailable_in_compiler_crate() {
    // mindc reaches the IR layer for any target but defers final
    // emission to mind-runtime (which ships the per-backend .so/.dylib
    // libraries). Selecting Cerebras as the target through
    // `compile_source` must surface `BackendUnavailable` so callers
    // know to route via the runtime crate rather than the compiler.
    let opts = CompileOptions {
        target: BackendTarget::Cerebras,
        ..CompileOptions::default()
    };
    let err = compile_source("module m { fn id(x: i32) -> i32 { x } }\n", &opts)
        .expect_err("Cerebras target should be unavailable inside libmind crate");
    match err {
        CompileError::BackendUnavailable { target } => {
            assert_eq!(target, BackendTarget::Cerebras);
        }
        other => panic!("expected BackendUnavailable, got {other:?}"),
    }
}
