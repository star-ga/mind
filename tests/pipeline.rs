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

use mind::pipeline::{compile_source, CompileOptions};
use mind::runtime::types::BackendTarget;
#[cfg(feature = "autodiff")]
use mind::ir::Instr;

#[test]
fn compile_source_stabilizes_ir() {
    let src = "3 * 3";
    let opts = CompileOptions {
        func: None,
        enable_autodiff: false,
        target: BackendTarget::Cpu,
    };

    let first = compile_source(src, &opts).expect("first compile");
    let second = compile_source(src, &opts).expect("second compile");

    assert_eq!(format!("{}", first.ir), format!("{}", second.ir));
    let rendered = format!("{}", first.ir);
    assert!(rendered.to_lowercase().contains("output"));
}

#[cfg(feature = "autodiff")]
#[test]
fn compile_source_runs_autodiff() {
    let src = "3 * 3";
    let opts = CompileOptions {
        func: Some("main".to_string()),
        enable_autodiff: true,
        target: BackendTarget::Cpu,
    };

    let ir_only = compile_source(
        src,
        &CompileOptions {
            func: Some("main".to_string()),
            enable_autodiff: false,
            target: BackendTarget::Cpu,
        },
    )
    .expect("compiled without autodiff");
    let outputs = ir_only
        .ir
        .instrs
        .iter()
        .filter(|instr| matches!(instr, Instr::Output(_)))
        .count();
    assert_eq!(outputs, 1, "expected a single output in canonical IR");

    let compiled = compile_source(src, &opts).expect("compiled with autodiff");
    let grad = compiled.grad.expect("gradient IR present");

    let repeated = compile_source(src, &opts).expect("repeat compile");

    assert_eq!(format!("{}", compiled.ir), format!("{}", repeated.ir));
    let grad_text = format!("{}", grad.gradient_module);
    let grad_text_repeat = format!(
        "{}",
        repeated
            .grad
            .as_ref()
            .expect("second gradient")
            .gradient_module
    );
    assert_eq!(grad_text, grad_text_repeat);
    assert!(grad_text.to_lowercase().contains("output"));
}

#[cfg(feature = "mlir-lowering")]
#[test]
fn lower_to_mlir_produces_stable_text() {
    use mind::pipeline::lower_to_mlir;

    let src = "1 + 2";
    let opts = CompileOptions {
        func: None,
        enable_autodiff: false,
        target: BackendTarget::Cpu,
    };

    let compiled = compile_source(src, &opts).expect("compiled IR");
    let mlir = lower_to_mlir(&compiled.ir, None).expect("lowered mlir");
    let mlir_again = lower_to_mlir(&compiled.ir, None).expect("second lowering");

    assert_eq!(mlir.primal_mlir, mlir_again.primal_mlir);
    assert!(mlir.primal_mlir.contains("func.func @main"));
}
