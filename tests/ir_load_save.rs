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
//
// Part of the MIND project (Machine Intelligence Native Design).
//
// Stability tests for the public IR `load`/`save` contract introduced in
// mindc 0.2.5. These pin the runtime-facing entry points used by
// `mind-runtime` and other backends to consume pre-compiled IR without
// re-running the surface parser.

use libmind::ir;
use libmind::pipeline::{CompileOptions, compile_source, compile_to_mic_text};

const SAMPLE_PROGRAM: &str = r#"
fn main() {
    let a = tensor.rand(2, 2);
    let b = tensor.rand(2, 2);
    let c = tensor.matmul(a, b);
    return c;
}
"#;

#[test]
fn ir_save_then_load_round_trips() {
    let opts = CompileOptions::default();
    let products = compile_source(SAMPLE_PROGRAM, &opts).expect("compile");
    let mic_text = ir::save(&products.ir);
    assert!(
        mic_text.starts_with("mic@1\n"),
        "save() must produce mic@1 text"
    );
    let loaded = ir::load(mic_text.as_bytes()).expect("load");
    let re_emitted = ir::save(&loaded);
    assert_eq!(
        re_emitted, mic_text,
        "save→load→save must be a fixed point (RFC-0001 determinism)"
    );
}

#[test]
fn compile_to_mic_text_matches_save() {
    let opts = CompileOptions::default();
    let direct = compile_to_mic_text(SAMPLE_PROGRAM, &opts).expect("compile");
    let products = compile_source(SAMPLE_PROGRAM, &opts).expect("compile");
    let via_save = ir::save(&products.ir);
    assert_eq!(
        direct, via_save,
        "compile_to_mic_text == save(compile_source.ir)"
    );
}

#[test]
fn ir_save_is_deterministic() {
    let opts = CompileOptions::default();
    let products = compile_source(SAMPLE_PROGRAM, &opts).expect("compile");
    let a = ir::save(&products.ir);
    let b = ir::save(&products.ir);
    let c = ir::save(&products.ir);
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[test]
fn ir_load_rejects_unknown_format() {
    let err = ir::load(b"not a MIC document").unwrap_err();
    assert!(
        matches!(err, ir::LoadError::UnknownFormat),
        "got: {:?}",
        err
    );
}

#[test]
fn ir_load_rejects_invalid_utf8() {
    let bytes = [0xFF, 0xFE, 0xFD];
    let err = ir::load(&bytes).unwrap_err();
    assert!(
        matches!(err, ir::LoadError::InvalidUtf8(_)),
        "got: {:?}",
        err
    );
}

/// Regression: `tensor.relu` must survive the mic@1 text round-trip. The text
/// is the canonical `trace_hash` source (RFC 0016) — a dropped relu would
/// attest IR missing the activation. (Found by audit on the relu lowering work.)
const RELU_PROGRAM: &str = r#"
fn main() {
    let a = tensor.rand(4, 8);
    let r = tensor.relu(a);
    return r;
}
"#;

/// Regression: `tensor.relu` must survive the mic@1 text round-trip. The text
/// is the canonical `trace_hash` source (RFC 0016) — a dropped relu would
/// attest IR missing the activation. (Found by audit on the relu lowering work.)
#[test]
fn ir_save_load_round_trips_relu() {
    let opts = CompileOptions::default();
    let products = compile_source("let a: Tensor[f32,(4,8)] = 1;\ntensor.relu(a)", &opts)
        .expect("compile relu");
    let mic_text = ir::save(&products.ir);
    assert!(
        mic_text.contains(" relu "),
        "mic@1 text must contain the relu node, got:\n{mic_text}"
    );
    let loaded = ir::load(mic_text.as_bytes()).expect("load relu mic@1");
    assert_eq!(
        ir::save(&loaded),
        mic_text,
        "relu save->load->save must be a fixed point"
    );
}

/// Regression: backward ReLU (`ReluGrad`) must survive the mic@1 text round-trip
/// (exercises `parse_relu_grad`). Built directly — the op is autodiff-internal
/// with no surface syntax. trace_hash is SHA-256 of this text (RFC 0016), so a
/// dropped/mis-parsed backward op would attest incomplete gradient IR.
#[test]
fn ir_save_load_round_trips_relu_grad() {
    use libmind::ir::{IRModule, Instr};
    use libmind::types::{DType, ShapeDim};

    let mut module = IRModule::new();
    let g = module.fresh();
    module.instrs.push(Instr::ConstTensor(
        g,
        DType::F32,
        vec![ShapeDim::Known(4)],
        None,
    ));
    let x = module.fresh();
    module.instrs.push(Instr::ConstTensor(
        x,
        DType::F32,
        vec![ShapeDim::Known(4)],
        None,
    ));
    let dx = module.fresh();
    module.instrs.push(Instr::ReluGrad {
        dst: dx,
        grad: g,
        src: x,
    });
    module.instrs.push(Instr::Output(dx));

    let mic_text = ir::save(&module);
    assert!(
        mic_text.contains(" relu_grad "),
        "mic@1 text must contain the relu_grad node, got:\n{mic_text}"
    );
    let loaded = ir::load(mic_text.as_bytes()).expect("load relu_grad mic@1");
    assert_eq!(
        ir::save(&loaded),
        mic_text,
        "relu_grad save->load->save must be a fixed point"
    );
}
