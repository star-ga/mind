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
use libmind::pipeline::{compile_source, compile_to_mic_text, CompileOptions};

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
    assert_eq!(direct, via_save, "compile_to_mic_text == save(compile_source.ir)");
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
