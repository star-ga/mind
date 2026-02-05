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

use libmind::diagnostics::{
    ColorChoice, Diagnostic, DiagnosticEmitter, DiagnosticFormat, Severity, Span,
};

fn sample_diag() -> Diagnostic {
    Diagnostic {
        phase: "parse",
        code: "E1001",
        severity: Severity::Error,
        message: "unexpected token".to_string(),
        span: Some(Span {
            file: Some("sample.mind".to_string()),
            line: 3,
            column: 5,
            length: 1,
        }),
        notes: vec!["while parsing function `main`".to_string()],
        help: Some("remove the stray token".to_string()),
    }
}

#[test]
fn human_output_smoke() {
    let diag = sample_diag();
    let emitter = DiagnosticEmitter::new(DiagnosticFormat::Human, ColorChoice::Never);
    let rendered = emitter.render_human(&diag, Some("let x = );"));

    assert!(rendered.contains("error[parse][E1001]"));
    assert!(rendered.contains("sample.mind:3:5"));
    assert!(rendered.contains("while parsing function"));
    assert!(rendered.contains("help: remove the stray token"));
}

#[test]
fn json_output_smoke() {
    let diag = sample_diag();
    let emitter = DiagnosticEmitter::new(DiagnosticFormat::Json, ColorChoice::Never);
    let mut buf = Vec::new();
    emitter.emit(&diag, None, &mut buf);
    let body = String::from_utf8(buf).expect("utf8");
    let value: serde_json::Value = serde_json::from_str(body.trim()).expect("json diagnostic");

    assert_eq!(value["phase"], "parse");
    assert_eq!(value["code"], "E1001");
    assert_eq!(value["severity"], "error");
}

#[test]
fn json_includes_span() {
    let diag = sample_diag();
    let emitter = DiagnosticEmitter::new(DiagnosticFormat::Json, ColorChoice::Never);
    let mut buf = Vec::new();
    emitter.emit(&diag, None, &mut buf);
    let body = String::from_utf8(buf).expect("utf8");
    let value: serde_json::Value = serde_json::from_str(body.trim()).expect("json diagnostic");

    assert_eq!(value["span"]["file"], "sample.mind");
    assert_eq!(value["span"]["line"], 3);
    assert_eq!(value["span"]["column"], 5);
    assert_eq!(value["span"]["length"], 1);
}
