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

//! Fleet-audit r1#4 — re-audited as ALREADY HANDLED (characterization lock, not
//! a fix). The claim was that a bogus `grad(..., wrt=[typo])` target "slips
//! through" name resolution. It does NOT: type-checking reports it with a
//! SPECIFIC diagnostic (E2001 "unknown tensor `X` in `wrt`"). A resolve-pass
//! check would only replace that precise message with a generic E2002. This test
//! LOCKS the correct behaviour — a bogus wrt is reported exactly once, a valid
//! wrt is clean — so it can never regress to actually slipping through.

#![cfg(feature = "std-surface")]

use libmind::parser::parse;
use libmind::type_checker::check_module_types_in_file;

fn diagnostics(src: &str) -> Vec<String> {
    let module = parse(src).expect("parse");
    check_module_types_in_file(&module, src, Some("t.mind"), &Default::default())
        .iter()
        .map(|e| format!("{e:?}"))
        .collect()
}

fn has_e2002(errs: &[String]) -> bool {
    errs.iter().any(|e| e.contains("E2002"))
}

#[test]
fn grad_wrt_undefined_var_reported_exactly_once() {
    // `nonexistent_var` is never declared. It MUST be reported — and exactly
    // once. Type-checking already flags it as E2001 ("unknown tensor in wrt");
    // the resolve pass must not ALSO emit a second (E2002) diagnostic for the
    // same name, which would be confusing duplicate output.
    let errs =
        diagnostics("let x: Tensor[f32,(2,3)] = 0; grad(tensor.sum(x), wrt=[nonexistent_var])");
    let mentions: Vec<&String> = errs
        .iter()
        .filter(|e| e.contains("nonexistent_var"))
        .collect();
    assert!(
        !mentions.is_empty(),
        "a bogus wrt target must be reported; got {errs:?}"
    );
    assert_eq!(
        mentions.len(),
        1,
        "a bogus wrt target must be reported exactly once (no duplicate \
         resolve+type-check diagnostic); got {mentions:?}"
    );
}

#[test]
fn grad_wrt_defined_var_is_clean() {
    // `x` is declared — a valid wrt target must NOT be flagged (additive: no
    // new false positives).
    let errs = diagnostics("let x: Tensor[f32,(2,3)] = 0; grad(tensor.sum(x), wrt=[x])");
    assert!(
        !has_e2002(&errs),
        "a valid wrt target must not flag E2002; got {errs:?}"
    );
}
