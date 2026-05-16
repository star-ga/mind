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

use libmind::eval;
use libmind::parser;

#[test]
fn lower_and_eval_add_ints() {
    let src = "1 + 2 * 3";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    let value = eval::eval_ir(&ir);
    let rendered = eval::format_value_human(&value);
    assert_eq!(rendered, "7");
}

#[test]
fn lower_tensor_preview() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    let value = eval::eval_ir(&ir);
    let rendered = eval::format_value_human(&value);
    assert!(rendered.contains("Tensor["), "{rendered}");
}

// RFC 0002 deliverable 1 — `Node::Export` lowers into `IRModule.exports`.
#[test]
fn lower_export_block_populates_ir_exports() {
    let src = "export { foo, bar }";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    assert_eq!(ir.exports.len(), 2);
    assert!(ir.exports.contains("foo"));
    assert!(ir.exports.contains("bar"));
    // The lowering MUST NOT add an Output instruction for the export block;
    // it's metadata, not a value.
    assert!(
        !ir.instrs
            .iter()
            .any(|i| matches!(i, libmind::ir::Instr::Output(_))),
        "export block must not produce an Output instr"
    );
}

#[test]
fn lower_no_export_keeps_exports_empty() {
    let src = "1 + 2";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    assert!(
        ir.exports.is_empty(),
        "default code path must leave IRModule.exports empty"
    );
}
