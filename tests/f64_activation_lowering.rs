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

//! Fleet-audit r3 #1/#2/#3 regression: `DType::F64` tensors must lower through
//! the FLOAT arith path. Before the fix, `F64` fell through to the integer `_`
//! arm in `Relu` (→ `arith.maxsi` on an `f64`), `ReluGrad` (→ `arith.cmpi`), and
//! `format_fill` (→ truncated/int `0` fill) — all invalid MLIR or wrong values.
//! These assertions FAIL on the pre-fix compiler (RED) and pass after (GREEN).

#![cfg(feature = "mlir-lowering")]

use libmind::compile_ir_to_mlir_text;
use libmind::ir::{IRModule, Instr, ValueId};
use libmind::types::{DType, ShapeDim};

fn f64_tensor_const(module: &mut IRModule, fill: Option<f64>) -> ValueId {
    let id = module.fresh();
    module.instrs.push(Instr::ConstTensor(
        id,
        DType::F64,
        vec![ShapeDim::Known(2), ShapeDim::Known(2)],
        fill,
    ));
    id
}

#[test]
fn f64_relu_uses_float_max_not_integer_max() {
    let mut module = IRModule::new();
    let src = f64_tensor_const(&mut module, Some(1.0));
    let dst = module.fresh();
    module.instrs.push(Instr::Relu { dst, src });
    module.instrs.push(Instr::Output(dst));

    let text = compile_ir_to_mlir_text(&mut module).expect("f64 relu must lower");
    assert!(
        text.contains("arith.maximumf"),
        "f64 relu must use arith.maximumf, got:\n{text}"
    );
    assert!(
        !text.contains("arith.maxsi"),
        "f64 relu must NOT emit integer arith.maxsi:\n{text}"
    );
    assert!(text.contains("f64"), "must lower as an f64 tensor:\n{text}");
}

#[test]
fn f64_relu_grad_uses_float_cmp_not_integer_cmp() {
    let mut module = IRModule::new();
    let grad = f64_tensor_const(&mut module, Some(1.0));
    let src = f64_tensor_const(&mut module, Some(1.0));
    let dst = module.fresh();
    module.instrs.push(Instr::ReluGrad { dst, grad, src });
    module.instrs.push(Instr::Output(dst));

    let text = compile_ir_to_mlir_text(&mut module).expect("f64 relu_grad must lower");
    assert!(
        text.contains("arith.cmpf"),
        "f64 relu_grad must gate with arith.cmpf:\n{text}"
    );
    assert!(
        !text.contains("arith.cmpi"),
        "f64 relu_grad must NOT emit integer arith.cmpi:\n{text}"
    );
}

#[test]
fn f64_const_fill_keeps_fraction_not_truncated() {
    let mut module = IRModule::new();
    // A fractional fill must survive as `0.5` — the integer arm `.trunc()`s it to
    // `0`, which is both the wrong value and (as a bare int) invalid for an f64
    // tensor element.
    let t = f64_tensor_const(&mut module, Some(0.5));
    module.instrs.push(Instr::Output(t));

    let text = compile_ir_to_mlir_text(&mut module).expect("f64 const fill must lower");
    assert!(
        text.contains("0.5"),
        "f64 fill 0.5 must not be truncated to an integer:\n{text}"
    );
}

#[test]
fn f64_zero_fill_is_float_zero() {
    let mut module = IRModule::new();
    // A `None` (zeroed) f64 fill must emit `0.0`, not the integer `0`.
    let t = f64_tensor_const(&mut module, None);
    module.instrs.push(Instr::Output(t));

    let text = compile_ir_to_mlir_text(&mut module).expect("f64 zero fill must lower");
    assert!(
        text.contains("0.0"),
        "f64 zero fill must be the float literal 0.0:\n{text}"
    );
}
