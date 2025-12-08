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

use std::collections::BTreeMap;
use std::fmt;

use crate::ir::{BinOp, IRModule, Instr, ValueId};

use super::rules;

/// Result of a differentiation run.
#[derive(Debug, Clone)]
pub struct GradientResult {
    /// Gradient IR that includes the primal computation followed by gradient ops.
    pub gradient_module: IRModule,
    /// Gradients keyed by the primal value they differentiate.
    pub gradients: BTreeMap<ValueId, ValueId>,
}

/// Errors returned by the autodiff engine.
#[derive(Debug, thiserror::Error)]
pub enum AutodiffError {
    /// The requested function name was not found. The public IR uses a single
    /// module without named functions; the engine therefore only accepts
    /// `"main"` to make intent explicit.
    #[error("function '{0}' not found in module (only 'main' is supported)")]
    FunctionNotFound(String),
    /// The IR module did not provide an output instruction.
    #[error("IR module does not contain any Output instruction")]
    MissingOutput,
    /// Multiple outputs are not yet supported by the static autodiff pipeline.
    #[error("IR module contains multiple outputs; expected a single scalar output")]
    MultipleOutputs,
    /// The autodiff engine does not have a rule for the encountered operation.
    #[error("unsupported operation for autodiff: {0}")]
    UnsupportedOp(&'static str),
    /// A deterministic error when shape or operation requirements are not met.
    #[error("invalid autodiff input: {0}")]
    InvalidInput(String),
}

/// Entry point for generating gradient IR from a public MIND IR module.
pub fn differentiate_function(
    module: &IRModule,
    fn_name: &str,
) -> Result<GradientResult, AutodiffError> {
    if fn_name != "main" {
        return Err(AutodiffError::FunctionNotFound(fn_name.to_string()));
    }

    let outputs: Vec<ValueId> = module
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            Instr::Output(id) => Some(*id),
            _ => None,
        })
        .collect();

    let output = match outputs.as_slice() {
        [] => return Err(AutodiffError::MissingOutput),
        [single] => *single,
        _ => return Err(AutodiffError::MultipleOutputs),
    };

    let mut builder = GradientBuilder::new(module);
    builder.seed_output(output);
    builder.propagate_gradients()?;
    Ok(builder.finish())
}

struct GradientBuilder<'a> {
    primal: &'a IRModule,
    gradient: IRModule,
    grads: BTreeMap<ValueId, ValueId>,
    primal_defs: BTreeMap<ValueId, &'a Instr>,
    leaves: BTreeMap<ValueId, &'static str>,
}

impl<'a> GradientBuilder<'a> {
    fn new(primal: &'a IRModule) -> Self {
        let mut gradient = IRModule::new();
        gradient.instrs.extend(primal.instrs.clone());
        gradient.next_id = primal.next_id;

        let mut primal_defs = BTreeMap::new();
        let mut leaves = BTreeMap::new();
        for instr in &primal.instrs {
            if let Some(dst) = instruction_dst(instr) {
                primal_defs.insert(dst, instr);
                if matches!(instr, Instr::ConstI64(..) | Instr::ConstTensor(..)) {
                    leaves.insert(dst, "leaf");
                }
            }
        }

        Self {
            primal,
            gradient,
            grads: BTreeMap::new(),
            primal_defs,
            leaves,
        }
    }

    fn seed_output(&mut self, output: ValueId) {
        let seed = self.add_const_i64(1);
        self.grads.insert(output, seed);
    }

    fn propagate_gradients(&mut self) -> Result<(), AutodiffError> {
        for instr in self.primal.instrs.iter().rev() {
            let Some(dst) = instruction_dst(instr) else {
                continue;
            };
            let Some(&upstream) = self.grads.get(&dst) else {
                continue;
            };
            rules::apply_rule(self, instr, upstream)?;
        }
        Ok(())
    }

    fn finish(self) -> GradientResult {
        let mut gradients = BTreeMap::new();
        for (id, _) in &self.leaves {
            if let Some(&grad) = self.grads.get(id) {
                gradients.insert(*id, grad);
            }
        }
        GradientResult {
            gradient_module: self.gradient,
            gradients,
        }
    }

    fn add_const_i64(&mut self, value: i64) -> ValueId {
        let dst = ValueId(self.gradient.next_id);
        self.gradient.next_id += 1;
        self.gradient.instrs.push(Instr::ConstI64(dst, value));
        dst
    }

    fn add_binop(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        let dst = ValueId(self.gradient.next_id);
        self.gradient.next_id += 1;
        self.gradient
            .instrs
            .push(Instr::BinOp { dst, op, lhs, rhs });
        dst
    }

    fn add_transpose(&mut self, src: ValueId, perm: Vec<i64>) -> ValueId {
        let dst = ValueId(self.gradient.next_id);
        self.gradient.next_id += 1;
        self.gradient
            .instrs
            .push(Instr::Transpose { dst, src, perm });
        dst
    }

    fn add_matmul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let dst = ValueId(self.gradient.next_id);
        self.gradient.next_id += 1;
        self.gradient.instrs.push(Instr::MatMul { dst, a, b });
        dst
    }

    fn add_grad(&mut self, target: ValueId, contribution: ValueId) {
        match self.grads.get(&target).copied() {
            None => {
                self.grads.insert(target, contribution);
            }
            Some(existing) => {
                let summed = self.add_binop(BinOp::Add, existing, contribution);
                self.grads.insert(target, summed);
            }
        }
    }
}

fn instruction_dst(instr: &Instr) -> Option<ValueId> {
    match instr {
        Instr::ConstI64(dst, ..)
        | Instr::ConstTensor(dst, ..)
        | Instr::BinOp { dst, .. }
        | Instr::Sum { dst, .. }
        | Instr::Mean { dst, .. }
        | Instr::Reshape { dst, .. }
        | Instr::ExpandDims { dst, .. }
        | Instr::Squeeze { dst, .. }
        | Instr::Transpose { dst, .. }
        | Instr::Dot { dst, .. }
        | Instr::MatMul { dst, .. }
        | Instr::Conv2d { dst, .. }
        | Instr::Index { dst, .. }
        | Instr::Slice { dst, .. }
        | Instr::Gather { dst, .. } => Some(*dst),
        Instr::Output(_) => None,
    }
}

impl fmt::Display for GradientResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Gradients:")?;
        for (src, grad) in &self.gradients {
            writeln!(f, "  {:?} -> {:?}", src, grad)?;
        }
        writeln!(f, "\nGradient IR:")?;
        write!(f, "{}", self.gradient_module)
    }
}

// Helper API used by derivative rules.
pub(super) trait GradientOps {
    fn gradient(&self) -> &IRModule;
    fn add_const_i64(&mut self, value: i64) -> ValueId;
    fn add_binop(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId) -> ValueId;
    fn add_transpose(&mut self, src: ValueId, perm: Vec<i64>) -> ValueId;
    fn add_matmul(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn add_grad(&mut self, target: ValueId, contribution: ValueId);
}

impl<'a> GradientOps for GradientBuilder<'a> {
    fn gradient(&self) -> &IRModule {
        &self.gradient
    }

    fn add_const_i64(&mut self, value: i64) -> ValueId {
        self.add_const_i64(value)
    }

    fn add_binop(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.add_binop(op, lhs, rhs)
    }

    fn add_transpose(&mut self, src: ValueId, perm: Vec<i64>) -> ValueId {
        self.add_transpose(src, perm)
    }

    fn add_matmul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.add_matmul(a, b)
    }

    fn add_grad(&mut self, target: ValueId, contribution: ValueId) {
        self.add_grad(target, contribution)
    }
}

pub(super) fn as_invalid(msg: impl Into<String>) -> AutodiffError {
    AutodiffError::InvalidInput(msg.into())
}
