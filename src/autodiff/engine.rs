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

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ir::{self, BinOp, IRModule, Instr, ValueId};

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
    #[error("unsupported operation for autodiff: {op}")]
    UnsupportedOp { op: &'static str },
    /// A deterministic error when shape or operation requirements are not met.
    #[error("invalid autodiff input: {0}")]
    InvalidInput(String),
    /// Verification of the generated IR failed.
    #[error("autodiff verification failed: {0}")]
    Verification(String),
    /// The provided axes or permutation were invalid for autodiff.
    #[error("invalid axes for autodiff: {reason}")]
    InvalidAxis { reason: String },
    /// Unsupported shape manipulation pattern.
    #[error("unsupported shape pattern for autodiff: {reason}")]
    UnsupportedShape { reason: String },
}

/// Entry point for generating gradient IR from a public MIND IR module.
pub fn differentiate_function(
    module: &IRModule,
    fn_name: &str,
) -> Result<GradientResult, AutodiffError> {
    differentiate_with_options(module, fn_name, GradientOptions::default())
}

/// Options controlling gradient emission.
#[derive(Debug, Clone, Copy)]
pub struct GradientOptions {
    /// When true, run IR verification on both the primal and gradient modules.
    pub verify_gradients: bool,
}

impl Default for GradientOptions {
    fn default() -> Self {
        Self {
            verify_gradients: true,
        }
    }
}

/// Entry point for generating gradient IR from a public MIND IR module with
/// configuration.
pub fn differentiate_with_options(
    module: &IRModule,
    fn_name: &str,
    opts: GradientOptions,
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

    if opts.verify_gradients {
        ir::verify_module(module).map_err(|e| AutodiffError::Verification(e.to_string()))?;
    }

    let mut builder = GradientBuilder::new(module);
    builder.seed_output(output);
    builder.propagate_gradients()?;
    let mut result = builder.finish();

    canonicalize_gradients(&mut result);
    if opts.verify_gradients {
        ir::verify_module(&result.gradient_module)
            .map_err(|e| AutodiffError::Verification(e.to_string()))?;
    }

    Ok(result)
}

fn canonicalize_gradients(result: &mut GradientResult) {
    let gradient_outputs: BTreeSet<ValueId> = result.gradients.values().copied().collect();

    for root in &gradient_outputs {
        result.gradient_module.instrs.push(Instr::Output(*root));
    }

    crate::opt::ir_canonical::canonicalize_module(&mut result.gradient_module);

    result.gradient_module.instrs.retain(|instr| {
        if let Instr::Output(id) = instr {
            if gradient_outputs.contains(id) {
                return false;
            }
        }
        true
    });

    let mut max_seen = 0usize;
    for instr in &result.gradient_module.instrs {
        if let Some(dst) = instruction_dst(instr) {
            max_seen = max_seen.max(dst.0 + 1);
        }
    }
    result.gradient_module.next_id = max_seen;
}

struct GradientBuilder<'a> {
    primal: &'a IRModule,
    gradient: IRModule,
    grads: BTreeMap<ValueId, ValueId>,
    leaves: BTreeMap<ValueId, &'static str>,
}

impl<'a> GradientBuilder<'a> {
    fn new(primal: &'a IRModule) -> Self {
        let mut gradient = IRModule::new();
        gradient.instrs.extend(primal.instrs.clone());
        gradient.next_id = primal.next_id;

        let mut leaves = BTreeMap::new();
        for instr in &primal.instrs {
            if let Some(dst) = instruction_dst(instr) {
                if matches!(instr, Instr::ConstI64(..) | Instr::ConstTensor(..)) {
                    leaves.insert(dst, "leaf");
                }
            }
        }

        Self {
            primal,
            gradient,
            grads: BTreeMap::new(),
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
        for id in self.leaves.keys() {
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
    fn add_const_i64(&mut self, value: i64) -> ValueId;
    fn add_binop(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId) -> ValueId;
    fn add_transpose(&mut self, src: ValueId, perm: Vec<i64>) -> ValueId;
    fn add_matmul(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn add_grad(&mut self, target: ValueId, contribution: ValueId);
}

impl<'a> GradientOps for GradientBuilder<'a> {
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
