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

use std::collections::BTreeSet;

use crate::ir::{IRModule, Instr, ValueId, instruction_dst};
use crate::opt::ir_canonical::instruction_operands;

/// Which SSA rule a [`SsaViolation`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsaRule {
    /// A value id was produced by two different instructions.
    SingleAssignment,
    /// An operand was referenced before any instruction defined it (in
    /// linearized program order; see [`check_ssa_well_formed`]).
    DefineBeforeUse,
}

impl SsaRule {
    /// Short, stable identifier used in CLI / JSON output.
    pub fn as_str(self) -> &'static str {
        match self {
            SsaRule::SingleAssignment => "single-assignment",
            SsaRule::DefineBeforeUse => "define-before-use",
        }
    }
}

/// A single SSA well-formedness violation: the offending value id and the
/// rule it breaks. Carries a human-readable reason via [`std::fmt::Display`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsaViolation {
    /// The value id (`%N`) that breaks the rule.
    pub value: ValueId,
    /// Which SSA rule was violated.
    pub rule: SsaRule,
}

impl std::fmt::Display for SsaViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.rule {
            SsaRule::SingleAssignment => write!(
                f,
                "single-assignment violated: value {} is defined more than once",
                self.value
            ),
            SsaRule::DefineBeforeUse => write!(
                f,
                "define-before-use violated: value {} is used before it is defined",
                self.value
            ),
        }
    }
}

/// Statically check SSA well-formedness over an [`IRModule`] parsed from a
/// mic@3 artifact, for the `mindc verify` surface (RFC 0017, second slice).
///
/// Two properties are enforced:
///
/// 1. **Single-assignment** — every value id (`%N`) is the result of at most
///    one instruction. A second definition of the same id is a violation.
/// 2. **Define-before-use** — every operand `%N` an instruction reads was
///    defined by an *earlier* instruction (function parameters and a region's
///    enclosing-scope values count as already-defined).
///
/// ### Scoping model — dominance-aware
///
/// The check walks the instruction tree in pre-order, but it is **dominance
/// aware** about nested control flow. The earlier linear-program-order slice
/// over-rejected: it let a region interior's definitions leak into the
/// enclosing/sibling scope and checked a region node's own operands (e.g. an
/// `If` merge's `then_val`/`else_val`) *before* recursing into the branch that
/// defines them — flagging programs WITH control flow that actually compile and
/// run deterministically (MIND-Fuzz bug #5).
///
/// The model now mirrors the control-flow graph:
///
/// * A value defined inside a branch / loop body **does not dominate** code
///   after the region (it is not defined on every path / iteration). Each
///   nested region is therefore checked in a **clone** of the enclosing
///   `defined` set so its interior definitions never leak back out.
/// * Only the values the region *exposes as block arguments* of its join block
///   dominate post-region uses: an `If` exposes its post-merge `dst` plus each
///   merge id (`^if_after` block args); a `While` exposes its `live_vars`
///   post-body ids plus its `exit_ids` (`^while_after` block args); a `Region`
///   exposes its `result`. Those — and only those — are inserted into the
///   enclosing scope after the region.
/// * A region node's own forwarded operands are checked where they are
///   *visible*: an `If` merge's `then_val` against the then-branch scope and its
///   `else_val` against the else-branch scope (each a value that dominates that
///   edge), never against the enclosing scope before recursion. `While.init_ids`
///   are genuine enclosing-scope reads and are still checked up front.
///
/// A use that is genuinely not dominated by any definition on a reaching path is
/// still reported; the check only stops *over*-rejecting well-dominated SSA.
///
/// ### Per-function SSA namespaces
///
/// In MIND's IR a function's parameter ids and its body value ids are numbered
/// in a namespace **local to that function** (`src/eval/lower.rs` resets the
/// value counter per `FnDef`). The same numeric id (e.g. `%0`) therefore recurs
/// across functions and even between a function's param `%0` and a top-level
/// `%0` that follows the `FnDef`. To avoid false single-assignment collisions,
/// every `FnDef` body is checked in a **fresh, function-local `defined` scope**
/// (params seeded into it); the module/top-level stream keeps its own scope.
/// A genuine duplicate result id *within one stream* is still rejected because
/// the collision is detected against that stream's own scope.
///
/// Returns `Ok(())` if both properties hold, or the first [`SsaViolation`] in
/// program order otherwise.
pub fn check_ssa_well_formed(module: &IRModule) -> Result<(), SsaViolation> {
    let mut defined: BTreeSet<ValueId> = BTreeSet::new();
    check_ssa_stream(&module.instrs, &mut defined)
}

/// Walk one instruction stream in program order, threading the running
/// `defined` set through nested regions. Operands are checked against
/// definitions seen *earlier* in the walk; each instruction's result is then
/// added (rejecting a duplicate).
fn check_ssa_stream(instrs: &[Instr], defined: &mut BTreeSet<ValueId>) -> Result<(), SsaViolation> {
    for instr in instrs {
        // Dispatch the nested control-flow / region variants first. Each is
        // checked in a CLONE of the enclosing `defined` set (so its interior
        // definitions cannot leak into this or a sibling scope — they do not
        // dominate code outside the region), then only the ids the region
        // exposes as join-block arguments are inserted into `defined`.
        match instr {
            Instr::FnDef { params, body, .. } => {
                // A function body is its own SSA namespace: ids are numbered
                // local to the function (the lowering value counter resets per
                // `FnDef`), so a body `%0` and an enclosing/top-level `%0` are
                // distinct values. Check the body in a FRESH scope seeded with
                // the parameter ids; do NOT thread the enclosing `defined` in
                // (that caused the false single-assignment collision reported by
                // MIND-Fuzz on `scalar_arith.mind`). A genuine duplicate result
                // id inside the body still collides within this fresh scope.
                let mut body_scope: BTreeSet<ValueId> = BTreeSet::new();
                for (_name, pid) in params {
                    // A param id may be repeated across the params list only if
                    // the IR is malformed; reject that as a single-assignment
                    // fault. (The body's own `Param` instruction re-stating a
                    // seeded param id is handled idempotently below.)
                    if !body_scope.insert(*pid) {
                        return Err(SsaViolation {
                            value: *pid,
                            rule: SsaRule::SingleAssignment,
                        });
                    }
                }
                check_ssa_stream(body, &mut body_scope)?;
                // A `FnDef` exposes no value into the enclosing module scope.
                continue;
            }
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs,
                body,
                init_ids,
                live_vars,
                exit_ids,
                ..
            } => {
                // `init_ids` are the pre-loop (enclosing-scope) values fed into
                // the header — genuine reads that must already dominate the
                // loop. Check them against the enclosing scope.
                for id in init_ids {
                    if !defined.contains(id) {
                        return Err(SsaViolation {
                            value: *id,
                            rule: SsaRule::DefineBeforeUse,
                        });
                    }
                }
                // The condition and body live in the loop's own region: they see
                // the enclosing dominators but their definitions must not escape
                // except through the join-block args. Check each in a clone.
                let mut loop_scope = defined.clone();
                check_ssa_stream(cond_instrs, &mut loop_scope)?;
                check_ssa_stream(body, &mut loop_scope)?;
                // Post-loop code references the `^while_after` block args: the
                // `live_vars` post-body ids and the F2 `exit_ids`. Those — and
                // only those — dominate code after the loop. (They decode to
                // empty from a mic@3 artifact; populated on a freshly-lowered
                // module, which is exactly where the false positive bit.)
                for (_name, post_id) in live_vars {
                    defined.insert(*post_id);
                }
                for exit_id in exit_ids {
                    defined.insert(*exit_id);
                }
                continue;
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                dst,
                merges,
                ..
            } => {
                // The condition is evaluated before the branch and sees the
                // enclosing dominators; its result does not escape the region.
                let mut cond_scope = defined.clone();
                check_ssa_stream(cond_instrs, &mut cond_scope)?;
                // Each branch is its own region: it sees the enclosing
                // dominators (plus the just-checked condition) but its
                // definitions do not leak out. A merge's `then_val` is the
                // then-branch incarnation (visible in the then scope); its
                // `else_val` the else-branch incarnation (visible in the else
                // scope). Check each forwarded value where it is actually
                // defined — never against the enclosing scope before recursing,
                // which over-rejected a value defined inside the branch.
                let mut then_scope = cond_scope.clone();
                check_ssa_stream(then_instrs, &mut then_scope)?;
                let mut else_scope = cond_scope;
                check_ssa_stream(else_instrs, &mut else_scope)?;
                for (_merge_id, then_val, else_val) in merges {
                    if !then_scope.contains(then_val) {
                        return Err(SsaViolation {
                            value: *then_val,
                            rule: SsaRule::DefineBeforeUse,
                        });
                    }
                    if !else_scope.contains(else_val) {
                        return Err(SsaViolation {
                            value: *else_val,
                            rule: SsaRule::DefineBeforeUse,
                        });
                    }
                }
                // The `If` exposes its post-merge `dst` plus each merge id
                // (`^if_after` block args) into the enclosing scope; they
                // dominate post-branch uses on every path.
                defined.insert(*dst);
                for (merge_id, _t, _e) in merges {
                    defined.insert(*merge_id);
                }
                continue;
            }
            #[cfg(feature = "std-surface")]
            Instr::Region { body, result, .. } => {
                // A region body is a nested scope; only its `result` (the last
                // body value) escapes. Check the body in a clone, then expose
                // the result.
                let mut region_scope = defined.clone();
                check_ssa_stream(body, &mut region_scope)?;
                defined.insert(*result);
                continue;
            }
            _ => {}
        }

        // Straight-line instruction (no nested region).
        //
        // 1. Define-before-use: every operand must already be defined. For a
        //    mic@3-parsed module the F2 block-argument forwarding ids decode to
        //    empty, so `instruction_operands` returns only genuine in-scope SSA
        //    reads here. `Break`/`Continue` `live` snapshots are reads of values
        //    that dominate the snapshot point — checked here too.
        for operand in instruction_operands(instr) {
            if !defined.contains(&operand) {
                return Err(SsaViolation {
                    value: operand,
                    rule: SsaRule::DefineBeforeUse,
                });
            }
        }

        // 2. Single-assignment: the instruction's own result id (if any) must
        //    not already be defined.
        //
        //    EXCEPTION: a `Param` body instruction re-states an id that the
        //    enclosing `FnDef` already seeded into this function-local scope from
        //    its `params` list (materialized both in `FnDef.params` and as a
        //    leading `Param` instruction, with the SAME id). That is one logical
        //    definition of the parameter, so its insert is idempotent here.
        let is_param = matches!(instr, Instr::Param { .. });
        if let Some(dst) = instruction_dst(instr) {
            if !defined.insert(dst) && !is_param {
                return Err(SsaViolation {
                    value: dst,
                    rule: SsaRule::SingleAssignment,
                });
            }
        }
    }
    Ok(())
}

/// Structured errors returned by the IR verifier.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IrVerifyError {
    /// Multiple instructions attempted to define the same SSA value.
    #[error("duplicate definition for value %{0}")]
    DuplicateDefinition(ValueId),
    /// A value was referenced before it had been defined.
    #[error("use of undefined value %{value} at instruction {instr_index}")]
    UseBeforeDefinition { value: ValueId, instr_index: usize },
    /// The module contains no `Output` instruction.
    #[error("module is missing an Output instruction")]
    MissingOutput,
    /// The module's `next_id` counter does not match the SSA IDs in use.
    #[error("next_id {found} is smaller than required {expected}")]
    NextIdOutOfSync { found: usize, expected: usize },
    /// Operand validation failed (e.g., a negative axis or stride).
    #[error("invalid operand in instruction {instr_index}: {message}")]
    InvalidOperand { instr_index: usize, message: String },
}

/// Verify that an [`IRModule`] is well-formed and deterministic.
///
/// The verifier enforces SSA discipline (unique definitions, no use-before-def),
/// basic operand sanity, and synchronization of the module's `next_id` counter.
/// It returns structured errors instead of panicking on invalid input.
pub fn verify_module(module: &IRModule) -> Result<(), IrVerifyError> {
    let mut defined: BTreeSet<ValueId> = BTreeSet::new();
    let mut saw_output = false;
    let mut max_seen = 0usize;

    for (idx, instr) in module.instrs.iter().enumerate() {
        validate_operands(idx, instr, &defined)?;

        if let Some(dst) = instruction_dst(instr) {
            if !defined.insert(dst) {
                return Err(IrVerifyError::DuplicateDefinition(dst));
            }
            max_seen = max_seen.max(dst.0 + 1);
        }

        if matches!(instr, Instr::Output(_)) {
            saw_output = true;
        }
    }

    // A module is well-formed if it has either at least one `Instr::Output`
    // *or* a non-empty `IRModule.exports` set (RFC 0002 deliverable 1 —
    // an export-only module's contract surface is the export declaration,
    // not an SSA output).
    if !saw_output && module.exports.is_empty() {
        return Err(IrVerifyError::MissingOutput);
    }

    if module.next_id < max_seen {
        return Err(IrVerifyError::NextIdOutOfSync {
            found: module.next_id,
            expected: max_seen,
        });
    }

    Ok(())
}

fn validate_operands(
    instr_index: usize,
    instr: &Instr,
    defined: &BTreeSet<ValueId>,
) -> Result<(), IrVerifyError> {
    let check_defined = |value: ValueId| {
        if !defined.contains(&value) {
            Err(IrVerifyError::UseBeforeDefinition { value, instr_index })
        } else {
            Ok(())
        }
    };

    match instr {
        Instr::ConstI64(_, _) | Instr::ConstF64(_, _) | Instr::ConstTensor(_, _, _, _) => {}
        Instr::BinOp { lhs, rhs, .. } => {
            check_defined(*lhs)?;
            check_defined(*rhs)?;
        }
        Instr::Sum { src, axes, .. } | Instr::Mean { src, axes, .. } => {
            check_defined(*src)?;
            if axes.iter().any(|axis| *axis < 0) {
                return Err(IrVerifyError::InvalidOperand {
                    instr_index,
                    message: "axes must be non-negative".to_string(),
                });
            }
        }
        Instr::Reshape { src, .. }
        | Instr::ExpandDims { src, .. }
        | Instr::Squeeze { src, .. }
        | Instr::Transpose { src, .. }
        | Instr::Relu { src, .. }
        | Instr::Index { src, .. }
        | Instr::Slice { src, .. } => {
            check_defined(*src)?;
        }
        Instr::ReluGrad { grad, src, .. } => {
            check_defined(*grad)?;
            check_defined(*src)?;
        }
        Instr::Dot { a, b, .. } | Instr::MatMul { a, b, .. } => {
            check_defined(*a)?;
            check_defined(*b)?;
        }
        Instr::Conv2d {
            input,
            filter,
            stride_h,
            stride_w,
            ..
        } => {
            check_defined(*input)?;
            check_defined(*filter)?;
            if *stride_h == 0 || *stride_w == 0 {
                return Err(IrVerifyError::InvalidOperand {
                    instr_index,
                    message: "conv2d strides must be positive".to_string(),
                });
            }
        }
        Instr::Conv2dGradInput {
            dy,
            filter,
            stride_h,
            stride_w,
            ..
        } => {
            check_defined(*dy)?;
            check_defined(*filter)?;
            if *stride_h == 0 || *stride_w == 0 {
                return Err(IrVerifyError::InvalidOperand {
                    instr_index,
                    message: "conv2d_grad_input strides must be positive".to_string(),
                });
            }
        }
        Instr::Conv2dGradFilter {
            input,
            dy,
            stride_h,
            stride_w,
            ..
        } => {
            check_defined(*input)?;
            check_defined(*dy)?;
            if *stride_h == 0 || *stride_w == 0 {
                return Err(IrVerifyError::InvalidOperand {
                    instr_index,
                    message: "conv2d_grad_filter strides must be positive".to_string(),
                });
            }
        }
        Instr::Gather {
            src, indices, axis, ..
        } => {
            check_defined(*src)?;
            check_defined(*indices)?;
            if *axis < 0 {
                return Err(IrVerifyError::InvalidOperand {
                    instr_index,
                    message: "axis must be non-negative".to_string(),
                });
            }
        }
        Instr::Output(id) => {
            check_defined(*id)?;
        }
        Instr::FnDef { body, .. } => {
            // FnDef bodies contain their own scope; params are defined within.
            // Verify body instructions have internally consistent definitions.
            let mut body_defined: BTreeSet<ValueId> = BTreeSet::new();
            for (body_idx, body_instr) in body.iter().enumerate() {
                if let Some(dst) = crate::ir::instruction_dst(body_instr) {
                    body_defined.insert(dst);
                }
                // RFC 0005 Gap 1: While loops output their live_vars post-body
                // values into the enclosing scope.  The MLIR emitter threads
                // them as block arguments; the IR verifier must treat those
                // post-body ValueIds as defined in the fn body after the While.
                #[cfg(feature = "std-surface")]
                if let Instr::While {
                    live_vars,
                    exit_ids,
                    ..
                } = body_instr
                {
                    for (_name, post_id) in live_vars {
                        body_defined.insert(*post_id);
                    }
                    // F2: post-loop code references the EXIT ids (^while_after
                    // block args), so they too are defined in fn scope after
                    // the While.
                    for exit_id in exit_ids {
                        body_defined.insert(*exit_id);
                    }
                }
                // F2: an If exposes its merge ids (^if_after block args) plus
                // its `dst` into the enclosing scope after the branch.
                #[cfg(feature = "std-surface")]
                if let Instr::If { merges, .. } = body_instr {
                    for (merge_id, _t, _e) in merges {
                        body_defined.insert(*merge_id);
                    }
                }
                // Check operand references within body scope.
                //
                // Bug 2 fix: previously only `BinOp` operands were validated,
                // so a use-before-def of a value consumed by a `Call` arg,
                // `Return` value, `ArrayLoad`/`Vec*` etc. went undetected.
                // We now validate *every* body instruction's operands against
                // `body_defined`, reusing the same exhaustive operand
                // enumeration as the DCE pass (`instruction_operands`).
                //
                // EXCEPTION: nested control-flow / region / fn instructions
                // (`While`/`If`/`Region`/`FnDef`) carry operands that live in
                // their *own* SSA sub-namespaces (e.g. a fall-through `If`
                // merge value defined inside a branch body, or a `usize::MAX`
                // non-fall-through placeholder). Those are not defined in this
                // fn-body scope, so validating them here would be a false
                // positive — the verifier already treats those nodes as opaque
                // control-flow units (see the `validate_operands` arms above),
                // and their exposed ids are inserted into `body_defined` for
                // subsequent instructions. We therefore skip operand checking
                // for those variants and validate only straight-line ops.
                let is_nested_region = matches!(body_instr, Instr::FnDef { .. });
                #[cfg(feature = "std-surface")]
                let is_nested_region = is_nested_region
                    || matches!(
                        body_instr,
                        Instr::While { .. } | Instr::If { .. } | Instr::Region { .. }
                    );
                if !is_nested_region {
                    for operand in crate::opt::ir_canonical::instruction_operands(body_instr) {
                        if !body_defined.contains(&operand) {
                            return Err(IrVerifyError::UseBeforeDefinition {
                                value: operand,
                                instr_index: body_idx,
                            });
                        }
                    }
                }
            }
        }
        Instr::Call { args, .. } => {
            for arg in args {
                check_defined(*arg)?;
            }
        }
        Instr::Return { value } => {
            if let Some(id) = value {
                check_defined(*id)?;
            }
        }
        Instr::Param { .. } => {
            // Parameters define values; no operands to check.
        }
        Instr::SparseAttr { src, .. } => {
            check_defined(*src)?;
        }
        // RFC 0005 Gap 1: While loop — condition and body each reside in their
        // own sub-module (separate SSA namespaces).  The outer verifier treats
        // the node as an opaque control-flow unit; no use-before-def check
        // at the module level is applicable. Gated.
        //
        // NOTE: This arm is called from the FnDef body-verifier loop (which
        // uses body_defined, not the outer `defined`). We do NOT check operands
        // here because the while body has its own SSA namespace.  The outer
        // check_defined closure references the wrong scope.
        #[cfg(feature = "std-surface")]
        Instr::While { .. } => {}
        // Loop control markers: pure terminators with no operands.
        #[cfg(feature = "std-surface")]
        Instr::Break { .. } | Instr::Continue { .. } => {}
        // RFC 0005 Phase 6.2b Gap 2: array constant — values are literals,
        // no SSA operand references to check.
        #[cfg(feature = "std-surface")]
        Instr::ConstArray { .. } => {}
        // RFC 0005 Phase 6.2b Gap 2: array load — base and index must be defined.
        #[cfg(feature = "std-surface")]
        Instr::ArrayLoad { base, index, .. } => {
            check_defined(*base)?;
            check_defined(*index)?;
        }
        // Phase 6.5 Stage 1a: If — condition, then, and else each reside in
        // their own sub-instruction streams (separate SSA namespaces from the
        // outer scope).  The outer verifier treats the node as an opaque
        // control-flow unit. Gated.
        #[cfg(feature = "std-surface")]
        Instr::If { .. } => {}
        // RFC 0006 Track B: SIMD vector primitives. Each operand is an
        // ordinary SSA value that must be defined before use; the lane
        // count is a compile-time literal, nothing to check there. Gated.
        #[cfg(feature = "std-surface")]
        Instr::VecLoad { base, offset, .. } => {
            check_defined(*base)?;
            check_defined(*offset)?;
        }
        #[cfg(feature = "std-surface")]
        Instr::VecFma { a, b, acc, .. } => {
            check_defined(*a)?;
            check_defined(*b)?;
            check_defined(*acc)?;
        }
        #[cfg(feature = "std-surface")]
        Instr::VecReduceAdd { src, .. } => {
            check_defined(*src)?;
        }
        // RFC 0006 Track B (increment 2) — symmetric / Q16.16 vector
        // primitives. Same operand-defined-before-use discipline. Gated.
        #[cfg(feature = "std-surface")]
        Instr::VecStore {
            src, base, offset, ..
        } => {
            check_defined(*src)?;
            check_defined(*base)?;
            check_defined(*offset)?;
        }
        #[cfg(feature = "std-surface")]
        Instr::VecLoadI32 { base, offset, .. } => {
            check_defined(*base)?;
            check_defined(*offset)?;
        }
        #[cfg(feature = "std-surface")]
        Instr::VecMulAddQ16 { a, b, acc, .. } => {
            check_defined(*a)?;
            check_defined(*b)?;
            check_defined(*acc)?;
        }
        #[cfg(feature = "std-surface")]
        Instr::VecReduceAddI64 { src, .. } => {
            check_defined(*src)?;
        }
        // RFC 0010 Phase A: extern declaration — no SSA operands to verify.
        #[cfg(feature = "std-surface")]
        Instr::ExternFnDecl { .. } => {}
        // RFC 0010 Phase J-A: region block — body instructions reside in
        // their own sub-stream (separate SSA namespace from the outer scope).
        // The outer verifier treats the node as an opaque unit. Gated.
        #[cfg(feature = "std-surface")]
        Instr::Region { .. } => {}
    }

    Ok(())
}
