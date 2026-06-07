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
/// ### Scoping model — linear program order, NOT dominance
///
/// mic@3 carries the full nested `IRModule` tree (`FnDef.body`, `While`
/// cond/body, `If` then/else, `Region.body`) but **not** the F2 block-argument
/// metadata (`If.merges`, `While.exit_ids` are lowering-internal and decode to
/// empty). A true dominance check needs that control-flow-graph structure, so
/// this first slice walks the instruction tree in **pre-order (program order)**
/// and treats an operand as defined if any earlier instruction in that walk —
/// including instructions in an enclosing region — produced it. Region
/// interiors inherit the enclosing `defined` set; their own definitions become
/// visible to subsequent instructions in the same and inner regions. This is a
/// sound *necessary* condition for SSA dominance and an accepted first slice;
/// the dominance-precise check is future work (tracked with the RFC 0017 SMT
/// extension).
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

/// Insert into `scope` exactly the SSA ids an instruction exposes into its
/// **enclosing** scope — the values that subsequent instructions in the same
/// (and inner) regions may legitimately reference.
///
/// This is the SINGLE source of truth for control-flow region exposure, shared
/// by both SSA verifiers ([`check_ssa_stream`] consumer-side and
/// [`verify_module`] in-pipeline) so they can never diverge on which ids a
/// control-flow op makes visible. When a new control-flow op is added, it is
/// taught here once.
///
/// The exposed ids are the region-EXIT / MERGE values that are synthesized at
/// region exit and are NOT defined by any instruction inside the region body:
///
/// * `If`  — the post-merge `dst` plus every F2 merge id (`^if_after` block
///   args, the `merges[i].0`).
/// * `While` — every `live_vars` post-body id plus every `exit_ids` id
///   (`^while_after` block args). The loop-carried values seen after the loop.
/// * `Region` — its `result` id (the last body value, exposed to the enclosing
///   scope).
/// * everything else — its plain `dst` if it produces one (the simple
///   straight-line value case).
///
/// `insert` is idempotent: when an interior recursion already inserted an id
/// (e.g. a `Region.result` produced inside the body), re-exposing it is a no-op
/// and is NOT a single-assignment violation, because the region is one logical
/// producer of that value.
/// Does a branch sub-stream fall through to `^if_after` (i.e. does it reach the
/// merge), or does it terminate early with a `return`?
///
/// This MUST mirror the lowering rule in `src/eval/lower.rs` (§ "A branch that
/// ends in `return` does not fall through to `^if_after`"): a branch whose last
/// instruction is `Return` has its `cf.br` omitted and does NOT pass a merge
/// value. Its slot in the `(merge_id, then_val, else_val)` tuple is a PLACEHOLDER
/// — the *other* branch's value, or a `usize::MAX` sentinel when neither branch
/// assigns the name — and therefore must NOT be validated against this branch's
/// scope (it is not a real edge into the merge). Only a falling-through branch
/// contributes a genuine merge operand that must be defined in its own scope.
#[cfg(feature = "std-surface")]
fn branch_falls_through(instrs: &[Instr]) -> bool {
    !matches!(instrs.last(), Some(Instr::Return { .. }))
}

/// Validate the F2 merge operands of an `If` against PER-BRANCH scopes — the
/// soundness-critical check (issue #24).
///
/// `then_val` is the merged variable's value at the EXIT of the then-branch, so
/// it must be defined in `then_scope` (enclosing ∪ then-branch defs) ONLY; symm-
/// etrically `else_val` must be defined in `else_scope` (enclosing ∪ else-branch
/// defs) ONLY. Checking both against the union (enclosing ∪ then ∪ else) is
/// UNSOUND: a tampered artifact whose `then_val` points at an else-branch-only
/// value would be wrongly accepted, defeating the consumer verifier's purpose.
///
/// Only a FALLING-THROUGH branch contributes a real merge operand (see
/// [`branch_falls_through`]); a returning branch's slot is a placeholder and is
/// skipped. This single helper is shared by both verifiers so they can never
/// diverge. On the first violation it returns `Err(offending_operand)`.
#[cfg(feature = "std-surface")]
fn validate_if_merges(
    then_scope: &BTreeSet<ValueId>,
    else_scope: &BTreeSet<ValueId>,
    then_instrs: &[Instr],
    else_instrs: &[Instr],
    merges: &[(ValueId, ValueId, ValueId)],
) -> Result<(), ValueId> {
    let then_ft = branch_falls_through(then_instrs);
    let else_ft = branch_falls_through(else_instrs);
    for (_merge_id, then_val, else_val) in merges {
        // then_val is only a real edge value when the then-branch falls through;
        // it must dominate the merge via the then-branch scope alone.
        if then_ft && !then_scope.contains(then_val) {
            return Err(*then_val);
        }
        // else_val symmetrically against the else-branch scope alone.
        if else_ft && !else_scope.contains(else_val) {
            return Err(*else_val);
        }
    }
    Ok(())
}

fn expose_region_definitions(instr: &Instr, scope: &mut BTreeSet<ValueId>) {
    match instr {
        #[cfg(feature = "std-surface")]
        Instr::If { dst, merges, .. } => {
            scope.insert(*dst);
            for (merge_id, _then_val, _else_val) in merges {
                scope.insert(*merge_id);
            }
        }
        #[cfg(feature = "std-surface")]
        Instr::While {
            live_vars,
            exit_ids,
            ..
        } => {
            for (_name, post_id) in live_vars {
                scope.insert(*post_id);
            }
            for exit_id in exit_ids {
                scope.insert(*exit_id);
            }
        }
        #[cfg(feature = "std-surface")]
        Instr::Region { result, .. } => {
            scope.insert(*result);
        }
        // Simple-value case: expose the instruction's own result id, if any.
        _ => {
            if let Some(dst) = instruction_dst(instr) {
                scope.insert(dst);
            }
        }
    }
}

/// Walk one instruction stream in program order, threading the running
/// `defined` set through nested regions. Operands are checked against
/// definitions seen *earlier* in the walk; each instruction's result is then
/// added (rejecting a duplicate).
fn check_ssa_stream(instrs: &[Instr], defined: &mut BTreeSet<ValueId>) -> Result<(), SsaViolation> {
    for instr in instrs {
        // Classify region-bearing nodes: their own `dst`/`result` is produced
        // *inside* a sub-stream (e.g. `Region.result` is the last body value,
        // `If.dst` is the post-merge value), so the node-level definition must
        // be inserted only AFTER recursing — never before, which would create a
        // false single-assignment collision with the interior definition.
        let is_region = matches!(instr, Instr::FnDef { .. });
        #[cfg(feature = "std-surface")]
        let is_region = is_region
            || matches!(
                instr,
                Instr::While { .. } | Instr::If { .. } | Instr::Region { .. }
            );

        // 1. Define-before-use: every operand must already be defined in the
        //    enclosing scope *at this point in program order*.
        //
        //    SCOPE ORDERING CAVEAT — an `If`'s F2 merge `then_val`/`else_val`
        //    (returned by `instruction_operands(If)`) are NOT enclosing-scope
        //    reads: each is the value of a variable at the EXIT of its branch,
        //    so it may be defined *inside* the corresponding branch sub-stream
        //    (e.g. `then_result` flowing into the merge). Those operands are
        //    therefore validated AFTER the branch is walked, in step 3, against
        //    the post-branch scope where the branch's definitions are visible —
        //    never here, which would falsely flag a branch-internal value as
        //    use-before-def. Every other instruction (including `While`, whose
        //    only operands are `init_ids` — genuine enclosing-scope pre-loop
        //    values) is checked here in the enclosing scope.
        let defer_operand_check = {
            #[cfg(feature = "std-surface")]
            {
                matches!(instr, Instr::If { .. })
            }
            #[cfg(not(feature = "std-surface"))]
            {
                false
            }
        };
        if !defer_operand_check {
            for operand in instruction_operands(instr) {
                if !defined.contains(&operand) {
                    return Err(SsaViolation {
                        value: operand,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }
            }
        }

        // 2. Single-assignment for straight-line ops: the instruction's own
        //    result id (if any) must not already be defined. Region nodes defer
        //    this to step 4.
        //
        //    EXCEPTION: a `Param` body instruction re-states an id that the
        //    enclosing `FnDef` already seeded into this function-local scope
        //    from its `params` list (the parameter is materialized both in the
        //    `FnDef.params` list and as a leading `Param` instruction in the
        //    body, with the SAME id). That is one logical definition of the
        //    parameter, not a duplicate, so its insert is idempotent here.
        if !is_region {
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

        // 3. Recurse into nested regions. Parameters are definitions visible to
        //    the body; the body sees the enclosing scope plus its own earlier
        //    definitions (pre-order program-order visibility).
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
                    // seeded param id is handled idempotently in `check_ssa_stream`.)
                    if !body_scope.insert(*pid) {
                        return Err(SsaViolation {
                            value: *pid,
                            rule: SsaRule::SingleAssignment,
                        });
                    }
                }
                check_ssa_stream(body, &mut body_scope)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                check_ssa_stream(cond_instrs, defined)?;
                check_ssa_stream(body, defined)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                merges,
                ..
            } => {
                // The condition is evaluated in the enclosing scope before the
                // branch split, so its defs are visible to both branches.
                check_ssa_stream(cond_instrs, defined)?;

                // SOUNDNESS (issue #24): each branch is walked in its OWN scoped
                // copy of `defined` so that the then-branch's interior defs are
                // NOT visible when validating the else-branch's merge operand,
                // and vice versa. `then_val` is the merged var's value at the
                // exit of the then-branch and so must dominate via the then-scope
                // ALONE; `else_val` via the else-scope alone. Validating both
                // against a single merged (enclosing ∪ then ∪ else) scope would
                // wrongly accept a tampered artifact whose `then_val` points at an
                // else-branch-only value — exactly the cross-branch forgery this
                // consumer verifier exists to catch.
                let mut then_scope = defined.clone();
                check_ssa_stream(then_instrs, &mut then_scope)?;
                let mut else_scope = defined.clone();
                check_ssa_stream(else_instrs, &mut else_scope)?;

                if let Err(operand) =
                    validate_if_merges(&then_scope, &else_scope, then_instrs, else_instrs, merges)
                {
                    return Err(SsaViolation {
                        value: operand,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }

                // Merge both branches' interior defs back into the enclosing
                // scope so straight-line code AFTER the `If` (which the lowering
                // only lets reference the dominating merge ids — exposed in
                // step 4 — but whose ids we keep visible to avoid false
                // use-before-def on any value the codec preserved) is checked
                // against the post-`If` program state. This is the same union the
                // previous threaded walk produced; only the MERGE-operand check
                // above is now per-branch.
                defined.extend(then_scope);
                defined.extend(else_scope);
            }
            #[cfg(feature = "std-surface")]
            Instr::Region { body, .. } => {
                check_ssa_stream(body, defined)?;
            }
            _ => {}
        }

        // 4. Expose a region node's exit/merge ids into the enclosing scope via
        //    the shared `expose_region_definitions` helper. This is the SAME
        //    helper `verify_module` uses, so the two verifiers can never diverge
        //    on control-flow exposure. It adds the synthesized region-exit ids
        //    (`If.merges`, `While.exit_ids`/`live_vars` post-body, `Region.result`,
        //    plus the node `dst`) that post-region code legitimately references
        //    but which are NOT defined by any instruction inside the region body.
        //    `insert` is idempotent — a no-op for an interior id, a fresh
        //    definition for a node-level merge/exit id — so this is NOT a
        //    single-assignment violation (the region is one logical producer).
        if is_region {
            expose_region_definitions(instr, defined);
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

/// Walk a branch sub-stream for the in-pipeline verifier, threading the exposed
/// region/straight-line definitions into a SCOPED copy seeded from the enclosing
/// scope, and recursively validating any nested `If` merge operands encountered
/// (so a cross-branch forgery nested inside a branch is still caught). Returns
/// the branch's post-walk scope (enclosing ∪ this branch's exposed defs), which
/// the caller uses to validate the branch's own merge operand.
///
/// This mirrors the exposure model of the consumer-side `check_ssa_stream`
/// (using the SHARED `expose_region_definitions`) so `verify_module` and
/// `check_ssa_well_formed` agree on which ids a branch makes visible — and now
/// also agree, per-branch, on `If` merge soundness (issue #24).
#[cfg(feature = "std-surface")]
fn collect_branch_scope(
    instrs: &[Instr],
    enclosing: &BTreeSet<ValueId>,
) -> Result<BTreeSet<ValueId>, IrVerifyError> {
    let mut scope = enclosing.clone();
    for instr in instrs {
        // Recurse FIRST for an `If` so its branches' interior defs are exposed
        // (and its nested merges validated) before we expose the node-level ids.
        if let Instr::If {
            then_instrs,
            else_instrs,
            merges,
            ..
        } = instr
        {
            validate_if_node(then_instrs, else_instrs, merges, &scope)?;
        }
        expose_region_definitions(instr, &mut scope);
    }
    Ok(scope)
}

/// Validate one `If` node's F2 merge operands against PER-BRANCH scopes for the
/// in-pipeline verifier, recursing into nested `If`s. The soundness core of the
/// #24 fix on the `verify_module` side: it builds each branch's own scope via
/// [`collect_branch_scope`] and defers the actual operand test to the SHARED
/// [`validate_if_merges`], so both verifiers reject identical cross-branch
/// forgeries.
#[cfg(feature = "std-surface")]
fn validate_if_node(
    then_instrs: &[Instr],
    else_instrs: &[Instr],
    merges: &[(ValueId, ValueId, ValueId)],
    enclosing: &BTreeSet<ValueId>,
) -> Result<(), IrVerifyError> {
    let then_scope = collect_branch_scope(then_instrs, enclosing)?;
    let else_scope = collect_branch_scope(else_instrs, enclosing)?;
    if let Err(operand) =
        validate_if_merges(&then_scope, &else_scope, then_instrs, else_instrs, merges)
    {
        return Err(IrVerifyError::UseBeforeDefinition {
            value: operand,
            instr_index: 0,
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
                // SOUNDNESS (issue #24): a nested `If` inside this fn body carries
                // F2 `merges` whose `then_val`/`else_val` must be validated against
                // PER-BRANCH scopes built from the enclosing (fn-body) scope as it
                // stands NOW — before the `If`'s own merge ids are exposed below.
                // `validate_if_node` recurses into both branches (and any deeper
                // nested `If`s) and defers to the shared `validate_if_merges`.
                #[cfg(feature = "std-surface")]
                if let Instr::If {
                    then_instrs,
                    else_instrs,
                    merges,
                    ..
                } = body_instr
                {
                    validate_if_node(then_instrs, else_instrs, merges, &body_defined)?;
                }
                // Expose every id this instruction makes visible to the enclosing
                // (fn-body) scope via the SHARED `expose_region_definitions`
                // helper — the same one the consumer-side `check_ssa_stream` uses,
                // so the two verifiers can never diverge. For a straight-line op
                // this inserts its plain `dst`; for a `While` it adds the
                // `live_vars` post-body + `exit_ids` (^while_after block args);
                // for an `If` it adds the `merges` ids (^if_after block args) plus
                // `dst`; for a `Region` its `result`. RFC 0005 Gap 1 + F2.
                expose_region_definitions(body_instr, &mut body_defined);
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
        // outer scope). The node is otherwise an opaque control-flow unit, BUT
        // its F2 `merges` carry `then_val`/`else_val` operands that MUST be
        // validated against PER-BRANCH scopes (issue #24 soundness): `then_val`
        // against (enclosing ∪ then-defs), `else_val` against (enclosing ∪
        // else-defs). Validating both against the union would wrongly accept a
        // cross-branch forgery. `defined` here is the enclosing scope. Gated.
        #[cfg(feature = "std-surface")]
        Instr::If {
            then_instrs,
            else_instrs,
            merges,
            ..
        } => {
            validate_if_node(then_instrs, else_instrs, merges, defined)?;
        }
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
