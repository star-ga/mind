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
/// * `While` — every `exit_ids` id (`^while_after` block args): the
///   loop-carried values seen AFTER the loop, which the lowering rebinds each
///   carried var to. The `live_vars` post-body ids are NOT exposed: they are the
///   loop's BACK-EDGE operands, defined inside the body and validated there (the
///   loop analog of `If`'s `then_val`/`else_val`), never visible to post-loop
///   code.
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
/// This MUST mirror the lowering rule in `src/eval/lower.rs` (`branch_terminates`,
/// § "A branch that ends in a block terminator does not fall through to
/// `^if_after`") and the MLIR emitter (`instr_is_block_terminator` in
/// `src/mlir/lowering.rs`): a branch whose last instruction is a terminator has
/// its `cf.br` omitted and does NOT pass a merge value. Its slot in the
/// `(merge_id, then_val, else_val)` tuple is a PLACEHOLDER — the *other* branch's
/// value, or a `usize::MAX` sentinel when neither branch assigns the name — and
/// therefore must NOT be validated against this branch's scope (it is not a real
/// edge into the merge). Only a falling-through branch contributes a genuine merge
/// operand that must be defined in its own scope.
///
/// `Return` always terminates. Under `std-surface`, a `match`-arm/branch body that
/// exits or re-tests the enclosing loop ends in `Break`/`Continue`, which likewise
/// emits a `cf.br` to `^while_after`/`^while_header` and NOT to `^if_after` — so it
/// too does not fall through. Recognizing only `Return` here (while the lowering
/// and MLIR emitter recognized `Break`/`Continue`) made the verifier reject a
/// well-formed `if { outer = … } else { continue }` inside a loop: it validated the
/// terminating branch's dead placeholder edge (the *other* branch's value, defined
/// in the *other* scope) and flagged it as an undefined operand (E3001). All three
/// surfaces must classify terminators identically.
#[cfg(feature = "std-surface")]
fn branch_falls_through(instrs: &[Instr]) -> bool {
    !matches!(
        instrs.last(),
        Some(Instr::Return { .. }) | Some(Instr::Break { .. }) | Some(Instr::Continue { .. })
    )
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
        Instr::While { exit_ids, .. } => {
            // ONLY the `exit_ids` (`^while_after` block args) are region-EXIT
            // values visible to post-loop code; the lowering rebinds every
            // loop-carried var to its `exit_id` after the loop
            // (`src/eval/lower.rs`: "code AFTER the loop is rebound to these exit
            // ids instead of the body-internal post_id`s").
            //
            // The `live_vars` post-body ids are NOT exposed here: they are the
            // loop's BACK-EDGE operands (the value each carried var holds at the
            // end of the body, fed to the header on the next iteration), defined
            // INSIDE the body and consumed only by the loop itself. Exposing them
            // to the enclosing scope would (a) leak body-internal ids to code that
            // can never legitimately reference them and (b) mask a forged
            // `live_vars` id by silently treating it as "defined". Their
            // definedness is instead VALIDATED against the body scope where they
            // are produced (the `While` arms of `check_ssa_stream` / `verify_module`),
            // exactly as `If` validates its `then_val`/`else_val` per-branch.
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
                cond_id,
                cond_instrs,
                body,
                live_vars,
                ..
            } => {
                // Snapshot the enclosing scope BEFORE threading in the loop's own
                // defs — the back-edge validation below needs the pre-loop state
                // to tell a genuine BODY definition apart from an enclosing value.
                let enclosing_before = defined.clone();
                // Walk cond + body for SSA discipline inside the loop's own
                // namespace (threaded through the enclosing `defined`, as before).
                check_ssa_stream(cond_instrs, defined)?;
                check_ssa_stream(body, defined)?;
                // SOUNDNESS (finding #9): the loop's `cond_id` (the boolean the
                // header tests each iteration) is a REGION operand omitted from
                // `instruction_operands` on purpose — it lives in the loop's own
                // namespace (it may read loop-carried / back-edge incarnations), so
                // checking it against the ENCLOSING scope would be a false positive.
                // Validate it here against the loop namespace (enclosing ∪ cond ∪
                // body): a genuine cond result is produced by `cond_instrs`; a
                // crafted mic@3 with a dangling `cond_id` is defined NOWHERE and is
                // rejected.
                if !defined.contains(cond_id) {
                    return Err(SsaViolation {
                        value: *cond_id,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }
                // SOUNDNESS (loop-carry): each `live_vars` post-body id is the
                // BACK-EDGE value of a loop-carried var. It must be DEFINED in
                // `enclosing ∪ cond ∪ body` (a body def, or an enclosing value a
                // rebinding like `a = b` re-points the var at). An id defined
                // NOWHERE is a forgery. Validate via the SHARED helper so this
                // verifier and `verify_module` reject identical loop-carry
                // forgeries — the loop analog of `validate_if_merges` (issue #24).
                if let Err(post_id) =
                    validate_while_backedges(cond_instrs, body, live_vars, &enclosing_before)
                {
                    return Err(SsaViolation {
                        value: post_id,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_id,
                cond_instrs,
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                merges,
                ..
            } => {
                // The condition is evaluated in the enclosing scope before the
                // branch split, so its defs are visible to both branches.
                check_ssa_stream(cond_instrs, defined)?;

                // SOUNDNESS (finding #9): `cond_id` is a REGION operand omitted from
                // `instruction_operands` on purpose. It is produced by `cond_instrs`
                // in the enclosing ∪ cond scope; a crafted mic@3 with a dangling
                // `cond_id` is defined nowhere and is rejected here.
                if !defined.contains(cond_id) {
                    return Err(SsaViolation {
                        value: *cond_id,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }

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
                // SOUNDNESS (finding #9): `then_result` is the if-expression's value
                // on the then path — a BRANCH-INTERNAL region operand (omitted from
                // `instruction_operands`), always produced inside the then-branch by
                // the lowering (at minimum its `ConstI64` placeholder). Validate it
                // against `then_scope` (enclosing ∪ cond ∪ then) ONLY; a dangling
                // `then_result` in a crafted artifact is rejected, with no false
                // positive on the branch-internal value.
                if !then_scope.contains(then_result) {
                    return Err(SsaViolation {
                        value: *then_result,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }
                let mut else_scope = defined.clone();
                check_ssa_stream(else_instrs, &mut else_scope)?;
                // Symmetric to `then_result`: validate `else_result` against
                // `else_scope` (enclosing ∪ cond ∪ else) alone.
                if !else_scope.contains(else_result) {
                    return Err(SsaViolation {
                        value: *else_result,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }

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
            Instr::Region { body, result, .. } => {
                check_ssa_stream(body, defined)?;
                // SOUNDNESS: `result` is the region's exit value — documented as
                // the "last body value" (`ir/mod.rs`) and exposed to the enclosing
                // scope by `expose_region_definitions` (step 4). Nothing enforced
                // that it is actually PRODUCED by the body: a forged mic@3 artifact
                // could set `result` to an id defined nowhere, and it would still
                // be exposed as "defined" — certifying an undefined-value module as
                // well-formed. Validate it against the post-body scope (enclosing ∪
                // body), exactly as the `If` arm validates `then_result`/
                // `else_result` against their branch scopes. A dangling / cross /
                // undefined region result is now rejected fail-CLOSED.
                if !defined.contains(result) {
                    return Err(SsaViolation {
                        value: *result,
                        rule: SsaRule::DefineBeforeUse,
                    });
                }
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

/// In-pipeline analog of [`check_ssa_stream`]: walk one instruction stream in
/// program order, threading `defined` through nested regions, and validating the
/// SAME SSA discipline (define-before-use + single-assignment) but returning the
/// structured [`IrVerifyError`] the pipeline consumes instead of an
/// [`SsaViolation`].
///
/// This is a FAITHFUL mirror of `check_ssa_stream` — same classification, same
/// deferred `If`-merge check, same per-branch scopes, same shared helpers
/// (`validate_while_backedges`, `validate_if_merges`, `expose_region_definitions`).
/// Keeping it structurally identical is what guarantees `verify_module` and
/// `check_ssa_well_formed` can never disagree on which IRs are SSA-well-formed
/// (the differential gates in `tests/verify_ssa.rs`). Before this existed,
/// `verify_module` walked ONLY the top-level stream and the `FnDef` body,
/// treating a `While`/`If`/`Region` interior as an opaque unit — so a
/// duplicate-def or undefined operand INSIDE a loop/branch body passed clean
/// (finding #123, one region deeper).
fn validate_ssa_stream(
    instrs: &[Instr],
    defined: &mut BTreeSet<ValueId>,
) -> Result<(), IrVerifyError> {
    for (idx, instr) in instrs.iter().enumerate() {
        // Region-bearing nodes produce their own `dst`/`result` INSIDE a
        // sub-stream, so the node-level definition is exposed only after the
        // recursion (step 4), never before (which would false-collide with the
        // interior def).
        let is_region = matches!(instr, Instr::FnDef { .. });
        #[cfg(feature = "std-surface")]
        let is_region = is_region
            || matches!(
                instr,
                Instr::While { .. } | Instr::If { .. } | Instr::Region { .. }
            );

        // 1. Define-before-use. Deferred for `If`: its `merges`' `then_val` /
        //    `else_val` (yielded by `instruction_operands(If)`) are branch-EXIT
        //    values validated per-branch in step 3, not enclosing-scope reads.
        //    Every other instruction — including `While` (`init_ids` only) and
        //    `Break`/`Continue` (their `live` snapshot ids) — is checked here.
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
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: operand,
                        instr_index: idx,
                    });
                }
            }
        }

        // 2. Single-assignment for straight-line ops. `Param` re-stating a seeded
        //    id is idempotent (mirrors `check_ssa_stream`).
        if !is_region {
            let is_param = matches!(instr, Instr::Param { .. });
            if let Some(dst) = instruction_dst(instr) {
                if !defined.insert(dst) && !is_param {
                    return Err(IrVerifyError::DuplicateDefinition(dst));
                }
            }
        }

        // 3. Recurse into nested regions.
        match instr {
            Instr::FnDef { body, .. } => {
                // Fresh, function-local namespace (the lowering resets the value
                // counter per `FnDef`). Deliberately NOT seeded with the param ids:
                // real fn bodies materialize each parameter as a leading
                // `Instr::Param` that defines its id, so an empty seed suffices and
                // matches the previously-shipped `verify_module` FnDef behavior
                // (param-seeding would newly reject a hand-built body that re-uses a
                // param id as a straight-line `dst` — a synthetic case `fresh()`
                // never produces, guarded by tests/verify_audit.rs).
                let mut body_scope: BTreeSet<ValueId> = BTreeSet::new();
                validate_ssa_stream(body, &mut body_scope)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_id,
                cond_instrs,
                body,
                live_vars,
                ..
            } => {
                let enclosing_before = defined.clone();
                validate_ssa_stream(cond_instrs, defined)?;
                validate_ssa_stream(body, defined)?;
                // `cond_id` lives in the loop namespace (enclosing ∪ cond ∪ body);
                // a dangling one in a crafted artifact is defined nowhere.
                if !defined.contains(cond_id) {
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: *cond_id,
                        instr_index: idx,
                    });
                }
                // Loop-carry back-edge soundness (the loop analog of the `If`
                // merge check), via the SHARED helper.
                if let Err(post_id) =
                    validate_while_backedges(cond_instrs, body, live_vars, &enclosing_before)
                {
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: post_id,
                        instr_index: idx,
                    });
                }
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_id,
                cond_instrs,
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                merges,
                ..
            } => {
                validate_ssa_stream(cond_instrs, defined)?;
                if !defined.contains(cond_id) {
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: *cond_id,
                        instr_index: idx,
                    });
                }
                // Per-branch scopes (issue #24): the then-branch's interior defs
                // must NOT be visible when validating the else-branch merge value,
                // and vice versa.
                let mut then_scope = defined.clone();
                validate_ssa_stream(then_instrs, &mut then_scope)?;
                if !then_scope.contains(then_result) {
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: *then_result,
                        instr_index: idx,
                    });
                }
                let mut else_scope = defined.clone();
                validate_ssa_stream(else_instrs, &mut else_scope)?;
                if !else_scope.contains(else_result) {
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: *else_result,
                        instr_index: idx,
                    });
                }
                if let Err(operand) =
                    validate_if_merges(&then_scope, &else_scope, then_instrs, else_instrs, merges)
                {
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: operand,
                        instr_index: idx,
                    });
                }
                // Merge both branches' interior defs back into the enclosing scope
                // for straight-line code after the `If` (same union the shared
                // consumer walk produces).
                defined.extend(then_scope);
                defined.extend(else_scope);
            }
            #[cfg(feature = "std-surface")]
            Instr::Region { body, result, .. } => {
                validate_ssa_stream(body, defined)?;
                // Mirror of the `check_ssa_stream` Region guard (and the symmetric
                // `If` then_result/else_result checks above): the region `result`
                // must be produced by the body's own instructions, not a dangling
                // id a forged artifact points nowhere. Exposed as "defined" by
                // `expose_region_definitions` otherwise — a verifier fail-open.
                if !defined.contains(result) {
                    return Err(IrVerifyError::UseBeforeDefinition {
                        value: *result,
                        instr_index: idx,
                    });
                }
            }
            _ => {}
        }

        // 4. Expose the region node's exit/merge ids into the enclosing scope via
        //    the SAME shared helper `check_ssa_stream` uses.
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

    /// Step D (AGGREGATE_TYPE_INVARIANT / Amendment 1) — a canonical
    /// `value_types` entry for a ValueId disagrees with the intrinsic type of the
    /// `ConstDenseTensor` node that also defines it. Two authorities may not
    /// disagree; this is fail-closed rather than silently preferring one.
    #[error("aggregate type conflict for value %{value}: {message}")]
    AggregateTypeConflict { value: ValueId, message: String },

    /// Step D — a `ConstDenseTensor` node is not a valid FIXED aggregate: its
    /// shape has a non-`Known` dim (e.g. a `Sym` dim decoded from a hostile mic@3
    /// artifact) or its `data` length disagrees with the shape's element product.
    /// Rejected fail-closed so the canonical aggregate-type reconstruction can
    /// never invent a confident `Fixed(N)` from an unknown / inconsistent extent.
    #[error("malformed ConstDenseTensor for value %{value}: {message}")]
    MalformedConstDenseTensor { value: ValueId, message: String },
}

/// Verify that an [`IRModule`] is well-formed and deterministic.
///
/// The verifier enforces SSA discipline (unique definitions, no use-before-def),
/// basic operand sanity, and synchronization of the module's `next_id` counter.
/// It returns structured errors instead of panicking on invalid input.
/// Step D — recursively verify `ConstDenseTensor` well-formedness across the FULL
/// nested instruction tree (FnDef / While / If / Region bodies), so a malformed
/// node buried in a control-flow body cannot evade the fail-closed check. Each
/// node's shape must be a valid FIXED extent and its `data` length must equal the
/// shape's element product. The check is per-node intrinsic; SSA namespaces do
/// not matter (only the `dst` is read, and only for the error message).
#[cfg(feature = "std-surface")]
fn check_const_dense_wellformed(instrs: &[Instr]) -> Result<(), IrVerifyError> {
    for instr in instrs {
        match instr {
            Instr::ConstDenseTensor {
                dst, shape, data, ..
            } => match crate::ir::dense_shape_len(shape) {
                None => {
                    return Err(IrVerifyError::MalformedConstDenseTensor {
                        value: *dst,
                        message: "shape is not a fully-known fixed extent".to_string(),
                    });
                }
                Some(n) if n as usize != data.len() => {
                    return Err(IrVerifyError::MalformedConstDenseTensor {
                        value: *dst,
                        message: format!("shape element count {n} != data length {}", data.len()),
                    });
                }
                Some(_) => {}
            },
            Instr::FnDef { body, .. } | Instr::Region { body, .. } => {
                check_const_dense_wellformed(body)?;
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                check_const_dense_wellformed(cond_instrs)?;
                check_const_dense_wellformed(body)?;
            }
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                check_const_dense_wellformed(cond_instrs)?;
                check_const_dense_wellformed(then_instrs)?;
                check_const_dense_wellformed(else_instrs)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Step D — validate the `value_types`-vs-node consistency for EVERY SSA scope:
/// the module (its top-level stream + module table) and, recursively, each
/// function (its body + its OWN table). Function bodies are separate namespaces,
/// so a fn-local aggregate is checked against the FUNCTION's scope, never the
/// module's — this is what makes a `%N` in `f` and a `%N` in `g` independently
/// typed. Only `FnDef` opens a new scope; nested `If`/`While`/`Region` share the
/// enclosing one and are handled by `canonical_array_type_in`'s recursive node
/// search. Dormant while all tables are empty (S1/S2a); load-bearing from S3.
#[cfg(feature = "std-surface")]
fn check_value_types_scope(
    instrs: &[Instr],
    table: &std::collections::BTreeMap<ValueId, crate::ir::ArrayType>,
) -> Result<(), IrVerifyError> {
    for (vid, _at) in table {
        match crate::ir::canonical_array_type_in(instrs, table, *vid) {
            Ok(_) => {} // an explicit entry with no conflict resolves to Ok(Some(entry))
            Err(crate::ir::AggregateTypeError::Conflict { table: t, node }) => {
                return Err(IrVerifyError::AggregateTypeConflict {
                    value: *vid,
                    message: format!(
                        "explicit value_types entry {t:?} disagrees with the \
                         ConstDenseTensor node type {node:?} for the same value"
                    ),
                });
            }
            Err(crate::ir::AggregateTypeError::MalformedConstDense) => {
                return Err(IrVerifyError::MalformedConstDenseTensor {
                    value: *vid,
                    message: "value_types entry references a malformed ConstDenseTensor"
                        .to_string(),
                });
            }
        }
    }
    // Each function is its OWN scope: find EVERY `FnDef` reachable in THIS scope's
    // stream — including one buried inside an `If`/`While`/`Region` body (those
    // share this scope, so we must descend them to reach a nested `FnDef`) — and
    // check each against its own body+table. Without descending control flow, a
    // hostile mic@3 artifact (S2b+) could bury a `FnDef` with a lying `value_types`
    // table inside a `Region` and evade the conflict check.
    check_nested_fndef_tables(instrs)?;
    Ok(())
}

/// Step D — visit EVERY `FnDef` reachable within one SSA scope's instruction
/// stream (descending same-scope `If`/`While`/`Region` bodies) and validate each
/// function's own `value_types` table against its own scope via
/// `check_value_types_scope` (which in turn hunts FnDefs nested inside THAT
/// function). Descent stops implicitly at a found `FnDef`: we hand its body to
/// `check_value_types_scope`, not back to this hunt, so a callee's scope is never
/// merged into the caller's.
#[cfg(feature = "std-surface")]
fn check_nested_fndef_tables(instrs: &[Instr]) -> Result<(), IrVerifyError> {
    for instr in instrs {
        match instr {
            Instr::FnDef {
                body, value_types, ..
            } => {
                check_value_types_scope(body, value_types)?;
            }
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                check_nested_fndef_tables(cond_instrs)?;
                check_nested_fndef_tables(then_instrs)?;
                check_nested_fndef_tables(else_instrs)?;
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                check_nested_fndef_tables(cond_instrs)?;
                check_nested_fndef_tables(body)?;
            }
            Instr::Region { body, .. } => {
                check_nested_fndef_tables(body)?;
            }
            _ => {}
        }
    }
    Ok(())
}

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
        }
        // `next_id` must cover EVERY id this instruction contributes to the
        // top-level stream — not just `instruction_dst`, which understates a
        // control-flow node (see `instruction_id_bound`: it folds in
        // `While.exit_ids`, `If.merges[i].0`, `If.branch_bindings`, and the read
        // ids from `for_each_operand`). Using the dst alone let a top-level
        // control-flow node carry a `fresh()`-minted id ABOVE `max_seen`, so a
        // `next_id` lowered to that understated max passed the sync check while
        // sitting below a live id.
        max_seen = max_seen.max(crate::opt::ir_canonical::instruction_id_bound(instr));

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

    // Step D — ConstDenseTensor well-formedness (fail closed on a hostile mic@3
    // artifact): every const-dense node — at ANY nesting depth (FnDef / While /
    // If / Region bodies, which the mic@3 parser accepts and lowering itself
    // re-emits into fn bodies) — must have a valid FIXED extent (each dim
    // `Known`, product non-overflowing) and an element count equal to
    // `data.len()`. Without this a `Sym`-dim / mismatched node buried inside a
    // loop body would evade the check and let the canonical lookup guess a
    // `Fixed(N)`. The check is per-node intrinsic (shape vs data.len), so a
    // function's separate SSA id-numbering does not matter.
    #[cfg(feature = "std-surface")]
    check_const_dense_wellformed(&module.instrs)?;

    // Step D — AGGREGATE_TYPE_INVARIANT well-formedness for EVERY scope (dormant in
    // S1/S2a: every table is empty, so this is a no-op; load-bearing from S3). The
    // module scope and each function scope are checked against THEIR OWN
    // instruction stream + table, so a fn-local aggregate is never validated
    // against — nor able to conflict with — a same-numbered value in the module or
    // another function. `canonical_array_type_in` returns `Err` (never `Ok(None)`)
    // on a conflict or malformed node — a corruption may not masquerade as absence.
    #[cfg(feature = "std-surface")]
    check_value_types_scope(&module.instrs, &module.value_types)?;

    Ok(())
}

/// Validate a `While` node's loop-carried back-edge values — the soundness
/// core of the loop analog of the #24 `If` fix, SHARED by both verifiers so
/// they can never diverge.
///
/// Each `live_vars` post-body id is the value a loop-carried var holds at the
/// END of the body — the back-edge value forwarded to the loop header on the
/// next iteration. It must be an SSA value that is DEFINED and dominates the
/// back-edge: either a definition produced INSIDE the loop (`cond` ∪ `body`) or
/// a value live from the ENCLOSING (pre-loop) scope. The latter is legitimate
/// because a plain rebinding such as `a = b` re-points a carried var at another
/// already-live id without emitting a new instruction (e.g. `a`'s back-edge
/// becomes `b`'s pre-loop id), and a no-op `s = s` keeps the init id.
///
/// What is NOT legitimate — and what this check rejects — is a `live_vars`
/// back-edge id that is defined NOWHERE in `enclosing ∪ cond ∪ body`: a forged
/// or dangling id whose value the loop header would read out of thin air (a
/// use-before-def on the back-edge). Pre-fix the verifier exposed the post-body
/// ids unconditionally and never checked them, so such a forgery passed clean.
///
/// Returns `Err(offending_post_id)` on the first undefined back-edge id.
#[cfg(feature = "std-surface")]
fn validate_while_backedges(
    cond_instrs: &[Instr],
    body: &[Instr],
    live_vars: &[(String, ValueId)],
    enclosing: &BTreeSet<ValueId>,
) -> Result<(), ValueId> {
    // Reachable = everything defined before the loop, plus every id the cond and
    // body produce. A genuine back-edge value is reachable in this set.
    let mut reachable = enclosing.clone();
    for instr in cond_instrs {
        expose_region_definitions(instr, &mut reachable);
    }
    for instr in body {
        expose_region_definitions(instr, &mut reachable);
    }
    for (_name, post_id) in live_vars {
        if !reachable.contains(post_id) {
            return Err(*post_id);
        }
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
        Instr::ConstI64(_, _)
        | Instr::ConstF64(_, _)
        | Instr::ConstTensor(_, _, _, _)
        | Instr::ConstDenseTensor { .. } => {}
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
        Instr::FnDef { .. } => {
            // A function body is its own SSA namespace. Delegate to the shared
            // recursive stream validator, which walks the body — and RECURSES
            // into any nested `While`/`If`/`Region` interiors (finding #123, one
            // region deeper) — applying define-before-use + single-assignment and
            // the shared per-branch merge / loop-carry back-edge soundness checks.
            // The `FnDef` arm inside `validate_ssa_stream` builds the fresh
            // param-seeded body scope, so the scope passed here is only used for
            // this node's (operand-less, dst-less) top-level slot.
            let mut scope: BTreeSet<ValueId> = BTreeSet::new();
            validate_ssa_stream(std::slice::from_ref(instr), &mut scope)?;
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
        // RFC 0005 Gap 1: While loop — condition and body reside in their own
        // sub-streams (loop-local SSA namespace that still reads enclosing-scope
        // ids). Delegate to the shared recursive validator so a duplicate-def or
        // undefined operand INSIDE `cond_instrs`/`body` is caught (finding #123,
        // one region deeper), the `cond_id` is validated, and the loop-carry
        // back-edge ids are checked via `validate_while_backedges`. Seeded from a
        // clone of the enclosing `defined` (the loop body legitimately reads
        // pre-loop values); the clone keeps top-level exposure unchanged
        // (`verify_module` still exposes only the node `dst`). Gated.
        #[cfg(feature = "std-surface")]
        Instr::While { .. } => {
            let mut scope = defined.clone();
            validate_ssa_stream(std::slice::from_ref(instr), &mut scope)?;
        }
        // Loop control markers: NOT operand-free. Break/Continue snapshot the
        // CURRENT value of every in-scope var in `live` (`for_each_operand`
        // yields these ids); the lowering forwards them as `^while_after` /
        // `^while_header` block-args, so each is a genuine operand that must be
        // defined. Validate them against the enclosing scope. Gated.
        #[cfg(feature = "std-surface")]
        Instr::Break { live } | Instr::Continue { live } => {
            for (_name, id) in live {
                check_defined(*id)?;
            }
        }
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
        // Phase 6.5 Stage 1a: If — condition, then, and else reside in their own
        // sub-instruction streams. Delegate to the shared recursive validator: it
        // walks each branch (catching a duplicate-def / undefined operand INSIDE
        // a branch body — finding #123, one region deeper), validates the
        // `cond_id`, and enforces the issue #24 per-branch `merges` soundness
        // (`then_val` against enclosing ∪ then only, `else_val` against enclosing
        // ∪ else only) via the shared `validate_if_merges`. Seeded from a clone of
        // the enclosing `defined`; top-level exposure is unchanged (`verify_module`
        // still exposes only the node `dst`). Gated.
        #[cfg(feature = "std-surface")]
        Instr::If { .. } => {
            let mut scope = defined.clone();
            validate_ssa_stream(std::slice::from_ref(instr), &mut scope)?;
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
        // RFC 0010 Phase J-A: region block — body instructions reside in their
        // own sub-stream (still reading enclosing-scope ids). Delegate to the
        // shared recursive validator so a duplicate-def / undefined operand inside
        // the region body is caught. Seeded from a clone of the enclosing scope.
        // Gated.
        #[cfg(feature = "std-surface")]
        Instr::Region { .. } => {
            let mut scope = defined.clone();
            validate_ssa_stream(std::slice::from_ref(instr), &mut scope)?;
        }
    }

    Ok(())
}

/// Step D S1 — the canonical aggregate-type invariant infrastructure: the
/// `canonical_array_type` lookup (which distinguishes ABSENCE from CORRUPTION)
/// and the fail-closed verifier rules. S1 acceptance controls exercised here:
/// CANONICAL_ARRAY_TYPE_API, CONST_DENSE_CANONICAL_ARRAY_TYPE, ARRAY_TYPE_CONFLICT
/// (fail closed), the permanent negative control
/// ARRAY_TYPE_CONFLICT_CANNOT_MASQUERADE_AS_NOT_AGGREGATE, and malformed-node
/// fail-closed — against the real IR types. (CANONICAL_ARRAY_TYPE_LOOKUP_TOTAL is
/// deliberately NOT claimed until S3 populates every supported aggregate surface.)
#[cfg(all(test, feature = "std-surface"))]
mod step_d_aggregate_type_tests {
    use crate::ir::{
        AggregateTypeError, ArraySize, ArrayType, IRModule, Instr, IrVerifyError, verify_module,
    };
    use crate::types::{DType, ShapeDim};

    fn const_dense_f64x4() -> (IRModule, crate::ir::ValueId) {
        // A Tier-B const-dense `[f64; 4]` + an Output (verify_module requires one).
        let mut m = IRModule::new();
        let v = m.fresh();
        m.instrs.push(Instr::ConstDenseTensor {
            dst: v,
            dtype: DType::F64,
            shape: vec![ShapeDim::Known(4)],
            data: vec![0u64; 4],
        });
        m.instrs.push(Instr::Output(v));
        (m, v)
    }

    #[test]
    fn canonical_array_type_total_for_const_dense() {
        // CONST_DENSE_CANONICAL_ARRAY_TYPE: a ConstDenseTensor dst is fully typed
        // through the canonical lookup EVEN WITH an empty value_types table (it is
        // reconstructed from the node's dtype+shape) — canonical typing is total,
        // and Tier-B needs no wire entry.
        let (m, v) = const_dense_f64x4();
        assert!(m.value_types.is_empty(), "S1: value_types starts empty");
        assert_eq!(
            m.canonical_array_type(v),
            Ok(Some(ArrayType {
                elem_dtype: DType::F64,
                size: ArraySize::Fixed(4),
            }))
        );
    }

    #[test]
    fn canonical_array_type_ok_none_for_non_aggregate() {
        // Negative control A: a scalar ValueId is genuinely non-aggregate -> the
        // API returns Ok(None) (ABSENCE), distinct from any Err (CORRUPTION).
        let mut m = IRModule::new();
        let v = m.fresh();
        m.instrs.push(Instr::ConstI64(v, 7));
        m.instrs.push(Instr::Output(v));
        assert_eq!(m.canonical_array_type(v), Ok(None));
    }

    fn const_dense(v: crate::ir::ValueId, dt: DType, n: usize) -> Instr {
        Instr::ConstDenseTensor {
            dst: v,
            dtype: dt,
            shape: vec![ShapeDim::Known(n)],
            data: vec![0u64; n],
        }
    }

    fn typed_fn(name: &str, body: Vec<Instr>) -> Instr {
        Instr::FnDef {
            name: name.to_string(),
            params: vec![],
            ret_id: None,
            body,
            reap_threshold: None,
            value_types: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn same_numeric_valueid_different_functions_do_not_alias() {
        // SAME_NUMERIC_VALUEID_DIFFERENT_FUNCTIONS_DO_NOT_ALIAS: function bodies are
        // SEPARATE SSA namespaces, so the SAME numeric ValueId can define a `f32`
        // aggregate in `f` and a `f64` aggregate in `g`. Each must be typed against
        // ITS OWN scope; a flat/global key would alias them.
        let v = crate::ir::ValueId(0);
        let f_body = vec![const_dense(v, DType::F32, 3)];
        let g_body = vec![const_dense(v, DType::F64, 3)];
        assert_eq!(
            crate::ir::canonical_array_type_in(&f_body, &std::collections::BTreeMap::new(), v),
            Ok(Some(ArrayType {
                elem_dtype: DType::F32,
                size: ArraySize::Fixed(3),
            }))
        );
        assert_eq!(
            crate::ir::canonical_array_type_in(&g_body, &std::collections::BTreeMap::new(), v),
            Ok(Some(ArrayType {
                elem_dtype: DType::F64,
                size: ArraySize::Fixed(3),
            }))
        );
        // And both coexist in one module WITHOUT a false cross-function conflict.
        let mut m = IRModule::new();
        m.instrs.push(typed_fn("f", f_body));
        m.instrs.push(typed_fn("g", g_body));
        let w = m.fresh();
        m.instrs.push(Instr::ConstI64(w, 0));
        m.instrs.push(Instr::Output(w));
        assert_eq!(verify_module(&m), Ok(()));
    }

    #[test]
    fn const_dense_intrinsic_lookup_scoped() {
        // CONST_DENSE_INTRINSIC_LOOKUP_SCOPED: the SAME numeric id defines a
        // DIFFERENT-dtype const-dense at module scope, in `f`, and in `g`. The
        // module lookup must NOT descend into the functions (no global first-match),
        // and each function scope resolves its own.
        let v = crate::ir::ValueId(3);
        let mut m = IRModule::new();
        // module scope: %3 = f32 (module lookup must return THIS, not a fn's).
        while m.next_id <= 3 {
            m.fresh();
        }
        m.instrs.push(const_dense(v, DType::F32, 2));
        m.instrs.push(Instr::Output(v));
        // `fpriv` is defined ONLY inside `f` (never at module scope) — the sharp
        // no-descend probe: if the module lookup wrongly walked into FnDef bodies,
        // it would resolve `fpriv`; it must return `Ok(None)`.
        let fpriv = crate::ir::ValueId(9);
        let f_body = vec![
            const_dense(v, DType::F64, 2),
            const_dense(fpriv, DType::F32, 2),
        ];
        let g_body = vec![const_dense(v, DType::I64, 2)];
        m.instrs.push(typed_fn("f", f_body.clone()));
        m.instrs.push(typed_fn("g", g_body.clone()));
        // Module scope resolves f32 — it does NOT walk into FnDef bodies.
        assert_eq!(
            m.canonical_array_type(v),
            Ok(Some(ArrayType {
                elem_dtype: DType::F32,
                size: ArraySize::Fixed(2),
            }))
        );
        // Non-vacuous no-descend: `fpriv` exists ONLY in `f` -> module scope must
        // NOT find it (would fail if FnDef descent were wrongly reintroduced).
        assert_eq!(m.canonical_array_type(fpriv), Ok(None));
        // Each function scope resolves its own dtype.
        assert_eq!(
            crate::ir::canonical_array_type_in(&f_body, &std::collections::BTreeMap::new(), v)
                .unwrap()
                .unwrap()
                .elem_dtype,
            DType::F64
        );
        assert_eq!(
            crate::ir::canonical_array_type_in(&g_body, &std::collections::BTreeMap::new(), v)
                .unwrap()
                .unwrap()
                .elem_dtype,
            DType::I64
        );
    }

    #[test]
    fn const_dense_lookup_descends_control_flow() {
        // The scope-local lookup must RECURSE into nested control-flow bodies
        // (same SSA namespace), closing the latent S1 gap where a flat
        // `instrs.iter().find_map` missed a const-dense buried in a Region/If/While.
        let mut m = IRModule::new();
        let v = m.fresh();
        let r = m.fresh();
        let en = m.fresh();
        let ex = m.fresh();
        m.instrs.push(Instr::Region {
            body: vec![const_dense(v, DType::F64, 4)],
            result: r,
            enter_id: en,
            exit_id: ex,
            alloc_ids: vec![],
        });
        m.instrs.push(Instr::Output(v));
        // Found via recursive descent even though it is NOT in the flat top-level
        // instruction list.
        assert_eq!(
            m.canonical_array_type(v),
            Ok(Some(ArrayType {
                elem_dtype: DType::F64,
                size: ArraySize::Fixed(4),
            }))
        );
    }

    #[test]
    fn verify_catches_conflicting_table_in_fndef_nested_in_region() {
        // NOTE-4 hardening: a FnDef whose OWN value_types table lies about a
        // const-dense in its body must be caught EVEN when the FnDef is buried
        // inside a Region (same enclosing scope). Without descending control flow
        // in the FnDef hunt, a hostile mic@3 artifact could hide the lie here.
        let fv = crate::ir::ValueId(0);
        let mut fn_table = std::collections::BTreeMap::new();
        fn_table.insert(
            fv,
            ArrayType {
                elem_dtype: DType::I64, // lies: the body node is F64
                size: ArraySize::Fixed(2),
            },
        );
        let fndef = Instr::FnDef {
            name: "buried".to_string(),
            params: vec![],
            ret_id: None,
            body: vec![const_dense(fv, DType::F64, 2)],
            reap_threshold: None,
            value_types: fn_table,
        };
        let mut m = IRModule::new();
        let r = m.fresh();
        let en = m.fresh();
        let ex = m.fresh();
        m.instrs.push(Instr::Region {
            body: vec![fndef, Instr::ConstI64(r, 0)],
            result: r,
            enter_id: en,
            exit_id: ex,
            alloc_ids: vec![],
        });
        m.instrs.push(Instr::Output(r));
        match verify_module(&m) {
            Err(IrVerifyError::AggregateTypeConflict { value, .. }) => assert_eq!(value, fv),
            other => panic!("expected AggregateTypeConflict for the buried FnDef, got {other:?}"),
        }
    }

    #[test]
    fn verify_rejects_conflicting_value_types() {
        // Negative control E (FAIL CLOSED): an explicit value_types entry that
        // DISAGREES with the ConstDenseTensor node for the same value is a hard
        // AggregateTypeConflict, never a silent single-authority pick.
        let (mut m, v) = const_dense_f64x4();
        m.value_types.insert(
            v,
            ArrayType {
                elem_dtype: DType::I64, // conflicts with the node's F64
                size: ArraySize::Fixed(4),
            },
        );
        match verify_module(&m) {
            Err(IrVerifyError::AggregateTypeConflict { value, .. }) => assert_eq!(value, v),
            other => panic!("expected AggregateTypeConflict, got {other:?}"),
        }
    }

    #[test]
    fn array_type_conflict_cannot_masquerade_as_not_aggregate() {
        // PERMANENT NEGATIVE CONTROL
        // (ARRAY_TYPE_CONFLICT_CANNOT_MASQUERADE_AS_NOT_AGGREGATE): a conflict
        // between the table and the node MUST surface as `Err(Conflict)` — never
        // as `Ok(None)`. If it collapsed to absence, a downstream
        // `None => scalar / legacy / re-infer` path would silently bypass the
        // backend on a corrupt canonical type.
        let (mut m, v) = const_dense_f64x4();
        m.value_types.insert(
            v,
            ArrayType {
                elem_dtype: DType::I64,
                size: ArraySize::Fixed(4),
            },
        );
        let got = m.canonical_array_type(v);
        assert!(
            matches!(got, Err(AggregateTypeError::Conflict { .. })),
            "conflict must be Err(Conflict), got {got:?}"
        );
        assert_ne!(got, Ok(None), "a conflict must NOT masquerade as absence");
    }

    #[test]
    fn malformed_const_dense_fails_closed() {
        // A ConstDenseTensor with a non-Known (Sym) shape dim cannot be typed to a
        // Fixed(N): the lookup returns Err(MalformedConstDense) (never a confident
        // guess), and verify_module rejects it as MalformedConstDenseTensor.
        let mut m = IRModule::new();
        let v = m.fresh();
        m.instrs.push(Instr::ConstDenseTensor {
            dst: v,
            dtype: DType::F64,
            shape: vec![ShapeDim::Sym("B")], // not a fixed extent
            data: vec![0u64; 4],
        });
        m.instrs.push(Instr::Output(v));
        assert_eq!(
            m.canonical_array_type(v),
            Err(AggregateTypeError::MalformedConstDense)
        );
        match verify_module(&m) {
            Err(IrVerifyError::MalformedConstDenseTensor { value, .. }) => assert_eq!(value, v),
            other => panic!("expected MalformedConstDenseTensor, got {other:?}"),
        }
    }

    #[test]
    fn nested_malformed_const_dense_fails_closed() {
        // Fail closed at ANY depth: a malformed ConstDenseTensor buried inside a
        // FnDef body is still rejected (the well-formedness check walks the full
        // nested instruction tree, not just top-level instrs). A flat,
        // top-level-only check would let this hostile-artifact shape through.
        let mut m = IRModule::new();
        let r = m.fresh();
        m.instrs.push(Instr::FnDef {
            name: "g".to_string(),
            params: vec![],
            ret_id: Some(r),
            body: vec![Instr::ConstDenseTensor {
                dst: r,
                dtype: DType::F64,
                shape: vec![ShapeDim::Sym("B")], // malformed, nested one level down
                data: vec![0u64; 4],
            }],
            reap_threshold: None,
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
        });
        let v = m.fresh();
        m.instrs.push(Instr::ConstI64(v, 0));
        m.instrs.push(Instr::Output(v));
        match verify_module(&m) {
            Err(IrVerifyError::MalformedConstDenseTensor { value, .. }) => assert_eq!(value, r),
            other => panic!("expected MalformedConstDenseTensor for a nested node, got {other:?}"),
        }
    }

    #[test]
    fn verify_accepts_agreeing_value_types() {
        // An explicit entry that AGREES with the node is well-formed.
        let (mut m, v) = const_dense_f64x4();
        m.value_types.insert(
            v,
            ArrayType {
                elem_dtype: DType::F64,
                size: ArraySize::Fixed(4),
            },
        );
        assert!(verify_module(&m).is_ok());
        assert_eq!(
            m.canonical_array_type(v),
            Ok(Some(ArrayType {
                elem_dtype: DType::F64,
                size: ArraySize::Fixed(4)
            }))
        );
    }
}
