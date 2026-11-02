// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Check-time gate for the two lowering refusals that used to reach the user as
//! a Rust `panic!` — issue #237 CASE 2 (a collection mutator whose realloc'd
//! handle cannot be rebound) and CASE 3 (a non-final bare-identifier match arm
//! whose name collides with a registered enum variant).
//!
//! Both refusals are CORRECT: each stands in front of a demonstrated silent
//! miscompile. What was wrong is that they only existed during lowering, so
//! `mindc check` exited 0 on the same source that made `mindc build` abort with
//! rc=101 and a backtrace. An IDE saw a crash, CI saw a crash, and `check` gave
//! a false green.
//!
//! # Why the rules live here and not in a second copy
//!
//! This module does NOT re-derive "is this receiver a tracked collection" or "is
//! this bare name a variant". Two copies of a refusal rule are two answers, and
//! a drifted gate is worse than no gate: it either refuses a program that builds
//! (a regression) or admits one straight into the panic it was meant to replace.
//! So the predicates and the expression walk are DEFINED here and CALLED by
//! `eval::lower`; where a rule cannot be lifted out of lowering (the match
//! desugar), the gate runs the real lowering code with a thread-local refusal
//! SINK armed. The two report sites consult the sink: armed, they record and the
//! walk continues; absent — every real lowering — they panic exactly as before,
//! which keeps the fail-closed backstop for a gate/lowering drift.
//!
//! # Why one call site is enough to make check and build agree
//!
//! `check_module_types_in_file` calls [`check`], and
//! `pipeline::compile_source_with_name` runs that same type-check BEFORE
//! `lower_to_ir`, returning `CompileError::TypeError` on any diagnostic. So both
//! `mindc check` and `mindc build` reach this gate through one function — they
//! cannot disagree — and a refused build exits 1 having written no artifact.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use super::Pretty;
use crate::ast::{Literal, MatchArm, Module, Node, Pattern, Span as AstSpan, TypeAnn};
use crate::diagnostics::{Severity, Span};
use crate::eval::lower;
use crate::ir::IRModule;

/// A collection mutating method used where its realloc'd handle cannot be
/// rebound, so the mutation would be silently lost (#306 / #237 CASE 2).
pub(crate) const E_COLLECTION_MUTATION_IN_EXPR: &str = "E2300";

/// A non-final unguarded bare-identifier arm whose name is a registered enum
/// variant: neither a discriminant test nor a catch-all (#237 CASE 3).
pub(crate) const E_NON_FINAL_VARIANT_BINDING: &str = "E2301";

/// `==`/`!=` that would compare string handles, not bytes: a string against an
/// operand whose type is not a string, or two operands of unknown type (see
/// `eval::string_equality`).
pub(crate) const E_STRING_EQUALITY_OPERAND: &str = "E2038";

/// The collection mutating-method names whose std implementation returns a
/// FRESH handle on realloc (`vec_push`, `map_insert`, …).
///
/// ONE list, read by both the statement-rebind matcher in
/// `lower::rewrite_collection_mutations` and the refusal walk below, so a
/// mutator is refused in an unrebindable position iff it would have been
/// rebound as a bare statement. The two used to be a `const` and a hand-written
/// `matches!` kept in sync by a comment.
// `vec_set` mutates existing storage and returns an i64 status, never a new
// owner. Rebinding that status would replace the collection handle with zero.
pub(crate) const COLLECTION_MUTATORS: &[&str] = &["insert", "push", "add"];

// ---------------------------------------------------------------------------
// Refusal sink
// ---------------------------------------------------------------------------

/// One refused construct, recorded by lowering instead of aborting it.
pub(crate) struct Refusal {
    code: &'static str,
    message: String,
    help: &'static str,
    span: AstSpan,
}

thread_local! {
    static SINK: RefCell<Option<Vec<Refusal>>> = const { RefCell::new(None) };
}

/// Disarms the sink on drop.
///
/// Not housekeeping: if an unwind escaped a collection run with the sink still
/// armed, the NEXT real lowering on this thread would RECORD its refusal and
/// carry on instead of panicking — turning the fail-closed backstop into the
/// silent miscompile it exists to prevent. This is cleanup on the way out, not
/// a caught panic.
struct SinkGuard;

impl Drop for SinkGuard {
    fn drop(&mut self) {
        SINK.with(|s| {
            *s.borrow_mut() = None;
        });
    }
}

/// Run `f` with refusal collection armed and return everything it recorded.
fn collect_with<R>(f: impl FnOnce() -> R) -> Vec<Refusal> {
    SINK.with(|s| *s.borrow_mut() = Some(Vec::new()));
    let _guard = SinkGuard;
    let _ = f();
    SINK.with(|s| s.borrow_mut().take()).unwrap_or_default()
}

/// Is a refusal-collection run in progress on this thread?
///
/// Read by the one lowering site whose refusal is owned elsewhere (the dangling
/// variant tag of #237 CASE 1), so that running the match desugar at check time
/// cannot make `mindc check` start aborting on a diagnostic this gate does not
/// own.
pub(crate) fn is_collecting() -> bool {
    SINK.with(|s| s.borrow().is_some())
}

/// Record `refusal` when collecting; otherwise panic, exactly as lowering did
/// before this gate existed.
fn report(refusal: Refusal) {
    let message = refusal.message.clone();
    let help = refusal.help;
    let code = refusal.code;
    let recorded = SINK.with(|s| match s.borrow_mut().as_mut() {
        Some(found) => {
            found.push(refusal);
            true
        }
        None => false,
    });
    if !recorded {
        // Unreachable from user input: the gate refuses first on every path that
        // can emit an artifact. Reaching it means the gate and lowering drifted,
        // which is a compiler defect and must stay loud.
        panic!("[{code}] {message}. {help}");
    }
}

/// CASE 3 — called from `lower::desugar_match_to_if`'s `(None, None)` arm.
pub(crate) fn non_final_variant_binding(pattern: &Pattern, span: AstSpan) {
    report(Refusal {
        code: E_NON_FINAL_VARIANT_BINDING,
        message: format!(
            "match arm pattern `{pattern:?}` is an irrefutable bare-identifier binding whose \
             name collides with a registered enum variant, in a non-final (test) position — \
             it is neither a discriminant test nor a catch-all, so the match cannot be lowered"
        ),
        help: "qualify the variant (`Enum::V`), rename the binding, or move the catch-all to \
               the final arm; falling back to a sequential evaluation here would ignore the \
               scrutinee and return the last arm — a silent miscompile",
        span,
    });
}

/// Called from `eval::string_equality::normalize`, which the CASE 2 run below
/// executes, so check and build refuse the same comparisons.
pub(crate) fn string_equality_operand(refused: &crate::eval::string_equality::Refused) {
    report(Refusal {
        code: E_STRING_EQUALITY_OPERAND,
        message: crate::eval::string_equality::refusal_message(refused),
        help: crate::eval::string_equality::refusal_help(refused),
        span: refused.span,
    });
}

fn collection_mutation_in_expr(receiver: &str, method: &str, span: AstSpan) {
    report(Refusal {
        code: E_COLLECTION_MUTATION_IN_EXPR,
        message: format!(
            "collection mutation `{receiver}.{method}(...)` in expression position is not \
             supported: the std `{method}` returns a fresh handle on realloc that cannot be \
             rebound here, so the mutation would be silently lost"
        ),
        help: "use it as its own statement (`recv.method(...)` on its own line), or as a \
               same-binding update (`recv = recv.method(...)`) — #306: refusing a known \
               silent miscompile",
        span,
    });
}

// ---------------------------------------------------------------------------
// CASE 2 — collection mutators in unrebindable positions
// ---------------------------------------------------------------------------

/// The lexical facts a receiver is judged against: which names are tracked
/// collections here, which `(struct, field)` pairs are collections, and which
/// locals have a struct type. Bundled so the walk keeps ONE parameter instead of
/// four and its stack frame does not grow with the rule set — the walk mirrors
/// `lower_expr`'s descent, and lowering-frame depth is a standing Windows
/// constraint.
#[derive(Clone)]
pub(crate) struct CollectionScope<'a> {
    pub scope: &'a HashSet<String>,
    pub struct_collection_fields: &'a HashSet<(String, String)>,
    pub struct_collection_element_fields: &'a HashSet<(String, String)>,
    pub vtypes: &'a HashMap<String, String>,
}

/// Does `receiver` name a tracked collection — a local/param of `array<T>` /
/// `map<K,V>` / `set<T>`, a struct collection FIELD, or a struct
/// `array<collection>` ELEMENT?
///
/// Mirrors the rebind matcher in `lower::rewrite_collection_mutations` so a
/// mutator is refused in an unrebindable position on exactly the receiver shapes
/// statement-position rebinding handles. The test is the receiver's TYPE, never
/// the method's spelling: a user-defined `.add` on a non-collection receiver is
/// not a tracked-collection mutator and passes through untouched.
fn receiver_is_tracked_collection(receiver: &Node, ctx: &CollectionScope<'_>) -> bool {
    match receiver {
        Node::Lit(Literal::Ident(v), _) => ctx.scope.contains(v),
        Node::FieldAccess {
            receiver: base,
            field,
            ..
        } => matches!(base.as_ref(), Node::Lit(Literal::Ident(obj), _)
        if ctx.vtypes.get(obj).is_some_and(|s| {
            ctx.struct_collection_fields.contains(&(s.clone(), field.clone()))
        })),
        Node::IndexAccess { receiver: base, .. } => matches!(
            base.as_ref(),
            Node::FieldAccess { receiver: obj, field, .. }
                if matches!(obj.as_ref(), Node::Lit(Literal::Ident(ov), _)
                    if ctx.vtypes.get(ov).is_some_and(|s| {
                        ctx.struct_collection_element_fields.contains(&(s.clone(), field.clone()))
                    }))
        ),
        _ => false,
    }
}

/// Is `value` a SAME-BINDING functional update of `name` (`m = m.insert(k, v)` /
/// `let m = m.insert(k, v)`)?
///
/// The one shape that rebinds the fresh handle back onto the receiver itself, so
/// the old handle becomes unreachable and nothing is lost. It is also, for the
/// assignment form, byte-for-byte the node `rewrite_collection_mutations`
/// synthesises for a bare `m.insert(k, v)` statement — the compiler emitted it
/// while refusing to accept it written by hand, which is what made
/// `out = out.push(x)` a rc=101 panic in mind-codegraph's BFS (#237 CASE 2, gap
/// G6). A DIFFERENT-name target (`w = v.push(x)`) leaves `v` on the freed handle
/// and is NOT this shape.
pub(crate) fn is_same_name_rebind(name: &str, value: &Node) -> bool {
    matches!(
        value,
        Node::MethodCall { receiver, method, .. }
            if COLLECTION_MUTATORS.contains(&method.as_str())
                && matches!(receiver.as_ref(), Node::Lit(Literal::Ident(r), _) if r == name)
    )
}

/// Apply one lexical `let` after its initializer has been inspected. Lowering
/// and the refusal walk share this update so they agree on shadowing.
pub(crate) fn update_let_binding(
    name: &str,
    ann: &Option<TypeAnn>,
    value: &Node,
    scope: &mut HashSet<String>,
    vtypes: &mut HashMap<String, String>,
) {
    let inferred_struct = match value {
        Node::Lit(Literal::Ident(src), _) => vtypes.get(src).cloned(),
        _ => None,
    };
    let remains_collection = lower::let_declares_collection(ann, value)
        || (scope.contains(name) && is_same_name_rebind(name, value));
    scope.remove(name);
    vtypes.remove(name);
    if remains_collection {
        scope.insert(name.to_string());
    } else if let Some(TypeAnn::Named(sname)) = ann {
        vtypes.insert(name.to_string(), sname.clone());
    } else if let Some(sname) = inferred_struct {
        vtypes.insert(name.to_string(), sname);
    }
}

pub(crate) fn shadow_tuple_bindings(
    names: &[String],
    scope: &mut HashSet<String>,
    vtypes: &mut HashMap<String, String>,
) {
    for name in names {
        scope.remove(name);
        vtypes.remove(name);
    }
}

/// Reject inside a call whose TOP node is an ALLOWED mutation — a bare
/// statement mutator, or a same-binding update.
///
/// The exemption covers exactly one node: the call whose handle is written back.
/// Its receiver sub-expressions and its ARGUMENTS are ordinary expression
/// positions, so `out = out.push(v.push(5))` rebinds `out` and still drops
/// `v`'s handle — the same silent loss, one level down. Skipping the whole
/// sub-tree because the top was exempt would be a hole in the middle of the
/// gate, so the operands are always walked.
pub(crate) fn reject_rebound_call_operands(call: &Node, ctx: &CollectionScope<'_>) {
    let Node::MethodCall { receiver, args, .. } = call else {
        // Not a method call: nothing is exempt, walk it whole.
        reject_collection_mutation_in_expr(call, ctx);
        return;
    };
    let mut roots = Vec::with_capacity(args.len() + 1);
    roots.push(receiver.as_ref());
    roots.extend(args.iter());
    reject_collection_mutations(roots, ctx);
}

/// Walk an expression-position sub-tree and report every collection mutator in
/// it whose realloc'd handle cannot be rebound.
///
/// Every such call — `w.push(v.push(5))`, `f(a.push(x))`, `let n = a.push(x)`,
/// `match a.push(x) { … }`, `0 => a.push(x)` — evaluates for its value and
/// discards the fresh handle, so the mutation is lost the first time the buffer
/// reallocs. A value-returning method (`a.length`, `a.get(i)`) is not a mutator
/// and a non-collection receiver is not tracked, so both pass through.
///
/// Descends into the STATEMENT bodies of a nested `match` arm as well
/// ([`reject_stmts_in_expr_position`]). That is not symmetry for its own sake:
/// a block reached from expression position never meets the statement-rebind
/// pass, so a bare `a.push(x)` inside it is dropped just as silently as one in
/// an argument. It was the last uncovered position in the #237 CASE 2 family.
pub(crate) fn reject_collection_mutation_in_expr(expr: &Node, ctx: &CollectionScope<'_>) {
    reject_collection_mutations(vec![expr], ctx);
}

enum MutationWork<'a> {
    Expr(&'a Node, Rc<HashSet<String>>, Rc<HashMap<String, String>>),
    Stmts(
        &'a [Node],
        usize,
        Rc<HashSet<String>>,
        Rc<HashMap<String, String>>,
    ),
}

/// Exhaustive non-recursive walk. A statement-list work item carries lexical
/// collection facts forward without growing the native stack.
fn reject_collection_mutations(roots: Vec<&Node>, ctx: &CollectionScope<'_>) {
    let scope = Rc::new(ctx.scope.clone());
    let vtypes = Rc::new(ctx.vtypes.clone());
    let mut work = Vec::with_capacity(roots.len().max(8));
    for root in roots.into_iter().rev() {
        work.push(MutationWork::Expr(root, scope.clone(), vtypes.clone()));
    }
    while let Some(item) = work.pop() {
        if let MutationWork::Stmts(stmts, index, scope, vtypes) = item {
            let Some(stmt) = stmts.get(index) else {
                continue;
            };
            let mut next_scope = scope.clone();
            let mut next_vtypes = vtypes.clone();
            match stmt {
                Node::Let {
                    name, ann, value, ..
                } => update_let_binding(
                    name,
                    ann,
                    value,
                    Rc::make_mut(&mut next_scope),
                    Rc::make_mut(&mut next_vtypes),
                ),
                Node::LetTuple { names, .. } => shadow_tuple_bindings(
                    names,
                    Rc::make_mut(&mut next_scope),
                    Rc::make_mut(&mut next_vtypes),
                ),
                _ => {}
            }
            work.push(MutationWork::Stmts(
                stmts,
                index + 1,
                next_scope,
                next_vtypes,
            ));
            work.push(MutationWork::Expr(stmt, scope, vtypes));
            continue;
        }
        let MutationWork::Expr(node, scope, vtypes) = item else {
            unreachable!()
        };
        let local = CollectionScope {
            scope: &scope,
            struct_collection_fields: ctx.struct_collection_fields,
            struct_collection_element_fields: ctx.struct_collection_element_fields,
            vtypes: &vtypes,
        };
        if let Node::MethodCall {
            receiver, method, ..
        } = node
        {
            if COLLECTION_MUTATORS.contains(&method.as_str())
                && receiver_is_tracked_collection(receiver, &local)
            {
                collection_mutation_in_expr(
                    &lower::describe_receiver(receiver),
                    method,
                    node.span(),
                );
            }
        }
        match node {
            Node::Let { name, value, .. } | Node::Assign { name, value, .. }
                if is_same_name_rebind(name, value) =>
            {
                let Node::MethodCall { receiver, args, .. } = value.as_ref() else {
                    unreachable!()
                };
                for child in args.iter().rev() {
                    work.push(MutationWork::Expr(child, scope.clone(), vtypes.clone()));
                }
                work.push(MutationWork::Expr(receiver, scope, vtypes));
            }
            Node::Block { stmts, .. } => {
                work.push(MutationWork::Stmts(stmts, 0, scope, vtypes));
            }
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                if let Some(branch) = else_branch {
                    work.push(MutationWork::Stmts(
                        branch,
                        0,
                        scope.clone(),
                        vtypes.clone(),
                    ));
                }
                work.push(MutationWork::Stmts(
                    then_branch,
                    0,
                    scope.clone(),
                    vtypes.clone(),
                ));
                work.push(MutationWork::Expr(cond, scope, vtypes));
            }
            Node::For {
                start, end, body, ..
            } => {
                work.push(MutationWork::Stmts(body, 0, scope.clone(), vtypes.clone()));
                work.push(MutationWork::Expr(end, scope.clone(), vtypes.clone()));
                work.push(MutationWork::Expr(start, scope, vtypes));
            }
            Node::ForEach {
                collection, body, ..
            } => {
                work.push(MutationWork::Stmts(body, 0, scope.clone(), vtypes.clone()));
                work.push(MutationWork::Expr(collection, scope, vtypes));
            }
            #[cfg(feature = "std-surface")]
            Node::While { cond, body, .. } => {
                work.push(MutationWork::Stmts(body, 0, scope.clone(), vtypes.clone()));
                work.push(MutationWork::Expr(cond, scope, vtypes));
            }
            #[cfg(feature = "std-surface")]
            Node::Region { body, .. } => {
                work.push(MutationWork::Stmts(body, 0, scope, vtypes));
            }
            Node::Match {
                scrutinee, arms, ..
            } => {
                for arm in arms.iter().rev() {
                    match &arm.body {
                        Node::Block { stmts, .. } => {
                            work.push(MutationWork::Stmts(stmts, 0, scope.clone(), vtypes.clone()))
                        }
                        body => work.push(MutationWork::Expr(body, scope.clone(), vtypes.clone())),
                    }
                    if let Some(guard) = &arm.guard {
                        work.push(MutationWork::Expr(guard, scope.clone(), vtypes.clone()));
                    }
                }
                work.push(MutationWork::Expr(scrutinee, scope, vtypes));
            }
            // Definitions have independent parameter/local scopes; the real
            // lowering prepass processes them separately.
            Node::FnDef(..) | Node::Closure(..) | Node::ImplBlock { .. } => {}
            _ => {
                let mut children = Vec::new();
                super::nerve_walk::for_each_child(node, &mut |child| children.push(child));
                for child in children.into_iter().rev() {
                    work.push(MutationWork::Expr(child, scope.clone(), vtypes.clone()));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Append every #237 CASE 2 / CASE 3 refusal in `module` to `errors`.
pub(super) fn check(module: &Module, src: &str, file: Option<&str>, errors: &mut Vec<Pretty>) {
    let mut found = collect_collection_mutations(module);
    found.extend(collect_match_refusals(module));
    // Deterministic order whatever the walk order: source position, then code
    // (the only way two refusals can share a span is across the two rules).
    found.sort_by_key(|r| (r.span.start(), r.span.end(), r.code));
    for r in found {
        errors.push(Pretty {
            phase: "type-check",
            code: r.code,
            severity: Severity::Error,
            message: r.message,
            span: Some(Span::from_offsets(src, r.span.start(), r.span.end(), file)),
            notes: Vec::new(),
            help: Some(r.help.to_string()),
        });
    }
}

/// CASE 2: run the real statement-rewrite pass under the sink.
///
/// `preprocess_collection_mutations` returns early for a module that declares no
/// collection, so a collection-free source — the keystone, the frontend
/// benchmark fixtures — pays that existing pre-scan and nothing else.
fn collect_collection_mutations(module: &Module) -> Vec<Refusal> {
    collect_with(|| lower::preprocess_collection_mutations(module))
}

/// CASE 3: run the real match desugar under the sink, for the matches that can
/// possibly reach the refusal.
fn collect_match_refusals(module: &Module) -> Vec<Refusal> {
    // Cheap EXACT prefilter, ahead of any registry work. The refusal needs a
    // non-final unguarded `Pattern::Ident` arm; that is decidable from the arm
    // list alone, with no enum registry and no allocation, and it is a strict
    // SUPERSET of `may_refuse` (which additionally requires the name to collide
    // with a variant). So a module this rejects cannot contain a refusal — no
    // false negatives, and it is not a bypass of any rule.
    //
    // Without it every type-check of a scalar, match-free module paid for an
    // `IRModule`, the ~14 prelude allocations, a global-enum-registry merge and
    // a second full AST walk. That is the whole cost of this gate for the
    // overwhelming majority of modules, so it is worth deciding first.
    if !module.items.iter().any(|item| {
        super::nerve_walk::any(
            item,
            &|n| matches!(n, Node::Match { arms, .. } if has_non_final_ident_arm(arms)),
        )
    }) {
        return Vec::new();
    }

    let mut ir = IRModule::new();
    lower::install_enum_prelude_and_globals(&mut ir, module);
    lower::register_module_wrapped_enum_tags(&mut ir, &module.items);

    let mut out: Vec<Refusal> = Vec::new();
    for item in &module.items {
        // Mirror lowering's item ORDER. A top-level `enum` registers when the
        // main loop REACHES it, so a function declared above it does not see its
        // variants and a bare arm there is still a catch-all. Registering the
        // whole module up front would refuse that program — a false positive on
        // code that builds today.
        if let Node::EnumDef { name, variants, .. } = item {
            lower::register_enum_metadata(&mut ir, name, variants);
            continue;
        }
        super::nerve_walk::walk(item, &mut |node| {
            if let Node::Match {
                scrutinee, arms, ..
            } = node
            {
                if !may_refuse(arms, &ir.enum_variant_tags) {
                    return;
                }
                out.extend(collect_with(|| {
                    lower::desugar_match_to_if(
                        scrutinee,
                        None,
                        arms,
                        &ir.enum_variant_tags,
                        &ir.boxed_enums,
                        &ir.enum_payload_types,
                        &ir.enum_struct_field_names,
                    )
                }));
            }
        });
    }
    out
}

/// Is a bare identifier pattern a CATCH-ALL rather than a reference to a
/// variant?
///
/// The single definition, shared with `lower::desugar_match_to_if`, which calls
/// it to decide where to truncate the arm list. One definition is the point:
/// this predicate IS the line between a program that builds and one this gate
/// refuses, so the gate and lowering must read the same line.
pub(crate) fn ident_is_catch_all(name: &str, tags: &BTreeMap<String, i64>) -> bool {
    !tags.contains_key(name)
        && !tags
            .keys()
            .any(|k| k.rsplit_once("::").map(|(_, v)| v == name).unwrap_or(false))
}

/// Can this match possibly reach the CASE 3 refusal?
///
/// The refusal needs an UNGUARDED irrefutable arm left in a TEST slot. A
/// `Wildcard` never survives there (it IS a catch-all, so the arm list is
/// truncated to it), and neither does a bare `Ident` whose name is not a
/// registered variant. Only a non-final unguarded `Pattern::Ident` colliding
/// with a variant name can — so every other match skips the desugar entirely,
/// which is what keeps this gate off the compile-time hot path.
fn may_refuse(arms: &[MatchArm], tags: &BTreeMap<String, i64>) -> bool {
    has_non_final_ident_arm(arms)
        && arms[..arms.len() - 1].iter().any(|a| {
            a.guard.is_none()
                && matches!(&a.pattern, Pattern::Ident(name)
                    if !is_dotted_variant(name, tags) && !ident_is_catch_all(name, tags))
        })
}

/// The registry-free half of [`may_refuse`]: is there an unguarded bare-`Ident`
/// arm in a non-final slot at all?
///
/// Split out so the module-level prefilter and the per-match test are the SAME
/// predicate rather than two that could drift — the prefilter's whole safety
/// argument is that it is a superset of the per-match test.
fn has_non_final_ident_arm(arms: &[MatchArm]) -> bool {
    arms.len() >= 2
        && arms[..arms.len() - 1]
            .iter()
            .any(|a| a.guard.is_none() && matches!(a.pattern, Pattern::Ident(_)))
}

/// A fieldless variant written with DOT notation (`Kind.Scalar`) parses as a
/// bare `Ident` but the desugar normalises it to an `EnumVariant`, so it is
/// never an irrefutable arm. Mirrors that normalisation.
fn is_dotted_variant(name: &str, tags: &BTreeMap<String, i64>) -> bool {
    name.contains('.') && tags.contains_key(&name.replacen('.', "::", 1))
}

#[cfg(test)]
#[path = "lowering_refusals_tests.rs"]
mod tests;
