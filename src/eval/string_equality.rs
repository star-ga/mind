// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Typed string equality normalization before recursive expression lowering.
//!
//! `==` and `!=` on two strings compare bytes. An operand is a string when its
//! declared type says so (see `facts` and `classify`): a literal, a
//! `string`/`String` parameter or annotated binding, a string constant, call
//! result (including a method call that lowers to a `string_<method>` or
//! `<type>_<method>` function), struct field, collection element, an `if` /
//! `match` whose every branch is a string, or a payload of an `Option` /
//! `Result` declared (or constructed) to carry one. Such a comparison is
//! rewritten to the compiler-owned `__mind_string_eq` before the interpreter
//! or any lowering sees it, so every backend computes the same byte-wise
//! answer. The null handle `0` is a `string` value equal only to itself; no
//! record is read through it.
//!
//! Every other comparison that could compare string handles, which is false
//! for equal text, is refused at check time with `E2038`, never lowered
//! silently:
//!
//! * a string compared with an operand that is not a string (an integer
//!   literal is exempt: a handle compared with it is a null or sentinel test);
//! * two operands of which neither has a known type, so both may be strings;
//! * a string literal `match` arm on a scrutinee not known to be a string
//!   (an arm on a string compares bytes on every backend);
//! * in a project build, a call of a function another module refuses such a
//!   comparison in (see `project`);
//! * a cast to `string` of an integer literal other than the null handle `0`
//!   (`5 as string`), which no string record is at.
//!
//! An operand whose type is known and is not a string makes the comparison an
//! integer comparison on every backend. A declaration proves what its slot
//! holds only while nothing else is stored there (see `slots` and
//! `flows`): a parameter, result, field, payload, module or local binding,
//! or collection element given a value that may be a string where none is
//! declared, or an integer other than the null handle where a string is
//! declared, is read as maybe a string. A value whose type the pass did not
//! establish may be a string, so storing it counts, also when it is handed
//! to a callee whose body is not analysed here (a std collection, raw
//! memory): the integer result of such a callee is then maybe a string; a
//! raw 64-bit load reads any collection's storage. A cast to `string` of a
//! value not proven a string, other than the null handle, is maybe a string.
//! A value of a type parameter, or the result of an imported function with
//! no declared result type, may be anything. A function with no declared
//! result type that still returns a value has an unknown result. Stores from
//! other modules of a project build count too.
//!
//! `mindc check`, `mindc build` (MLIR and native) and `mindc test` refuse the
//! same source: each analyses every module against its own declarations, the
//! project registry and the project summary. The module `mindc test` merges is
//! split back into its modules first (see [`segments`]); no module learns a
//! fact from another module's merged body.
//!
//! A local binding is a slot like any other: each assignment is read in its
//! own scope, and the module is walked again until no binding changes (see
//! `Decls::local_fact`); facts are never joined across branches of
//! statements, which avoids branch, loop, and shadowing ambiguity.
//!
//! `.length` on a string reads a member no string record has; it is refused
//! at check time with the code the build gives it (`E6009`). For `mindc test`
//! a method call on a string becomes the `string_<method>` call the MLIR
//! lowering makes for it (`analyze_interpreted`).

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{BinOp, Literal, Module, Node, Span};

mod classify;
mod facts;
#[cfg(test)]
mod flow_tests;
mod flows;
#[cfg(feature = "cross-module-imports")]
mod project;
#[cfg(test)]
mod segment_tests;
pub mod segments;
#[cfg(test)]
mod slot_tests;
mod slots;
mod sums;
#[cfg(test)]
mod tests;
mod walk;

use facts::Class;
#[cfg(feature = "cross-module-imports")]
pub(crate) use project::ProjectModules;

type Targets = BTreeSet<usize>;
/// Method call ordinal → whether its receiver is passed as the first
/// argument (a value receiver) or names the type (`string.from(x)`).
type MethodTargets = BTreeMap<usize, bool>;
type Assigned = BTreeSet<String>;

/// Deterministic preorder numbers of the original equality and method call
/// nodes, by node address in the analysed AST.
#[derive(Default)]
struct Ordinals {
    equalities: BTreeMap<usize, usize>,
    methods: BTreeMap<usize, usize>,
    /// Lowering only: each `print` operand that is not a string literal. Its
    /// numbers follow the equalities', so one target set holds both.
    print_args: BTreeMap<usize, usize>,
}

/// Why a comparison is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefusedKind {
    /// A string compared with an operand not known to be a string.
    Mixed,
    /// Two operands, neither of a known type.
    Untyped,
    /// A string literal pattern matched against a scrutinee not known to be
    /// a string.
    Pattern,
    /// A call of a function of another module of the project that compares
    /// a value that may be a string there (see `project`).
    Callee,
    /// `s.length` on a string: no string record has that member, and no
    /// backend lowers the read.
    Member,
    /// `5 as string`: an integer literal other than the null handle `0` cast
    /// to `string`; no string record is at that address.
    Cast,
    /// A `print` operand that is a value of the named aggregate kind: a
    /// runnable artifact would write its handle as an integer.
    PrintAggregate(&'static str),
    /// A `print` operand whose type is not known, so it may be a string.
    PrintUntyped,
}

impl RefusedKind {
    /// Whether this refuses a `print` operand. Only a runnable artifact
    /// refuses one (see `print_blockers`): the interpreter prints every value.
    pub(crate) fn is_print(self) -> bool {
        matches!(self, Self::PrintAggregate(_) | Self::PrintUntyped)
    }
}

/// A comparison that would compare string handles instead of bytes.
pub(crate) struct Refused {
    pub(crate) op: BinOp,
    pub(crate) kind: RefusedKind,
    pub(crate) span: Span,
    /// The function whose body holds the comparison (`None` at module level
    /// and in a closure); for [`RefusedKind::Callee`], the function called.
    pub(crate) function: Option<String>,
}

/// The `E2038` message for `refused`, naming the byte-wise std function.
pub(crate) fn refusal_message(refused: &Refused) -> String {
    let symbol = if refused.op == BinOp::Eq { "==" } else { "!=" };
    let operands = match refused.kind {
        RefusedKind::Mixed => "a string with an operand not known to be a string",
        RefusedKind::Untyped => {
            "two operands whose types are not known at compile time, so both may be strings"
        }
        RefusedKind::Pattern => {
            return "a string literal `match` pattern is compared with a scrutinee not known \
                    to be a string; it would compare a handle or read an integer as a \
                    string record — use `string_eq(s, \"...\")` for byte-wise equality"
                .to_string();
        }
        RefusedKind::Member => {
            return "`.length` reads a member of a string, which no string record has, and \
                    no backend lowers the read — use `string_len(s)` for the byte length"
                .to_string();
        }
        RefusedKind::Cast => {
            return "`as string` casts an integer other than the null handle `0`; no string \
                    record is at that address, and `==` or a std.string function would read \
                    one through it"
                .to_string();
        }
        RefusedKind::PrintAggregate(kind) => {
            return format!(
                "`print` of {kind}: a runnable artifact would write its handle as an \
                 integer, not its contents"
            );
        }
        RefusedKind::PrintUntyped => {
            return "`print` of a value whose type is not known at compile time: it may be \
                    a string, and a runnable artifact would write its handle as an integer"
                .to_string();
        }
        RefusedKind::Callee => {
            let callee = refused.function.as_deref().unwrap_or("?");
            return format!(
                "`{callee}`, defined in another module of this project, compares two \
                 operands that may be strings (refused there with E2038); it would compare \
                 handles, not bytes — use `string_eq(a, b)` for byte-wise equality"
            );
        }
    };
    format!(
        "`{symbol}` compares {operands}; it would compare handles, not bytes, and \
         report equal text as unequal — use `string_eq(a, b)` for byte-wise equality"
    )
}

/// The diagnostic code of `refused`: `E6009` (the code the build gives the
/// same read) for a string member read, the print key for a `print` operand,
/// `E2038` for every comparison.
pub(crate) fn refusal_code(refused: &Refused) -> &'static str {
    match refused.kind {
        RefusedKind::Member => crate::type_checker::lowering_refusals::E_STRING_MEMBER,
        RefusedKind::PrintAggregate(_) | RefusedKind::PrintUntyped => {
            crate::type_checker::lowering_refusals::E_PRINT_OPERAND
        }
        _ => crate::type_checker::lowering_refusals::E_STRING_EQUALITY_OPERAND,
    }
}

/// The `E2038` help text for `refused`.
pub(crate) fn refusal_help(refused: &Refused) -> &'static str {
    match refused.kind {
        RefusedKind::Mixed => {
            "give the other operand a `string` type (parameter, annotated binding, \
             field or function return) and store only strings (or the null handle \
             `0`) where `string` is declared, or compare bytes explicitly with \
             `string_eq(a, b)` from std.string"
        }
        RefusedKind::Untyped => {
            "give an operand a declared type (an annotated binding such as \
             `let n: i64 = ...`, a parameter, a field or a function return `-> T`) \
             and store no string where another type is declared, or compare \
             strings explicitly with `string_eq(a, b)` from std.string"
        }
        RefusedKind::Pattern => {
            "give the scrutinee a `string` type (parameter, annotated binding, \
             field or function return) and store only strings there, or test \
             it with `string_eq(s, \"...\")` from std.string"
        }
        RefusedKind::Member => {
            "call `string_len(s)` from std.string; std.string is a set of free \
             functions, and a string has no `.length` member"
        }
        RefusedKind::Cast => {
            "cast only a string or the null handle `0` to `string`; write text as a \
             string literal, or build it with the std.string functions"
        }
        RefusedKind::PrintAggregate(_) => {
            "print each scalar or string part on its own (`print(p.x)`), or run the \
             program with the `mind` interpreter"
        }
        RefusedKind::PrintUntyped => {
            "give the value a declared type (an annotated binding such as \
             `let n: i64 = ...`, a parameter, a field or a function return `-> T`)"
        }
        RefusedKind::Callee => {
            "declare the compared slots `string` where strings are passed, or \
             compare bytes there with `string_eq(a, b)` from std.string; \
             `mindc build` reports the comparison in its own module"
        }
    }
}

/// Decide one equality from its operand classes: `Ok(true)` rewrites it to the
/// byte-wise comparison, `Ok(false)` keeps it an integer comparison, and
/// `Err` refuses it.
fn decide(left: Class, right: Class) -> Result<bool, RefusedKind> {
    match (left, right) {
        (Class::Str, Class::Str) => Ok(true),
        (Class::Str, Class::IntLiteral) | (Class::IntLiteral, Class::Str) => Ok(false),
        (Class::Str, _) | (_, Class::Str) => Err(RefusedKind::Mixed),
        (Class::Unknown, Class::Unknown) => Err(RefusedKind::Untyped),
        _ => Ok(false),
    }
}

/// Decide one `print` operand from its fact: `Ok(true)` prints the bytes of
/// a string, `Ok(false)` keeps the integer print of a scalar, and `Err`
/// refuses a value no runnable artifact prints correctly.
fn decide_print(fact: facts::Fact) -> Result<bool, RefusedKind> {
    use facts::Fact;
    match fact {
        Fact::Str => Ok(true),
        Fact::NotStr | Fact::Never => Ok(false),
        Fact::Record(_) => Err(RefusedKind::PrintAggregate("a struct value")),
        Fact::Elems(_) => Err(RefusedKind::PrintAggregate(
            "an array, slice, set or tensor",
        )),
        Fact::Map(_) => Err(RefusedKind::PrintAggregate("a map")),
        Fact::Tuple(_) => Err(RefusedKind::PrintAggregate("a tuple")),
        Fact::Sum(..) => Err(RefusedKind::PrintAggregate("an `Option` or `Result` value")),
        Fact::Unknown | Fact::Any | Fact::MaybeStr => Err(RefusedKind::PrintUntyped),
    }
}

/// The callee a string `print` operand is wrapped in for lowering.
const PRINT_TEXT: &str = "__mind_print_text";

/// The expression whose string record a `print` operand writes as bytes: a
/// string literal, or an operand this pass proved a string. `None` for an
/// operand printed as an integer.
pub(crate) fn print_text(arg: &Node) -> Option<&Node> {
    match arg {
        Node::Lit(Literal::Str(_), _) => Some(arg),
        Node::Call { callee, args, .. } if callee == PRINT_TEXT => args.first(),
        _ => None,
    }
}

/// Number every `print` operand that is not a string literal, in the
/// preorder `rewrite` visits them, after the equalities.
fn number_print_args(module: &Module, ordinals: &mut Ordinals) {
    let mut pending: Vec<&Node> = module.items.iter().rev().collect();
    while let Some(node) = pending.pop() {
        if let Node::Print { args, .. } = node {
            for arg in args.iter().filter(|a| print_text(a).is_none()) {
                let next = ordinals.equalities.len() + ordinals.print_args.len();
                ordinals
                    .print_args
                    .insert(arg as *const Node as usize, next);
            }
        }
        let children = children_ref(node);
        pending.extend(children.into_iter().rev());
    }
}

fn children_ref(node: &Node) -> Vec<&Node> {
    match node {
        Node::FnDef(fd, _) => fd.body.iter().collect(),
        Node::Closure(data, _) => data.body.iter().collect(),
        _ => super::closures::node_children_ref(node),
    }
}

fn children_mut(node: &mut Node) -> Vec<&mut Node> {
    match node {
        Node::FnDef(fd, _) => fd.body.iter_mut().collect(),
        Node::Closure(data, _) => data.body.iter_mut().collect(),
        _ => super::closures::node_children_mut(node),
    }
}

fn assigned_in(stmts: &[Node]) -> Assigned {
    let mut assigned = Assigned::new();
    let mut pending: Vec<&Node> = stmts.iter().rev().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Assign { name, value, .. } => {
                assigned.insert(name.clone());
                pending.push(value);
            }
            // Each declaration owns its own assignment analysis.
            Node::FnDef(..) | Node::Closure(..) => {}
            _ => {
                let children = super::closures::node_children_ref(node);
                pending.extend(children.into_iter().rev());
            }
        }
    }
    assigned
}

/// Give every original equality and method call node a deterministic
/// preorder number. Pointer identity is used only inside this immutable AST
/// snapshot; the numbers, not pointers or source-local spans, are carried
/// into the rewritten clone.
fn equality_ordinals(module: &Module) -> Ordinals {
    let mut ordinals = Ordinals::default();
    let mut pending: Vec<&Node> = module.items.iter().rev().collect();
    while let Some(node) = pending.pop() {
        let numbered = match node {
            Node::Binary {
                op: BinOp::Eq | BinOp::Ne,
                ..
            } => Some(&mut ordinals.equalities),
            Node::MethodCall { .. } => Some(&mut ordinals.methods),
            _ => None,
        };
        if let Some(numbered) = numbered {
            let next = numbered.len();
            numbered.insert(node as *const Node as usize, next);
        }
        let children = children_ref(node);
        pending.extend(children.into_iter().rev());
    }
    ordinals
}

/// Whether a call in the module being analysed may be refused: it is a
/// module of a project build in which another module refuses a comparison
/// (see `project`).
fn calls_may_be_refused() -> bool {
    #[cfg(feature = "cross-module-imports")]
    return project::current().is_some_and(|summary| !summary.refusing.is_empty());
    #[cfg(not(feature = "cross-module-imports"))]
    false
}

/// Whether `module` has a form the walk decides besides `==` / `!=`: a
/// `match` with a string literal arm, which compares a string too, a
/// `.length` member read, which is refused on a string, or a cast to a named
/// type that may be `string`, refused for an integer literal (see `walk`).
fn has_other_string_form(module: &Module) -> bool {
    let mut pending: Vec<&Node> = module.items.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Match { arms, .. } => {
                let literal = |arm: &crate::ast::MatchArm| {
                    matches!(arm.pattern, crate::ast::Pattern::Literal(Literal::Str(_)))
                };
                if arms.iter().any(literal) {
                    return true;
                }
            }
            Node::FieldAccess { field, .. } if field == "length" => return true,
            Node::As {
                ty: crate::ast::TypeAnn::Named(name),
                ..
            } if !facts::SCALAR_NAMES.contains(&name.as_str()) => return true,
            _ => {}
        }
        pending.extend(children_ref(node));
    }
    false
}

/// Rewrite in deterministic preorder on an explicit heap worklist. A target is
/// replaced before its original operands are queued, so the synthesized `!= 0`
/// comparison is not counted as another source equality, and a desugared
/// method call keeps its receiver and arguments in their preorder.
fn rewrite(
    module: &mut Module,
    targets: &mut Targets,
    methods: &mut MethodTargets,
    equalities: usize,
) {
    let mut next_equality = 0;
    let mut next_method = 0;
    let mut next_print = equalities;
    let mut pending: Vec<&mut Node> = module.items.iter_mut().rev().collect();
    while let Some(node) = pending.pop() {
        if let Node::Print { args, .. } = node {
            for arg in args.iter_mut().filter(|a| print_text(a).is_none()) {
                let ordinal = next_print;
                next_print += 1;
                if targets.remove(&ordinal) {
                    let span = arg.span();
                    let text = std::mem::replace(arg, Node::Lit(Literal::Int(0), span));
                    *arg = Node::Call {
                        callee: PRINT_TEXT.to_string(),
                        args: vec![text],
                        span,
                    };
                }
            }
        }
        if matches!(node, Node::MethodCall { .. }) {
            let ordinal = next_method;
            next_method += 1;
            if let Some(passes_receiver) = methods.remove(&ordinal) {
                desugar_string_method(node, passes_receiver);
            }
        }
        let target = if matches!(
            node,
            Node::Binary {
                op: BinOp::Eq | BinOp::Ne,
                ..
            }
        ) {
            let ordinal = next_equality;
            next_equality += 1;
            targets.remove(&ordinal)
        } else {
            false
        };
        if target {
            let old = std::mem::replace(node, Node::Lit(Literal::Int(0), node.span()));
            let Node::Binary {
                op,
                left,
                right,
                span,
            } = old
            else {
                unreachable!("target ordinal did not name an equality node")
            };
            let eq = Node::Call {
                callee: "__mind_string_eq".to_string(),
                args: vec![*left, *right],
                span,
            };
            *node = if op == BinOp::Eq {
                eq
            } else {
                Node::Binary {
                    op: BinOp::Eq,
                    left: Box::new(eq),
                    right: Box::new(Node::Lit(Literal::Int(0), span)),
                    span,
                }
            };
        }
        let children = children_mut(node);
        pending.extend(children.into_iter().rev());
    }
}

/// `s.method(args)` on a string `s` → `string_<method>(s, args)` (or, for a
/// type-name receiver `string.method(args)`, `string_<method>(args)`): the
/// call the MLIR lowering makes for a string receiver (`receiver_is_string`).
fn desugar_string_method(node: &mut Node, passes_receiver: bool) {
    let old = std::mem::replace(node, Node::Lit(Literal::Int(0), node.span()));
    let Node::MethodCall {
        receiver,
        method,
        args,
        span,
    } = old
    else {
        unreachable!("method ordinal did not name a method call")
    };
    let receiver = passes_receiver.then_some(*receiver);
    *node = Node::Call {
        callee: format!("string_{method}"),
        args: receiver.into_iter().chain(args).collect(),
        span,
    };
}

/// The analysis of one module: the rewritten clone (only when a string
/// comparison exists) and every refused comparison in source order.
pub(crate) struct Analysis {
    pub(crate) rewritten: Option<Module>,
    pub(crate) refused: Vec<Refused>,
}

/// Analyze `module` for lowering. The walk is iterative: modules with no
/// target keep lower_expr's stack shape, emitted bytes, and the Windows
/// debug-stack margin unchanged.
pub(crate) fn analyze(module: &Module) -> Analysis {
    analyze_with(module, false)
}

/// Analyze `module` for the interpreter: as for lowering, and a method call
/// on a string also becomes the `string_<method>` call the MLIR lowering
/// makes for it, so `mindc test` runs the same std function.
pub(crate) fn analyze_interpreted(module: &Module) -> Analysis {
    analyze_with(module, true)
}

fn analyze_with(module: &Module, desugar_methods: bool) -> Analysis {
    // The interpreter analyses on a thread where the project's module table
    // may not be installed; read the one its check reads.
    #[cfg(feature = "cross-module-imports")]
    let _scope = desugar_methods.then(project::install_scope);
    let mut ordinals = equality_ordinals(module);
    if !desugar_methods {
        number_print_args(module, &mut ordinals);
    }
    let methods_wanted = desugar_methods && !ordinals.methods.is_empty();
    let decides = !ordinals.equalities.is_empty()
        || !ordinals.print_args.is_empty()
        || methods_wanted
        || calls_may_be_refused()
        || has_other_string_form(module);
    if !decides {
        return Analysis {
            rewritten: None,
            refused: Vec::new(),
        };
    }
    // The interpreter's module merges the std and project modules a test
    // reaches into the test file; each is decided with its own facts, as the
    // build compiles each alone (see `segments`).
    let (mut targets, methods, refused) = if desugar_methods {
        segments::collect_targets(module, &ordinals)
    } else {
        walk::collect_targets(module, &ordinals)
    };
    let mut methods = if desugar_methods {
        methods
    } else {
        MethodTargets::new()
    };
    if targets.is_empty() && methods.is_empty() {
        return Analysis {
            rewritten: None,
            refused,
        };
    }
    let mut rewritten = module.clone();
    rewrite(
        &mut rewritten,
        &mut targets,
        &mut methods,
        ordinals.equalities.len(),
    );
    assert!(
        targets.is_empty() && methods.is_empty(),
        "string equality normalization lost an analyzed AST target"
    );
    Analysis {
        rewritten: Some(rewritten),
        refused,
    }
}

/// Lowering entry: report each refused comparison through the check-time
/// refusal sink (which `mindc check` and `mindc build` consult before
/// lowering), then return the rewritten clone.
pub(crate) fn normalize(module: &Module) -> Option<Module> {
    let analysis = analyze(module);
    for refused in analysis.refused.iter().filter(|r| !r.kind.is_print()) {
        segments::report_refused(refused);
    }
    analysis.rewritten
}

/// The `print` operands of `module` no runnable artifact prints correctly,
/// as runnable-artifact blockers: an aggregate value, or a value whose type
/// is not known. `mindc check` and the interpreter accept them.
#[cfg(feature = "std-surface")]
pub(crate) fn print_blockers(
    module: &Module,
    src: &str,
    file: Option<&str>,
) -> Vec<crate::diagnostics::Diagnostic> {
    use crate::diagnostics::{Diagnostic, Span as DiagSpan};
    if !module.items.iter().any(|item| {
        crate::type_checker::nerve_walk::any(item, &|n| matches!(n, Node::Print { .. }))
    }) {
        return Vec::new();
    }
    let refused = analyze(module).refused.into_iter();
    refused
        .filter(|r| r.kind.is_print())
        .map(|r| {
            let span = DiagSpan::from_offsets(src, r.span.start(), r.span.end(), file);
            Diagnostic::error("lower", refusal_code(&r), refusal_message(&r))
                .with_span(span)
                .with_help(refusal_help(&r))
        })
        .collect()
}
