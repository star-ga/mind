// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Scope walk behind the `native.loop_body_shadow` / `native.block_shadow`
//! refusals.
//!
//! The frozen native compiler resolves variables by NAME across nested
//! blocks. A binding that re-declares a name already bound in an ENCLOSING
//! scope is merged with that outer binding:
//!
//! * inside a loop (a `let`, a `let (a, b)`, a `for`/for-each induction
//!   variable, a `match` arm binding, a closure parameter):
//!   `let mut x = 5; while .. { let mut x = 7; x = 3; } return x;` exits 3
//!   natively and 5 under MLIR;
//! * outside any loop, once the name is written from a further-nested block:
//!   `let mut x = 5; if c { let mut x = 9; if c { x = 40; } } return x;`
//!   exits 40 natively and 5 under MLIR (also with a loop in the inner `if`).
//!
//! A single-level `if` shadow happens to run correctly, but that was measured
//! at one nesting only, so the walk reports EVERY cross-frame re-declaration.
//!
//! Soundness over precision: an enclosing scope is ANY lexical frame outside
//! the binding's own -- module constants, function parameters, outer `let`s,
//! an enclosing loop body, `if` block, `match` arm or `{ }` block, a loop's own
//! induction variable. Admitted are only a re-declaration inside the SAME
//! frame, and a `let` at a function's top level re-declaring a parameter: the
//! parameters share the body's top frame, because nothing after the body can
//! observe the parameter again, so name-merging and lexical scoping agree
//! (measured: a top-level `let mut x = x + 1` written from a nested `if` runs
//! the same natively and under MLIR). Unknown nodes are walked through the
//! exhaustive `nerve_walk::for_each_child` inside a frame of their own, so a
//! binding in a statement container this walk does not model is checked
//! against every outer frame, including the one it appears in (the refusing
//! side).
//!
//! Each finding also says whether its shape is PROVEN to be scoped correctly by
//! a compiler whose `main.mind` gives every nested `let` its own storage
//! (`Shadow::proven`): a plain `let` re-declaring an outer `let` or parameter,
//! both annotated with a scalar type the proof exercises (`i64`, `i32`, `f64`),
//! where every frame between the two was opened by a statement-level `if` /
//! `else` block or a `while` loop. Those shapes are what
//! `examples/mindc_mind/self_host_native_shadow_scope_smoke.py` generates and
//! checks against a lexically scoped model (block depths 1-4, with `break`,
//! `continue` and `return`; the MLIR build agrees except where it carries a
//! loop-body shadow out through a `break`/`continue`, a known MLIR defect).
//! Everything else -- `for` / for-each variables and bodies, `match` bindings
//! and arms, closures, bare `{ }` blocks, regions, tuple `let`s, module
//! constants, a binding inside an expression (a value-`if` branch), an
//! unannotated `let` or any other type (structs, strings, references, tensors)
//! and any node this walk does not model -- stays unproven.

use std::collections::BTreeMap;

use crate::ast::{Module, Node, Pattern, Span, TypeAnn};
use crate::type_checker::nerve_walk::for_each_child;

/// One binding that re-declares a name bound in an enclosing scope.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Shadow {
    /// The re-declared name.
    pub name: String,
    /// The function whose body holds the inner binding.
    pub fn_name: String,
    /// What the inner binding is (`let`, `for variable`, ...).
    pub inner_kind: &'static str,
    /// Span of the inner binding.
    pub inner: Span,
    /// What the enclosing binding is (`parameter`, `let`, ...).
    pub outer_kind: &'static str,
    /// Span of the enclosing binding.
    pub outer: Span,
    /// True when the inner binding sits inside a loop (header or body).
    pub in_loop: bool,
    /// True when the shape is one the scoping smoke proves (see the module
    /// docs): admitted once the compiler that runs carries the scoping mark.
    pub proven: bool,
}

/// What opened a lexical frame. Only statement-level `if`/`else` blocks and
/// `while` loops are frames the scoping proof covers.
#[derive(Clone, Copy, PartialEq)]
enum FrameKind {
    Module,
    FnTop,
    IfBranch,
    While,
    Other,
}

#[derive(Clone, Copy)]
struct Binding {
    kind: &'static str,
    span: Span,
    /// Annotated with a scalar type the scoping proof covers (`proven_scalar`).
    scalar: bool,
}

/// The binding types the scoping smoke exercises: a shadow is proven only when
/// both bindings carry one of these annotations.
fn proven_scalar(t: Option<&TypeAnn>) -> bool {
    matches!(
        t,
        Some(TypeAnn::ScalarI64 | TypeAnn::ScalarI32 | TypeAnn::ScalarF64)
    )
}

struct Walk<'a> {
    /// Lexical frames, innermost last. Frame 0 holds module-level names.
    frames: Vec<BTreeMap<&'a str, Binding>>,
    /// What opened each frame of `frames` (same length).
    kinds: Vec<FrameKind>,
    /// Number of loops enclosing the current position in the current function.
    loop_depth: usize,
    fn_name: &'a str,
    found: Vec<Shadow>,
}

/// Every binding of `module` that re-declares a name from an enclosing scope.
pub(super) fn find(module: &Module) -> Vec<Shadow> {
    let mut module_frame = BTreeMap::new();
    for item in &module.items {
        match item {
            Node::Const { name, span, .. } | Node::ExternConst { name, span, .. } => {
                module_frame.insert(
                    name.as_str(),
                    Binding {
                        kind: "const",
                        span: *span,
                        scalar: false,
                    },
                );
            }
            _ => {}
        }
    }
    let mut walk = Walk {
        frames: vec![module_frame],
        kinds: vec![FrameKind::Module],
        loop_depth: 0,
        fn_name: "",
        found: Vec::new(),
    };
    for item in &module.items {
        walk.node(item);
    }
    walk.found
}

impl<'a> Walk<'a> {
    fn push(&mut self, kind: FrameKind) {
        self.frames.push(BTreeMap::new());
        self.kinds.push(kind);
    }

    fn pop(&mut self) {
        self.frames.pop();
        self.kinds.pop();
    }

    /// Walk an expression position in a frame of its own, so an `if` used as a
    /// value (and any binding inside it) never counts as a statement block.
    fn expr(&mut self, n: &'a Node) {
        self.push(FrameKind::Other);
        self.node(n);
        self.pop();
    }

    fn bind(&mut self, name: &'a str, span: Span, kind: &'static str) {
        self.bind_as(name, span, kind, false);
    }

    /// `scalar`: the binding is a single-name `let` statement or a parameter
    /// annotated with a type `proven_scalar` accepts (the only bindings the
    /// scoping proof covers).
    fn bind_as(&mut self, name: &'a str, span: Span, kind: &'static str, scalar: bool) {
        if name == "_" {
            return;
        }
        let top = self.frames.len() - 1;
        let enclosing = self.frames[..top]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, f)| f.get(name).copied().map(|b| (i, b)));
        if let Some((at, outer)) = enclosing {
            let proven = scalar
                && kind == "let"
                && outer.scalar
                && matches!(outer.kind, "let" | "parameter")
                && self.kinds[at + 1..]
                    .iter()
                    .all(|k| matches!(k, FrameKind::IfBranch | FrameKind::While));
            self.found.push(Shadow {
                name: name.to_string(),
                fn_name: self.fn_name.to_string(),
                inner_kind: kind,
                inner: span,
                outer_kind: outer.kind,
                outer: outer.span,
                in_loop: self.loop_depth > 0,
                proven,
            });
        }
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name, Binding { kind, span, scalar });
        }
    }

    fn scoped(&mut self, stmts: &'a [Node], kind: FrameKind) {
        self.push(kind);
        for s in stmts {
            self.node(s);
        }
        self.pop();
    }

    /// A loop: optional induction variable in its own frame, body in a child
    /// frame, everything (header expressions included) at `loop_depth + 1`.
    fn looped(&mut self, header: &[&'a Node], var: Option<(&'a str, Span)>, body: &'a [Node]) {
        self.loop_depth += 1;
        for h in header {
            self.expr(h);
        }
        // Only a `while` (no induction variable) is a frame the proof covers.
        let kind = if var.is_none() {
            FrameKind::While
        } else {
            FrameKind::Other
        };
        self.push(kind);
        if let Some((name, span)) = var {
            self.bind(name, span, "for variable");
        }
        self.scoped(body, kind);
        self.pop();
        self.loop_depth -= 1;
    }

    fn pattern(&mut self, p: &'a Pattern, span: Span) {
        match p {
            Pattern::Ident(name) => self.bind(name, span, "match binding"),
            Pattern::EnumVariant { args, .. } | Pattern::Tuple(args) => {
                for a in args {
                    self.pattern(a, span);
                }
            }
            Pattern::EnumStruct { fields, .. } => {
                for (_, sub) in fields {
                    self.pattern(sub, span);
                }
            }
            Pattern::Literal(_) | Pattern::Wildcard => {}
        }
    }

    fn function(&mut self, name: &'a str, params: &'a [crate::ast::Param], body: &'a [Node]) {
        // A function body starts a fresh scope chain: only module names enclose it.
        let saved_frames = self.frames.split_off(1);
        let saved_kinds = self.kinds.split_off(1);
        let saved_depth = std::mem::replace(&mut self.loop_depth, 0);
        let saved_name = std::mem::replace(&mut self.fn_name, name);
        // The parameters share the body's top frame: a top-level `let` that
        // re-declares one is a same-frame re-declaration (nothing after the
        // body can observe the parameter again).
        self.push(FrameKind::FnTop);
        for p in params {
            self.bind_as(&p.name, p.span, "parameter", proven_scalar(Some(&p.ty)));
        }
        for s in body {
            self.node(s);
        }
        self.pop();
        self.fn_name = saved_name;
        self.loop_depth = saved_depth;
        self.frames.extend(saved_frames);
        self.kinds.extend(saved_kinds);
    }

    fn node(&mut self, n: &'a Node) {
        match n {
            Node::FnDef(fd, _) => self.function(&fd.name, &fd.params, &fd.body),
            Node::Let {
                name,
                ann,
                value,
                span,
                ..
            } => {
                self.expr(value);
                self.bind_as(name, *span, "let", proven_scalar(ann.as_ref()));
            }
            Node::LetTuple {
                names, value, span, ..
            } => {
                self.expr(value);
                for name in names {
                    self.bind(name, *span, "let");
                }
            }
            Node::Const {
                name, value, span, ..
            } => {
                self.expr(value);
                self.bind(name, *span, "const");
            }
            Node::Block { stmts, .. } => self.scoped(stmts, FrameKind::Other),
            #[cfg(feature = "std-surface")]
            Node::Region { body, .. } => self.scoped(body, FrameKind::Other),
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                self.expr(cond);
                self.scoped(then_branch, FrameKind::IfBranch);
                if let Some(e) = else_branch {
                    self.scoped(e, FrameKind::IfBranch);
                }
            }
            #[cfg(feature = "std-surface")]
            Node::While { cond, body, .. } => self.looped(&[cond.as_ref()], None, body),
            Node::For {
                var,
                start,
                end,
                body,
                span,
                ..
            } => self.looped(&[start.as_ref(), end.as_ref()], Some((var, *span)), body),
            Node::ForEach {
                var,
                collection,
                body,
                span,
            } => self.looped(&[collection.as_ref()], Some((var, *span)), body),
            Node::Match {
                scrutinee, arms, ..
            } => {
                self.expr(scrutinee);
                for arm in arms {
                    self.push(FrameKind::Other);
                    self.pattern(&arm.pattern, arm.span);
                    if let Some(g) = &arm.guard {
                        self.expr(g);
                    }
                    self.push(FrameKind::Other);
                    self.node(&arm.body);
                    self.pop();
                    self.pop();
                }
            }
            Node::Closure(cd, _) => {
                // Conservative: the closure body keeps the enclosing chain and
                // loop depth, so a closure parameter or `let` that re-declares a
                // captured-scope name is reported like any nested binding.
                self.push(FrameKind::Other);
                for p in &cd.params {
                    self.bind(&p.name, p.span, "closure parameter");
                }
                self.scoped(&cd.body, FrameKind::Other);
                self.pop();
            }
            _ => {
                // Any other node's children get a frame of their own, so a
                // binding inside a statement container this walk does not
                // know is checked against EVERY frame outside it -- including
                // the current one -- and a re-declaration there is refused.
                self.push(FrameKind::Other);
                for_each_child(n, &mut |child| self.node(child));
                self.pop();
            }
        }
    }
}
