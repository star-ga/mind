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

//! Scoped name-resolution pass for function bodies (issue #23).
//!
//! `mindc check` historically accepted undefined references inside `fn`
//! bodies (`return r;` / `return nope(3);` produced `rc=0`, no diagnostic).
//! The proximate cause was the additive FnDef-body shape pass: it re-ran the
//! module type-checker on the body as a "mini-module" and then *filtered* the
//! result down to shape codes + `E2004` narrowing, dropping the
//! unknown-identifier / undefined-call diagnostics. The filter existed because
//! the mini-module env was incomplete (params default to `ScalarI64`, `&expr`
//! / match / struct constructs mis-type) and the resolution set is spread
//! across four sources — so un-dropping those diagnostics naively
//! false-positived on ~12/22 std files.
//!
//! This module replaces that hack for the undefined-reference question with a
//! dedicated, purpose-built pass that does ONE thing: decide whether a
//! referenced name (a bare identifier, or a call callee) is *resolvable*. It
//! never infers types, so the `ScalarI64`-param / `&expr` / match mis-typing
//! that drove the false positives simply does not apply here.
//!
//! Resolvability unions ALL symbol sources (the design's `is_resolvable`):
//!   1. lexical scope frames — params + every binding form (`let`, `for` var,
//!      `match`-arm bindings), with proper nested `Block` / `If` / `While` /
//!      `For` / `Match` frames so an intermediate `let` inside a nested block
//!      is visible to later references in that scope (the asymmetric
//!      binding-tracking gap the root-cause doc identifies);
//!   2. module-level names — fn / const / let / struct / enum / type-alias
//!      names and in-file `extern "C"` declarations;
//!   3. the std-surface intrinsic table (`__mind_*`);
//!   4. cross-module imported symbols (project table, when populated).
//!
//! Soundness guard for the single-file `mindc check` path: that path runs with
//! an EMPTY project table, so `import std.vec` cannot enumerate the names it
//! brings in. We therefore record whether the module has any *unresolved*
//! import and, when it does, SUPPRESS undefined-call diagnostics (those calls
//! may legitimately come from the import). Bare-variable references are still
//! reported even then — imports bring functions/types, never loose variables.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::ast::{Literal, Module, Node, Pattern, TypeAnn};

/// Closed set of bare math / tensor / autodiff builtin call names — the
/// builtins the corpus invokes WITHOUT an `import` line or a `tensor.` prefix
/// (e.g. `sqrt(x)`, `matmul(a, b)`, `backward(loss)`). These are first-class
/// language builtins lowered directly by the codegen path, so a call to one is
/// never a genuinely-undefined reference.
///
/// EMPIRICALLY SWEPT — NOT a guessed list. Derived by running
/// `mindc check std/ examples/ --no-fmt` over the full corpus (`std/`,
/// `examples/`, `examples/zoo/`, `examples/compliance/`) under BOTH feature
/// configs (`std-surface` and `std-surface cross-module-imports`) with E2003
/// detection live, then folding in every flagged name that is a legitimate
/// builtin (and FIXING the genuine bug — `vec_free` → `__mind_vec_free`).
///
/// MUST stay sorted ascending — looked up with `binary_search`. A `debug_assert`
/// in `resolve_fn_body` enforces the ordering so a careless insert fails tests.
const BARE_BUILTINS: &[&str] = &[
    "abs",
    "argmax",
    "backward",
    "conv2d",
    "cos",
    "exp",
    "fft",
    "fft2d",
    "floor",
    "full",
    "gather",
    "ifft",
    "linspace",
    "log",
    "log10",
    "log2",
    "log_softmax",
    "matmul",
    "max",
    "maxpool2d",
    "mean",
    "ones",
    "pow",
    "rand_normal",
    "rand_uniform",
    "random",
    "random_normal",
    "relu",
    "reshape",
    "round",
    "sigmoid",
    "sin",
    "sqrt",
    "sum",
];

/// Undefined-variable diagnostic code (issue #23). A bare identifier that
/// resolves to nothing in any scope frame or module-level symbol source.
pub const UNKNOWN_IDENT_CODE: &str = "E2002";
/// Undefined-call diagnostic code (issue #23). A call whose callee resolves to
/// nothing — not a local, not a module fn, not an intrinsic, not an import.
pub const UNKNOWN_CALL_CODE: &str = "E2003";

/// An unresolved reference found by the pass, with the kind that selects the
/// diagnostic code/message. Spans are AST byte-offset spans; the caller renders
/// them through the same `diag_from_span` localization as every other code.
pub struct Unresolved {
    pub name: String,
    pub span: crate::ast::Span,
    pub is_call: bool,
    /// The closest in-scope name (edit-distance "did you mean"), if any.
    pub suggestion: Option<String>,
}

/// The module-level resolution context, built once per module.
struct ModuleSyms {
    /// All names resolvable without a local binding: module fns/consts/lets,
    /// struct/enum/type-alias names, in-file extern fn names, and (when the
    /// project table is populated) cross-module imported symbols.
    names: BTreeSet<String>,
}

/// Collect the top-level declaration names a module defines (fn / const / let /
/// struct / enum / type-alias / extern), recursing into the `Block` that a
/// `module NAME { ... }` body unwraps into. Pure: it does NOT resolve imports,
/// so it is safe to call while building the std-export cache (no re-entrancy).
pub(crate) fn collect_decl_names(module: &Module, out: &mut BTreeSet<String>) {
    for item in &module.items {
        match item {
            Node::FnDef { name, .. }
            | Node::Const { name, .. }
            | Node::Let { name, .. }
            | Node::StructDef { name, .. }
            | Node::EnumDef { name, .. }
            | Node::TypeAlias { name, .. } => {
                out.insert(name.clone());
            }
            Node::ExternBlock { fns, .. } => {
                for efn in fns {
                    out.insert(efn.name.clone());
                }
            }
            Node::Block { stmts, .. } => {
                let inner = Module {
                    items: stmts.clone(),
                };
                collect_decl_names(&inner, out);
            }
            _ => {}
        }
    }
}

/// Cached map of std-module path (`"std.vec"`) to the set of top-level names it
/// exports. Built exactly once from the bundled stdlib registry so the
/// single-file `mindc check` path resolves std-surface calls (`vec_new`,
/// `sha256`, ...) PRECISELY — no false positive, and no blanket suppression
/// that would also hide a genuine typo.
fn stdlib_exports() -> &'static BTreeMap<String, BTreeSet<String>> {
    static CACHE: OnceLock<BTreeMap<String, BTreeSet<String>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // `project::stdlib` (the bundled std-surface registry) is gated behind
        // `cross-module-imports`. With the feature off there is no std surface
        // to resolve, so this map stays empty — keeps `--no-default-features`
        // compiling while leaving the feature-on behaviour byte-for-byte
        // unchanged.
        #[allow(unused_mut)]
        let mut map = BTreeMap::new();
        #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
        for (path, module) in crate::project::stdlib::parsed_stdlib_modules() {
            let mut names = BTreeSet::new();
            collect_decl_names(&module, &mut names);
            map.insert(path, names);
        }
        map
    })
}

/// Collect the module-level resolvable-name set. `injected` carries any symbol
/// names the cross-module resolver already merged into the type env (so a
/// populated project table resolves imports precisely). On top of that we
/// resolve `import std.X` against the bundled stdlib registry so the single-file
/// path knows the std-surface API; a non-std import whose symbols cannot be
/// enumerated here suppresses undefined-CALL reports only — bare undefined
/// variables are still reported, since imports bring fns/types, never vars.
fn collect_module_syms(module: &Module, injected: &BTreeSet<String>) -> ModuleSyms {
    let mut names: BTreeSet<String> = injected.clone();
    collect_decl_names(module, &mut names);
    let std_exports = stdlib_exports();
    // The std-surface is the language standard library: its public names are
    // always resolvable. std modules reference each other WITHOUT explicit
    // imports (they are compiled as one bundle — e.g. std/io.mind calls
    // std.string's `string_push_byte` with no import line), and a std-surface
    // function is never a *genuinely-undefined* reference. So make the whole
    // surface visible rather than per-import. (Whether a user module should be
    // forced to `import std.X` before calling it is a separate "missing import"
    // lint, not the undefined-reference question issue #23 owns.)
    for exports in std_exports.values() {
        names.extend(exports.iter().cloned());
    }
    ModuleSyms { names }
}

/// Collect the symbolic shape-dimension identifiers declared in a type
/// annotation into `out`. A tensor type's `dims` are raw strings: a numeric
/// dim (`32`, `800`, `[]`) is a literal, while a non-numeric dim (`batch`,
/// `N`) is a symbolic shape variable that is in scope throughout the function
/// body wherever that shape is referenced (e.g. `reshape(x, [batch, 800])`).
/// We recurse through compound annotations (`Slice`/`Ref`/`Array`/`Tuple`/
/// `Generic`/`FnPtr`/`RawPtr`/`SparseTensor`) so a shape var nested inside,
/// say, a `Vec<tensor<f32[N]>>` parameter is also pre-bound. This only ever
/// ADDS resolvable names, so it can never produce a false positive.
pub fn collect_shape_vars(ty: &TypeAnn, out: &mut BTreeSet<String>) {
    match ty {
        TypeAnn::Tensor { dims, .. } | TypeAnn::DiffTensor { dims, .. } => {
            for d in dims {
                add_shape_dim_str(d, out);
            }
        }
        TypeAnn::SparseTensor { element, shape, .. } => {
            collect_shape_vars(element, out);
            for sd in shape {
                if let crate::types::ShapeDim::Sym(s) = sd {
                    out.insert((*s).to_string());
                }
            }
        }
        TypeAnn::Slice { element, .. } => collect_shape_vars(element, out),
        TypeAnn::Array { element, .. } => collect_shape_vars(element, out),
        TypeAnn::Ref { target, .. } => collect_shape_vars(target, out),
        TypeAnn::RawPtr { pointee, .. } => collect_shape_vars(pointee, out),
        TypeAnn::Tuple { elements } => {
            for e in elements {
                collect_shape_vars(e, out);
            }
        }
        TypeAnn::Generic { args, .. } => {
            for a in args {
                collect_shape_vars(a, out);
            }
        }
        TypeAnn::FnPtr { params, ret } => {
            for p in params {
                collect_shape_vars(p, out);
            }
            if let Some(r) = ret {
                collect_shape_vars(r, out);
            }
        }
        // Scalars and user-named types carry no shape dimensions.
        TypeAnn::ScalarI32
        | TypeAnn::ScalarI64
        | TypeAnn::ScalarF32
        | TypeAnn::ScalarF64
        | TypeAnn::ScalarBool
        | TypeAnn::ScalarU32
        | TypeAnn::Named(_) => {}
    }
}

/// Insert a tensor dim string into `out` iff it is a symbolic identifier (not a
/// numeric literal and not the empty scalar-shape marker).
fn add_shape_dim_str(dim: &str, out: &mut BTreeSet<String>) {
    let d = dim.trim();
    if d.is_empty() {
        return;
    }
    // A numeric dim is a literal extent, not an identifier.
    if d.bytes().all(|b| b.is_ascii_digit()) {
        return;
    }
    out.insert(d.to_string());
}

/// A stack of lexical scope frames. Frame 0 holds the fn parameters; nested
/// blocks push a fresh frame on entry and pop it on exit.
struct Scopes {
    frames: Vec<BTreeSet<String>>,
}

impl Scopes {
    fn new() -> Self {
        Scopes {
            frames: vec![BTreeSet::new()],
        }
    }
    fn push(&mut self) {
        self.frames.push(BTreeSet::new());
    }
    fn pop(&mut self) {
        self.frames.pop();
    }
    fn bind(&mut self, name: &str) {
        if let Some(top) = self.frames.last_mut() {
            top.insert(name.to_string());
        }
    }
    fn contains(&self, name: &str) -> bool {
        self.frames.iter().any(|f| f.contains(name))
    }
}

/// True for tensor builtins whose `Node::Call` arguments include dtype/shape
/// identifier literals (`tensor.zeros(f32, (..))`) rather than value
/// expressions. We must not descend into their args as references — `f32` and
/// symbolic shape names are not variables. All such callees start with
/// `tensor.`; their callee is always a builtin, so it is never undefined.
fn is_builtin_callee(callee: &str) -> bool {
    callee.starts_with("tensor.")
}

/// Did-you-mean suggestion drawn from the in-scope names + module symbols.
fn suggest(name: &str, scopes: &Scopes, syms: &ModuleSyms) -> Option<String> {
    let max_dist = (name.chars().count() / 2).clamp(1, 2);
    let mut best: Option<(usize, &str)> = None;
    let candidates = scopes
        .frames
        .iter()
        .flat_map(|f| f.iter())
        .map(String::as_str)
        .chain(syms.names.iter().map(String::as_str));
    for known in candidates {
        let d = super::levenshtein(name, known);
        if d == 0 || d > max_dist {
            continue;
        }
        let better = match &best {
            None => true,
            Some((bd, bk)) => d < *bd || (d == *bd && known < *bk),
        };
        if better {
            best = Some((d, known));
        }
    }
    best.map(|(_, k)| k.to_string())
}

/// Resolver state threaded through the body walk.
struct Resolver<'a> {
    syms: &'a ModuleSyms,
    scopes: Scopes,
    out: Vec<Unresolved>,
}

impl<'a> Resolver<'a> {
    fn ident_resolvable(&self, name: &str) -> bool {
        self.scopes.contains(name)
            || self.syms.names.contains(name)
            // A qualified path used as a *value* — an enum-variant constructor
            // (`Mode::On`), an associated const, etc. — folds its `::` segments
            // into one ident string. This pass owns the BARE undefined-reference
            // question (E2002); verifying that a `Type::Member` actually exists is
            // a separate type-resolution concern. Mirror `call_resolvable`'s `::`
            // treatment so a qualified value reference is never a bare-undefined.
            || name.contains("::")
    }

    fn call_resolvable(&self, name: &str) -> bool {
        // #23 sound call resolution: a callee resolves iff ANY symbol source
        // claims it. Unions (in cheapest-first order):
        //   1. a lexical binding (a `let f = ...` closure / fn-ptr param) or a
        //      module-level name (fn/const/struct/extern/cross-module-injected
        //      + the std-surface export set) — the same set `ident_resolvable`
        //      uses;
        //   2. the `__mind_*` intrinsic namespace (raw runtime-support ABI) —
        //      also subsumes the two non-`__mind_` STD_SURFACE_INTRINSICS
        //      entries (`int-dot`, `det.igemm`) via (4), and those are internal
        //      lowering tiers never written as a bare callee anyway;
        //   3. `tensor.*` qualified builtins (`tensor.zeros`, `tensor.matmul`);
        //   4. the std-surface intrinsic table proper (`std_surface_intrinsic_arity`);
        //   5. the cross-module imported-fn table (`cm_lookup_fn`);
        //   6. `gen_deref` — the RFC 0010 generational-ref deref builtin;
        //   7. BARE_BUILTINS — the closed set of bare math / tensor / autodiff
        //      builtin names the corpus calls without an import or a `tensor.`
        //      qualifier. EMPIRICALLY SWEPT (see the BARE_BUILTINS doc-comment):
        //      `mindc check std/ examples/` was run under both feature configs
        //      and every legitimate builtin it flagged was folded in here.
        // Anything matching none of these is a genuinely-undefined call → E2003.
        if self.ident_resolvable(name) {
            return true;
        }
        // A QUALIFIED path callee (`Vec::new`, `String::with_capacity`,
        // `Vec::with_capacity`) references a *type's associated function*, not a
        // bare module-level name. The parser folds the `::` segments into one
        // contiguous callee string. This pass owns the BARE undefined-reference
        // question (E2003) — verifying that an associated function actually
        // exists on a type is a separate type-resolution concern — so a callee
        // carrying `::` is never a bare undefined reference and must not be
        // flagged. A bare `foo()` (no `::`) is still fully checked below.
        if name.contains("::") {
            return true;
        }
        if name.starts_with("__mind_") || name.starts_with("tensor.") {
            return true;
        }
        if name == "gen_deref" {
            return true;
        }
        if BARE_BUILTINS.binary_search(&name).is_ok() {
            return true;
        }
        #[cfg(feature = "std-surface")]
        if super::std_surface_intrinsic_arity(name).is_some() {
            return true;
        }
        #[cfg(feature = "cross-module-imports")]
        if super::cm_lookup_fn(name).is_some() {
            return true;
        }
        false
    }

    /// Register every binding introduced by a pattern (`match` arms). Bare
    /// idents and enum-variant payload sub-patterns bind names; literals and
    /// `_` bind nothing.
    fn bind_pattern(&mut self, pat: &Pattern) {
        match pat {
            Pattern::Ident(name) => self.scopes.bind(name),
            Pattern::EnumVariant { args, .. } => {
                for a in args {
                    self.bind_pattern(a);
                }
            }
            Pattern::Tuple(elems) => {
                for e in elems {
                    self.bind_pattern(e);
                }
            }
            Pattern::Literal(_) | Pattern::Wildcard => {}
        }
    }

    /// Walk a sequence of statements as a fresh lexical scope: push a frame,
    /// visit each statement in order (so a `let` is visible only to later
    /// statements), then pop.
    fn walk_block(&mut self, stmts: &[Node]) {
        self.scopes.push();
        for s in stmts {
            self.walk(s);
        }
        self.scopes.pop();
    }

    /// Visit a node: register any binding it introduces (into the CURRENT
    /// frame, for sequential visibility) and recurse into sub-expressions,
    /// reporting unresolved references.
    fn walk(&mut self, node: &Node) {
        match node {
            // ── Reference sites ───────────────────────────────────────────
            Node::Lit(Literal::Ident(name), span) => {
                if !self.ident_resolvable(name) {
                    let suggestion = suggest(name, &self.scopes, self.syms);
                    self.out.push(Unresolved {
                        name: name.clone(),
                        span: *span,
                        is_call: false,
                        suggestion,
                    });
                }
            }
            Node::Lit(_, _) => {}
            Node::Call { callee, args, span } => {
                if !self.call_resolvable(callee) {
                    let suggestion = suggest(callee, &self.scopes, self.syms);
                    self.out.push(Unresolved {
                        name: callee.clone(),
                        span: *span,
                        is_call: true,
                        suggestion,
                    });
                }
                // Skip arg descent for tensor builtins whose args carry
                // dtype/shape identifier literals (not variables).
                if !is_builtin_callee(callee) {
                    for a in args {
                        self.walk(a);
                    }
                }
            }

            // ── Binding sites (register into the current frame) ───────────
            Node::Let { name, value, .. } => {
                // RHS is evaluated before the name is in scope.
                self.walk(value);
                self.scopes.bind(name);
            }
            Node::Assign { name, value, .. } => {
                // `name` is an l-value; an undefined assign target is a
                // distinct concern (and the parser/lowerer handles it). We
                // only resolve the RHS here to avoid false positives on
                // forward-declared module state.
                self.walk(value);
                let _ = name;
            }
            Node::For {
                var,
                start,
                end,
                body,
                ..
            } => {
                self.walk(start);
                self.walk(end);
                self.scopes.push();
                self.scopes.bind(var);
                for s in body {
                    self.walk(s);
                }
                self.scopes.pop();
            }
            Node::Match {
                scrutinee, arms, ..
            } => {
                self.walk(scrutinee);
                for arm in arms {
                    self.scopes.push();
                    self.bind_pattern(&arm.pattern);
                    self.walk(&arm.body);
                    self.scopes.pop();
                }
            }

            // ── Control flow / blocks (fresh frames) ──────────────────────
            Node::Block { stmts, .. } => self.walk_block(stmts),
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                self.walk(cond);
                self.walk_block(then_branch);
                if let Some(eb) = else_branch {
                    self.walk_block(eb);
                }
            }
            #[cfg(feature = "std-surface")]
            Node::While { cond, body, .. } => {
                self.walk(cond);
                self.walk_block(body);
            }
            #[cfg(feature = "std-surface")]
            Node::Region { body, .. } => self.walk_block(body),
            #[cfg(feature = "std-surface")]
            Node::Break { .. } | Node::Continue { .. } => {}
            Node::Return { value, .. } => {
                if let Some(v) = value {
                    self.walk(v);
                }
            }

            // ── Compound expressions: recurse into child expressions ──────
            Node::Binary { left, right, .. }
            | Node::Logical { left, right, .. }
            | Node::Bitwise { left, right, .. } => {
                self.walk(left);
                self.walk(right);
            }
            Node::Paren(inner, _)
            | Node::Neg { operand: inner, .. }
            | Node::Not { operand: inner, .. }
            | Node::Ref { inner, .. } => {
                self.walk(inner);
            }
            Node::As { expr, .. } => self.walk(expr),
            Node::Tuple { elements, .. } | Node::ArrayLit { elements, .. } => {
                for e in elements {
                    self.walk(e);
                }
            }
            Node::Print { args, .. } => {
                for a in args {
                    self.walk(a);
                }
            }
            Node::Assert { cond, .. } => self.walk(cond),
            Node::MethodCall { receiver, args, .. } => {
                // The method name is a member, not a binding. Resolve the
                // receiver and the value arguments only.
                self.walk(receiver);
                for a in args {
                    self.walk(a);
                }
            }
            Node::FieldAccess { receiver, .. } => self.walk(receiver),
            Node::FieldAssign {
                receiver, value, ..
            } => {
                self.walk(receiver);
                self.walk(value);
            }
            Node::IndexAccess {
                receiver, index, ..
            } => {
                self.walk(receiver);
                self.walk(index);
            }
            Node::IndexAssign {
                receiver,
                index,
                value,
                ..
            } => {
                self.walk(receiver);
                self.walk(index);
                self.walk(value);
            }
            Node::StructLit { fields, .. } => {
                // Field names are members; only the field VALUES are exprs.
                for f in fields {
                    self.walk(&f.value);
                }
            }

            // ── Tensor-builtin AST variants: recurse into expr children ───
            // These carry only `Box<Node>` value children (no bare dtype/shape
            // idents), so descending is safe.
            Node::CallGrad { loss, .. } => self.walk(loss),
            Node::CallTensorSum { x, .. }
            | Node::CallTensorMean { x, .. }
            | Node::CallReshape { x, .. }
            | Node::CallExpandDims { x, .. }
            | Node::CallSqueeze { x, .. }
            | Node::CallTranspose { x, .. }
            | Node::CallIndex { x, .. }
            | Node::CallSlice { x, .. }
            | Node::CallSliceStride { x, .. }
            | Node::CallTensorRelu { x, .. } => self.walk(x),
            Node::CallGather { x, idx, .. } => {
                self.walk(x);
                self.walk(idx);
            }
            Node::CallDot { a, b, .. } | Node::CallMatMul { a, b, .. } => {
                self.walk(a);
                self.walk(b);
            }
            Node::TensorMatmul { lhs, rhs, .. } | Node::TensorElemwise { lhs, rhs, .. } => {
                self.walk(lhs);
                self.walk(rhs);
            }
            Node::CallTensorConv2d { x, w, .. } => {
                self.walk(x);
                self.walk(w);
            }

            // ── Leaf / non-reference nodes ────────────────────────────────
            // Declarations and constructs that introduce no resolvable
            // references inside a fn body (or are handled at module level).
            Node::FnDef { .. }
            | Node::Const { .. }
            | Node::TypeAlias { .. }
            | Node::Export { .. }
            | Node::StructDef { .. }
            | Node::EnumDef { .. }
            | Node::Import { .. }
            | Node::ExternBlock { .. }
            | Node::CallTensorRand { .. } => {}
        }
    }
}

/// Resolve undefined references inside one fn body. `params` are pre-bound into
/// the base frame; `injected` is the set of cross-module symbol names already
/// merged into the type env (empty in the single-file check path).
pub fn resolve_fn_body(
    body: &[Node],
    param_names: &[String],
    module: &Module,
    injected: &BTreeSet<String>,
) -> Vec<Unresolved> {
    debug_assert!(
        BARE_BUILTINS.windows(2).all(|w| w[0] < w[1]),
        "BARE_BUILTINS must be sorted ascending and unique for binary_search"
    );
    let syms = collect_module_syms(module, injected);
    let mut scopes = Scopes::new();
    for p in param_names {
        scopes.bind(p);
    }
    let mut resolver = Resolver {
        syms: &syms,
        scopes,
        out: Vec::new(),
    };
    for stmt in body {
        resolver.walk(stmt);
    }
    resolver.out
}
