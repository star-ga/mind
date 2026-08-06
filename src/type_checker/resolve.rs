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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

use crate::ast::{Literal, Module, Node, Pattern, TypeAnn};

/// Deterministic FxHash-backed tables for the lookup-only symbol sets built
/// per `resolve_fn_body` call (the type checker's `FxBuild` — the same
/// deterministic hasher swap that sped the per-compile side tables). BTree
/// membership tests are O(log n) pointer-chases over per-node allocations;
/// these sets are queried once per identifier/call in the body walk, so the
/// flat open-addressed layout is the structural win. Every iteration over
/// these structures feeds only order-independent consumers (`suggest`
/// tie-breaks lexicographically on equal edit distance, so candidate order
/// never reaches output; the std caches feed a set union), and the diagnostic
/// vectors are filled in AST-walk order — emitted bytes and diagnostics are
/// unchanged.
type FxSet = HashSet<String, super::FxBuild>;
type FxMap<V> = HashMap<String, V, super::FxBuild>;

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
/// Unknown-enum-variant diagnostic code. A qualified `Enum::Variant` path whose
/// HEAD names a locally-declared enum but whose variant is not one of that
/// enum's declared variants — a silent-miscompile typo (the lowerer would fold
/// the unknown variant to discriminant tag 0). Deferred for non-local /
/// cross-module / associated-fn heads (those are a separate resolution concern).
pub const UNKNOWN_VARIANT_CODE: &str = "E2008";
/// Undeclared-assignment-target diagnostic code. An assignment whose l-value
/// name is bound NOWHERE — not in any active local scope frame and not a
/// module-level name. The loose lowerer would otherwise silently auto-create a
/// fresh variable, turning a typo (`conter = counter + 1`) into wrong output.
pub const UNDECLARED_ASSIGN_CODE: &str = "E2009";
/// Function-value-call diagnostic code. A call whose callee resolves ONLY as a
/// local lexical binding (`let f = add1` / a value param) and to NOTHING
/// emittable — not a module fn, intrinsic (`__mind_*`), tensor.* builtin,
/// BARE_BUILTIN, std-surface intrinsic, cross-module import, or `::`-qualified
/// path. First-class functions do not exist yet: the front-end used to ACCEPT
/// such a call and the backend then synthesised a call to an undefined
/// `func.func private @<name>`, so `--emit-shared` produced an .so with an
/// UNDEFINED symbol (`nm -D` shows `U f`) that fails at dlopen. Fail loud at
/// COMPILE instead of at link.
pub const FN_VALUE_CALL_CODE: &str = "E2012";

/// Self-host-only-intrinsic diagnostic code. A call whose callee resolves
/// (via `call_resolvable`'s blanket `__mind_*` acceptance) but is NOT one of
/// the arity-checked, cross-backend `STD_SURFACE_INTRINSICS` entries. Advisory
/// only — `mindc check` still passes (this is a Warning, not an Error), but it
/// tells the user `mindc build`'s Rust/MLIR backends cannot emit this call.
/// Before this code existed, such a call was silently accepted with zero
/// diagnostic at check-time, then either failed to link or (via the
/// launcher-script fallback in `src/project/mod.rs`) produced a script instead
/// of a native binary while `mindc build` still reported "Finished" exit 0.
pub const SELF_HOST_ONLY_CALL_CODE: &str = "E2024";

/// An unresolved reference found by the pass, with the kind that selects the
/// diagnostic code/message. Spans are AST byte-offset spans; the caller renders
/// them through the same `diag_from_span` localization as every other code.
pub struct Unresolved {
    pub name: String,
    pub span: crate::ast::Span,
    pub is_call: bool,
    /// The closest in-scope name (edit-distance "did you mean"), if any.
    pub suggestion: Option<String>,
    /// `Some(enum_name)` iff this is a qualified `Enum::Variant` path whose head
    /// names a locally-declared enum but whose variant is unknown — selects the
    /// `UNKNOWN_VARIANT_CODE` diagnostic instead of the bare-undefined codes.
    pub variant_of: Option<String>,
    /// `true` iff this is an assignment to a name bound nowhere — selects the
    /// `UNDECLARED_ASSIGN_CODE` diagnostic.
    pub undeclared_assign: bool,
    /// `true` iff this is a call whose callee resolves ONLY as a local lexical
    /// binding (a function value) and to nothing emittable — selects the
    /// `FN_VALUE_CALL_CODE` diagnostic.
    pub fn_value_call: bool,
    /// `true` iff this is a call whose callee is a MODULE-LEVEL NON-FUNCTION
    /// declaration (const / module-`let` / struct / enum-type / type-alias name)
    /// and resolves to nothing emittable as a function — the module-scope twin
    /// of `fn_value_call`. Selects the `FN_VALUE_CALL_CODE` diagnostic (same
    /// broken-artifact class) with an "`X` is not a function" message.
    pub non_fn_call: bool,
}

/// A call that resolves (never blocks `mindc check`) but whose callee is only
/// accepted through `call_resolvable`'s blanket `__mind_*` acceptance rather
/// than a registered, arity-checked `STD_SURFACE_INTRINSICS` entry — today
/// that's the native-ELF self-host-only intrinsics (`__mind_argc`/
/// `__mind_argv`/`__mind_open`), but it also catches a genuinely-misspelled
/// `__mind_*` name, which otherwise had zero diagnostic coverage at all.
pub struct SelfHostOnlyCall {
    pub name: String,
    pub span: crate::ast::Span,
}

/// The module-level resolution context, built once per module.
struct ModuleSyms {
    /// All names resolvable without a local binding: module fns/consts/lets,
    /// struct/enum/type-alias names, in-file extern fn names, and (when the
    /// project table is populated) cross-module imported symbols.
    names: FxSet,
    /// Locally-declared enums: enum name -> its declared variant-name set. Used
    /// ONLY to validate a qualified `Enum::Variant` value/constructor path whose
    /// head is one of THESE enums; a head not in this map (imported enum,
    /// associated fn on a struct, cross-module type) is deferred as before.
    enums: FxMap<FxSet>,
    /// Module-level declaration names that are NOT functions and NOT callable
    /// constructors — const / extern-const / module-`let` / struct / enum-TYPE /
    /// type-alias names, minus any name that also has a function definition.
    /// A call whose callee resolves only to one of these targets a data/global
    /// symbol; see `is_module_non_fn_call`.
    ///
    /// Membership-only (`contains` / `retain`), never iterated in a way that
    /// reaches emitted output or diagnostic order, so an `FxSet` (matching the
    /// `names`/`enums` siblings) is byte-neutral and drops the per-`contains`
    /// O(log n) BTree string-compare to O(1).
    non_fn_decls: FxSet,
}

impl ModuleSyms {
    /// A name is module-resolvable iff it is a per-module declaration/import name
    /// OR a std-surface export. The std set is checked separately (not copied
    /// into `names`) so building `ModuleSyms` stays O(module decls), not
    /// O(module decls + whole std surface), on every recursive check call.
    fn name_resolvable(&self, name: &str) -> bool {
        self.names.contains(name) || stdlib_export_names().contains(name)
    }
}

/// Collect the top-level declaration names a module defines (fn / const / let /
/// struct / enum / type-alias / extern), recursing into the `Block` that a
/// `module NAME { ... }` body unwraps into. Pure: it does NOT resolve imports,
/// so it is safe to call while building the std-export cache (no re-entrancy).
pub(crate) fn collect_decl_names(module: &Module, out: &mut BTreeSet<String>) {
    collect_decl_names_into(module, out);
}

/// Set-generic body of [`collect_decl_names`] so the per-compile FxHash-backed
/// tables are populated directly instead of being rebuilt from a BTree copy.
fn collect_decl_names_into<S: Extend<String>>(module: &Module, out: &mut S) {
    for item in &module.items {
        match item {
            Node::FnDef(fd, _) => {
                out.extend([fd.name.clone()]);
            }
            Node::Const { name, .. }
            | Node::ExternConst { name, .. }
            | Node::Let { name, .. }
            | Node::StructDef { name, .. }
            | Node::TypeAlias { name, .. } => {
                out.extend([name.clone()]);
            }
            Node::EnumDef { name, variants, .. } => {
                // The enum name, plus per variant BOTH the qualified `Enum::Variant`
                // form AND the BARE name. The qualified form (matching the parser's
                // `Enum.Variant` → `Enum::Variant` normalisation) is what keeps
                // variant resolution precise: `Res::Ok` resolves to exactly this
                // enum, so two enums that share a variant name (`Ok`, `None`) no
                // longer alias when referenced qualified. The bare name is kept so
                // an UNQUALIFIED constructor or unit value (`Some(x)`, `None`,
                // `Ok(v)`, `Err(e)`) still resolves — MIND links all modules into
                // one unit and the exact tag/enum is recovered in lowering. (This
                // pass owns only the undefined-reference question; verifying a
                // bare variant belongs to the EXPECTED enum is a later type-check
                // concern.)
                out.extend([name.clone()]);
                for v in variants {
                    out.extend([format!("{name}::{}", v.name)]);
                    out.extend([v.name.clone()]);
                }
            }
            Node::ExternBlock { fns, .. } => {
                for efn in fns {
                    out.extend([efn.name.clone()]);
                }
            }
            Node::Block { stmts, .. } => {
                let inner = Module {
                    items: stmts.clone(),
                };
                collect_decl_names_into(&inner, out);
            }
            _ => {}
        }
    }
}

/// Collect every locally-declared enum's variant-name set, recursing into the
/// `Block` that a `module NAME { ... }` body unwraps into. Mirrors
/// `collect_decl_names`'s recursion so an enum declared inside a `module { }`
/// block is registered too. Pure metadata — never resolves imports.
fn collect_enum_variants(module: &Module, out: &mut FxMap<FxSet>) {
    for item in &module.items {
        match item {
            Node::EnumDef { name, variants, .. } => {
                let set = out.entry(name.clone()).or_default();
                for v in variants {
                    set.insert(v.name.clone());
                }
            }
            Node::Block { stmts, .. } => {
                let inner = Module {
                    items: stmts.clone(),
                };
                collect_enum_variants(&inner, out);
            }
            _ => {}
        }
    }
}

/// Collect the module-level declaration names that are NOT functions and NOT
/// callable constructors — const / extern-const / module-`let` / struct /
/// enum-TYPE / type-alias names. A call whose callee resolves ONLY to one of
/// these (and to nothing emittable as a function) targets a data/global symbol;
/// the lowerer would synthesise `func.call @<name>` against a non-callable
/// global, and `--emit-shared` then yields an undefined/non-callable `.so` —
/// the exact E2012 broken-artifact class. Enum VARIANT names (bare + qualified)
/// are deliberately EXCLUDED: they are valid unit/payload constructors
/// (`Some(x)`, `Ok(v)`, `Color::Red`). Function and extern-function names are
/// recorded in `fns` so a name that also has a function definition is never
/// treated as a non-function (defensive against a same-name redeclaration).
/// Recurses into `module { }` blocks exactly like `collect_decl_names`.
fn collect_non_fn_decl_names(module: &Module, out: &mut FxSet, fns: &mut FxSet) {
    for item in &module.items {
        match item {
            Node::FnDef(fd, _) => {
                fns.insert(fd.name.clone());
            }
            Node::ExternBlock { fns: efns, .. } => {
                for efn in efns {
                    fns.insert(efn.name.clone());
                }
            }
            Node::Const { name, .. }
            | Node::ExternConst { name, .. }
            | Node::Let { name, .. }
            | Node::StructDef { name, .. }
            | Node::TypeAlias { name, .. }
            | Node::EnumDef { name, .. } => {
                out.insert(name.clone());
            }
            Node::Block { stmts, .. } => {
                let inner = Module {
                    items: stmts.clone(),
                };
                collect_non_fn_decl_names(&inner, out, fns);
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
fn stdlib_exports() -> &'static FxMap<FxSet> {
    static CACHE: OnceLock<FxMap<FxSet>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // `project::stdlib` (the bundled std-surface registry) is gated behind
        // `cross-module-imports`. With the feature off there is no std surface
        // to resolve, so this map stays empty — keeps `--no-default-features`
        // compiling while leaving the feature-on behaviour byte-for-byte
        // unchanged.
        #[allow(unused_mut)]
        let mut map = FxMap::default();
        #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
        for (path, module) in crate::project::stdlib::parsed_stdlib_modules() {
            let mut names = FxSet::default();
            collect_decl_names_into(&module, &mut names);
            map.insert(path, names);
        }
        map
    })
}

/// The flattened set of every std-surface export name, built once.
///
/// The std surface is constant across a compile, so its (large) name set must
/// NOT be copied into every per-module `ModuleSyms::names` — `collect_module_syms`
/// is run once per `check_module_types_in_file` invocation, and that function
/// recurses per top-level item / block / fn body, so re-`extend`-ing thousands of
/// std names into a fresh `BTreeSet` each call was O(items x std_names) — the
/// dominant cost of `mindc check main.mind`. Instead cache the flat set here and
/// check membership against it separately (see `ModuleSyms::name_resolvable` and
/// the `suggest` candidate chain). Diagnostics are unchanged: a name is resolvable
/// iff it is in the per-module set OR this std set (same union as before), and the
/// suggestion candidate set is the same (`closest_identifier` tie-breaks
/// lexicographically, so iteration order never reaches output).
fn stdlib_export_names() -> &'static FxSet {
    static CACHE: OnceLock<FxSet> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut s = FxSet::default();
        for exports in stdlib_exports().values() {
            s.extend(exports.iter().cloned());
        }
        s
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
    let mut names: FxSet = injected.iter().cloned().collect();
    collect_decl_names_into(module, &mut names);
    // Built-in Result/Option prelude: the bare variant names are always
    // resolvable (the compiler registers the enums in the lowering side-tables;
    // they have no source-level `EnumDef` to feed `collect_decl_names`). A user
    // module that defines its own `Result`/`Option` already adds these via
    // `collect_decl_names` — this only ever ADDS names, never removes.
    for v in ["Result", "Option", "Ok", "Err", "Some", "None"] {
        names.insert(v.to_string());
    }
    // The std-surface is the language standard library: its public names are
    // always resolvable. std modules reference each other WITHOUT explicit
    // imports (they are compiled as one bundle — e.g. std/io.mind calls
    // std.string's `string_push_byte` with no import line), and a std-surface
    // function is never a *genuinely-undefined* reference. So the whole surface
    // is visible rather than per-import. (Whether a user module should be forced
    // to `import std.X` before calling it is a separate "missing import" lint,
    // not the undefined-reference question issue #23 owns.)
    //
    // The std name set is CONSTANT across the compile, so it is NOT copied into
    // this per-module `names` — that copy was O(items x std_names), the dominant
    // cost of `mindc check main.mind`. It is checked separately via
    // `stdlib_export_names()` in `name_resolvable` and the `suggest` chain.
    let mut enums: FxMap<FxSet> = FxMap::default();
    collect_enum_variants(module, &mut enums);
    // Module-level non-function decls, used to reject a call whose callee names a
    // data/global symbol (const/let/struct/enum-type/type-alias) rather than a
    // function. Function names are collected alongside and subtracted so a name
    // that also has a `fn` definition is never treated as a non-function.
    let mut non_fn_decls: FxSet = FxSet::default();
    let mut fn_decls: FxSet = FxSet::default();
    collect_non_fn_decl_names(module, &mut non_fn_decls, &mut fn_decls);
    non_fn_decls.retain(|n| !fn_decls.contains(n));
    ModuleSyms {
        names,
        enums,
        non_fn_decls,
    }
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
        // A user-named annotation can be a *type alias for a shape variable*:
        // `type N = u32` then `fn f(x: [i64; N]) -> i64 { reshape(x, [N]) }`
        // parses the dim/extent `N` as `Named("N")`, and that same `N` is read
        // in the body. Pre-bind the name so the body reference resolves instead
        // of false-positiving E2002. This only ADDS a resolvable name: when the
        // name is instead a real struct/enum/alias type it is ALREADY a module
        // decl-name (so the union is unchanged), and a never-referenced type
        // name in the set is harmless — so this can never produce a false
        // positive, matching the rest of this collector's contract.
        TypeAnn::Named(name) => {
            out.insert(name.clone());
        }
        // Scalars carry no shape dimensions.
        TypeAnn::ScalarI32
        | TypeAnn::ScalarI64
        | TypeAnn::ScalarF32
        | TypeAnn::ScalarF64
        | TypeAnn::ScalarBool
        | TypeAnn::ScalarU32 => {}
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
    frames: Vec<FxSet>,
}

impl Scopes {
    fn new() -> Self {
        Scopes {
            frames: vec![FxSet::default()],
        }
    }
    fn push(&mut self) {
        self.frames.push(FxSet::default());
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

/// True ONLY for the dtype/shape-LITERAL tensor builtins — those whose leading
/// `Node::Call` argument is a bare dtype identifier (`f32`, `f64`, …) rather
/// than a value expression (`tensor.zeros(f32, (..))`). We must not descend into
/// these args as references: `f32` is not a resolvable variable, so walking it
/// would false-positive E2002.
///
/// CRIT — missed diagnostic: this used to return `true` for ALL `tensor.*`
/// callees, so the resolver skipped the value arguments of *every* tensor
/// builtin (e.g. `tensor.sum(x, undefined_var)`) and an undefined identifier
/// inside them was never reported. The skip is now scoped to exactly the
/// dtype-literal constructors; all other `tensor.*` generic calls have their
/// value arguments walked, so undefined idents are caught (E2002).
///
/// `tensor.rand` parses to a dedicated `Node::CallTensorRand` leaf and never
/// reaches this generic-call path, but it is listed for documentation /
/// forward-compatibility (if it ever falls back to a generic call its first
/// arg is still a dtype literal). Likewise `tensor.full` is reserved for the
/// same `(dtype, shape, fill)` constructor shape.
fn is_dtype_literal_builtin(callee: &str) -> bool {
    matches!(
        callee,
        "tensor.zeros" | "tensor.ones" | "tensor.full" | "tensor.rand"
    )
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
        .chain(syms.names.iter().map(String::as_str))
        .chain(stdlib_export_names().iter().map(String::as_str));
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

/// Feature-neutral shim over the cross-module export lookup. On a `mindc`
/// built without `cross-module-imports` there is no project table, so this is a
/// constant `false` and the resolver behaves byte-identically to the pre-change
/// single-file path.
#[cfg(feature = "cross-module-imports")]
#[inline]
fn cm_symbol_exported_res(name: &str) -> bool {
    super::cm_symbol_exported(name)
}
#[cfg(not(feature = "cross-module-imports"))]
#[inline]
fn cm_symbol_exported_res(_name: &str) -> bool {
    false
}

/// Resolver state threaded through the body walk.
struct Resolver<'a> {
    syms: &'a ModuleSyms,
    scopes: Scopes,
    out: Vec<Unresolved>,
    self_host_only: Vec<SelfHostOnlyCall>,
}

impl<'a> Resolver<'a> {
    fn ident_resolvable(&self, name: &str) -> bool {
        self.scopes.contains(name)
            || self.syms.name_resolvable(name)
            // `bytes` is a builtin fixed-byte-buffer type usable as a value base
            // for `bytes[N].zero()` (a zeroed N-byte buffer). A user binding named
            // `bytes` shadows it via the `scopes` check above.
            || name == "bytes"
            // A qualified path used as a *value* — an enum-variant constructor
            // (`Mode::On`), an associated const, etc. — folds its `::` segments
            // into one ident string. This pass owns the BARE undefined-reference
            // question (E2002); verifying that a `Type::Member` actually exists is
            // a separate type-resolution concern. Mirror `call_resolvable`'s `::`
            // treatment so a qualified value reference is never a bare-undefined.
            || name.contains("::")
            // A symbol exported by a SIBLING project module (cross-module import).
            // Covers module-qualified consts the parser normalised to bare names
            // (`fixed_point.Q16_ONE` → `Q16_ONE`) and any other exported value.
            // Empty (false) on the single-file / default path (no project table).
            || cm_symbol_exported_res(name)
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
        // A name explicitly `export`ed by a sibling module (e.g. an
        // `export fn q16_add` inside `module fixed_point { ... }`, which the
        // parser normalises a `fixed_point.q16_add(...)` call down to). The
        // explicit-export surface carries names for every kind, whereas
        // `cm_lookup_fn` only sees typed fn signatures — so this catches the
        // `export`-block fn form the signature table does not.
        if cm_symbol_exported_res(name) {
            return true;
        }
        false
    }

    /// A CALL whose callee resolves ONLY as a local lexical binding (a
    /// `let f = ...` name or a value param — a *function value*) and to nothing
    /// the backend can emit a direct call to. First-class functions do not
    /// exist yet: `call_resolvable` accepts such a callee (item 1 of its union,
    /// anticipating closures/fn-ptrs), and the lowerer then synthesises a call
    /// to an undefined `func.func private @<name>`, so `--emit-shared` yields an
    /// .so with an UNDEFINED symbol. We reject it at compile (E2012) instead.
    ///
    /// Sound-condition only — mirrors the E2008/E2009 discipline. Returns true
    /// iff the callee is a local binding AND is NOT resolvable through ANY
    /// emittable source (module fn/const/struct/extern/cross-module-injected +
    /// std surface, a `::`-qualified associated path, the `bytes` value-type,
    /// `__mind_*` / `tensor.*` / `gen_deref` builtins, BARE_BUILTINS, the
    /// std-surface intrinsic table, or the cross-module imported-fn table). If
    /// it ALSO resolves as any of those, we defer to the emittable meaning and
    /// never reject — so this can only fire on a genuine function-value call.
    /// The extern-C fn-ptr PARAM path (RFC 0010 Phase B) declares/type-checks
    /// `fn(T)->R`-typed params but never calls through the value, so this never
    /// fires there.
    fn is_fn_value_call(&self, name: &str) -> bool {
        if !self.scopes.contains(name) {
            return false;
        }
        if self.syms.name_resolvable(name)
            || name.contains("::")
            || name == "bytes"
            || name.starts_with("__mind_")
            || name.starts_with("tensor.")
            || name == "gen_deref"
            || BARE_BUILTINS.binary_search(&name).is_ok()
        {
            return false;
        }
        #[cfg(feature = "std-surface")]
        if super::std_surface_intrinsic_arity(name).is_some() {
            return false;
        }
        #[cfg(feature = "cross-module-imports")]
        if super::cm_lookup_fn(name).is_some() {
            return false;
        }
        true
    }

    /// A CALL whose callee is a MODULE-LEVEL NON-FUNCTION declaration (const /
    /// extern-const / module-`let` / struct / enum-TYPE / type-alias name) and
    /// resolves to nothing emittable as a function. The module-scope twin of
    /// `is_fn_value_call`: that guard only inspects LOCAL `scopes`, so a call to
    /// a module-level const/struct/enum name slips through — the name is in
    /// `syms.names` (so `call_resolvable` accepts it → no E2003), yet it is not a
    /// function, so the lowerer synthesises `func.call @<name>` to a data/global
    /// symbol and `--emit-shared` yields an undefined/non-callable `.so`. Reject
    /// at compile (E2012) instead of at link.
    ///
    /// Fail-CLOSED sound condition: fires ONLY when the callee is a known non-fn
    /// module decl AND resolves through NO emittable function source (a
    /// `::`-qualified associated path, `bytes`, `__mind_*` / `tensor.*` /
    /// `gen_deref` builtins, `BARE_BUILTINS`, the std-surface intrinsic table, or
    /// the cross-module imported-fn table). Enum VARIANT constructors are never
    /// in `non_fn_decls`, so `Some(x)` / `Ok(v)` / `Color::Red` never fire here.
    /// `name_resolvable` is intentionally NOT consulted — a non-fn module decl
    /// IS name-resolvable (that is exactly the leak this closes).
    fn is_module_non_fn_call(&self, name: &str) -> bool {
        if !self.syms.non_fn_decls.contains(name) {
            return false;
        }
        if name.contains("::")
            || name == "bytes"
            || name.starts_with("__mind_")
            || name.starts_with("tensor.")
            || name == "gen_deref"
            || BARE_BUILTINS.binary_search(&name).is_ok()
        {
            return false;
        }
        #[cfg(feature = "std-surface")]
        if super::std_surface_intrinsic_arity(name).is_some() {
            return false;
        }
        #[cfg(feature = "cross-module-imports")]
        if super::cm_lookup_fn(name).is_some() {
            return false;
        }
        true
    }

    /// The E2009 (`UNDECLARED_ASSIGN`) predicate: an assignment target name that
    /// is PROVABLY neither (i) bound in any active local scope frame NOR (ii) a
    /// known module-level name (fn/const/let/struct/enum/type-alias/extern +
    /// injected cross-module symbols). The loose lowerer auto-creates a fresh
    /// variable for an assignment to such an unbound name, turning a typo
    /// (`conter = counter + 1`) into wrong runtime output with no diagnostic;
    /// flagging it fails that shape closed. Forward-declared module state is a
    /// module-level name, so it still resolves — when in doubt (any known name),
    /// we keep accepting.
    ///
    /// Pure sibling of `ident_resolvable`/`call_resolvable`/`is_fn_value_call`:
    /// the negated scope-frame membership plus module-symbol resolvability, with
    /// no I/O or diagnostic emission — the caller owns the `Unresolved` push.
    fn is_undeclared_assign(&self, name: &str) -> bool {
        !self.scopes.contains(name) && !self.syms.name_resolvable(name)
    }

    /// Classify a qualified `Enum::Variant` value/constructor path. Returns
    /// `Some(enum_name)` ONLY when the head names a locally-declared enum but the
    /// trailing segment is NOT one of that enum's variants — a typo the lowerer
    /// would silently fold to discriminant tag 0. Everything else returns `None`
    /// (bare names, deeper `a::b::c` paths, and heads that are not a local enum —
    /// imported enums, struct associated fns, cross-module types — which remain a
    /// deferred separate-resolution concern, never a false positive here).
    fn unknown_variant(&self, name: &str) -> Option<String> {
        let (head, rest) = name.split_once("::")?;
        // Only a simple two-segment `Enum::Variant` path is validated; a deeper
        // module-qualified path (`a::B::C`) is deferred.
        if rest.contains("::") {
            return None;
        }
        let variants = self.syms.enums.get(head)?;
        if variants.contains(rest) {
            None
        } else {
            Some(head.to_string())
        }
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
            Pattern::EnumStruct { fields, .. } => {
                for (_, sub) in fields {
                    self.bind_pattern(sub);
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
                // A qualified `Enum::Variant` value whose head is a local enum
                // but whose variant is unknown is a silent-miscompile typo
                // (folds to tag 0) — flag it before the generic resolvability
                // check, which `::` otherwise makes pass unconditionally.
                if let Some(enum_name) = self.unknown_variant(name) {
                    let suggestion = suggest(name, &self.scopes, self.syms);
                    self.out.push(Unresolved {
                        name: name.clone(),
                        span: *span,
                        is_call: false,
                        suggestion,
                        variant_of: Some(enum_name),
                        undeclared_assign: false,
                        fn_value_call: false,
                        non_fn_call: false,
                    });
                } else if !self.ident_resolvable(name) {
                    let suggestion = suggest(name, &self.scopes, self.syms);
                    self.out.push(Unresolved {
                        name: name.clone(),
                        span: *span,
                        is_call: false,
                        suggestion,
                        variant_of: None,
                        undeclared_assign: false,
                        fn_value_call: false,
                        non_fn_call: false,
                    });
                }
            }
            Node::Lit(_, _) => {}
            Node::Call { callee, args, span } => {
                // Advisory, orthogonal to the resolvable/unresolved decision
                // below: a callee that resolves ONLY through the blanket
                // `__mind_*` acceptance in `call_resolvable` (not a registered
                // `STD_SURFACE_INTRINSICS` entry) gets zero arity/existence
                // checking from either this pass or the type-checker's
                // `check_std_surface_intrinsic`. Flag it as a Warning so
                // `mindc check` stays green (no behavior change for legitimate
                // self-host-only callers like `std/cli.mind`/`std/fs.mind`)
                // but the gap is no longer silent.
                #[cfg(feature = "std-surface")]
                if callee.starts_with("__mind_")
                    && super::std_surface_intrinsic_arity(callee).is_none()
                {
                    self.self_host_only.push(SelfHostOnlyCall {
                        name: callee.clone(),
                        span: *span,
                    });
                }
                // A payload-variant constructor typo (`Color::Rde(x)`) is the
                // call-position twin of the value-position case above: head is a
                // local enum, variant is unknown → silent fold to tag 0.
                if let Some(enum_name) = self.unknown_variant(callee) {
                    let suggestion = suggest(callee, &self.scopes, self.syms);
                    self.out.push(Unresolved {
                        name: callee.clone(),
                        span: *span,
                        is_call: true,
                        suggestion,
                        variant_of: Some(enum_name),
                        undeclared_assign: false,
                        fn_value_call: false,
                        non_fn_call: false,
                    });
                } else if self.is_fn_value_call(callee) {
                    // Calling a function value (`let f = add1  f(41)`): the
                    // callee resolves only as a local binding and to nothing
                    // emittable. Reject at compile (E2012) rather than letting
                    // the lowerer synthesise a call to an undefined extern.
                    self.out.push(Unresolved {
                        name: callee.clone(),
                        span: *span,
                        is_call: true,
                        suggestion: None,
                        variant_of: None,
                        undeclared_assign: false,
                        fn_value_call: true,
                        non_fn_call: false,
                    });
                } else if self.is_module_non_fn_call(callee) {
                    // Calling a module-level NON-function (`const ADD: i64 = 1
                    // ADD(2)`): the callee is a data/global symbol, not a fn, so
                    // the lowerer would synthesise `func.call @ADD` to a
                    // non-callable global. Reject at compile (E2012, same class)
                    // rather than emitting an undefined/non-callable `.so`.
                    self.out.push(Unresolved {
                        name: callee.clone(),
                        span: *span,
                        is_call: true,
                        suggestion: None,
                        variant_of: None,
                        undeclared_assign: false,
                        fn_value_call: false,
                        non_fn_call: true,
                    });
                } else if !self.call_resolvable(callee) {
                    let suggestion = suggest(callee, &self.scopes, self.syms);
                    self.out.push(Unresolved {
                        name: callee.clone(),
                        span: *span,
                        is_call: true,
                        suggestion,
                        variant_of: None,
                        undeclared_assign: false,
                        fn_value_call: false,
                        non_fn_call: false,
                    });
                }
                // Skip arg descent ONLY for the dtype-literal tensor
                // constructors, whose leading arg is a bare dtype ident
                // (`f32`) — not a variable. Every other call (including all
                // other `tensor.*` builtins) has its value args walked so an
                // undefined ident inside them is reported (E2002).
                if !is_dtype_literal_builtin(callee) {
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
            Node::LetTuple { names, value, .. } => {
                // RHS (the tuple) is evaluated before any name is in scope; then
                // every destructured name binds.
                self.walk(value);
                for nm in names {
                    self.scopes.bind(nm);
                }
            }
            Node::Assign { name, value, span } => {
                // `name` is an l-value. Resolve the RHS first.
                self.walk(value);
                // Silent-miscompile guard (see `is_undeclared_assign`): the loose
                // lowerer auto-creates a fresh variable for an assignment to an
                // unbound name, turning a typo (`conter = counter + 1`) into wrong
                // runtime output with no diagnostic. Flag the undeclared target.
                if self.is_undeclared_assign(name) {
                    let suggestion = suggest(name, &self.scopes, self.syms);
                    self.out.push(Unresolved {
                        name: name.clone(),
                        span: *span,
                        is_call: false,
                        suggestion,
                        variant_of: None,
                        undeclared_assign: true,
                        fn_value_call: false,
                        non_fn_call: false,
                    });
                }
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
            Node::ForEach {
                var,
                collection,
                body,
                ..
            } => {
                self.walk(collection);
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
                    // A guard resolves names with the pattern bindings in scope.
                    if let Some(guard) = &arm.guard {
                        self.walk(guard);
                    }
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
            | Node::BitNot { operand: inner, .. }
            | Node::Ref { inner, .. } => {
                self.walk(inner);
            }
            Node::As { expr, .. } => self.walk(expr),
            // W1.5f: postfix `?` binds nothing of its own — resolve `inner`.
            Node::Try { inner, .. } => self.walk(inner),
            Node::Tuple { elements, .. } | Node::ArrayLit { elements, .. } => {
                for e in elements {
                    self.walk(e);
                }
            }
            Node::MapLit { entries, .. } => {
                for (k, v) in entries {
                    self.walk(k);
                    self.walk(v);
                }
            }
            Node::SetLit { elements, .. } => {
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
                //
                // EXCEPTION: a static/associated type-name call
                // (`string.from_utf8_bytes(..)`) has a TYPE name as the receiver,
                // not a binding — don't flag it "unknown identifier `string`". A
                // real local of that name (resolvable) keeps the normal path.
                let static_type_recv = matches!(
                    receiver.as_ref(),
                    Node::Lit(Literal::Ident(n), _)
                        if (n == "string" || n == "String") && !self.ident_resolvable(n)
                );
                if !static_type_recv {
                    self.walk(receiver);
                }
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
            Node::SliceRange {
                receiver,
                start,
                end,
                ..
            } => {
                self.walk(receiver);
                self.walk(start);
                self.walk(end);
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
            // r1#4 (re-audited: NOT a resolve-pass gap). A bogus `wrt`
            // differentiation target is NOT unvalidated — type-checking already
            // reports it with a SPECIFIC diagnostic (E2001 "unknown tensor `X`
            // in `wrt`"). Adding a resolve-pass check here would only replace
            // that precise message with a generic E2002. Walk only `loss`; the
            // wrt targets are a type-check concern. tests/grad_wrt_resolve.rs
            // locks that the bogus target stays reported (exactly once).
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
            Node::FnDef(..)
            // #267: closures are desugared before the resolver runs.
            | Node::Closure(..)
            // #268: trait/impl are desugared (impls to free fns, traits dropped)
            // before the resolver runs.
            | Node::TraitDef { .. }
            | Node::ImplBlock { .. }
            | Node::Const { .. }
            | Node::ExternConst { .. }
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
/// merged into the type env (empty in the single-file check path). Returns the
/// genuinely-unresolved references alongside advisory self-host-only-intrinsic
/// call sites (see `SelfHostOnlyCall`) — the latter never blocks `mindc check`.
pub fn resolve_fn_body(
    body: &[Node],
    param_names: &[String],
    module: &Module,
    injected: &BTreeSet<String>,
) -> (Vec<Unresolved>, Vec<SelfHostOnlyCall>) {
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
        self_host_only: Vec::new(),
    };
    for stmt in body {
        resolver.walk(stmt);
    }
    (resolver.out, resolver.self_host_only)
}
