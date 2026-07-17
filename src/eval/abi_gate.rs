//! Runnable-artifact ABI gate (release-readiness P1.1).
//!
//! A pure, read-only pre-pass over the parsed + type-checked AST that fails
//! LOUD on the constructs the *runnable* (object / shared-library) lowering
//! path would otherwise SILENTLY MISCOMPILE — so a stranger never receives a
//! wrong-but-`rc=0` executable artifact. For a deterministic, evidence-signing
//! compiler a confidently-wrong `.so` is the worst outcome: it carries a valid
//! signature over incorrect code.
//!
//! Scope is exactly the i64-scalar ABI the shipped backend lowers correctly.
//! The gate is enforced ONLY when emitting a runnable artifact (`--emit-obj` /
//! `--emit-shared`). The inspection / IR surfaces (`mindc check`, `--emit-ir`,
//! `--emit-mlir`, `--emit-mic3`) are intentionally left untouched: there a
//! `i32`/`u32`/`tensor` annotation is a perfectly valid *type* (and the test
//! suite type-checks such programs heavily) — it simply does not yet *lower*
//! to a correct runnable artifact.
//!
//! The walk NEVER mutates the module/IR and produces zero diagnostics for an
//! all-i64 program, so the mic@3 self-host fixed point stays byte-identical
//! (keystone 7/7). Verified false-positive-free against `std/`, `examples/`,
//! and `examples/mindc_mind/main.mind`.
//!
//! Gate set (the *silent* sub-i64-ABI miscompiles; genuinely loud paths —
//! e.g. float struct fields, unsupported tensor ops — are left to the existing
//! lowering errors). Sub-i64 struct *fields* are no longer gated: the
//! width-aware struct ABI lowers them with a canonical offset table + typed
//! store/load. The remaining gated constructs:
//!   * a `tensor` / `diff tensor` function parameter or return: erases to the
//!     i64 ABI and is silently treated as a scalar integer.
//!   * a function parameter or return declared at a sub-i64 integer width
//!     (`i32`/`u32`/`i8`/`u8`/`i16`/`u16`): the width and, for unsigned types,
//!     the semantics are silently lost.
//!
//! `extern "C"` blocks are EXEMPT — the C-ABI boundary legitimately declares
//! narrow ints and maps them to the platform contract.

use crate::ast::{Module, Node, Span as AstSpan, TypeAnn};
use crate::diagnostics::{Diagnostic, Span};

const PHASE: &str = "lower";
const HELP: &str = "the shipped backend lowers only the i64-scalar ABI; this construct is not yet \
     lowerable to a runnable artifact (RUNS burndown). Run it with the `mind` interpreter, or keep \
     the compiled path to the i64 subset.";

/// Walk the module and return one error diagnostic per construct the runnable
/// (object / shared-library) lowering path would silently miscompile. The
/// returned `Vec` is empty for an all-i64 program (no allocation on the happy
/// path beyond the empty `Vec`), and the module is never mutated.
pub fn check_runnable_lowerable(module: &Module, src: &str, file: Option<&str>) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    // Only function signatures gate now. Struct declarations no longer gate: the
    // width-aware struct ABI lowers sub-i64 fields (i32/u32/i16/u16/i8/u8/bool)
    // with a canonical per-field offset table + typed store/load, and float
    // fields remain loud via the downstream non-i64-call check. `extern "C"`
    // declarations and every other item are exempt — no other top-level item
    // reaches the runnable path with a silently-miscompiled non-i64 signature.
    for item in &module.items {
        let Node::FnDef {
            name,
            params,
            ret_type,
            span,
            ..
        } = item
        else {
            continue;
        };
        for p in params {
            if let Some(reason) = param_non_i64(&p.ty) {
                out.push(mk(
                    src,
                    file,
                    p.span,
                    "lower::non_i64_param",
                    format!(
                        "parameter `{}` of `{name}` is not lowerable to a runnable artifact: \
                         {reason}",
                        p.name
                    ),
                ));
            }
        }
        if let Some(reason) = ret_type.as_ref().and_then(sig_non_i64) {
            out.push(mk(
                src,
                file,
                *span,
                "lower::non_i64_return",
                format!(
                    "return type of `{name}` is not lowerable to a runnable artifact: {reason}"
                ),
            ));
        }
    }
    out
}

/// Reason a function PARAMETER `TypeAnn` cannot lower in the runnable ABI, or
/// `None` when it lowers correctly. Identical to [`sig_non_i64`] EXCEPT that a
/// STATIC-SHAPE (all extents compile-time-known) `tensor` parameter is now
/// allowed: it lowers to a real memref/tensor C ABI (ptr + baked-in static
/// extents), not the erased i64 scalar. The `func.func` signature carries the
/// true `tensor<..>` type (`type_ann_to_abi_mlir`), the param seeds a real
/// `ValueKind::Tensor` (`type_ann_to_value_kind`), the build routes to the
/// `arith-linalg` preset whose `one-shot-bufferize{bufferize-function-
/// boundaries=true}` converts the boundary to a memref, and the pinned
/// reduction fold reads it via `tensor.extract %param[..]`. A DYNAMIC/symbolic
/// dim still gates (no static extent to bake into the memref descriptor), and a
/// `diff tensor` param still gates (autodiff boundary is a separate change).
/// Tensor RETURNS are unchanged — still routed through [`sig_non_i64`] and
/// gated (the out-param C ABI is a separate, larger slice).
fn param_non_i64(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        TypeAnn::Tensor { dims, .. } if tensor_dims_all_static(dims) => None,
        _ => sig_non_i64(ty),
    }
}

/// `true` when a tensor annotation's dims are all statically-known numeric
/// extents (`tensor<f64[4]>` → `["4"]`), so the shape can be baked into a memref
/// descriptor with no dynamic dim. A rank-0 tensor (`dims` empty) is NOT treated
/// as static-lowerable here (there is no reduction surface for it yet), and a
/// symbolic dim (`["N"]`) makes the whole annotation dynamic.
fn tensor_dims_all_static(dims: &[String]) -> bool {
    !dims.is_empty() && dims.iter().all(|d| d.parse::<usize>().is_ok())
}

/// Reason a function parameter/return `TypeAnn` cannot lower in the runnable
/// i64 ABI, or `None` when it lowers correctly (`i64`, `f32`, `f64`, `bool`,
/// a struct handle (`Named`), slice/ref/array/tuple — handled elsewhere or
/// already loud).
fn sig_non_i64(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        // `i32`/`u32` params & returns now lower correctly (real i32 MLIR with
        // signed/unsigned op selection + deterministic two's-complement wrap), so
        // they are no longer gated.
        TypeAnn::Tensor { .. } => Some(
            "a tensor-typed parameter/return erases to the i64 ABI and is treated as a scalar \
             integer",
        ),
        TypeAnn::DiffTensor { .. } => Some("a diff-tensor parameter/return erases to the i64 ABI"),
        // i8/u8/i16/u16 (as `Named`) still widen to i64 in a signature — they have
        // no dedicated ValueKind yet, so the i32 ABI does not cover them.
        TypeAnn::Named(n) if is_narrow_int_name(n) => {
            Some("a sub-i64 integer type silently widens to `i64`")
        }
        _ => None,
    }
}

/// Narrow integer type names that may reach the AST as `TypeAnn::Named`
/// (the lexer maps `i32`/`u32` to dedicated scalars; `i8`/`u8`/`i16`/`u16`
/// can arrive as named scalars). `u64`/`i64` are full-width and lower fine.
fn is_narrow_int_name(n: &str) -> bool {
    matches!(n, "i8" | "u8" | "i16" | "u16")
}

fn mk(src: &str, file: Option<&str>, span: AstSpan, code: &'static str, msg: String) -> Diagnostic {
    Diagnostic::error(PHASE, code, msg)
        .with_span(Span::from_offsets(src, span.start(), span.end(), file))
        .with_help(HELP)
}

// ── Fail-closed gate: unresolved generic calls (CRITICAL #2) ──────────────────
//
// A generic call `id(x)` whose argument's concrete type cannot be inferred in
// the bounded monomorphization slice was a SILENT MISCOMPILE: the lowering left
// a bare `@id` reference (the `id$T` body was never emitted), so `--emit-shared`
// wrote an EXIT=0 `.so` with an UNDEFINED symbol (`nm -D` => `U id`; dlopen =>
// "undefined symbol: id"). This gate flags exactly those calls so the runnable
// path fails LOUD with a file:line span instead of shipping a broken artifact.
//
// ZERO false positives BY CONSTRUCTION: it can only fire on a call whose callee
// is in the declared-generic-template set (a `FnDef` with a non-empty
// `type_params`), which is EMPTY for every non-generic / intrinsic / extern-C /
// BLAS program — so a `__mind_*` runtime intrinsic, an `extern "C"` symbol, a
// monomorphized `id$i64`, or any ordinary user fn can NEVER be flagged. The
// accept/reject decision shares the SAME `is_monomorphizable` predicate the
// lowering uses (`crate::eval::lower`), so the gate flags exactly the calls the
// lowering would leave dangling — they cannot drift. Empty `Vec` for a
// non-generic module => keystone / canaries byte-identical.

const GENERIC_HELP: &str = "a generic call monomorphizes only when its argument's concrete type is \
     inferable in the shipped slice: an int/float literal, or an enclosing-fn parameter of scalar \
     type. Bind the argument to a typed parameter, or run it with the `mind` interpreter.";

struct GenCtx<'a> {
    templates: &'a std::collections::HashSet<&'a str>,
    /// Enclosing-fn `param/let name -> declared/inferred scalar type`. Grows as
    /// the forward body walk records each top-level Let (lockstep with lowering).
    bindings: &'a std::collections::HashMap<String, TypeAnn>,
    /// Every non-generic fn's declared scalar return type (`id(g(3))` resolution).
    fn_returns: &'a std::collections::HashMap<String, TypeAnn>,
    src: &'a str,
    file: Option<&'a str>,
}

/// Flag every call to a generic template whose argument cannot be monomorphized
/// — the calls the lowering would otherwise leave as a dangling bare-template
/// reference (an undefined symbol in the runnable artifact). Enforced ONLY on
/// the emit path (wired into `runnable_blockers` in `pipeline.rs`).
pub fn check_generic_resolvable(module: &Module, src: &str, file: Option<&str>) -> Vec<Diagnostic> {
    // Fast path (the overwhelmingly common case): no generic templates at all.
    // A pure iterator scan with ZERO allocation — no `HashSet`, no per-fn binding
    // map, no body walk. The gate is then structurally inert (and byte-identity
    // safe) for every non-generic / intrinsic / extern-C program, so the compile
    // hot path is untouched.
    let has_templates = module
        .items
        .iter()
        .any(|item| matches!(item, Node::FnDef { type_params, .. } if !type_params.is_empty()));
    if !has_templates {
        return Vec::new();
    }
    // Declared-generic-template set: IDENTICAL to the set the lowering registers
    // (src/eval/lower.rs — `FnDef` with non-empty `type_params`).
    let mut templates: std::collections::HashSet<&str> = std::collections::HashSet::new();
    // `fn_returns`: each NON-generic fn's declared scalar return type — built by
    // the SAME forward pass + filter the lowering's FN_RETURNS uses (same AST,
    // same `mangle_suffix` scalar filter), so a nested-call arg resolves
    // identically in the gate and the lowering (lockstep).
    let mut fn_returns: std::collections::HashMap<String, TypeAnn> =
        std::collections::HashMap::new();
    for item in &module.items {
        if let Node::FnDef {
            name,
            type_params,
            ret_type,
            ..
        } = item
        {
            if !type_params.is_empty() {
                templates.insert(name.as_str());
            } else if let Some(rt) = ret_type
                && crate::eval::lower::mangle_suffix(rt).is_some()
            {
                fn_returns.insert(name.clone(), rt.clone());
            }
        }
    }
    let mut out = Vec::new();
    for item in &module.items {
        if let Node::FnDef {
            params,
            body,
            type_params,
            ..
        } = item
        {
            // A generic TEMPLATE is never lowered directly — only its concrete
            // instances are (synthesized post-typecheck, lowered via the FnDef
            // path where the type-params are concrete and the same Part-1
            // resolution applies). So there is no directly-emitted body to check
            // here; skip it. Checking a template body's `id(y)` over a type-param
            // `y` would be a FALSE POSITIVE (it resolves at instantiation).
            // deferred: an instance whose own inner generic call is unresolvable
            // (e.g. a non-scalar local inside the instantiated body) is not caught
            // by this module-level gate — rare (scalar instances' inner calls
            // resolve); upgrade path: a lowering-time check in the mono drain.
            if !type_params.is_empty() {
                continue;
            }
            // `bindings`: the enclosing fn's params first (mirrors the lowering's
            // seed), then grown by each top-level Let via the SHARED `bind_let`
            // in the SAME forward order — so the gate's resolvability decision at
            // every call site matches the lowering's monomorphization decision.
            let mut bindings: std::collections::HashMap<String, TypeAnn> =
                std::collections::HashMap::new();
            for p in params {
                bindings.insert(p.name.clone(), p.ty.clone());
            }
            for stmt in body {
                {
                    let ctx = GenCtx {
                        templates: &templates,
                        bindings: &bindings,
                        fn_returns: &fn_returns,
                        src,
                        file,
                    };
                    walk_generic_calls(stmt, &ctx, &mut out);
                }
                // Record a top-level Let's type AFTER walking its calls and
                // BEFORE the next statement — identical helper + order to the
                // lowering (lockstep). A no-op for non-Let statements.
                crate::eval::lower::bind_let(&mut bindings, stmt, &fn_returns);
            }
        }
    }
    out
}

/// Recursive walk mirroring `lower::node_mentions_type_name`'s variant coverage,
/// flagging each `Call` to a generic template that is not monomorphizable.
fn walk_generic_calls(node: &Node, ctx: &GenCtx, out: &mut Vec<Diagnostic>) {
    use Node as N;
    match node {
        N::Call { callee, args, span } => {
            if ctx.templates.contains(callee.as_str())
                && !crate::eval::lower::is_monomorphizable(args, ctx.bindings, ctx.fn_returns)
            {
                out.push(
                    Diagnostic::error(
                        PHASE,
                        "lower::unresolved_generic",
                        format!(
                            "call to generic `{callee}` cannot be monomorphized: its argument's \
                             concrete type is not inferable here, so the lowering would emit a \
                             dangling `@{callee}` reference (an undefined symbol in the artifact)"
                        ),
                    )
                    .with_span(Span::from_offsets(
                        ctx.src,
                        span.start(),
                        span.end(),
                        ctx.file,
                    ))
                    .with_help(GENERIC_HELP),
                );
            }
            for a in args {
                walk_generic_calls(a, ctx, out);
            }
        }
        N::Let { value, .. } => walk_generic_calls(value, ctx, out),
        N::Const { value, .. } => walk_generic_calls(value, ctx, out),
        N::As { expr, .. } => walk_generic_calls(expr, ctx, out),
        N::Block { stmts, .. } => stmts.iter().for_each(|s| walk_generic_calls(s, ctx, out)),
        N::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            walk_generic_calls(cond, ctx, out);
            then_branch
                .iter()
                .for_each(|s| walk_generic_calls(s, ctx, out));
            if let Some(e) = else_branch {
                e.iter().for_each(|s| walk_generic_calls(s, ctx, out));
            }
        }
        #[cfg(feature = "std-surface")]
        N::While { cond, body, .. } => {
            walk_generic_calls(cond, ctx, out);
            body.iter().for_each(|s| walk_generic_calls(s, ctx, out));
        }
        N::Match {
            scrutinee, arms, ..
        } => {
            walk_generic_calls(scrutinee, ctx, out);
            arms.iter().for_each(|a| {
                // A guard is a full expression evaluated on the runnable path;
                // walk it too so an unsupported construct in `pat if <guard>`
                // is still flagged (pattern-guards W1.5a).
                if let Some(g) = &a.guard {
                    walk_generic_calls(g, ctx, out);
                }
                walk_generic_calls(&a.body, ctx, out);
            });
        }
        N::Return { value: Some(v), .. } => walk_generic_calls(v, ctx, out),
        N::Assign { value, .. } => walk_generic_calls(value, ctx, out),
        N::FieldAssign {
            receiver, value, ..
        } => {
            walk_generic_calls(receiver, ctx, out);
            walk_generic_calls(value, ctx, out);
        }
        N::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            walk_generic_calls(receiver, ctx, out);
            walk_generic_calls(index, ctx, out);
            walk_generic_calls(value, ctx, out);
        }
        N::Paren(inner, _) | N::Neg { operand: inner, .. } | N::Ref { inner, .. } => {
            walk_generic_calls(inner, ctx, out)
        }
        N::Binary { left, right, .. }
        | N::Logical { left, right, .. }
        | N::Bitwise { left, right, .. } => {
            walk_generic_calls(left, ctx, out);
            walk_generic_calls(right, ctx, out);
        }
        N::MethodCall { receiver, args, .. } => {
            walk_generic_calls(receiver, ctx, out);
            args.iter().for_each(|a| walk_generic_calls(a, ctx, out));
        }
        N::FieldAccess { receiver, .. } => walk_generic_calls(receiver, ctx, out),
        N::IndexAccess {
            receiver, index, ..
        } => {
            walk_generic_calls(receiver, ctx, out);
            walk_generic_calls(index, ctx, out);
        }
        _ => {}
    }
}

// ── Fail-closed gate: enum handle returned where a bare scalar is declared ─────
//
// A function declared `-> i64` (or any bare scalar) that returns a payload-
// carrying enum constructor on some path was a SILENT MISCOMPILE: the enum value
// is a heap-record HANDLE (an i64 address), so `divide(5,0)` returning
// `Res::Err(0)` leaked a raw pointer (e.g. `369049072`) as the result. The
// type-checker has no declared-return-vs-body unification, so this compiled.
//
// This gate flags exactly that shape so the runnable path fails LOUD with a span
// instead of leaking a pointer. ZERO false positives by construction: it has a
// no-enum fast path (empty for every program without an `enum` decl — the
// keystone has none, so it stays byte-identical), only fires when the declared
// return is a BARE SCALAR (a `-> Enum` return is the correct shape and is never
// flagged), and only on a PAYLOAD constructor in RETURN POSITION (a fieldless
// variant lowers to a bare ordinal tag, not a handle, and an enum ctor in
// argument position is not in return position — neither is flagged).

const ENUM_RETURN_HELP: &str = "a function returning a bare scalar must not return an enum value on \
     any path — an enum value is a heap-record handle (an i64 address), so returning it where a \
     scalar is declared leaks a pointer. Declare the return type as the enum, or `match` the enum \
     to a scalar before returning.";

/// `true` for a declared return type that is a bare scalar (mixing an enum
/// handle into it is the unsound shape). A `Named` (struct/enum) return, tensor,
/// slice, etc. are NOT bare scalars and are never flagged.
fn is_bare_scalar_ann(ty: &TypeAnn) -> bool {
    matches!(
        ty,
        TypeAnn::ScalarI64
            | TypeAnn::ScalarI32
            | TypeAnn::ScalarU32
            | TypeAnn::ScalarF64
            | TypeAnn::ScalarF32
            | TypeAnn::ScalarBool
    )
}

/// Flag a function with a bare-scalar declared return that returns a payload-
/// carrying enum constructor on any return path (`Node::Return` value, or the
/// tail expression of the body / an if-branch / a match arm / a block).
pub fn check_enum_handle_scalar_return(
    module: &Module,
    src: &str,
    file: Option<&str>,
) -> Vec<Diagnostic> {
    // Payload-carrying enum constructors `Enum::Variant`, keyed exactly as the
    // ctor `Node::Call` callee. Empty for any program with no `enum` decl (a
    // pure no-allocation iterator scan) -> the gate is inert + byte-identity-safe.
    let mut payload_ctors: std::collections::HashSet<String> = std::collections::HashSet::new();
    for item in &module.items {
        if let Node::EnumDef { name, variants, .. } = item {
            for v in variants {
                if !v.payload.is_empty() {
                    payload_ctors.insert(format!("{name}::{}", v.name));
                }
            }
        }
    }
    let mut out = Vec::new();
    if payload_ctors.is_empty() {
        return out;
    }
    for item in &module.items {
        if let Node::FnDef { ret_type, body, .. } = item {
            if !ret_type.as_ref().is_some_and(is_bare_scalar_ann) {
                continue;
            }
            for (i, stmt) in body.iter().enumerate() {
                // Every explicit `return E`, including those nested inside an
                // if/while/match/block (`if b == 0 { return Res::Err(0) }`).
                find_returns(stmt, &payload_ctors, src, file, &mut out);
                // The body's tail expression is an implicit return (skip if it is
                // itself a `return`, already covered by find_returns).
                if i + 1 == body.len() && !matches!(stmt, Node::Return { .. }) {
                    flag_enum_return(stmt, &payload_ctors, src, file, &mut out);
                }
            }
        }
    }
    out
}

/// Recurse through control-flow STATEMENT lists to find every explicit
/// `return E` (incl. nested in an if/while/match/block) and flag `E` if it is —
/// or yields, in tail position — a payload-ctor handle. Does NOT descend into
/// expression operands / call args / let values (those are not return
/// positions), so an enum ctor in argument position is never flagged.
fn find_returns(
    node: &Node,
    ctors: &std::collections::HashSet<String>,
    src: &str,
    file: Option<&str>,
    out: &mut Vec<Diagnostic>,
) {
    use Node as N;
    match node {
        N::Return { value: Some(v), .. } => flag_enum_return(v, ctors, src, file, out),
        N::If {
            then_branch,
            else_branch,
            ..
        } => {
            for s in then_branch {
                find_returns(s, ctors, src, file, out);
            }
            if let Some(e) = else_branch {
                for s in e {
                    find_returns(s, ctors, src, file, out);
                }
            }
        }
        #[cfg(feature = "std-surface")]
        N::While { body, .. } => {
            for s in body {
                find_returns(s, ctors, src, file, out);
            }
        }
        N::Match { arms, .. } => {
            for a in arms {
                find_returns(&a.body, ctors, src, file, out);
            }
        }
        N::Block { stmts, .. } => {
            for s in stmts {
                find_returns(s, ctors, src, file, out);
            }
        }
        _ => {}
    }
}

/// Walk only the RETURN POSITIONS reachable from `node` (NOT a full-body walk —
/// an enum ctor in argument or let-value position is legitimate), flagging a
/// payload-ctor `Call`.
fn flag_enum_return(
    node: &Node,
    ctors: &std::collections::HashSet<String>,
    src: &str,
    file: Option<&str>,
    out: &mut Vec<Diagnostic>,
) {
    use Node as N;
    match node {
        N::Call { callee, span, .. } if ctors.contains(callee) => {
            out.push(
                Diagnostic::error(
                    PHASE,
                    "lower::enum_handle_in_scalar_return",
                    format!(
                        "function with a bare-scalar return returns the enum constructor `{callee}` \
                         (a heap-record handle) on this path — returning it as a scalar leaks a \
                         pointer; declare the return type as the enum or match it to a scalar first"
                    ),
                )
                .with_span(Span::from_offsets(src, span.start(), span.end(), file))
                .with_help(ENUM_RETURN_HELP),
            );
        }
        N::Return { value: Some(v), .. } => flag_enum_return(v, ctors, src, file, out),
        N::Paren(inner, _) => flag_enum_return(inner, ctors, src, file, out),
        N::If {
            then_branch,
            else_branch,
            ..
        } => {
            if let Some(t) = then_branch.last() {
                flag_enum_return(t, ctors, src, file, out);
            }
            if let Some(e) = else_branch.as_ref().and_then(|b| b.last()) {
                flag_enum_return(e, ctors, src, file, out);
            }
        }
        N::Match { arms, .. } => {
            for a in arms {
                flag_enum_return(&a.body, ctors, src, file, out);
            }
        }
        N::Block { stmts, .. } => {
            if let Some(t) = stmts.last() {
                flag_enum_return(t, ctors, src, file, out);
            }
        }
        _ => {}
    }
}

// ── Fail-closed gate: enum match sub-pattern the runnable path miscompiles ─────
//
// v1 enum match lowers a payload sub-pattern only when each field is an `Ident`
// (bound from its record slot) or a `Wildcard` (skipped). A NESTED or LITERAL
// sub-pattern (`Some(Some(x))`, `Some(0)`) bails `desugar_match_to_if` to `None`,
// and the `Node::Match` lowering's fallback then evaluates every arm sequentially
// and returns the LAST one's value — a SILENT MISCOMPILE (no error, no crash,
// wrong value). Multi-field tuple variants DO lower now (each field is stored
// into its own record slot), so they are no longer gated.
//
// This gate flags the unsupported sub-patterns on the runnable path so they fail
// LOUD with a span instead of silently miscompiling. Inert (empty,
// byte-identity-safe) for any program with no payload-carrying enum.

const MATCH_RUNNABLE_HELP: &str = "v1 enum match lowers a payload sub-pattern only when each field is a binding (`Ident`), a \
     wildcard (`_`), or a single-level tuple of bindings/wildcards — e.g. `Some(v)`, \
     `Pair::P(a, b)`, `Pair::P(a, _)`, `Ok((a, b))`. A deeper-nested or literal sub-pattern \
     (`Some(Some(x))`, `Some(0)`, `Ok(((a, b), c))`) is not yet lowerable: the match would fall \
     back to a sequential evaluation that returns the wrong arm. Match the field with a binding \
     and test it in the arm body, or run it with the `mind` interpreter.";

/// `true` when a match arm's payload sub-patterns are all ones the v1 desugar
/// lowers correctly: each field is an `Ident` (bind it from its slot), a
/// `Wildcard` (skip it), or a single-level `Tuple` of Idents/Wildcards
/// (`Ok((a, b))` — the payload slot holds the tuple aggregate's handle; the
/// desugar binds the handle then loads each element). MUST mirror `sub_ok` in
/// `desugar_match_to_if`'s `build_binds` (`src/eval/lower.rs`) exactly: a
/// DEEPER-nested or LITERAL sub-pattern (`Some(Some(x))`, `Some(0)`) bails
/// the desugar, so it must stay rejected here (loud, fail-closed).
fn payload_subpattern_supported(args: &[crate::ast::Pattern]) -> bool {
    use crate::ast::Pattern as P;
    args.iter().all(|p| match p {
        P::Ident(_) | P::Wildcard => true,
        P::Tuple(inner) => inner.iter().all(|q| matches!(q, P::Ident(_) | P::Wildcard)),
        _ => false,
    })
}

/// Flag match shapes the runnable lowering would SILENTLY MISCOMPILE. Two
/// classes: (a) an `EnumVariant`/`EnumStruct` arm with a nested or literal
/// payload sub-pattern (`Some(Some(x))`, `E::V { x: 0 }`); (b) a whole-arm
/// FLOAT / STRING literal pattern (`1.0 =>`, `"x" =>`) or a top-level TUPLE
/// pattern (`(a, b) =>`). All of these bail `desugar_match_to_if` to a
/// sequential fallback (`src/eval/lower.rs`) that IGNORES the scrutinee and
/// returns the LAST arm's value — a silent wrong result (reproduced:
/// `match x { 1.0 => 10, 2.0 => 20 }` returns 20 for every `x`;
/// `match (1,2) { (a,b) => a+b, _ => 0 }` returns 0). Inert (empty) for a module
/// whose matches use only int-literal / wildcard / ident / enum-variant arms
/// with binding-or-wildcard payloads, so the mic@3 self-host fixed point stays
/// byte-identical (keystone 7/7).
pub fn check_match_runnable(module: &Module, src: &str, file: Option<&str>) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for item in &module.items {
        if let Node::FnDef { body, .. } = item {
            for stmt in body {
                walk_match_runnable(stmt, src, file, &mut out);
            }
        }
    }
    out
}

const AMBIG_CTOR_HELP: &str = "a bare payload constructor `V(...)` resolves against every `Enum::V` in the \
     program in deterministic map order, so when the SAME variant name carries a payload in TWO enums \
     the bare form silently picks the first — a wrong-enum tag that then matches the wrong arm. Qualify \
     it with the target enum: `Zeta::V(...)` instead of `V(...)`.";

/// Flag a BARE payload constructor call `V(x)` whose short name `V` carries a
/// payload in more than one declared enum. The lowering (`src/eval/lower.rs`
/// bare-ctor resolution) resolves such a name to the FIRST `Enum::V` in
/// deterministic BTreeMap order and stamps THAT enum's discriminant tag — so
/// `let z: Zeta = V(7)` where `Alpha::V` sorts first gets Alpha's tag and then
/// matches the WRONG arm in a `match z` (reproduced: returned 1007 for a correct
/// answer of 7). This is a SILENT miscompile the type-checker's `mindc check`
/// passes. Reject it fail-loud on the runnable path — the fix is to qualify the
/// ctor (`Zeta::V(7)`). Inert (empty) unless a payload variant name collides
/// across enums, so keystone byte-identity is unaffected.
pub fn check_ambiguous_bare_ctor(
    module: &Module,
    src: &str,
    file: Option<&str>,
) -> Vec<Diagnostic> {
    // short variant name -> count of DISTINCT enums that declare it with a payload.
    let mut short_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for item in &module.items {
        if let Node::EnumDef { variants, .. } = item {
            for v in variants {
                if !v.payload.is_empty() {
                    *short_counts.entry(v.name.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    // Ambiguous = a payload variant name declared in >= 2 enums.
    let ambiguous: std::collections::HashSet<String> = short_counts
        .into_iter()
        .filter(|(_, n)| *n >= 2)
        .map(|(name, _)| name)
        .collect();
    let mut out = Vec::new();
    if ambiguous.is_empty() {
        return out;
    }
    for item in &module.items {
        if let Node::FnDef { body, .. } = item {
            for stmt in body {
                walk_ambiguous_ctor(stmt, &ambiguous, src, file, &mut out);
            }
        }
    }
    out
}

/// Full-expression walk flagging a bare payload ctor call whose short name is in
/// `ambiguous`. Mirrors `walk_match_runnable`'s node coverage. A missed nesting
/// only misses a rejection (fail-safe) — it can never produce a false positive.
fn walk_ambiguous_ctor(
    node: &Node,
    ambiguous: &std::collections::HashSet<String>,
    src: &str,
    file: Option<&str>,
    out: &mut Vec<Diagnostic>,
) {
    use Node as N;
    let recur =
        |n: &Node, out: &mut Vec<Diagnostic>| walk_ambiguous_ctor(n, ambiguous, src, file, out);
    match node {
        N::Call {
            callee, args, span, ..
        } => {
            // A BARE (unqualified) callee with a payload whose short name collides.
            if !callee.contains("::") && !args.is_empty() && ambiguous.contains(callee.as_str()) {
                out.push(
                    Diagnostic::error(
                        PHASE,
                        "lower::ambiguous_bare_ctor",
                        format!(
                            "bare constructor `{callee}` is ambiguous: the payload variant name \
                             `{callee}` is declared in more than one enum, so the lowering would \
                             resolve it to the wrong enum's discriminant and match the wrong arm"
                        ),
                    )
                    .with_span(Span::from_offsets(src, span.start(), span.end(), file))
                    .with_help(AMBIG_CTOR_HELP),
                );
            }
            for a in args {
                recur(a, out);
            }
        }
        N::Match {
            scrutinee, arms, ..
        } => {
            recur(scrutinee, out);
            for a in arms {
                if let Some(g) = &a.guard {
                    recur(g, out);
                }
                recur(&a.body, out);
            }
        }
        N::Let { value, .. }
        | N::Const { value, .. }
        | N::Assign { value, .. }
        | N::Return {
            value: Some(value), ..
        } => recur(value, out),
        N::As { expr, .. } => recur(expr, out),
        N::Paren(inner, _) | N::Neg { operand: inner, .. } | N::Ref { inner, .. } => {
            recur(inner, out)
        }
        N::Block { stmts, .. } => stmts.iter().for_each(|s| recur(s, out)),
        N::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            recur(cond, out);
            then_branch.iter().for_each(|s| recur(s, out));
            if let Some(e) = else_branch {
                e.iter().for_each(|s| recur(s, out));
            }
        }
        #[cfg(feature = "std-surface")]
        N::While { cond, body, .. } => {
            recur(cond, out);
            body.iter().for_each(|s| recur(s, out));
        }
        N::Binary { left, right, .. } | N::Bitwise { left, right, .. } => {
            recur(left, out);
            recur(right, out);
        }
        N::MethodCall { receiver, args, .. } => {
            recur(receiver, out);
            args.iter().for_each(|a| recur(a, out));
        }
        N::FieldAccess { receiver, .. } => recur(receiver, out),
        N::IndexAccess {
            receiver, index, ..
        } => {
            recur(receiver, out);
            recur(index, out);
        }
        _ => {}
    }
}

/// Full-expression walk flagging unsupported enum construct/match shapes. Mirrors
/// `walk_generic_calls`'s variant coverage so every nested position is visited.
fn walk_match_runnable(node: &Node, src: &str, file: Option<&str>, out: &mut Vec<Diagnostic>) {
    use Node as N;
    match node {
        N::Call { args, .. } => {
            // Multi-field constructors now lower (every field is stored into its
            // own record slot), so a ctor call no longer gates. Recurse into the
            // arguments in case one nests an unsupported match.
            for a in args {
                walk_match_runnable(a, src, file, out);
            }
        }
        N::Match {
            scrutinee, arms, ..
        } => {
            walk_match_runnable(scrutinee, src, file, out);
            for a in arms {
                // Flag an arm whose payload sub-patterns v1 cannot lower (a nested
                // or literal field pattern) — for BOTH tuple-payload variants
                // (`EnumVariant`, `E::V(0)`) AND struct-payload variants
                // (`EnumStruct`, `E::V { x: 0 }`). The `EnumStruct` case was
                // previously UNCHECKED here: its desugar (`lower.rs` normalises the
                // named fields to positional args, `.unwrap_or(Wildcard)`) drops a
                // non-binding field sub-pattern to a wildcard, so `E::V { x: 0 }`
                // matched a value with `x = 5` and returned a garbage value — a
                // SILENT MISCOMPILE — instead of failing loud like the tuple form.
                let unsupported_path = match &a.pattern {
                    crate::ast::Pattern::EnumVariant { path, args }
                        if !payload_subpattern_supported(args) =>
                    {
                        Some(path.as_str())
                    }
                    crate::ast::Pattern::EnumStruct { path, fields }
                        if !fields.iter().all(|(_, p)| {
                            matches!(
                                p,
                                crate::ast::Pattern::Ident(_) | crate::ast::Pattern::Wildcard
                            )
                        }) =>
                    {
                        Some(path.as_str())
                    }
                    _ => None,
                };
                if let Some(path) = unsupported_path {
                    out.push(
                        Diagnostic::error(
                            PHASE,
                            "lower::enum_match_unsupported_payload",
                            format!(
                                "match arm `{path}` binds a nested or literal payload sub-pattern, \
                                 which v1 does not lower — only field bindings (`Ident`), \
                                 wildcards (`_`), and single-level tuples of bindings \
                                 (`Ok((a, b))`) are supported; the match would otherwise silently \
                                 fall back to a sequential evaluation and return the wrong arm"
                            ),
                        )
                        .with_span(Span::from_offsets(src, a.span.start(), a.span.end(), file))
                        .with_help(MATCH_RUNNABLE_HELP),
                    );
                }
                // A WHOLE-ARM pattern the desugar cannot lower — a FLOAT/STRING
                // literal or a top-level TUPLE — bails `desugar_match_to_if` to the
                // sequential fallback that ignores the scrutinee and returns the
                // last arm (a silent wrong value). `desugar_match_to_if` lowers
                // int-literal / wildcard / ident / enum-variant patterns only.
                let unsupported_arm = match &a.pattern {
                    crate::ast::Pattern::Literal(crate::ast::Literal::Float(_)) => {
                        Some("a float-literal")
                    }
                    crate::ast::Pattern::Literal(crate::ast::Literal::Str(_)) => {
                        Some("a string-literal")
                    }
                    crate::ast::Pattern::Tuple(_) => Some("a tuple-destructure"),
                    _ => None,
                };
                if let Some(kind) = unsupported_arm {
                    out.push(
                        Diagnostic::error(
                            PHASE,
                            "lower::match_pattern_unsupported",
                            format!(
                                "match arm uses {kind} pattern, which the runnable v1 lowering does \
                                 not support — the match would silently fall back to a sequential \
                                 evaluation that ignores the scrutinee and returns the last arm (a \
                                 wrong value). Match on an integer/enum discriminant, or run it with \
                                 the `mind` interpreter"
                            ),
                        )
                        .with_span(Span::from_offsets(src, a.span.start(), a.span.end(), file))
                        .with_help(MATCH_RUNNABLE_HELP),
                    );
                }
                // Walk the guard expression too — its lowering is subject to the
                // same runnable constraints as the arm body (pattern-guards
                // W1.5a). A guarded `_`/ident arm is a refutable TEST arm, not a
                // catch-all (drift #131), and the desugar routes it correctly, so
                // no extra whole-arm flag is needed here.
                if let Some(g) = &a.guard {
                    walk_match_runnable(g, src, file, out);
                }
                walk_match_runnable(&a.body, src, file, out);
            }
        }
        N::Let { value, .. } => walk_match_runnable(value, src, file, out),
        N::Const { value, .. } => walk_match_runnable(value, src, file, out),
        N::As { expr, .. } => walk_match_runnable(expr, src, file, out),
        N::Block { stmts, .. } => stmts
            .iter()
            .for_each(|s| walk_match_runnable(s, src, file, out)),
        N::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            walk_match_runnable(cond, src, file, out);
            then_branch
                .iter()
                .for_each(|s| walk_match_runnable(s, src, file, out));
            if let Some(e) = else_branch {
                e.iter()
                    .for_each(|s| walk_match_runnable(s, src, file, out));
            }
        }
        #[cfg(feature = "std-surface")]
        N::While { cond, body, .. } => {
            walk_match_runnable(cond, src, file, out);
            body.iter()
                .for_each(|s| walk_match_runnable(s, src, file, out));
        }
        N::Return { value: Some(v), .. } => walk_match_runnable(v, src, file, out),
        N::Assign { value, .. } => walk_match_runnable(value, src, file, out),
        N::FieldAssign {
            receiver, value, ..
        } => {
            walk_match_runnable(receiver, src, file, out);
            walk_match_runnable(value, src, file, out);
        }
        N::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            walk_match_runnable(receiver, src, file, out);
            walk_match_runnable(index, src, file, out);
            walk_match_runnable(value, src, file, out);
        }
        N::Paren(inner, _) | N::Neg { operand: inner, .. } | N::Ref { inner, .. } => {
            walk_match_runnable(inner, src, file, out)
        }
        N::Binary { left, right, .. }
        | N::Logical { left, right, .. }
        | N::Bitwise { left, right, .. } => {
            walk_match_runnable(left, src, file, out);
            walk_match_runnable(right, src, file, out);
        }
        N::MethodCall { receiver, args, .. } => {
            walk_match_runnable(receiver, src, file, out);
            args.iter()
                .for_each(|a| walk_match_runnable(a, src, file, out));
        }
        N::FieldAccess { receiver, .. } => walk_match_runnable(receiver, src, file, out),
        N::IndexAccess {
            receiver, index, ..
        } => {
            walk_match_runnable(receiver, src, file, out);
            walk_match_runnable(index, src, file, out);
        }
        _ => {}
    }
}
