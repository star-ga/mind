// Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
// Generic-function codegen monomorphization: a generic fn called at a concrete
// type must lower through the IR backend, and that lowering must be deterministic
// (byte-identical mic@3 across compiles) — the cross-substrate wedge applies to
// monomorphized instances just like everything else.

use libmind::ir::compact::emit_mic3;
// Only the std-surface-gated `resolves`/`fails_closed` helpers use this; gating
// the import to match keeps the no-default-features clippy lane warning-clean.
#[cfg(feature = "std-surface")]
use libmind::pipeline::compile_source_with_name;
use libmind::{CompileOptions, compile_source};

#[test]
fn generic_fn_call_lowers_and_is_deterministic() {
    let src = "fn id<T>(x: T) -> T { x }\nlet y = id(5)\ny";

    let a = compile_source(src, &CompileOptions::default())
        .expect("a generic function call should compile through lowering");
    let b = compile_source(src, &CompileOptions::default())
        .expect("a generic function call should compile through lowering");

    // The monomorphization drain must emit a REAL instance body, not the
    // pre-drain body-less decl that failed to link (`undefined symbol: id$i64`).
    // The mangled instance name is interned in the emitted mic@3 string table.
    assert!(
        emit_mic3(&a.ir).windows(6).any(|w| w == b"id$i64"),
        "the monomorphized generic instance `id$i64` must be emitted with a body"
    );

    // Determinism (the wedge): the same source produces byte-identical mic@3.
    assert_eq!(
        emit_mic3(&a.ir),
        emit_mic3(&b.ir),
        "monomorphized generic lowering must be deterministic"
    );
}

/// A generic call over `src` must MONOMORPHIZE: emit the `id$<suffix>` instance
/// body AND record no `lower::unresolved_generic` blocker (0 fail-closed).
// The 0-fail-closed shapes exercise fn-body `let` / `struct` / `as` which are
// std-surface constructs, so these tests + helpers are gated on that feature
// (the no-default-features lanes compile them out).
#[cfg(feature = "std-surface")]
fn resolves(src: &str, suffix: &str) {
    let needle = format!("id${suffix}");
    let a = compile_source(src, &CompileOptions::default())
        .unwrap_or_else(|e| panic!("should compile: {e:?}\nsrc:\n{src}"));
    assert!(
        emit_mic3(&a.ir)
            .windows(needle.len())
            .any(|w| w == needle.as_bytes()),
        "expected `{needle}` instance body emitted for:\n{src}"
    );
    let p = compile_source_with_name(src, None, &CompileOptions::default())
        .expect("source parses + type-checks");
    assert!(
        !p.runnable_blockers
            .iter()
            .any(|d| d.code == "lower::unresolved_generic"),
        "a resolvable generic call must NOT be gated: {:?}\nsrc:\n{src}",
        p.runnable_blockers
    );
}

/// A genuinely-unresolvable generic call must FAIL-CLOSED: a `lower::unresolved_generic`
/// runnable blocker (loud `--emit-shared` error), never a broken artifact.
#[cfg(feature = "std-surface")]
fn fails_closed(src: &str) {
    let p = compile_source_with_name(src, None, &CompileOptions::default())
        .expect("source parses + type-checks");
    assert!(
        p.runnable_blockers
            .iter()
            .any(|d| d.code == "lower::unresolved_generic"),
        "expected a fail-closed blocker, got: {:?}\nsrc:\n{src}",
        p.runnable_blockers
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn generic_call_resolves_every_well_typed_scalar_arg_shape() {
    // CRITICAL #2 follow-up — 0 fail-closed for well-typed scalar programs. Every
    // arg shape whose concrete scalar type is statically inferable must
    // monomorphize (was: only literals + params; the rest silently dangled then,
    // after the fail-closed net, errored loud — now they all resolve).
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn f(n: i64) -> i64 { id(n) }",
        "i64",
    ); // param
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn f() -> i64 { let z: i64 = 5\n id(z) }",
        "i64",
    ); // annotated let
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn f() -> i64 { let z = 5\n id(z) }",
        "i64",
    ); // inferred let
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn g(n: i64) -> i64 { n }\nfn f() -> i64 { id(g(3)) }",
        "i64",
    ); // nested call
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn f(a: i64, b: i64) -> i64 { id(a + b) }",
        "i64",
    ); // arithmetic
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn f(x: i64) -> i64 { id(x as i64) }",
        "i64",
    ); // cast
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn f(x: f64) -> f64 { id(x) }",
        "f64",
    ); // f64 param
    // A generic that calls another generic, used at a concrete type, must NOT be
    // falsely flagged (the template body is resolved at instantiation).
    resolves(
        "fn id<T>(x: T) -> T { x }\nfn wrap<U>(y: U) -> U { id(y) }\nfn f() -> i64 { wrap(5) }",
        "i64",
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn generic_call_genuinely_unresolvable_fails_closed() {
    // The fail-closed net still fires (correctly) when the argument's concrete
    // scalar type genuinely cannot be inferred — a non-scalar (struct) local —
    // so a broken `.so` is never written. Loud blocker, not a silent miscompile.
    fails_closed(
        "struct P { x: i64 }\nfn id<T>(x: T) -> T { x }\nfn f() -> i64 { let p = P { x: 1 }\n let q = id(p)\n 0 }",
    );
}
