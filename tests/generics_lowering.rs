// Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
// Generic-function codegen monomorphization: a generic fn called at a concrete
// type must lower through the IR backend, and that lowering must be deterministic
// (byte-identical mic@3 across compiles) — the cross-substrate wedge applies to
// monomorphized instances just like everything else.

use libmind::ir::compact::emit_mic3;
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

#[test]
fn generic_call_with_variable_arg_monomorphizes() {
    // CRITICAL #2 fix (Part 1): a generic called with a VARIABLE argument bound
    // to a scalar parameter must monomorphize. Before, `id(n)` kept the bare
    // `@id` name and the `.so` shipped an undefined symbol (EXIT=0 silent
    // miscompile); now the enclosing fn's param type resolves the instance.
    let src = "fn id<T>(x: T) -> T { x }\nfn use_it(n: i64) -> i64 { id(n) }";
    let a = compile_source(src, &CompileOptions::default())
        .expect("a generic call over a scalar parameter should compile");
    assert!(
        emit_mic3(&a.ir).windows(6).any(|w| w == b"id$i64"),
        "id(n) for n: i64 must monomorphize to a real `id$i64` body"
    );
    // The now-resolvable call must NOT be recorded as a runnable blocker.
    let p = compile_source_with_name(src, None, &CompileOptions::default())
        .expect("source parses + type-checks");
    assert!(
        !p.runnable_blockers
            .iter()
            .any(|d| d.code == "lower::unresolved_generic"),
        "a resolvable generic call must not be gated: {:?}",
        p.runnable_blockers
    );
}

#[test]
fn generic_call_unresolvable_fails_closed() {
    // CRITICAL #2 fail-closed net (Part 2): a generic call whose argument type
    // is NOT inferable in the bounded slice (here a Let-bound local) must be
    // recorded as a runnable blocker — a loud file:line error on
    // `--emit-shared` — instead of writing a broken `.so` with an undefined
    // symbol. The source parses + type-checks (it is a valid program); only the
    // RUNNABLE artifact is refused.
    let src = "fn id<T>(x: T) -> T { x }\nfn use_it() -> i64 {\n  let z: i64 = 5\n  id(z)\n}";
    let p = compile_source_with_name(src, None, &CompileOptions::default())
        .expect("source parses + type-checks");
    assert!(
        p.runnable_blockers
            .iter()
            .any(|d| d.code == "lower::unresolved_generic"),
        "an unresolvable generic call must be gated as a runnable blocker, got: {:?}",
        p.runnable_blockers
    );
}
