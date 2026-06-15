// Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
// Generic-function codegen monomorphization: a generic fn called at a concrete
// type must lower through the IR backend, and that lowering must be deterministic
// (byte-identical mic@3 across compiles) — the cross-substrate wedge applies to
// monomorphized instances just like everything else.

use libmind::ir::compact::emit_mic3;
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
