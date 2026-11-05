// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A call to a registered `__mind_*` intrinsic carries exactly the declared
//! arity. `__mind_write(fd, buf, count, offset)` with three arguments used to
//! pass verification and build, so the C runtime read an undefined offset.

#![cfg(feature = "std-surface")]

use libmind::ir::{IRModule, Instr, ValueId, verify_module};

fn module_calling_write(arg_count: usize) -> IRModule {
    let mut body: Vec<Instr> = (0..arg_count)
        .map(|i| Instr::ConstI64(ValueId(i), 1))
        .collect();
    body.push(Instr::Call {
        dst: ValueId(arg_count),
        name: "__mind_write".to_string(),
        args: (0..arg_count).map(ValueId).collect(),
        resolved_callee: None,
    });
    body.push(Instr::Return {
        value: Some(ValueId(arg_count)),
    });
    let mut m = IRModule::new();
    m.instrs.push(Instr::FnDef {
        name: "main".to_string(),
        params: vec![],
        ret_id: Some(ValueId(arg_count)),
        body,
        reap_threshold: None,
        semantic_types: None,
        value_types: std::collections::BTreeMap::new(),
    });
    m.instrs.push(Instr::ConstI64(ValueId(0), 0));
    m.instrs.push(Instr::Output(ValueId(0)));
    m.next_id = arg_count + 1;
    m
}

#[test]
fn a_short_intrinsic_call_fails_verification() {
    let err = verify_module(&module_calling_write(3))
        .expect_err("a 3-argument __mind_write must not verify");
    assert!(
        err.to_string()
            .contains("intrinsic `__mind_write` expects 4 i64 argument(s)"),
        "the error names the intrinsic and its arity: {err}"
    );
}

#[test]
fn an_extra_argument_fails_verification() {
    let err = verify_module(&module_calling_write(5))
        .expect_err("a 5-argument __mind_write must not verify");
    assert!(
        err.to_string()
            .contains("intrinsic `__mind_write` expects 4 i64 argument(s)")
            && err.to_string().contains("got 5"),
        "the error names the intrinsic, its arity and the count: {err}"
    );
}

#[test]
fn the_declared_arity_verifies() {
    verify_module(&module_calling_write(4)).expect("a 4-argument __mind_write verifies");
}

/// `n` constants and an `n`-argument `__mind_write`, ids from `base`.
fn write_call(base: usize, n: usize) -> Vec<Instr> {
    let mut v: Vec<Instr> = (base..base + n)
        .map(|i| Instr::ConstI64(ValueId(i), 1))
        .collect();
    v.push(Instr::Call {
        dst: ValueId(base + n),
        name: "__mind_write".to_string(),
        args: (base..base + n).map(ValueId).collect(),
        resolved_callee: None,
    });
    v
}

fn short_write(base: usize) -> Vec<Instr> {
    write_call(base, 3)
}

/// `instrs` followed by the definition of `id`, so a region operand the
/// verifier checks (an if condition or branch result, a while condition, a
/// region result) is produced where it must be.
fn then_define(mut instrs: Vec<Instr>, id: usize) -> Vec<Instr> {
    instrs.push(Instr::ConstI64(ValueId(id), 1));
    instrs
}

fn main_with(body: Vec<Instr>) -> IRModule {
    let mut m = IRModule::new();
    m.instrs.push(Instr::FnDef {
        name: "main".to_string(),
        params: vec![],
        ret_id: None,
        body,
        reap_threshold: None,
        semantic_types: None,
        value_types: std::collections::BTreeMap::new(),
    });
    m.instrs.push(Instr::ConstI64(ValueId(0), 0));
    m.instrs.push(Instr::Output(ValueId(0)));
    m.next_id = 100;
    m
}

fn if_with(cond: Vec<Instr>, then: Vec<Instr>, otherwise: Vec<Instr>) -> Instr {
    Instr::If {
        cond_id: ValueId(50),
        cond_instrs: then_define(cond, 50),
        then_instrs: then_define(then, 51),
        then_result: ValueId(51),
        else_instrs: then_define(otherwise, 52),
        else_result: ValueId(52),
        dst: ValueId(53),
        branch_bindings: vec![],
        merges: vec![],
    }
}

fn while_with(cond: Vec<Instr>, body: Vec<Instr>) -> Instr {
    Instr::While {
        cond_id: ValueId(60),
        cond_instrs: then_define(cond, 60),
        body,
        live_vars: vec![],
        init_ids: vec![],
        exit_ids: vec![],
    }
}

fn region_with(body: Vec<Instr>) -> Instr {
    Instr::Region {
        body: then_define(body, 70),
        result: ValueId(70),
        enter_id: ValueId(71),
        exit_id: ValueId(72),
        alloc_ids: vec![],
    }
}

fn assert_refused(where_: &str, m: IRModule) {
    let err = verify_module(&m).expect_err(where_);
    assert!(
        err.to_string()
            .contains("intrinsic `__mind_write` expects 4 i64 argument(s)"),
        "a short call {where_} is refused for its arity: {err}"
    );
}

#[test]
fn a_short_call_inside_an_if_is_refused() {
    assert_refused(
        "in an if condition",
        main_with(vec![if_with(short_write(1), vec![], vec![])]),
    );
    assert_refused(
        "in an if then branch",
        main_with(vec![if_with(vec![], short_write(1), vec![])]),
    );
    assert_refused(
        "in an if else branch",
        main_with(vec![if_with(vec![], vec![], short_write(1))]),
    );
}

#[test]
fn a_short_call_inside_a_while_is_refused() {
    assert_refused(
        "in a while condition",
        main_with(vec![while_with(short_write(1), vec![])]),
    );
    assert_refused(
        "in a while body",
        main_with(vec![while_with(vec![], short_write(1))]),
    );
}

#[test]
fn a_short_call_inside_a_region_and_deeper_is_refused() {
    assert_refused("in a region", main_with(vec![region_with(short_write(1))]));
    assert_refused(
        "in a while inside an if inside a region",
        main_with(vec![region_with(vec![if_with(
            vec![],
            vec![while_with(vec![], short_write(1))],
            vec![],
        )])]),
    );
}

#[test]
fn the_declared_arity_verifies_at_every_depth() {
    // The positive control for the nested refusals above: the same shapes with a
    // four-argument call verify, so each refusal comes from the arity rule.
    let ok = |n: usize| write_call(1, n);
    for (where_, m) in [
        (
            "in an if condition",
            main_with(vec![if_with(ok(4), vec![], vec![])]),
        ),
        (
            "in an if then branch",
            main_with(vec![if_with(vec![], ok(4), vec![])]),
        ),
        (
            "in an if else branch",
            main_with(vec![if_with(vec![], vec![], ok(4))]),
        ),
        (
            "in a while condition",
            main_with(vec![while_with(ok(4), vec![])]),
        ),
        (
            "in a while body",
            main_with(vec![while_with(vec![], ok(4))]),
        ),
        ("in a region", main_with(vec![region_with(ok(4))])),
        (
            "in a while inside an if inside a region",
            main_with(vec![region_with(vec![if_with(
                vec![],
                vec![while_with(vec![], ok(4))],
                vec![],
            )])]),
        ),
    ] {
        verify_module(&m).unwrap_or_else(|e| panic!("a 4-argument call {where_} verifies: {e}"));
    }
}

// ── The type checker reports E2005 in every expression position ─────────
//
// The IR verifier above refuses a wrong-arity call at any depth, but it only
// runs on the lowering paths; `mindc check` runs the type checker alone. The
// checker used to report the rule (E2005) only where its `infer_expr` walk
// reached the call, so `while` bodies and conditions (RFC 0005 Gap 1), `else
// if` branches, and a call nested as an argument of a call whose checker never
// re-infers its arguments all passed `check`. These tests drive the checker
// entry point `mindc check` runs (`check_module_types`) through every one of
// those shapes.

use libmind::type_checker::{TypeEnv, check_module_types};

/// Type-check `src` and return the diagnostics, exactly as `mindc check` does.
fn check_src(src: &str) -> Vec<libmind::diagnostics::Diagnostic> {
    let module = libmind::parser::parse(src).expect("the fixture must parse");
    check_module_types(&module, src, &TypeEnv::default())
}

fn has_intrinsic_arity_e2005(diags: &[libmind::diagnostics::Diagnostic]) -> bool {
    diags.iter().any(|d| {
        d.code == "E2005"
            && d.message
                .contains("intrinsic `__mind_write` expects 4 i64 argument(s)")
            && d.message.contains("got 3")
    })
}

fn assert_e2005(where_: &str, src: &str) {
    let diags = check_src(src);
    assert!(
        has_intrinsic_arity_e2005(&diags),
        "a 3-argument __mind_write {where_} must report E2005; got {diags:?}"
    );
}

fn assert_no_e2005(where_: &str, src: &str) {
    let diags = check_src(src);
    assert!(
        !has_intrinsic_arity_e2005(&diags),
        "a 4-argument __mind_write {where_} must not report the arity rule; got {diags:?}"
    );
}

/// The wrong-arity call of each shape, and its four-argument twin.
fn shape_fixtures(short: &str, ok: &str) -> Vec<(&'static str, String, String)> {
    vec![
        (
            "in a while body",
            format!(
                "fn main() -> i64 {{\n    let mut i: i64 = 0;\n    while i < 1 {{\n        {short};\n        i = i + 1;\n    }}\n    return 0\n}}\n"
            ),
            format!(
                "fn main() -> i64 {{\n    let mut i: i64 = 0;\n    while i < 1 {{\n        {ok};\n        i = i + 1;\n    }}\n    return 0\n}}\n"
            ),
        ),
        (
            "in a while condition",
            format!(
                "fn main() -> i64 {{\n    let mut i: i64 = 0;\n    while {short} > 100 {{\n        i = i + 1;\n    }}\n    return i\n}}\n"
            ),
            format!(
                "fn main() -> i64 {{\n    let mut i: i64 = 0;\n    while {ok} > 100 {{\n        i = i + 1;\n    }}\n    return i\n}}\n"
            ),
        ),
        (
            "in an else if branch",
            format!(
                "fn main() -> i64 {{\n    if 1 > 5 {{\n        return 1\n    }} else if 1 > 0 {{\n        {short};\n    }}\n    return 0\n}}\n"
            ),
            format!(
                "fn main() -> i64 {{\n    if 1 > 5 {{\n        return 1\n    }} else if 1 > 0 {{\n        {ok};\n    }}\n    return 0\n}}\n"
            ),
        ),
        (
            "nested as an argument",
            format!(
                "fn id(x: i64) -> i64 {{\n    return x\n}}\n\nfn main() -> i64 {{\n    let n = id({short});\n    return n\n}}\n"
            ),
            format!(
                "fn id(x: i64) -> i64 {{\n    return x\n}}\n\nfn main() -> i64 {{\n    let n = id({ok});\n    return n\n}}\n"
            ),
        ),
    ]
}

#[test]
fn a_wrong_arity_call_reports_e2005_at_every_body_shape() {
    for (where_, src, _) in shape_fixtures("__mind_write(1, 0, 0)", "__mind_write(1, 0, 0, 0 - 1)")
    {
        assert_e2005(where_, &src);
    }
}

#[test]
fn the_declared_arity_passes_the_type_checker_at_every_body_shape() {
    for (where_, _, ok) in shape_fixtures("__mind_write(1, 0, 0)", "__mind_write(1, 0, 0, 0 - 1)") {
        assert_no_e2005(where_, &ok);
    }
}
