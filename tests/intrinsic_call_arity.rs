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
// The IR verifier above refuses a wrong-arity call at any depth, but it runs
// only on the lowering paths; `mindc check` runs the type checker alone. The
// checker used to report the rule (E2005) only where its `infer_expr` walk
// reached the call, so `while` bodies and conditions (RFC 0005 Gap 1), `else
// if` branches, a non-final statement of an `if` or `for` body, and a call
// nested as an argument of a call whose arguments the checker never re-infers
// all passed `check`. These tests drive the checker entry point `check_module_types` on
// the parsed module (before any desugar, so closure and impl bodies are still
// nested) over every shape the IR tests above refuse and every body the
// checker descends into. Each wrong-arity call is reported exactly once: a
// second report would mean two passes own one rule.

use libmind::diagnostics::Diagnostic;
use libmind::type_checker::{TypeEnv, check_module_types};

const SHORT_CALL: &str = "__mind_write(1, 0, 0)";
const LONG_CALL: &str = "__mind_write(1, 0, 0, 0 - 1, 0)";
const DECLARED_CALL: &str = "__mind_write(1, 0, 0, 0 - 1)";

/// Every position the checker's walk reaches, as a program with `CALL` where
/// the intrinsic call goes.
const BODY_SHAPES: &[(&str, &str)] = &[
    (
        "as a statement in a fn body",
        "fn main() -> i64 {\n    CALL;\n    return 0\n}\n",
    ),
    (
        "in a let initializer",
        "fn main() -> i64 {\n    let n = CALL;\n    return n\n}\n",
    ),
    (
        "in a return expression",
        "fn main() -> i64 {\n    return CALL\n}\n",
    ),
    (
        "in an if condition",
        "fn main() -> i64 {\n    if CALL > 0 {\n        return 1\n    }\n    return 0\n}\n",
    ),
    (
        "in an if then branch",
        "fn main() -> i64 {\n    if 1 > 0 {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "before another statement in an if branch",
        "fn main() -> i64 {\n    let mut t: i64 = 0;\n    if 1 > 0 {\n        CALL;\n        t = 1;\n    }\n    return t\n}\n",
    ),
    (
        "before another statement in a for body",
        "fn main() -> i64 {\n    let mut t: i64 = 0;\n    for i in 0..2 {\n        CALL;\n        t = t + i;\n    }\n    return t\n}\n",
    ),
    (
        "in an else branch",
        "fn main() -> i64 {\n    if 1 > 5 {\n        return 1\n    } else {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "in an else if branch",
        "fn main() -> i64 {\n    if 1 > 5 {\n        return 1\n    } else if 1 > 0 {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "in a while body",
        "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while i < 1 {\n        CALL;\n        i = i + 1;\n    }\n    return 0\n}\n",
    ),
    (
        "in a while condition",
        "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while CALL > 100 {\n        i = i + 1;\n    }\n    return i\n}\n",
    ),
    (
        "nested as an argument",
        "fn id(x: i64) -> i64 {\n    return x\n}\n\nfn main() -> i64 {\n    let n = id(CALL);\n    return n\n}\n",
    ),
    (
        "in a for body",
        "fn main() -> i64 {\n    for i in 0..2 {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "in a match arm",
        "fn pick(n: i64) -> i64 {\n    match n { 0 => CALL, _ => 0 }\n}\n\nfn main() -> i64 {\n    return pick(1)\n}\n",
    ),
    (
        "in a region",
        "fn main() -> i64 {\n    let r = region {\n        let w = CALL\n        0\n    }\n    return r\n}\n",
    ),
    (
        "in a while inside an if inside a region",
        "fn main() -> i64 {\n    let r = region {\n        let mut i: i64 = 0\n        if i < 1 {\n            while i < 1 {\n                CALL;\n                i = i + 1\n            }\n        }\n        i\n    }\n    return r\n}\n",
    ),
    (
        "in a closure body",
        "fn main() -> i64 {\n    let k: i64 = 1\n    let f = |k; x: i64| -> i64 {\n        let w = CALL\n        x + k\n    }\n    return f(5)\n}\n",
    ),
    (
        "in an impl method body",
        "struct Foo { x: i64 }\n\ntrait Speak {\n    fn val(self) -> i64\n}\n\nimpl Speak for Foo {\n    fn val(self) -> i64 {\n        let w = CALL\n        return self.x\n    }\n}\n\nfn main() -> i64 {\n    let foo = Foo { x: 1 }\n    return foo.val()\n}\n",
    ),
    (
        "in a fn inside a module block",
        "module m {\n    pub fn run() -> i64 {\n        CALL;\n        return 0\n    }\n}\n\nfn main() -> i64 {\n    return 0\n}\n",
    ),
];

/// Type-check `src` with the checker entry point and return the diagnostics.
fn check_src(where_: &str, src: &str) -> Vec<Diagnostic> {
    let module = libmind::parser::parse(src)
        .unwrap_or_else(|e| panic!("the fixture {where_} must parse: {e:?}\n{src}"));
    check_module_types(&module, src, &TypeEnv::default())
}

/// The E2005 reports for a `__mind_write` call with `got` arguments.
fn arity_reports(diags: &[Diagnostic], got: usize) -> usize {
    let tail = format!("got {got}");
    diags
        .iter()
        .filter(|d| {
            d.code == "E2005"
                && d.message
                    .contains("intrinsic `__mind_write` expects 4 i64 argument(s)")
                && d.message.ends_with(&tail)
        })
        .count()
}

#[test]
fn a_wrong_arity_call_reports_e2005_once_in_every_body_shape() {
    let mut missed = Vec::new();
    for (call, got) in [(SHORT_CALL, 3), (LONG_CALL, 5)] {
        for (where_, shape) in BODY_SHAPES {
            let diags = check_src(where_, &shape.replace("CALL", call));
            let reports = arity_reports(&diags, got);
            if reports != 1 {
                missed.push(format!(
                    "{got} arguments {where_}: {reports} E2005 report(s); {diags:?}"
                ));
            }
        }
    }
    assert!(
        missed.is_empty(),
        "each wrong-arity __mind_write must report E2005 exactly once:\n{}",
        missed.join("\n")
    );
}

#[test]
fn the_declared_arity_reports_no_arity_error_in_any_body_shape() {
    // Positive control: the same programs with the four-argument call carry no
    // arity diagnostic, so each report above comes from the argument count.
    let mut reported = Vec::new();
    for (where_, shape) in BODY_SHAPES {
        let diags = check_src(where_, &shape.replace("CALL", DECLARED_CALL));
        if diags
            .iter()
            .any(|d| d.message.contains("intrinsic `__mind_write` expects"))
        {
            reported.push(format!("{where_}: {diags:?}"));
        }
    }
    assert!(
        reported.is_empty(),
        "a four-argument __mind_write must not report the arity rule:\n{}",
        reported.join("\n")
    );
}
