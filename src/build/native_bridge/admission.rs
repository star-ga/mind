// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Source-level refusals the native bridge applies to the merged program.
//!
//! * An entry `main` that declares parameters is refused. The frozen compiler
//!   does not bind `main`'s parameters to the process arguments, so such a
//!   program would run with every parameter 0: `fn main(argc: i64, argv: i64)`
//!   built and exited 40 natively where the MLIR build of the same source
//!   exited 23. Refused until the seed binds them.
//! * The name check the MLIR build path treats as fatal (an unknown identifier
//!   E2002, a function used as a value E2037) and an unknown enum variant
//!   (E2008) are refused with the type checker's own diagnostics. Without the
//!   check these reached lowering's fail-closed `panic!` and the build died
//!   with exit 101.
//! * A symbolic tensor dimension read as a VALUE (`reshape(x, [K, 1])` where
//!   `K` is only a dimension of a parameter's type) is refused. The type
//!   checker deliberately lets the name resolve, but lowering has no value for
//!   it and panicked on the same undefined-identifier guard.
//! * Lowering itself runs behind a panic boundary, so any other construct that
//!   still reaches a lowering `panic!` becomes a refusal with its message, not
//!   an exit 101.

use std::collections::BTreeSet;

use crate::ast::{Module, Node};
use crate::diagnostics::{
    ColorChoice, Diagnostic, DiagnosticEmitter, DiagnosticFormat, Severity, Span,
};
use crate::type_checker::resolve;

/// Refuse a parameterized entry outside the exact wave-three argc/argv ABI.
pub(super) fn refuse_entry_main_with_params(ast: &Module, path: &str) -> Result<(), String> {
    for item in &ast.items {
        let Node::FnDef(f, _) = item else { continue };
        if f.name != "main" || f.params.is_empty() {
            continue;
        }
        if f.type_params.is_empty()
            && f.params.len() == 2
            && f.params[0].name == "argc"
            && f.params[1].name == "argv"
            && f.params
                .iter()
                .all(|p| p.ty == crate::ast::TypeAnn::ScalarI64)
            && f.ret_type == Some(crate::ast::TypeAnn::ScalarI64)
        {
            continue;
        }
        let params: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        return Err(format!(
            "`{path}`: the entry `fn main` declares parameter(s) {params:?}, but the native \
             backend admits only `fn main()` or `fn main(argc: i64, argv: i64) -> i64`. \
             Correct the entry signature or rerun with `--backend mlir`. No artifact written. \
             (native.entry_main_has_params)"
        ));
    }
    Ok(())
}

/// The type-check codes whose construct lowering cannot materialise.
const NAME_FATAL_CODES: [&str; 3] = [
    resolve::UNKNOWN_IDENT_CODE,
    resolve::UNKNOWN_VARIANT_CODE,
    crate::type_checker::FN_AS_VALUE_CODE,
];

fn render(diags: &[Diagnostic], text: &str) -> String {
    let emitter = DiagnosticEmitter::new(DiagnosticFormat::Human, ColorChoice::Never);
    diags
        .iter()
        .map(|d| emitter.render_human(d, Some(text)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Refuse unresolved names, using the type checker's own diagnostics.
pub(super) fn refuse_unresolved_names(ast: &Module, text: &str, path: &str) -> Result<(), String> {
    let fatal: Vec<Diagnostic> = crate::type_checker::check_module_types_in_file(
        ast,
        text,
        Some(path),
        &crate::type_checker::TypeEnv::default(),
    )
    .into_iter()
    .filter(|d| d.severity == Severity::Error && NAME_FATAL_CODES.contains(&d.code))
    .collect();
    if fatal.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{}\n`{path}`: unresolved name(s) above; the native backend refuses a program it \
         cannot lower. No artifact written. (native.unresolved_name)",
        render(&fatal, text)
    ))
}

/// Refuse a symbolic tensor dimension that a function body reads as a value.
pub(super) fn refuse_symbolic_dims_read_as_values(
    ast: &Module,
    text: &str,
    path: &str,
) -> Result<(), String> {
    let mut found: Vec<Diagnostic> = Vec::new();
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            let Node::FnDef(f, _) = node else { return };
            let mut dims = BTreeSet::new();
            for p in &f.params {
                resolve::collect_shape_vars(&p.ty, &mut dims);
            }
            if let Some(rt) = &f.ret_type {
                resolve::collect_shape_vars(rt, &mut dims);
            }
            if dims.is_empty() {
                return;
            }
            // Resolve WITHOUT the dimension names the type checker pre-binds:
            // what is then unresolved and IS a dimension name is a dimension
            // read as a value. Everything else keeps the type checker's verdict.
            let injected: BTreeSet<String> = f.type_params.iter().cloned().collect();
            let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
            let (unresolved, _) = resolve::resolve_fn_body(&f.body, &params, ast, &injected);
            for u in unresolved {
                if u.is_call || u.variant_of.is_some() || u.undeclared_assign {
                    continue;
                }
                if !dims.contains(&u.name) {
                    continue;
                }
                let start = u.span.start().min(text.len());
                let end = u.span.end().min(text.len());
                let mut d = Diagnostic::error(
                    "native",
                    "native.symbolic_dim_as_value",
                    format!(
                        "symbolic tensor dimension `{}` of fn `{}` is read as a value; the \
                         native backend has no runtime value for a symbolic dimension",
                        u.name, f.name
                    ),
                );
                d.span = Some(Span::from_offsets(text, start, end, Some(path)));
                d.help = Some(
                    "pass the extent as an `i64` parameter, or rerun with `--backend mlir`"
                        .to_string(),
                );
                found.push(d);
            }
        });
    }
    if found.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{}\n`{path}`: symbolic dimension(s) read as values; no artifact written. \
         (native.symbolic_dim_as_value)",
        render(&found, text)
    ))
}

/// Lower behind a panic boundary: a lowering `panic!` becomes a refusal.
pub(super) fn lower_without_panicking(
    ast: &Module,
    path: &str,
) -> Result<crate::ir::IRModule, String> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let lowered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::eval::lower_to_ir(ast)
    }));
    std::panic::set_hook(previous);
    match lowered {
        Ok(Ok(ir)) => Ok(ir),
        Ok(Err(refusal)) => Err(format!(
            "`{path}`: {refusal} (lower.materialization_refused)"
        )),
        Err(payload) => {
            let why = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("(no message)");
            Err(format!(
                "`{path}`: lowering refused the program: {why} No artifact written. \
                 (lower.refused_construct)"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(src: &str) -> Result<(), String> {
        let ast = crate::parser::parse(src).expect("parse");
        refuse_symbolic_dims_read_as_values(&ast, src, "t.mind")
    }

    #[test]
    fn a_dimension_read_as_a_value_is_refused() {
        let err =
            dims("fn f(x: tensor<f32[K]>) -> tensor<f32[K, 1]> { return reshape(x, [K, 1]); }\n")
                .unwrap_err();
        assert!(err.contains("`K`") && err.contains("t.mind:1:"), "{err}");
    }

    #[test]
    fn a_dimension_used_only_in_types_is_admitted() {
        assert!(dims("fn f(x: tensor<f32[K]>) -> tensor<f32[K]> { return x; }\n").is_ok());
    }

    #[test]
    fn a_local_or_module_binding_of_the_same_name_is_admitted() {
        assert!(dims("fn f(x: tensor<f32[K]>) -> i64 { let K = 3; return K; }\n").is_ok());
        assert!(dims("const K: i64 = 4;\nfn f(x: tensor<f32[K]>) -> i64 { return K; }\n").is_ok());
        assert!(dims("fn f(x: tensor<f32[K]>, K: i64) -> i64 { return K; }\n").is_ok());
    }

    #[test]
    fn only_a_parameterised_main_is_refused() {
        let ok = crate::parser::parse("fn main() -> i64 { return 0; }\n").expect("parse");
        assert!(refuse_entry_main_with_params(&ok, "t.mind").is_ok());
        let helper = crate::parser::parse(
            "fn h(a: i64) -> i64 { return a; }\nfn main() -> i64 { return h(1); }\n",
        )
        .expect("parse");
        assert!(refuse_entry_main_with_params(&helper, "t.mind").is_ok());
        let bad =
            crate::parser::parse("fn main(argc: i64) -> i64 { return argc; }\n").expect("parse");
        assert!(refuse_entry_main_with_params(&bad, "t.mind").is_err());
    }

    #[test]
    fn a_lowering_panic_becomes_a_refusal() {
        // An unknown variant of a known enum reaches lowering's fail-closed
        // panic when the name check is skipped; the boundary reports it.
        let src =
            "enum Color { Red, Blue }\nfn main() -> i64 { let c = Color::Green; return 0; }\n";
        let ast = crate::parser::parse(src).expect("parse");
        let err = lower_without_panicking(&ast, "t.mind").unwrap_err();
        assert!(err.contains("lower.refused_construct"), "{err}");
    }
}
