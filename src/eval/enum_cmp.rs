// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Equality of two enum values in the tree evaluator.
//!
//! A compiled program compares an enum with no payload variant by its tag, and
//! a boxed enum (one with any payload variant) by its heap record, not by its
//! value: two `Shape::Empty` values compare unequal there. Only an enum known
//! to be unit-only therefore has an equality the evaluator and the compiled
//! program agree on. Any other enum, and any operator other than `==` / `!=`,
//! stays unsupported, the same as the catch-all arm.

use std::cell::RefCell;
use std::collections::BTreeMap;

use super::{EvalError, Value};
use crate::ast::{BinOp, Node};

thread_local! {
    /// Enum name -> whether any variant carries a payload, for the module
    /// under test.
    static ENUM_DEFS: RefCell<BTreeMap<String, bool>> = const { RefCell::new(BTreeMap::new()) };
}

/// Restores the previous module's enum declarations on drop.
pub(crate) struct EnumDefsGuard(BTreeMap<String, bool>);

impl Drop for EnumDefsGuard {
    fn drop(&mut self) {
        ENUM_DEFS.with(|defs| *defs.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Record the enum declarations in `items` for one evaluation; the `mindc test`
/// runner installs the module under test.
pub(crate) fn install_enum_defs(items: &[Node]) -> EnumDefsGuard {
    let mut defs = BTreeMap::new();
    let mut pending: Vec<&Node> = items.iter().collect();
    while let Some(item) = pending.pop() {
        match item {
            Node::EnumDef { name, variants, .. } => {
                let boxed = variants.iter().any(|variant| !variant.payload.is_empty());
                defs.insert(name.clone(), boxed);
            }
            Node::Block { stmts, .. } => pending.extend(stmts),
            _ => {}
        }
    }
    EnumDefsGuard(ENUM_DEFS.with(|cell| cell.replace(defs)))
}

pub(super) fn cmp(op: BinOp, left: Value, right: Value) -> Result<Value, EvalError> {
    let (
        Value::Enum {
            variant: left_variant,
            payload: left_payload,
        },
        Value::Enum {
            variant: right_variant,
            payload: right_payload,
        },
    ) = (left, right)
    else {
        return Err(EvalError::Unsupported);
    };
    if !matches!(op, BinOp::Eq | BinOp::Ne) {
        return Err(EvalError::Unsupported);
    }
    for variant in [&left_variant, &right_variant] {
        match unit_only(variant) {
            Some(true) => {}
            Some(false) => {
                return Err(EvalError::UnsupportedMsg(format!(
                    "`{variant}` belongs to an enum with a payload variant; a compiled \
                     program compares those values by heap record, not by value, so \
                     the evaluator does not compare them"
                )));
            }
            None => return Err(EvalError::Unsupported),
        }
    }
    let equal = left_variant == right_variant && left_payload == right_payload;
    Ok(Value::Int(i64::from(equal == matches!(op, BinOp::Eq))))
}

/// `Some(true)` when `variant` names a variant of an enum with no payload
/// variant, `Some(false)` when its enum is boxed, `None` when no declaration
/// of its enum is known. The project registry answers first; the module under
/// evaluation answers for its own declarations.
fn unit_only(variant: &str) -> Option<bool> {
    let (enum_key, _) = variant.rsplit_once("::")?;
    crate::ir::with_global_enums(|global| registry_unit_only(global, variant, enum_key)).or_else(
        || {
            let name = enum_key.rsplit("::").next().unwrap_or(enum_key);
            ENUM_DEFS.with(|defs| defs.borrow().get(name).map(|boxed| !boxed))
        },
    )
}

fn registry_unit_only(
    global: &crate::ir::GlobalEnums,
    variant: &str,
    enum_key: &str,
) -> Option<bool> {
    #[cfg(feature = "cross-module-imports")]
    if global.qualified.variant_tags.contains_key(variant) {
        return Some(!global.qualified.boxed.contains(enum_key));
    }
    #[cfg(feature = "std-surface")]
    if global.variant_tags.contains_key(variant) {
        return Some(!global.boxed.contains(enum_key));
    }
    let _ = (global, variant, enum_key);
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(variant: &str, payload: Vec<Value>) -> Value {
        Value::Enum {
            variant: variant.to_string(),
            payload,
        }
    }

    fn with_defs<R>(src: &str, f: impl FnOnce() -> R) -> R {
        let module = crate::parser::parse(src).expect("parse enum declarations");
        let _defs = install_enum_defs(&module.items);
        f()
    }

    const DECLS: &str =
        "enum Color {\n    Red,\n    Blue,\n}\nenum Shape {\n    Circle(i64),\n    Empty,\n}\n";

    #[test]
    fn unit_only_enum_compares_by_variant() {
        with_defs(DECLS, || {
            let red = || value("Color::Red", Vec::new());
            let blue = || value("Color::Blue", Vec::new());
            assert!(matches!(cmp(BinOp::Eq, red(), red()), Ok(Value::Int(1))));
            assert!(matches!(cmp(BinOp::Eq, red(), blue()), Ok(Value::Int(0))));
            assert!(matches!(cmp(BinOp::Ne, red(), blue()), Ok(Value::Int(1))));
            assert!(matches!(cmp(BinOp::Ne, red(), red()), Ok(Value::Int(0))));
        });
    }

    #[test]
    fn ordering_operators_stay_unsupported() {
        with_defs(DECLS, || {
            for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
                let got = cmp(
                    op,
                    value("Color::Red", Vec::new()),
                    value("Color::Blue", Vec::new()),
                );
                assert!(matches!(got, Err(EvalError::Unsupported)));
            }
        });
    }

    #[test]
    fn a_boxed_enum_is_not_compared() {
        with_defs(DECLS, || {
            let unit = cmp(
                BinOp::Eq,
                value("Shape::Empty", Vec::new()),
                value("Shape::Empty", Vec::new()),
            );
            assert!(
                matches!(&unit, Err(EvalError::UnsupportedMsg(m)) if m.contains("heap record")),
                "a unit variant of a boxed enum: {unit:?}"
            );
            let payload = cmp(
                BinOp::Ne,
                value("Shape::Circle", vec![Value::Int(1)]),
                value("Shape::Circle", vec![Value::Int(2)]),
            );
            assert!(matches!(payload, Err(EvalError::UnsupportedMsg(_))));
        });
    }

    #[test]
    fn an_undeclared_enum_is_not_compared() {
        let got = cmp(
            BinOp::Eq,
            value("Color::Red", Vec::new()),
            value("Color::Red", Vec::new()),
        );
        assert!(matches!(got, Err(EvalError::Unsupported)));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn the_project_registry_decides_boxing() {
        let mut enums = crate::ir::GlobalEnums::default();
        enums.variant_tags.insert("Mode::On".to_string(), 0);
        enums.variant_tags.insert("Tree::Leaf".to_string(), 0);
        enums.boxed.insert("Tree".to_string());
        crate::ir::set_global_enums(enums);
        let on = || value("Mode::On", Vec::new());
        let leaf = || value("Tree::Leaf", Vec::new());
        let unit = cmp(BinOp::Eq, on(), on());
        let boxed = cmp(BinOp::Eq, leaf(), leaf());
        crate::ir::set_global_enums(crate::ir::GlobalEnums::default());
        assert!(matches!(unit, Ok(Value::Int(1))));
        assert!(matches!(boxed, Err(EvalError::UnsupportedMsg(_))));
    }
}
