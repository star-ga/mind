// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `Ok(..)` / `Err(..)` in the tree evaluator.
//!
//! The compiled build resolves a bare `Ok` / `Err` to the prelude variant
//! `Result::Ok` / `Result::Err` (`lower::resolve_bare_variant`: a bare name with
//! one owner is that owner), builds the same boxed record for either spelling,
//! and tests the same tag in a `match` arm. The evaluator keeps the spelling it
//! was given in the [`Value::Enum`](super::Value) it builds, so two spellings of
//! one variant must compare equal here, at construction and at every pattern.

/// Is `callee` a `Result` constructor call (`Ok(..)` / `Err(..)`, bare or
/// qualified)? A qualified `Result::Ok(..)` already carries `::`.
pub(super) fn is_result_ctor(callee: &str) -> bool {
    matches!(callee, "Ok" | "Err")
}

/// The qualified prelude spelling of a bare `Ok` / `Err`; every other name is
/// returned as written.
fn qualified(name: &str) -> &str {
    match name {
        "Ok" => "Result::Ok",
        "Err" => "Result::Err",
        other => other,
    }
}

/// Does the variant of an evaluated enum value (`variant`) name the same
/// variant as a pattern path (`path`)? Equal spellings match, and so do the
/// bare and qualified spellings of a `Result` variant.
pub(super) fn same_variant(variant: &str, path: &str) -> bool {
    variant == path || qualified(variant) == qualified(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_and_qualified_result_variants_are_one_variant() {
        assert!(same_variant("Ok", "Ok"));
        assert!(same_variant("Ok", "Result::Ok"));
        assert!(same_variant("Result::Err", "Err"));
        assert!(!same_variant("Ok", "Err"));
        assert!(!same_variant("Ok", "Result::Err"));
    }

    #[test]
    fn other_variants_keep_exact_spelling() {
        assert!(same_variant("Mode::On", "Mode::On"));
        assert!(!same_variant("Mode::On", "On"));
        assert!(!same_variant("Some", "Option::Some"));
        assert!(!same_variant("Other::Ok", "Ok"));
    }

    #[test]
    fn only_ok_and_err_are_result_constructors() {
        assert!(is_result_ctor("Ok") && is_result_ctor("Err"));
        assert!(
            !is_result_ctor("Okay") && !is_result_ctor("Some") && !is_result_ctor("Result::Ok")
        );
    }
}
