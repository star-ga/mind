// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A function name used as a VALUE (`let x = null`, `return helper`).
//!
//! First-class functions do not exist: no backend can materialise a function
//! as a value, so native lowering stops at its fail-closed undefined-identifier
//! panic when such a reference reaches it. The resolver nevertheless accepted
//! the name, because every function — a module `fn`, an `extern` fn, or any
//! std-surface export such as `std.json`'s `null()` — is a resolvable name for
//! the CALL question. This pass answers the VALUE question: a bare identifier in
//! value position whose only meaning is a function is refused at check time
//! with `FN_AS_VALUE_CODE` (E2037). `mindc check` and `mindc build` then agree,
//! and the project builder, which refuses to embed a module carrying that code
//! as a runtime fallback, never lowers it into the panic.
//!
//! The E2002 (unknown identifier) and E2012 (function-value call) verdicts are
//! deliberately unchanged at such a position: the name is resolvable and no
//! call is made, and the self-hosted checker ports of those two rules are
//! verified against `mindc check` position by position.
//!
//! Covered: the module's own functions and std-surface names that are only
//! ever functions. A function name that resolves solely through another
//! project module's exports is not classified here.
//!
//! Member-access receivers are walked with `value = false`: `sha256.hash(..)`
//! names the `std.sha256` module even though that module also exports a
//! `sha256` function.

use std::sync::OnceLock;

use super::{FxSet, Resolver, Unresolved, cse, qvalue};
use crate::ast::{Literal, Node, Span};

/// Names declared as functions and as nothing else in the same declaration set.
pub(super) fn only_fns(fns: &FxSet, non_fns: &FxSet) -> FxSet {
    fns.iter()
        .filter(|name| !non_fns.contains(*name))
        .cloned()
        .collect()
}

/// Std-surface names that are functions and never a const, `let`, type, or
/// enum variant anywhere in the bundled std sources. Built once, like the
/// std export-name cache it refines.
fn stdlib_fn_only_names() -> &'static FxSet {
    static CACHE: OnceLock<FxSet> = OnceLock::new();
    CACHE.get_or_init(|| {
        #[allow(unused_mut)]
        let (mut fns, mut values) = (FxSet::default(), FxSet::default());
        #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
        for (_, module) in crate::project::stdlib::parsed_stdlib_modules() {
            super::collect_non_fn_decl_names(&module, &mut values, &mut fns);
            let mut enums = super::FxMap::default();
            super::collect_enum_variants(&module, &mut enums);
            values.extend(enums.into_values().flatten());
        }
        only_fns(&fns, &values)
    })
}

/// Does a project enum (any module) declare a variant spelled `name`? Such a
/// bare name may be a visible imported unit variant, so it is never refused
/// here; the variant-visibility check owns that question.
fn project_variant_named(name: &str) -> bool {
    #[cfg(feature = "cross-module-imports")]
    {
        crate::ir::with_global_enums(|g| g.qualified.has_bare_variant(name))
    }
    #[cfg(not(feature = "cross-module-imports"))]
    {
        let _ = name;
        false
    }
}

/// Is the project module that exports `name` to the current one exporting a
/// FUNCTION (the same owner selection the call resolver uses)? In a project
/// build the bundled std modules sit in the module table too, so a std
/// function is also a visible exported symbol; one exported as a value stays
/// a value.
fn visible_sibling_fn(name: &str) -> bool {
    #[cfg(feature = "cross-module-imports")]
    {
        super::super::cm_lookup_fn(name).is_some()
    }
    #[cfg(not(feature = "cross-module-imports"))]
    {
        let _ = name;
        false
    }
}

impl Resolver<'_> {
    /// Walk a member-access receiver. A bare identifier there may name a
    /// module or type rather than a value, so it is resolved without the
    /// function-as-value rule; any other receiver expression is walked as usual.
    pub(super) fn walk_receiver(&mut self, receiver: &Node) {
        match receiver {
            Node::Lit(Literal::Ident(name), span) => self.check_ident(name, *span, false),
            other => self.walk(other),
        }
    }

    /// True iff `name`, in value position, resolves ONLY as a function: it IS
    /// a function of this module or a function-only std-surface export, and it
    /// is not a local binding, a module-level const / `let` / type, an enum
    /// variant, or a value exported by another project module. Local and
    /// module-level values are excluded first, so they never build the std
    /// function table (parsed once, on first use).
    pub(super) fn fn_used_as_value(&self, name: &str, span: Span) -> bool {
        // `bytes[N]` is the builtin fixed-byte-buffer value base.
        if name == "bytes"
            || self.scopes.contains(name)
            || self.syms.non_fn_decls.contains(name)
            || self
                .syms
                .enums
                .values()
                .any(|variants| variants.contains(name))
        {
            return false;
        }
        let is_fn = self.syms.fns.contains(name) || stdlib_fn_only_names().contains(name);
        is_fn
            && !qvalue(span, name)
            && (!cse(name) || visible_sibling_fn(name))
            && !project_variant_named(name)
            // A sibling module's `const` inlines at lowering (PROJECT_CONSTS).
            && crate::ir::module_const_value(name).is_none()
    }

    /// Record a function-as-value reference. `fn_value_call` with
    /// `is_call == false` selects the value-position message (E2037).
    pub(super) fn push_fn_as_value(&mut self, name: &str, span: Span) {
        self.out.push(Unresolved {
            name: name.to_string(),
            span,
            is_call: false,
            suggestion: None,
            variant_of: None,
            undeclared_assign: false,
            fn_value_call: true,
            non_fn_call: false,
        });
    }
}
