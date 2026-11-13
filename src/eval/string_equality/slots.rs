// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Declared slots whose declaration a stored value no longer proves.
//!
//! `mindc check` accepts a value of another type in many slots: a string for
//! an `i64` parameter, result, field, payload, element or module binding, and
//! an integer for a `string` one. A declaration proves what its slot holds
//! only while the module stores nothing else there, so a walk records every
//! such store (see [`super::flows`]) and the next walk reads the slot as
//! [`Fact::MaybeStr`]: two of them compared are refused, never compiled to a
//! handle compare (false for equal text) or to a byte compare through an
//! integer that is not a string record.
//!
//! The null handle `0` is the one integer a `string` slot may hold: the
//! byte-wise comparison defines it as equal only to itself, and reads no
//! record through it.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{Literal, Node};

use super::facts::Fact;

/// Every declared slot of one module that a store contradicts.
#[derive(Clone, Debug, Default)]
pub(super) struct Contradicted {
    /// Local function → indices of parameters given a contradicting value.
    pub(super) params: BTreeMap<String, BTreeSet<usize>>,
    /// Local functions that return a value their result type does not hold,
    /// or a string with no declared result type.
    pub(super) returns: BTreeSet<String>,
    /// Struct (by [`struct_key`]) → fields given a contradicting value.
    pub(super) fields: BTreeMap<String, BTreeSet<String>>,
    /// Variant (`Enum::Variant`) → payload positions given one.
    pub(super) payloads: BTreeMap<String, BTreeSet<usize>>,
    /// Module-level `let` / `const` bindings assigned one.
    pub(super) globals: BTreeSet<String>,
    /// Element facts of collections given an element they do not hold (by
    /// index assignment or a `push` / `set` / `insert` / `add`), keyed by
    /// the element fact rather than by binding so an alias of the collection
    /// is covered too. [`Fact::Unknown`] stands for a collection whose
    /// element type is not known: every element read is then maybe a string.
    pub(super) elements: Vec<Fact>,
    /// Record types given to a callee whose body is not analysed here
    /// together with a string through a non-string parameter (the `Vec` of
    /// `vec_push(v, "abc")`).
    pub(super) escaped_records: BTreeSet<String>,
    /// A string was handed to such a callee with no record beside it (the
    /// address of `__mind_store_i64(p, s)`).
    pub(super) escaped_raw: bool,
    /// Local binding site (see [`super::facts::site`]) → the joined facts of
    /// the values assigned to it that do not keep its fact ([`Fact::MaybeStr`]
    /// for a declared binding an assignment contradicts). Sites are AST
    /// addresses of one module, so these never leave the module's analysis.
    pub(super) locals: BTreeMap<usize, Fact>,
}

impl Contradicted {
    /// Add every slot of `other`; whether any was new.
    pub(super) fn absorb(&mut self, other: Contradicted) -> bool {
        fn absorb_map<K: Ord, V: Ord>(
            into: &mut BTreeMap<K, BTreeSet<V>>,
            from: BTreeMap<K, BTreeSet<V>>,
        ) -> bool {
            let mut grew = false;
            for (key, values) in from {
                let at = into.entry(key).or_default();
                for value in values {
                    grew |= at.insert(value);
                }
            }
            grew
        }
        fn absorb_set<V: Ord>(into: &mut BTreeSet<V>, from: BTreeSet<V>) -> bool {
            from.into_iter()
                .fold(false, |grew, v| into.insert(v) | grew)
        }
        let mut grew = absorb_map(&mut self.params, other.params);
        grew |= absorb_map(&mut self.fields, other.fields);
        grew |= absorb_map(&mut self.payloads, other.payloads);
        grew |= absorb_set(&mut self.returns, other.returns);
        grew |= absorb_set(&mut self.globals, other.globals);
        grew |= absorb_set(&mut self.escaped_records, other.escaped_records);
        for element in other.elements {
            grew |= self.add_element(element);
        }
        for (site, fact) in other.locals {
            grew |= self.add_local(site, fact);
        }
        grew |= other.escaped_raw && !std::mem::replace(&mut self.escaped_raw, true);
        grew
    }

    /// The slots other modules of a project build can declare or read:
    /// every one but the module's own local and module-level bindings and
    /// its element facts. Element facts are keyed by fact, not by slot, so
    /// one module's store into its own `[u8; N]` names every `[u8; N]` of
    /// the project: the project summary keeps them per module instead and
    /// gives them only to the modules that can exchange a collection with it
    /// (see `project::Summary::seed`).
    #[cfg(feature = "cross-module-imports")]
    pub(super) fn shared(&self) -> Contradicted {
        Contradicted {
            locals: BTreeMap::new(),
            globals: BTreeSet::new(),
            elements: Vec::new(),
            ..self.clone()
        }
    }

    /// Record a value with `fact` assigned to the local binding at `site`;
    /// whether the recorded fact changed.
    pub(super) fn add_local(&mut self, site: usize, fact: Fact) -> bool {
        match self.locals.get_mut(&site) {
            Some(at) => {
                let joined = super::facts::join(at.clone(), fact);
                let grew = joined != *at;
                *at = joined;
                grew
            }
            None => {
                self.locals.insert(site, fact);
                true
            }
        }
    }

    /// Record an element fact; whether it was new.
    pub(super) fn add_element(&mut self, element: Fact) -> bool {
        let new = !self.elements.contains(&element);
        if new {
            self.elements.push(element);
        }
        new
    }

    /// The fact of one element of a collection whose elements have `element`.
    pub(super) fn element(&self, element: Fact) -> Fact {
        let stored = self.elements.contains(&element) || self.elements.contains(&Fact::Unknown);
        if stored && !matches!(element, Fact::MaybeStr | Fact::Never | Fact::Any) {
            Fact::MaybeStr
        } else {
            element
        }
    }
}

/// Whether a value with `fact` may be, or have a part that may be, a string
/// on some path: a string, a maybe-string, a value of a type this module
/// cannot see ([`Fact::Any`]), or a value whose type the pass did not
/// establish ([`Fact::Unknown`]). A value is taken to hold no string only
/// when its type is known to hold none.
pub(super) fn holds_str(fact: &Fact) -> bool {
    match fact {
        Fact::Str | Fact::MaybeStr | Fact::Any | Fact::Unknown => true,
        Fact::Elems(part) | Fact::Map(part) => holds_str(part),
        Fact::Sum(ok, err) => holds_str(ok) || holds_str(err),
        Fact::Tuple(parts) => parts.iter().any(holds_str),
        Fact::Record(_) | Fact::NotStr | Fact::Never => false,
    }
}

/// Whether storing a value with fact `value` where `declared` is declared
/// stores what the declaration says the slot cannot hold, at any depth: a
/// value that may be a string ([`holds_str`], so also a value whose type the
/// pass did not establish) where no string is declared, or a value not
/// proven a string where a string is declared. A slot whose type is not
/// known or not visible proves nothing, so no store contradicts it.
pub(super) fn contradicts(declared: &Fact, value: &Fact) -> bool {
    match (declared, value) {
        (Fact::Unknown | Fact::MaybeStr | Fact::Any, _) | (_, Fact::Never) => false,
        (Fact::Str, value) => *value != Fact::Str,
        (Fact::Elems(d), Fact::Elems(v)) | (Fact::Map(d), Fact::Map(v)) => contradicts(d, v),
        (Fact::Tuple(ds), Fact::Tuple(vs)) => ds.iter().zip(vs).any(|(d, v)| contradicts(d, v)),
        (Fact::Sum(d, e), Fact::Sum(v, f)) => contradicts(d, v) || contradicts(e, f),
        (_, value) => holds_str(value),
    }
}

/// The key of struct `owner` in [`Contradicted::fields`]: its bare name, so a
/// struct named through its module (`provider.Box`, `crate.provider.Box`)
/// and by its own module (`Box`) share one entry.
pub(super) fn struct_key(owner: &str) -> &str {
    owner.rsplit('.').next().unwrap_or(owner)
}

/// Whether `node` is the integer literal `0`: the null handle, the one
/// integer a `string` slot may hold.
pub(super) fn is_null(node: &Node) -> bool {
    match node {
        Node::Lit(Literal::Int(0), _) => true,
        Node::Paren(inner, _) => is_null(inner),
        _ => false,
    }
}
