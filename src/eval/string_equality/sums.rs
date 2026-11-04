// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Payload facts of the built-in `Option` / `Result`.
//!
//! The prelude enums are generic, so their registered payload type (`T`)
//! says nothing. Their payload facts come from the declared type instead
//! (`Option<string>`: the `Some` payload is a string) and from the value a
//! constructor is given (`Some("abc")`), so a payload bound by a `match` arm
//! is typed, and a string payload stored where an integer is declared is a
//! recorded store like any other.

use super::facts::{Decls, Fact};

/// A variant of the built-in `Option` / `Result` prelude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Variant {
    /// `Some(v)` or `Ok(v)`: the payload is the first part of [`Fact::Sum`].
    Ok,
    /// `Err(e)`: the payload is the second part.
    Err,
    /// `None`: no payload.
    None,
}

impl Variant {
    /// The fact of this variant built with a payload of fact `value`.
    pub(super) fn build(self, value: Fact) -> Fact {
        let never = || Box::new(Fact::Never);
        match self {
            Variant::Ok => Fact::Sum(Box::new(value), never()),
            Variant::Err => Fact::Sum(never(), Box::new(value)),
            Variant::None => Fact::Sum(never(), never()),
        }
    }

    /// The fact of this variant's payload in a value with fact `fact`;
    /// `None` when `fact` is not a sum.
    pub(super) fn payload_of(self, fact: &Fact) -> Option<Fact> {
        let part = match (self, fact) {
            (Variant::Ok, Fact::Sum(ok, _)) => (**ok).clone(),
            (Variant::Err, Fact::Sum(_, err)) => (**err).clone(),
            (_, Fact::MaybeStr) => Fact::MaybeStr,
            _ => return None,
        };
        // A payload the sum never holds: the arm is unreachable.
        Some(if part == Fact::Never {
            Fact::Unknown
        } else {
            part
        })
    }
}

impl Decls {
    /// The built-in variant `path` names, unless a module enum of that name
    /// (or, for a bare name, a module variant of that name) shadows it.
    pub(super) fn builtin_variant(&self, path: &str) -> Option<Variant> {
        let (owner, variant) = match path.rsplit_once("::") {
            Some((head, variant)) => (Some(head.rsplit('.').next().unwrap_or(head)), variant),
            None => (None, path),
        };
        let which = match (owner, variant) {
            (None | Some("Option"), "Some") | (None | Some("Result"), "Ok") => Variant::Ok,
            (None | Some("Result"), "Err") => Variant::Err,
            (None | Some("Option"), "None") => Variant::None,
            _ => return None,
        };
        let shadowed = match owner {
            Some(owner) => self.local_enum(owner),
            None => self.local_variant(variant),
        };
        (!shadowed).then_some(which)
    }
}
