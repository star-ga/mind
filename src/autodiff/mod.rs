// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

#![allow(dead_code, unused_variables, unused_imports)]

//! Minimal autodiff prototype (Phase 1).
//!
//! `grad(f)` returns a closure that runs `f()` and also returns a placeholder
//! gradient marker. This is a stub to unblock examples/tests and API design.
//!
//! # Example
//! ```
//! #[cfg(feature = "autodiff")]
//! {
//!     let g = mind::autodiff::grad(|| 21 + 21);
//!     let (v, d) = g();
//!     assert_eq!(v, 42);
//!     assert!(d.contains("placeholder"));
//! }
//! ```

pub fn grad<F, T>(f: F) -> impl Fn() -> (T, &'static str)
where
    F: Fn() -> T,
{
    move || {
        let v = f();
        (v, "∂(placeholder)")
    }
}
