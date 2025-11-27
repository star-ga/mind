// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
