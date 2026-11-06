// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Whether a bare call name is a math, tensor, or autodiff builtin.
//!
//! The closed list stays in `resolve` with the check that uses it. This
//! predicate is the string pass's view of that list, and it lives here so
//! `resolve.rs` stays inside its pinned line budget.

/// Whether `name` is one of the bare math / tensor / autodiff builtins
/// ([`super::resolve::BARE_BUILTINS`]): none of them takes or returns a string.
#[cfg(feature = "std-surface")]
pub(crate) fn is_bare_builtin(name: &str) -> bool {
    super::resolve::BARE_BUILTINS.binary_search(&name).is_ok()
}
