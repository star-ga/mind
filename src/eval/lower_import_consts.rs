// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Imported-constant resolution for lowering.
//!
//! `crate::ir::module_const_value` unions the per-module const table and the
//! whole-project cross-module table, but `lower_expr`'s `Lit(Ident)` arm
//! consulted it only under `std-surface`. A `cross-module-imports` build
//! without that feature therefore dropped an imported constant the type
//! checker had accepted onto the fail-closed `undefined identifier` panic.
//! This sibling (lower.rs is over its size ceiling) re-opens the lookup
//! unconditionally, with the same const-reference cycle guard, so the panic
//! stays unreachable for every imported constant name resolution accepted —
//! and a constant is never silently emitted as `const 0`.

use super::ast;

/// Inline `name`'s constant value through `lower`, or `None` when `name` is
/// not a module/project constant. Cycles (`const A = B; const B = A`) fail
/// loud rather than recursing forever.
pub(super) fn lower_imported_const<T>(
    name: &str,
    lower: impl FnOnce(&ast::Node) -> T,
) -> Option<T> {
    let value = crate::ir::module_const_value(name)?;
    if !crate::ir::begin_resolving_const(name) {
        panic!(
            "const `{name}` is defined in terms of itself (cyclic const \
             reference) — refusing to inline (that would loop)."
        );
    }
    let out = lower(&value);
    crate::ir::end_resolving_const(name);
    Some(out)
}
