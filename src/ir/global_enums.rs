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

//! Whole-project enum registry. It lives here so `ir/mod.rs` stays inside
//! its pinned line budget.

/// Whole-project enum registry; single-file compilation leaves it empty.
#[derive(Debug, Clone, Default)]
pub struct GlobalEnums {
    /// Enum type names declared across the project. The parser checks this to
    /// decide whether a dotted `First.Second` reference is a variant access.
    pub names: Vec<String>,
    /// `"Enum::Variant"` → ordinal tag, mirroring the per-module `EnumDef`
    /// lowering. Merged into `IRModule::enum_variant_tags`.
    #[cfg(feature = "std-surface")]
    pub variant_tags: std::collections::BTreeMap<String, i64>,
    /// `"Enum::Variant"` → declared positional payload field types.
    #[cfg(feature = "std-surface")]
    pub payload_types: std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
    /// `"Enum"` → heap-record slot count (`1 + max payload arity`).
    #[cfg(feature = "std-surface")]
    pub slots: std::collections::BTreeMap<String, usize>,
    /// Enum type names that carry a payload on ≥1 variant (boxed enums).
    #[cfg(feature = "std-surface")]
    pub boxed: std::collections::BTreeSet<String>,
    /// `"Enum::Variant"` → struct-variant field NAMES (parallel to
    /// `payload_types`), present only for a struct variant `V { f: T }`.
    #[cfg(feature = "std-surface")]
    pub struct_field_names: std::collections::BTreeMap<String, Vec<String>>,
    /// Legacy bare-name project struct schemas in declaration order.
    #[cfg(feature = "std-surface")]
    pub structs: std::collections::BTreeMap<String, (Vec<String>, Vec<crate::ast::TypeAnn>)>,
    /// Collision-safe transient schemas keyed by owner, e.g. `crate.parser.Node`.
    #[cfg(feature = "std-surface")]
    pub qualified_structs:
        std::collections::BTreeMap<String, (Vec<String>, Vec<crate::ast::TypeAnn>)>,
    /// Every FUNCTION declared across the project → its declared return type.
    /// Lets a `let x = f(...)` infer `x`'s type (string / collection / struct)
    /// for method-call resolution — `let raw = decorator_arg_string(d); raw.split(…)`.
    #[cfg(feature = "std-surface")]
    pub fn_returns: std::collections::BTreeMap<String, crate::ast::TypeAnn>,
    #[cfg(feature = "cross-module-imports")]
    pub(crate) qualified: crate::qualified_enums::Registry,
    /// The project's modules for the cross-module string slots; `None` below two.
    #[cfg(feature = "cross-module-imports")]
    pub(crate) string_project: Option<std::sync::Arc<crate::eval::string_equality::ProjectModules>>,
}
