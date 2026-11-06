// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Owner-preserving type-alias resolution for checker and lowering metadata.
//!
//! Source type spellings stay in the AST. MLIR ABI lowering consumes the
//! structural `TypeAnn` target through `IRModule::fn_signatures`; scalar
//! coercion helpers use the same scoped targets while lowering function bodies.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{Node, TypeAnn};
use crate::ir::IRModule;

thread_local! {
    /// Aliases owned by the module currently being lowered. Lowering has many
    /// recursive statement paths, so keeping the owner-scoped table here lets
    /// scalar coercion helpers canonicalize annotations without flattening
    /// aliases from sibling source modules into one global namespace.
    static ACTIVE_ALIASES: std::cell::RefCell<LocalTypeAliases> =
        std::cell::RefCell::new(LocalTypeAliases::default());
}

/// Module-local aliases collected once and reused by all signature/type users
/// in a pass. Checker and lowering signatures share this local alias policy.
#[derive(Clone, Default)]
pub(crate) struct LocalTypeAliases {
    aliases: BTreeMap<String, TypeAnn>,
    #[cfg(feature = "cross-module-imports")]
    imports: Vec<String>,
}

impl LocalTypeAliases {
    pub(crate) fn new(items: &[Node]) -> Self {
        let mut aliases = BTreeMap::new();
        collect_local_type_aliases(items, &mut aliases);
        Self {
            aliases,
            #[cfg(feature = "cross-module-imports")]
            imports: collect_imports(items),
        }
    }

    pub(crate) fn resolve(&self, ty: &TypeAnn) -> TypeAnn {
        resolve_or_original(ty, &self.aliases)
    }

    /// [`Self::resolve`], then read the result through the project registry:
    /// every visible project alias, at any nesting depth, becomes its target
    /// and every other project type its source key, so `array<defs.Q>` and
    /// `array<i32>` compare equal when `defs` exports `type Q = i32`.
    pub(crate) fn resolve_with_imports(&self, ty: &TypeAnn) -> TypeAnn {
        self.read_through_registry(self.resolve(ty))
    }

    #[cfg(feature = "cross-module-imports")]
    fn read_through_registry(&self, ty: TypeAnn) -> TypeAnn {
        crate::qualified_enums::canonical_type_here(&ty, &self.imports)
    }

    #[cfg(not(feature = "cross-module-imports"))]
    fn read_through_registry(&self, ty: TypeAnn) -> TypeAnn {
        ty
    }

    /// Make this module's aliases available to recursive lowering helpers.
    /// The guard restores the prior owner on every exit path, including a
    /// materialization refusal or a nested lowering invocation.
    pub(crate) fn install(&self) -> ActiveAliasesGuard {
        ACTIVE_ALIASES.with(|active| {
            let previous = std::mem::replace(&mut *active.borrow_mut(), self.clone());
            ActiveAliasesGuard(previous)
        })
    }
}

pub(crate) struct ActiveAliasesGuard(LocalTypeAliases);

impl Drop for ActiveAliasesGuard {
    fn drop(&mut self) {
        ACTIVE_ALIASES.with(|active| {
            *active.borrow_mut() = std::mem::take(&mut self.0);
        });
    }
}

/// Resolve a type annotation against the current source module, then resolve a
/// visible owner-qualified imported alias from the project registry. Cycles
/// preserve the original named annotation, matching `LocalTypeAliases`.
pub(crate) fn resolve_active(ty: &TypeAnn) -> TypeAnn {
    ACTIVE_ALIASES.with(|active| active.borrow().resolve_with_imports(ty))
}

#[cfg(feature = "cross-module-imports")]
fn collect_imports(items: &[Node]) -> Vec<String> {
    let mut imports = Vec::new();
    for item in items {
        match item {
            Node::Import { path, .. } => imports.push(path.join(".")),
            Node::Block { stmts, .. } => imports.extend(collect_imports(stmts)),
            _ => {}
        }
    }
    imports
}

/// Populate local function-signature metadata after resolving aliases declared
/// in the same parsed module. `Node::Block` is transparent here because the
/// parser represents a source-level `module name { ... }` wrapper as that node;
/// the project export and local-const collectors use the same rule.
pub(super) fn collect_local_fn_signatures(
    items: &[Node],
    aliases: &LocalTypeAliases,
    ir: &mut IRModule,
) {
    collect_signatures(items, aliases, ir);
}

fn collect_local_type_aliases(items: &[Node], aliases: &mut BTreeMap<String, TypeAnn>) {
    for item in items {
        match item {
            Node::TypeAlias { name, target, .. } => {
                aliases.insert(name.clone(), target.clone());
            }
            Node::Block { stmts, .. } => collect_local_type_aliases(stmts, aliases),
            _ => {}
        }
    }
}

fn collect_signatures(items: &[Node], aliases: &LocalTypeAliases, ir: &mut IRModule) {
    for item in items {
        match item {
            Node::FnDef(fd, _) if fd.type_params.is_empty() => {
                let param_types = fd
                    .params
                    .iter()
                    .map(|param| aliases.resolve(&param.ty))
                    .collect();
                let ret_type = fd.ret_type.as_ref().map(|ret| aliases.resolve(ret));
                ir.fn_signatures
                    .insert(fd.name.clone(), (param_types, ret_type));
            }
            Node::Block { stmts, .. } => collect_signatures(stmts, aliases, ir),
            _ => {}
        }
    }
}

fn resolve_or_original(ty: &TypeAnn, aliases: &BTreeMap<String, TypeAnn>) -> TypeAnn {
    resolve(ty, aliases, &mut BTreeSet::new()).unwrap_or_else(|| ty.clone())
}

/// Resolve aliases and wrapped element/target types. A cycle returns `None`
/// through the entire enclosing type, so callers preserve the original named
/// annotation and the existing MLIR missing-type refusal remains fail-closed.
fn resolve(
    ty: &TypeAnn,
    aliases: &BTreeMap<String, TypeAnn>,
    resolving: &mut BTreeSet<String>,
) -> Option<TypeAnn> {
    match ty {
        TypeAnn::Named(name) => {
            let Some(target) = aliases.get(name) else {
                return Some(ty.clone());
            };
            if !resolving.insert(name.clone()) {
                return None;
            }
            let resolved = resolve(target, aliases, resolving);
            resolving.remove(name);
            resolved
        }
        TypeAnn::Slice { mutable, element } => Some(TypeAnn::Slice {
            mutable: *mutable,
            element: Box::new(resolve(element, aliases, resolving)?),
        }),
        TypeAnn::Array { element, length } => Some(TypeAnn::Array {
            element: Box::new(resolve(element, aliases, resolving)?),
            length: *length,
        }),
        TypeAnn::Ref { mutable, target } => Some(TypeAnn::Ref {
            mutable: *mutable,
            target: Box::new(resolve(target, aliases, resolving)?),
        }),
        TypeAnn::Generic { name, args } => Some(TypeAnn::Generic {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| resolve(arg, aliases, resolving))
                .collect::<Option<Vec<_>>>()?,
        }),
        TypeAnn::Tuple { elements } => Some(TypeAnn::Tuple {
            elements: elements
                .iter()
                .map(|element| resolve(element, aliases, resolving))
                .collect::<Option<Vec<_>>>()?,
        }),
        TypeAnn::SparseTensor {
            layout,
            element,
            shape,
        } => Some(TypeAnn::SparseTensor {
            layout: *layout,
            element: Box::new(resolve(element, aliases, resolving)?),
            shape: shape.clone(),
        }),
        TypeAnn::RawPtr { mutable, pointee } => Some(TypeAnn::RawPtr {
            mutable: *mutable,
            pointee: Box::new(resolve(pointee, aliases, resolving)?),
        }),
        TypeAnn::FnPtr { params, ret } => Some(TypeAnn::FnPtr {
            params: params
                .iter()
                .map(|param| resolve(param, aliases, resolving))
                .collect::<Option<Vec<_>>>()?,
            ret: match ret {
                Some(ret) => Some(Box::new(resolve(ret, aliases, resolving)?)),
                None => None,
            },
        }),
        _ => Some(ty.clone()),
    }
}
