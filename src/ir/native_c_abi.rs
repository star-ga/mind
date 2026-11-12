// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Source-derived C declaration contracts, not native-call admission.
//!
//! Legacy IR can erase signedness, pointer pointees and safety annotations.
//! Derive this inventory from the original captured AST before lowering. The
//! caller binds that AST, resolved namespace and native descriptor to the SAME
//! source image. Equality here proves agreement of declaration metadata only;
//! it does not implement calls, linking, pointer safety or a runtime ABI.

use crate::ast::{CallConv, Module, Node, TypeAnn};
use std::collections::BTreeSet;

pub const MAX_DECLARATIONS: usize = 4096;
pub const MAX_PARAMETERS: usize = 256;
pub const MAX_POINTER_DEPTH: usize = 32;
pub const MAX_SYMBOL_BYTES: usize = 255;
pub const DESCRIPTOR_MAGIC: &[u8; 8] = b"MCABI001";

/// Metadata distinctions are retained even when a target uses equal widths.
/// In particular `Isize`, `Usize`, `Bool` and `C` require a target ABI decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CAbiType {
    Void,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Isize,
    Usize,
    F32,
    F64,
    Bool,
    Pointer {
        mutable: bool,
        pointee: Box<CAbiType>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CDeclaration {
    pub name: String,
    pub callconv: CallConv,
    /// Preserved source annotation, not an enforced safety proof.
    pub is_unsafe: bool,
    pub is_varargs: bool,
    pub parameters: Vec<CAbiType>,
    pub result: CAbiType,
}

/// Resolved names come from the same source capture's module resolver.
/// `None` explicitly means no import resolution is available. An import in
/// the AST then refuses. `Some` must contain ALL resolved imported bindings;
/// this module cannot certify a caller-supplied resolver's completeness.
pub struct SymbolContext<'a> {
    pub imported: Option<&'a BTreeSet<String>>,
    /// Additional compiler/std symbols reserved by the selected capability.
    /// The complete `__mind_` namespace is always reserved independently.
    pub reserved: &'a BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CAbiRejection {
    pub kind: &'static str,
    pub symbol: Option<String>,
}
type Result<T> = std::result::Result<T, CAbiRejection>;

fn reject<T>(kind: &'static str, symbol: Option<&str>) -> Result<T> {
    Err(CAbiRejection {
        kind,
        symbol: symbol.map(str::to_owned),
    })
}

/// Validated inventory in captured source declaration order. The private
/// vector prevents callers from mutating the expected descriptor after checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCAbi {
    declarations: Vec<CDeclaration>,
}

fn symbol_valid(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_SYMBOL_BYTES
        && name
            .as_bytes()
            .first()
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
}

fn source_type(
    ty: &TypeAnn,
    depth: usize,
    symbol: &str,
    shadowed_types: &BTreeSet<String>,
) -> Result<CAbiType> {
    use CAbiType as C;
    // Some scalar spellings use Named in the AST. A user alias called `u64`
    // must not impersonate that primitive, including behind a raw pointer.
    let spelling = match ty {
        TypeAnn::ScalarI32 => Some("i32"),
        TypeAnn::ScalarU32 => Some("u32"),
        TypeAnn::ScalarI64 => Some("i64"),
        TypeAnn::ScalarF32 => Some("f32"),
        TypeAnn::ScalarF64 => Some("f64"),
        TypeAnn::ScalarBool => Some("bool"),
        TypeAnn::Named(name) => Some(name.as_str()),
        _ => None,
    };
    if spelling.is_some_and(|name| shadowed_types.contains(name)) {
        return reject("cabi.unsupported_type", Some(symbol));
    }
    Ok(match ty {
        TypeAnn::ScalarI32 => C::I32,
        TypeAnn::ScalarU32 => C::U32,
        TypeAnn::ScalarI64 => C::I64,
        TypeAnn::ScalarF32 => C::F32,
        TypeAnn::ScalarF64 => C::F64,
        TypeAnn::ScalarBool => C::Bool,
        TypeAnn::Named(name) => match name.as_str() {
            "i8" => C::I8,
            "u8" => C::U8,
            "i16" => C::I16,
            "u16" => C::U16,
            "u64" => C::U64,
            "isize" => C::Isize,
            "usize" => C::Usize,
            // Do not resolve aliases by spelling or silently erase opaque
            // aggregates, including `void` and pointers to named `void`.
            _ => return reject("cabi.unsupported_type", Some(symbol)),
        },
        TypeAnn::RawPtr { mutable, pointee } => {
            if depth >= MAX_POINTER_DEPTH {
                return reject("cabi.pointer_depth", Some(symbol));
            }
            C::Pointer {
                mutable: *mutable,
                pointee: Box::new(source_type(pointee, depth + 1, symbol, shadowed_types)?),
            }
        }
        TypeAnn::FnPtr { .. } => return reject("cabi.callback_unsupported", Some(symbol)),
        TypeAnn::Tensor { .. }
        | TypeAnn::DiffTensor { .. }
        | TypeAnn::Slice { .. }
        | TypeAnn::Array { .. }
        | TypeAnn::Ref { .. }
        | TypeAnn::Generic { .. }
        | TypeAnn::Tuple { .. }
        | TypeAnn::SparseTensor { .. } => {
            return reject("cabi.unsupported_type", Some(symbol));
        }
    })
}

impl NativeCAbi {
    pub fn from_ast(ast: &Module, context: &SymbolContext<'_>) -> Result<Self> {
        let mut local = BTreeSet::new();
        // Resolved imported names have no type/value distinction here. Refuse
        // ambiguous scalar spellings conservatively instead of guessing.
        let mut shadowed_types = context.imported.cloned().unwrap_or_default();
        let mut import_aliases = BTreeSet::new();
        let mut scan_error = None;
        for item in &ast.items {
            let top_extern = matches!(item, Node::ExternBlock { .. });
            crate::type_checker::nerve_walk::walk(item, &mut |node| {
                if scan_error.is_some() {
                    return;
                }
                let name = match node {
                    Node::FnDef(f, _) => Some(f.name.as_str()),
                    Node::Const { name, .. } => Some(name.as_str()),
                    Node::TypeAlias { name, .. }
                    | Node::StructDef { name, .. }
                    | Node::EnumDef { name, .. } => {
                        shadowed_types.insert(name.clone());
                        Some(name.as_str())
                    }
                    Node::ExternConst { name, .. } => {
                        scan_error = Some(CAbiRejection {
                            kind: "cabi.data_import_unsupported",
                            symbol: Some(name.clone()),
                        });
                        None
                    }
                    Node::ExternBlock { .. } if !top_extern => {
                        scan_error = Some(CAbiRejection {
                            kind: "cabi.nested_extern",
                            symbol: None,
                        });
                        None
                    }
                    Node::Import { path, alias, .. } => {
                        if context.imported.is_none() {
                            scan_error = Some(CAbiRejection {
                                kind: "cabi.unresolved_imports",
                                symbol: None,
                            });
                        }
                        if let Some(name) = alias.as_ref().or_else(|| path.last()) {
                            import_aliases.insert(name.clone());
                            shadowed_types.insert(name.clone());
                        }
                        None
                    }
                    _ => None,
                };
                if let Some(name) = name {
                    local.insert(name.to_owned());
                }
            });
        }
        if let Some(error) = scan_error {
            return Err(error);
        }

        let mut declarations = Vec::new();
        let mut seen = BTreeSet::new();
        for item in &ast.items {
            let Node::ExternBlock { callconv, fns, .. } = item else {
                continue;
            };
            for f in fns {
                let name = f.name.as_str();
                if !symbol_valid(name) {
                    return reject("cabi.symbol", Some(name));
                }
                if !seen.insert(name) {
                    return reject("cabi.duplicate_declaration", Some(name));
                }
                if name.starts_with("__mind_") || context.reserved.contains(name) {
                    return reject("cabi.reserved_symbol", Some(name));
                }
                if local.contains(name) {
                    return reject("cabi.local_conflict", Some(name));
                }
                if import_aliases.contains(name)
                    || context.imported.is_some_and(|names| names.contains(name))
                {
                    return reject("cabi.import_conflict", Some(name));
                }
                if declarations.len() >= MAX_DECLARATIONS {
                    return reject("cabi.declaration_limit", Some(name));
                }
                if f.params.len() > MAX_PARAMETERS {
                    return reject("cabi.parameter_limit", Some(name));
                }
                let mut parameter_names = BTreeSet::new();
                let mut parameters = Vec::new();
                for p in &f.params {
                    if !parameter_names.insert(&p.name) {
                        return reject("cabi.duplicate_parameter", Some(name));
                    }
                    parameters.push(source_type(&p.ty, 0, name, &shadowed_types)?);
                }
                declarations.push(CDeclaration {
                    name: f.name.clone(),
                    callconv: *callconv,
                    is_unsafe: f.is_unsafe,
                    is_varargs: f.is_varargs,
                    parameters,
                    result: f
                        .ret_type
                        .as_ref()
                        .map(|ty| source_type(ty, 0, name, &shadowed_types))
                        .transpose()?
                        .unwrap_or(CAbiType::Void),
                });
            }
        }
        Ok(Self { declarations })
    }

    pub fn declarations(&self) -> &[CDeclaration] {
        &self.declarations
    }

    /// A prerequisite contract check for a fixed-signature SysV AMD64 slice.
    /// Does not implement argument classification, stack alignment, extensions,
    /// linkage or memory ownership. Variadic promotions need a separate gate.
    pub fn require_fixed_sysv_amd64(&self) -> Result<()> {
        for d in &self.declarations {
            if !matches!(d.callconv, CallConv::C | CallConv::SysV) {
                return reject("cabi.callconv_unsupported", Some(&d.name));
            }
            if d.is_varargs {
                return reject("cabi.varargs_unsupported", Some(&d.name));
            }
        }
        Ok(())
    }

    /// Canonical MCABI001 bytes, distinct from historical MIC extern encoding.
    /// Source order is significant; spans/parameter names are not ABI fields.
    pub fn descriptor_bytes(&self) -> Vec<u8> {
        let mut out = DESCRIPTOR_MAGIC.to_vec();
        out.extend_from_slice(&(self.declarations.len() as u32).to_le_bytes());
        for d in &self.declarations {
            out.extend_from_slice(&(d.name.len() as u32).to_le_bytes());
            out.extend_from_slice(d.name.as_bytes());
            out.push(match d.callconv {
                CallConv::C => 0,
                CallConv::SysV => 1,
                CallConv::Win64 => 2,
                CallConv::Aapcs => 3,
            });
            out.push(u8::from(d.is_unsafe));
            out.push(u8::from(d.is_varargs));
            out.extend_from_slice(&(d.parameters.len() as u32).to_le_bytes());
            emit_type(&d.result, &mut out);
            for ty in &d.parameters {
                emit_type(ty, &mut out);
            }
        }
        out
    }

    /// Hash of declaration metadata only, not a full source/artifact identity.
    pub fn descriptor_hash(&self) -> [u8; 32] {
        crate::deps::mini_sha256(&self.descriptor_bytes())
    }

    /// Compare independently produced native metadata to this AST inventory.
    /// Unknown versions/tags, truncation, trailing bytes and changed ordering
    /// cannot match. A successful comparison does not authorize execution.
    pub fn check_native_descriptor(&self, bytes: &[u8]) -> Result<()> {
        if bytes == self.descriptor_bytes() {
            Ok(())
        } else {
            reject("cabi.native_descriptor_mismatch", None)
        }
    }
}

fn emit_type(ty: &CAbiType, out: &mut Vec<u8>) {
    use CAbiType as C;
    let tag = match ty {
        C::Void => 0,
        C::I8 => 1,
        C::U8 => 2,
        C::I16 => 3,
        C::U16 => 4,
        C::I32 => 5,
        C::U32 => 6,
        C::I64 => 7,
        C::U64 => 8,
        C::Isize => 9,
        C::Usize => 10,
        C::F32 => 11,
        C::F64 => 12,
        C::Bool => 13,
        C::Pointer { mutable, pointee } => {
            out.push(if *mutable { 33 } else { 32 });
            emit_type(pointee, out);
            return;
        }
    };
    out.push(tag);
}

#[cfg(test)]
mod tests;
