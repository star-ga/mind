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

// Part of the MIND project (Machine Intelligence Native Design).

//! MIC@3 binary emitter — serializes [`IRModule`] to deterministic bytes.

use std::collections::HashMap;
use std::io::Write;

use crate::ir::{BinOp, IRModule, Instr, ValueId};
use crate::types::{ConvPadding, DType, ShapeDim};

use super::{MIC3_MAGIC, MIC3_VERSION};
use crate::ir::compact::v2::{uleb128_write, zigzag_encode};

// ─── Opcode constants ────────────────────────────────────────────────────────

pub(super) const OP_CONST_I64: u8 = 0x01;
pub(super) const OP_CONST_F64: u8 = 0x02;
pub(super) const OP_CONST_TENSOR: u8 = 0x03;
pub(super) const OP_BINOP: u8 = 0x04;
pub(super) const OP_SUM: u8 = 0x05;
pub(super) const OP_MEAN: u8 = 0x06;
pub(super) const OP_RESHAPE: u8 = 0x07;
pub(super) const OP_EXPAND_DIMS: u8 = 0x08;
pub(super) const OP_SQUEEZE: u8 = 0x09;
pub(super) const OP_TRANSPOSE: u8 = 0x0A;
pub(super) const OP_DOT: u8 = 0x0B;
pub(super) const OP_MATMUL: u8 = 0x0C;
pub(super) const OP_CONV2D: u8 = 0x0D;
pub(super) const OP_CONV2D_GRAD_INPUT: u8 = 0x0E;
pub(super) const OP_CONV2D_GRAD_FILTER: u8 = 0x0F;
pub(super) const OP_INDEX: u8 = 0x10;
pub(super) const OP_SLICE: u8 = 0x11;
pub(super) const OP_GATHER: u8 = 0x12;
pub(super) const OP_OUTPUT: u8 = 0x13;
pub(super) const OP_SPARSE_ATTR: u8 = 0x14;
pub(super) const OP_FN_DEF: u8 = 0x15;
pub(super) const OP_CALL: u8 = 0x16;
pub(super) const OP_RETURN: u8 = 0x17;
pub(super) const OP_PARAM: u8 = 0x18;
pub(super) const OP_CONST_ARRAY: u8 = 0x19;
pub(super) const OP_ARRAY_LOAD: u8 = 0x1A;
pub(super) const OP_WHILE: u8 = 0x1B;
pub(super) const OP_IF: u8 = 0x1C;
pub(super) const OP_VEC_LOAD: u8 = 0x1D;
pub(super) const OP_VEC_FMA: u8 = 0x1E;
pub(super) const OP_VEC_REDUCE_ADD: u8 = 0x1F;
pub(super) const OP_VEC_STORE: u8 = 0x20;
pub(super) const OP_VEC_LOAD_I32: u8 = 0x21;
pub(super) const OP_VEC_MUL_ADD_Q16: u8 = 0x22;
pub(super) const OP_VEC_REDUCE_ADD_I64: u8 = 0x23;
pub(super) const OP_REGION: u8 = 0x24;
pub(super) const OP_EXTERN_FN_DECL: u8 = 0x25;
/// Elementwise ReLU (`{dst, src}`). Appended (never inserted) so existing
/// mic@3 byte streams — and their `trace_hash` — are unchanged.
pub(super) const OP_RELU: u8 = 0x26;
/// Backward ReLU (`{dst, grad, src}`): `dx = grad * step(src)`. Appended after
/// `OP_RELU` so existing mic@3 byte streams and their `trace_hash` are unchanged.
pub(super) const OP_RELU_GRAD: u8 = 0x27;
/// Loop `break` marker (zero operands). Appended in previously-unused op-space
/// so existing mic@3 byte streams and their `trace_hash` are unchanged
/// (backward-compatible additive extension — no MIC3_VERSION bump).
pub(super) const OP_BREAK: u8 = 0x28;
/// Loop `continue` marker (zero operands). Appended after `OP_BREAK`.
pub(super) const OP_CONTINUE: u8 = 0x29;

// ─── DType byte tags ─────────────────────────────────────────────────────────

pub(super) fn dtype_to_byte(d: &DType) -> u8 {
    match d {
        DType::I32 => 0x00,
        DType::I64 => 0x01,
        DType::F32 => 0x02,
        DType::F64 => 0x03,
        DType::BF16 => 0x04,
        DType::F16 => 0x05,
        DType::Q16 => 0x06,
    }
}

pub(super) fn byte_to_dtype(b: u8) -> Option<DType> {
    match b {
        0x00 => Some(DType::I32),
        0x01 => Some(DType::I64),
        0x02 => Some(DType::F32),
        0x03 => Some(DType::F64),
        0x04 => Some(DType::BF16),
        0x05 => Some(DType::F16),
        0x06 => Some(DType::Q16),
        _ => None,
    }
}

// ─── ConvPadding byte tags ────────────────────────────────────────────────────

pub(super) fn padding_to_byte(p: ConvPadding) -> u8 {
    match p {
        ConvPadding::Valid => 0x00,
        ConvPadding::Same => 0x01,
    }
}

pub(super) fn byte_to_padding(b: u8) -> Option<ConvPadding> {
    match b {
        0x00 => Some(ConvPadding::Valid),
        0x01 => Some(ConvPadding::Same),
        _ => None,
    }
}

// ─── BinOp byte tags ─────────────────────────────────────────────────────────

pub(super) fn binop_to_byte(op: BinOp) -> u8 {
    match op {
        BinOp::Add => 0x00,
        BinOp::Sub => 0x01,
        BinOp::Mul => 0x02,
        BinOp::Div => 0x03,
        BinOp::Mod => 0x04,
        BinOp::Lt => 0x05,
        BinOp::Le => 0x06,
        BinOp::Gt => 0x07,
        BinOp::Ge => 0x08,
        BinOp::Eq => 0x09,
        BinOp::Ne => 0x0A,
        #[cfg(feature = "std-surface")]
        BinOp::BitAnd => 0x0B,
        #[cfg(feature = "std-surface")]
        BinOp::BitOr => 0x0C,
        #[cfg(feature = "std-surface")]
        BinOp::BitXor => 0x0D,
        #[cfg(feature = "std-surface")]
        BinOp::Shl => 0x0E,
        #[cfg(feature = "std-surface")]
        BinOp::Shr => 0x0F,
    }
}

pub(super) fn byte_to_binop(b: u8) -> Option<BinOp> {
    match b {
        0x00 => Some(BinOp::Add),
        0x01 => Some(BinOp::Sub),
        0x02 => Some(BinOp::Mul),
        0x03 => Some(BinOp::Div),
        0x04 => Some(BinOp::Mod),
        0x05 => Some(BinOp::Lt),
        0x06 => Some(BinOp::Le),
        0x07 => Some(BinOp::Gt),
        0x08 => Some(BinOp::Ge),
        0x09 => Some(BinOp::Eq),
        0x0A => Some(BinOp::Ne),
        #[cfg(feature = "std-surface")]
        0x0B => Some(BinOp::BitAnd),
        #[cfg(feature = "std-surface")]
        0x0C => Some(BinOp::BitOr),
        #[cfg(feature = "std-surface")]
        0x0D => Some(BinOp::BitXor),
        #[cfg(feature = "std-surface")]
        0x0E => Some(BinOp::Shl),
        #[cfg(feature = "std-surface")]
        0x0F => Some(BinOp::Shr),
        _ => None,
    }
}

// ─── SparseLayout byte tags ──────────────────────────────────────────────────

pub(super) fn sparse_layout_to_byte(l: crate::ast::SparseLayout) -> u8 {
    match l {
        crate::ast::SparseLayout::Csr => 0x00,
        crate::ast::SparseLayout::Csc => 0x01,
        crate::ast::SparseLayout::Coo => 0x02,
        crate::ast::SparseLayout::Bsr => 0x03,
    }
}

pub(super) fn byte_to_sparse_layout(b: u8) -> Option<crate::ast::SparseLayout> {
    match b {
        0x00 => Some(crate::ast::SparseLayout::Csr),
        0x01 => Some(crate::ast::SparseLayout::Csc),
        0x02 => Some(crate::ast::SparseLayout::Coo),
        0x03 => Some(crate::ast::SparseLayout::Bsr),
        _ => None,
    }
}

// ─── CallConv byte tags (std-surface) ────────────────────────────────────────

#[cfg(feature = "std-surface")]
pub(super) fn callconv_to_byte(c: crate::ast::CallConv) -> u8 {
    match c {
        crate::ast::CallConv::C => 0x00,
        crate::ast::CallConv::SysV => 0x01,
        crate::ast::CallConv::Win64 => 0x02,
        crate::ast::CallConv::Aapcs => 0x03,
    }
}

#[cfg(feature = "std-surface")]
pub(super) fn byte_to_callconv(b: u8) -> Option<crate::ast::CallConv> {
    match b {
        0x00 => Some(crate::ast::CallConv::C),
        0x01 => Some(crate::ast::CallConv::SysV),
        0x02 => Some(crate::ast::CallConv::Win64),
        0x03 => Some(crate::ast::CallConv::Aapcs),
        _ => None,
    }
}

// ─── TypeAnn serialization (std-surface) ─────────────────────────────────────

#[cfg(feature = "std-surface")]
pub(super) fn encode_type_ann<W: Write>(
    w: &mut W,
    ann: &crate::ast::TypeAnn,
    st: &StringTable,
) -> std::io::Result<()> {
    use crate::ast::TypeAnn;
    let tag: u8 = match ann {
        TypeAnn::ScalarI32 => 0x01,
        TypeAnn::ScalarI64 => 0x02,
        TypeAnn::ScalarF32 => 0x03,
        TypeAnn::ScalarF64 => 0x04,
        TypeAnn::ScalarBool => 0x05,
        TypeAnn::ScalarU32 => 0x06,
        TypeAnn::Tensor { .. } => 0x07,
        TypeAnn::DiffTensor { .. } => 0x08,
        TypeAnn::Named(_) => 0x09,
        TypeAnn::Slice { .. } => 0x0A,
        TypeAnn::Array { .. } => 0x0B,
        TypeAnn::Ref { .. } => 0x0C,
        TypeAnn::Generic { .. } => 0x0D,
        TypeAnn::Tuple { .. } => 0x0E,
        TypeAnn::SparseTensor { .. } => 0x0F,
        TypeAnn::RawPtr { .. } => 0x10,
        TypeAnn::FnPtr { .. } => 0x11,
    };
    w.write_all(&[tag])?;
    match ann {
        TypeAnn::ScalarI32
        | TypeAnn::ScalarI64
        | TypeAnn::ScalarF32
        | TypeAnn::ScalarF64
        | TypeAnn::ScalarBool
        | TypeAnn::ScalarU32 => {}
        TypeAnn::Tensor { dtype, dims } | TypeAnn::DiffTensor { dtype, dims } => {
            uleb128_write(w, st.get(dtype) as u64)?;
            uleb128_write(w, dims.len() as u64)?;
            for d in dims {
                uleb128_write(w, st.get(d) as u64)?;
            }
        }
        TypeAnn::Named(name) => {
            uleb128_write(w, st.get(name) as u64)?;
        }
        TypeAnn::Slice { mutable, element } => {
            w.write_all(&[*mutable as u8])?;
            encode_type_ann(w, element, st)?;
        }
        TypeAnn::Array { element, length } => {
            uleb128_write(w, *length as u64)?;
            encode_type_ann(w, element, st)?;
        }
        TypeAnn::Ref { mutable, target } => {
            w.write_all(&[*mutable as u8])?;
            encode_type_ann(w, target, st)?;
        }
        TypeAnn::Generic { name, args } => {
            uleb128_write(w, st.get(name) as u64)?;
            uleb128_write(w, args.len() as u64)?;
            for a in args {
                encode_type_ann(w, a, st)?;
            }
        }
        TypeAnn::Tuple { elements } => {
            uleb128_write(w, elements.len() as u64)?;
            for e in elements {
                encode_type_ann(w, e, st)?;
            }
        }
        TypeAnn::SparseTensor {
            layout,
            element,
            shape,
        } => {
            w.write_all(&[sparse_layout_to_byte(*layout)])?;
            encode_type_ann(w, element, st)?;
            uleb128_write(w, shape.len() as u64)?;
            for dim in shape {
                encode_shape_dim(w, dim, st)?;
            }
        }
        TypeAnn::RawPtr { mutable, pointee } => {
            w.write_all(&[*mutable as u8])?;
            encode_type_ann(w, pointee, st)?;
        }
        TypeAnn::FnPtr { params, ret } => {
            uleb128_write(w, params.len() as u64)?;
            for p in params {
                encode_type_ann(w, p, st)?;
            }
            if let Some(r) = ret {
                w.write_all(&[1u8])?;
                encode_type_ann(w, r, st)?;
            } else {
                w.write_all(&[0u8])?;
            }
        }
    }
    Ok(())
}

// ─── String table ─────────────────────────────────────────────────────────────

/// String table interned in first-seen order (deterministic across runs since
/// we traverse `module.instrs` sequentially).
pub(super) struct StringTable {
    entries: Vec<String>,
    index: HashMap<String, usize>,
}

impl StringTable {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Intern a string and return its index (idempotent).
    pub fn intern(&mut self, s: &str) -> usize {
        if let Some(&idx) = self.index.get(s) {
            return idx;
        }
        let idx = self.entries.len();
        self.entries.push(s.to_string());
        self.index.insert(s.to_string(), idx);
        idx
    }

    /// Look up an already-interned string.  Panics in debug builds if missing;
    /// returns 0 in release builds (should never occur after a complete pass).
    pub fn get(&self, s: &str) -> usize {
        debug_assert!(
            self.index.contains_key(s),
            "string '{}' not interned before emit",
            s
        );
        *self.index.get(s).unwrap_or(&0)
    }

    pub fn entries(&self) -> &[String] {
        &self.entries
    }
}

// ─── Low-level helpers ────────────────────────────────────────────────────────

fn write_u64_le<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_vid<W: Write>(w: &mut W, id: ValueId) -> std::io::Result<()> {
    uleb128_write(w, id.0 as u64)?;
    Ok(())
}

fn write_i64<W: Write>(w: &mut W, v: i64) -> std::io::Result<()> {
    uleb128_write(w, zigzag_encode(v))?;
    Ok(())
}

fn write_bool<W: Write>(w: &mut W, b: bool) -> std::io::Result<()> {
    w.write_all(&[b as u8])
}

fn encode_shape_dim<W: Write>(w: &mut W, dim: &ShapeDim, st: &StringTable) -> std::io::Result<()> {
    match dim {
        ShapeDim::Known(n) => {
            w.write_all(&[0u8])?; // tag: known
            uleb128_write(w, *n as u64)?;
        }
        ShapeDim::Sym(s) => {
            w.write_all(&[1u8])?; // tag: symbolic
            uleb128_write(w, st.get(s) as u64)?;
        }
    }
    Ok(())
}

fn encode_i64_vec<W: Write>(w: &mut W, v: &[i64]) -> std::io::Result<()> {
    uleb128_write(w, v.len() as u64)?;
    for &x in v {
        write_i64(w, x)?;
    }
    Ok(())
}

fn encode_vid_vec<W: Write>(w: &mut W, ids: &[ValueId]) -> std::io::Result<()> {
    uleb128_write(w, ids.len() as u64)?;
    for id in ids {
        write_vid(w, *id)?;
    }
    Ok(())
}

fn encode_string_idx<W: Write>(w: &mut W, s: &str, st: &StringTable) -> std::io::Result<()> {
    uleb128_write(w, st.get(s) as u64)?;
    Ok(())
}

fn encode_opt_vid<W: Write>(w: &mut W, id: Option<ValueId>) -> std::io::Result<()> {
    match id {
        None => w.write_all(&[0u8]),
        Some(v) => {
            w.write_all(&[1u8])?;
            write_vid(w, v)
        }
    }
}

fn encode_opt_f64<W: Write>(w: &mut W, v: Option<f64>) -> std::io::Result<()> {
    match v {
        None => w.write_all(&[0u8]),
        Some(f) => {
            w.write_all(&[1u8])?;
            write_u64_le(w, f.to_bits())
        }
    }
}

#[cfg(feature = "std-surface")]
fn encode_opt_string<W: Write>(
    w: &mut W,
    s: &Option<String>,
    st: &StringTable,
) -> std::io::Result<()> {
    match s {
        None => w.write_all(&[0u8]),
        Some(name) => {
            w.write_all(&[1u8])?;
            encode_string_idx(w, name, st)
        }
    }
}

fn encode_named_vids<W: Write>(
    w: &mut W,
    pairs: &[(String, ValueId)],
    st: &StringTable,
) -> std::io::Result<()> {
    uleb128_write(w, pairs.len() as u64)?;
    for (name, id) in pairs {
        encode_string_idx(w, name, st)?;
        write_vid(w, *id)?;
    }
    Ok(())
}

// ─── String collection (pre-pass) ─────────────────────────────────────────────

/// Walk all instructions recursively and intern every string that appears,
/// in traversal order (preserves determinism).
fn collect_strings(instrs: &[Instr], st: &mut StringTable) {
    for instr in instrs {
        collect_instr_strings(instr, st);
    }
}

fn collect_instr_strings(instr: &Instr, st: &mut StringTable) {
    match instr {
        Instr::ConstTensor(_, _, shape, _) => {
            for dim in shape {
                if let ShapeDim::Sym(s) = dim {
                    st.intern(s);
                }
            }
        }
        Instr::Reshape { new_shape, .. } => {
            for dim in new_shape {
                if let ShapeDim::Sym(s) = dim {
                    st.intern(s);
                }
            }
        }
        Instr::FnDef {
            name, params, body, ..
        } => {
            st.intern(name);
            for (pname, _) in params {
                st.intern(pname);
            }
            collect_strings(body, st);
        }
        Instr::Call { name, .. } => {
            st.intern(name);
        }
        Instr::Param { name, .. } => {
            st.intern(name);
        }
        #[cfg(feature = "std-surface")]
        Instr::ConstArray { name: Some(n), .. } => {
            st.intern(n);
        }
        #[cfg(feature = "std-surface")]
        Instr::While {
            cond_instrs,
            body,
            live_vars,
            ..
        } => {
            collect_strings(cond_instrs, st);
            collect_strings(body, st);
            for (name, _) in live_vars {
                st.intern(name);
            }
        }
        #[cfg(feature = "std-surface")]
        Instr::If {
            cond_instrs,
            then_instrs,
            else_instrs,
            branch_bindings,
            ..
        } => {
            collect_strings(cond_instrs, st);
            collect_strings(then_instrs, st);
            collect_strings(else_instrs, st);
            for (name, _) in branch_bindings {
                st.intern(name);
            }
        }
        #[cfg(feature = "std-surface")]
        Instr::Region { body, .. } => {
            collect_strings(body, st);
        }
        #[cfg(feature = "std-surface")]
        Instr::ExternFnDecl {
            name,
            param_types,
            ret_type,
            vararg_hints,
            ..
        } => {
            st.intern(name);
            for t in param_types {
                st.intern(t);
            }
            if let Some(r) = ret_type {
                st.intern(r);
            }
            for h in vararg_hints {
                st.intern(h);
            }
        }
        _ => {}
    }
}

/// Collect strings needed for exports.
fn collect_export_strings(exports: &std::collections::HashSet<String>, st: &mut StringTable) {
    // Sort for determinism before interning so indices reflect sorted order.
    let mut sorted: Vec<&String> = exports.iter().collect();
    sorted.sort();
    for s in sorted {
        st.intern(s);
    }
}

/// Collect strings needed for struct_defs (std-surface).
#[cfg(feature = "std-surface")]
fn collect_struct_def_strings(
    struct_defs: &std::collections::BTreeMap<String, Vec<String>>,
    st: &mut StringTable,
) {
    for (name, fields) in struct_defs {
        st.intern(name);
        for f in fields {
            st.intern(f);
        }
    }
}

/// Collect strings needed for const_array_defs (std-surface).
#[cfg(feature = "std-surface")]
fn collect_const_array_def_strings(
    defs: &std::collections::BTreeMap<String, Vec<i64>>,
    st: &mut StringTable,
) {
    for name in defs.keys() {
        st.intern(name);
    }
}

/// Collect strings needed for repr_c_structs (std-surface).
#[cfg(feature = "std-surface")]
fn collect_repr_c_strings(
    structs: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
    st: &mut StringTable,
) {
    for (name, fields) in structs {
        st.intern(name);
        for f in fields {
            collect_type_ann_strings(f, st);
        }
    }
}

#[cfg(feature = "std-surface")]
fn collect_type_ann_strings(ann: &crate::ast::TypeAnn, st: &mut StringTable) {
    use crate::ast::TypeAnn;
    match ann {
        TypeAnn::Tensor { dtype, dims } | TypeAnn::DiffTensor { dtype, dims } => {
            st.intern(dtype);
            for d in dims {
                st.intern(d);
            }
        }
        TypeAnn::Named(name) => {
            st.intern(name);
        }
        TypeAnn::Generic { name, args } => {
            st.intern(name);
            for a in args {
                collect_type_ann_strings(a, st);
            }
        }
        TypeAnn::Slice { element, .. }
        | TypeAnn::Array { element, .. }
        | TypeAnn::Ref {
            target: element, ..
        }
        | TypeAnn::RawPtr {
            pointee: element, ..
        } => {
            collect_type_ann_strings(element, st);
        }
        TypeAnn::Tuple { elements } => {
            for e in elements {
                collect_type_ann_strings(e, st);
            }
        }
        TypeAnn::SparseTensor { element, shape, .. } => {
            collect_type_ann_strings(element, st);
            for dim in shape {
                if let ShapeDim::Sym(s) = dim {
                    st.intern(s);
                }
            }
        }
        TypeAnn::FnPtr { params, ret } => {
            for p in params {
                collect_type_ann_strings(p, st);
            }
            if let Some(r) = ret {
                collect_type_ann_strings(r, st);
            }
        }
        _ => {}
    }
}

// ─── Main emitter ─────────────────────────────────────────────────────────────

/// Emit an [`IRModule`] as MIC@3 binary bytes.
///
/// Output is deterministic: identical `IRModule` content always produces
/// byte-identical output regardless of run order or HashSet iteration order.
pub fn emit_mic3(module: &IRModule) -> Vec<u8> {
    let mut st = StringTable::new();

    // Pre-pass: collect all strings in traversal order (deterministic).
    collect_strings(&module.instrs, &mut st);
    collect_export_strings(&module.exports, &mut st);
    #[cfg(feature = "std-surface")]
    {
        collect_struct_def_strings(&module.struct_defs, &mut st);
        collect_const_array_def_strings(&module.const_array_defs, &mut st);
        collect_repr_c_strings(&module.repr_c_structs, &mut st);
    }

    let mut out = Vec::with_capacity(256);

    // Header
    out.write_all(&MIC3_MAGIC).unwrap();
    out.write_all(&[MIC3_VERSION]).unwrap();

    // String table
    let entries = st.entries().to_vec();
    uleb128_write(&mut out, entries.len() as u64).unwrap();
    for s in &entries {
        let b = s.as_bytes();
        uleb128_write(&mut out, b.len() as u64).unwrap();
        out.write_all(b).unwrap();
    }

    // next_id
    uleb128_write(&mut out, module.next_id as u64).unwrap();

    // Exports — serialised SORTED for determinism (HashSet iteration is unordered).
    let mut sorted_exports: Vec<&String> = module.exports.iter().collect();
    sorted_exports.sort();
    uleb128_write(&mut out, sorted_exports.len() as u64).unwrap();
    for e in &sorted_exports {
        uleb128_write(&mut out, st.get(e) as u64).unwrap();
    }

    // Instructions
    uleb128_write(&mut out, module.instrs.len() as u64).unwrap();
    for instr in &module.instrs {
        emit_instr(&mut out, instr, &st);
    }

    // std-surface registries
    #[cfg(feature = "std-surface")]
    {
        // struct_defs (BTreeMap — already sorted by key)
        uleb128_write(&mut out, module.struct_defs.len() as u64).unwrap();
        for (name, fields) in &module.struct_defs {
            uleb128_write(&mut out, st.get(name) as u64).unwrap();
            uleb128_write(&mut out, fields.len() as u64).unwrap();
            for f in fields {
                uleb128_write(&mut out, st.get(f) as u64).unwrap();
            }
        }

        // const_array_defs (BTreeMap — already sorted by key)
        uleb128_write(&mut out, module.const_array_defs.len() as u64).unwrap();
        for (name, vals) in &module.const_array_defs {
            uleb128_write(&mut out, st.get(name) as u64).unwrap();
            uleb128_write(&mut out, vals.len() as u64).unwrap();
            for &v in vals {
                uleb128_write(&mut out, zigzag_encode(v)).unwrap();
            }
        }

        // repr_c_structs (BTreeMap — already sorted by key)
        uleb128_write(&mut out, module.repr_c_structs.len() as u64).unwrap();
        for (name, fields) in &module.repr_c_structs {
            uleb128_write(&mut out, st.get(name) as u64).unwrap();
            uleb128_write(&mut out, fields.len() as u64).unwrap();
            for f in fields {
                encode_type_ann(&mut out, f, &st).unwrap();
            }
        }
    }

    out
}

// ─── Instruction emitter ──────────────────────────────────────────────────────

fn emit_instr<W: Write>(w: &mut W, instr: &Instr, st: &StringTable) {
    match instr {
        Instr::ConstI64(dst, v) => {
            w.write_all(&[OP_CONST_I64]).unwrap();
            write_vid(w, *dst).unwrap();
            write_i64(w, *v).unwrap();
        }
        Instr::ConstF64(dst, v) => {
            w.write_all(&[OP_CONST_F64]).unwrap();
            write_vid(w, *dst).unwrap();
            write_u64_le(w, v.to_bits()).unwrap();
        }
        Instr::ConstTensor(dst, dtype, shape, fill) => {
            w.write_all(&[OP_CONST_TENSOR]).unwrap();
            write_vid(w, *dst).unwrap();
            w.write_all(&[dtype_to_byte(dtype)]).unwrap();
            uleb128_write(w, shape.len() as u64).unwrap();
            for dim in shape {
                encode_shape_dim(w, dim, st).unwrap();
            }
            encode_opt_f64(w, *fill).unwrap();
        }
        Instr::BinOp { dst, op, lhs, rhs } => {
            w.write_all(&[OP_BINOP]).unwrap();
            write_vid(w, *dst).unwrap();
            w.write_all(&[binop_to_byte(*op)]).unwrap();
            write_vid(w, *lhs).unwrap();
            write_vid(w, *rhs).unwrap();
        }
        Instr::Sum {
            dst,
            src,
            axes,
            keepdims,
        } => {
            w.write_all(&[OP_SUM]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            encode_i64_vec(w, axes).unwrap();
            write_bool(w, *keepdims).unwrap();
        }
        Instr::Mean {
            dst,
            src,
            axes,
            keepdims,
        } => {
            w.write_all(&[OP_MEAN]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            encode_i64_vec(w, axes).unwrap();
            write_bool(w, *keepdims).unwrap();
        }
        Instr::Relu { dst, src } => {
            w.write_all(&[OP_RELU]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
        }
        Instr::ReluGrad { dst, grad, src } => {
            w.write_all(&[OP_RELU_GRAD]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *grad).unwrap();
            write_vid(w, *src).unwrap();
        }
        Instr::Reshape {
            dst,
            src,
            new_shape,
        } => {
            w.write_all(&[OP_RESHAPE]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            uleb128_write(w, new_shape.len() as u64).unwrap();
            for dim in new_shape {
                encode_shape_dim(w, dim, st).unwrap();
            }
        }
        Instr::ExpandDims { dst, src, axis } => {
            w.write_all(&[OP_EXPAND_DIMS]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            write_i64(w, *axis).unwrap();
        }
        Instr::Squeeze { dst, src, axes } => {
            w.write_all(&[OP_SQUEEZE]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            encode_i64_vec(w, axes).unwrap();
        }
        Instr::Transpose { dst, src, perm } => {
            w.write_all(&[OP_TRANSPOSE]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            encode_i64_vec(w, perm).unwrap();
        }
        Instr::Dot { dst, a, b } => {
            w.write_all(&[OP_DOT]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *a).unwrap();
            write_vid(w, *b).unwrap();
        }
        Instr::MatMul { dst, a, b } => {
            w.write_all(&[OP_MATMUL]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *a).unwrap();
            write_vid(w, *b).unwrap();
        }
        Instr::Conv2d {
            dst,
            input,
            filter,
            stride_h,
            stride_w,
            padding,
        } => {
            w.write_all(&[OP_CONV2D]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *input).unwrap();
            write_vid(w, *filter).unwrap();
            uleb128_write(w, *stride_h as u64).unwrap();
            uleb128_write(w, *stride_w as u64).unwrap();
            w.write_all(&[padding_to_byte(*padding)]).unwrap();
        }
        Instr::Conv2dGradInput {
            dst,
            dy,
            filter,
            input_shape,
            stride_h,
            stride_w,
            padding,
        } => {
            w.write_all(&[OP_CONV2D_GRAD_INPUT]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *dy).unwrap();
            write_vid(w, *filter).unwrap();
            for &s in input_shape {
                uleb128_write(w, s as u64).unwrap();
            }
            uleb128_write(w, *stride_h as u64).unwrap();
            uleb128_write(w, *stride_w as u64).unwrap();
            w.write_all(&[padding_to_byte(*padding)]).unwrap();
        }
        Instr::Conv2dGradFilter {
            dst,
            input,
            dy,
            filter_shape,
            stride_h,
            stride_w,
            padding,
        } => {
            w.write_all(&[OP_CONV2D_GRAD_FILTER]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *input).unwrap();
            write_vid(w, *dy).unwrap();
            for &s in filter_shape {
                uleb128_write(w, s as u64).unwrap();
            }
            uleb128_write(w, *stride_h as u64).unwrap();
            uleb128_write(w, *stride_w as u64).unwrap();
            w.write_all(&[padding_to_byte(*padding)]).unwrap();
        }
        Instr::Index { dst, src, indices } => {
            w.write_all(&[OP_INDEX]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            uleb128_write(w, indices.len() as u64).unwrap();
            for idx in indices {
                write_i64(w, idx.axis).unwrap();
                write_i64(w, idx.index).unwrap();
            }
        }
        Instr::Slice { dst, src, dims } => {
            w.write_all(&[OP_SLICE]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            uleb128_write(w, dims.len() as u64).unwrap();
            for d in dims {
                write_i64(w, d.axis).unwrap();
                write_i64(w, d.start).unwrap();
                encode_opt_i64(w, d.end).unwrap();
                write_i64(w, d.stride).unwrap();
            }
        }
        Instr::Gather {
            dst,
            src,
            indices,
            axis,
        } => {
            w.write_all(&[OP_GATHER]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            write_vid(w, *indices).unwrap();
            write_i64(w, *axis).unwrap();
        }
        Instr::Output(id) => {
            w.write_all(&[OP_OUTPUT]).unwrap();
            write_vid(w, *id).unwrap();
        }
        Instr::SparseAttr { src, dst, layout } => {
            w.write_all(&[OP_SPARSE_ATTR]).unwrap();
            write_vid(w, *src).unwrap();
            write_vid(w, *dst).unwrap();
            w.write_all(&[sparse_layout_to_byte(*layout)]).unwrap();
        }
        Instr::FnDef {
            name,
            params,
            ret_id,
            body,
            reap_threshold,
        } => {
            w.write_all(&[OP_FN_DEF]).unwrap();
            encode_string_idx(w, name, st).unwrap();
            encode_named_vids(w, params, st).unwrap();
            encode_opt_vid(w, *ret_id).unwrap();
            encode_opt_f64(w, *reap_threshold).unwrap();
            // Emit body recursively
            uleb128_write(w, body.len() as u64).unwrap();
            for bi in body {
                emit_instr(w, bi, st);
            }
        }
        Instr::Call { dst, name, args } => {
            w.write_all(&[OP_CALL]).unwrap();
            write_vid(w, *dst).unwrap();
            encode_string_idx(w, name, st).unwrap();
            encode_vid_vec(w, args).unwrap();
        }
        Instr::Return { value } => {
            w.write_all(&[OP_RETURN]).unwrap();
            encode_opt_vid(w, *value).unwrap();
        }
        Instr::Param { dst, name, index } => {
            w.write_all(&[OP_PARAM]).unwrap();
            write_vid(w, *dst).unwrap();
            encode_string_idx(w, name, st).unwrap();
            uleb128_write(w, *index as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::ConstArray { dst, name, values } => {
            w.write_all(&[OP_CONST_ARRAY]).unwrap();
            write_vid(w, *dst).unwrap();
            encode_opt_string(w, name, st).unwrap();
            uleb128_write(w, values.len() as u64).unwrap();
            for &v in values {
                uleb128_write(w, zigzag_encode(v)).unwrap();
            }
        }
        #[cfg(feature = "std-surface")]
        Instr::ArrayLoad { dst, base, index } => {
            w.write_all(&[OP_ARRAY_LOAD]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *base).unwrap();
            write_vid(w, *index).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::While {
            cond_id,
            cond_instrs,
            body,
            live_vars,
            init_ids,
            // F2 exit_ids is lowering-internal: not serialised to the mic
            // wire format (fn bodies are not persisted to mic; hash-neutral).
            ..
        } => {
            w.write_all(&[OP_WHILE]).unwrap();
            write_vid(w, *cond_id).unwrap();
            uleb128_write(w, cond_instrs.len() as u64).unwrap();
            for ci in cond_instrs {
                emit_instr(w, ci, st);
            }
            uleb128_write(w, body.len() as u64).unwrap();
            for bi in body {
                emit_instr(w, bi, st);
            }
            encode_named_vids(w, live_vars, st).unwrap();
            encode_vid_vec(w, init_ids).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::If {
            cond_id,
            cond_instrs,
            then_instrs,
            then_result,
            else_instrs,
            else_result,
            dst,
            branch_bindings,
            // F2 merges is lowering-internal: not serialised to the mic wire
            // format (fn bodies are not persisted to mic; hash-neutral).
            ..
        } => {
            w.write_all(&[OP_IF]).unwrap();
            write_vid(w, *cond_id).unwrap();
            uleb128_write(w, cond_instrs.len() as u64).unwrap();
            for ci in cond_instrs {
                emit_instr(w, ci, st);
            }
            uleb128_write(w, then_instrs.len() as u64).unwrap();
            for ti in then_instrs {
                emit_instr(w, ti, st);
            }
            write_vid(w, *then_result).unwrap();
            uleb128_write(w, else_instrs.len() as u64).unwrap();
            for ei in else_instrs {
                emit_instr(w, ei, st);
            }
            write_vid(w, *else_result).unwrap();
            write_vid(w, *dst).unwrap();
            encode_named_vids(w, branch_bindings, st).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::VecLoad {
            dst,
            base,
            offset,
            lanes,
        } => {
            w.write_all(&[OP_VEC_LOAD]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *base).unwrap();
            write_vid(w, *offset).unwrap();
            uleb128_write(w, *lanes as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::VecFma {
            dst,
            a,
            b,
            acc,
            lanes,
        } => {
            w.write_all(&[OP_VEC_FMA]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *a).unwrap();
            write_vid(w, *b).unwrap();
            write_vid(w, *acc).unwrap();
            uleb128_write(w, *lanes as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::VecReduceAdd { dst, src, lanes } => {
            w.write_all(&[OP_VEC_REDUCE_ADD]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            uleb128_write(w, *lanes as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::VecStore {
            src,
            base,
            offset,
            lanes,
        } => {
            w.write_all(&[OP_VEC_STORE]).unwrap();
            write_vid(w, *src).unwrap();
            write_vid(w, *base).unwrap();
            write_vid(w, *offset).unwrap();
            uleb128_write(w, *lanes as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::VecLoadI32 {
            dst,
            base,
            offset,
            lanes,
        } => {
            w.write_all(&[OP_VEC_LOAD_I32]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *base).unwrap();
            write_vid(w, *offset).unwrap();
            uleb128_write(w, *lanes as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::VecMulAddQ16 {
            dst,
            a,
            b,
            acc,
            lanes,
        } => {
            w.write_all(&[OP_VEC_MUL_ADD_Q16]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *a).unwrap();
            write_vid(w, *b).unwrap();
            write_vid(w, *acc).unwrap();
            uleb128_write(w, *lanes as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::VecReduceAddI64 { dst, src, lanes } => {
            w.write_all(&[OP_VEC_REDUCE_ADD_I64]).unwrap();
            write_vid(w, *dst).unwrap();
            write_vid(w, *src).unwrap();
            uleb128_write(w, *lanes as u64).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::Region {
            body,
            result,
            enter_id,
            exit_id,
            alloc_ids,
        } => {
            w.write_all(&[OP_REGION]).unwrap();
            uleb128_write(w, body.len() as u64).unwrap();
            for bi in body {
                emit_instr(w, bi, st);
            }
            write_vid(w, *result).unwrap();
            write_vid(w, *enter_id).unwrap();
            write_vid(w, *exit_id).unwrap();
            encode_vid_vec(w, alloc_ids).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::ExternFnDecl {
            name,
            param_types,
            ret_type,
            is_varargs,
            vararg_hints,
            callconv,
        } => {
            w.write_all(&[OP_EXTERN_FN_DECL]).unwrap();
            encode_string_idx(w, name, st).unwrap();
            uleb128_write(w, param_types.len() as u64).unwrap();
            for t in param_types {
                encode_string_idx(w, t, st).unwrap();
            }
            // ret_type: Option<String>
            encode_opt_string(w, ret_type, st).unwrap();
            write_bool(w, *is_varargs).unwrap();
            uleb128_write(w, vararg_hints.len() as u64).unwrap();
            for h in vararg_hints {
                encode_string_idx(w, h, st).unwrap();
            }
            w.write_all(&[callconv_to_byte(*callconv)]).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::Break { live } => {
            w.write_all(&[OP_BREAK]).unwrap();
            encode_named_vids(w, live, st).unwrap();
        }
        #[cfg(feature = "std-surface")]
        Instr::Continue { live } => {
            w.write_all(&[OP_CONTINUE]).unwrap();
            encode_named_vids(w, live, st).unwrap();
        }
    }
}

fn encode_opt_i64<W: Write>(w: &mut W, v: Option<i64>) -> std::io::Result<()> {
    match v {
        None => w.write_all(&[0u8]),
        Some(n) => {
            w.write_all(&[1u8])?;
            write_i64(w, n)
        }
    }
}
