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

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use crate::types::DType;

use crate::types::ShapeDim;

use crate::types::TensorType;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub String);

#[cfg(feature = "cpu-buffers")]
#[derive(Debug, Clone, PartialEq)]
pub enum Buffer {
    I32(Vec<i32>),
    F32(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorVal {
    pub dtype: DType,
    pub shape: Vec<ShapeDim>,
    /// Optional constant fill value for all elements (preview only).
    pub fill: Option<f64>,
    #[cfg(feature = "cpu-buffers")]
    pub buf: Option<Buffer>,
}

impl TensorVal {
    pub fn from_type(t: &TensorType, fill: Option<f64>) -> Self {
        Self {
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            fill,
            #[cfg(feature = "cpu-buffers")]
            buf: None,
        }
    }

    pub fn new(dtype: DType, shape: Vec<ShapeDim>, fill: Option<f64>) -> Self {
        Self {
            dtype,
            shape,
            fill,
            #[cfg(feature = "cpu-buffers")]
            buf: None,
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn from_materialized_f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            dtype: DType::F32,
            shape: shape.into_iter().map(ShapeDim::Known).collect(),
            fill: None,
            buf: Some(Buffer::F32(data)),
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn is_materialized_f32(&self) -> bool {
        matches!(self.buf, Some(Buffer::F32(_)))
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.buf {
            Some(Buffer::F32(values)) => Some(values.as_slice()),
            _ => None,
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.buf {
            Some(Buffer::F32(values)) => Some(values.as_mut_slice()),
            _ => None,
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn shape_as_usize(&self) -> Option<Vec<usize>> {
        let mut out = Vec::with_capacity(self.shape.len());
        for dim in &self.shape {
            match dim {
                ShapeDim::Known(n) => out.push(*n),
                ShapeDim::Sym(_) => return None,
            }
        }
        Some(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Value>),
    Tensor(TensorVal),
    GradMap(BTreeMap<VarId, TensorVal>),
    /// Sum-type value (Phase 10.7): an enum/`Option`/`Result` variant with an
    /// optional positional payload. `variant` is the full path string as written
    /// in source (e.g. `"Result::Ok"`, `"Some"`, `"Mode::On"`); `payload` holds
    /// the constructor arguments (empty for unit variants).
    Enum {
        variant: String,
        payload: Vec<Value>,
    },
    /// Plain-struct value (interpreter): the struct's name plus its fields as
    /// `(field_name, value)` pairs in literal order. Field access resolves by
    /// NAME (`eval_field_access`), so declared-vs-literal field order never
    /// matters. Deterministic: construction evaluates fields in source order.
    /// The fields are a shared [`Record`]: a copy of the value denotes the
    /// SAME record, so a field store is seen through every copy.
    Struct {
        name: String,
        fields: Record,
    },
    /// A dynamic `array<T>` element run (a std.vec handle in the compiled
    /// build). The elements live in ONE shared backing ([`ArrayBuf`]), so a
    /// copy of the value denotes the SAME run: an element store through one
    /// reference is seen through every alias, the way a copied handle shares
    /// the heap backing. Only a struct field DECLARED `array<T>` carries this
    /// variant (the `record_store` schema decides at construction); a fixed
    /// `[T; N]` field stays a value container ([`Value::Tuple`]), exactly as
    /// the compiled build stores it inline.
    Array(ArrayBuf),
}

/// The fields of one struct record (types.md "Record identity"): binding,
/// assigning, passing or returning a struct value shares the record and never
/// clones its fields, so a field store through one reference is visible
/// through all others. A struct literal is the only way to make a new record.
#[derive(Clone)]
pub struct Record(Rc<RefCell<Vec<(String, Value)>>>);

impl Record {
    pub fn new(fields: Vec<(String, Value)>) -> Self {
        Record(Rc::new(RefCell::new(fields)))
    }

    /// The current value of `field`, if the record has it.
    pub fn get(&self, field: &str) -> Option<Value> {
        let fields = self.0.borrow();
        fields
            .iter()
            .find(|(f, _)| f == field)
            .map(|(_, v)| v.clone())
    }

    /// Store `value` into `field`; `false` when the record has no such field.
    pub fn set(&self, field: &str, value: Value) -> bool {
        let mut fields = self.0.borrow_mut();
        match fields.iter_mut().find(|(f, _)| f == field) {
            Some(slot) => {
                slot.1 = value;
                true
            }
            None => false,
        }
    }

    /// A copy of the `(field, value)` pairs, in literal order.
    pub fn fields(&self) -> Vec<(String, Value)> {
        self.0.borrow().clone()
    }
}

impl PartialEq for Record {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0) || *self.0.borrow() == *other.0.borrow()
    }
}

/// The shared element run of a dynamic `array<T>` value: binding, assigning,
/// passing or returning the value shares the run and never clones it, so an
/// element store made through one reference is visible through every alias.
/// That is the tree evaluator's model of the std.vec handle the compiled
/// build copies by value while its heap backing stays shared. A struct
/// literal whose field is DECLARED `array<T>` is the only producer
/// (`record_store::wrap_struct_field`); the interpreter keeps carrying every
/// other array shape as a plain [`Value::Tuple`].
#[derive(Clone)]
pub struct ArrayBuf(Rc<RefCell<Vec<Value>>>);

impl ArrayBuf {
    pub fn new(items: Vec<Value>) -> Self {
        ArrayBuf(Rc::new(RefCell::new(items)))
    }

    /// The current element at `idx`, if the run has it.
    pub fn get(&self, idx: usize) -> Option<Value> {
        self.0.borrow().get(idx).cloned()
    }

    /// Store `value` into `idx`; `false` (storing nothing) when `idx` is out
    /// of bounds.
    pub fn set(&self, idx: usize, value: Value) -> bool {
        let mut items = self.0.borrow_mut();
        match items.get_mut(idx) {
            Some(slot) => {
                *slot = value;
                true
            }
            None => false,
        }
    }

    /// The run's current length.
    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }

    /// An empty run is a legal `array<T>` value (a never-pushed vec).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A value-snapshot of the whole run, as the plain `Value::Tuple` the
    /// interpreter uses for array copies.
    pub fn items(&self) -> Vec<Value> {
        self.0.borrow().clone()
    }
}

impl PartialEq for ArrayBuf {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0) || *self.0.borrow() == *other.0.borrow()
    }
}

thread_local! {
    static PRINT_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Run `f` one record-nesting level deeper, or return `None` past a fixed
/// depth: a record can reach itself through its fields, and printing one must
/// stop rather than recurse until the stack overflows.
fn nested<R>(f: impl FnOnce() -> R) -> Option<R> {
    struct Restore(u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRINT_DEPTH.with(|d| d.set(self.0));
        }
    }
    let depth = PRINT_DEPTH.with(Cell::get);
    if depth >= 16 {
        return None;
    }
    PRINT_DEPTH.with(|d| d.set(depth + 1));
    let _restore = Restore(depth);
    Some(f())
}

impl fmt::Debug for Record {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        nested(|| f.debug_list().entries(self.fields()).finish())
            .unwrap_or_else(|| f.write_str("[..]"))
    }
}

impl fmt::Debug for ArrayBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Same recursion guard as `Record`: a run can reach itself through a
        // record field, and printing one must stop, not overflow the stack.
        nested(|| f.debug_list().entries(self.items()).finish())
            .unwrap_or_else(|| f.write_str("[..]"))
    }
}

impl Value {
    pub fn is_int(&self) -> bool {
        matches!(self, Value::Int(_))
    }

    pub fn as_int(&self) -> Option<i64> {
        if let Value::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// A copy of this value for a NEW read of a module `const`. The compiled
    /// build inlines a const's literal at every use, so every use gets its
    /// own records and array runs and a store through one use can never leak
    /// into another use — or back into the const. Shared backings ([`Value::Struct`]
    /// records, [`Value::Array`] runs, and the same nested inside tuples and
    /// enum payloads) are rebuilt fresh; scalars, strings and tensors pass
    /// through unchanged (they share nothing).
    pub fn fresh_for_const_use(&self) -> Value {
        match self {
            Value::Struct { name, fields } => Value::Struct {
                name: name.clone(),
                fields: Record::new(
                    fields
                        .fields()
                        .into_iter()
                        .map(|(fname, fval)| (fname, fval.fresh_for_const_use()))
                        .collect(),
                ),
            },
            Value::Array(run) => Value::Array(ArrayBuf::new(
                run.items()
                    .into_iter()
                    .map(|item| item.fresh_for_const_use())
                    .collect(),
            )),
            Value::Tuple(items) => Value::Tuple(
                items
                    .iter()
                    .map(|item| item.fresh_for_const_use())
                    .collect(),
            ),
            Value::Enum { variant, payload } => Value::Enum {
                variant: variant.clone(),
                payload: payload
                    .iter()
                    .map(|item| item.fresh_for_const_use())
                    .collect(),
            },
            other => other.clone(),
        }
    }
}

pub fn format_value_human(v: &Value) -> String {
    match v {
        Value::Int(n) => format!("{n}"),
        Value::Float(f) => trim_float(*f),
        Value::Str(s) => s.clone(),
        Value::Tuple(items) => {
            let mut out = String::from("(");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format_value_human(item));
            }
            out.push(')');
            out
        }
        Value::Array(run) => {
            let mut out = String::from("(");
            for (i, item) in run.items().iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format_value_human(item));
            }
            out.push(')');
            out
        }
        Value::Tensor(t) => {
            let mut shape = String::from("(");
            for (i, d) in t.shape.iter().enumerate() {
                if i > 0 {
                    shape.push(',');
                }
                match d {
                    ShapeDim::Known(n) => shape.push_str(&n.to_string()),
                    ShapeDim::Sym(sym) => shape.push_str(sym),
                }
            }
            shape.push(')');
            #[cfg(feature = "cpu-buffers")]
            if let Some(buf) = &t.buf {
                let sample = format_buffer_sample(buf, 8);
                return format!(
                    "Tensor[{dtype:?},{shape}] materialized (sample=[{sample}])",
                    dtype = t.dtype,
                    shape = shape,
                    sample = sample
                );
            }
            match t.fill {
                Some(f) => format!(
                    "Tensor[{dtype:?},{shape}] fill={fill}",
                    dtype = t.dtype,
                    shape = shape,
                    fill = trim_float(f)
                ),
                None => format!("Tensor[{dtype:?},{shape}]", dtype = t.dtype, shape = shape),
            }
        }
        Value::GradMap(m) => {
            let mut parts = Vec::new();
            for (var, tensor) in m {
                let tensor_str = format_value_human(&Value::Tensor(tensor.clone()));
                parts.push(format!("{}: {}", var.0, tensor_str));
            }
            format!("grad{{ {} }}", parts.join(", "))
        }
        Value::Enum { variant, payload } => {
            if payload.is_empty() {
                variant.clone()
            } else {
                let mut out = String::new();
                out.push_str(variant);
                out.push('(');
                for (i, item) in payload.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&format_value_human(item));
                }
                out.push(')');
                out
            }
        }
        Value::Struct { name, fields } => nested(|| {
            let mut out = String::new();
            out.push_str(name);
            out.push_str(" { ");
            for (i, (fname, fval)) in fields.fields().iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(fname);
                out.push_str(": ");
                out.push_str(&format_value_human(fval));
            }
            out.push_str(" }");
            out
        })
        .unwrap_or_else(|| format!("{name} {{ .. }}")),
    }
}

fn trim_float(x: f64) -> String {
    let s = format!("{:.6}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(feature = "cpu-buffers")]
fn format_buffer_sample(buf: &Buffer, limit: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    match buf {
        Buffer::I32(values) => {
            let max = values.len().min(limit);
            for v in &values[..max] {
                parts.push(v.to_string());
            }
            if values.len() > limit {
                parts.push("...".to_string());
            }
        }
        Buffer::F32(values) => {
            let max = values.len().min(limit);
            for v in &values[..max] {
                parts.push(trim_float(*v as f64));
            }
            if values.len() > limit {
                parts.push("...".to_string());
            }
        }
    }
    parts.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DType;
    use crate::types::ShapeDim;

    #[test]
    fn format_tensor_fill_trim() {
        let t = TensorVal::new(DType::F32, vec![ShapeDim::Known(2)], Some(1.2300001));
        let s = format_value_human(&Value::Tensor(t));
        assert!(s.contains("1.23"));
    }

    #[test]
    fn format_tensor_shape_symbols() {
        let t = TensorVal::new(
            DType::I32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(4)],
            None,
        );
        let s = format_value_human(&Value::Tensor(t));
        assert!(s.contains("(B,4)"));
    }
}
