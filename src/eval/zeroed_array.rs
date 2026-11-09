// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `vec_zeroed(n)` buffers held by an `array<T>` slot.
//!
//! `vec_zeroed(n)` allocates `n` zeroed i64 cells and returns the base address
//! as a plain integer (the runtime-support intrinsic and `std.vec` agree). The
//! compiled build never looks at the call: it types the slot, and a slot
//! declared `array<T>` (a `let`, a struct field, a parameter, a return value)
//! is a std.vec handle, so `a[i]` is `vec_get`, `a[i] = v` is `vec_set`,
//! `a.len()` is `vec_len`, and passing `a` to a `&mut [T]` parameter passes
//! the same handle, so the callee's stores are the caller's.
//!
//! The evaluator models that handle with the shared element run
//! ([`Value::Array`], see `record_store`). This module is the missing edge: a
//! buffer `vec_zeroed` returned becomes that run when an `array<T>` slot takes
//! it. One buffer adopted twice yields the SAME run (one backing, as one
//! handle), and a slot declared as anything else (`let z: i64 = vec_zeroed(2)`
//! read through `__mind_load_i64`) keeps the plain integer.
//!
//! Only a buffer `vec_zeroed` returned in THIS test is ever adopted (the
//! registry below is empty otherwise), so a program that never calls it sees no
//! change, and the reassignment check is one empty-map test.
//!
//! shortcut: an element type that is a float alias (`type F = f64`) gets an
//! integer zero. Upgrade path: carry the resolved element type through here.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use crate::ast::TypeAnn;

use super::value::ArrayBuf;
use super::{EvalError, Value, declared_width, interp_mem};

/// One `vec_zeroed` buffer: its length in cells, and the run once a slot has
/// adopted it (so a second slot adopting the same buffer shares the run).
struct Buffer {
    len: usize,
    run: Option<ArrayBuf>,
}

thread_local! {
    /// Buffers keyed by arena base address. A `BTreeMap`, so nothing here is
    /// iterated in a hash order.
    static BUFFERS: RefCell<BTreeMap<i64, Buffer>> = const { RefCell::new(BTreeMap::new()) };
}

/// Forget every buffer. Called with the arena reset, once per test.
pub(super) fn reset() {
    BUFFERS.with(|b| b.borrow_mut().clear());
}

/// Record the buffer a `vec_zeroed(len)` call just returned at `base`. A
/// non-positive count allocates nothing and returns 0: the empty buffer.
pub(super) fn register(base: i64, len: usize) {
    BUFFERS.with(|b| b.borrow_mut().insert(base, Buffer { len, run: None }));
}

/// The element type of a dynamic `array<T>` annotation or a slice `&[T]` /
/// `&mut [T]`; `None` for any other type, so a scalar slot never adopts.
fn element(ann: &TypeAnn) -> Option<&TypeAnn> {
    match ann {
        TypeAnn::Generic { name, args } if name == "array" => args.first(),
        TypeAnn::Slice { element, .. } => Some(element.as_ref()),
        _ => None,
    }
}

fn is_float(elem: &TypeAnn) -> bool {
    match elem {
        TypeAnn::ScalarF32 | TypeAnn::ScalarF64 => true,
        TypeAnn::Named(n) => matches!(n.as_str(), "f32" | "f64"),
        _ => false,
    }
}

/// Does the collection annotation `ann` hold float elements?
pub(super) fn has_float_elements(ann: &TypeAnn) -> bool {
    element(ann).is_some_and(is_float)
}

/// What a slot declared `ann` holds when it is given `value`: the run of the
/// `vec_zeroed` buffer when `value` is one and `ann` is a collection type,
/// `value` itself otherwise.
pub(super) fn adopt(ann: &TypeAnn, value: Value) -> Result<Value, EvalError> {
    let Some(elem) = element(ann) else {
        return Ok(value);
    };
    adopt_field(value, is_float(elem))
}

/// A `let` binding: the declared type decides whether the slot adopts a
/// `vec_zeroed` buffer, then the declared-width layer materialises it.
pub(super) fn bind_let(
    ann: &Option<TypeAnn>,
    name: &str,
    value: Value,
) -> Result<Value, EvalError> {
    let value = match ann {
        Some(ann) => adopt(ann, value)?,
        None => value,
    };
    declared_width::bind_let(ann, name, value)
}

/// A fn return: the declared return type decides, then the declared-width
/// layer materialises the value.
pub(super) fn narrow_return(
    ret: &Option<TypeAnn>,
    callee: &str,
    value: Value,
) -> Result<Value, EvalError> {
    let value = match ret {
        Some(ann) => adopt(ann, value)?,
        None => value,
    };
    declared_width::narrow_return(ret, callee, value)
}

/// Is `base` a buffer `vec_zeroed` returned in this test?
fn is_buffer(base: i64) -> bool {
    BUFFERS.with(|b| {
        let b = b.borrow();
        !b.is_empty() && b.contains_key(&base)
    })
}

/// A reassignment `a = vec_zeroed(n)` carries no annotation: the slot keeps
/// the kind it holds, so a variable that holds an array adopts the buffer.
pub(super) fn reassign(
    env: &HashMap<String, Value>,
    name: &str,
    value: Value,
) -> Result<Value, EvalError> {
    let Value::Int(base) = value else {
        return Ok(value);
    };
    if !is_buffer(base) {
        return Ok(value);
    }
    let first = match env.get(name) {
        Some(Value::Array(run)) => run.get(0),
        Some(Value::Tuple(items)) => items.first().cloned(),
        _ => return Ok(value),
    };
    adopt_field(value, matches!(first, Some(Value::Float(_))))
}

/// A slot known to hold a collection (a struct field declared `array<T>`):
/// `value` becomes the run of its `vec_zeroed` buffer, or stays as it is.
pub(super) fn adopt_field(value: Value, float_elements: bool) -> Result<Value, EvalError> {
    let Value::Int(base) = value else {
        return Ok(value);
    };
    match run_of(base, float_elements) {
        Some(run) => run.map(Value::Array),
        None => Ok(value),
    }
}

/// The run for the buffer at `base`; `None` when `base` is not a buffer
/// `vec_zeroed` returned in this test.
fn run_of(base: i64, float_elements: bool) -> Option<Result<ArrayBuf, EvalError>> {
    let (len, shared) =
        BUFFERS.with(|b| b.borrow().get(&base).map(|buf| (buf.len, buf.run.clone())))?;
    if let Some(run) = shared {
        return Some(Ok(run));
    }
    Some(read_cells(base, len, float_elements).map(|cells| {
        let run = ArrayBuf::new(cells);
        BUFFERS.with(|b| {
            if let Some(buf) = b.borrow_mut().get_mut(&base) {
                buf.run = Some(run.clone());
            }
        });
        run
    }))
}

/// The buffer's cells as they stand in the arena (a program may have stored
/// through the integer before the slot took it).
fn read_cells(base: i64, len: usize, float_elements: bool) -> Result<Vec<Value>, EvalError> {
    (0..len)
        .map(|i| -> Result<Value, EvalError> {
            let addr = i64::try_from(i)
                .ok()
                .and_then(|i| i.checked_mul(8))
                .and_then(|off| base.checked_add(off))
                .ok_or_else(|| EvalError::UnsupportedMsg("vec_zeroed: size overflow".into()))?;
            let cell = interp_mem::eval_intrinsic("__mind_load_i64", &[addr])
                .map_err(EvalError::UnsupportedMsg)?;
            Ok(if float_elements {
                Value::Float(f64::from_bits(cell as u64))
            } else {
                Value::Int(cell)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array_of(elem: TypeAnn) -> TypeAnn {
        TypeAnn::Generic {
            name: "array".to_string(),
            args: vec![elem],
        }
    }

    /// A fresh arena holding one `vec_zeroed(n)` buffer; returns its base.
    fn zeroed(n: i64) -> i64 {
        interp_mem::reset();
        interp_mem::eval_intrinsic("__mind_vec_zeroed", &[n]).expect("vec_zeroed")
    }

    #[test]
    fn an_array_slot_takes_the_buffer_as_a_zeroed_run() {
        let base = zeroed(3);
        let held = adopt(&array_of(TypeAnn::ScalarI32), Value::Int(base)).unwrap();
        let Value::Array(run) = held else {
            panic!("expected a run, got {held:?}");
        };
        assert_eq!(run.items(), vec![Value::Int(0); 3]);
    }

    #[test]
    fn a_slice_slot_takes_it_too() {
        let base = zeroed(2);
        let slice = TypeAnn::Slice {
            mutable: true,
            element: Box::new(TypeAnn::ScalarI32),
        };
        assert!(matches!(
            adopt(&slice, Value::Int(base)).unwrap(),
            Value::Array(_)
        ));
    }

    #[test]
    fn one_buffer_is_one_backing() {
        let base = zeroed(2);
        let ty = array_of(TypeAnn::ScalarI64);
        let (Value::Array(a), Value::Array(b)) = (
            adopt(&ty, Value::Int(base)).unwrap(),
            adopt(&ty, Value::Int(base)).unwrap(),
        ) else {
            panic!("expected two runs");
        };
        assert!(a.set(1, Value::Int(9)));
        assert_eq!(b.get(1), Some(Value::Int(9)));
    }

    #[test]
    fn other_slots_and_other_integers_keep_the_integer() {
        let base = zeroed(2);
        assert_eq!(
            adopt(&TypeAnn::ScalarI64, Value::Int(base)).unwrap(),
            Value::Int(base)
        );
        let ty = array_of(TypeAnn::ScalarI64);
        assert_eq!(
            adopt(&ty, Value::Int(base + 1)).unwrap(),
            Value::Int(base + 1)
        );
        assert_eq!(adopt(&ty, Value::Float(1.5)).unwrap(), Value::Float(1.5));
    }

    #[test]
    fn the_empty_buffer_and_float_elements() {
        let ty = array_of(TypeAnn::ScalarI64);
        assert_eq!(zeroed(0), 0);
        let Value::Array(empty) = adopt(&ty, Value::Int(0)).unwrap() else {
            panic!("expected an empty run");
        };
        assert!(empty.is_empty());
        let base = zeroed(1);
        let float_ty = array_of(TypeAnn::ScalarF64);
        assert!(has_float_elements(&float_ty) && !has_float_elements(&ty));
        let Value::Array(run) = adopt(&float_ty, Value::Int(base)).unwrap() else {
            panic!("expected a float run");
        };
        assert_eq!(run.items(), vec![Value::Float(0.0)]);
    }

    #[test]
    fn a_store_made_before_adoption_is_kept() {
        let base = zeroed(2);
        interp_mem::eval_intrinsic("__mind_store_i64", &[base + 8, 7]).unwrap();
        let held = adopt(&array_of(TypeAnn::ScalarI64), Value::Int(base)).unwrap();
        let Value::Array(run) = held else {
            panic!("expected a run, got {held:?}");
        };
        assert_eq!(run.items(), vec![Value::Int(0), Value::Int(7)]);
    }

    #[test]
    fn a_program_that_never_called_vec_zeroed_keeps_every_integer() {
        interp_mem::reset();
        let ty = array_of(TypeAnn::ScalarI64);
        assert_eq!(adopt(&ty, Value::Int(0)).unwrap(), Value::Int(0));
        assert_eq!(adopt(&ty, Value::Int(8)).unwrap(), Value::Int(8));
    }

    #[test]
    fn a_reassignment_adopts_only_into_an_array_variable() {
        let base = zeroed(2);
        let env: HashMap<String, Value> = [
            ("held".to_string(), Value::Tuple(vec![Value::Int(1)])),
            ("plain".to_string(), Value::Int(5)),
        ]
        .into_iter()
        .collect();
        assert!(matches!(
            reassign(&env, "held", Value::Int(base)).unwrap(),
            Value::Array(_)
        ));
        assert_eq!(
            reassign(&env, "plain", Value::Int(base)).unwrap(),
            Value::Int(base)
        );
        assert_eq!(
            reassign(&env, "absent", Value::Int(base)).unwrap(),
            Value::Int(base)
        );
        assert_eq!(
            reassign(&env, "held", Value::Int(base + 1)).unwrap(),
            Value::Int(base + 1)
        );
    }
}
