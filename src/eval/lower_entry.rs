// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Thin lowering entry wrapper used to keep the legacy lowerer under its
//! established source-size ceiling.
//!
//! It also owns the `std.posix` link-name step. That module keeps its source
//! names (`close`, `open`, ...) but is emitted as `mind_posix_*`, so a program
//! that also declares the C name in an `extern "C"` block still links to libc.

#[cfg(feature = "cross-module-imports")]
use super::canonical_producers;
use super::{HashMap, IRModule, Instr, LoweringContext, ValueId, ast};
use crate::eval::materialization::ConstRefusal;
use crate::opt::fold;
use crate::types::{DType, ShapeDim};

/// Lower one whole module. Every module lowering enters here, so the
/// file-local `std.posix` admission matches type-checking: the exports are
/// registered only when this module imports `std.posix`, and a local
/// `extern "C"` name is left to its declaration. Installed before
/// `visible_fns` runs and dropped when lowering returns.
pub(super) fn lower_module(module: &ast::Module, context: &mut LoweringContext) -> IRModule {
    #[cfg(feature = "cross-module-imports")]
    let _file_scope = crate::project::active_module_table::FileScopeGuard::install(module);
    super::lower_to_ir_inner(module, context)
}

impl IRModule {
    /// Push a function definition, renaming a `std.posix` function to its link name.
    pub(super) fn push_fn_def(&mut self, def: Instr) {
        #[cfg(feature = "cross-module-imports")]
        let def = {
            let mut def = def;
            if let Instr::FnDef { name, .. } = &mut def {
                let symbol = posix::link_name(name);
                posix::alias_signature(self, name, &symbol);
                *name = symbol;
            }
            def
        };
        self.instrs.push(def);
    }
}

/// The callee symbol of a call: the generic instance name when the call
/// matches the bounded mono slice, else the callee, then its `std.posix`
/// link name when it resolves there.
pub(super) fn call_name(
    ir: &mut IRModule,
    callee: &str,
    args: &[ast::Node],
    span: ast::Span,
) -> String {
    let name =
        super::try_register_mono_instance(callee, args).unwrap_or_else(|| callee.to_string());
    #[cfg(feature = "cross-module-imports")]
    let name = posix::call_symbol(ir, name, callee, span);
    #[cfg(not(feature = "cross-module-imports"))]
    let _ = (ir, span);
    name
}

#[cfg(feature = "cross-module-imports")]
mod posix {
    use super::IRModule;
    use crate::project::active_module_table as table;

    pub(super) fn link_name(name: &str) -> String {
        table::link_name(
            crate::qualified_enums::current_module_path().as_deref(),
            name,
        )
    }

    /// The renamed symbol inherits the source name's signature, if any.
    pub(super) fn alias_signature(ir: &mut IRModule, name: &str, symbol: &str) {
        if symbol != name {
            if let Some(sig) = ir.fn_signatures.get(name).cloned() {
                ir.fn_signatures.entry(symbol.to_string()).or_insert(sig);
            }
        }
    }

    /// A call qualified with a `std.posix` path (`posix.close(fd)`) always
    /// links to the module, even when a local extern has the bare name; a
    /// bare call does when it resolves to `std.posix`.
    pub(super) fn call_symbol(
        ir: &mut IRModule,
        name: String,
        callee: &str,
        span: crate::ast::Span,
    ) -> String {
        if !table::is_qualified_posix_call(span) {
            let symbol = link_name(&name);
            alias_signature(ir, &name, &symbol);
            return symbol;
        }
        let symbol = format!("mind_posix_{name}");
        if let Some(sig) = table::qualified_import_fn(span, callee) {
            ir.fn_signatures
                .insert(symbol.clone(), (sig.param_types, sig.ret_type));
        }
        alias_signature(ir, &name, &symbol);
        symbol
    }
}

#[inline]
pub(super) fn lower_expr(
    node: &ast::Node,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
    context: &mut LoweringContext,
) -> ValueId {
    let id = super::lower_expr_inner(node, ir, env, struct_env, receiver_types, context);
    #[cfg(feature = "cross-module-imports")]
    if context.canonical_active() && !context.canonical_failed() {
        canonical_producers::record_checked_expr(node, id, context);
    }
    id
}

#[inline]
pub(super) fn validated_value(
    _context: &mut LoweringContext,
    _source: &ast::Node,
    value: ValueId,
) -> ValueId {
    #[cfg(feature = "cross-module-imports")]
    _context.canonical_validate_return(_source.span(), value);
    value
}

#[inline]
pub(super) fn ret(
    context: &mut LoweringContext,
    source: Option<&ast::Node>,
    value: Option<ValueId>,
) -> Option<ValueId> {
    if let (Some(source), Some(value)) = (source, value) {
        Some(validated_value(context, source, value))
    } else {
        value
    }
}

/// Stop lowering at `at` with a named constant refusal (the historical paths
/// stored 0 or a saturated value for it, a silent wrong value).
pub(super) fn refuse_const(
    cx: &mut LoweringContext,
    kind: ConstRefusal,
    op: &'static str,
    at: &ast::Node,
) {
    let span = at.span();
    cx.refusal
        .get_or_insert(kind.at(op, span.start(), span.end()));
}

/// The dense bits of a literal's elements (`fold::dense_elem_bits`), or `None`
/// after a named refusal at `at`.
pub(super) fn dense_bits<'a>(
    elements: impl IntoIterator<Item = &'a ast::Node>,
    dtype: &DType,
    op: &'static str,
    at: &ast::Node,
    cx: &mut LoweringContext,
) -> Option<Vec<u64>> {
    let bits = elements
        .into_iter()
        .map(|e| fold::dense_elem_bits(e, dtype));
    bits.collect::<Result<Vec<u64>, _>>()
        .map_err(|e| refuse_const(cx, e.into(), op, at))
        .ok()
}

/// The shape of a nested array literal and its leaves in row-major order.
#[derive(Default)]
struct Nest<'a> {
    dims: Vec<usize>,
    leaf_depth: Option<usize>,
    leaves: Vec<&'a ast::Node>,
}

impl<'a> Nest<'a> {
    /// False when the nest is ragged: siblings of unequal length, or a leaf and a
    /// list at one level, or leaves at two depths.
    fn walk(&mut self, elems: &'a [ast::Node], depth: usize) -> bool {
        if depth == self.dims.len() {
            self.dims.push(elems.len());
        }
        if self.dims[depth] != elems.len() {
            return false;
        }
        for e in elems {
            if let ast::Node::ArrayLit { elements, .. } = e {
                if !self.walk(elements, depth + 1) {
                    return false;
                }
            } else {
                if *self.leaf_depth.get_or_insert(depth + 1) != depth + 1 {
                    return false;
                }
                self.leaves.push(e);
            }
        }
        true
    }
}

/// A dense tensor literal `[e0, ..]` or a nested `[[..], ..]` of constants,
/// flattened row-major into one `ConstDenseTensor`. `Some` when this handled
/// the literal (built or refused by name); `None` leaves a flat literal whose
/// count is not the shape product, or a symbolic shape, to the type checker.
/// A nest that is ragged or does not have the declared shape is `RaggedNest`; a
/// leaf count that is not the shape product is `DenseLeafCount`.
pub(super) fn dense_literal(
    ir: &mut IRModule,
    elements: &[ast::Node],
    dtype: &DType,
    shape: &[ShapeDim],
    value: &ast::Node,
    cx: &mut LoweringContext,
) -> Option<ValueId> {
    let op = "dense_tensor_literal";
    let known: Option<Vec<usize>> = shape
        .iter()
        .map(|d| match d {
            ShapeDim::Known(n) => Some(*n),
            _ => None,
        })
        .collect();
    let nested = elements
        .iter()
        .any(|e| matches!(e, ast::Node::ArrayLit { .. }));
    let leaves: Vec<&ast::Node> = if nested {
        let known = known?;
        let mut nest = Nest::default();
        let regular =
            nest.walk(elements, 0) && nest.leaf_depth.is_none_or(|d| d == nest.dims.len());
        if !regular {
            refuse_const(cx, ConstRefusal::RaggedNest, op, value);
            return Some(ir.fresh());
        }
        let want = known.iter().try_fold(1usize, |n, k| n.checked_mul(*k));
        if want != Some(nest.leaves.len()) {
            refuse_const(cx, ConstRefusal::DenseLeafCount, op, value);
            return Some(ir.fresh());
        }
        if nest.dims != known {
            refuse_const(cx, ConstRefusal::RaggedNest, op, value);
            return Some(ir.fresh());
        }
        nest.leaves
    } else {
        let expected: usize = known.map_or(0, |k| k.iter().product());
        if expected != elements.len() {
            return None;
        }
        elements.iter().collect()
    };
    if !cx.charge_static_elements(leaves.len(), false) {
        return Some(ir.fresh());
    }
    let Some(data) = dense_bits(leaves, dtype, op, value, cx) else {
        return Some(ir.fresh());
    };
    let dst = ir.fresh();
    let (dtype, shape) = (dtype.clone(), shape.to_vec());
    ir.instrs.push(Instr::ConstDenseTensor {
        dst,
        dtype,
        shape,
        data,
    });
    Some(dst)
}

/// A constant scalar fill of a tensor binding. Q16.16 stores Q(x) in every
/// element as a dense constant (spec 3.3; the scalar `ConstTensor` fill has no
/// Q16.16 form). An integer tensor refuses a fill its element type does not
/// hold exactly. `Some` when this handled the fill (built or refused).
pub(super) fn const_fill(
    ir: &mut IRModule,
    value: &ast::Node,
    dtype: &DType,
    shape: &[ShapeDim],
    cx: &mut LoweringContext,
) -> Option<ValueId> {
    if matches!(value, ast::Node::ArrayLit { .. }) {
        return None;
    }
    let x = fold::const_float(value)?;
    if !matches!(dtype, DType::Q16) {
        // The IR fill is binary64: an integer it rounds (2^53 + 1) is refused, never stored rounded.
        let int_fill = matches!(dtype, DType::I32 | DType::I64);
        let held = !int_fill || fold::const_int(value).is_none_or(fold::f64_holds_int);
        if held && fold::exact_int_fill(x, dtype) {
            return None;
        }
        refuse_const(cx, ConstRefusal::InexactIntegerFill, "integer_fill", value);
        return Some(ir.fresh());
    }
    let Some(q) = fold::q16_const(x) else {
        refuse_const(cx, ConstRefusal::Q16OutOfRange, "q16_fill", value);
        return Some(ir.fresh());
    };
    let known = |n: usize, d: &ShapeDim| match d {
        ShapeDim::Known(k) => n.checked_mul(*k),
        _ => None,
    };
    let Some(n) = shape.iter().try_fold(1usize, known) else {
        refuse_const(
            cx,
            ConstRefusal::Q16Unsupported,
            "q16_fill_symbolic_shape",
            value,
        );
        return Some(ir.fresh());
    };
    if !cx.charge_static_elements(n, false) {
        return Some(ir.fresh());
    }
    let (dst, data) = (ir.fresh(), vec![q as u32 as u64; n]);
    let (dtype, shape) = (DType::Q16, shape.to_vec());
    ir.instrs.push(Instr::ConstDenseTensor {
        dst,
        dtype,
        shape,
        data,
    });
    Some(dst)
}
