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
