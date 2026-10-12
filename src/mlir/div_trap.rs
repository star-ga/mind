// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! W2-trap: the deterministic integer-division trap of the MLIR lowering.
//! `x / 0`, `x % 0` (every integer width) and i64 `INT_MIN / -1`,
//! `INT_MIN % -1` exit the process with [`MIND_DIV_TRAP_EXIT_CODE`] through an
//! internal `@__mind_div_trap` trampoline (docs/errors.md "Integer division
//! trap"); the pure-MIND native emitter uses the same status.

use super::{LoweringContext, MlirLowerError, extern_ret_mlir_ty};

/// Process exit status of the integer-division trap (W2-trap, docs/errors.md
/// "Integer division trap"): i64 `x / 0`, `x % 0`, `INT_MIN / -1` and
/// `INT_MIN % -1` (and a zero divisor at every narrower or unsigned width)
/// terminate the process with this status and write nothing further. The
/// pure-MIND native emitter uses the same value (`nb_div_trap_code` in
/// examples/mindc_mind/main.mind), so both backends agree.
pub(super) const MIND_DIV_TRAP_EXIT_CODE: i32 = 79;

/// The call a division trap guard emits. `@__mind_div_trap` is an internal
/// trampoline the module assembler defines once (scanning for this text) that
/// calls libc `_exit(MIND_DIV_TRAP_EXIT_CODE)`; routing through it keeps the
/// guard independent of how (or whether) the program declares `_exit` itself.
pub(super) const DIV_TRAP_CALL: &str = "llvm.call @__mind_div_trap()";

impl LoweringContext {
    /// W2-trap: emit `scf.if <bad> { __mind_div_trap() }` ahead of a guarded
    /// division (the trampoline `_exit`s with MIND_DIV_TRAP_EXIT_CODE). `bad` is the
    /// i1 trap predicate. `_exit` runs no atexit handler and flushes no stdio buffer, so nothing is written after
    /// the trap (the MIND runtime's writes are unbuffered `write(2)`s already).
    /// The divisor the caller then feeds the `arith.div*`/`rem*` stays
    /// substituted, so the division itself is never undefined behaviour.
    pub(super) fn emit_div_trap_guard(&mut self, bad: &str) {
        self.emit_line(&format!("    scf.if {bad} {{"));
        self.emit_line(&format!("      {DIV_TRAP_CALL} : () -> ()"));
        self.emit_line("    }");
    }
}

/// Define the internal `@__mind_div_trap` trampoline once iff a division trap
/// guard was emitted (authoritative buffer scan, like `@malloc`). It calls libc
/// `_exit(MIND_DIV_TRAP_EXIT_CODE)`. A module with no integer division emits
/// nothing new.
pub(super) fn append_trampoline(
    ctx: &LoweringContext,
    out: &mut String,
) -> Result<(), MlirLowerError> {
    #[cfg(feature = "std-surface")]
    let references_div_trap = ctx.body.contains(DIV_TRAP_CALL)
        || ctx.user_fns.contains(DIV_TRAP_CALL)
        || ctx.mt_workers.contains(DIV_TRAP_CALL);
    #[cfg(not(feature = "std-surface"))]
    let references_div_trap = ctx.body.contains(DIV_TRAP_CALL);
    if references_div_trap {
        out.push_str(&trampoline(ctx)?);
    }
    Ok(())
}

/// The libc process-exit symbol the trampoline calls: `_exit`, or its C99 twin
/// `_Exit` (same contract: no atexit handlers, no stdio flush) when the program
/// defines its own MIND fn named `_exit` (a second `_exit` symbol would clash
/// with it, and calling it would run the program's function instead of
/// exiting). A program that defines both names is refused.
fn exit_symbol(ctx: &LoweringContext) -> Result<&'static str, MlirLowerError> {
    #[cfg(feature = "std-surface")]
    let defines = |name: &str| ctx.defined_fns.contains(name);
    #[cfg(not(feature = "std-surface"))]
    let defines = |_name: &str| {
        let _ = ctx;
        false
    };
    match (defines("_exit"), defines("_Exit")) {
        (false, _) => Ok("_exit"),
        (true, false) => Ok("_Exit"),
        (true, true) => Err(MlirLowerError::ShapeError(
            "the program defines functions named both `_exit` and `_Exit`; the integer \
             division trap exits the process through one of these libc symbols"
                .to_string(),
        )),
    }
}

/// The W2-trap trampoline `llvm.func internal @__mind_div_trap()`, plus the
/// declaration of the exit symbol it needs ([`exit_symbol`]). When the program
/// declares that symbol itself (`extern "C" fn _exit`), the declaration (emitted
/// from `extern_c_fns`, whose integer parameter may be widened to `i64`) is
/// reused and the call is typed to match it; a declaration that does not take
/// exactly one integer is refused rather than called with a guessed signature.
fn trampoline(ctx: &LoweringContext) -> Result<String, MlirLowerError> {
    let sym = exit_symbol(ctx)?;
    #[cfg(feature = "std-surface")]
    let user_decl = ctx.extern_c_fns.get(sym);
    #[cfg(not(feature = "std-surface"))]
    let user_decl: Option<&(Vec<String>, Option<String>)> = None;
    let (decl, param, ret) = match user_decl {
        None => (
            format!("  llvm.func @{sym}(i32)\n"),
            "i32".to_string(),
            None,
        ),
        Some(sig) => {
            let (params, ret_type) = (&sig.0, &sig.1);
            let is_int = |t: &str| matches!(t, "i8" | "i16" | "i32" | "i64");
            if params.len() != 1 || !is_int(params[0].as_str()) {
                return Err(MlirLowerError::ShapeError(format!(
                    "the program declares `{sym}` with parameters {params:?}; the integer \
                     division trap calls `{sym}(status)` and needs exactly one integer \
                     parameter"
                )));
            }
            let ret = ret_type
                .as_deref()
                .map(extern_ret_mlir_ty)
                .unwrap_or("i64")
                .to_string();
            (String::new(), params[0].clone(), Some(ret))
        }
    };
    let call = match &ret {
        None => format!("    llvm.call @{sym}(%0) : ({param}) -> ()\n"),
        Some(r) => format!("    %1 = llvm.call @{sym}(%0) : ({param}) -> {r}\n"),
    };
    Ok(format!(
        "{decl}  llvm.func internal @__mind_div_trap() {{\n    \
         %0 = llvm.mlir.constant({MIND_DIV_TRAP_EXIT_CODE} : {param}) : {param}\n\
         {call}    llvm.unreachable\n  }}\n"
    ))
}
