// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mind-native` — compile a `.mind` file straight to a runnable x86-64 ELF with
//! **zero LLVM / MLIR / clang / assembler / linker**, via the experimental
//! native backend (`libmind::native`). Built only with `--features
//! native-backend` (see the `required-features` on the `[[bin]]`).
//!
//! Scope: the scalar-i64 subset the native backend supports (functions, the full
//! integer `BinOp` set, `if`-expressions, `while` loops, recursion). Anything
//! outside it fails loud with a typed `NativeError` — never a broken artifact.
//!
//! ```text
//! mind-native examples/fib.mind /tmp/fib && /tmp/fib ; echo $?   # 55
//! ```

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.mind> <output-elf>", args[0]);
        return ExitCode::from(2);
    }
    let (input, output) = (&args[1], &args[2]);

    let src = match std::fs::read_to_string(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("mind-native: cannot read {input}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let module = match libmind::parser::parse(&src) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mind-native: parse error: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    // Seed the bundled stdlib so `use std.vec` / `use std.string` / … resolve.
    //
    // The native backend lowers EVERY top-level `FnDef` in the IR module into a
    // linkable `Func` and resolves intra-module `call`s by name (`native::link`).
    // A `.mind` file that calls a std free-function — e.g. the self-host
    // compiler's `vec_push(toks, kind)` (examples/mindc_mind/main.mind) — parses
    // and lowers fine on its own, but the callee is DEFINED in `std/vec.mind`,
    // not in the user file, so the linker rejects it as an undefined callee
    // ("native: call to undefined `vec_push`"). The single-file lowering had no
    // definition to link against.
    //
    // The MLIR/project path (`project::compile_project`) already solves this by
    // prepending `project::stdlib::parsed_stdlib_modules()` to the parsed-source
    // set before lowering. We mirror that here: build one combined `ast::Module`
    // whose items are the bundled stdlib items FIRST, then the user file's items
    // LAST, and lower the combined module. User items coming last gives the same
    // last-write-wins shadowing contract as the project path (a user `fn vec_push`
    // overrides the bundled one). Lowering is a pure function of the AST and the
    // std bricks are i64-ABI built only from the inlined `__mind_*` intrinsics, so
    // the std `FnDef`s lower to plain scalar-i64 functions the linker can resolve;
    // any std body using an IR construct outside the native slice still surfaces
    // as the real next blocker rather than being masked.
    // Roots for the dead-code prune below: the names of every function DEFINED
    // in the user file. Captured BEFORE `module.items` is moved into `combined`.
    let user_fn_names: Vec<String> = module
        .items
        .iter()
        .filter_map(|item| match item {
            libmind::ast::Node::FnDef { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();

    let mut combined = libmind::ast::Module::default();
    for (_path, std_module) in libmind::project::stdlib::parsed_stdlib_modules() {
        combined.items.extend(std_module.items);
    }
    combined.items.extend(module.items);

    let mut ir = libmind::eval::lower::lower_to_ir(&combined);

    // Dead-code prune the lowered std functions to the user file's CALL GRAPH.
    //
    // We seed ALL ~21 bundled std modules so any `use std.*` resolves, but most
    // programs touch only a few (the self-host compiler uses vec/string/map/io).
    // Lowering every std `FnDef` would otherwise drag in float/tensor/SIMD bodies
    // (`std.blas`'s `__mind_blas_dot_f32`, the async/reactor kernels, …) that are
    // outside the native backend's scalar-i64 slice — so `compile_to_elf` would
    // fail on an intrinsic the program never actually calls. Pruning to the
    // reachable set keeps those out: an unsupported std body only blocks the
    // build if the user code can actually reach it.
    //
    // Roots = every function DEFINED in the user file (its functions are the
    // program's surface, incl. the library/no-`main` self-host form). BFS over
    // `Instr::Call` callee names (recursing through `If`/`While` sub-streams)
    // pulls in transitively-reachable std functions; everything else is dropped.
    // Pure function of the IR (name-set reachability), so byte-identity holds.
    prune_unreachable_fns(&mut ir, &user_fn_names);

    let ir = ir;

    let elf = match libmind::native::compile_to_elf(&ir) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("mind-native: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = std::fs::write(output, &elf) {
        eprintln!("mind-native: cannot write {output}: {e}");
        return ExitCode::FAILURE;
    }
    // The output is a Linux x86-64 ELF; mark it executable on unix hosts. (The
    // tool still cross-emits on macOS/Windows — the artifact just won't run there,
    // and there is no unix permission bit to set.)
    #[cfg(unix)]
    if let Ok(meta) = std::fs::metadata(output) {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = meta.permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(output, perms);
    }

    let trace: String = libmind::ir::ir_trace_hash(&ir)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    eprintln!(
        "mind-native: wrote {output} ({} bytes, zero LLVM); trace_hash {trace}",
        elf.len()
    );
    ExitCode::SUCCESS
}

use libmind::ir::{IRModule, Instr};

/// Drop every top-level `FnDef` in `ir` that the user file cannot reach through
/// the call graph.
///
/// `ir` is the lowering of the bundled stdlib + the user file combined;
/// `user_fn_names` are the functions defined in the user file (the program
/// surface — works for both a `main` entry and the library/no-`main` self-host
/// form). This keeps only the functions transitively called from those roots.
/// Non-`FnDef` top-level instructions are always retained (they are not callees,
/// and `compile_to_elf` only links `FnDef`s anyway). A pure function of the IR:
/// the same module always prunes to the same set, so the native backend's
/// byte-identity wedge is preserved.
fn prune_unreachable_fns(ir: &mut IRModule, user_fn_names: &[String]) {
    use std::collections::BTreeSet;

    // Roots: the user file's own functions. Anything they (transitively) call
    // must be kept; everything else (unused std bricks) is dropped.
    let mut reachable: BTreeSet<String> = user_fn_names.iter().cloned().collect();

    // Index the lowered FnDef bodies by name so BFS can look up callees' calls.
    let mut bodies: std::collections::HashMap<String, &[Instr]> =
        std::collections::HashMap::new();
    for ins in &ir.instrs {
        if let Instr::FnDef { name, body, .. } = ins {
            bodies.insert(name.clone(), body.as_slice());
        }
    }

    // Worklist BFS over call-graph edges. `worklist` carries the names whose
    // bodies still need their call edges expanded.
    let mut worklist: Vec<String> = reachable.iter().cloned().collect();
    while let Some(name) = worklist.pop() {
        if let Some(body) = bodies.get(name.as_str()) {
            let mut callees = Vec::new();
            collect_call_names(body, &mut callees);
            for callee in callees {
                if reachable.insert(callee.clone()) {
                    worklist.push(callee);
                }
            }
        }
    }

    // Keep non-FnDef instrs and FnDefs whose name is reachable.
    ir.instrs.retain(|ins| match ins {
        Instr::FnDef { name, .. } => reachable.contains(name.as_str()),
        _ => true,
    });
}

/// Collect every `Instr::Call` callee name in `body`, recursing into the
/// sub-instruction streams of `If` / `While` (where nested calls live). `__mind_*`
/// intrinsic callees are gathered too — harmless, since they are never `FnDef`
/// names and the native backend inlines them rather than linking them.
fn collect_call_names(body: &[Instr], out: &mut Vec<String>) {
    for ins in body {
        match ins {
            Instr::Call { name, .. } => out.push(name.clone()),
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                collect_call_names(cond_instrs, out);
                collect_call_names(then_instrs, out);
                collect_call_names(else_instrs, out);
            }
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                collect_call_names(cond_instrs, out);
                collect_call_names(body, out);
            }
            _ => {}
        }
    }
}
