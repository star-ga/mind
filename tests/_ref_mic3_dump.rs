// Committed self-host reference generator (A9b): reconstruct
// emit_mic3(seeded+pruned combined IR) exactly as the deleted `mind-native`
// backend did, for every native_elf fixture + main, EXCLUDING std.http/llvm/mlir
// to match the frozen native_elf oracle (21 modules). Prints one
// `REF <name>: ... note=<64-hex>` line per case (and writes _ref_<name>.note) —
// the SHA-256 of emit_mic3(pruned IR), i.e. the ELF's PT_NOTE trace-hash.
//
// This is the Rust `emit_mic3` reference the native-ELF smoke
// (examples/mindc_mind/self_host_native_elf_smoke.py) DERIVES at test time to
// check the pure-MIND self-computed PT_NOTE against — so the note tracks the live
// std/*.mind content instead of a frozen 32-byte blob that re-stales on any benign
// stdlib growth. `mindc --emit-mic3` cannot substitute: it emits the standard
// pipeline IR, not mind-native's seed-then-prune IR (different id namespace), so
// this helper reproduces the exact seeding + call-graph prune the note is defined on.
//
// Run: cargo test --features "std-surface cross-module-imports" --test _ref_mic3_dump -- --nocapture
#![cfg(feature = "std-surface")]

use std::collections::BTreeSet;

use libmind::ast::Node;
use libmind::ir::{IRModule, Instr};

const FIXTURES: &[(&str, &str)] = &[
    (
        "add",
        "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\nfn main() -> i64 {\n    return add(2, 3);\n}\n",
    ),
    (
        "if_ret",
        "fn f(c: i64) -> i64 {\n    if c == 0 {\n        return 1;\n    }\n    return 2;\n}\nfn main() -> i64 {\n    return f(0);\n}\n",
    ),
    (
        "value_if",
        "fn f(a: i64, b: i64) -> i64 {\n    let m: i64 = if a > b { a } else { b };\n    return m;\n}\nfn main() -> i64 {\n    return f(3, 7);\n}\n",
    ),
    (
        "recursion",
        "fn fib(n: i64) -> i64 {\n    if n < 2 {\n        return n;\n    }\n    return fib(n - 1) + fib(n - 2);\n}\nfn main() -> i64 {\n    return fib(7);\n}\n",
    ),
    (
        "struct_field",
        "struct P {\n    x: i64,\n    y: i64,\n}\nfn main() -> i64 {\n    let p: P = P { x: 7, y: 9 };\n    return p.x;\n}\n",
    ),
];

const KEEP: &[&str] = &[
    "std.arena",
    "std.async",
    "std.blas",
    "std.cli",
    "std.fs",
    "std.io",
    "std.io_canon",
    "std.iouring",
    "std.json",
    "std.map",
    "std.net",
    "std.process",
    "std.reactor",
    "std.regex",
    "std.ring",
    "std.sha256",
    "std.string",
    "std.time",
    "std.toml",
    "std.tui",
    "std.vec",
];

fn collect_call_names(body: &[Instr], out: &mut Vec<String>) {
    for ins in body {
        match ins {
            Instr::Call { name, .. } => out.push(name.clone()),
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

fn prune_unreachable_fns(ir: &mut IRModule, user_fn_names: &[String]) {
    let mut reachable: BTreeSet<String> = user_fn_names.iter().cloned().collect();
    let mut bodies: std::collections::HashMap<String, Vec<Instr>> =
        std::collections::HashMap::new();
    for ins in &ir.instrs {
        if let Instr::FnDef { name, body, .. } = ins {
            bodies.insert(name.clone(), body.clone());
        }
    }
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
    ir.instrs.retain(|ins| match ins {
        Instr::FnDef { name, .. } => reachable.contains(name.as_str()),
        _ => true,
    });
}

/// Bare-name struct schemas of every `StructDef` in the combined module, in the
/// shape the project builder installs into the global registry
/// (`src/project/mod.rs`, `enums.structs`).
///
/// Why the oracle needs this: the 21 std modules are concatenated in
/// `std.X`-alphabetical order, so a std fn can read a field of a struct that a
/// LATER module declares (`std.cli::args_push` reads `s.addr` on `String`,
/// declared in `std.string`). `lower_to_ir` registers a local `StructDef` only
/// when its item is reached, so that read has no schema yet. Lowering used to
/// paper over this with a `ConstI64(0)` placeholder; it now refuses
/// (`UnsupportedLoweringOperation { "struct field read" }`), which made this
/// generator panic before writing any note.
///
/// A real multi-module compile never hits the ordering hazard: the project
/// builder seeds every module's structs into the global registry and
/// `lower_to_ir` merges them into `ir.struct_defs` before the item walk. The
/// oracle seeds the same registry with the same bare names and field lists.
/// Only BARE names are seeded (no `qualified_structs`), so `ir.struct_defs`, the
/// one serialized struct registry, holds exactly the keys and field-name lists
/// the local `StructDef` arms insert. Fn bodies have their own id space, so the
/// module-level instr stream and `next_id` are untouched. The trace hash
/// therefore stays a function of the same pruned module.
fn combined_struct_registry(combined: &libmind::ast::Module) -> libmind::ir::GlobalEnums {
    let mut registry = libmind::ir::GlobalEnums::default();
    for item in &combined.items {
        if let Node::StructDef { name, fields, .. } = item {
            registry.structs.insert(
                name.clone(),
                (
                    fields.iter().map(|f| f.name.clone()).collect(),
                    fields.iter().map(|f| f.ty.clone()).collect(),
                ),
            );
        }
    }
    registry
}

/// Unpruned seeded-combined IR (std KEEP set ahead of the user source), or the
/// lowering refusal. Split out so the regression test can assert on it.
fn try_combined_ir(
    user_src: &str,
) -> Result<(IRModule, Vec<String>), libmind::eval::materialization::MaterializationRefusal> {
    let module = libmind::parser::parse(user_src).expect("parse user");
    let user_fn_names: Vec<String> = module
        .items
        .iter()
        .filter_map(|item| match item {
            Node::FnDef(fd, _) => Some(fd.name.clone()),
            _ => None,
        })
        .collect();

    let mut combined = libmind::ast::Module::default();
    let keep: BTreeSet<&str> = KEEP.iter().copied().collect();
    for (path, std_module) in libmind::project::stdlib::parsed_stdlib_modules() {
        if keep.contains(path.as_str()) {
            combined.items.extend(std_module.items);
        }
    }
    combined.items.extend(module.items);

    libmind::ir::set_global_enums(combined_struct_registry(&combined));
    let lowered = libmind::eval::lower::lower_to_ir(&combined);
    libmind::ir::clear_global_enums();
    lowered.map(|ir| (ir, user_fn_names))
}

fn ref_ir(user_src: &str) -> IRModule {
    let (mut ir, user_fn_names) = try_combined_ir(user_src).expect("lowering");
    prune_unreachable_fns(&mut ir, &user_fn_names);
    ir
}

fn note_of(ir: &IRModule) -> String {
    libmind::ir::ir_trace_hash(ir)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// Manual reference-regeneration utility, not a CI assertion: it writes
// `_ref_*.note` fixtures next to the smoke and makes no assertions, so it is
// `#[ignore]`d and run explicitly with `cargo test dump_ref -- --ignored` when
// the mic@3 references are refreshed.
//
// The output directory is resolved from THIS checkout, never a fixed absolute
// path: `$MIND_REF_OUT_DIR` if set, else `$CARGO_MANIFEST_DIR/examples/mindc_mind`
// (CARGO_MANIFEST_DIR is the crate root of the tree being built). For the default
// main-repo build this is `/home/n/mind/examples/mindc_mind` — byte-identical to
// the old hardcoded path — but a git-worktree / CI checkout now regenerates into
// ITS OWN tree instead of clobbering the main repo's committed references.
#[test]
#[ignore = "manual mic@3 reference-dump utility; writes _ref_*.note into this checkout, run with --ignored"]
fn dump_ref() {
    let out_dir = std::env::var("MIND_REF_OUT_DIR")
        .unwrap_or_else(|_| format!("{}/examples/mindc_mind", env!("CARGO_MANIFEST_DIR")));
    for (name, src) in FIXTURES {
        let ir = ref_ir(src);
        let bytes = libmind::ir::compact::emit_mic3(&ir);
        let note = note_of(&ir);
        if let Ok(dir) = std::env::var("MIND_MIC3_BYTES_DIR") {
            std::fs::write(format!("{dir}/rust_{name}.mic3"), &bytes).unwrap();
        }
        std::fs::write(format!("{out_dir}/_ref_{name}.note"), &note).unwrap();
        eprintln!(
            "REF {name}: {} B  next_id={}  instrs={}  note={note}",
            bytes.len(),
            ir.next_id,
            ir.instrs.len()
        );
    }
    // main.mind
    let main_src = std::fs::read_to_string(format!("{out_dir}/main.mind")).unwrap();
    let ir = ref_ir(&main_src);
    let bytes = libmind::ir::compact::emit_mic3(&ir);
    let note = note_of(&ir);
    std::fs::write(format!("{out_dir}/_ref_main.note"), &note).unwrap();
    eprintln!(
        "REF main: {} B  next_id={}  instrs={}  note={note}",
        bytes.len(),
        ir.next_id,
        ir.instrs.len()
    );
}

// Regression gate for the oracle itself (runs in normal `cargo test`, writes
// nothing). The concatenated std KEEP set puts `std.cli` (whose `args_push`
// reads `s.addr` on `String`) ahead of `std.string` (which declares `String`).
// Before the registry seed, lowering refused that read and `dump_ref` panicked
// with `UnsupportedLoweringOperation { operation: "struct field read" }`, so the
// pinned `_ref_*.note` cache could not be refreshed at all.
#[test]
fn dump_ref_lowers_seeded_std_cli_before_std_string() {
    let (name, src) = FIXTURES[0];
    assert_eq!(name, "add");
    let (ir, user_fn_names) = match try_combined_ir(src) {
        Ok(lowered) => lowered,
        Err(refusal) => panic!("seeded std + `{name}` must lower, got {refusal:?}"),
    };
    // Positive control: std.cli really was lowered (the reader that used to be
    // refused is present as a FnDef), so the green above is not a vacuous pass
    // over a KEEP set that silently dropped std.cli.
    assert!(
        ir.instrs
            .iter()
            .any(|i| matches!(i, Instr::FnDef { name, .. } if name == "args_push")),
        "std.cli::args_push must be present in the unpruned combined IR"
    );
    // The seed must not leak qualified keys into the serialized struct registry.
    assert!(ir.struct_defs.contains_key("String"));
    assert!(
        ir.struct_defs.keys().all(|k| !k.contains('.')),
        "only bare struct names may reach ir.struct_defs: {:?}",
        ir.struct_defs.keys().collect::<Vec<_>>()
    );
    // The registry is transient: a later single-file compile must see it empty.
    assert!(libmind::ir::with_global_enums(|g| g.structs.is_empty()));

    let mut pruned = ir;
    prune_unreachable_fns(&mut pruned, &user_fn_names);
    let bytes = libmind::ir::compact::emit_mic3(&pruned);
    assert!(bytes.len() > 4 && bytes.starts_with(b"MIC3"));
    assert_eq!(note_of(&pruned).len(), 64);
}
