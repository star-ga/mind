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

fn ref_ir(user_src: &str) -> IRModule {
    let module = libmind::parser::parse(user_src).expect("parse user");
    let user_fn_names: Vec<String> = module
        .items
        .iter()
        .filter_map(|item| match item {
            Node::FnDef { name, .. } => Some(name.clone()),
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

    let mut ir = libmind::eval::lower::lower_to_ir(&combined);
    prune_unreachable_fns(&mut ir, &user_fn_names);
    ir
}

fn note_of(ir: &IRModule) -> String {
    libmind::ir::ir_trace_hash(ir)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn dump_ref() {
    let out_dir = "/home/n/mind/examples/mindc_mind";
    for (name, src) in FIXTURES {
        let ir = ref_ir(src);
        let bytes = libmind::ir::compact::emit_mic3(&ir);
        let note = note_of(&ir);
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
