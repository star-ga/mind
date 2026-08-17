// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! IR-audit finding #2 (task #270) — the `NARROW_LOCALS` thread-local must
//! not leak narrow-width type entries ACROSS modules on a shared thread, and a
//! BLOCK-local narrow `let` must not clobber an outer same-named i64 binding
//! after the block ends. Both are silent miscompiles: without the fix, a later
//! i64 `let c = 300; c = c + 1` gets re-masked to 8 bits (301 -> 45), and an
//! outer i64 `c = 300` after a block that declared a `let c: u8` becomes 44.
//!
//! Two harnesses, because the two facets manifest at different scopes:
//!  * Cross-module leak — thread-local state that survives between two
//!    `lower_to_ir` calls on the SAME thread. A subprocess `mindc` compile
//!    starts a fresh thread-local and cannot reproduce it, so this facet is
//!    proven IN-PROCESS: lowering module B must be byte-identical whether or
//!    not module A (with a top-level `let c: u8 = 200`) was lowered first.
//!  * Block sub-case — reproduces within a SINGLE module compile (one thread),
//!    so it is proven by compiling to a `.so` and asserting the runtime result
//!    is 300 (not the masked 44).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test narrow_locals_leak_run`

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::compact::emit_mic3;
use libmind::parser::parse;

/// Cross-module leak: module A leaves a `c -> u8` entry; module B's i64 `c`
/// must lower IDENTICALLY whether or not A ran first on the same thread. The
/// reset added to `lower_to_ir` makes B independent of A; without it, B's
/// `c = c + 1` picks up A's stale narrow mask (301 -> 45), perturbing the IR.
#[test]
fn narrow_locals_do_not_leak_across_modules() {
    // Module A leaves a top-level `c -> u8` entry in NARROW_LOCALS. A top-level
    // narrow `let` records at MODULE scope (the item loop), and — unlike code
    // inside a fn body, which `enter_narrow_scope`'s take() would clear on fn
    // entry — nothing re-scopes the module level. So B's re-lowering must be
    // driven by its OWN module-scope reassignment `c = c + 1` (item-loop Assign
    // arm), which is exactly the path that consults the leaked entry and would
    // re-mask 301 -> 45.
    let module_scope_a = "let c: u8 = 200\npub fn main() -> i64 {\n    return 0\n}\n";
    let src_b = "let c = 300\nc = c + 1\npub fn main() -> i64 {\n    return c\n}\n";

    // Lower B FIRST on this fresh thread — thread-local starts empty, so this is
    // the true clean baseline.
    let b_clean = emit_mic3(&lower_to_ir(&parse(src_b).expect("parse B")));

    // Pollute the thread-local with A's top-level narrow `let`, then lower B.
    let _ = lower_to_ir(&parse(module_scope_a).expect("parse A(module-scope)"));
    let b_after_a = emit_mic3(&lower_to_ir(&parse(src_b).expect("parse B again")));
    assert_eq!(
        b_clean, b_after_a,
        "module B's i64 `let c` lowered differently after module A leaked a \
         `c -> u8` narrow entry (module-scope) — NARROW_LOCALS leak"
    );
}

// ── Block sub-case: single-module compile+run (reproduces on one thread). ──

#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
mod block_run {
    use std::process::Command;

    fn mindc_bin() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_BIN_EXE_mindc"))
    }

    /// Compile `pub fn run() -> i64 { <body> }` to a `.so` and call it.
    fn run_body(body: &str, tag: &str) -> i64 {
        let mindc = mindc_bin();
        let dir = std::env::temp_dir();
        let s = dir.join(format!("mind_nleak_{tag}.mind"));
        let so = dir.join(format!("mind_nleak_{tag}.so"));
        let src = format!("pub fn run() -> i64 {{\n{body}\n}}\n");
        std::fs::write(&s, src).expect("write");
        let out = Command::new(&mindc)
            .args([s.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
            .output()
            .expect("run mindc");
        assert!(
            out.status.success(),
            "narrow-leak block compile failed ({tag}):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let py = format!(
            "import ctypes\nlib=ctypes.CDLL(r'{}')\nlib.run.restype=ctypes.c_int64\n\
             print(lib.run())\n",
            so.to_string_lossy()
        );
        let out = Command::new("python3")
            .args(["-c", &py])
            .output()
            .expect("py");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .unwrap_or(i64::MIN)
    }

    #[test]
    fn block_local_narrow_let_does_not_mask_outer_i64() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("narrow-leak block: mindc not found; skipping");
            return;
        }
        // probe for mlir-build availability
        {
            let dir = std::env::temp_dir();
            let s = dir.join("mind_nleak_probe.mind");
            std::fs::write(&s, "pub fn run() -> i64 { return 0 }\n").unwrap();
            let o = Command::new(&mindc)
                .args([
                    s.to_str().unwrap(),
                    "--emit-shared",
                    dir.join("p_nleak.so").to_str().unwrap(),
                ])
                .output()
                .unwrap();
            let e = String::from_utf8_lossy(&o.stderr);
            if e.contains("mlir-build") && e.contains("requires") {
                println!("narrow-leak block: needs mlir-build; skipping");
                return;
            }
        }

        // Outer i64 `c`; a block-valued match arm declares a same-named narrow
        // `let c: u8` (routes through the value-`Block` lowering arm); the outer
        // `c = 300` after the block must NOT be masked to 8 bits (44).
        assert_eq!(
            run_body(
                "    let c: i64 = 0\n    let _b = match c {\n        _ => {\n            let c: u8 = 5\n            c\n        }\n    }\n    c = 300\n    return c",
                "block"
            ),
            300
        );

        // Control: a genuinely narrow outer `let c: u8` still re-masks a later
        // reassignment (the block scoping must not disable real masking).
        assert_eq!(
            run_body("    let c: u8 = 200\n    c = c + 100\n    return c", "ctrl"),
            44
        );
    }
}
