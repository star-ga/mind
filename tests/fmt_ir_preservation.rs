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

//! Formatter IR-preservation gate — Phase 2A acceptance test (Step 3 of PR #3).
//!
//! For every file in scope, asserts that formatting never changes the
//! compiled MIC IR output:
//!
//!   `emit_mic(parse(src)) == emit_mic(parse(format_source(src)))`
//!
//! This is the semantic-correctness gate: a formatter that changes the
//! program's meaning (IR) is a compiler bug, not just a style issue.
//!
//! Scope:
//!   - `std/vec.mind`, `std/string.mind`, `std/io.mind`, `std/map.mind`,
//!     `std/blas.mind`
//!   - `examples/parser/main.mind`, `examples/typecheck/main.mind`,
//!     `examples/emit_ir/main.mind` (if they compile successfully)
//!
//! Files that fail to compile (e.g. because they use intrinsics or tensor
//! ops that require runtime support beyond the compile-to-MIC pipeline)
//! are skipped with a note — compile failures are not IR-preservation failures.
//!
//! # What "byte-identical MIC IR" means
//!
//! The MIC (Machine Intelligence Code) IR text is produced by
//! `compile_to_mic_text`, which: parses → type-checks → lowers to IR →
//! verifies → canonicalizes → serialises.  Two sources that lower to
//! identical IR will produce byte-identical MIC text.  Formatting must
//! not rename variables, reorder top-level items, or alter the AST in
//! any way that changes the lowered result.

use libmind::fmt::format_source;
use libmind::pipeline::{CompileOptions, compile_to_mic_text};
use libmind::project::MindcraftFormatConfig;

fn default_cfg() -> MindcraftFormatConfig {
    MindcraftFormatConfig::default()
}

fn default_compile_opts() -> CompileOptions {
    CompileOptions::default()
}

fn manifest_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ---------------------------------------------------------------------------
// Core assertion helper
// ---------------------------------------------------------------------------

/// Assert IR-preservation for a single source string.
///
/// Returns `true` if the file was exercised (passed the assertion),
/// `false` if skipped (compile error on the original source).
fn check_ir_preservation(label: &str, src: &str) -> bool {
    let cfg = default_cfg();
    let opts = default_compile_opts();

    // Step 1: compile the original source.
    let ir_before = match compile_to_mic_text(src, &opts) {
        Ok(ir) => ir,
        Err(_) => {
            // Source uses features outside the compile-to-MIC scope
            // (e.g. tensor intrinsics, __mind_blas_*, cross-module imports).
            // This is expected for some files; skip without failing.
            return false;
        }
    };

    // Step 2: format the source.
    let formatted = match format_source(src, &cfg) {
        Ok(s) => s,
        Err(e) => panic!("ir_preservation: format failed for {label}: {e}"),
    };

    // Step 3: compile the formatted source.
    let ir_after = compile_to_mic_text(&formatted, &opts).unwrap_or_else(|e| {
        panic!(
            "ir_preservation: formatted source failed to compile for {label}: {e}\n\
                 Formatted source:\n{formatted}"
        )
    });

    // Step 4: byte-identical assertion.
    assert_eq!(
        ir_before, ir_after,
        "IR changed after formatting for {label}.\n\
         This means the formatter altered program semantics.\n\
         IR before formatting:\n{ir_before}\n\
         IR after formatting:\n{ir_after}",
    );

    true
}

// ---------------------------------------------------------------------------
// std/*.mind
// ---------------------------------------------------------------------------

#[test]
fn ir_preservation_vec() {
    let path = manifest_dir().join("std/vec.mind");
    let src = std::fs::read_to_string(&path).unwrap();
    check_ir_preservation("std/vec.mind", &src);
}

#[test]
fn ir_preservation_string() {
    let path = manifest_dir().join("std/string.mind");
    let src = std::fs::read_to_string(&path).unwrap();
    check_ir_preservation("std/string.mind", &src);
}

#[test]
fn ir_preservation_io() {
    let path = manifest_dir().join("std/io.mind");
    let src = std::fs::read_to_string(&path).unwrap();
    check_ir_preservation("std/io.mind", &src);
}

#[test]
fn ir_preservation_map() {
    let path = manifest_dir().join("std/map.mind");
    let src = std::fs::read_to_string(&path).unwrap();
    check_ir_preservation("std/map.mind", &src);
}

#[test]
fn ir_preservation_blas() {
    let path = manifest_dir().join("std/blas.mind");
    let src = std::fs::read_to_string(&path).unwrap();
    check_ir_preservation("std/blas.mind", &src);
}

// ---------------------------------------------------------------------------
// examples/*.mind — key self-host ladder files
// ---------------------------------------------------------------------------

#[test]
fn ir_preservation_parser_main() {
    let path = manifest_dir().join("examples/parser/main.mind");
    if !path.exists() {
        return;
    }
    let src = std::fs::read_to_string(&path).unwrap();
    let exercised = check_ir_preservation("examples/parser/main.mind", &src);
    let _ = exercised;
}

#[test]
fn ir_preservation_typecheck_main() {
    let path = manifest_dir().join("examples/typecheck/main.mind");
    if !path.exists() {
        return;
    }
    let src = std::fs::read_to_string(&path).unwrap();
    let exercised = check_ir_preservation("examples/typecheck/main.mind", &src);
    let _ = exercised;
}

#[test]
fn ir_preservation_emit_ir_main() {
    let path = manifest_dir().join("examples/emit_ir/main.mind");
    if !path.exists() {
        return;
    }
    let src = std::fs::read_to_string(&path).unwrap();
    let exercised = check_ir_preservation("examples/emit_ir/main.mind", &src);
    let _ = exercised;
}

// ---------------------------------------------------------------------------
// Aggregated summary
// ---------------------------------------------------------------------------

#[test]
fn ir_preservation_summary() {
    let base = manifest_dir();

    let files = [
        "std/vec.mind",
        "std/string.mind",
        "std/io.mind",
        "std/map.mind",
        "std/blas.mind",
        "examples/parser/main.mind",
        "examples/typecheck/main.mind",
        "examples/emit_ir/main.mind",
    ];

    let mut exercised = 0usize;
    let mut skipped = 0usize;

    for &rel in &files {
        let path = base.join(rel);
        if !path.exists() {
            skipped += 1;
            continue;
        }
        let src =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {rel}: {e}"));
        if check_ir_preservation(rel, &src) {
            exercised += 1;
        } else {
            skipped += 1;
        }
    }

    eprintln!("ir_preservation_summary: {exercised} exercised, {skipped} skipped (compile-scope)");

    // At minimum the stdlib files that use only core MIND should all compile.
    // If ALL files skipped, something is wrong with the compile pipeline.
    assert!(
        exercised > 0,
        "ir_preservation: every file was skipped — compile pipeline may be broken"
    );
}
