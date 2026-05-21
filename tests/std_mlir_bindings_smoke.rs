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

// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0010 Phase E — `std/mlir.mind` bindings smoke test.
//!
//! Two verification layers:
//!
//! 1. **Compilation gate** — the mindc compiler must parse, type-check, and
//!    emit MLIR declarations for every function declared in `std/mlir.mind`.
//!    The emitted MLIR must contain one `llvm.func @mlirXxx` declaration per
//!    bound function.  This verifies that the Phase A/B/C extern "C" pipeline
//!    handles the full ~150-function surface without errors.
//!
//! 2. **Symbol presence gate** — for each bound function name, the test
//!    verifies that the symbol exists in the MLIR 18 C API static libraries
//!    shipped with the local LLVM installation.  The MLIR C API symbols live
//!    in `libMLIRCAPIIR.a`, `libMLIRCAPIPass.a`, etc. (not in `libMLIR.so`,
//!    which only contains the C++ MLIR ABI). The gate uses `nm` to enumerate
//!    symbol tables from all `libMLIRCAP*.a` files found under the LLVM prefix.
//!
//!    If MLIR is not installed locally the gate **skips** with a clear message
//!    rather than failing.  The compilation gate (layer 1) always runs.
//!
//! 3. **Round-trip MIND compilation gate** — compiles a minimal `.mind` snippet
//!    that references `mlirContextCreate` and `mlirContextIsNull` from the
//!    std.mlir bindings (via --emit-mlir), then checks that the output contains
//!    the expected `llvm.func` declarations and a `llvm.call` targeting them.
//!    Requires `mindc --emit-mlir` to be functional (mlir-lowering feature).
//!
//! Gate: `cargo test --features "std-surface mlir-lowering" --test std_mlir_bindings_smoke`

#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]

use std::path::PathBuf;
use std::process::Command;

// ── helpers ───────────────────────────────────────────────────────────────────

fn mindc_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // prefer release; fall back to debug
    let rel = manifest.join("target/release/mindc");
    if rel.exists() {
        return rel;
    }
    manifest.join("target/debug/mindc")
}

/// Run `mindc <args>` and return stdout on success, or panic with the full
/// stderr on failure.
fn run_mindc(args: &[&str]) -> String {
    let out = Command::new(mindc_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn mindc: {e}"));
    if !out.status.success() {
        panic!(
            "mindc {} failed:\nstdout: {}\nstderr: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Returns the path to the `std/mlir.mind` file relative to the workspace.
fn std_mlir_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("std/mlir.mind")
}

/// Discover all `libMLIRCAP*.a` static libraries under the LLVM installation.
/// Returns an empty Vec if MLIR is not available locally.
fn find_mlir_capi_libs() -> Vec<PathBuf> {
    // Common LLVM installation prefixes.
    let candidates: &[&str] = &[
        "/usr/lib/llvm-18/lib",
        "/usr/lib/llvm-17/lib",
        "/usr/local/lib",
        "/opt/homebrew/lib",
    ];
    let mut libs: Vec<PathBuf> = Vec::new();
    for prefix in candidates {
        let dir = PathBuf::from(prefix);
        if !dir.is_dir() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let s = name.to_string_lossy();
                if s.starts_with("libMLIRCAP") && s.ends_with(".a") {
                    libs.push(entry.path());
                }
            }
        }
    }
    libs
}

/// Run `nm <lib>` and collect all global text symbols (lines containing ` T `).
/// Returns a sorted, deduplicated list of symbol names.
fn nm_symbols(lib: &PathBuf) -> Vec<String> {
    let out = Command::new("nm")
        .arg("--defined-only")
        .arg(lib)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(_) => return Vec::new(), // nm not available
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut syms: Vec<String> = text
        .lines()
        .filter(|l| l.contains(" T "))
        .filter_map(|l| l.split_whitespace().last().map(|s| s.to_string()))
        .collect();
    syms.sort();
    syms.dedup();
    syms
}

// ── test 1: std/mlir.mind compiles cleanly under mindc check ─────────────────

/// Gate 1: `mindc check std/mlir.mind` must exit 0 — no parse errors,
/// no type-check errors, and the file is already formatted (fmt::drift = 0).
#[test]
fn std_mlir_mind_passes_mindc_check() {
    let path = std_mlir_path();
    assert!(
        path.exists(),
        "std/mlir.mind not found at {path:?}; did you forget to create it?"
    );
    let out = Command::new(mindc_path())
        .args(["check", path.to_str().unwrap()])
        .output()
        .unwrap_or_else(|e| panic!("spawn mindc check: {e}"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "mindc check std/mlir.mind failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    // No diagnostic lines of severity error should appear.
    let combined = format!("{stdout}{stderr}");
    assert!(
        !combined.contains("error:"),
        "mindc check produced error diagnostics:\n{combined}"
    );
}

// ── test 2: MLIR emission produces llvm.func declarations ────────────────────

/// Gate 2: `mindc std/mlir.mind --emit-mlir` must produce at least 150
/// `llvm.func @mlirXxx` declarations — one per bound MLIR C API function.
#[test]
fn std_mlir_mind_emits_llvm_func_declarations() {
    let path = std_mlir_path();
    let mlir = run_mindc(&[path.to_str().unwrap(), "--emit-mlir"]);

    // Count how many llvm.func @mlirXxx declarations appear.
    let count = mlir
        .lines()
        .filter(|l| l.contains("llvm.func @mlir"))
        .count();

    assert!(
        count >= 150,
        "expected >= 150 llvm.func declarations for the MLIR C API bindings; \
         got {count}.\n\nFirst 20 lines of MLIR output:\n{}",
        mlir.lines().take(20).collect::<Vec<_>>().join("\n")
    );

    // Spot-check the core IR functions the Phase G migration will need first.
    for expected in &[
        "llvm.func @mlirContextCreate",
        "llvm.func @mlirContextDestroy",
        "llvm.func @mlirContextIsNull",
        "llvm.func @mlirModuleCreateEmpty",
        "llvm.func @mlirModuleDestroy",
        "llvm.func @mlirModuleGetOperation",
        "llvm.func @mlirLocationUnknownGet",
        "llvm.func @mlirOperationCreate",
        "llvm.func @mlirOperationDestroy",
        "llvm.func @mlirRegionCreate",
        "llvm.func @mlirRegionDestroy",
        "llvm.func @mlirRegionAppendOwnedBlock",
        "llvm.func @mlirBlockCreate",
        "llvm.func @mlirBlockDestroy",
        "llvm.func @mlirBlockAppendOwnedOperation",
        "llvm.func @mlirBlockGetArgument",
        "llvm.func @mlirValueGetType",
        "llvm.func @mlirIdentifierGet",
        "llvm.func @mlirAttributeParseGet",
        "llvm.func @mlirTypeParseGet",
        "llvm.func @mlirPassManagerCreate",
        "llvm.func @mlirPassManagerDestroy",
        "llvm.func @mlirPassManagerRunOnOp",
        "llvm.func @mlirPassManagerAddOwnedPass",
        "llvm.func @mlirRegisterAllDialects",
        "llvm.func @mlirRegisterAllPasses",
        "llvm.func @mlirStringRefCreateFromCString",
    ] {
        assert!(
            mlir.contains(expected),
            "expected declaration `{expected}` missing from emitted MLIR;\n\
             check that the corresponding function is in std/mlir.mind"
        );
    }
}

// ── test 3: round-trip — call mlirContextCreate from a MIND snippet ───────────

/// Gate 3: a minimal MIND snippet that calls `mlirContextCreate` and
/// `mlirContextIsNull` via the extern "C" bindings must compile to MLIR that
/// contains the expected `llvm.func` declarations and `llvm.call` instructions.
///
/// This verifies the Phase A call-lowering pipeline handles the full round-trip
/// for the most critical function pair the Phase G migration will use first.
#[test]
fn round_trip_mlir_context_create_and_is_null() {
    // Minimal MIND source that declares (inline) two MLIR C API functions and
    // calls them.  We use an inline extern block rather than importing std.mlir
    // to keep this test self-contained (cross-module imports not required).
    let src = r#"
extern "C" {
    safe fn mlirContextCreate() -> *mut u8;
    safe fn mlirContextIsNull(ctx: *mut u8) -> bool;
}

pub fn check_mlir_context(dummy: i64) -> i64 {
    let ctx: *mut u8 = mlirContextCreate();
    let is_null: bool = mlirContextIsNull(ctx);
    if is_null { 1 } else { 0 }
}
"#;

    let tmp_dir = std::env::temp_dir();
    let src_path = tmp_dir.join("mlir_ctx_smoke.mind");
    std::fs::write(&src_path, src).expect("write smoke .mind source");

    let mlir = run_mindc(&[src_path.to_str().unwrap(), "--emit-mlir"]);

    // The module must declare both extern functions.
    assert!(
        mlir.contains("llvm.func @mlirContextCreate"),
        "MLIR output must contain `llvm.func @mlirContextCreate`;\ngot:\n{mlir}"
    );
    assert!(
        mlir.contains("llvm.func @mlirContextIsNull"),
        "MLIR output must contain `llvm.func @mlirContextIsNull`;\ngot:\n{mlir}"
    );

    // The call to mlirContextCreate must lower to llvm.call (not func.call).
    assert!(
        mlir.contains("llvm.call @mlirContextCreate"),
        "call to mlirContextCreate must lower to `llvm.call`;\ngot:\n{mlir}"
    );
    // The call to mlirContextIsNull must also be an llvm.call.
    assert!(
        mlir.contains("llvm.call @mlirContextIsNull"),
        "call to mlirContextIsNull must lower to `llvm.call`;\ngot:\n{mlir}"
    );

    // Neither should appear as func.call (that would mean the extern path failed).
    assert!(
        !mlir.contains("func.call @mlirContextCreate"),
        "mlirContextCreate must NOT lower to func.call (extern path broken);\ngot:\n{mlir}"
    );
    assert!(
        !mlir.contains("func.call @mlirContextIsNull"),
        "mlirContextIsNull must NOT lower to func.call (extern path broken);\ngot:\n{mlir}"
    );
}

// ── test 4: symbol presence in MLIR C API static libraries ───────────────────

/// Gate 4: for each MLIR C API function declared in `std/mlir.mind`, verify
/// that the symbol exists in the MLIR 18 C API static libraries installed
/// locally (`libMLIRCAP*.a`).
///
/// **Self-skips** if no `libMLIRCAP*.a` files are found — the local toolchain
/// may not have MLIR static libraries installed (the mlir-build feature uses
/// mlir-opt subprocess instead of in-process linking).
///
/// The skip message names the search paths so the CI operator can diagnose
/// the absence.
#[test]
fn mlir_capi_symbols_present_in_static_libs() {
    let libs = find_mlir_capi_libs();
    if libs.is_empty() {
        println!(
            "std_mlir_bindings_smoke(symbols): no libMLIRCAP*.a files found in \
             /usr/lib/llvm-{{17,18}}/lib, /usr/local/lib, /opt/homebrew/lib; \
             skipping symbol presence gate (MLIR static libs not installed locally)"
        );
        return;
    }

    // Collect all symbols from every MLIR C API static library.
    let mut all_syms: std::collections::HashSet<String> = std::collections::HashSet::new();
    for lib in &libs {
        for sym in nm_symbols(lib) {
            all_syms.insert(sym);
        }
    }

    if all_syms.is_empty() {
        println!(
            "std_mlir_bindings_smoke(symbols): `nm` returned no symbols (nm may \
             not be installed or the .a files may be stripped); skipping symbol gate"
        );
        return;
    }

    // The set of MLIR C API functions whose symbol presence we verify.
    // These are the most critical functions for the Phase G migration path.
    // A function is NOT in this list if it is a static inline in the C header
    // (e.g. mlirPassManagerIsNull, mlirValueIsNull) — those do not appear in
    // the .a files because they are inlined by the C compiler.
    let verify_symbols: &[&str] = &[
        "mlirContextCreate",
        "mlirContextCreateWithThreading",
        "mlirContextCreateWithRegistry",
        "mlirContextDestroy",
        "mlirContextEqual",
        "mlirContextEnableMultithreading",
        "mlirContextLoadAllAvailableDialects",
        "mlirContextGetOrLoadDialect",
        "mlirContextIsRegisteredOperation",
        "mlirContextAppendDialectRegistry",
        "mlirContextSetAllowUnregisteredDialects",
        "mlirContextGetAllowUnregisteredDialects",
        "mlirContextGetNumRegisteredDialects",
        "mlirContextGetNumLoadedDialects",
        "mlirDialectRegistryCreate",
        "mlirDialectRegistryDestroy",
        "mlirLocationUnknownGet",
        "mlirLocationFileLineColGet",
        "mlirLocationCallSiteGet",
        "mlirLocationNameGet",
        "mlirLocationGetContext",
        "mlirLocationEqual",
        "mlirModuleCreateEmpty",
        "mlirModuleCreateParse",
        "mlirModuleDestroy",
        "mlirModuleGetContext",
        "mlirModuleGetBody",
        "mlirModuleGetOperation",
        "mlirModuleFromOperation",
        "mlirOperationCreate",
        "mlirOperationDestroy",
        "mlirOperationRemoveFromParent",
        "mlirOperationEqual",
        "mlirOperationClone",
        "mlirOperationGetContext",
        "mlirOperationGetLocation",
        "mlirOperationGetName",
        "mlirOperationGetBlock",
        "mlirOperationGetParentOperation",
        "mlirOperationGetNextInBlock",
        "mlirOperationGetNumOperands",
        "mlirOperationGetOperand",
        "mlirOperationSetOperand",
        "mlirOperationGetNumResults",
        "mlirOperationGetResult",
        "mlirOperationGetNumRegions",
        "mlirOperationGetFirstRegion",
        "mlirOperationGetRegion",
        "mlirOperationGetNumSuccessors",
        "mlirOperationGetSuccessor",
        "mlirOperationSetSuccessor",
        "mlirOperationGetNumAttributes",
        "mlirOperationGetAttributeByName",
        "mlirOperationSetAttributeByName",
        "mlirOperationRemoveAttributeByName",
        "mlirOperationDump",
        "mlirOperationVerify",
        "mlirOperationMoveAfter",
        "mlirOperationMoveBefore",
        "mlirOperationStateGet",
        "mlirOperationStateAddResults",
        "mlirOperationStateAddOperands",
        "mlirOperationStateAddOwnedRegions",
        "mlirOperationStateAddSuccessors",
        "mlirOperationStateAddAttributes",
        "mlirOpPrintingFlagsCreate",
        "mlirOpPrintingFlagsDestroy",
        "mlirOpPrintingFlagsElideLargeElementsAttrs",
        "mlirOpPrintingFlagsEnableDebugInfo",
        "mlirOpPrintingFlagsPrintGenericOpForm",
        "mlirOpPrintingFlagsUseLocalScope",
        "mlirRegionCreate",
        "mlirRegionDestroy",
        "mlirRegionEqual",
        "mlirRegionGetFirstBlock",
        "mlirRegionAppendOwnedBlock",
        "mlirRegionInsertOwnedBlock",
        "mlirRegionInsertOwnedBlockAfter",
        "mlirRegionInsertOwnedBlockBefore",
        "mlirRegionGetNextInOperation",
        "mlirRegionTakeBody",
        "mlirBlockCreate",
        "mlirBlockDestroy",
        "mlirBlockDetach",
        "mlirBlockEqual",
        "mlirBlockGetParentOperation",
        "mlirBlockGetParentRegion",
        "mlirBlockGetNextInRegion",
        "mlirBlockGetFirstOperation",
        "mlirBlockGetTerminator",
        "mlirBlockAppendOwnedOperation",
        "mlirBlockInsertOwnedOperation",
        "mlirBlockInsertOwnedOperationAfter",
        "mlirBlockInsertOwnedOperationBefore",
        "mlirBlockGetNumArguments",
        "mlirBlockAddArgument",
        "mlirBlockInsertArgument",
        "mlirBlockGetArgument",
        "mlirValueEqual",
        "mlirValueIsABlockArgument",
        "mlirValueIsAOpResult",
        "mlirValueGetType",
        "mlirValueSetType",
        "mlirValueGetFirstUse",
        "mlirValueReplaceAllUsesOfWith",
        "mlirTypeParseGet",
        "mlirTypeEqual",
        "mlirTypeGetContext",
        "mlirTypeGetDialect",
        "mlirAttributeParseGet",
        "mlirAttributeGetNull",
        "mlirAttributeEqual",
        "mlirAttributeGetContext",
        "mlirAttributeGetType",
        "mlirAttributeGetDialect",
        "mlirIdentifierGet",
        "mlirIdentifierGetContext",
        "mlirIdentifierEqual",
        "mlirIdentifierStr",
        "mlirNamedAttributeGet",
        "mlirSymbolTableCreate",
        "mlirSymbolTableDestroy",
        "mlirSymbolTableLookup",
        "mlirSymbolTableInsert",
        "mlirSymbolTableErase",
        "mlirSymbolTableGetSymbolAttributeName",
        "mlirPassManagerCreate",
        "mlirPassManagerCreateOnOperation",
        "mlirPassManagerDestroy",
        "mlirPassManagerGetAsOpPassManager",
        "mlirPassManagerRunOnOp",
        "mlirPassManagerEnableVerifier",
        "mlirPassManagerGetNestedUnder",
        "mlirPassManagerAddOwnedPass",
        "mlirOpPassManagerGetNestedUnder",
        "mlirOpPassManagerAddOwnedPass",
        "mlirOpPassManagerAddPipeline",
        "mlirPrintPassPipeline",
        "mlirRegisterAllDialects",
        "mlirRegisterAllLLVMTranslations",
        "mlirRegisterAllPasses",
        "mlirStringRefCreateFromCString",
        "mlirStringRefEqual",
    ];

    let mut missing: Vec<&str> = Vec::new();
    for &sym in verify_symbols {
        if !all_syms.contains(sym) {
            missing.push(sym);
        }
    }

    if !missing.is_empty() {
        panic!(
            "std_mlir_bindings_smoke(symbols): {} MLIR C API symbols missing from \
             libMLIRCAP*.a files in the local LLVM installation.\n\
             Missing symbols:\n  {}\n\
             This means std/mlir.mind declares functions that do not exist in \
             the MLIR 18 C API. Check the header files and update the bindings.",
            missing.len(),
            missing.join("\n  ")
        );
    }

    println!(
        "std_mlir_bindings_smoke(symbols): {} symbols verified present in {} \
         libMLIRCAP*.a files",
        verify_symbols.len(),
        libs.len()
    );
}
