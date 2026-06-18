// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! High-level compiler pipeline utilities for the public MIND front-end.
//!
//! The helpers in this module parse, type-check, and lower MIND source code
//! into the public IR. They optionally run the static autodiff pipeline and
//! can lower the resulting IR into MLIR text when the `mlir-lowering` feature
//! is enabled.

use std::collections::HashMap;

use crate::diagnostics::Diagnostic;
use crate::eval;
use crate::ir;
use crate::opt;
use crate::parser;
use crate::runtime::types::BackendTarget;
use crate::type_checker;

#[cfg(feature = "autodiff")]
use crate::autodiff;
#[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
use crate::mlir;

/// RFC 0002 D3 guard — cap the number of manifest-declared exports to a
/// generous-but-bounded value. Larger lists are rejected as `Mind.toml`
/// content is user-controlled (and on CI may be PR-controlled). The cap
/// is well above any realistic project's public surface; pick a higher
/// value if a legitimate use case appears.
const MAX_MANIFEST_EXPORTS: usize = 1024;

/// RFC 0002 D3 guard — validate a manifest export name is a C-style
/// identifier (`[A-Za-z_][A-Za-z0-9_]*`). Names land directly in the
/// C ABI wrapper symbol once D2 lands, so anything outside that grammar
/// is a future symbol-injection vector.
fn validate_manifest_export_name(name: &str) -> Result<(), &'static str> {
    if name.is_empty() {
        return Err("empty export name");
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err("export name must start with a letter or underscore");
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return Err("export name must be ASCII alphanumeric or underscore");
        }
    }
    Ok(())
}

/// Options controlling the compiler pipeline.
#[derive(Debug, Default, Clone)]
pub struct CompileOptions {
    /// Optional function name to focus on (used by autodiff and MLIR).
    pub func: Option<String>,
    /// Whether to run autodiff for the selected function.
    pub enable_autodiff: bool,
    /// Requested execution backend (CPU is the only supported target).
    pub target: BackendTarget,
    /// RFC 0002 deliverable 3 — extra export names from
    /// `Mind.toml [exports] c_abi`. Merged into `IRModule.exports`
    /// after AST → IR lowering, alongside any in-source `export { ... }`
    /// block. Default empty; only the build pipeline populates this.
    pub manifest_exports: Vec<String>,
    /// RFC 0002 deliverable 5 — language profile (`default` /
    /// `systems` / `embedded`). Reaches the cache-key fingerprint so
    /// the same `Mind.toml` produces a distinct artifact per profile.
    /// Default = `ProfileTag::Default`.
    pub profile: crate::cache::ProfileTag,
}

/// Artifacts produced by [`compile_source`].
#[derive(Debug, Clone)]
pub struct CompileProducts {
    /// Verified and canonicalized IR for the input source.
    pub ir: ir::IRModule,
    /// Gradient IR when autodiff is enabled.
    #[cfg(feature = "autodiff")]
    pub grad: Option<autodiff::GradientResult>,
    /// Constructs that parse + type-check but cannot lower to a *correct*
    /// runnable artifact in the shipped i64-scalar ABI (release-readiness
    /// P1.1). Empty for an all-i64 program. The runnable-artifact emit paths
    /// (`--emit-obj` / `--emit-shared`) fail loud on these; the inspection
    /// paths (`--emit-ir` / `--emit-mlir` / `mindc check`) ignore them because
    /// the constructs are valid *types*, just not yet lowerable.
    pub runnable_blockers: Vec<Diagnostic>,
}

/// Errors surfaced by the high-level compilation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// Parsing failed with one or more diagnostics.
    #[error("parse error")]
    ParseError(Vec<Diagnostic>),
    /// Type checking failed with one or more diagnostics.
    #[error("type error")]
    TypeError(Vec<Diagnostic>),
    /// The IR module did not pass verification.
    #[error("IR verification failed: {0}")]
    IrVerify(#[from] ir::IrVerifyError),
    /// Autodiff requested without specifying a function.
    #[error("autodiff requested but no function was provided")]
    MissingFunctionName,
    /// Requested backend target is not available in this build.
    #[error("backend unavailable: {target}")]
    BackendUnavailable { target: BackendTarget },
    /// `Mind.toml [exports] c_abi` contains an invalid entry (DoS / symbol
    /// injection guard: bound the list to MAX_MANIFEST_EXPORTS and reject
    /// strings that aren't C-style identifiers).
    #[error("invalid manifest export `{name}`: {reason}")]
    InvalidManifestExport { name: String, reason: &'static str },
    /// Autodiff failed with a structured error.
    #[cfg(feature = "autodiff")]
    #[error("autodiff failed: {0}")]
    Autodiff(#[from] autodiff::AutodiffError),
    /// Autodiff was requested but the feature is not enabled.
    #[cfg(not(feature = "autodiff"))]
    #[error("autodiff requested but the 'autodiff' feature is not enabled")]
    AutodiffDisabled,
}

impl CompileError {
    pub fn into_diagnostics(self, source_name: Option<&str>) -> Vec<Diagnostic> {
        match self {
            CompileError::ParseError(diags) | CompileError::TypeError(diags) => diags
                .into_iter()
                .map(|d| d.fill_file(source_name))
                .collect(),
            CompileError::IrVerify(e) => {
                vec![Diagnostic::error("ir-verify", "E3001", e.to_string())]
            }
            CompileError::MissingFunctionName => vec![Diagnostic::error(
                "autodiff",
                "E4002",
                "--autodiff requires --func <name>",
            )],
            CompileError::BackendUnavailable { target } => vec![Diagnostic::error(
                "backend",
                "E5001",
                format!("no backend available for target {target}"),
            )],
            CompileError::InvalidManifestExport { name, reason } => vec![Diagnostic::error(
                "manifest",
                "E5002",
                format!("invalid Mind.toml [exports] c_abi entry `{name}`: {reason}"),
            )],
            #[cfg(feature = "autodiff")]
            CompileError::Autodiff(e) => {
                vec![Diagnostic::error("autodiff", "E4001", e.to_string())]
            }
            #[cfg(not(feature = "autodiff"))]
            CompileError::AutodiffDisabled => vec![Diagnostic::error(
                "autodiff",
                "E4003",
                "autodiff requested but the 'autodiff' feature is not enabled",
            )],
        }
    }
}

/// Compile-time phase instrumentation (feature `compile-timings`).
///
/// Entirely cfg-gated: in a default build this module does not exist and the
/// `mark`/`report` call sites compile to nothing, so the frozen frontend
/// criterion baselines and the bit-identity hot path pay ZERO cost. Build with
/// `--features compile-timings` and set `MIND_TIMINGS=1` to print a per-phase
/// waterfall to stderr (timings go to stderr only — never into any artifact,
/// so `trace_hash` / cross-substrate byte-identity is unaffected).
#[cfg(feature = "compile-timings")]
pub(crate) mod timings {
    use std::time::{Duration, Instant};

    pub struct Timings {
        start: Instant,
        last: Instant,
        phases: Vec<(&'static str, Duration)>,
    }

    impl Timings {
        pub fn start() -> Self {
            let now = Instant::now();
            Self {
                start: now,
                last: now,
                phases: Vec::new(),
            }
        }

        pub fn mark(&mut self, name: &'static str) {
            let now = Instant::now();
            self.phases.push((name, now.duration_since(self.last)));
            self.last = now;
        }

        pub fn report(&self, what: &str) {
            // Opt-in at runtime too: a `compile-timings` build stays quiet
            // unless explicitly asked, so it can be used as a normal binary.
            if std::env::var_os("MIND_TIMINGS").is_none() {
                return;
            }
            let total = self.last.duration_since(self.start).as_secs_f64() * 1e3;
            eprintln!("[mind-timings] {what}: total {total:.3} ms");
            for (name, d) in &self.phases {
                eprintln!(
                    "[mind-timings]   {name:<14} {:.3} ms",
                    d.as_secs_f64() * 1e3
                );
            }
        }
    }
}

/// High-level compiler pipeline entry point: parse, type-check, lower to IR,
/// verify, and canonicalize. Optionally run autodiff for the requested
/// function.
pub fn compile_source(
    source: &str,
    opts: &CompileOptions,
) -> Result<CompileProducts, CompileError> {
    compile_source_with_name(source, None, opts)
}

pub fn compile_source_with_name(
    source: &str,
    source_name: Option<&str>,
    opts: &CompileOptions,
) -> Result<CompileProducts, CompileError> {
    // Non-CPU targets (Gpu, Cerebras, etc.) lower in this crate to the
    // shared canonical IR, but final code emission requires the matching
    // `mind-runtime` backend library. mindc surfaces the target as
    // unavailable here so callers can branch on `BackendUnavailable`
    // and route to the runtime crate instead of attempting in-tree
    // emission.
    if !matches!(opts.target, BackendTarget::Cpu) {
        return Err(CompileError::BackendUnavailable {
            target: opts.target,
        });
    }

    #[cfg(feature = "compile-timings")]
    let mut _tm = timings::Timings::start();

    let module = parser::parse_with_diagnostics_in_file(source, source_name)
        .map_err(CompileError::ParseError)?;
    #[cfg(feature = "compile-timings")]
    _tm.mark("parse");

    let type_diags =
        type_checker::check_module_types_in_file(&module, source, source_name, &HashMap::new());
    if !type_diags.is_empty() {
        return Err(CompileError::TypeError(type_diags));
    }
    #[cfg(feature = "compile-timings")]
    _tm.mark("typecheck");

    let mut ir = eval::lower_to_ir(&module);
    // RFC 0002 D3: merge manifest-declared exports into the IR set so
    // both `export { ... }` source blocks and `Mind.toml [exports]
    // c_abi` reach the same codegen pass. Empty in the default code
    // path — no scan, no allocation. Validated for length + identifier
    // shape to neutralise the symbol-injection / DoS surface flagged in
    // the v0.2.8 security audit.
    if !opts.manifest_exports.is_empty() {
        if opts.manifest_exports.len() > MAX_MANIFEST_EXPORTS {
            return Err(CompileError::InvalidManifestExport {
                name: format!("(count={})", opts.manifest_exports.len()),
                reason: "Mind.toml [exports] c_abi exceeds MAX_MANIFEST_EXPORTS",
            });
        }
        for name in &opts.manifest_exports {
            if let Err(reason) = validate_manifest_export_name(name) {
                return Err(CompileError::InvalidManifestExport {
                    name: name.clone(),
                    reason,
                });
            }
        }
        ir.exports.extend(opts.manifest_exports.iter().cloned());
    }
    #[cfg(feature = "compile-timings")]
    _tm.mark("ir-lower");

    ir::verify_module(&ir)?;
    opt::ir_canonical::canonicalize_module(&mut ir);
    ir::verify_module(&ir)?;
    #[cfg(feature = "compile-timings")]
    {
        _tm.mark("verify+canon");
        _tm.report("compile_source");
    }

    #[cfg(feature = "autodiff")]
    let grad = if opts.enable_autodiff {
        let func = opts
            .func
            .as_deref()
            .ok_or(CompileError::MissingFunctionName)?;
        let grad = autodiff::differentiate_function(&ir, func)?;
        Some(grad)
    } else {
        None
    };

    #[cfg(not(feature = "autodiff"))]
    if opts.enable_autodiff {
        return Err(CompileError::AutodiffDisabled);
    }

    // P1.1 runnable-artifact ABI gate: record constructs that parse + type-check
    // but cannot lower to a correct runnable artifact in the shipped i64 ABI.
    // Pure read-only walk; empty for an all-i64 program, so the mic@3 self-host
    // fixed point and the keystone stay byte-identical. Enforced only by the
    // `--emit-obj` / `--emit-shared` paths; inspection surfaces ignore it.
    let mut runnable_blockers =
        crate::eval::abi_gate::check_runnable_lowerable(&module, source, source_name);
    // Fail-closed on a generic call whose argument cannot be monomorphized — the
    // lowering would otherwise leave a dangling bare-template reference, writing
    // an EXIT=0 `.so` with an undefined symbol (a silent miscompile). Inert
    // (empty) for any module with no generic templates, so byte-identity holds.
    runnable_blockers.extend(crate::eval::abi_gate::check_generic_resolvable(
        &module,
        source,
        source_name,
    ));
    // Fail-closed on a function with a bare-scalar return that returns a payload-
    // carrying enum constructor (a heap handle) on some path — otherwise it leaks
    // a raw pointer as the i64 result. Inert (empty) for any module with no `enum`
    // declaration, so byte-identity holds (the keystone has none).
    runnable_blockers.extend(crate::eval::abi_gate::check_enum_handle_scalar_return(
        &module,
        source,
        source_name,
    ));
    // Fail-closed on an enum construct/match shape v1 cannot lower: a multi-field
    // constructor (the ctor drops the extra fields) or a multi-field/nested match
    // arm (the desugar bails to a sequential fallback that returns the wrong arm).
    // Both were SILENT miscompiles. Inert for a module with no payload enum.
    runnable_blockers.extend(crate::eval::abi_gate::check_match_runnable(
        &module,
        source,
        source_name,
    ));

    Ok(CompileProducts {
        ir,
        #[cfg(feature = "autodiff")]
        grad,
        runnable_blockers,
    })
}

/// Compile MIND source to deterministic mic@1 text suitable for runtime
/// consumption.
///
/// This is the AOT pipeline used to break the runtime-parser coupling
/// historically present in `mind-runtime/src/eval_entry.rs`: emit MIC at
/// build time, then have the runtime call [`crate::ir::load`] instead of
/// re-running [`parser::parse`] per inference.
///
/// # Stability
/// The mic@1 textual form is part of the v0.2.x stability surface — see
/// `docs/versioning.md` and `docs/ir-stability.md`.
pub fn compile_to_mic_text(source: &str, opts: &CompileOptions) -> Result<String, CompileError> {
    let products = compile_source(source, opts)?;
    Ok(crate::ir::save(&products.ir))
}

/// MLIR lowering artifacts for the canonical IR (and optional gradient IR).
#[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
#[derive(Debug, Clone)]
pub struct MlirProducts {
    /// Textual MLIR for the canonical IR module.
    pub primal_mlir: String,
    /// Textual MLIR for the gradient IR module when autodiff is enabled.
    pub grad_mlir: Option<String>,
}

/// Lower canonical IR (and optional gradient IR) into MLIR text.
///
/// The `grad` parameter is only meaningful when the `autodiff` feature is
/// compiled in; in builds without `autodiff` callers must pass `None`. The
/// concrete reference type stays internal so this function can be linked
/// from `mlir-build`-only configurations where `autodiff` is not enabled.
#[cfg(all(
    any(feature = "mlir-lowering", feature = "mlir-build"),
    feature = "autodiff"
))]
pub fn lower_to_mlir(
    ir: &ir::IRModule,
    grad: Option<&autodiff::GradientResult>,
) -> Result<MlirProducts, mlir::MlirLowerError> {
    let mut canonical_ir = ir.clone();
    ir::verify_module(&canonical_ir)?;
    opt::ir_canonical::canonicalize_module(&mut canonical_ir);
    ir::verify_module(&canonical_ir)?;

    let primal_mlir = mlir::lower_ir_to_mlir(&canonical_ir)?.text;

    let grad_mlir = if let Some(grad) = grad {
        let mut grad_ir = grad.gradient_module.clone();
        ir::verify_module(&grad_ir)?;
        opt::ir_canonical::canonicalize_module(&mut grad_ir);
        ir::verify_module(&grad_ir)?;
        Some(mlir::lower_ir_to_mlir(&grad_ir)?.text)
    } else {
        None
    };

    Ok(MlirProducts {
        primal_mlir,
        grad_mlir,
    })
}

/// Lower canonical IR into MLIR text without autodiff support.
///
/// This is the build matrix where `mlir-lowering` (or `mlir-build`) is
/// enabled but `autodiff` is not — the gradient leg is unreachable, so
/// the parameter is folded out entirely and we always return
/// `grad_mlir: None`.
#[cfg(all(
    any(feature = "mlir-lowering", feature = "mlir-build"),
    not(feature = "autodiff")
))]
pub fn lower_to_mlir(ir: &ir::IRModule) -> Result<MlirProducts, mlir::MlirLowerError> {
    let mut canonical_ir = ir.clone();
    ir::verify_module(&canonical_ir)?;
    opt::ir_canonical::canonicalize_module(&mut canonical_ir);
    ir::verify_module(&canonical_ir)?;

    let primal_mlir = mlir::lower_ir_to_mlir(&canonical_ir)?.text;

    Ok(MlirProducts {
        primal_mlir,
        grad_mlir: None,
    })
}
