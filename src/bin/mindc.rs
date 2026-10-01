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

//! MIND command-line compiler: parse, type-check, lower to IR/MLIR, and
//! optionally run autodiff.

// Small-object primary allocator, opted in per-binary (not registered by the
// library — see `libmind::SmallHeapAlloc`). Cuts allocation overhead on the
// compile hot path; produces no values, so emitted artifacts are unaffected.
#[global_allocator]
static GLOBAL_SMALL_HEAP: libmind::SmallHeapAlloc = libmind::SmallHeapAlloc;

#[path = "mindc/mic3_output.rs"]
mod mic3_output;

use std::fs;
use std::process;

use clap::{ArgAction, Parser, Subcommand};

use libmind::build::{BuildOpts, run_build};
use libmind::check::{CheckOptions, ReporterKind, run_check};
use libmind::deps::{CleanOpts, FetchOpts, LockOpts, run_clean, run_fetch, run_lock};
use libmind::doc::{DocOptions, run_doc};
use libmind::fmt::cli as mindc_fmt;
use libmind::test::{ReporterKind as TestReporterKind, TestOptions as MindTestOptions, run_tests};
use libmind::workspace::{WorkspaceOpts, resolve_workspace_members, toposort_members};

use libmind::BackendTarget;
use libmind::diagnostics::{ColorChoice, DiagnosticEmitter, DiagnosticFormat};
use libmind::ops::core_v1;
use libmind::pipeline::{CompileOptions, compile_source_with_name};
use libmind::project::{
    Backend, BenchOptions, BuildOptions, EmitKind, OptimizeLevel, bench_project, run_project,
};
use libmind::{ConformanceOptions, ConformanceProfile, conformance};

#[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
use libmind::pipeline::{MlirProducts, lower_to_mlir};

#[cfg(any(feature = "mlir-build", feature = "cross-module-imports"))]
use std::path::Path;

#[derive(Parser, Debug)]
#[command(
    author,
    about = None,
    long_about = None,
    disable_version_flag = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    compile: CompileArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build a MIND project (reads Mind.toml).
    ///
    /// RFC 0008 Phase A — single-crate orchestrator.
    /// Reads `[build]` from Mind.toml; CLI flags override the manifest.
    Build {
        /// Source files to compile.  When omitted, uses `[build].entry` or
        /// auto-detects src/main.mind / src/lib.mind.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Build in release mode (equivalent to --optimize=release).
        #[arg(long)]
        release: bool,
        /// Target backend (cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras).
        /// Overrides `[build].target` in Mind.toml.
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,
        /// Code-generation backend: mlir (default) | native.
        ///
        /// `mlir` drives the production MLIR-text → mlir-opt/clang pipeline.
        /// `native` (RI-D Option A) bridges the build to the frozen pure-MIND
        /// x86-64 native-ELF compiler for a runnable ELF with zero MLIR/LLVM/clang;
        /// fail-closed on any construct the pure-MIND subset cannot lower (never a
        /// silent MLIR fallback). The default `mlir` build is unaffected.
        #[arg(long, value_name = "BACKEND", default_value = "mlir",
              value_parser = ["mlir", "native"])]
        backend: String,
        /// Output artifact type: binary | cdylib | object.
        /// Overrides `[build].emit` in Mind.toml.
        #[arg(long, value_name = "EMIT")]
        emit: Option<String>,
        /// Optimization level: debug | release | size.
        /// Overrides `[build].optimize` in Mind.toml. --release is shorthand.
        #[arg(long, value_name = "LEVEL", conflicts_with = "release")]
        optimize: Option<String>,
        /// Custom output path.  Overrides the default `target/<profile>/<name>`.
        #[arg(long, value_name = "PATH")]
        out: Option<String>,
        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
        /// Build only the named workspace member (and its prerequisites).
        /// Alias: -p.  RFC 0008 Phase C.
        #[arg(long, short = 'p', value_name = "NAME")]
        package: Option<String>,
        /// Explicitly build all workspace members (no-op when at workspace root;
        /// included for parity with cargo).
        #[arg(long)]
        workspace: bool,
        /// Bypass the incremental object cache for this build (RFC 0008 Phase F).
        ///
        /// New objects are still written to cache so subsequent runs benefit.
        /// Use this when you suspect a stale cache entry.
        #[arg(long)]
        no_cache: bool,
    },
    /// Build and run a MIND project.
    Run {
        /// Build in release mode with optimizations.
        #[arg(long)]
        release: bool,
        /// Target backend (cpu, cuda, cuda-ampere, rocm, metal, webgpu, etc.).
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,
        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
        /// Arguments to pass to the program (after --).
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Run tests marked with `[test]` in MIND source files (RFC 0008 Phase B).
    ///
    /// Discovers all `[test]`-annotated functions in the specified paths (or
    /// the current directory when none are given), compiles and runs each as an
    /// isolated test case, and reports pass/fail in cargo-test–compatible output.
    ///
    /// Exit code 0 = all tests passed.  Exit code 1 = one or more failed.
    Test {
        /// Source files or directories to search for `[test]` functions.
        /// When omitted, walks the current directory for *.mind files.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Run only tests whose name contains this substring.
        #[arg(long, value_name = "SUBSTR")]
        filter: Option<String>,
        /// Do not capture test stdout/stderr; print it immediately.
        #[arg(long)]
        no_capture: bool,
        /// Maximum parallel worker threads (0 = use available parallelism).
        #[arg(long, value_name = "N", default_value = "0")]
        threads: usize,
        /// List test names and exit without running any tests.
        #[arg(long)]
        list: bool,
        /// Diagnostic reporter: human (default) or json.
        #[arg(long, value_name = "REPORTER", default_value = "human",
              value_parser = ["human", "json"])]
        reporter: String,
        /// Run tests for only the named workspace member (and its prerequisites).
        /// Alias: -p.  RFC 0008 Phase C.
        #[arg(long, short = 'p', value_name = "NAME")]
        package: Option<String>,
    },
    /// Run project benchmarks (bench/*.mind).
    Bench {
        /// Target backend (cpu, cuda, etc.).
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,
        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
        /// Filter benchmarks by name.
        #[arg(long, value_name = "PATTERN")]
        filter: Option<String>,
        /// Number of iterations.
        #[arg(long, value_name = "N")]
        iterations: Option<u32>,
        /// Output results as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run the Core v1 conformance suite.
    Conformance {
        /// Which profile to execute (cpu|gpu).
        #[arg(long, default_value = "cpu")]
        profile: String,
    },
    /// Run format-check + lint + type-check over MIND source files.
    ///
    /// Exit code 0 = all passes clean; 1 = one or more error-severity
    /// diagnostics detected.
    Check {
        /// Files or directories to check.  Directories are walked recursively
        /// for *.mind files.  Defaults to the current directory when omitted.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Diagnostic reporter: human (default), json, or lsp.
        ///
        /// `lsp` emits LSP-compatible Diagnostic JSON objects (RFC 0007 §C).
        #[arg(long, value_name = "REPORTER", default_value = "human",
              value_parser = ["human", "json", "lsp"])]
        reporter: String,
        /// Skip the format-check pass.
        #[arg(long)]
        no_fmt: bool,
        /// Skip the lint pass.
        #[arg(long)]
        no_lint: bool,
        /// Skip the type-check pass.
        #[arg(long)]
        no_typecheck: bool,
        /// Apply machine-applicable fixes and rewrite files.
        ///
        /// For every fmt::drift diagnostic, writes the formatted file.
        /// For every lint rule with an auto-fix, applies the byte-range edit.
        /// Iterates up to 5 rounds; warns if convergence is not reached.
        /// Prints: "Fixed N files, M unfixable diagnostics remaining."
        #[arg(long)]
        fix: bool,
    },
    /// Format MIND source files (or directories of *.mind files).
    Fmt {
        /// Files or directories to format. Directories are walked recursively
        /// for *.mind files. Defaults to the current directory when omitted.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Check whether files are already formatted; exit 1 if any would
        /// change. No files are written.
        #[arg(long)]
        check: bool,
        /// Print a unified diff between the original and formatted source;
        /// exit 1 if any file would change. No files are written.
        #[arg(long)]
        diff: bool,
        /// Read source from stdin and write the formatted result to stdout.
        /// Cannot be combined with positional PATHS.
        #[arg(long)]
        stdin: bool,
        /// Explicitly format files in-place (same as the default write mode)
        /// and print a summary: "Formatted N files, M unchanged."
        #[arg(long)]
        fix: bool,
    },
    /// Inspect compiler knowledge about Core profiles.
    Ops {
        /// Show the Core v1 operator catalog.
        #[arg(long, default_value_t = true, action = ArgAction::SetTrue)]
        core_v1: bool,
    },
    /// Regenerate Mind.lock from the current Mind.toml (RFC 0008 Phase E).
    ///
    /// Resolves all path and git dependencies, fetches git deps if needed,
    /// and writes a fully pinned Mind.lock. Mandatory before `mindc build`.
    Lock {
        /// Only verify — do not write Mind.lock; exit 1 if stale.
        #[arg(long)]
        check: bool,
        /// Re-resolve only the named package (update its entry in Mind.lock).
        #[arg(long, value_name = "PKG")]
        update: Option<String>,
    },
    /// Populate ~/.mindenv/cache/ from Mind.lock (RFC 0008 Phase E).
    ///
    /// Idempotent: already-cached deps are not re-fetched unless --update is given.
    Fetch {
        /// Re-fetch all git deps even if already cached. Does NOT modify Mind.lock.
        #[arg(long)]
        update: bool,
    },
    /// Generate HTML documentation from `///` doc-comments in MIND source files.
    ///
    /// Walks *.mind files, extracts `pub` items and their preceding `///`
    /// doc-comment blocks, and renders one HTML page per source file plus a
    /// top-level `index.html` and `search-index.json`.
    ///
    /// Exit code 0 = success, 1 = parse or I/O error, 2 = invalid CLI args.
    Doc {
        /// Source files or directories to document.  Directories are walked
        /// recursively for *.mind files.  Defaults to the current directory.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Output directory for generated HTML (default: `./target/doc`).
        #[arg(long, value_name = "DIR", default_value = "target/doc")]
        out: String,
        /// Do not render dependency files; only document the given paths.
        #[arg(long)]
        no_deps: bool,
        /// Open the generated `index.html` in a browser after rendering.
        #[arg(long)]
        open: bool,
    },
    /// Remove build artifacts and/or the dependency cache (RFC 0008 Phase E).
    Clean {
        /// Wipe ~/.mindenv/cache/ entries for this project's deps.
        #[arg(long)]
        cache: bool,
        /// Wipe both target/ and the entire ~/.mindenv/cache/.
        #[arg(long)]
        all: bool,
    },
    /// Verify the evidence chain embedded in a mic@3 artifact (RFC 0021 §4.2).
    ///
    /// Reads an artifact written by `mindc build --emit-evidence` (or
    /// `--emit-mic3` plus a MAP epilogue), peels the `evidence_chain.*` MAP,
    /// recomputes the canonical mic@3 `trace_hash` (RFC 0016 §3.2) over the
    /// parsed IR body, and confirms it matches the stored hash.  This is the
    /// consumer-side half of the wedge: generation without verification is
    /// security theatre (RFC 0021 §4 / #288 / #290 / #309).
    ///
    /// Exit code 0 = SSA well-formed and — when the artifact is attested — the
    /// trace_hash is valid (untampered). An unattested-but-SSA-valid artifact
    /// ALSO exits 0 with `attested: false`: attestation is opt-in (RFC 0017), so
    /// a consumer that requires a guarantee must fail closed with
    /// `--require-strict-fp`, `--require-deterministic`, or `--signer-pubkey`.
    /// 1 = tampered/forged chain, malformed artifact, SSA fault, or a failed
    /// `--require-*` / pinned-signer gate; 2 = I/O or CLI error.
    Verify {
        /// Path to the mic@3 evidence artifact to verify.
        #[arg(value_name = "ARTIFACT")]
        artifact: String,
        /// Emit the report as a JSON object instead of human-readable text.
        #[arg(long)]
        json: bool,
        /// Fail verification (exit 1) unless the artifact's FP-contract mode is
        /// `strict` — i.e. it used no FMA-contraction / f32-reassociation op.
        /// Off by default so existing relaxed-but-untampered f32 artifacts still
        /// pass a plain `verify`; a consumer that requires bit-identical floats
        /// opts in. Fail-closed: a `relaxed` mode, an `unknown` mode, AND an
        /// unattested artifact (no evidence_chain, so no trace_hash attesting
        /// the mode) are all rejected — the flag never silently passes.
        #[arg(long)]
        require_strict_fp: bool,
        /// Trust anchor: pin the expected signer public key(s) as hex (repeatable).
        /// When set, a signed artifact's `signature.pubkey` / `signature.mldsa_pubkey`
        /// MUST be in this allowlist or verify fails (exit 1) — this is what turns
        /// "the embedded signature is internally consistent" into "signed by a key I
        /// trust". Additional keys may be supplied via the
        /// `MIND_EVIDENCE_VERIFY_PUBKEYS` env var (comma/space-separated hex).
        /// Pinning a key makes a signature REQUIRED: an unsigned (or
        /// signature-stripped) artifact is rejected fail-closed even if its
        /// trace_hash is intact, so the pin cannot be bypassed by simply not
        /// signing. When NO allowlist is given, verify still passes an
        /// internally-consistent signature but prints the signer key(s) for
        /// out-of-band pinning and does not claim authenticity.
        #[arg(long = "signer-pubkey", value_name = "HEX", action = ArgAction::Append)]
        signer_pubkey: Vec<String>,
        /// Fail verification (exit 1) unless the artifact is `deterministic` — i.e.
        /// it calls no PRNG / wall-clock / stdin builtin. The mode is RE-DERIVED
        /// from the hashed mic@3 body (not read from the forgeable MAP field), so
        /// this is fail-closed against a tampered `determinism` label on an
        /// unsigned artifact: a `relaxed`/nondeterministic mode, an unattested
        /// artifact, OR a stored label that disagrees with the re-derived truth
        /// all fail. Off by default so a legitimately-labelled `nondeterministic`
        /// artifact still passes a plain `verify`; a consumer that requires
        /// reproducibility opts in.
        #[arg(long)]
        require_deterministic: bool,
        /// Fail verification (exit 1) unless the artifact carries a VALID signature
        /// (any signer). Weaker than `--signer-pubkey`, which additionally requires
        /// the signer be in a pinned allowlist: use `--require-signed` for an "every
        /// artifact must be signed" policy without pinning a specific key. Fail-closed:
        /// an unsigned, signature-stripped, or malformed-signature artifact is rejected
        /// even when its trace_hash is intact. Off by default (signing is opt-in).
        #[arg(long)]
        require_signed: bool,
    },
    /// Decode + inspect a mic@3 binary artifact — the consumer/debug counterpart
    /// of `--emit-mic3`. Pretty-prints the canonical IR body plus a structural
    /// summary (instruction count, SSA value count, exports, byte size) and, when
    /// the artifact carries an `evidence_chain` MAP, the trace_hash / determinism
    /// / fp_mode.
    ///
    /// With `--diff OTHER`, structurally compares two artifacts and reports the
    /// FIRST diverging byte plus each side's parse status and instruction count —
    /// the tool the self-host byte-identity gates need when a reseed or loop stops
    /// being byte-identical and "bytes differ" is not enough. mic@3 is canonical
    /// (RFC 0021), so byte-identity IS structural identity.
    ///
    /// Exit 0 = decoded (and, with `--diff`, identical); 1 = artifacts differ
    /// (`--diff`) or a malformed artifact; 2 = I/O error.
    Inspect {
        /// Path to the mic@3 artifact to inspect.
        #[arg(value_name = "ARTIFACT")]
        artifact: String,
        /// Emit the summary as a JSON object instead of human-readable text.
        #[arg(long)]
        json: bool,
        /// Structurally diff ARTIFACT against a second mic@3 artifact; report the
        /// first diverging byte. Exit 1 unless the two are byte-identical.
        #[arg(long, value_name = "OTHER")]
        diff: Option<String>,
    },
}

#[derive(Parser, Debug, Default)]
struct CompileArgs {
    /// Print the compiler version and component stability versions.
    #[arg(long, action = ArgAction::SetTrue)]
    version: bool,
    /// Print a short description of the public stability model.
    #[arg(long, action = ArgAction::SetTrue)]
    stability: bool,
    /// Input .mind file to compile.
    #[arg(value_name = "FILE")]
    input: Option<String>,
    /// Emit canonical IR for the module.
    #[arg(long)]
    emit_ir: bool,
    /// Emit MIC (compact serializable IR) for the module.
    #[arg(long)]
    emit_mic: bool,
    /// Emit gradient IR for the selected function (requires --autodiff).
    #[arg(long)]
    emit_grad_ir: bool,
    /// Emit MLIR text for the canonical IR (requires feature mlir-lowering).
    #[arg(long)]
    emit_mlir: bool,
    /// Focus on a specific function (used for autodiff and MLIR).
    #[arg(long, value_name = "NAME")]
    func: Option<String>,
    /// Run autodiff for the selected function and expose the gradient IR/MLIR.
    #[arg(long)]
    autodiff: bool,
    /// Only verify the pipeline without emitting artifacts.
    #[arg(long)]
    verify_only: bool,
    /// Emit MIC@3 binary artifact to the specified path (RFC 0021 step 3).
    ///
    /// Writes the binary mic@3 encoding of the compiled IR module.  The output
    /// is identical to calling `compact::v3::emit_mic3` on the compiled IR.
    #[arg(long, value_name = "PATH")]
    emit_mic3: Option<String>,
    /// Emit MIC@3 binary artifact with RFC 0021 evidence MAP to the specified path.
    ///
    /// Equivalent to `--emit-mic3` plus an appended `evidence_chain.*` MAP
    /// epilogue containing substrate, toolchain, determinism declaration, and
    /// a SHA-256 trace hash of the canonical IR.  Use `mic3_evidence_report`
    /// to verify the artifact offline.
    #[arg(long, value_name = "PATH")]
    emit_evidence: Option<String>,
    /// Chain this artifact to a PARENT artifact's evidence (Phase 17.7).
    ///
    /// The value is either a 64-hex-char `trace_hash` or a path to a parent mic@3
    /// evidence artifact whose `trace_hash` is read and recorded as this build's
    /// `evidence_chain.parent`. Lets provenance form a chain (child references
    /// parent). Only meaningful together with `--emit-evidence`. The parent link
    /// lives in the MAP epilogue (outside the `trace_hash` preimage), so it never
    /// perturbs this artifact's own anchor / byte-identity.
    #[arg(long, value_name = "HASH_OR_PATH")]
    evidence_parent: Option<String>,
    /// Attach an application-namespace evidence attribute (Phase 17.8), repeatable.
    ///
    /// `KEY=VALUE` where `KEY` is a dotted, non-reserved namespace
    /// (`org.example.build_id=42`). Reserved `evidence_chain.*` / `signature.*`
    /// keys are rejected. Attributes are byte-additive: none supplied ⇒ the
    /// artifact is byte-identical to the closed-key encoder. Only meaningful with
    /// `--emit-evidence`.
    #[arg(long = "evidence-attr", value_name = "KEY=VALUE")]
    evidence_attr: Vec<String>,
    /// Compile a NON-DETERMINISTIC program (one that calls a PRNG / wall-clock /
    /// stdin builtin such as `random()` / `now()`). MIND programs are
    /// deterministic by default — such a program is REJECTED fail-loud unless this
    /// flag is passed, which points the author at the seeded `Random(seed=…)` API.
    /// Non-determinism never leaks untraced: WITH the flag the program compiles,
    /// and its evidence chain still honestly attests `nondeterministic` (the flag
    /// authorises the build, it never touches the attestation). A whole-artifact
    /// property — the artifact IS non-deterministic if any part of it is.
    #[arg(long)]
    allow_nondeterministic: bool,
    /// Emit object file (.o) to the specified path.
    #[arg(long, value_name = "PATH")]
    emit_obj: Option<String>,
    /// Emit a shared library (`.so` on Linux, `.dylib` on macOS) to the
    /// specified path. Equivalent to `--emit-obj` followed by a shared-
    /// library link. Phase 10.8 / mindc 0.3.0 cdylib-emit foundation.
    /// Requires the `mlir-build` feature.
    #[arg(long, value_name = "PATH")]
    emit_shared: Option<String>,
    /// Select the execution target backend (cpu|gpu).
    #[arg(long, value_name = "TARGET", default_value = "cpu")]
    target: String,
    /// Language profile (default|systems|embedded). RFC 0002 deliverable 5:
    /// the same Mind.toml produces a distinct artifact per profile via the
    /// cache fingerprint, so cross-mode rebuilds never hit a stale entry.
    /// Strict on the CLI surface: unknown values are rejected by clap
    /// before reaching `ProfileTag::parse`'s permissive fallback.
    #[arg(
        long,
        value_name = "PROFILE",
        default_value = "default",
        value_parser = ["default", "systems", "embedded"],
    )]
    profile: String,
    /// Diagnostic output format (human|short|json).
    #[arg(long, value_name = "FORMAT", default_value = "human")]
    diagnostic_format: String,
    /// ANSI color handling (auto|always|never).
    #[arg(long, value_name = "WHEN")]
    color: Option<String>,
}

fn main() {
    #[cfg(windows)]
    {
        // The PE main thread has a smaller stack than Rust's ordinary worker
        // threads. Run the compiler driver on the standard thread default.
        match std::thread::Builder::new()
            .name("mindc-driver".to_string())
            .spawn(mindc_main)
            .expect("failed to start mindc driver thread")
            .join()
        {
            Ok(()) => {}
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    #[cfg(not(windows))]
    mindc_main();
}

fn mindc_main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Command::Build {
            paths,
            release,
            target,
            backend,
            emit,
            optimize,
            out,
            verbose,
            package,
            workspace: _,
            no_cache,
        }) => {
            run_mindc_build(
                paths,
                *release,
                target,
                backend,
                emit,
                optimize,
                out,
                *verbose,
                package.as_deref(),
                *no_cache,
            );
            return;
        }
        Some(Command::Run {
            release,
            target,
            verbose,
            args,
        }) => {
            run_run_command(*release, target.clone(), *verbose, args.clone());
            return;
        }
        Some(Command::Test {
            paths,
            filter,
            no_capture: _,
            threads,
            list,
            reporter,
            package,
        }) => {
            run_mindc_test(
                paths,
                filter.as_deref(),
                *threads,
                *list,
                reporter,
                package.as_deref(),
            );
            return;
        }
        Some(Command::Bench {
            target,
            verbose,
            filter,
            iterations,
            json,
        }) => {
            let opts = BenchOptions {
                target: target.clone(),
                verbose: *verbose,
                filter: filter.clone(),
                iterations: *iterations,
                json: *json,
            };
            match bench_project(&opts) {
                Ok(code) => process::exit(code),
                Err(err) => {
                    eprintln!("error: {}", err);
                    process::exit(1);
                }
            }
        }
        Some(Command::Conformance { profile }) => {
            run_conformance(profile);
            return;
        }
        Some(Command::Check {
            paths,
            reporter,
            no_fmt,
            no_lint,
            no_typecheck,
            fix,
        }) => {
            let reporter_kind = match reporter.as_str() {
                "json" => ReporterKind::Json,
                "lsp" => ReporterKind::Lsp,
                _ => ReporterKind::Human,
            };
            let opts = CheckOptions {
                run_fmt: !no_fmt,
                run_lint: !no_lint,
                run_typecheck: !no_typecheck,
                reporter: reporter_kind,
                paths: paths.clone(),
                fix: *fix,
            };
            process::exit(run_check(&opts));
        }
        Some(Command::Fmt {
            paths,
            check,
            diff,
            stdin,
            fix,
        }) => {
            process::exit(mindc_fmt::run_fmt(paths, *check, *diff, *stdin, *fix));
        }
        Some(Command::Doc {
            paths,
            out,
            no_deps,
            open,
        }) => {
            let opts = DocOptions {
                paths: paths.clone(),
                out_dir: std::path::PathBuf::from(out),
                no_deps: *no_deps,
                open: *open,
            };
            process::exit(run_doc(&opts));
        }
        Some(Command::Ops { .. }) => {
            print_ops(&cli.command);
            return;
        }
        Some(Command::Lock { check, update }) => {
            run_mindc_lock(*check, update.as_deref());
            return;
        }
        Some(Command::Fetch { update }) => {
            run_mindc_fetch(*update);
            return;
        }
        Some(Command::Clean { cache, all }) => {
            run_mindc_clean(*cache, *all);
            return;
        }
        Some(Command::Verify {
            artifact,
            json,
            require_strict_fp,
            signer_pubkey,
            require_deterministic,
            require_signed,
        }) => {
            let trusted = match collect_trusted_pubkeys(signer_pubkey) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error[verify]: {e}");
                    process::exit(2);
                }
            };
            process::exit(run_verify(
                artifact,
                *json,
                *require_strict_fp,
                *require_deterministic,
                *require_signed,
                &trusted,
            ));
        }
        Some(Command::Inspect {
            artifact,
            json,
            diff,
        }) => {
            process::exit(run_inspect(artifact, *json, diff.as_deref()));
        }
        None => {}
    }

    if cli.compile.version {
        print_version();
        return;
    }

    if cli.compile.stability {
        print_stability();
        return;
    }

    let input = match &cli.compile.input {
        Some(path) => path.clone(),
        None => {
            eprintln!("error[cli]: expected an input file or subcommand");
            process::exit(1);
        }
    };

    if cli.compile.autodiff && cli.compile.func.is_none() {
        eprintln!("error[autodiff]: --autodiff requires --func <name>");
        process::exit(1);
    }

    let target = match parse_target(&cli.compile.target) {
        Ok(target) => target,
        Err(msg) => {
            eprintln!("error[backend]: {msg}");
            process::exit(1);
        }
    };

    let diagnostic_format =
        DiagnosticFormat::parse(&cli.compile.diagnostic_format).unwrap_or(DiagnosticFormat::Human);
    let color_choice = resolve_color_choice(&cli.compile.color);
    let emitter = DiagnosticEmitter::new(diagnostic_format, color_choice);

    let source = match fs::read_to_string(&input) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("failed to read {}: {err}", input);
            process::exit(1);
        }
    };

    let opts = CompileOptions {
        func: cli.compile.func.clone(),
        enable_autodiff: cli.compile.autodiff,
        target,
        profile: libmind::cache::ProfileTag::parse(&cli.compile.profile),
        ..Default::default()
    };

    #[cfg(feature = "cross-module-imports")]
    let single_file_scope = match libmind::project::single_file_scope::discover_with_source(
        Path::new(&input),
        &cli.compile.target,
        &source,
    ) {
        Ok(libmind::project::single_file_scope::Discovery::Project(scope)) => Some(scope),
        Ok(libmind::project::single_file_scope::Discovery::MissingProject(imports)) => {
            eprintln!(
                "error[type-check][E2003]: local import(s) {} require an enclosing Mind.toml \
                 project that declares the source set",
                imports.join(", ")
            );
            process::exit(1);
        }
        Ok(libmind::project::single_file_scope::Discovery::SingleTranslationUnit) => None,
        Err(err) => {
            eprintln!("error[project]: cannot establish single-file import scope: {err}");
            process::exit(1);
        }
    };
    #[cfg(feature = "cross-module-imports")]
    let _single_file_guard = single_file_scope.as_ref().map(|scope| scope.install());

    let products = match compile_source_with_name(&source, Some(&input), &opts) {
        Ok(products) => products,
        Err(err) => {
            let diags = err.into_diagnostics(Some(&input));
            emitter.emit_all(&diags, Some(&source));
            process::exit(1);
        }
    };

    if cli.compile.verify_only {
        return;
    }

    // Determinism-by-default gate (Option C): a program that calls a HARD
    // non-deterministic builtin (PRNG / wall-clock / stdin, e.g. `random()` /
    // `now()`) is REJECTED fail-loud when producing a runnable (`--emit-obj` /
    // `--emit-shared`) or attested (`--emit-evidence`) artifact, unless
    // `--allow-nondeterministic` authorises it. This gates SHIPPING, not
    // inspecting — `--emit-ir` / `--emit-mlir` / `--emit-mic` / `check` still show
    // the IR of a non-deterministic program. Non-determinism can never leak
    // untraced: WITH the flag the artifact compiles AND still attests
    // `nondeterministic` (the flag authorises the build, never the label). The
    // check is whole-program (an artifact IS non-deterministic if any part is).
    let produces_artifact = cli.compile.emit_obj.is_some()
        || cli.compile.emit_shared.is_some()
        || cli.compile.emit_evidence.is_some();
    if produces_artifact && !cli.compile.allow_nondeterministic {
        // HARD-only: an unclassified `extern "C"` callee is UNKNOWN, not nondeterministic.
        // Attestation stays conservative via ir_first_nondeterministic_call.
        if let Some(offender) = libmind::ir::ir_first_hard_nondeterministic_call(&products.ir) {
            eprintln!("error[determinism]: `{offender}()` introduces unseeded nondeterminism");
            eprintln!();
            eprintln!("MIND programs are deterministic by default. Use a seeded generator such as");
            eprintln!("`Random(seed = 42)`, or rebuild with `--allow-nondeterministic`.");
            eprintln!(
                "Artifacts built with `--allow-nondeterministic` are attested as `nondeterministic`."
            );
            process::exit(1);
        }
    }

    let emit_ir = cli.compile.emit_ir
        || (!cli.compile.emit_grad_ir
            && !cli.compile.emit_mlir
            && !cli.compile.emit_mic
            && cli.compile.emit_mic3.is_none()
            && cli.compile.emit_evidence.is_none());
    if emit_ir {
        println!("{}", products.ir);
    }

    if cli.compile.emit_mic {
        let mic = libmind::ir::compact::emit_mic(&products.ir);
        println!("{}", mic);
    }

    mic3_output::emit_mic3_if_requested(&cli.compile, &products);
    emit_evidence_if_requested(&cli.compile, &products);

    #[cfg(feature = "autodiff")]
    if cli.compile.autodiff && cli.compile.emit_grad_ir {
        match products.grad.as_ref() {
            Some(grad) => println!("{}", grad.gradient_module),
            None => {
                eprintln!("autodiff did not produce gradient IR");
                process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "autodiff"))]
    if cli.compile.autodiff && cli.compile.emit_grad_ir {
        eprintln!("gradient IR emission requires building with the 'autodiff' feature");
        process::exit(1);
    }

    emit_mlir_if_requested(&cli.compile, &products);

    // P1.1: a runnable artifact (`--emit-obj` / `--emit-shared`) must never be a
    // silent miscompile. If the source uses a construct outside the i64-scalar
    // ABI the backend lowers correctly, fail loud here with file:line + RC!=0.
    // Inspection emits above (`--emit-ir` / `--emit-mlir`) are intentionally
    // unaffected — `i32`/`tensor` etc. are valid *types*, just not yet lowerable
    // to a runnable artifact.
    if (cli.compile.emit_obj.is_some() || cli.compile.emit_shared.is_some())
        && !products.runnable_blockers.is_empty()
    {
        emitter.emit_all(&products.runnable_blockers, Some(&source));
        process::exit(1);
    }

    emit_obj_if_requested(&cli.compile, &products);
    #[cfg(feature = "cross-module-imports")]
    emit_shared_if_requested(&cli.compile, &products, &source, single_file_scope.as_ref());
    #[cfg(not(feature = "cross-module-imports"))]
    emit_shared_if_requested(&cli.compile, &products, &source);
}

#[allow(clippy::too_many_arguments)]
fn run_mindc_build(
    paths: &[String],
    release: bool,
    target: &Option<String>,
    backend: &str,
    emit: &Option<String>,
    optimize: &Option<String>,
    out: &Option<String>,
    verbose: bool,
    package: Option<&str>,
    no_cache: bool,
) {
    // Code-generation backend dispatch (RI-D seam). The default `mlir` path is
    // left fully inert: it falls through to the existing pipeline unchanged.
    // `native` (RI-D Option A) hands the build to the pure-MIND x86-64 native-ELF
    // compiler (zero MLIR/LLVM/clang), fail-closed — never a silent MLIR fallback.
    match Backend::parse(backend) {
        Ok(Backend::Mlir) => {}
        Ok(Backend::Native) => {
            // RI-D Option A: hand the whole build to the pure-MIND native-ELF
            // compiler (zero MLIR/LLVM/clang). Returns after writing the artifact.
            libmind::build::native_bridge::run_native_backend_bridge(paths, out, emit);
            return;
        }
        Err(msg) => {
            eprintln!("error[build]: {}", msg);
            process::exit(2);
        }
    }

    // Workspace detection: if we are at a workspace root and no explicit
    // source paths are given, delegate to the workspace build path.
    if paths.is_empty() {
        if let Some(root) = detect_workspace_root() {
            run_workspace_build(
                &root, release, target, emit, optimize, out, verbose, package,
            );
            return;
        }
    }

    // --target is passed RAW to the build layer, which resolves it against the
    // manifest (`[targets.<name>]` block name wins, then backend class, else a
    // hard error). Classifying here would reject a declared block name (e.g.
    // `windows`) before the manifest is even loaded.

    // Parse --emit override.
    let eff_emit: Option<EmitKind> = match emit {
        None => None,
        Some(e) => match EmitKind::parse(e) {
            Ok(ek) => Some(ek),
            Err(msg) => {
                eprintln!("error[build]: {}", msg);
                process::exit(2);
            }
        },
    };

    // --release is shorthand for --optimize=release.
    let eff_optimize: Option<OptimizeLevel> = if release {
        Some(OptimizeLevel::Release)
    } else {
        match optimize {
            None => None,
            Some(o) => match OptimizeLevel::parse(o) {
                Ok(ol) => Some(ol),
                Err(msg) => {
                    eprintln!("error[build]: {}", msg);
                    process::exit(2);
                }
            },
        }
    };

    let opts = BuildOpts {
        paths: paths.iter().map(std::path::PathBuf::from).collect(),
        target: target.clone(),
        emit: eff_emit,
        optimize: eff_optimize,
        out: out.as_ref().map(std::path::PathBuf::from),
        verbose,
        no_cache,
    };

    match run_build(&opts) {
        Ok(output) => {
            println!(
                "   Finished {} [{}] {}",
                output.target,
                output.emit.as_str(),
                output.artifact_path.display()
            );
            println!("   Artifact: {} bytes", output.byte_count);
        }
        Err(err) => {
            // `render()` owns the prefix so a coded refusal keeps its code on
            // the wire (`error[build][E5003]: …`); a `{err}` here would drop it.
            eprintln!("{}", err.render());
            process::exit(err.exit_code());
        }
    }
}

fn run_mindc_test(
    paths: &[String],
    filter: Option<&str>,
    threads: usize,
    list: bool,
    reporter: &str,
    package: Option<&str>,
) {
    // Workspace detection: if invoked in a workspace root with no explicit
    // source paths, run tests for all members (or the named member).
    if paths.is_empty() {
        if let Some(root) = detect_workspace_root() {
            run_workspace_test(&root, filter, threads, list, reporter, package);
            return;
        }
    }

    let reporter_kind = if reporter == "json" {
        TestReporterKind::Json
    } else {
        TestReporterKind::Human
    };

    let opts = MindTestOptions {
        paths: paths.iter().map(std::path::PathBuf::from).collect(),
        filter: filter.unwrap_or("").to_string(),
        capture: true,
        threads,
        list,
        reporter: reporter_kind,
    };

    match run_tests(&opts) {
        Ok(summary) => {
            // Phase 17.8: fail-loud on a ZERO-test run. `mindc test` on a suite
            // with no `#[test]` previously printed "running 0 tests / ok" and
            // exited 0, so any CI gate built on the return code was silently
            // green (a no-false-green violation). A discovery that finds nothing
            // to run is a failure, not a pass — exit non-zero. `--list` is
            // exempt: it deliberately enumerates without running.
            if !list && summary.passed == 0 && summary.failed == 0 {
                eprintln!(
                    "error[test]: no tests found (0 tests ran); \
                     nothing was verified — treating as failure"
                );
                process::exit(1);
            }
            if summary.all_passed() {
                process::exit(0);
            } else {
                process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("error[test]: {}", err);
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// RFC 0008 Phase C — workspace dispatch helpers
// ---------------------------------------------------------------------------

/// Detect whether the current working directory (or a parent) is a workspace
/// root (has a `Mind.toml` with a `[workspace]` block).
///
/// Returns `Some(root)` when a workspace root is found, `None` otherwise.
fn detect_workspace_root() -> Option<std::path::PathBuf> {
    use libmind::project::find_project_root;
    let root = find_project_root().ok()?;
    let text = std::fs::read_to_string(root.join("Mind.toml")).ok()?;
    if text.contains("[workspace]") {
        Some(root)
    } else {
        None
    }
}

/// Build all workspace members (or a filtered subset) in topological order.
#[allow(clippy::too_many_arguments)]
fn run_workspace_build(
    workspace_root: &std::path::Path,
    release: bool,
    target: &Option<String>,
    emit: &Option<String>,
    optimize: &Option<String>,
    out: &Option<String>,
    verbose: bool,
    package: Option<&str>,
) {
    let members = match resolve_workspace_members(workspace_root) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let sorted = match toposort_members(&members) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let ws_opts = WorkspaceOpts {
        package_filter: package.map(|s| s.to_string()),
    };
    let selected = ws_opts.filter_members(&members, &sorted);

    if selected.is_empty() {
        if let Some(pkg) = package {
            eprintln!("error[workspace]: package '{pkg}' not found in workspace");
            process::exit(2);
        }
    }

    let mut any_failed = false;
    for member in &selected {
        if verbose {
            eprintln!("   Building workspace member: {}", member.name);
        }
        // Change into the member directory and delegate to the single-crate
        // builder by temporarily pushing the manifest path.
        let member_paths: Vec<String> = vec![];
        let eff_out: Option<String> = if out.is_some() && selected.len() == 1 {
            out.clone()
        } else {
            None // each member uses its own default output path
        };
        let member_out = std::env::current_dir().ok().and(eff_out);

        // Run the Phase A build for this member's root.
        let mut build_opts = BuildOpts {
            paths: member_paths.iter().map(std::path::PathBuf::from).collect(),
            target: target.clone(),
            emit: parse_emit_opt(emit),
            optimize: parse_optimize_opt(release, optimize),
            out: member_out.map(std::path::PathBuf::from),
            verbose,
            no_cache: false,
        };
        // Override paths to use the member root's entry point resolution.
        // The member root is passed via a synthetic path pointing to the member.
        build_opts.paths = vec![member.root.clone()];

        // Temporarily change working directory to the member root so that
        // find_project_root() inside run_build picks up the member's Mind.toml.
        let saved_dir = std::env::current_dir().unwrap_or_else(|_| workspace_root.to_path_buf());
        if std::env::set_current_dir(&member.root).is_ok() {
            build_opts.paths = vec![];
        }

        match run_build(&build_opts) {
            Ok(output) => {
                println!(
                    "   Finished {} ({}) [{}] {}",
                    member.name,
                    output.target,
                    output.emit.as_str(),
                    output.artifact_path.display()
                );
            }
            Err(err) => {
                // Keep the cause code on the wire here too (`code_tag()` is the
                // single owner of the token's shape).
                eprintln!(
                    "error[workspace][{}]{}: {}",
                    member.name,
                    err.code_tag(),
                    err
                );
                any_failed = true;
            }
        }

        // Restore working directory.
        let _ = std::env::set_current_dir(&saved_dir);
    }

    if any_failed {
        process::exit(1);
    }
}

fn parse_emit_opt(emit: &Option<String>) -> Option<EmitKind> {
    match emit {
        None => None,
        Some(e) => match EmitKind::parse(e) {
            Ok(ek) => Some(ek),
            Err(msg) => {
                eprintln!("error[build]: {msg}");
                process::exit(2);
            }
        },
    }
}

fn parse_optimize_opt(release: bool, optimize: &Option<String>) -> Option<OptimizeLevel> {
    if release {
        Some(OptimizeLevel::Release)
    } else {
        match optimize {
            None => None,
            Some(o) => match OptimizeLevel::parse(o) {
                Ok(ol) => Some(ol),
                Err(msg) => {
                    eprintln!("error[build]: {msg}");
                    process::exit(2);
                }
            },
        }
    }
}

/// Run tests for all workspace members (or a filtered subset).
fn run_workspace_test(
    workspace_root: &std::path::Path,
    filter: Option<&str>,
    threads: usize,
    list: bool,
    reporter: &str,
    package: Option<&str>,
) {
    let members = match resolve_workspace_members(workspace_root) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let sorted = match toposort_members(&members) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let ws_opts = WorkspaceOpts {
        package_filter: package.map(|s| s.to_string()),
    };
    let selected = ws_opts.filter_members(&members, &sorted);

    if selected.is_empty() {
        if let Some(pkg) = package {
            eprintln!("error[workspace]: package '{pkg}' not found in workspace");
            process::exit(2);
        }
    }

    let reporter_kind = if reporter == "json" {
        TestReporterKind::Json
    } else {
        TestReporterKind::Human
    };

    let mut any_failed = false;
    let saved_dir = std::env::current_dir().unwrap_or_else(|_| workspace_root.to_path_buf());

    for member in &selected {
        if std::env::set_current_dir(&member.root).is_err() {
            eprintln!(
                "error[workspace]: cannot enter member directory: {}",
                member.root.display()
            );
            any_failed = true;
            continue;
        }

        let opts = MindTestOptions {
            paths: vec![],
            filter: filter.unwrap_or("").to_string(),
            capture: true,
            threads,
            list,
            reporter: reporter_kind.clone(),
        };

        match run_tests(&opts) {
            Ok(summary) => {
                if !summary.all_passed() {
                    any_failed = true;
                }
            }
            Err(err) => {
                eprintln!("error[test][{}]: {}", member.name, err);
                any_failed = true;
            }
        }

        let _ = std::env::set_current_dir(&saved_dir);
    }

    if any_failed {
        process::exit(1);
    }
}

fn run_run_command(release: bool, target: Option<String>, verbose: bool, args: Vec<String>) {
    let opts = BuildOptions {
        release,
        target,
        verbose,
        ..Default::default()
    };

    match run_project(&args, &opts) {
        Ok(code) => {
            process::exit(code);
        }
        Err(err) => {
            eprintln!("error: {}", err);
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// RFC 0008 Phase D + E — lock / fetch / clean handlers
// ---------------------------------------------------------------------------

fn run_mindc_lock(check: bool, update_pkg: Option<&str>) {
    use libmind::project::{find_project_root, load_manifest};
    let root = match find_project_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error[lock]: cannot find Mind.toml: {e}");
            process::exit(1);
        }
    };
    let manifest = match load_manifest(&root) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[lock]: {e}");
            process::exit(1);
        }
    };
    let opts = LockOpts {
        check,
        update_pkg: update_pkg.map(|s| s.to_string()),
    };
    match run_lock(&root, &manifest, &opts) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error[lock]: {e}");
            process::exit(e.exit_code());
        }
    }
}

fn run_mindc_fetch(update: bool) {
    use libmind::project::find_project_root;
    let root = match find_project_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error[fetch]: cannot find Mind.toml: {e}");
            process::exit(1);
        }
    };
    let opts = FetchOpts { update };
    match run_fetch(&root, &opts) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error[fetch]: {e}");
            process::exit(e.exit_code());
        }
    }
}

fn run_mindc_clean(cache: bool, all: bool) {
    use libmind::build::cache::clean_all_caches;
    use libmind::project::find_project_root;

    let root = match find_project_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error[clean]: cannot find Mind.toml: {e}");
            process::exit(1);
        }
    };

    // Phase F: --cache wipes the incremental build object cache (.cache/ dirs
    // under target/), leaving the previously linked binaries intact.
    if cache && !all {
        match clean_all_caches(&root) {
            Ok(()) => println!("   Removed incremental cache (target/*/.cache/)."),
            Err(e) => {
                eprintln!("error[clean]: {e}");
                process::exit(1);
            }
        }
        // Also clean the deps git cache via the deps subsystem.
        let opts = CleanOpts {
            cache: true,
            all: false,
        };
        match run_clean(&root, &opts) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error[clean]: {e}");
                process::exit(e.exit_code());
            }
        }
        return;
    }

    let opts = CleanOpts { cache, all };
    match run_clean(&root, &opts) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error[clean]: {e}");
            process::exit(e.exit_code());
        }
    }
}

fn print_ops(command: &Option<Command>) {
    if let Some(Command::Ops { core_v1 }) = command {
        if *core_v1 {
            println!("Core v1 operators (name | arity | dtypes | autodiff)");
            for op in core_v1::core_v1_ops() {
                let arity = match op.arity {
                    core_v1::Arity::Fixed(n) => format!("{n}"),
                    core_v1::Arity::Variadic { min } => format!("{min}+"),
                };
                let dtypes = if op.allowed_dtypes.is_empty() {
                    "shape-dependent".to_string()
                } else {
                    op.allowed_dtypes
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let autodiff = if op.differentiable { "yes" } else { "no" };
                println!(
                    "{:<18} | {:<6} | {:<24} | {}",
                    op.name, arity, dtypes, autodiff
                );
            }
        }
    }
}

fn print_version() {
    println!("mind {}", env!("CARGO_PKG_VERSION"));

    // Advertise ONLY the components actually compiled into this binary — a build
    // without the `autodiff` / `mlir-lowering` features must not claim them, since
    // `--autodiff` / `--emit-mlir` feature-error there (release-readiness: no false
    // capability advertisement to installed users).
    // `mut` is only exercised when a feature below pushes a component; in a
    // build with neither `autodiff` nor `mlir-lowering` the vec is never mutated,
    // so scope the `unused_mut` allowance to exactly that configuration (clippy
    // `-D warnings` runs the no-default-features job).
    #[cfg_attr(
        not(any(feature = "autodiff", feature = "mlir-lowering")),
        allow(unused_mut, clippy::useless_vec)
    )]
    let mut components = vec!["core-ir=1.0"];
    #[cfg(feature = "autodiff")]
    components.push("core-autodiff=1.0");
    #[cfg(feature = "mlir-lowering")]
    components.push("mlir-lowering=0.1");

    println!("{}", components.join("  "));
}

fn print_stability() {
    println!(
        "MIND Core v1 stability: stable IR/autodiff/CLI surfaces; MLIR lowering is\
         conditionally stable within a minor release; new ops & feature flags are\
         experimental. See docs/versioning.md for details."
    );
}

fn run_conformance(profile: &str) {
    let profile = match profile.to_ascii_lowercase().as_str() {
        "cpu" => ConformanceProfile::CpuBaseline,
        "gpu" => ConformanceProfile::CpuAndGpu,
        other => {
            eprintln!("error[conformance]: unknown profile '{other}' (expected cpu|gpu)");
            process::exit(1);
        }
    };

    match conformance::run_conformance(ConformanceOptions { profile }) {
        Ok(report) => {
            // The suite owns what its own pass attests (`attestation_lines`);
            // the CLI only puts it on the wire. Formatting it here once meant a
            // second, drifting statement of what a green run proves.
            for line in report.attestation_lines() {
                println!("{line}");
            }
        }
        Err(err) => {
            eprintln!("conformance failures detected:");
            for failure in err.0.iter() {
                eprintln!("- {failure}");
            }
            process::exit(1);
        }
    }
}

#[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
fn emit_mlir_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    if !cli.emit_mlir {
        return;
    }

    let mlir: MlirProducts = match lower_to_mlir_compat(products) {
        Ok(mlir) => mlir,
        Err(err) => {
            eprintln!("error[mlir]: {err}");
            process::exit(1);
        }
    };

    println!("{}", mlir.primal_mlir);

    if cli.autodiff {
        if let Some(grad_mlir) = mlir.grad_mlir {
            println!("{}", grad_mlir);
        }
    }
}

/// Thin wrapper around `pipeline::lower_to_mlir` that erases the
/// `autodiff`-feature signature difference for the `mindc` binary.
#[cfg(all(
    any(feature = "mlir-lowering", feature = "mlir-build"),
    feature = "autodiff"
))]
fn lower_to_mlir_compat(
    products: &libmind::pipeline::CompileProducts,
) -> Result<MlirProducts, libmind::MlirLowerError> {
    lower_to_mlir(&products.ir, products.grad.as_ref())
}

#[cfg(all(
    any(feature = "mlir-lowering", feature = "mlir-build"),
    not(feature = "autodiff")
))]
fn lower_to_mlir_compat(
    products: &libmind::pipeline::CompileProducts,
) -> Result<MlirProducts, libmind::MlirLowerError> {
    lower_to_mlir(&products.ir)
}

#[cfg(not(any(feature = "mlir-lowering", feature = "mlir-build")))]
fn emit_mlir_if_requested(cli: &CompileArgs, _products: &libmind::pipeline::CompileProducts) {
    if cli.emit_mlir {
        eprintln!(
            "error[mlir][{}]: MLIR emission requires building with the 'mlir-lowering' or \
             'mlir-build' feature",
            libmind::diagnostics::capability::NO_NATIVE_BACKEND
        );
        process::exit(1);
    }
}

fn emit_evidence_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    let path = match &cli.emit_evidence {
        Some(p) => p,
        None => return,
    };
    let substrate = cli.target.as_str();
    // Honest-by-derivation determinism declaration (Option C, phase 1): the
    // artifact attests `nondeterministic` iff its IR actually calls a PRNG /
    // wall-clock / stdin builtin (`random`/`now`/…), else `deterministic`. This
    // closes the forge where a `random()` program attested `deterministic` — the
    // one claim `mind verify` reports. Deterministic programs (incl. seeded
    // `randn(shape, seed)`) are unchanged. The `determinism` field lives in the
    // MAP epilogue (not the trace_hash), so this never perturbs byte-identity.
    // (Phase 2 — determinism-by-default with an explicit `#[nondeterministic]`
    // opt-in that REJECTS hidden non-determinism — is tracked as the RFC 0012
    // follow-up, superseding the old TODO(#289).)
    let determinism = if libmind::ir::ir_declares_deterministic(&products.ir) {
        libmind::ir::compact::Determinism::Deterministic
    } else {
        libmind::ir::compact::Determinism::Nondeterministic
    };
    let toolchain = env!("CARGO_PKG_VERSION");

    // Optional crypto-agile signing (RFC 0021 §6), opt-in via env-supplied seeds
    // (never a hardcoded key). The supported signing profile is the exact
    // ML-DSA-87 + SLH-DSA-SHAKE-256s pair. The historical Ed25519 and
    // Ed25519+ML-DSA-65 variables remain recognized as retired configuration and
    // refuse explicitly; they are available only for historical inspection.
    // Each supported seed is supplied as hex. No env ⇒ the unsigned path,
    // byte-identical to the pre-signing encoder (the determinism gate is untouched).
    use libmind::ir::compact::SigningKey;
    // No legacy seed is READ. The scheme is retired, so parsing it would only
    // create a path where an invalid retired seed reports "invalid" instead of
    // "retired" — the wrong cause, and one that implies a fixable configuration.
    let mldsa_seed = read_seed_env(libmind::ir::compact::v3::evidence::ENV_MLDSA_SEED);
    // Bulletproof PQC-hybrid (max-security offline release profile): BOTH the
    // ML-DSA-87 and SLH-DSA-256s seeds must be supplied. It takes precedence over
    // the transition schemes when present; one leg alone is a partial, NON-
    // bulletproof config and is refused fail-closed (never a silent downgrade).
    let mldsa87_seed = read_seed_env(libmind::ir::compact::v3::mldsa::ENV_MLDSA87_SEED);
    let slhdsa_seed = read_seed_env_96(libmind::ir::compact::v3::slhdsa::ENV_SLHDSA_SEED);
    // RETIRED LEGACY SEEDS refuse EXPLICITLY, before any key is selected.
    //
    // Ed25519 signing is permanently retired. A supplied legacy seed is an
    // explicit request for a retired scheme, so it is an error — not something
    // to ignore while quietly choosing a different key. That distinction is the
    // whole point: silently falling back to the hybrid would mean an operator
    // who believes they configured Ed25519 gets something else without being
    // told, and an operator who configured BOTH would never learn their legacy
    // config was dead. Both cases refuse here, including the mixed
    // supported-plus-legacy configuration.
    // Presence alone, via `var_os`: a retired scheme needs no seed parsing, and
    // an unparseable or non-UTF-8 legacy value must still refuse rather than be
    // read as absent.
    if std::env::var_os(libmind::ir::compact::v3::evidence::ENV_ED25519_SEED).is_some() {
        eprintln!(
            "error[emit-evidence]: Ed25519 evidence signing is permanently retired \
             ({}). Remove the legacy seed; the supported signing mode is the \
             post-quantum hybrid ML-DSA-87 + SLH-DSA-SHAKE-256s via \
             MIND_EVIDENCE_MLDSA87_KEY and MIND_EVIDENCE_SLHDSA_KEY. Refusing \
             rather than silently signing under a different scheme. \
             (signing.retired_scheme)",
            libmind::ir::compact::v3::evidence::ENV_ED25519_SEED
        );
        process::exit(1);
    }

    let signing_key: Option<SigningKey> = match (mldsa87_seed, slhdsa_seed) {
        (Some(m87), Some(slh)) => Some(SigningKey::PqcHybrid {
            mldsa87: m87,
            slhdsa: slh,
        }),
        (Some(_), None) | (None, Some(_)) => {
            eprintln!(
                "error[emit-evidence]: the bulletproof PQC-hybrid needs BOTH \
                 MIND_EVIDENCE_MLDSA87_KEY and MIND_EVIDENCE_SLHDSA_KEY — one is missing (fail-closed)"
            );
            process::exit(1);
        }
        // Ed-bearing arms are unreachable: a legacy seed already refused above,
        // which also retires the old Ed25519+ML-DSA-65 hybrid, since that hybrid
        // cannot be requested without the Ed seed.
        //
        // Standalone ML-DSA-65 remains selectable and is NOT part of the
        // supported production release profile, which requires the exact
        // ML-DSA-87 + SLH-DSA pair. It is left reachable deliberately rather
        // than removed by association: retiring Ed is mandated, widening that to
        // an unrelated post-quantum primitive is not.
        (None, None) => mldsa_seed.map(SigningKey::MlDsa65),
    };
    let sig_label = match &signing_key {
        Some(SigningKey::PqcHybrid { .. }) => ", pqc-hybrid-ml-dsa-87-slh-dsa-256s-signed",
        Some(SigningKey::Ed25519(_)) | Some(SigningKey::Hybrid { .. }) => {
            ", retired-signing-refused"
        }
        Some(SigningKey::MlDsa65(_)) => ", ml-dsa-65-signed",
        None => "",
    };

    // Parent linkage (Phase 17.7): resolve `--evidence-parent` to the parent's
    // 32-byte trace_hash (either a literal hex hash or a parent artifact path),
    // recorded as `evidence_chain.parent` so chained artifacts reference their
    // parent. The link sits in the epilogue, outside the trace_hash preimage, so
    // it never perturbs THIS artifact's anchor.
    let parent: Option<[u8; 32]> = cli
        .evidence_parent
        .as_deref()
        .map(resolve_evidence_parent)
        .transpose()
        .unwrap_or_else(|()| {
            // resolve_evidence_parent already emitted a specific diagnostic.
            process::exit(1)
        });

    // Application-namespace attributes (Phase 17.8): parse `KEY=VALUE` pairs and
    // validate them fail-closed (non-reserved, dotted, no duplicates) before
    // emit. Empty ⇒ byte-identical to the closed-key encoder.
    let app_entries: Vec<(String, String)> = cli
        .evidence_attr
        .iter()
        .map(|kv| parse_evidence_attr(kv))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|msg| {
            eprintln!("error[emit-evidence]: {msg}");
            process::exit(1);
        });
    if let Err(msg) = libmind::ir::compact::validate_app_entries(&app_entries) {
        eprintln!("error[emit-evidence]: {msg}");
        process::exit(1);
    }

    // Emit body + evidence MAP, carrying the Salov loop-collapse receipts (S4)
    // the pipeline produced (empty for a source with no constant-folding
    // collapse — then byte-identical to the pre-S4 encoder, trace_hash unchanged).
    let bytes = match libmind::ir::compact::emit_mic3_with_evidence_and_receipts(
        &products.ir,
        substrate,
        parent,
        determinism,
        toolchain,
        signing_key.as_ref(),
        &products.collapse_receipts,
        &app_entries,
    ) {
        Ok(b) => b,
        Err(msg) => {
            eprintln!("error[emit-evidence]: {msg}");
            process::exit(1);
        }
    };
    // Built-in self-check (RFC 0016 Phase B verifier-core round-trip): peel the
    // freshly-emitted MAP, recompute the canonical mic@3 `trace_hash` over the
    // parsed IR body, and confirm it matches the stored hash before we hand the
    // artifact to the user. Generation without verification is security theatre
    // (RFC 0021 §4); this catches an emit/serialization regression at its source
    // rather than letting an unverifiable artifact escape. The check runs only on
    // the opt-in `--emit-evidence` path, so the default build is untouched.
    match libmind::ir::compact::mic3_evidence_report(&bytes) {
        Ok(report) if report.trace_hash_valid => {}
        Ok(_) => {
            eprintln!(
                "error[emit-evidence]: self-check failed — emitted evidence trace_hash \
                 does not validate against the IR body (internal emitter bug, not your input)"
            );
            process::exit(1);
        }
        Err(err) => {
            eprintln!(
                "error[emit-evidence]: self-check could not parse the artifact just emitted: {err:?}"
            );
            process::exit(1);
        }
    }
    // When signing was requested, the self-check must also confirm the signature
    // verifies — a signing/serialization regression must not ship a bad signature.
    if signing_key.is_some() {
        match libmind::ir::compact::mic3_signature_status(&bytes) {
            Ok(libmind::ir::compact::SignatureStatus::Valid(_)) => {}
            other => {
                eprintln!(
                    "error[emit-evidence]: signature self-check failed — emitted signature \
                     does not verify ({other:?}) (internal signing bug, not your input)"
                );
                process::exit(1);
            }
        }
    }
    // Collapse-receipt self-check (S4): re-derive every embedded receipt in O(1)
    // and confirm it re-derives + binds to the body before shipping. Generation
    // without verification is security theatre; catch an emitter regression here.
    let collapse_note = match libmind::ir::compact::mic3_collapse_verify(&bytes) {
        Ok(libmind::ir::compact::CollapseVerifyStatus::Verified(n)) => {
            format!(", {n} collapse receipt(s) re-derived")
        }
        Ok(libmind::ir::compact::CollapseVerifyStatus::Absent) => String::new(),
        other => {
            eprintln!(
                "error[emit-evidence]: collapse-receipt self-check failed ({other:?}) \
                 (internal emitter bug, not your input)"
            );
            process::exit(1);
        }
    };
    if let Err(err) = fs::write(path, &bytes) {
        eprintln!("error[emit-evidence]: failed to write {path}: {err}");
        process::exit(1);
    }
    eprintln!(
        "Wrote mic@3 evidence artifact: {path} ({} bytes, self-check ok{sig_label}{collapse_note})",
        bytes.len(),
    );
}

/// Read a 32-byte seed from a hex env var. `None` if unset; hard-exits on a
/// set-but-invalid value (fail-closed — never silently fall back to unsigned).
fn read_seed_env(var: &str) -> Option<[u8; 32]> {
    // A SET-BUT-UNREADABLE value must never look like an absent one.
    // `std::env::var` collapses NotPresent and NotUnicode into one Err, so a
    // non-UTF-8 seed would have been silently ignored and the build would have
    // produced unsigned (or differently signed) output while the operator
    // believed a key was configured. Presence is decided by `var_os`, which
    // does not require UTF-8.
    //
    // The variable NAME is reported; its VALUE never is.
    let raw = std::env::var_os(var)?;
    let Some(hex) = raw.to_str() else {
        eprintln!(
            "error[emit-evidence]: {var} is set but is not valid UTF-8, so it \
             cannot be a hex seed. Refusing rather than ignoring a configured \
             key. (signing.invalid_seed_encoding)"
        );
        process::exit(1);
    };
    match parse_ed25519_seed(hex.trim()) {
        Ok(seed) => Some(seed),
        Err(msg) => {
            eprintln!("error[emit-evidence]: {var} is set but invalid: {msg}");
            process::exit(1);
        }
    }
}

/// Decode a 64-hex-char string into a 32-byte seed. No `hex` crate dep.
fn parse_ed25519_seed(s: &str) -> Result<[u8; 32], String> {
    // Validate ASCII hex BEFORE indexing. `len()` counts BYTES and the loop
    // below slices by byte offset, so a multi-byte value whose byte length is 64
    // would slice at a non-char boundary and PANIC. A malformed configuration
    // must produce a structured error, never a crash.
    if !s.is_ascii() {
        return Err("seed must be ASCII hex".to_string());
    }
    if s.len() != 64 {
        return Err(format!(
            "expected 64 hex chars (32-byte seed), got {}",
            s.len()
        ));
    }
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("seed must be hex digits only".to_string());
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| "seed must be lowercase/uppercase hex".to_string())?;
    }
    Ok(out)
}

/// Read a 96-byte SLH-DSA seed (`SK.seed ‖ SK.prf ‖ PK.seed`, 192 hex chars) from
/// an env var. Fail-closed: a set-but-invalid value exits non-zero (never a
/// silently-dropped or truncated key).
fn read_seed_env_96(var: &str) -> Option<[u8; 96]> {
    // Same rule as `read_seed_env`: a set-but-non-UTF-8 value refuses rather
    // than reading as absent. The variable name is reported, never its value.
    let raw = std::env::var_os(var)?;
    let Some(hex) = raw.to_str() else {
        eprintln!(
            "error[emit-evidence]: {var} is set but is not valid UTF-8, so it \
             cannot be a hex seed. Refusing rather than ignoring a configured \
             key. (signing.invalid_seed_encoding)"
        );
        process::exit(1);
    };
    match parse_seed_96(hex.trim()) {
        Ok(seed) => Some(seed),
        Err(msg) => {
            eprintln!("error[emit-evidence]: {var} is set but invalid: {msg}");
            process::exit(1);
        }
    }
}

/// Decode a 192-hex-char string into a 96-byte seed.
fn parse_seed_96(s: &str) -> Result<[u8; 96], String> {
    if s.len() != 192 {
        return Err(format!(
            "expected 192 hex chars (96-byte seed), got {}",
            s.len()
        ));
    }
    if !s.is_ascii() {
        return Err("seed must be ASCII hex".to_string());
    }
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("seed must be hex digits only".to_string());
    }
    let mut out = [0u8; 96];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| "seed must be lowercase/uppercase hex".to_string())?;
    }
    Ok(out)
}

/// Resolve `--evidence-parent` (Phase 17.7) to a parent artifact's 32-byte
/// `trace_hash`. The value is either a 64-hex-char hash used verbatim, or a path
/// to a parent mic@3 evidence artifact whose `trace_hash` is read out via the
/// verifier core. Fail-closed: on any error a specific diagnostic is printed and
/// `Err(())` is returned (the caller exits non-zero) — a build must never silently
/// drop a requested parent link.
fn resolve_evidence_parent(value: &str) -> Result<[u8; 32], ()> {
    // A bare 64-hex-char string is a literal trace_hash.
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return parse_ed25519_seed(value).map_err(|msg| {
            eprintln!("error[emit-evidence]: --evidence-parent hex is invalid: {msg}");
        });
    }
    // Otherwise treat it as a path to a parent evidence artifact. The parent is
    // untrusted input; cap it at the 10 MiB mic@3 ceiling BEFORE slurping it into
    // memory (a FIFO / multi-GB file would otherwise balloon before the parser's
    // own cap rejects it).
    if let Ok(meta) = fs::metadata(value) {
        const MAX_PARENT_ARTIFACT: u64 = 10 * 1024 * 1024;
        if meta.len() > MAX_PARENT_ARTIFACT {
            eprintln!(
                "error[emit-evidence]: --evidence-parent `{value}` is {} bytes, over the \
                 {MAX_PARENT_ARTIFACT}-byte mic@3 cap",
                meta.len()
            );
            return Err(());
        }
    }
    let bytes = fs::read(value).map_err(|err| {
        eprintln!(
            "error[emit-evidence]: --evidence-parent `{value}` is neither a 64-hex-char \
             trace_hash nor a readable artifact: {err}"
        );
    })?;
    match libmind::ir::compact::mic3_evidence_report(&bytes) {
        Ok(report) if report.trace_hash_valid => Ok(report.trace_hash),
        Ok(_) => {
            eprintln!(
                "error[emit-evidence]: --evidence-parent `{value}` is body-tampered — its \
                 stored trace_hash does not match its canonical mic@3 body; refusing to \
                 chain to a corrupt parent"
            );
            Err(())
        }
        Err(err) => {
            eprintln!(
                "error[emit-evidence]: --evidence-parent `{value}` carries no readable \
                 evidence chain to link to ({err:?})"
            );
            Err(())
        }
    }
}

/// Parse one `--evidence-attr KEY=VALUE` (Phase 17.8) into a key/value pair. The
/// key namespace is validated separately by `validate_app_entries`; here we only
/// require a single `=` separator and a non-empty key.
fn parse_evidence_attr(kv: &str) -> Result<(String, String), String> {
    match kv.split_once('=') {
        Some((k, v)) if !k.is_empty() => Ok((k.to_string(), v.to_string())),
        _ => Err(format!(
            "--evidence-attr expects KEY=VALUE with a non-empty key, got `{kv}`"
        )),
    }
}

/// Lowercase hex-encode a byte slice (no `hex` crate dependency).
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Decode a hex string (optional `0x` prefix, case-insensitive) into bytes.
/// Returns `None` on odd length or a non-hex digit.
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if s.is_empty() || s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

/// Build the signer-key trust allowlist for `mindc verify` from `--signer-pubkey`
/// flags plus the `MIND_EVIDENCE_VERIFY_PUBKEYS` env var (comma/space-separated
/// hex). An invalid hex entry is a hard error (fail-closed on operator input),
/// never silently dropped. Returns decoded key bytes compared verbatim against
/// the artifact's embedded supported key(s); retired schemes never verify.
fn collect_trusted_pubkeys(flags: &[String]) -> Result<Vec<Vec<u8>>, String> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let add = |tok: &str, out: &mut Vec<Vec<u8>>| -> Result<(), String> {
        let tok = tok.trim();
        if tok.is_empty() {
            return Ok(());
        }
        match hex_decode(tok) {
            Some(b) => {
                out.push(b);
                Ok(())
            }
            None => Err(format!(
                "invalid --signer-pubkey / trusted-pubkey hex: {tok}"
            )),
        }
    };
    for f in flags {
        add(f, &mut out)?;
    }
    if let Ok(env) = std::env::var("MIND_EVIDENCE_VERIFY_PUBKEYS") {
        for tok in env.split(|c: char| c == ',' || c.is_whitespace()) {
            add(tok, &mut out)?;
        }
    }
    Ok(out)
}

/// Escape a string for embedding inside a hand-built JSON string literal.
///
/// `--json` output is assembled by interpolation rather than via serde, so any
/// free-form field (the artifact path, or the `substrate` / `toolchain` values
/// which a crafted artifact controls verbatim) must be escaped or it could
/// inject structure into the object — e.g. spoofing `trace_hash_valid` for a
/// consumer that parses the JSON instead of checking the exit code.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// `mindc inspect <artifact> [--diff OTHER]` — decode + pretty-print a mic@3
/// artifact and, with `--diff`, structurally compare two. The consumer/debug
/// counterpart of `--emit-mic3`: it surfaces MIND's deterministic canonical IR
/// and its tamper-evident evidence chain, and — via `--diff` — pinpoints WHERE
/// two artifacts diverge, exactly what the self-host byte-identity gates need
/// when a reseed or loop stops being byte-identical. Not a Rust `objdump` clone:
/// the value is the canonical-IR + evidence surface only MIND's wedge exposes.
///
/// Returns the process exit code: 0 = decoded (and, with `--diff`, identical);
/// 1 = artifacts differ (`--diff`) or a malformed artifact; 2 = I/O error.
fn run_inspect(artifact: &str, json: bool, diff: Option<&str>) -> i32 {
    use libmind::ir::compact::{Determinism, mic3_evidence_report, parse_mic3_envelope};

    let bytes = match fs::read(artifact) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("error[inspect]: cannot read artifact {artifact}: {err}");
            return 2;
        }
    };

    // --diff MODE: mic@3 is canonical (RFC 0021), so byte-identity IS structural
    // identity. When the bytes differ, locate the first diverging byte and parse
    // each side to report parse status + instruction-count delta (a parse failure
    // is itself a reported difference, never a crash).
    if let Some(other) = diff {
        let other_bytes = match fs::read(other) {
            Ok(b) => b,
            Err(err) => {
                eprintln!("error[inspect]: cannot read artifact {other}: {err}");
                return 2;
            }
        };
        if bytes == other_bytes {
            // Canonical mic@3: byte-identity IS structural identity — but the
            // contract is "exit 0 = decoded", so two byte-identical GARBAGE files
            // (e.g. both zero-length from an aborted build, or a self-diff of a
            // corrupt artifact) must still fail closed rather than report a
            // confident "identical: YES".
            if let Err(err) = parse_mic3_envelope(&bytes) {
                eprintln!("error[inspect]: {artifact} did not parse as mic@3: {err:?}");
                if json {
                    println!(
                        "{{\"a\":\"{}\",\"b\":\"{}\",\"identical\":true,\"decoded\":false,\"bytes\":{}}}",
                        json_escape(artifact),
                        json_escape(other),
                        bytes.len()
                    );
                }
                return 1;
            }
            if json {
                println!(
                    "{{\"a\":\"{}\",\"b\":\"{}\",\"identical\":true,\"decoded\":true,\"bytes\":{}}}",
                    json_escape(artifact),
                    json_escape(other),
                    bytes.len()
                );
            } else {
                println!("identical:  YES ({} bytes)", bytes.len());
            }
            return 0;
        }
        let common = bytes.len().min(other_bytes.len());
        let first_diff = (0..common)
            .find(|&i| bytes[i] != other_bytes[i])
            .unwrap_or(common);
        let a_instrs = parse_mic3_envelope(&bytes)
            .map(|m| m.instrs.len() as i64)
            .unwrap_or(-1);
        let b_instrs = parse_mic3_envelope(&other_bytes)
            .map(|m| m.instrs.len() as i64)
            .unwrap_or(-1);
        if json {
            println!(
                "{{\"a\":\"{}\",\"b\":\"{}\",\"identical\":false,\"a_bytes\":{},\"b_bytes\":{},\"first_diff_offset\":{},\"a_instrs\":{},\"b_instrs\":{}}}",
                json_escape(artifact),
                json_escape(other),
                bytes.len(),
                other_bytes.len(),
                first_diff,
                a_instrs,
                b_instrs
            );
        } else {
            let fmt_side = |n: i64| {
                if n < 0 {
                    "PARSE FAIL".to_string()
                } else {
                    format!("{n} instrs")
                }
            };
            println!("identical:        NO");
            println!(
                "a:                {artifact} ({} bytes, {})",
                bytes.len(),
                fmt_side(a_instrs)
            );
            println!(
                "b:                {other} ({} bytes, {})",
                other_bytes.len(),
                fmt_side(b_instrs)
            );
            println!("first_diff_byte:  {first_diff}");
            let lo = first_diff.saturating_sub(4);
            let a_hi = (first_diff + 4).min(bytes.len());
            let b_hi = (first_diff + 4).min(other_bytes.len());
            println!("  a[{lo}..]: {}", hex_encode(&bytes[lo..a_hi]));
            println!("  b[{lo}..]: {}", hex_encode(&other_bytes[lo..b_hi]));
        }
        return 1;
    }

    // INSPECT MODE: decode + pretty-print one artifact.
    let module = match parse_mic3_envelope(&bytes) {
        Ok(m) => m,
        Err(err) => {
            let reason = format!("{err:?}");
            eprintln!("error[inspect]: {artifact} did not parse as mic@3: {reason}");
            // Mirror run_verify: a scripted --json consumer reads stdout, so emit a
            // well-formed error object there too (not just stderr).
            if json {
                println!(
                    "{{\"artifact\":\"{}\",\"error\":\"{}\"}}",
                    json_escape(artifact),
                    json_escape(&reason)
                );
            }
            return 1;
        }
    };
    let evidence = mic3_evidence_report(&bytes).ok();
    let det_str = |d: &Determinism| {
        if matches!(d, Determinism::Deterministic) {
            "deterministic"
        } else {
            "nondeterministic"
        }
    };

    if json {
        let (attested, trace_hash, determinism, fp_mode) = match &evidence {
            Some(r) => (
                true,
                hex_encode(&r.trace_hash),
                det_str(&r.determinism),
                r.fp_mode.as_str(),
            ),
            None => (false, String::new(), "", ""),
        };
        println!(
            "{{\"artifact\":\"{}\",\"bytes\":{},\"instrs\":{},\"next_id\":{},\"exports\":{},\"attested\":{},\"trace_hash\":\"{}\",\"determinism\":\"{}\",\"fp_mode\":\"{}\"}}",
            json_escape(artifact),
            bytes.len(),
            module.instrs.len(),
            module.next_id,
            module.exports.len(),
            attested,
            trace_hash,
            determinism,
            fp_mode
        );
    } else {
        println!("artifact:         {artifact}");
        println!("bytes:            {}", bytes.len());
        println!("instrs:           {}", module.instrs.len());
        println!("ssa_next_id:      {}", module.next_id);
        println!("exports:          {}", module.exports.len());
        match &evidence {
            Some(r) => {
                println!("attested:         YES");
                println!("trace_hash:       {}", hex_encode(&r.trace_hash));
                println!("determinism:      {}", det_str(&r.determinism));
                println!("fp_mode:          {}", r.fp_mode.as_str());
            }
            None => println!("attested:         no (no evidence_chain MAP)"),
        }
        println!("--- canonical IR ---");
        println!("{module}");
    }
    0
}

/// `mindc verify <artifact>` — consumer-side static + evidence verification.
///
/// Two independent properties are reported (RFC 0017):
///   * SSA well-formedness — a property of the IR body alone, needing no
///     evidence chain. Always reported (`ssa_valid`); an SSA fault fails verify.
///   * trace_hash attestation — checked only when the artifact carries an
///     `evidence_chain` MAP. An unattested-but-SSA-valid artifact passes and is
///     reported with `attested: false`.
///
/// Returns the process exit code: 0 = valid (SSA well-formed, and — when
/// attested — trace_hash intact); 1 = verification failed (SSA fault, tampered
/// trace_hash, or malformed evidence chain); 2 = I/O error reading the artifact.
fn run_verify(
    artifact: &str,
    json: bool,
    require_strict_fp: bool,
    require_deterministic: bool,
    require_signed: bool,
    trusted: &[Vec<u8>],
) -> i32 {
    use libmind::ir::compact::{
        CollapseVerifyStatus, Determinism, EvidenceError, MAX_MIC3_INPUT, Mic3NonCanonical,
        TraceHashKind, mic3_canonical_check, mic3_evidence_report, parse_mic3_envelope,
    };
    use libmind::ir::{IrVerifyError, check_ssa_well_formed, verify_module};

    // Stat-before-read DoS guard: `parse_mic3` rejects input over MAX_MIC3_INPUT,
    // but only AFTER the bytes are in memory. Reading the whole file first means a
    // crafted `truncate -s 100G evil.mic3` aborts `mindc verify` on an allocation
    // failure before that cap is ever consulted. Reject on the file's declared
    // size up front so an oversized artifact fails closed (exit 2) instead of
    // OOM-killing the process.
    match fs::metadata(artifact) {
        Ok(meta) if meta.len() > MAX_MIC3_INPUT as u64 => {
            eprintln!(
                "error[verify]: artifact {artifact} is {} bytes, exceeds the mic@3 \
                 input cap of {MAX_MIC3_INPUT} bytes",
                meta.len()
            );
            return 2;
        }
        Ok(_) => {}
        Err(err) => {
            eprintln!("error[verify]: cannot stat artifact {artifact}: {err}");
            return 2;
        }
    }

    let bytes = match fs::read(artifact) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("error[verify]: cannot read artifact {artifact}: {err}");
            return 2;
        }
    };

    // CANONICAL-FORM GATE (SECURITY). Neither integrity layer covers the LITERAL
    // bytes: `trace_hash` anchors the re-emission of the PARSED IR, and the RFC 0021
    // signature preimage covers trace_hash + scheme tag + DECODED provenance entries.
    // So every mutation the decoder normalises away produced a DIFFERENT file, with a
    // different SHA-256, that still reported "signature is valid and signer key is
    // trusted" and exited 0 — appending a byte after the MAP epilogue, padding to the
    // MAX_MIC3_INPUT ceiling, re-encoding the MAP entry-count ULEB non-minimally,
    // downgrading the wire-version byte inside the accepted read window, or flipping a
    // normalised body byte. A valid signature therefore did not identify ONE byte
    // string, which is the whole point of signing it: any hash-pinning consumer — a
    // lockfile, a transparency log, a reproducible-build comparison — could be handed
    // a different file that `verify` blesses.
    //
    // `mic3_canonical_check` re-emits the artifact and byte-compares against the
    // input, so distinct byte streams can never share one verdict. It existed,
    // documented as a hard gate of `mindc verify`, with ZERO call sites outside its
    // own unit tests.
    //
    // `Unparseable` is deliberately NOT fatal here: canonicality is undecidable on a
    // body that does not decode, and the parse/SSA path immediately below reports the
    // authoritative diagnostic for it. Every other variant fails closed.
    if let Err(nc) = mic3_canonical_check(&bytes) {
        if !matches!(nc, Mic3NonCanonical::Unparseable) {
            eprintln!(
                "error[verify]: {artifact} is not in canonical mic@3 form: {nc}\n  \
                 the artifact's literal bytes differ from the canonical re-emission of \
                 the IR they decode to, so a signature or trace_hash over it would not \
                 identify these exact bytes (fail-closed)"
            );
            return 1;
        }
    }

    // SSA well-formedness (RFC 0017, second static-verification slice): parse
    // the mic@3 IR body and statically confirm single-assignment +
    // define-before-use over the instruction tree. This is independent of the
    // evidence-chain trace_hash check below; `verify` fails if EITHER property
    // fails. A parse failure here is a malformed artifact, not an SSA fault.
    // Parse the mic@3 body ONCE and derive from it both (a) SSA well-formedness
    // and (b) the determinism declaration RE-DERIVED from the hashed body. The
    // re-derivation is the Risk-2 fix: the `evidence_chain.determinism` MAP field
    // sits OUTSIDE the trace_hash anchor (like every MAP key), so on an unsigned
    // artifact it is post-hoc forgeable while the trace_hash still matches. By
    // recomputing `ir_declares_deterministic` from the same body the trace_hash
    // authenticates — exactly as `fp_mode` is re-derived — the true mode is
    // authenticated, and a stored MAP field that disagrees is a tamper indicator.
    let (ssa_valid, ssa_reason, rederived_deterministic): (bool, Option<String>, Option<bool>) =
        match parse_mic3_envelope(&bytes) {
            Ok(module) => {
                let det = libmind::ir::ir_declares_deterministic(&module);
                match check_ssa_well_formed(&module) {
                    Ok(()) => {
                        // check_ssa_well_formed covers single-assignment +
                        // define-before-use over the full instruction tree. Also run
                        // the in-pipeline `verify_module` so the untrusted-artifact
                        // surface gains its SEMANTIC operand sanity (negative axis /
                        // zero conv stride — `IrVerifyError::InvalidOperand`), which
                        // the SSA-only consumer check does not cover. `MissingOutput`
                        // is NOT a fault at this surface (a decoded fn-only / export
                        // artifact legitimately lacks a top-level `Output`), so it is
                        // not treated as a failure. The two verifiers agree on the SSA
                        // verdict (differential gates in tests/verify_ssa.rs), so this
                        // never contradicts the check above.
                        match verify_module(&module) {
                            Ok(()) | Err(IrVerifyError::MissingOutput) => (true, None, Some(det)),
                            Err(e) => (false, Some(e.to_string()), Some(det)),
                        }
                    }
                    Err(v) => (false, Some(v.to_string()), Some(det)),
                }
            }
            // Could not parse the IR body for the SSA check. The evidence path
            // below produces the authoritative parse-error diagnostic; here we only
            // record that SSA could not be established.
            Err(_) => (
                false,
                Some("mic@3 body did not parse for SSA check".into()),
                None,
            ),
        };

    // SSA well-formedness is a property of the IR body alone — independent of,
    // and gated BEFORE, the evidence chain (which an artifact may legitimately
    // lack). A structural SSA fault fails `verify` regardless of attestation,
    // so report it standalone and exit 1 here, *before* the evidence path.
    if !ssa_valid {
        let reason = ssa_reason.as_deref().unwrap_or("malformed IR");
        if json {
            println!(
                "{{\"artifact\":\"{}\",\"ssa_valid\":false,\"ssa_reason\":\"{}\",\"attested\":false}}",
                json_escape(artifact),
                json_escape(reason)
            );
        } else {
            println!("artifact:         {artifact}");
            println!("ssa_valid:        NO");
            println!("ssa_reason:       {reason}");
        }
        eprintln!("error[verify]: SSA well-formedness check FAILED — {reason}");
        return 1;
    }

    // Crypto-agile signature layer (RFC 0021 §6): checked when present, tolerated
    // when absent (back-compat). The `signature.scheme` (`alg`) tag selects the
    // verifier(s). Retired Ed25519 and old-hybrid tags are reported as retired;
    // a present-but-bad signature, unknown scheme, or required-but-uncompiled PQC
    // verifier all fail closed.
    use libmind::ir::compact::{SignatureStatus, mic3_signature_status};
    // Fail-closed (MED #4): a malformed/type-confused signature field yields `Err`
    // from the signature layer. Coercing that to `Absent` would read as "unsigned
    // but attested" (sig_ok = true) — a fail-OPEN. Map `Err` to a signature
    // FAILURE instead; only a genuine `Ok(Absent)` is a true unsigned artifact.
    let sig_status = match mic3_signature_status(&bytes) {
        Ok(s) => s,
        Err(_) => SignatureStatus::Malformed("signature"),
    };
    // `sig_label` names the scheme when valid; the failure kind otherwise. The two
    // pubkey fields carry the keys the signature(s) were checked against.
    #[allow(clippy::type_complexity)]
    let (sig_label, sig_ed_pubkey, sig_mldsa_pubkey, sig_mldsa87_pubkey, sig_slhdsa_pubkey): (
        String,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = match &sig_status {
        SignatureStatus::Absent => ("absent".to_string(), None, None, None, None),
        SignatureStatus::Valid(v) => (
            v.scheme.clone(),
            v.ed25519_pubkey.map(|pk| hex_encode(&pk)),
            v.mldsa_pubkey.as_deref().map(hex_encode),
            v.mldsa87_pubkey.as_deref().map(hex_encode),
            v.slhdsa_pubkey.as_deref().map(hex_encode),
        ),
        SignatureStatus::Invalid => ("invalid".to_string(), None, None, None, None),
        SignatureStatus::Malformed(_) => ("malformed".to_string(), None, None, None, None),
        SignatureStatus::Unsupported(_) => ("unsupported".to_string(), None, None, None, None),
        // Reported by NAME so the operator learns which retired scheme this
        // artifact carries, rather than a bare "invalid" that hides the reason.
        SignatureStatus::Retired(scheme) => (format!("retired ({scheme})"), None, None, None, None),
    };
    // `Retired` is deliberately NOT in this set. An artifact signed under a
    // retired scheme is not acceptable, and it is not the same as unsigned:
    // treating it as `Absent` would silently demote a signed artifact and hand
    // back a success, which is the quiet downgrade the retirement removes.
    let sig_ok = matches!(
        sig_status,
        SignatureStatus::Absent | SignatureStatus::Valid(_)
    );

    match mic3_evidence_report(&bytes) {
        Ok(report) => {
            // Risk-2: REPORT the determinism RE-DERIVED from the hashed body — the
            // authoritative value the trace_hash authenticates — not the forgeable
            // stored MAP field. The evidence path is only reached when the body
            // parsed (an SSA/parse fault returned 1 earlier), so `rederived_*` is
            // `Some`. `stored_deterministic` is kept only to detect a tampered MAP
            // field below.
            let stored_deterministic = matches!(report.determinism, Determinism::Deterministic);
            let effective_deterministic = rederived_deterministic.unwrap_or(stored_deterministic);
            let determinism = if effective_deterministic {
                "deterministic"
            } else {
                "nondeterministic"
            };
            let parent = report.parent.map(|p| hex_encode(&p));
            let trace_hash = hex_encode(&report.trace_hash);
            // `mic3-bytes` for every current artifact; a key-less legacy artifact
            // decodes to the same default (the anchor in use since 2026-05-31).
            let trace_hash_kind = match report.trace_hash_kind {
                TraceHashKind::Mic3Bytes => "mic3-bytes",
                TraceHashKind::Mic1Text => "mic1-text",
            };
            // Strict-FP contract mode, re-derived from the same hashed body
            // (strict / relaxed / unknown). Charset-safe (enum tag).
            let fp_mode = report.fp_mode.as_str();

            if json {
                // Hand-formatted JSON keeps the binary free of a serde dependency
                // and the output byte-stable for scripted consumers.  Free-form
                // fields are json_escape'd; `determinism`/`trace_hash`/`parent`
                // are charset-safe (enum / hex) by construction.
                let parent_field = match &parent {
                    Some(p) => format!("\"{p}\""),
                    None => "null".to_string(),
                };
                let ssa_reason_field = match &ssa_reason {
                    Some(r) => format!("\"{}\"", json_escape(r)),
                    None => "null".to_string(),
                };
                let sig_ed_pubkey_field = match &sig_ed_pubkey {
                    Some(pk) => format!("\"{pk}\""),
                    None => "null".to_string(),
                };
                let sig_mldsa_pubkey_field = match &sig_mldsa_pubkey {
                    Some(pk) => format!("\"{pk}\""),
                    None => "null".to_string(),
                };
                // Tier-2 provenance authentication for scripted consumers: true ONLY when the
                // artifact carries a valid signature AND a trusted signer key is pinned (the
                // signature preimage covers substrate/toolchain/parent). Unsigned or
                // untrusted-signer => false: the provenance MAP fields are not authenticated,
                // so a consumer must not trust substrate/toolchain/parent from trace_hash alone.
                let provenance_authenticated =
                    matches!(sig_status, SignatureStatus::Valid(_)) && !trusted.is_empty();
                // pqc-hybrid legs: surface both PQC pubkeys so scripted consumers
                // can out-of-band-pin the hybrid key material from JSON (matching
                // the ed25519 / ml-dsa-65 fields above).
                let sig_mldsa87_pubkey_field = match &sig_mldsa87_pubkey {
                    Some(pk) => format!("\"{pk}\""),
                    None => "null".to_string(),
                };
                let sig_slhdsa_pubkey_field = match &sig_slhdsa_pubkey {
                    Some(pk) => format!("\"{pk}\""),
                    None => "null".to_string(),
                };
                println!(
                    "{{\"artifact\":\"{}\",\"substrate\":\"{}\",\"determinism\":\"{determinism}\",\"toolchain\":\"{}\",\"parent\":{parent_field},\"trace_hash\":\"{trace_hash}\",\"trace_hash_kind\":\"{trace_hash_kind}\",\"trace_hash_valid\":{},\"fp_mode\":\"{fp_mode}\",\"ssa_valid\":{ssa_valid},\"ssa_reason\":{ssa_reason_field},\"signature\":\"{sig_label}\",\"signature_ed25519_pubkey\":{sig_ed_pubkey_field},\"signature_mldsa_pubkey\":{sig_mldsa_pubkey_field},\"signature_mldsa87_pubkey\":{sig_mldsa87_pubkey_field},\"signature_slhdsa_pubkey\":{sig_slhdsa_pubkey_field},\"provenance_authenticated\":{provenance_authenticated}}}",
                    json_escape(artifact),
                    json_escape(&report.substrate),
                    json_escape(&report.toolchain),
                    report.trace_hash_valid
                );
            } else {
                println!("artifact:         {artifact}");
                println!("substrate:        {}", report.substrate);
                println!("determinism:      {determinism}");
                println!("toolchain:        {}", report.toolchain);
                println!(
                    "parent:           {}",
                    parent.as_deref().unwrap_or("(root)")
                );
                println!("trace_hash:       {trace_hash}");
                println!("trace_hash_kind:  {trace_hash_kind}");
                println!(
                    "trace_hash_valid: {}",
                    if report.trace_hash_valid { "yes" } else { "NO" }
                );
                println!("fp_mode:          {fp_mode}");
                println!("ssa_valid:        {}", if ssa_valid { "yes" } else { "NO" });
                if let Some(r) = &ssa_reason {
                    println!("ssa_reason:       {r}");
                }
                println!("signature:        {sig_label}");
                if let Some(pk) = &sig_ed_pubkey {
                    println!("signature_ed25519_pubkey: {pk}");
                }
                if let Some(pk) = &sig_mldsa_pubkey {
                    println!("signature_mldsa_pubkey:   {pk}");
                }
                if let Some(pk) = &sig_mldsa87_pubkey {
                    println!("signature_mldsa87_pubkey: {pk}");
                }
                if let Some(pk) = &sig_slhdsa_pubkey {
                    println!("signature_slhdsa_pubkey:  {pk}");
                }
            }

            // SSA is already established valid above (an SSA fault returns 1
            // before this point). An attested artifact therefore reports BOTH
            // ssa_valid and the evidence-chain trace_hash result; it passes only
            // if the trace_hash also holds.
            if report.trace_hash_valid {
                // Signature layer fails closed: a present-but-bad signature is a
                // verification failure even though the trace_hash matched (an
                // attacker who re-hashed a tampered anchor cannot re-sign it).
                if !sig_ok {
                    // A retired scheme is not a failed signature, and saying so
                    // would misdescribe the cause. Name the real reason.
                    if let SignatureStatus::Retired(scheme) = &sig_status {
                        eprintln!(
                            "error[verify]: signature scheme `{scheme}` is PERMANENTLY RETIRED \
                             from trust verification — the artifact is structurally intact and \
                             remains inspectable, but a retired scheme can never verify as \
                             trusted. Supported signing is the post-quantum hybrid \
                             ML-DSA-87 + SLH-DSA-SHAKE-256s. (verify.retired_scheme)"
                        );
                    } else {
                        eprintln!(
                            "error[verify]: signature is {sig_label} — artifact signature does not verify over the trace_hash (fail-closed)"
                        );
                    }
                    return 1;
                }
                // Trust anchor, part 1 — signature-stripping downgrade: pinning a
                // signer key (--signer-pubkey / MIND_EVIDENCE_VERIFY_PUBKEYS) makes a
                // signature REQUIRED. An `Absent` signature — stripped from a real
                // artifact, or simply never signed — passes the `!sig_ok` gate above
                // and would otherwise skip the key-allowlist check below and be
                // reported valid. An attacker therefore emits their OWN body
                // attested-but-UNSIGNED (no private key needed) and it satisfies the
                // pin. A pinned key with no signature is a downgrade; refuse it.
                if !trusted.is_empty() && !matches!(sig_status, SignatureStatus::Valid(_)) {
                    eprintln!(
                        "error[verify]: a signer key is pinned (--signer-pubkey / MIND_EVIDENCE_VERIFY_PUBKEYS) but the artifact signature is {sig_label} — a pinned signer requires a valid signature (fail-closed)"
                    );
                    return 1;
                }
                // Trust anchor, part 2 (HIGH #3): a valid signature only proves the
                // artifact was signed by the holder of the EMBEDDED key. Without a
                // pinned allowlist that says nothing about WHO — an attacker self-signs
                // their own artifact with their own key. When `trusted` is set, the
                // signer key(s) MUST be in it or we refuse to report valid.
                if let SignatureStatus::Valid(v) = &sig_status {
                    if !trusted.is_empty() {
                        let mut present: Vec<Vec<u8>> = Vec::new();
                        if let Some(pk) = v.ed25519_pubkey {
                            present.push(pk.to_vec());
                        }
                        if let Some(pk) = &v.mldsa_pubkey {
                            present.push(pk.clone());
                        }
                        if let Some(pk) = &v.mldsa87_pubkey {
                            present.push(pk.clone());
                        }
                        if let Some(pk) = &v.slhdsa_pubkey {
                            present.push(pk.clone());
                        }
                        // Fail-closed on an empty key set: a pinned allowlist must
                        // match at least one ACTUAL signer key. `all()` over an empty
                        // vec is vacuously true, so guard it explicitly — otherwise a
                        // scheme whose pubkeys we failed to collect would pass the pin.
                        // For pqc-hybrid this ALSO means BOTH the ML-DSA-87 and SLH-DSA
                        // keys must be in the allowlist (both are in `present`), so a
                        // pinned STARGA hybrid identity requires both keys.
                        let all_trusted = !present.is_empty()
                            && present.iter().all(|pk| trusted.iter().any(|t| t == pk));
                        if !all_trusted {
                            eprintln!(
                                "error[verify]: signer key is NOT in the trusted allowlist (--signer-pubkey / MIND_EVIDENCE_VERIFY_PUBKEYS) — refusing to report valid"
                            );
                            if let Some(pk) = v.ed25519_pubkey {
                                eprintln!("  artifact ed25519 pubkey:   {}", hex_encode(&pk));
                            }
                            if let Some(pk) = &v.mldsa_pubkey {
                                eprintln!("  artifact ml-dsa-65 pubkey: {}", hex_encode(pk));
                            }
                            if let Some(pk) = &v.mldsa87_pubkey {
                                eprintln!("  artifact ml-dsa-87 pubkey: {}", hex_encode(pk));
                            }
                            if let Some(pk) = &v.slhdsa_pubkey {
                                eprintln!("  artifact slh-dsa pubkey:   {}", hex_encode(pk));
                            }
                            return 1;
                        }
                    }
                }
                if !json {
                    // Tier 1 (trace_hash-covered, tamper-evident): the canonical IR body,
                    // plus the determinism / fp_mode labels which are RE-DERIVED from the
                    // hashed bytes below. The provenance MAP fields (substrate / toolchain /
                    // parent) sit OUTSIDE trace_hash and are authenticated only by a trusted
                    // signature (tier 2, in the signature match below). A bare "untampered"
                    // over-claims for an unsigned artifact whose substrate field is editable.
                    eprintln!(
                        "verified: IR body attested (tamper-evident) — trace_hash matches the re-emitted canonical IR"
                    );
                    eprintln!("verified: IR body is SSA well-formed");
                    match &sig_status {
                        SignatureStatus::Valid(v) => {
                            if trusted.is_empty() {
                                // No trust root supplied: do NOT claim authenticity.
                                // Report internal consistency and print the signer
                                // key(s) so a consumer can pin them out-of-band.
                                eprintln!(
                                    "verified: signature is internally consistent (scheme: {}) — verify the signer key out-of-band",
                                    v.scheme
                                );
                                if let Some(pk) = v.ed25519_pubkey {
                                    eprintln!("  signer ed25519 pubkey:   {}", hex_encode(&pk));
                                }
                                if let Some(pk) = &v.mldsa_pubkey {
                                    eprintln!("  signer ml-dsa-65 pubkey: {}", hex_encode(pk));
                                }
                                if let Some(pk) = &v.mldsa87_pubkey {
                                    eprintln!("  signer ml-dsa-87 pubkey: {}", hex_encode(pk));
                                }
                                if let Some(pk) = &v.slhdsa_pubkey {
                                    eprintln!("  signer slh-dsa pubkey:   {}", hex_encode(pk));
                                }
                            } else {
                                eprintln!(
                                    "verified: signature is valid and signer key is trusted (scheme: {})",
                                    v.scheme
                                );
                            }
                        }
                        SignatureStatus::Absent => {
                            eprintln!(
                                "note: artifact carries no signature (unsigned but attested)"
                            );
                            // Tier 2 (provenance authentication): the substrate / toolchain /
                            // parent MAP fields are NOT covered by trace_hash, so on an
                            // unsigned artifact they are editable without changing the
                            // (still-valid) trace_hash. Say so plainly — do not let the
                            // tier-1 attestation imply the provenance is authenticated.
                            eprintln!(
                                "note: provenance (substrate/toolchain/parent) is NOT authenticated — these MAP fields sit outside trace_hash; sign the artifact and pin --signer-pubkey to authenticate them"
                            );
                        }
                        _ => {}
                    }
                    if !effective_deterministic {
                        eprintln!(
                            "note: artifact is nondeterministic (calls a PRNG / wall-clock / stdin builtin); trace_hash matches but reproducibility is not asserted"
                        );
                    }
                }
                // Risk-2 fail-closed: the stored `determinism` MAP field sits
                // OUTSIDE the trace_hash anchor, so on this (trace_hash-VALID)
                // artifact a field that disagrees with the value RE-DERIVED from
                // the hashed body has been tampered with — the body genuinely
                // calls (or does not call) a nondeterministic builtin, but the
                // label says otherwise. Reject: the attestation cannot lie.
                if rederived_deterministic.is_some()
                    && stored_deterministic != effective_deterministic
                {
                    eprintln!(
                        "error[verify]: determinism label TAMPERED — the evidence_chain.determinism field says `{}` but the hashed body is `{}`",
                        if stored_deterministic {
                            "deterministic"
                        } else {
                            "nondeterministic"
                        },
                        if effective_deterministic {
                            "deterministic"
                        } else {
                            "nondeterministic"
                        },
                    );
                    return 1;
                }
                // Opt-in strict-FP gate: an untampered artifact still fails
                // verification if the consumer demanded strict-FP and the
                // re-derived mode isn't strict (relaxed OR unknown → fail
                // closed). The trace_hash already attests the mode is genuine.
                if require_strict_fp && !report.fp_mode.is_strict() {
                    eprintln!(
                        "error[verify]: fp_mode is {} — artifact used FMA-contraction / f32 reassociation (or was not scanned); strict-FP required",
                        report.fp_mode.as_str()
                    );
                    return 1;
                }
                // Opt-in determinism gate (mirrors --require-strict-fp): a consumer
                // that requires reproducibility rejects a nondeterministic artifact.
                // Uses the RE-DERIVED value (authoritative), so a forged
                // `deterministic` label cannot slip past it.
                if require_deterministic && !effective_deterministic {
                    eprintln!(
                        "error[verify]: artifact is nondeterministic (re-derived from the hashed body) — deterministic build required"
                    );
                    return 1;
                }
                // Opt-in signed gate: require a VALID signature (any signer). Weaker
                // than a pinned --signer-pubkey (which also requires the signer be
                // trusted); this is the "every artifact must be signed" policy.
                // Fail-closed on unsigned / signature-stripped / malformed.
                if require_signed && !matches!(sig_status, SignatureStatus::Valid(_)) {
                    eprintln!(
                        "error[verify]: artifact carries no valid signature (signature: {sig_label}) — --require-signed demands a signed artifact"
                    );
                    return 1;
                }
                // Salov loop-collapse receipts (S4): independently RE-DERIVE every
                // folded constant in O(1) (the loop is never re-run) and confirm it
                // (a) matches the receipt's recorded value and (b) is materialised
                // in the hashed body. A tampered constant/parameter fails closed.
                match libmind::ir::compact::mic3_collapse_verify(&bytes) {
                    Ok(CollapseVerifyStatus::Verified(n)) => {
                        if !json {
                            eprintln!(
                                "verified: {n} loop-collapse receipt(s) re-derived (O(1) closed form, loop not re-run)"
                            );
                        }
                    }
                    Ok(CollapseVerifyStatus::Absent) => {}
                    Ok(CollapseVerifyStatus::Rederivation { rederived, claimed }) => {
                        eprintln!(
                            "error[verify]: collapse-receipt FORGERY — recorded constant {claimed} but the loop parameters re-derive to {rederived} (fail-closed)"
                        );
                        return 1;
                    }
                    Ok(CollapseVerifyStatus::NotInBody { constant }) => {
                        eprintln!(
                            "error[verify]: collapse-receipt constant {constant} is not materialised in the hashed body (fail-closed)"
                        );
                        return 1;
                    }
                    Ok(CollapseVerifyStatus::Malformed) => {
                        eprintln!(
                            "error[verify]: collapse-receipt blob is malformed (fail-closed)"
                        );
                        return 1;
                    }
                    Ok(CollapseVerifyStatus::NonCanonical) => {
                        eprintln!(
                            "error[verify]: collapse-receipt blob is not in canonical form (fail-closed)"
                        );
                        return 1;
                    }
                    Err(_) => {
                        eprintln!(
                            "error[verify]: collapse-receipt layer could not be parsed (fail-closed)"
                        );
                        return 1;
                    }
                }
                0
            } else {
                eprintln!("error[verify]: trace_hash MISMATCH — artifact has been tampered with");
                1
            }
        }
        Err(EvidenceError::Missing) => {
            // Unattested but SSA well-formed (an SSA fault returned 1 above).
            // SSA is a property of the IR body alone and needs no evidence
            // chain, so report ssa_valid standalone and pass. Attestation is
            // reported separately as absent.
            if json {
                println!(
                    "{{\"artifact\":\"{}\",\"ssa_valid\":{ssa_valid},\"ssa_reason\":null,\"attested\":false}}",
                    json_escape(artifact)
                );
            } else {
                println!("artifact:         {artifact}");
                println!("ssa_valid:        {}", if ssa_valid { "yes" } else { "NO" });
                println!("attested:         no");
            }
            eprintln!("verified: IR body is SSA well-formed");
            eprintln!(
                "note: {artifact} carries no evidence_chain — unattested artifact (trace_hash not checked)"
            );
            // Fail-closed strict-FP gate on the UNATTESTED path. An artifact
            // with no evidence chain has no `trace_hash` attesting its body, so
            // its FP-contract mode cannot be re-derived from *attested* bytes —
            // it is effectively `unknown`. `--require-strict-fp` must never
            // silently pass such an artifact: the whole point of the flag is a
            // build-host-independent, attested strict-FP guarantee, and an
            // unattested artifact offers none. Reject it, mirroring the
            // attested-path relaxed/unknown rejection above (both fail closed).
            // Plain `verify` (no flag) still exits 0 here — attestation is
            // absent, not failed (RFC 0017).
            //
            // `--require-signed` gate (fail-closed) — SECURITY. The flag is an
            // explicit demand for a VALID signature, and it was honoured at only
            // one site (the attested arm), so an artifact with NO evidence_chain
            // took this arm and exited 0. Stripping the chain therefore turned
            // REJECTED into ACCEPTED: `verify --require-signed evil.mic3` passed
            // on a fully attacker-authored, unsigned artifact, so a CI gate
            // `verify --require-signed KEY && deploy` would deploy it. An
            // unattested artifact carries no signature at all and so can never
            // satisfy the demand. This is the identical downgrade the
            // pinned-signer guard immediately below already closes, and the two
            // must not disagree about whether a missing chain is benign.
            if require_signed {
                eprintln!(
                    "error[verify]: --require-signed was given but {artifact} carries no evidence_chain — an unattested artifact has no signature to verify (fail-closed)"
                );
                return 1;
            }

            // Pinned-signer gate (fail-closed) — SECURITY (audit rank 1). A
            // pinned signer (`--signer-pubkey` / `MIND_EVIDENCE_VERIFY_PUBKEYS`)
            // is an explicit demand for a valid, trusted signature. An
            // UNATTESTED artifact carries no evidence_chain and therefore no
            // signature at all, so it can never satisfy a pinned signer. Without
            // this the `Missing` arm returned 0 for
            // `verify --signer-pubkey KEY evil.mic3` on a fully attacker-authored,
            // unsigned artifact, so a CI gate `verify --signer-pubkey KEY &&
            // deploy` would deploy attacker code. Mirrors the attested-path
            // pinned-signer rejection (mindc.rs ~1866/1887) — a stripped
            // evidence_chain must never be a silent downgrade-to-benign path.
            if !trusted.is_empty() {
                eprintln!(
                    "error[verify]: a signer key is pinned (--signer-pubkey / MIND_EVIDENCE_VERIFY_PUBKEYS) but {artifact} carries no evidence_chain — an unattested artifact has no signature to verify; a pinned signer requires a valid, trusted signature (fail-closed)"
                );
                return 1;
            }
            if require_strict_fp {
                eprintln!(
                    "error[verify]: --require-strict-fp on an unattested artifact — no evidence_chain to attest the FP-contract mode; strict-FP cannot be proven (fail-closed)"
                );
                return 1;
            }
            // Same fail-closed shape for --require-deterministic: an unattested
            // artifact carries no evidence chain to certify reproducibility, even
            // though its body parsed. A consumer that DEMANDED determinism cannot
            // accept an unattested build.
            if require_deterministic {
                eprintln!(
                    "error[verify]: --require-deterministic on an unattested artifact — no evidence_chain to attest determinism (fail-closed)"
                );
                return 1;
            }
            0
        }
        Err(EvidenceError::MissingKey(k)) => {
            eprintln!("error[verify]: evidence chain is missing required key '{k}'");
            1
        }
        Err(EvidenceError::Malformed(k)) => {
            eprintln!("error[verify]: evidence chain key '{k}' is malformed");
            1
        }
        Err(EvidenceError::UnknownDeterminism(d)) => {
            eprintln!("error[verify]: evidence chain has unknown determinism value '{d}'");
            1
        }
    }
}

fn parse_target(raw: &str) -> Result<BackendTarget, String> {
    match raw.to_ascii_lowercase().as_str() {
        "cpu" => Ok(BackendTarget::Cpu),
        "gpu" | "cuda" | "rocm" | "metal" | "webgpu" => Ok(BackendTarget::Gpu),
        "tpu" => Ok(BackendTarget::Tpu),
        "npu" | "ane" | "hexagon" => Ok(BackendTarget::Npu),
        "lpu" | "groq" => Ok(BackendTarget::Lpu),
        "dpu" | "smartnic" | "bluefield" => Ok(BackendTarget::Dpu),
        "fpga" | "hls" => Ok(BackendTarget::Fpga),
        // Wafer-scale: distinct logical target from GPU because the
        // runtime backend lowers to CSL and reasons about a 2-D fabric
        // mesh rather than CUDA-style SMs. Accept all WSE generations
        // here; the wafer generation (WSE-2 / WSE-3) is selected at
        // runtime, not at the source-level target.
        "cerebras" | "wse" | "wse2" | "wse3" => Ok(BackendTarget::Cerebras),
        other => Err(format!(
            "unknown target '{other}' (expected cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras)"
        )),
    }
}

fn resolve_color_choice(flag: &Option<String>) -> ColorChoice {
    if let Some(value) = flag.as_deref() {
        return ColorChoice::parse(value).unwrap_or(ColorChoice::Auto);
    }
    if let Ok(env) = std::env::var("MINDC_COLOR") {
        return ColorChoice::parse(&env).unwrap_or(ColorChoice::Auto);
    }
    ColorChoice::Auto
}

#[cfg(feature = "mlir-build")]
fn emit_obj_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    let obj_path = match &cli.emit_obj {
        Some(path) => path,
        None => return,
    };

    // First lower to MLIR
    let mlir = match lower_to_mlir_compat(products) {
        Ok(mlir) => mlir,
        Err(err) => {
            eprintln!("error[mlir]: {err}");
            process::exit(1);
        }
    };

    // Resolve build tools
    let tools = match libmind::eval::mlir_build::resolve_tools() {
        Ok(tools) => tools,
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    };

    // Build object file
    let opts = libmind::eval::mlir_build::BuildOptions {
        preset: libmind::eval::mlir_build::preset_for_mlir(&mlir.primal_mlir),
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: Some(Path::new(obj_path)),
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };

    match libmind::eval::mlir_build::build_all(&mlir.primal_mlir, &tools, &opts) {
        Ok(_) => {
            eprintln!("Wrote object file: {}", obj_path);
        }
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    }
}

#[cfg(not(feature = "mlir-build"))]
fn emit_obj_if_requested(cli: &CompileArgs, _products: &libmind::pipeline::CompileProducts) {
    if cli.emit_obj.is_some() {
        eprintln!(
            "error[build][{}]: --emit-obj requires building with the 'mlir-build' feature",
            libmind::diagnostics::capability::NO_NATIVE_BACKEND
        );
        process::exit(1);
    }
}

/// Emit `--emit-shared <out.so>`.
///
/// Links the substrate objects for every `std` substrate module the entry
/// imports (transitively). Without that the emitted `.so` carries the imported
/// module's symbols as UNDEFINED and `dlopen` fails at load — measured as
/// `libio_canon.so: undefined symbol: sha256`, because `std/io_canon.mind`
/// imports `std.sha256` and this flat path (unlike `mindc build --emit=cdylib`)
/// ran no cross-module link walk at all. The walk is shared with the manifest
/// cdylib path via `libmind::project::substrate_link`; an entry importing no
/// substrate module yields an empty object set, leaving the link byte-identical
/// to the historical single-entry path.
#[cfg(feature = "mlir-build")]
fn emit_shared_if_requested(
    cli: &CompileArgs,
    products: &libmind::pipeline::CompileProducts,
    source: &str,
    #[cfg(feature = "cross-module-imports")] project_scope: Option<
        &libmind::project::single_file_scope::ProjectScope,
    >,
) {
    let shared_path = match &cli.emit_shared {
        Some(path) => path,
        None => return,
    };

    let mlir = match lower_to_mlir_compat(products) {
        Ok(mlir) => mlir,
        Err(err) => {
            eprintln!("error[mlir]: {err}");
            process::exit(1);
        }
    };

    let tools = match libmind::eval::mlir_build::resolve_tools() {
        Ok(tools) => tools,
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    };

    let opts = libmind::eval::mlir_build::BuildOptions {
        preset: libmind::eval::mlir_build::preset_for_mlir(&mlir.primal_mlir),
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: Some(Path::new(shared_path)),
        opt_pipeline: None,
        target_triple: None,
    };

    // The cross-module substrate archive, seeded and planned over the entry
    // and the project siblings it links (the scope's entry comes first). Held
    // until the link below is done, and dropped before any failing exit (an
    // exit skips destructors and would leave its private directory behind).
    // Fail-loud: a module that will not compile must abort the build, never
    // yield an `.so` whose symbol is still undefined at `dlopen`.
    #[cfg(feature = "cross-module-imports")]
    let substrate = {
        let linked_siblings: Vec<(&Path, &str)> = project_scope
            .map(|scope| {
                scope
                    .linked_sources()
                    .skip(1)
                    .map(|s| (s.path(), s.source()))
                    .collect()
            })
            .unwrap_or_default();
        match libmind::project::substrate_link::substrate_objects_for_entry(
            source,
            &linked_siblings,
            libmind::runtime::types::BackendTarget::Cpu,
            &tools,
        ) {
            Ok(archive) => archive,
            Err(err) => {
                eprintln!("error[build]: {err}");
                process::exit(1);
            }
        }
    };
    // Project sibling objects, with the archive's internal-linkage plan.
    #[cfg(feature = "cross-module-imports")]
    let mut extra_objects: Vec<std::path::PathBuf> = {
        let obj_dir = Path::new(shared_path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let mut objects = Vec::new();
        if let Some(scope) = project_scope {
            match libmind::project::compile_project_sibling_objects(
                scope,
                &obj_dir,
                libmind::runtime::types::BackendTarget::Cpu,
                &tools,
                Some(&substrate),
            ) {
                Ok(mut siblings) => objects.append(&mut siblings),
                Err(err) => {
                    eprintln!("error[build]: {err}");
                    drop(substrate);
                    process::exit(1);
                }
            }
        }
        objects
    };
    // The archive goes LAST so a sibling's substrate references pull their
    // members too.
    #[cfg(feature = "cross-module-imports")]
    extra_objects.extend(substrate.link_inputs());
    #[cfg(not(feature = "cross-module-imports"))]
    let extra_objects: Vec<std::path::PathBuf> = {
        // No embedded stdlib without `cross-module-imports`, so no substrate
        // module can be imported and the object set is necessarily empty —
        // byte-identical to the historical single-entry link.
        let _ = source;
        Vec::new()
    };

    // The entry's own non-`pub` fns that collide with (or are referenced by)
    // a linked std module get internal linkage.
    #[cfg(feature = "cross-module-imports")]
    let entry_mlir = substrate.apply_linkage(
        Path::new(libmind::project::substrate_link::ENTRY_KEY),
        &mlir.primal_mlir,
    );
    #[cfg(not(feature = "cross-module-imports"))]
    let entry_mlir = mlir.primal_mlir.clone();
    let linked = libmind::eval::mlir_build::build_all_with_objects(
        &entry_mlir,
        &tools,
        &opts,
        &extra_objects,
    );
    // Remove the private substrate directory before a failing exit, which
    // would otherwise skip its destructor and leave it behind.
    #[cfg(feature = "cross-module-imports")]
    drop(substrate);
    match linked {
        Ok(_) => {
            eprintln!("Wrote shared library: {}", shared_path);
        }
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    }
}

#[cfg(not(feature = "mlir-build"))]
fn emit_shared_if_requested(
    cli: &CompileArgs,
    _products: &libmind::pipeline::CompileProducts,
    _source: &str,
    #[cfg(feature = "cross-module-imports")] _scope: Option<
        &libmind::project::single_file_scope::ProjectScope,
    >,
) {
    if cli.emit_shared.is_some() {
        eprintln!(
            "error[build][{}]: --emit-shared requires building with the 'mlir-build' feature",
            libmind::diagnostics::capability::NO_NATIVE_BACKEND
        );
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{hex_encode, json_escape};

    #[test]
    fn hex_encode_is_lowercase_and_fixed_width() {
        assert_eq!(hex_encode(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_encode(&[0x5; 32]).len(), 64);
    }

    #[test]
    fn json_escape_passes_clean_strings_through() {
        assert_eq!(json_escape("cpu"), "cpu");
        assert_eq!(json_escape("0.7.0"), "0.7.0");
        assert_eq!(json_escape("/tmp/a.bin"), "/tmp/a.bin");
    }

    #[test]
    fn json_escape_neutralizes_structure_injection() {
        // A crafted substrate/toolchain or a path with a quote must not break
        // out of its JSON string literal (the MEDIUM finding being guarded).
        assert_eq!(
            json_escape(r#"cpu","trace_hash_valid":true,"x":""#),
            r#"cpu\",\"trace_hash_valid\":true,\"x\":\""#
        );
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("line\nbreak\ttab\r"), "line\\nbreak\\ttab\\r");
        assert_eq!(json_escape("\u{0001}"), "\\u0001");
    }
}
