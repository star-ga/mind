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

#[cfg(feature = "mlir-build")]
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

    // Phase 17.7: `--emit-evidence` must attest a RESOLVED multi-module project,
    // not just a lone translation unit. The flat compile path is single-TU by
    // default (only `mindc build` seeds the cross-module table), so a program that
    // `use`s a sibling module fails to resolve here. When emitting evidence, seed
    // the whole-project module table from the input file's sibling `.mind` sources
    // first, so `use crate.util` resolves and the attested `trace_hash` covers the
    // resolved program. Best-effort + cleared after compile; a true single file
    // (no siblings) seeds nothing and is byte-identical to the prior path.
    let seeded_project_table =
        cli.compile.emit_evidence.is_some() && seed_project_table_for_evidence(&input);

    let products = match compile_source_with_name(&source, Some(&input), &opts) {
        Ok(products) => products,
        Err(err) => {
            let diags = err.into_diagnostics(Some(&input));
            emitter.emit_all(&diags, Some(&source));
            process::exit(1);
        }
    };
    if seeded_project_table {
        // Restore the empty table so nothing downstream sees a stale project scope.
        #[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
        libmind::type_checker::cm_set_project_table(None);
    }

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
        if let Some(offender) = libmind::ir::ir_first_nondeterministic_call(&products.ir) {
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

    emit_mic3_if_requested(&cli.compile, &products);
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
    emit_shared_if_requested(&cli.compile, &products);
}

/// RI-D Option A (task #110): bridge `mindc build --backend native` to the frozen
/// pure-MIND x86-64 native-ELF compiler, producing a runnable ELF with ZERO
/// MLIR/LLVM/clang in the path. The Rust driver only composes the source image and
/// shells out; all code generation is the pure-MIND emitter (examples/mindc_mind,
/// RI-B/E). Byte-identity to the pure-MIND compiler's direct stdout is the invariant
/// (gate: backend_native_bridge_smoke.py) — the wrapper adds zero bytes.
///
/// Wire contract (selfhost_driver.mind `main`, proven byte-faithful by
/// self_host_loop_smoke.py): stdin fd 0 = [8B user_lo LE][8B src_len LE][std_blob ++
/// user_src]; the emitted static ELF is written to stdout fd 1. The compiler ELF is
/// resolved from MINDC_NATIVE_ELF, else the frozen stage0 seed
/// examples/mindc_mind/testdata/selfhost_loop/stage1.elf.
///
/// Fail-closed: a non-zero exit from the pure-MIND compiler is propagated verbatim and
/// NO artifact is written. There is deliberately NO MLIR fallback — a `native` request
/// that the pure-MIND subset cannot lower must fail loud, never silently degrade.
fn run_native_backend_bridge(paths: &[String], out: &Option<String>) {
    use std::io::Write as _;

    // Slice 1: exactly one source file. Multi-file/workspace native builds are a later slice.
    if paths.len() != 1 {
        eprintln!(
            "error[backend-native]: the native backend bridge currently accepts exactly one \
             source file (got {}). Multi-file/workspace native builds are a later RI-D slice.",
            paths.len()
        );
        process::exit(2);
    }
    let user_src = match std::fs::read(&paths[0]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "error[backend-native]: cannot read source '{}': {e}",
                paths[0]
            );
            process::exit(2);
        }
    };

    // Compose the std blob exactly as the pure-MIND compiler expects: the 21 std modules
    // in this fixed order, '\n'-joined with a trailing '\n'. This list is the twin of
    // self_host_standalone_driver_smoke.py::_STDLIB_MODULES; the bridge's byte-identity
    // gate fails loud if the two ever drift (bridge output != stage1-direct output).
    // deferred: unify both readers on one committed manifest — upgrade path:
    //   examples/mindc_mind/testdata/stdlib_manifest.txt read by smoke + bridge.
    const STD_MODULES: [&str; 21] = [
        "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring", "json", "map", "net",
        "process", "reactor", "regex", "ring", "sha256", "string", "time", "toml", "tui", "vec",
    ];
    let std_dir = std::env::var("MINDC_STD_DIR").unwrap_or_else(|_| "std".to_string());
    let mut combined: Vec<u8> = Vec::new();
    for m in STD_MODULES.iter() {
        let p = format!("{std_dir}/{m}.mind");
        match std::fs::read(&p) {
            Ok(b) => combined.extend_from_slice(&b),
            Err(e) => {
                eprintln!(
                    "error[backend-native]: cannot read std module '{p}': {e} \
                     (set MINDC_STD_DIR to the std/ directory)"
                );
                process::exit(2);
            }
        }
        combined.push(b'\n');
    }
    let user_lo = combined.len() as i64;
    combined.extend_from_slice(&user_src);
    let src_len = combined.len() as i64;

    // stdin image: [8B user_lo LE][8B src_len LE][combined]
    let mut image: Vec<u8> = Vec::with_capacity(16 + combined.len());
    image.extend_from_slice(&user_lo.to_le_bytes());
    image.extend_from_slice(&src_len.to_le_bytes());
    image.extend_from_slice(&combined);

    // Resolve the pure-MIND compiler ELF (the RI-D shell-out target).
    let elf = std::env::var("MINDC_NATIVE_ELF")
        .unwrap_or_else(|_| "examples/mindc_mind/testdata/selfhost_loop/stage1.elf".to_string());
    if !std::path::Path::new(&elf).exists() {
        eprintln!(
            "error[backend-native]: pure-MIND compiler ELF not found at '{elf}' \
             (set MINDC_NATIVE_ELF)"
        );
        process::exit(2);
    }

    let mut child = match std::process::Command::new(&elf)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error[backend-native]: failed to spawn compiler ELF '{elf}': {e}");
            process::exit(2);
        }
    };
    if let Some(mut si) = child.stdin.take() {
        if let Err(e) = si.write_all(&image) {
            eprintln!("error[backend-native]: failed to stream source image to compiler: {e}");
            process::exit(2);
        }
    }
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error[backend-native]: compiler ELF did not complete: {e}");
            process::exit(2);
        }
    };

    // Fail-closed: propagate the pure-MIND compiler's diagnostic + exit; no artifact.
    if !output.status.success() {
        let code = output.status.code().unwrap_or(1);
        if !output.stderr.is_empty() {
            let _ = std::io::stderr().write_all(&output.stderr);
        }
        eprintln!(
            "error[backend-native]: pure-MIND compiler rejected the program (exit {code}); \
             no artifact written (native backend is fail-closed — no MLIR fallback)."
        );
        process::exit(if code == 0 { 1 } else { code });
    }
    let elf_bytes = output.stdout;
    // A real static ELF (magic + ET_EXEC), never an empty/garbage artifact.
    if elf_bytes.len() < 256 || &elf_bytes[0..4] != b"\x7fELF" {
        eprintln!(
            "error[backend-native]: compiler produced a non-ELF/short artifact ({} bytes); \
             refusing to write.",
            elf_bytes.len()
        );
        process::exit(1);
    }

    let out_path = out.clone().unwrap_or_else(|| "a.out".to_string());
    if let Err(e) = std::fs::write(&out_path, &elf_bytes) {
        eprintln!("error[backend-native]: cannot write output '{out_path}': {e}");
        process::exit(2);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(md) = std::fs::metadata(&out_path) {
            let mut perm = md.permissions();
            perm.set_mode(perm.mode() | 0o755);
            let _ = std::fs::set_permissions(&out_path, perm);
        }
    }
    eprintln!(
        "native: wrote {} ({} bytes) via the pure-MIND native-ELF backend \
         (zero MLIR/LLVM/clang)",
        out_path,
        elf_bytes.len()
    );
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
            run_native_backend_bridge(paths, out);
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
            eprintln!("error[build]: {}", err);
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

    if selected.is_empty()
        && let Some(pkg) = package
    {
        eprintln!("error[workspace]: package '{pkg}' not found in workspace");
        process::exit(2);
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
                eprintln!("error[workspace][{}]: {}", member.name, err);
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

    if selected.is_empty()
        && let Some(pkg) = package
    {
        eprintln!("error[workspace]: package '{pkg}' not found in workspace");
        process::exit(2);
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
        Ok(()) => {
            println!("Core v1 conformance passed for profile: {:?}", profile);
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
            "error[mlir]: MLIR emission requires building with the 'mlir-lowering' or 'mlir-build' feature"
        );
        process::exit(1);
    }
}

fn emit_mic3_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    let path = match &cli.emit_mic3 {
        Some(p) => p,
        None => return,
    };
    let bytes = libmind::ir::compact::emit_mic3(&products.ir);
    if let Err(err) = fs::write(path, &bytes) {
        eprintln!("error[emit-mic3]: failed to write {path}: {err}");
        process::exit(1);
    }
    eprintln!("Wrote mic@3 artifact: {path} ({} bytes)", bytes.len());
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
    // (never a hardcoded key):
    //   * MIND_EVIDENCE_MLDSA_KEY   → post-quantum ML-DSA-65 (FIPS-204). PREFERRED
    //                                 for federal PQC compliance (EO 14412 / OMB
    //                                 M-26-15 / the FAR PQC rule).
    //   * MIND_EVIDENCE_ED25519_KEY → classical Ed25519 (legacy/interop).
    //   * BOTH set                  → hybrid (both must verify) — defence-in-depth.
    // Each seed is 32 bytes as 64 hex chars. No env ⇒ the unsigned path, byte-
    // identical to the pre-signing encoder (the determinism gate is untouched).
    use libmind::ir::compact::SigningKey;
    let ed_seed = read_seed_env(libmind::ir::compact::v3::evidence::ENV_ED25519_SEED);
    let mldsa_seed = read_seed_env(libmind::ir::compact::v3::evidence::ENV_MLDSA_SEED);
    let signing_key: Option<SigningKey> = match (ed_seed, mldsa_seed) {
        (Some(ed), Some(ml)) => Some(SigningKey::Hybrid {
            ed25519: ed,
            mldsa65: ml,
        }),
        (Some(ed), None) => Some(SigningKey::Ed25519(ed)),
        (None, Some(ml)) => Some(SigningKey::MlDsa65(ml)),
        (None, None) => None,
    };
    let sig_label = match &signing_key {
        Some(SigningKey::Hybrid { .. }) => ", hybrid-ed25519-ml-dsa-65-signed",
        Some(SigningKey::MlDsa65(_)) => ", ml-dsa-65-signed",
        Some(SigningKey::Ed25519(_)) => ", ed25519-signed",
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
    match std::env::var(var) {
        Ok(hex) => match parse_ed25519_seed(hex.trim()) {
            Ok(seed) => Some(seed),
            Err(msg) => {
                eprintln!("error[emit-evidence]: {var} is set but invalid: {msg}");
                process::exit(1);
            }
        },
        Err(_) => None,
    }
}

/// Decode a 64-hex-char string into a 32-byte seed. No `hex` crate dep.
fn parse_ed25519_seed(s: &str) -> Result<[u8; 32], String> {
    if s.len() != 64 {
        return Err(format!(
            "expected 64 hex chars (32-byte seed), got {}",
            s.len()
        ));
    }
    let mut out = [0u8; 32];
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

/// Seed the whole-project cross-module table (Phase 17.7) from the sibling
/// `.mind` files next to `input`, so a `--emit-evidence` compile of a project
/// entry resolves `use crate.<module>` references the same way `mindc build`
/// does. Best-effort: unparseable siblings are skipped (they simply do not
/// contribute exports). Returns `true` when a table WAS installed (there is at
/// least one sibling module beyond the entry), so the caller clears it after the
/// compile. A lone file with no siblings seeds nothing and returns `false`,
/// leaving the single-TU path byte-identical.
#[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
fn seed_project_table_for_evidence(input: &str) -> bool {
    use std::path::Path;
    let input_path = Path::new(input);
    let dir = match input_path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let mut parsed: Vec<(String, libmind::ast::Module)> =
        libmind::project::stdlib::parsed_stdlib_modules();
    let mut sibling_count = 0usize;
    let entry_name = input_path.file_name();
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut paths: Vec<std::path::PathBuf> = rd.flatten().map(|e| e.path()).collect();
        // Deterministic order so the seeded table is build-independent.
        paths.sort();
        for p in paths {
            if p.extension().map(|x| x == "mind").unwrap_or(false) {
                if let Ok(text) = std::fs::read_to_string(&p) {
                    if let Ok(m) = libmind::parser::parse(&text) {
                        if p.file_name() != entry_name {
                            sibling_count += 1;
                        }
                        let key = libmind::project::module_table::module_path_of(&p, dir);
                        parsed.push((key, m));
                    }
                }
            }
        }
    }
    if sibling_count == 0 {
        return false;
    }
    let refs: Vec<(String, &libmind::ast::Module)> =
        parsed.iter().map(|(k, m)| (k.clone(), m)).collect();
    let table = libmind::project::module_table::build_module_table(&refs);
    libmind::type_checker::cm_set_project_table(Some(table));
    true
}

/// No-op stub for builds without the cross-module machinery (`module_table` /
/// `cm_set_project_table` / bundled `stdlib`): there is no multi-module project
/// scope to seed, so `--emit-evidence` stays single-TU — byte-identical to the
/// pre-17.7 path. Keeps the `mindc` binary buildable under `--no-default-features`
/// and the default (`std-surface`-only) feature set.
#[cfg(not(all(feature = "std-surface", feature = "cross-module-imports")))]
fn seed_project_table_for_evidence(_input: &str) -> bool {
    false
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
/// never silently dropped. Returns the decoded key bytes (Ed25519 = 32 B,
/// ML-DSA-65 = 1952 B) compared verbatim against the artifact's embedded key(s).
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
    use libmind::ir::compact::{Determinism, mic3_evidence_report, parse_mic3};

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
            if let Err(err) = parse_mic3(&bytes) {
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
        let a_instrs = parse_mic3(&bytes)
            .map(|m| m.instrs.len() as i64)
            .unwrap_or(-1);
        let b_instrs = parse_mic3(&other_bytes)
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
    let module = match parse_mic3(&bytes) {
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
        CollapseVerifyStatus, Determinism, EvidenceError, MAX_MIC3_INPUT, TraceHashKind,
        mic3_evidence_report, parse_mic3,
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
        match parse_mic3(&bytes) {
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
    // verifier(s) — ed25519 / ml-dsa-65 / hybrid. A present-but-bad signature, an
    // unknown scheme, or a required-but-uncompiled PQC verifier all fail closed.
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
    let (sig_label, sig_ed_pubkey, sig_mldsa_pubkey): (String, Option<String>, Option<String>) =
        match &sig_status {
            SignatureStatus::Absent => ("absent".to_string(), None, None),
            SignatureStatus::Valid(v) => (
                v.scheme.clone(),
                v.ed25519_pubkey.map(|pk| hex_encode(&pk)),
                v.mldsa_pubkey.as_deref().map(hex_encode),
            ),
            SignatureStatus::Invalid => ("invalid".to_string(), None, None),
            SignatureStatus::Malformed(_) => ("malformed".to_string(), None, None),
            SignatureStatus::Unsupported(_) => ("unsupported".to_string(), None, None),
        };
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
                println!(
                    "{{\"artifact\":\"{}\",\"substrate\":\"{}\",\"determinism\":\"{determinism}\",\"toolchain\":\"{}\",\"parent\":{parent_field},\"trace_hash\":\"{trace_hash}\",\"trace_hash_kind\":\"{trace_hash_kind}\",\"trace_hash_valid\":{},\"fp_mode\":\"{fp_mode}\",\"ssa_valid\":{ssa_valid},\"ssa_reason\":{ssa_reason_field},\"signature\":\"{sig_label}\",\"signature_ed25519_pubkey\":{sig_ed_pubkey_field},\"signature_mldsa_pubkey\":{sig_mldsa_pubkey_field},\"provenance_authenticated\":{provenance_authenticated}}}",
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
                    eprintln!(
                        "error[verify]: signature is {sig_label} — artifact signature does not verify over the trace_hash (fail-closed)"
                    );
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
                        let all_trusted = present.iter().all(|pk| trusted.iter().any(|t| t == pk));
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
        eprintln!("error[build]: --emit-obj requires building with the 'mlir-build' feature");
        process::exit(1);
    }
}

#[cfg(feature = "mlir-build")]
fn emit_shared_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
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

    match libmind::eval::mlir_build::build_all(&mlir.primal_mlir, &tools, &opts) {
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
fn emit_shared_if_requested(cli: &CompileArgs, _products: &libmind::pipeline::CompileProducts) {
    if cli.emit_shared.is_some() {
        eprintln!("error[build]: --emit-shared requires building with the 'mlir-build' feature");
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
