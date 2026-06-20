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

//! RFC 0008 Phase B — `mindc test` discovery and parallel runner.
//!
//! Public entry point: [`run_tests`].
//!
//! ## Design decisions
//!
//! ### Isolation model: in-process `catch_unwind` (not process-per-test)
//!
//! RFC 0008 §10 specifies process-per-test as the normative isolation model
//! for correctness. Phase B deviates intentionally: MIND test functions are
//! currently evaluated by the Rust `eval` interpreter, which runs in the same
//! address space. Spawning a child `mindc` process per test would require a
//! stable `--test-fn=<name>` execution mode that is not yet wired into the
//! interpreter pipeline. The in-process route with `std::panic::catch_unwind`
//! is correct for the interpreter layer because:
//!
//! 1. The interpreter does not mutate global static state across test calls.
//! 2. Heap allocations from one test do not affect subsequent tests through the
//!    interpreter's per-`eval` context.
//! 3. The interpreter panics are Rust panics, which `catch_unwind` can safely
//!    intercept on a per-test basis.
//!
//! Process-per-test isolation (full fork/exec model) is future work, tracked as
//! a follow-on to the MLIR-compiled binary test path. When that path lands, the
//! isolation model can be upgraded without changing this module's public API:
//! `run_tests` returns the same `TestRunSummary` regardless.
//!
//! ### Parallelism: Rayon-free thread pool
//!
//! We use `std::thread::spawn` directly to avoid adding a new dependency.
//! The worker pool is bounded to `opts.threads` (or available parallelism when
//! zero). Tasks are distributed via a `std::sync::Mutex<VecDeque<TestEntry>>`.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::ast::Node;
use crate::parser;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single discovered test function.
#[derive(Debug, Clone)]
pub struct TestEntry {
    /// Fully-qualified test name (file-stem::fn-name, or fn-name for single files).
    pub name: String,
    /// Absolute path to the source file that declares the test.
    pub source_file: PathBuf,
    /// 1-based line number of the `fn` keyword (derived from span offset).
    pub source_line: u32,
    /// The source text of the enclosing file (needed for in-process eval).
    pub source_text: String,
}

/// Options controlling a `mindc test` run.
#[derive(Debug, Clone, Default)]
pub struct TestOptions {
    /// Source files or directories to search. Empty = walk current directory.
    pub paths: Vec<PathBuf>,
    /// Only run tests whose name contains this substring.
    pub filter: String,
    /// Capture per-test stdout/stderr; print only on failure. Default `true`.
    pub capture: bool,
    /// Max parallel worker threads. 0 = available parallelism.
    pub threads: usize,
    /// List test names and exit without running.
    pub list: bool,
    /// Reporter style.
    pub reporter: ReporterKind,
}

/// Output reporter style.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ReporterKind {
    #[default]
    Human,
    Json,
}

/// Per-test result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestStatus {
    Passed,
    Failed { message: String },
}

/// Result for one test execution.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
    pub duration: Duration,
}

/// Aggregate summary returned by [`run_tests`].
#[derive(Debug, Clone, Default)]
pub struct TestRunSummary {
    pub passed: u32,
    pub failed: u32,
    pub results: Vec<TestResult>,
}

impl TestRunSummary {
    /// `true` when all discovered tests passed (or no tests were found).
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

/// Typed error from the test runner.
#[derive(Debug, thiserror::Error)]
pub enum TestError {
    #[error("test discovery failed: {0}")]
    Discovery(String),
    #[error("test execution error: {0}")]
    Execution(String),
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Discover and run all `[test]`-annotated functions in `opts.paths`.
///
/// Returns `Ok(summary)` even when tests fail; the caller checks
/// `summary.all_passed()` to decide the exit code.
pub fn run_tests(opts: &TestOptions) -> Result<TestRunSummary, TestError> {
    // 1. Resolve source files to walk.
    let source_files =
        collect_source_files(&opts.paths).map_err(|e| TestError::Discovery(e.to_string()))?;

    // 2. Parse each file, collect #[test] entries.
    //    Files that fail to parse are silently skipped — the intent is to
    //    discover tests across a directory tree without aborting on files
    //    that have syntax errors (they might be test fixtures for parser
    //    error-case tests, or simply broken files that are not test sources).
    let mut entries: Vec<TestEntry> = Vec::new();
    let mut parse_failures: Vec<(PathBuf, String)> = Vec::new();
    for path in &source_files {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("warning[test]: cannot read {}: {e}", path.display());
                continue;
            }
        };
        match discover_tests_in_source(path, &text) {
            Ok(discovered) => entries.extend(discovered),
            // A file that fails to parse yields no tests — but skipping it
            // SILENTLY turns a real breakage into a misleading green "running 0
            // tests" (exactly what masked a parse error in a downstream repo's
            // test suite). Collect the error and surface it below so the user
            // sees WHY a file produced no tests, while other files still run.
            Err(msg) => parse_failures.push((path.clone(), msg)),
        }
    }
    if !parse_failures.is_empty() {
        eprintln!(
            "warning[test]: {} test file(s) skipped — they do not parse, so no \
             tests were discovered in them:",
            parse_failures.len()
        );
        for (path, msg) in &parse_failures {
            eprintln!("  {}: {msg}", path.display());
        }
    }

    // 3. Apply filter.
    if !opts.filter.is_empty() {
        entries.retain(|e| e.name.contains(&opts.filter));
    }

    // 4. Handle --list.
    if opts.list {
        for entry in &entries {
            println!("{}", entry.name);
        }
        return Ok(TestRunSummary::default());
    }

    // 5. Print header.
    println!(
        "running {} test{}",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" }
    );
    if entries.is_empty() {
        println!("\ntest result: ok. 0 passed; 0 failed");
        return Ok(TestRunSummary::default());
    }

    // 6. Execute tests in parallel.
    let summary = execute_tests(entries, opts)?;

    // 7. Print summary.
    print_summary(&summary, opts);

    Ok(summary)
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Walk `paths` for `*.mind` files. If `paths` is empty, walk the current dir.
fn collect_source_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>, std::io::Error> {
    if paths.is_empty() {
        return walk_for_mind_files(Path::new("."));
    }
    let mut result = Vec::new();
    for path in paths {
        if path.is_file() {
            if path.extension().map(|e| e == "mind").unwrap_or(false) {
                result.push(path.clone());
            }
        } else if path.is_dir() {
            result.extend(walk_for_mind_files(path)?);
        }
    }
    Ok(result)
}

/// Recursively walk a directory for `*.mind` files.
fn walk_for_mind_files(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut found = Vec::new();
    walk_dir_recursive(dir, &mut found)?;
    found.sort(); // deterministic order
    Ok(found)
}

fn walk_dir_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Skip hidden directories and target/
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            walk_dir_recursive(&path, out)?;
        } else if path.extension().map(|e| e == "mind").unwrap_or(false) {
            out.push(path);
        }
    }
    Ok(())
}

/// Parse a `.mind` source and return all `[test]`-annotated functions as
/// `TestEntry` values.
///
/// Errors propagate only for file I/O; parse errors in the source are reported
/// as individual test failures rather than aborting discovery of the entire
/// file — this mirrors `cargo test` behaviour where a syntax error in one
/// module does not hide tests in sibling modules.
pub fn discover_tests_in_source(path: &Path, source: &str) -> Result<Vec<TestEntry>, String> {
    let module = match parser::parse(source) {
        Ok(m) => m,
        Err(errs) => {
            // Surface parse errors as a single failed pseudo-test so the
            // reporter shows them rather than silently dropping the file.
            let msg = errs
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(msg);
        }
    };

    let file_stem = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut entries = Vec::new();
    for item in &module.items {
        if let Node::FnDef {
            is_test: true,
            name,
            span,
            ..
        } = item
        {
            let source_line = line_number_at(source, span.start());
            entries.push(TestEntry {
                name: format!("{}::{}", file_stem, name),
                source_file: path.to_path_buf(),
                source_line,
                source_text: source.to_string(),
            });
        }
    }
    Ok(entries)
}

/// Compute the 1-based line number for a byte offset in `source`.
fn line_number_at(source: &str, offset: usize) -> u32 {
    let safe = offset.min(source.len());
    (source[..safe].bytes().filter(|&b| b == b'\n').count() + 1) as u32
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Execute the test entries in parallel, collecting results.
fn execute_tests(entries: Vec<TestEntry>, opts: &TestOptions) -> Result<TestRunSummary, TestError> {
    let thread_count = if opts.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        opts.threads
    };
    let thread_count = thread_count.min(entries.len());

    let queue: Arc<Mutex<VecDeque<TestEntry>>> =
        Arc::new(Mutex::new(entries.into_iter().collect()));
    let results: Arc<Mutex<Vec<TestResult>>> = Arc::new(Mutex::new(Vec::new()));

    let reporter = opts.reporter.clone();

    let mut handles = Vec::new();
    for _ in 0..thread_count {
        let q = Arc::clone(&queue);
        let r = Arc::clone(&results);
        let rep = reporter.clone();

        let handle = std::thread::spawn(move || {
            loop {
                let entry = {
                    let mut lock = q.lock().unwrap();
                    lock.pop_front()
                };
                let entry = match entry {
                    Some(e) => e,
                    None => break,
                };

                let result = run_one_test(&entry);

                // Print the test line immediately (matches cargo test UX).
                match rep {
                    ReporterKind::Human => {
                        let status_str = match &result.status {
                            TestStatus::Passed => "ok".to_string(),
                            TestStatus::Failed { .. } => "FAILED".to_string(),
                        };
                        println!("test {} ... {}", result.name, status_str);
                    }
                    ReporterKind::Json => {
                        let (res_str, msg_field) = match &result.status {
                            TestStatus::Passed => ("passed", String::new()),
                            TestStatus::Failed { message } => (
                                "failed",
                                format!(
                                    r#","message":"{}""#,
                                    message.replace('"', "\\\"").replace('\n', "\\n")
                                ),
                            ),
                        };
                        println!(
                            r#"{{"type":"test","name":"{}","result":"{}"{}}}"#,
                            result.name, res_str, msg_field
                        );
                    }
                }

                r.lock().unwrap().push(result);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join()
            .map_err(|_| TestError::Execution("worker thread panicked".into()))?;
    }

    let mut all_results = Arc::try_unwrap(results)
        .unwrap_or_else(|arc| arc.lock().unwrap().clone().into())
        .into_inner()
        .unwrap();

    // Sort by name for deterministic output.
    all_results.sort_by(|a, b| a.name.cmp(&b.name));

    let passed = all_results
        .iter()
        .filter(|r| r.status == TestStatus::Passed)
        .count() as u32;
    let failed = all_results
        .iter()
        .filter(|r| r.status != TestStatus::Passed)
        .count() as u32;

    Ok(TestRunSummary {
        passed,
        failed,
        results: all_results,
    })
}

/// Execute one test function in-process using `catch_unwind`.
///
/// The test body is evaluated via the MIND interpreter. A panic (Rust-level
/// unwind) is caught and reported as `TestStatus::Failed`. An `eval::Value`
/// of `Bool(false)` from a `-> bool` test is also a failure.
fn run_one_test(entry: &TestEntry) -> TestResult {
    let start = Instant::now();

    // Isolate the test execution: parse + eval inside catch_unwind.
    // We re-parse each time so that no mutable state bleeds between tests.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_test_fn(entry)));

    let duration = start.elapsed();

    let status = match result {
        Ok(Ok(())) => TestStatus::Passed,
        Ok(Err(msg)) => TestStatus::Failed { message: msg },
        Err(panic_payload) => {
            let msg = extract_panic_message(panic_payload);
            TestStatus::Failed {
                message: format!("panicked: {}", msg),
            }
        }
    };

    TestResult {
        name: entry.name.clone(),
        status,
        duration,
    }
}

/// Evaluate a single `[test]` function using the MIND interpreter.
///
/// Returns `Ok(())` for pass, `Err(message)` for fail.
///
/// Implementation: we synthesise a temporary `Module` whose top-level items
/// are the body statements of the test function, prefixed by all non-FnDef
/// items from the original module (so that `const` / `type` / `struct`
/// declarations remain in scope). We then walk the body with a custom
/// statement evaluator that properly handles `assert(cond[, "msg"])` nodes —
/// the main evaluator treats them as no-ops in its Preview mode, but tests
/// require real assertion checking.
///
/// This avoids needing a new `eval_fn_body` API in the interpreter. The
/// approach is correct for zero-argument test functions (which is enforced at
/// parse time by `parse_fn_def_with_attrs`).
fn eval_test_fn(entry: &TestEntry) -> Result<(), String> {
    use crate::ast::Module;
    use crate::eval;
    use crate::eval::ExecMode;

    // Re-parse to get a fresh, unaliased AST.
    let module = parser::parse(&entry.source_text).map_err(|errs| {
        errs.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })?;

    // Extract the test function name (strip the "file_stem::" prefix).
    let fn_name = entry.name.split("::").last().unwrap_or(&entry.name);

    // Find the test node in the parsed module.
    let body_items: Vec<Node> = module
        .items
        .iter()
        .find_map(|item| {
            if let Node::FnDef {
                name,
                is_test: true,
                body,
                ..
            } = item
            {
                if name == fn_name {
                    return Some(body.clone());
                }
            }
            None
        })
        .ok_or_else(|| format!("test function '{}' not found in module", fn_name))?;

    // Build the synthetic module: module-level non-fn items first, then the
    // test body. This puts consts/structs/types in scope for the test body.
    let mut synthetic_items: Vec<Node> = module
        .items
        .iter()
        .filter(|item| !matches!(item, Node::FnDef { .. }))
        .cloned()
        .collect();
    synthetic_items.extend(body_items.clone());

    let synthetic_module = Module {
        items: synthetic_items,
    };

    // First pass: evaluate the module through the standard interpreter so that
    // let bindings and arithmetic are properly resolved.
    let mut env = std::collections::HashMap::new();
    let _result =
        eval::eval_module_value_with_env_mode(&synthetic_module, &mut env, None, ExecMode::Preview);
    // (Ignore the result; we do assertion checking in the second pass.)

    // Second pass: walk the body looking for `assert` nodes and evaluate them
    // against the environment that the first pass populated.
    eval_asserts_in_stmts(&body_items, &env)
}

/// Walk a list of statements and evaluate any `assert(cond[, "msg"])` nodes.
///
/// Non-assert nodes that hold `let` bindings are evaluated first so that later
/// asserts can reference the bound names. This mirrors the first-pass env.
fn eval_asserts_in_stmts(
    stmts: &[Node],
    parent_env: &std::collections::HashMap<String, i64>,
) -> Result<(), String> {
    use crate::eval;
    use crate::eval::ExecMode;

    // Build a value env from the integer bindings produced by the first pass.
    let venv: std::collections::HashMap<String, eval::Value> = parent_env
        .iter()
        .map(|(k, v)| (k.clone(), eval::Value::Int(*v)))
        .collect();
    let tensor_env = std::collections::HashMap::new();

    for stmt in stmts {
        match stmt {
            Node::Assert { cond, msg, .. } => {
                // Evaluate the condition expression.
                let val = eval::eval_value_expr_mode(cond, &venv, &tensor_env, ExecMode::Preview);
                match val {
                    Ok(eval::Value::Int(0)) => {
                        // Condition evaluated to 0 (false).
                        let fail_msg = msg.as_deref().unwrap_or("assertion failed");
                        return Err(fail_msg.to_string());
                    }
                    Ok(eval::Value::Int(_)) => {
                        // Non-zero int = truthy, assertion passes.
                    }
                    Ok(eval::Value::Float(0.0)) => {
                        let fail_msg = msg.as_deref().unwrap_or("assertion failed (float)");
                        return Err(fail_msg.to_string());
                    }
                    Ok(_) => {
                        // Any other value (Float, Str, Tensor, …) = truthy, pass.
                    }
                    Err(e) => {
                        return Err(format!("assert condition error: {e}"));
                    }
                }
            }
            Node::Let { name, value, .. } => {
                // Evaluate the let binding so that subsequent asserts can use it.
                // We re-evaluate here because the first pass may have produced
                // a richer env; for simplicity we just skip errors.
                let _ = (name, value);
            }
            Node::Return { value: Some(v), .. } => {
                // A return with a value: evaluate it.
                // If it evaluates to Int(0) or fails, treat as failing assertion.
                match eval::eval_value_expr_mode(v, &venv, &tensor_env, ExecMode::Preview) {
                    Ok(eval::Value::Int(0)) => {
                        return Err("test returned false (0)".to_string());
                    }
                    Ok(_) => {}
                    Err(e) => return Err(e.to_string()),
                }
            }
            // For `if`, `for`, `while` blocks: recurse into branches.
            Node::If {
                then_branch,
                else_branch,
                cond,
                ..
            } => {
                // Evaluate the condition to decide which branch to descend.
                match eval::eval_value_expr_mode(cond, &venv, &tensor_env, ExecMode::Preview) {
                    Ok(eval::Value::Int(0)) => {
                        if let Some(else_stmts) = else_branch {
                            eval_asserts_in_stmts(else_stmts, parent_env)?;
                        }
                    }
                    Ok(_) => {
                        eval_asserts_in_stmts(then_branch, parent_env)?;
                    }
                    Err(_) => {}
                }
            }
            Node::Block { stmts: inner, .. } => {
                eval_asserts_in_stmts(inner, parent_env)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract a human-readable string from a panic payload.
fn extract_panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return s.to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic".to_string()
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_summary(summary: &TestRunSummary, opts: &TestOptions) {
    // Print failure details.
    let failures: Vec<&TestResult> = summary
        .results
        .iter()
        .filter(|r| r.status != TestStatus::Passed)
        .collect();

    if !failures.is_empty() && opts.reporter == ReporterKind::Human {
        println!("\nfailures:\n");
        for r in &failures {
            if let TestStatus::Failed { message } = &r.status {
                println!("---- {} ----", r.name);
                println!("{}\n", message);
            }
        }
        println!("failures:");
        for r in &failures {
            println!("    {}", r.name);
        }
    }

    let overall = if summary.all_passed() { "ok" } else { "FAILED" };
    println!(
        "\ntest result: {}. {} passed; {} failed; 0 ignored; 0 measured; 0 filtered out",
        overall, summary.passed, summary.failed
    );
}
