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

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::ast::Node;
use crate::parser;

mod imports;
mod runner;
mod top_level;
use imports::{EvalSupport, prepare_eval_support};
use runner::execute_tests;

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

/// A discovered entry paired with its stable source-order ordinal.
///
/// The ordinal is deliberately kept out of [`TestEntry`]'s public shape. It
/// is an execution/reporting detail, while callers that construct a
/// `TestEntry` should not have to invent one.
#[derive(Debug, Clone)]
struct IndexedTestEntry {
    ordinal: usize,
    entry: TestEntry,
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

#[derive(Debug, Clone)]
struct IndexedTestResult {
    ordinal: usize,
    result: TestResult,
}

/// Aggregate summary returned by [`run_tests`].
#[derive(Debug, Clone, Default)]
pub struct TestRunSummary {
    pub passed: u32,
    pub failed: u32,
    pub results: Vec<TestResult>,
    /// Test files that could not be PARSED, so contributed no tests.
    ///
    /// Counted, not merely warned about. A file in a test suite that does not parse
    /// is a broken test, never a skip -- and the warning below already said so
    /// ("skipping it SILENTLY turns a real breakage into a misleading green") while
    /// the run still exited 0 whenever any OTHER file yielded a passing test. The
    /// diagnosis was written and the enforcement was not.
    pub unparsed_files: u32,
}

impl TestRunSummary {
    /// `true` when all discovered tests passed AND every test file parsed.
    ///
    /// An unparsed file is a failure of the suite even though it produced no test
    /// result to fail: the tests inside it did not run, and "did not run" must never
    /// read as "passed".
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.unparsed_files == 0
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
    let mut eval_support = BTreeMap::new();
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
            Ok(discovered) => {
                if !discovered.is_empty() {
                    eval_support.insert(path.clone(), prepare_eval_support(path, &text));
                }
                entries.extend(discovered);
            }
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
    let unparsed_files = parse_failures.len() as u32;

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
    let mut summary = execute_tests(entries, Arc::new(eval_support), opts)?;
    // Carried from THIS function: `execute_tests` never sees the discovery phase, so
    // the count of files that failed to parse is attached here.
    summary.unparsed_files = unparsed_files;

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

    let items = top_level::refs(&module.items);
    let mut function_counts = BTreeMap::<&str, (usize, bool)>::new();
    for item in &items {
        if let Node::FnDef(function, _) = item {
            let count = function_counts.entry(&function.name).or_insert((0, false));
            count.0 += 1;
            count.1 |= function.is_test;
        }
    }
    if let Some((name, (count, _))) = function_counts
        .iter()
        .find(|(_, (count, includes_test))| *includes_test && *count > 1)
    {
        return Err(format!(
            "ambiguous test function `{name}` has {count} module-level definitions"
        ));
    }

    let mut entries = Vec::new();
    for item in items {
        if let Node::FnDef(fd, span) = item {
            if fd.is_test {
                let name = &fd.name;
                let source_line = line_number_at(source, span.start());
                entries.push(TestEntry {
                    name: format!("{}::{}", file_stem, name),
                    source_file: path.to_path_buf(),
                    source_line,
                    source_text: source.to_string(),
                });
            }
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

/// Execute one test function in-process using `catch_unwind`.
///
/// The test body is evaluated via the MIND interpreter. A panic (Rust-level
/// unwind) is caught and reported as `TestStatus::Failed`. An `eval::Value`
/// of `Bool(false)` from a `-> bool` test is also a failure.
fn run_one_test(entry: &TestEntry, support: Option<&EvalSupport>) -> TestResult {
    let start = Instant::now();

    // Isolate the test execution: parse + eval inside catch_unwind.
    // We re-parse each time so that no mutable state bleeds between tests.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        eval_test_fn(entry, support)
    }));

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
/// ONE ordered interpreter pass. The synthetic module is the file's own items
/// (plus the bundled-std fns its imports name) followed by a single
/// `Node::Call` of the test function, so the body executes through exactly the
/// fn-body executor every helper it calls already runs on — `let`/`assign`
/// threading, `if`/`for`/`while` scoping, memory intrinsics, early `return` —
/// with the interpreter's assertion-checking guard held. Consequences, each of
/// which was a reported false verdict of the previous run-then-rewalk design:
///
/// * an `assert` is evaluated AT ITS POINT, after every statement before it
///   and before every statement after it, memory effects included (#241);
/// * a `let` of a call result, a struct, or anything else is simply a binding
///   in the executing env — nothing is re-evaluated or re-allocated (#240);
/// * ANY evaluation error in the body — an out-of-bounds load, an unresolved
///   name inside a callee, a non-boolean assert condition — fails the test
///   with the interpreter's own message, regardless of how a later assertion
///   would have read (#243, #242);
/// * asserts inside loop bodies, match arms, `region` blocks and CALLED helper
///   fns are executed, not refused.
///
/// The verdict from the returned value: a declared `-> bool` test returning
/// `false` fails ("test returned false (0)"), a returned `Result::Err(..)`
/// fails with its payload, anything else passes. Zero-arity is enforced at
/// parse time by `parse_fn_def_with_attrs`, so the call binds no params.
fn eval_test_fn(entry: &TestEntry, support: Option<&EvalSupport>) -> Result<(), String> {
    use crate::ast::Module;
    use crate::eval;
    use crate::eval::ExecMode;

    // Fresh, deterministic linear memory for THIS test: every test sees an
    // identical zeroed arena regardless of worker-thread assignment or run
    // order (the arena is thread-local and worker threads are reused).
    eval::interp_mem::reset();

    if let Some(error) = support.and_then(|support| support.setup_error.as_ref()) {
        return Err(error.clone());
    }

    // Re-parse to get a fresh, unaliased AST.
    let mut module = parser::parse(&entry.source_text).map_err(|errs| {
        errs.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })?;

    // Extract the test function name (strip the "file_stem::" prefix).
    let fn_name = entry.name.split("::").last().unwrap_or(&entry.name);

    // Find the test fn in the parsed module: its declared return type decides
    // how the call's value is graded, and its span labels the synthetic call.
    let matches = top_level::refs(&module.items)
        .into_iter()
        .filter_map(|item| {
            if let Node::FnDef(fd, span) = item {
                if fd.is_test && fd.name == fn_name {
                    return Some((fd.ret_type.clone(), *span));
                }
            }
            None
        })
        .collect::<Vec<_>>();
    let [(ret_type, span)] = matches.as_slice() else {
        return Err(format!(
            "test function `{fn_name}` resolved to {} module-level definitions",
            matches.len()
        ));
    };
    let (ret_type, span) = (ret_type.clone(), *span);

    let eval_fn_name = if let Some(owner) = support.and_then(|value| value.entry_owner.as_deref()) {
        imports::qualify_module_declarations(&mut module, owner, false);
        eval::eval_owned_symbol(owner, fn_name)
    } else {
        fn_name.to_string()
    };

    // Build the synthetic module: imported-std fn defs first (so file-local
    // fns shadow them in the fn table — later install wins), then ALL
    // module-level items, then ONE call of the test fn. `FnDef` items are
    // inert at module eval level (`Ok(Int(0))`) but are registered by
    // `fn_table_install`, which is what lets the call — and the helper fns
    // its body calls, and `pub fn`s of bundled std modules (e.g. the
    // std.sha256 KAT) — dispatch through the interpreter, with no native
    // runtime. The call MUST be an item of the same module eval: the fn
    // table is scoped to that eval and is torn down when it returns.
    let mut synthetic_items: Vec<Node> = Vec::new();
    #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
    synthetic_items.extend(resolve_std_import_fns(&module)?);
    if let Some(support) = support {
        synthetic_items.extend(support.imported_items.iter().cloned());
    }
    synthetic_items.extend(top_level::cloned(&module.items));
    synthetic_items.push(Node::Call {
        callee: eval_fn_name,
        args: Vec::new(),
        span,
    });
    let synthetic_module = Module {
        items: synthetic_items,
        spelling: Default::default(),
    };

    // Assertion checking is ON for exactly this evaluation. The guard drops on
    // every exit from this function — `?`, `return`, and the unwinding of a
    // panic that `run_one_test`'s `catch_unwind` will catch — so a reused
    // worker thread never carries the mode into the next test or, worse, into
    // a non-test interpreter consumer.
    let _checking = eval::assert_check_guard();
    let _bindings = support.map(|support| {
        eval::EvalBindingsGuard::install(support.bindings.clone(), support.entry_owner.clone())
    });
    let mut env = std::collections::HashMap::new();
    let outcome =
        eval::eval_module_value_with_env_mode(&synthetic_module, &mut env, None, ExecMode::Preview);

    match (outcome, eval::take_assert_failure()) {
        (Err(_), Some(message)) => Err(message),
        (Ok(value), None) => grade_returned_value(ret_type.as_ref(), value),
        // Any other error is the ROOT CAUSE that stopped the body: the
        // interpreter names it (`unknown variable: X`, `memory access out of
        // bounds: [a, a+n) outside requested allocation extent [..]`, ...).
        // Nothing after that point ran, so nothing after it can vouch for a
        // pass.
        (Err(root), None) => Err(root.to_string()),
        (Ok(_), Some(message)) => Err(format!(
            "internal: assertion failed without stopping evaluation: {message}"
        )),
    }
}

/// Grade the value a test fn call returned.
///
/// * `Result::Err(payload)` → failure carrying the payload (RFC 0008 §5.1);
/// * a declared `-> bool` test that returned `0` → "test returned false (0)";
/// * anything else — a unit body's last statement value, `Result::Ok`, a
///   `-> bool` returning `1` — passes.
fn grade_returned_value(
    ret_type: Option<&crate::ast::TypeAnn>,
    value: crate::eval::Value,
) -> Result<(), String> {
    use crate::ast::TypeAnn;
    use crate::eval::Value;

    match value {
        Value::Enum { variant, payload } if variant == "Err" || variant.ends_with("::Err") => {
            let rendered = match payload.as_slice() {
                [Value::Str(s)] => s.clone(),
                [Value::Int(n)] => n.to_string(),
                other => format!("{other:?}"),
            };
            Err(format!("test returned Err({rendered})"))
        }
        Value::Int(0) if matches!(ret_type, Some(TypeAnn::ScalarBool)) => {
            Err("test returned false (0)".to_string())
        }
        _ => Ok(()),
    }
}

/// Resolve `import std.X` / `use std::X` declarations of a test module (and,
/// transitively, of the modules they import) against the stdlib sources baked
/// into the binary (RFC 0005 Phase C, `project::stdlib`), returning the
/// imported modules' `FnDef` items so the interpreter's fn table can dispatch
/// cross-module calls — e.g. a `[test]` running the std.sha256 KAT — plus
/// their module-level `Const` items, so a fn body that reads a module const
/// (std.json's `MAX_DEPTH`) resolves it through the interpreter's const
/// table.
///
/// Deterministic: imports are processed in source-encounter order (FIFO
/// worklist, `BTreeSet` visited guard against cycles). An import that is not
/// a bundled `std.*` module contributes nothing (existing behavior for it is
/// unchanged); a bundled module that fails to parse is a fail-loud error.
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
fn resolve_std_import_fns(module: &crate::ast::Module) -> Result<Vec<Node>, String> {
    use crate::project::stdlib::STDLIB_MIND_SOURCES;
    use std::collections::VecDeque;

    fn import_key(item: &Node) -> Option<String> {
        if let Node::Import { path, .. } = item {
            if path.len() == 2 && path[0] == "std" {
                return Some(path.join("."));
            }
        }
        None
    }

    let mut out: Vec<Node> = Vec::new();
    let mut visited: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut queue: VecDeque<String> = top_level::refs(&module.items)
        .into_iter()
        .filter_map(import_key)
        .collect();
    while let Some(key) = queue.pop_front() {
        if !visited.insert(key.clone()) {
            continue;
        }
        let Some((_, src)) = STDLIB_MIND_SOURCES.iter().find(|(k, _)| *k == key) else {
            continue;
        };
        let imported = parser::parse(src).map_err(|errs| {
            format!(
                "failed to parse bundled std module '{key}': {}",
                errs.iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        })?;
        for item in top_level::refs(&imported.items) {
            match item {
                Node::FnDef(..) | Node::Const { .. } => out.push(item.clone()),
                Node::Import { .. } => {
                    if let Some(k) = import_key(item) {
                        queue.push_back(k);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(out)
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
