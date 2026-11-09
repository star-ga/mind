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

//! Parallel test coordination and stable result reporting.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[path = "worker.rs"]
mod worker;

use super::{
    EvalSupport, IndexedTestEntry, IndexedTestResult, ReporterKind, TestEntry, TestError,
    TestOptions, TestResult, TestRunSummary, TestStatus, run_one_test,
};
#[cfg(test)]
use worker::CancellationState;
use worker::{WorkerPanicGuard, WorkerPool, default_stack_ladder};

/// Execute the test entries in parallel, collecting results.
pub(super) fn execute_tests(
    entries: Vec<TestEntry>,
    eval_support: Arc<BTreeMap<PathBuf, EvalSupport>>,
    opts: &TestOptions,
) -> Result<TestRunSummary, TestError> {
    let runner: Arc<TestRunner> = Arc::new(run_one_test);
    let sink: Arc<ResultSink> = Arc::new(print_result);
    execute_tests_with(
        entries,
        eval_support,
        opts,
        runner,
        sink,
        CompletionControl::default(),
    )
}

type TestRunner = dyn Fn(&TestEntry, Option<&EvalSupport>) -> TestResult + Send + Sync;
type ResultSink = dyn Fn(&TestResult, &ReporterKind) + Send + Sync;

/// Tests may observe a completed result after it reaches the shared queue or
/// inject stack-size ladders and worker lifecycle hooks.
/// The production control is zero-sized and has no callback or branch.
#[derive(Clone, Default)]
struct CompletionControl {
    #[cfg(test)]
    after_store: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    #[cfg(test)]
    worker_stacks: Option<Arc<dyn Fn(usize) -> Vec<usize> + Send + Sync>>,
    #[cfg(test)]
    before_spawn: Option<Arc<dyn Fn(usize, &Arc<CancellationState>) + Send + Sync>>,
    #[cfg(test)]
    on_worker_spawned: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

fn execute_tests_with(
    entries: Vec<TestEntry>,
    eval_support: Arc<BTreeMap<PathBuf, EvalSupport>>,
    opts: &TestOptions,
    runner: Arc<TestRunner>,
    sink: Arc<ResultSink>,
    completion: CompletionControl,
) -> Result<TestRunSummary, TestError> {
    let thread_count = if opts.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        opts.threads
    };
    let thread_count = thread_count.min(entries.len());

    let indexed_entries: Vec<IndexedTestEntry> = entries
        .into_iter()
        .enumerate()
        .map(|(ordinal, entry)| IndexedTestEntry { ordinal, entry })
        .collect();
    let queue: Arc<Mutex<VecDeque<IndexedTestEntry>>> =
        Arc::new(Mutex::new(indexed_entries.iter().cloned().collect()));
    let results: Arc<Mutex<Vec<IndexedTestResult>>> = Arc::new(Mutex::new(Vec::new()));

    let reporter = opts.reporter.clone();
    let base_stacks = default_stack_ladder();
    let mut pool = WorkerPool::new(Arc::clone(&queue));
    let mut spawn_err = None;

    for worker_idx in 0..thread_count {
        #[cfg(test)]
        if let Some(before_spawn) = &completion.before_spawn {
            before_spawn(worker_idx, &pool.cancellation());
        }
        #[cfg(test)]
        let custom_stacks = completion.worker_stacks.as_ref().map(|f| f(worker_idx));
        #[cfg(test)]
        let stacks: &[usize] = custom_stacks.as_deref().unwrap_or(&base_stacks);
        #[cfg(not(test))]
        let stacks: &[usize] = &base_stacks;

        let q = Arc::clone(&queue);
        let r = Arc::clone(&results);
        let support = Arc::clone(&eval_support);
        let run = Arc::clone(&runner);
        let cancelled = pool.cancellation();
        let worker_completion = completion.clone();

        match pool.spawn_with_fallback(worker_idx, stacks, move || {
            let q = Arc::clone(&q);
            let r = Arc::clone(&r);
            let support = Arc::clone(&support);
            let run = Arc::clone(&run);
            let cancelled = Arc::clone(&cancelled);
            let _completion = worker_completion.clone();
            move || {
                let _panic_guard = WorkerPanicGuard {
                    cancellation: &cancelled,
                };
                loop {
                    if cancelled.is_cancelled() {
                        break;
                    }
                    let entry = {
                        let mut lock = q.lock().unwrap_or_else(|e| e.into_inner());
                        if cancelled.is_cancelled() {
                            lock.clear();
                            None
                        } else {
                            lock.pop_front()
                        }
                    };
                    let Some(entry) = entry else {
                        break;
                    };

                    let result = run(&entry.entry, support.get(&entry.entry.source_file));
                    if cancelled.is_cancelled() {
                        break;
                    }

                    r.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(IndexedTestResult {
                            ordinal: entry.ordinal,
                            result,
                        });
                    #[cfg(test)]
                    if let Some(after_store) = &_completion.after_store {
                        after_store(entry.ordinal);
                    }
                }
            }
        }) {
            Ok(_accepted_stack) =>
            {
                #[cfg(test)]
                if let Some(on_spawned) = &completion.on_worker_spawned {
                    on_spawned(worker_idx, _accepted_stack);
                }
            }
            Err(err) => {
                spawn_err = Some(err);
                break;
            }
        }
    }

    if let Some(err) = spawn_err {
        pool.cancel_and_fail_unrun(&indexed_entries, &results, &err);
    } else {
        pool.join_all()?;
    }

    let indexed_results = Arc::try_unwrap(results)
        .unwrap_or_else(|arc| arc.lock().unwrap_or_else(|e| e.into_inner()).clone().into())
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());

    // Workers finish in an arbitrary order. Discovery order is the stable
    // contract: names alone are insufficient when duplicate names occur in
    // different source files or repeated declarations.
    let all_results = order_results(indexed_results);

    let passed = all_results
        .iter()
        .filter(|r| r.status == TestStatus::Passed)
        .count() as u32;
    let failed = all_results
        .iter()
        .filter(|r| r.status != TestStatus::Passed)
        .count() as u32;

    for result in &all_results {
        sink(result, &reporter);
    }

    Ok(TestRunSummary {
        passed,
        failed,
        results: all_results,
        // Set by `run_tests`, which owns the discovery phase.
        unparsed_files: 0,
    })
}

fn print_result(result: &TestResult, reporter: &ReporterKind) {
    match reporter {
        ReporterKind::Human => {
            let status_str = match &result.status {
                TestStatus::Passed => "ok",
                TestStatus::Failed { .. } => "FAILED",
            };
            println!("test {} ... {}", result.name, status_str);
        }
        ReporterKind::Json => println!("{}", json_result_line(result)),
    }
}

fn order_results(mut indexed_results: Vec<IndexedTestResult>) -> Vec<TestResult> {
    indexed_results.sort_unstable_by_key(|result| result.ordinal);
    indexed_results
        .into_iter()
        .map(|result| result.result)
        .collect()
}

fn json_result_line(result: &TestResult) -> String {
    let mut object = serde_json::Map::new();
    object.insert("name".to_string(), serde_json::json!(result.name));
    match &result.status {
        TestStatus::Passed => {
            object.insert("result".to_string(), serde_json::json!("passed"));
        }
        TestStatus::Failed { message } => {
            object.insert("message".to_string(), serde_json::json!(message));
            object.insert("result".to_string(), serde_json::json!("failed"));
        }
    }
    object.insert("type".to_string(), serde_json::json!("test"));
    serde_json::Value::Object(object).to_string()
}

#[cfg(test)]
mod deterministic_report_tests {
    use super::{TestResult, TestStatus, json_result_line};
    use std::sync::{Arc, Barrier, Mutex, mpsc};
    use std::time::Duration;

    #[test]
    fn coordinator_drains_forced_reverse_completion_before_emitting_duplicate_names() {
        use super::{
            CompletionControl, EvalSupport, ReporterKind, TestEntry, TestOptions,
            execute_tests_with,
        };
        use std::collections::BTreeMap;

        let both_started = Arc::new(Barrier::new(2));
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let release_first_rx = Arc::new(Mutex::new(release_first_rx));
        let first_barrier = Arc::clone(&both_started);
        let first_release = Arc::clone(&release_first_rx);
        let runner = Arc::new(move |entry: &TestEntry, _support: Option<&EvalSupport>| {
            first_barrier.wait();
            if entry.source_line == 1 {
                first_release
                    .lock()
                    .expect("release first lock")
                    .recv()
                    .expect("release first result");
                TestResult {
                    name: entry.name.clone(),
                    status: TestStatus::Failed {
                        message: "first\nfailed \"with\" escapes \\ and unicode ☃".to_string(),
                    },
                    duration: Duration::ZERO,
                }
            } else {
                TestResult {
                    name: entry.name.clone(),
                    status: TestStatus::Passed,
                    duration: Duration::ZERO,
                }
            }
        });
        // Releasing the first runner before the second returns is a race:
        // the first could still reach the shared result queue first. Release
        // it only after the real worker has stored ordinal 1.
        let completion = CompletionControl {
            after_store: Some(Arc::new(move |ordinal| {
                if ordinal == 1 {
                    release_first_tx.send(()).expect("release first result");
                }
            })),
            ..CompletionControl::default()
        };

        let emitted = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let emitted_sink = Arc::clone(&emitted);
        let sink = Arc::new(move |result: &TestResult, reporter: &ReporterKind| {
            emitted_sink
                .lock()
                .expect("emitted lock")
                .push(match reporter {
                    ReporterKind::Json => json_result_line(result),
                    ReporterKind::Human => format!("test {}", result.name),
                });
        });
        let entry = |line| TestEntry {
            name: "same_name".to_string(),
            source_file: format!("source{line}.mind").into(),
            source_line: line,
            source_text: String::new(),
        };
        let opts = TestOptions {
            threads: 2,
            reporter: ReporterKind::Json,
            ..TestOptions::default()
        };
        let summary = execute_tests_with(
            vec![entry(1), entry(2)],
            Arc::new(BTreeMap::new()),
            &opts,
            runner,
            sink,
            completion,
        )
        .expect("coordinator execution");

        assert_eq!((summary.passed, summary.failed), (1, 1));
        let lines = emitted.lock().expect("emitted lock").clone();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(&lines[0]).expect("first JSON row");
        let second: serde_json::Value = serde_json::from_str(&lines[1]).expect("second JSON row");
        assert_eq!(first["result"], "failed");
        assert_eq!(second["result"], "passed");
        assert_eq!(first["name"], "same_name");
    }

    #[test]
    fn json_report_uses_strict_json_escaping_for_all_text() {
        let result = TestResult {
            name: "file::quote\"\\\n☃".to_string(),
            status: TestStatus::Failed {
                message: "line\nquote \" slash \\ tab\t".to_string(),
            },
            duration: Duration::ZERO,
        };
        let encoded = json_result_line(&result);
        let parsed: serde_json::Value =
            serde_json::from_str(&encoded).expect("report must be valid JSON");
        assert_eq!(parsed["type"], "test");
        assert_eq!(parsed["result"], "failed");
        assert_eq!(parsed["name"], result.name);
        assert_eq!(parsed["message"], "line\nquote \" slash \\ tab\t");
    }
}
