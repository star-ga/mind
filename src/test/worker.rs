// Copyright 2026 STARGA Inc.
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

//! Worker thread spawning with stack-size fallback and RAII join/cancel guard.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use super::super::{IndexedTestEntry, IndexedTestResult, TestError, TestResult, TestStatus};

/// Default stack size for each test worker thread (512 MiB).
///
/// Test bodies run on the tree-walking evaluator, which recurses per nested
/// expression and call; a large std program (the TLS client) overflows the
/// 2 MiB default thread stack in a debug build. Only the address range is
/// reserved; pages are committed as they are touched.
pub(super) const TEST_WORKER_STACK: usize = 512 * 1024 * 1024;

/// Fallback stack sizes tried in order when the primary 512 MiB reservation is
/// refused by the OS (for example under a low `RLIMIT_AS` or strict overcommit).
pub(super) const FALLBACK_WORKER_STACKS: &[usize] = &[
    64 * 1024 * 1024,
    16 * 1024 * 1024,
    8 * 1024 * 1024,
    2 * 1024 * 1024,
];

/// Environment variable overriding the worker stack-size ladder (bytes,
/// comma-separated byte list, or `only:<bytes>` to disable fallback).
pub(super) const ENV_WORKER_STACK: &str = "MINDC_TEST_WORKER_STACK";

/// Resolve the stack-size ladder to try when spawning a test worker thread.
pub(super) fn default_stack_ladder() -> Vec<usize> {
    let env_val = std::env::var(ENV_WORKER_STACK).ok();
    resolve_stack_ladder(env_val.as_deref())
}

/// Parse an optional `MINDC_TEST_WORKER_STACK` specification into the ordered
/// list of stack sizes (in bytes) to try when spawning a worker thread.
///
/// * `None` / empty: `[512 MiB, 64 MiB, 16 MiB, 8 MiB, 2 MiB]`.
/// * `"<bytes>"`: tries `<bytes>` first, then falls back through
///   [`FALLBACK_WORKER_STACKS`].
/// * `"<b1>,<b2>,..."`: tries the listed byte sizes in order.
/// * `"only:<b1>[,<b2>...]"` : tries only the listed byte sizes without
///   appending [`FALLBACK_WORKER_STACKS`].
pub(super) fn resolve_stack_ladder(raw: Option<&str>) -> Vec<usize> {
    let default_ladder = || {
        let mut out = Vec::with_capacity(1 + FALLBACK_WORKER_STACKS.len());
        out.push(TEST_WORKER_STACK);
        out.extend_from_slice(FALLBACK_WORKER_STACKS);
        out
    };
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return default_ladder();
    };
    if let Some(only) = raw.strip_prefix("only:") {
        let parsed = parse_size_list(only);
        return if parsed.is_empty() {
            default_ladder()
        } else {
            parsed
        };
    }
    if raw.contains(',') {
        let parsed = parse_size_list(raw);
        return if parsed.is_empty() {
            default_ladder()
        } else {
            parsed
        };
    }
    if let Ok(primary) = raw.parse::<usize>() {
        if primary > 0 {
            let mut out = Vec::with_capacity(1 + FALLBACK_WORKER_STACKS.len());
            out.push(primary);
            for &fallback in FALLBACK_WORKER_STACKS {
                if fallback != primary {
                    out.push(fallback);
                }
            }
            return out;
        }
    }
    default_ladder()
}

fn parse_size_list(list: &str) -> Vec<usize> {
    list.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|&sz| sz > 0)
        .collect()
}

/// Shared cancellation state between the coordinator and spawned workers.
#[derive(Default)]
pub(super) struct CancellationState {
    cancelled: AtomicBool,
    gate: Mutex<bool>,
    cv: Condvar,
}

impl CancellationState {
    pub(super) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        let mut guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        *guard = true;
        self.cv.notify_all();
    }

    #[cfg(test)]
    pub(super) fn wait_cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let mut guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        while !*guard && !self.is_cancelled() {
            guard = self.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Cancels the shared worker pool immediately if a worker thread unwinds from
/// an unrecovered panic, waking sibling workers regardless of join order.
pub(super) struct WorkerPanicGuard<'a> {
    pub(super) cancellation: &'a CancellationState,
}

impl Drop for WorkerPanicGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.cancellation.cancel();
        }
    }
}

/// Owns the spawned worker threads for a test run and guarantees that every
/// spawned thread is cancelled and joined on normal completion, spawn failure,
/// worker panic, or any early return.
pub(super) struct WorkerPool {
    queue: Arc<Mutex<VecDeque<IndexedTestEntry>>>,
    cancellation: Arc<CancellationState>,
    handles: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    pub(super) fn new(queue: Arc<Mutex<VecDeque<IndexedTestEntry>>>) -> Self {
        Self {
            queue,
            cancellation: Arc::new(CancellationState::default()),
            handles: Vec::new(),
        }
    }

    pub(super) fn cancellation(&self) -> Arc<CancellationState> {
        Arc::clone(&self.cancellation)
    }

    /// Spawn one worker thread using the first stack size in `stacks` that the
    /// OS accepts. Returns the accepted stack size on success, or the last OS
    /// error if every candidate stack size failed.
    pub(super) fn spawn_with_fallback<F, G>(
        &mut self,
        worker_idx: usize,
        stacks: &[usize],
        make_task: F,
    ) -> Result<usize, std::io::Error>
    where
        F: Fn() -> G,
        G: FnOnce() + Send + 'static,
    {
        let mut last_err = None;
        for &stack_size in stacks {
            let task = make_task();
            match std::thread::Builder::new()
                .name(format!("mindc-test-{worker_idx}"))
                .stack_size(stack_size)
                .spawn(task)
            {
                Ok(handle) => {
                    self.handles.push(handle);
                    return Ok(stack_size);
                }
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| std::io::Error::other("no worker stack sizes configured")))
    }

    /// Cancel queued work, join all already-spawned workers so no orphan
    /// remains, and record a per-test failure for every entry that did not
    /// complete.
    pub(super) fn cancel_and_fail_unrun(
        &mut self,
        all_entries: &[IndexedTestEntry],
        results: &Arc<Mutex<Vec<IndexedTestResult>>>,
        err: &std::io::Error,
    ) {
        self.cancel_and_clear_queue();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        let message = format!("could not start a test worker: {err}");
        let mut locked = results.lock().unwrap_or_else(|e| e.into_inner());
        for item in all_entries {
            if !locked.iter().any(|r| r.ordinal == item.ordinal) {
                locked.push(IndexedTestResult {
                    ordinal: item.ordinal,
                    result: TestResult {
                        name: item.entry.name.clone(),
                        status: TestStatus::Failed {
                            message: message.clone(),
                        },
                        duration: Duration::ZERO,
                    },
                });
            }
        }
    }

    /// Join every spawned worker thread. If any worker thread panics, cancel
    /// the remaining queue and still join all other spawned workers before
    /// returning the execution error.
    pub(super) fn join_all(mut self) -> Result<(), TestError> {
        let mut panicked = false;
        let mut handles = std::mem::take(&mut self.handles).into_iter();
        while let Some(handle) = handles.next() {
            if handle.join().is_err() {
                panicked = true;
                self.cancel_and_clear_queue();
                for rest in handles {
                    let _ = rest.join();
                }
                break;
            }
        }
        if panicked {
            Err(TestError::Execution("worker thread panicked".into()))
        } else {
            Ok(())
        }
    }

    fn cancel_and_clear_queue(&self) {
        self.cancellation.cancel();
        let mut lock = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        lock.clear();
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        if !self.handles.is_empty() {
            self.cancel_and_clear_queue();
            for handle in self.handles.drain(..) {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CancellationState, FALLBACK_WORKER_STACKS, TEST_WORKER_STACK, WorkerPool,
        resolve_stack_ladder,
    };
    use crate::test::runner::{CompletionControl, execute_tests_with};
    use crate::test::{
        EvalSupport, IndexedTestEntry, ReporterKind, TestEntry, TestOptions, TestResult, TestStatus,
    };
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Mutex, mpsc};
    use std::time::Duration;

    /// Exceeds the virtual address space on every target architecture, so
    /// `Builder::new().stack_size(UNSPAWNABLE_STACK).spawn(...)` fails
    /// immediately in the OS without allocating physical memory.
    const UNSPAWNABLE_STACK: usize = usize::MAX / 4;
    const SMALL_TEST_STACK: usize = 2 * 1024 * 1024;

    struct ActiveWorkerGuard {
        active: Arc<AtomicUsize>,
    }

    impl ActiveWorkerGuard {
        fn enter(active: &Arc<AtomicUsize>) -> Self {
            active.fetch_add(1, Ordering::SeqCst);
            Self {
                active: Arc::clone(active),
            }
        }
    }

    impl Drop for ActiveWorkerGuard {
        fn drop(&mut self) {
            self.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn dummy_entry(line: u32, name: &str) -> TestEntry {
        TestEntry {
            name: name.to_string(),
            source_file: format!("source{line}.mind").into(),
            source_line: line,
            source_text: String::new(),
        }
    }

    #[test]
    fn fallback_stack_runs_tests_when_primary_stack_cannot_spawn() {
        let spawned_stacks = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
        let recorded = Arc::clone(&spawned_stacks);
        let ladder = resolve_stack_ladder(Some(&UNSPAWNABLE_STACK.to_string()));
        assert_eq!(ladder[0], UNSPAWNABLE_STACK);
        assert_eq!(&ladder[1..], FALLBACK_WORKER_STACKS);

        let completion = CompletionControl {
            worker_stacks: Some(Arc::new(move |_| vec![UNSPAWNABLE_STACK, SMALL_TEST_STACK])),
            on_worker_spawned: Some(Arc::new(move |idx, stack| {
                recorded.lock().expect("spawned lock").push((idx, stack));
            })),
            ..CompletionControl::default()
        };
        let runner = Arc::new(
            |entry: &TestEntry, _support: Option<&EvalSupport>| TestResult {
                name: entry.name.clone(),
                status: TestStatus::Passed,
                duration: Duration::ZERO,
            },
        );
        let sink = Arc::new(|_result: &TestResult, _reporter: &ReporterKind| {});
        let opts = TestOptions {
            threads: 2,
            ..TestOptions::default()
        };
        let summary = execute_tests_with(
            vec![dummy_entry(1, "alpha"), dummy_entry(2, "beta")],
            Arc::new(BTreeMap::new()),
            &opts,
            runner,
            sink,
            completion,
        )
        .expect("fallback stack must succeed");

        assert_eq!((summary.passed, summary.failed), (2, 0));
        let chosen = spawned_stacks.lock().expect("spawned lock").clone();
        assert_eq!(chosen, vec![(0, SMALL_TEST_STACK), (1, SMALL_TEST_STACK)]);
    }

    #[test]
    fn unrecoverable_spawn_failure_reports_per_test_error_without_aborting() {
        let ladder = resolve_stack_ladder(Some(&format!("only:{UNSPAWNABLE_STACK}")));
        assert_eq!(ladder, vec![UNSPAWNABLE_STACK]);
        assert_eq!(resolve_stack_ladder(None)[0], TEST_WORKER_STACK);

        let completion = CompletionControl {
            worker_stacks: Some(Arc::new(move |_| ladder.clone())),
            ..CompletionControl::default()
        };
        let runner = Arc::new(
            |entry: &TestEntry, _support: Option<&EvalSupport>| TestResult {
                name: entry.name.clone(),
                status: TestStatus::Passed,
                duration: Duration::ZERO,
            },
        );
        let emitted = Arc::new(Mutex::new(Vec::<String>::new()));
        let emitted_sink = Arc::clone(&emitted);
        let sink = Arc::new(move |result: &TestResult, _reporter: &ReporterKind| {
            emitted_sink
                .lock()
                .expect("emitted lock")
                .push(result.name.clone());
        });
        let opts = TestOptions {
            threads: 2,
            ..TestOptions::default()
        };
        let summary = execute_tests_with(
            vec![dummy_entry(1, "first_test"), dummy_entry(2, "second_test")],
            Arc::new(BTreeMap::new()),
            &opts,
            runner,
            sink,
            completion,
        )
        .expect("spawn failure must report per-test failures instead of aborting");

        assert_eq!((summary.passed, summary.failed), (0, 2));
        assert_eq!(
            emitted.lock().expect("emitted lock").as_slice(),
            &["first_test", "second_test"]
        );
        for res in &summary.results {
            match &res.status {
                TestStatus::Failed { message } => {
                    assert!(
                        message.contains("could not start a test worker"),
                        "unexpected failure message: {message}"
                    );
                }
                TestStatus::Passed => panic!("unspawnable worker must not report pass"),
            }
        }
    }

    #[test]
    fn no_orphan_workers_remain_when_later_worker_spawn_fails() {
        let active_workers = Arc::new(AtomicUsize::new(0));
        let (worker0_started_tx, worker0_started_rx) = mpsc::channel();
        let worker0_started_rx = Arc::new(Mutex::new(worker0_started_rx));
        let shared_cancel: Arc<Mutex<Option<Arc<CancellationState>>>> = Arc::new(Mutex::new(None));

        let active_for_runner = Arc::clone(&active_workers);
        let cancel_for_runner = Arc::clone(&shared_cancel);
        let runner = Arc::new(move |entry: &TestEntry, _support: Option<&EvalSupport>| {
            let _guard = ActiveWorkerGuard::enter(&active_for_runner);
            let _ = worker0_started_tx.send(());
            if let Some(cancel) = cancel_for_runner.lock().expect("cancel lock").clone() {
                cancel.wait_cancelled();
                // Keep the worker alive after cancellation so only a real join (not
                // luck of scheduling) makes the active count reach zero.
                std::thread::sleep(Duration::from_millis(150));
            }
            TestResult {
                name: entry.name.clone(),
                status: TestStatus::Passed,
                duration: Duration::ZERO,
            }
        });

        let cancel_for_hook = Arc::clone(&shared_cancel);
        let completion = CompletionControl {
            before_spawn: Some(Arc::new(move |worker_idx, cancel| {
                if worker_idx == 0 {
                    *cancel_for_hook.lock().expect("cancel lock") = Some(Arc::clone(cancel));
                } else if worker_idx == 1 {
                    worker0_started_rx
                        .lock()
                        .expect("started rx lock")
                        .recv()
                        .expect("worker 0 must be running before worker 1 spawn fails");
                }
            })),
            worker_stacks: Some(Arc::new(|worker_idx| {
                if worker_idx == 0 {
                    vec![SMALL_TEST_STACK]
                } else {
                    vec![UNSPAWNABLE_STACK]
                }
            })),
            ..CompletionControl::default()
        };
        let sink = Arc::new(|_result: &TestResult, _reporter: &ReporterKind| {});
        let opts = TestOptions {
            threads: 2,
            ..TestOptions::default()
        };

        let summary = execute_tests_with(
            vec![
                dummy_entry(1, "worker0_item"),
                dummy_entry(2, "worker1_item"),
            ],
            Arc::new(BTreeMap::new()),
            &opts,
            runner,
            sink,
            completion,
        )
        .expect("partial spawn failure must join worker 0 and report per-test failure");

        assert_eq!(
            active_workers.load(Ordering::SeqCst),
            0,
            "already-spawned worker 0 must be joined before execute_tests_with returns"
        );
        assert_eq!((summary.passed, summary.failed), (0, 2));
    }

    #[test]
    fn no_orphan_workers_remain_when_worker_thread_panics() {
        let active_workers = Arc::new(AtomicUsize::new(0));
        let both_running = Arc::new(Barrier::new(2));
        let shared_cancel: Arc<Mutex<Option<Arc<CancellationState>>>> = Arc::new(Mutex::new(None));

        let active_for_runner = Arc::clone(&active_workers);
        let barrier_for_runner = Arc::clone(&both_running);
        let cancel_for_runner = Arc::clone(&shared_cancel);
        let runner = Arc::new(move |entry: &TestEntry, _support: Option<&EvalSupport>| {
            let _guard = ActiveWorkerGuard::enter(&active_for_runner);
            barrier_for_runner.wait();
            if entry.source_line == 1 {
                panic!("injected worker panic");
            }
            if let Some(cancel) = cancel_for_runner.lock().expect("cancel lock").clone() {
                cancel.wait_cancelled();
                // Keep the worker alive after cancellation so only a real join (not
                // luck of scheduling) makes the active count reach zero.
                std::thread::sleep(Duration::from_millis(150));
            }
            TestResult {
                name: entry.name.clone(),
                status: TestStatus::Passed,
                duration: Duration::ZERO,
            }
        });

        let cancel_for_hook = Arc::clone(&shared_cancel);
        let completion = CompletionControl {
            before_spawn: Some(Arc::new(move |worker_idx, cancel| {
                if worker_idx == 0 {
                    *cancel_for_hook.lock().expect("cancel lock") = Some(Arc::clone(cancel));
                }
            })),
            worker_stacks: Some(Arc::new(|_| vec![SMALL_TEST_STACK])),
            ..CompletionControl::default()
        };
        let sink = Arc::new(|_result: &TestResult, _reporter: &ReporterKind| {});
        let opts = TestOptions {
            threads: 2,
            ..TestOptions::default()
        };

        let err = execute_tests_with(
            vec![dummy_entry(1, "panicking"), dummy_entry(2, "sibling")],
            Arc::new(BTreeMap::new()),
            &opts,
            runner,
            sink,
            completion,
        )
        .expect_err("worker panic must surface as TestError::Execution");

        assert!(err.to_string().contains("worker thread panicked"));
        assert_eq!(
            active_workers.load(Ordering::SeqCst),
            0,
            "sibling worker thread must be cancelled and joined before panic error returns"
        );
    }

    #[test]
    fn dropping_worker_pool_cancels_queue_and_joins_spawned_workers() {
        let queue: Arc<Mutex<VecDeque<IndexedTestEntry>>> =
            Arc::new(Mutex::new(VecDeque::from([IndexedTestEntry {
                ordinal: 0,
                entry: dummy_entry(1, "queued"),
            }])));
        let active_workers = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let started_rx = Arc::new(Mutex::new(started_rx));

        {
            let mut pool = WorkerPool::new(Arc::clone(&queue));
            let cancel = pool.cancellation();
            let active = Arc::clone(&active_workers);
            pool.spawn_with_fallback(0, &[SMALL_TEST_STACK], move || {
                let cancel = Arc::clone(&cancel);
                let active = Arc::clone(&active);
                let started_tx = started_tx.clone();
                move || {
                    let _guard = ActiveWorkerGuard::enter(&active);
                    let _ = started_tx.send(());
                    cancel.wait_cancelled();
                }
            })
            .expect("spawn worker in pool");

            started_rx
                .lock()
                .expect("started rx")
                .recv()
                .expect("worker started");
            assert_eq!(active_workers.load(Ordering::SeqCst), 1);
            // Dropping `pool` here simulates an early `?` return before `join_all`.
        }

        assert_eq!(
            active_workers.load(Ordering::SeqCst),
            0,
            "Drop on WorkerPool must cancel and join every spawned thread"
        );
        assert!(queue.lock().expect("queue lock").is_empty());
    }
}
