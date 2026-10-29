# RFC 0011: Async + Structured Concurrency Model

| Field | Value |
|---|---|
| RFC | 0011 |
| Title | Async + structured concurrency model |
| Status | **Phase A shipped** — `std/async.mind` (sync + ReplayScheduler, pure MIND); Phases B–F deferred. Phase D amended 2026-10-28 (proposed): value determinism for data-parallel work (§5.4, §7.1) |
| Authors | STARGA Inc. |
| Created | 2026-05-21 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0008 (mindc build + test), RFC 0010 (memory safety + C ABI), RFC 0009 (federation package layer) |

---

## 1. Motivation

MIND's governance-substrate positioning demands a specific property that most
concurrency models do not supply: **deterministic execution**. An agent
execution trace that cannot be replayed byte-identically is not auditable. A
concurrent program whose output depends on scheduler timing cannot be placed
into an evidence chain.

The async model in this RFC is designed from that constraint outward. The
scheduler is an explicit, first-class value — not an implicit thread-local, not
a language keyword with hidden runtime state. Every point of non-determinism
is confined to the `Scheduler` argument. A program that uses a
`ReplayScheduler` produces byte-identical output given identical inputs across
runs, machines, and operating-system versions. This is the governance-substrate
primitive.

The second constraint is practical: MIND targets both single-core embedded
contexts (the fixed-point Q16.16 workloads) and multi-core server contexts (the
native encoder, the federation package fetch layer). A single async model must
work across both without forcing the fixed-point path to import threading
overhead. The Scheduler injection model satisfies this: a `SyncScheduler`
adds zero overhead for the single-core case; a `WorkStealingScheduler` provides
parallel execution for the multi-core case. The caller picks the scheduler at
the call site, not at compile time.

---

## 2. Non-goals

The following are explicitly outside the scope of RFC 0011.

**No `async fn` keyword infection.** A function that performs asynchronous work
does not need a different declaration syntax. The `Scheduler` argument is
explicit; there is no implicit coloring of functions into sync and async worlds.
A function is either synchronous (no `Scheduler` argument) or asynchronous
(takes a `&Scheduler` argument). Calling a synchronous function from an
asynchronous context, or the reverse, requires no adapter.

**No single mandatory executor.** The design ships four scheduler
implementations (§6), each appropriate for a different deployment context.
MIND does not designate one as "the" runtime. The program picks.

**No garbage-collected futures.** `Future<T>` is a Tier 3 (region-exterior)
heap allocation managed via `GenRef<Future<T>>` (RFC 0010 §3.3). Futures are
freed explicitly; there is no background collector.

**No stackful coroutines or green threads in Phase 1.** M:N scheduling with
per-task stacks is a Phase 2 consideration. Phase 1 tasks are stackless: a
`Future<T>` is a state-machine allocated on the heap, advanced by the scheduler
without a dedicated stack per task.

**No implicit cancellation propagation.** Cancellation in Phase 1 is
cooperative and explicit (§8). A task that does not observe cancellation
continues to run.

---

## 3. The Scheduler injection model

Async operations take an explicit `&Scheduler` argument. There is no implicit
scheduler in thread-local storage and no keyword that inserts one.

```mind
fn fetch(s: &Scheduler, url: String) -> Future<Response> {
    let req = http_request(url);
    s.submit(req)
}

fn main() {
    let s = ReplayScheduler::new();
    let resp = fetch(&s, "https://example.com").await_on(&s);
    print(resp.body);
}
```

The `Scheduler` is the *only* point that introduces non-determinism. Code that
receives a `&Scheduler` argument is transparent about the fact that it performs
asynchronous work. Code that does not receive a `&Scheduler` argument is
provably synchronous and determinism-preserving — the type signature is the
audit trail.

### 3.1 The Scheduler trait

`Scheduler` is a trait in `std.concurrent`. All scheduler implementations
satisfy it.

```mind
trait Scheduler {
    fn submit<T>(self: &Self, work: SenderOf<T>) -> Future<T>;
    fn run<T>(self: &Self, future: Future<T>) -> T;
    fn yield_now(self: &Self);
    fn trace_hash(self: &Self) -> u64;
}
```

- `submit` — enqueues a unit of work and returns a handle to its result.
- `run` — blocks until a future completes and returns its value. The entry
  point for top-level async code.
- `yield_now` — cooperative yield; the task returns control to the scheduler
  and will be rescheduled. The mechanism for cooperative cancellation.
- `trace_hash` — returns a hash over the execution trace accumulated so far
  (RFC 0011 §7). Every scheduler implements this; it is the governance-substrate
  primitive.

### 3.2 Await syntax

`.await_on(&s)` is the method-call form for resolving a future. The scheduler
is always explicit; there is no bare `.await` that resolves against an implicit
runtime.

```mind
let a: Future<i32> = compute_async(&s, 1);
let b: Future<i32> = compute_async(&s, 2);

// Sequential resolution
let x = a.await_on(&s);
let y = b.await_on(&s);

// Or use the algebra (§4) for pipeline composition
let result = compute_async(&s, 1)
    .then(|n| compute_async(&s, n + 1))
    .await_on(&s);
```

---

## 4. Senders + receivers algebra

Asynchronous values compose via a pipeline algebra. A `Sender<T>` produces a
value asynchronously. A `Receiver<T>` consumes it. The two types are the
building blocks of all asynchronous pipelines.

```mind
trait Sender<T> {
    fn then<U>(self, f: fn(T) -> Sender<U>) -> Sender<U>;
    fn map<U>(self, f: fn(T) -> U) -> Sender<U>;
    fn recover(self, f: fn(Error) -> T) -> Sender<T>;
    fn into_future(self, s: &Scheduler) -> Future<T>;
}

trait Receiver<T> {
    fn recv(self: &Self, s: &Scheduler) -> Option<T>;
}
```

Pipelines are values. They compose without allocation until `into_future` (or
`await_on`) is called to actually schedule execution.

```mind
let pipeline = http_request("https://api.example.com/data")
    .then(|resp| parse_json::<ApiResponse>(resp.body))
    .then(|data| store_record(db, data))
    .map(|_| "stored");

let msg = pipeline.into_future(&scheduler).await_on(&scheduler);
print(msg);  // "stored"
```

Error propagation is explicit: `recover` handles errors at the point they are
declared to be handled. Unhandled errors propagate to the `await_on` call site
and are returned to the caller as `Result<T, Error>`.

---

## 5. Structured concurrency

All tasks in MIND are tree-structured. A parent task waits for all of its
children before it completes. There are no detached tasks at the language level.

```mind
fn parallel_fetch(s: &Scheduler, urls: Vec<String>) -> Vec<Response> {
    let group = TaskGroup::new(s);

    for url in urls {
        group.spawn(http_request(url));
    }

    group.join_all()   // waits for every spawned task; returns Vec<Response>
}
```

The `TaskGroup` is the structured concurrency primitive. It is a scope: tasks
spawned into the group live no longer than the group. When `join_all` returns,
all child tasks have completed (or panicked — §5.2).

### 5.1 `select` and `race`

```mind
// select: wait for the first of N futures to complete
let first = select(&s, [fetch_from_a(&s), fetch_from_b(&s)]);

// race: like select, but cancels the losers cooperatively
let winner = race(&s, [slow_path(&s), fast_path(&s)]);
```

`select` returns the value of the first future that resolves. The remaining
futures are abandoned (not cancelled — they are simply not awaited). `race` is
`select` with cooperative cancellation: losing tasks receive a cancellation
signal and are expected to yield on the next `yield_now` call.

### 5.2 Task panics

If a child task panics, the panic propagates to `join_all` at the parent. The
parent task then observes a `TaskPanic` error containing the child's panic
message and task ID. Other child tasks in the same group complete normally before
`join_all` returns; the panic does not propagate laterally to sibling tasks.

This is the structured-concurrency guarantee: a parent that calls `join_all`
either gets all results or gets the full set of panics, never a partial result
with some tasks still running.

### 5.3 Detached tasks (Phase 2)

`spawn_detached` — a task that outlives its parent and joins with no one — is
explicitly deferred to Phase 2 and will be marked as the `unsafe`-equivalent
surface for concurrency: it requires an explicit `# safety:` annotation
explaining why the task lifetime is sound, and it is disabled by default in
governance-substrate contexts.

### 5.4 Data-parallel entry points (Phase D)

```mind
// Illustrative; exact signatures are fixed at implementation.
fn par_for(s: &Scheduler, len: i64, body: Fn(i64, i64, i64) -> ());
fn par_fold<T>(s: &Scheduler, len: i64, init: T,
               leaf: Fn(i64, i64, i64) -> T, combine: Fn(T, T) -> T) -> T;
```

`len` is the element count. Leaf `i` covers elements
`[i * 4096, min(len, (i + 1) * 4096))` (§7.1 rule 1); `body` and `leaf` are
called as `(i, start, end)` for every leaf. `par_fold` returns `init` when
`len == 0` and otherwise never combines `init`. `combine` is applied by the
fixed tree of §7.1 rule 2, whose shape depends only on the leaf count, never on
`T`. Both entry points are available on every scheduler and are
value-deterministic on every scheduler (§7.1).

---

## 6. Phasing

| Phase | Deliverable | Gate |
|---|---|---|
| A | **SHIPPED** `std/async.mind` — `SyncScheduler` + `ReplayScheduler` (heap-record i64 ABI, pure MIND, no threads). `Sender` / `Receiver` pipeline algebra (`submit`, `then`, `run`). `trace_hash` — FNV-1a rolling hash over the event log (governance-substrate determinism primitive, RFC 0011 §7). Thunk model: integer-addend pipeline (Phase A honest: sync = immediate evaluation). | `std_surface_async` test suite: parse + lower, cross-module resolver, IR structural checks. Phase A gates: `cargo test --features "std-surface cross-module-imports" --test std_surface_async` — all pass. `mlir-build` functional tests (determinism contract, native .so) gated for MLIR environments. |
| B | `ReplayScheduler` — records all scheduler events (submit order, yield points, completion order) into an append-only event log. Replay mode reads from the log and re-executes in recorded order. `trace_hash` is the SHA-256 of the event log. | Byte-identical execution traces across two independent runs on the same input; `trace_hash` values match. |
| C | `SingleThreadScheduler` — asynchronous I/O on one thread using the platform event loop abstraction (`mind.io`). No thread pool. Suitable for I/O-bound workloads on a single core. | Integration test: concurrent HTTP fetches complete without blocking; CPU does not spin. |
| D | `WorkStealingScheduler` — parallel execution across N threads using work-stealing queues. Explicit N parameter; defaults to the number of logical processors. Ships the data-parallel entry points `par_for` and `par_fold` (§5.4) under the value-determinism contract (§7.1); depends on the radix-4096 tiled fold (draft RFC `docs/rfcs/DRAFT-deterministic-format-frontend.md` §2e) being shipped, since `par_fold` is defined as that tree. General `submit` keeps the §7 non-replay semantics. | Throughput: N independent CPU-bound tasks complete in about 1/N wall time vs sequential. Value determinism (§7.1): `par_fold` output bytes identical for N ∈ {1, 2, 3, 4, logical processors, > leaf count}, over lengths {0, 1, 4095, 4096, 3·4096 + tail}, identical to `SyncScheduler`, and avx2 == neon for leaves that reduce with the pinned fold; identical across seeded runs with randomized steal order and injected delays (test-only seam, absent from the default build per RFC 0008 §4.7; a short seed set in the keystone, a 1,000-run sweep once per lane); a deliberately completion-ordered fold over an association-sensitive f64 input fails the gate (positive control); a denormal-bearing input is identical for every N; a test-only build with a reduced leaf size and radix (a seam, absent from the default build) exercises trees of three or more levels at small sizes and must equal a reference tree evaluator, so a flat-loop combine fails; the once-per-lane sweep also folds one input above 4096² elements at the production constants. |
| E | Structured concurrency primitives: `TaskGroup` (§5), `select` (§5.1), `race` (§5.1). Task panic propagation (§5.2). | `TaskGroup` join semantics verified: all children complete before join returns; panics propagate correctly. |
| F | Governance-substrate guarantees: `ReplayScheduler`'s trace hash is part of the evidence chain. `mindc test` (RFC 0008) gains a `--scheduler replay` flag that runs all async tests under `ReplayScheduler` and fails if two runs produce different `trace_hash` values. A non-replay scheduler's trace hash (`std.async.trace_hash`) is diagnostic only: it never enters `evidence_chain.trace_hash` (RFC 0016 §3.3) or `agent.replay_trace` (RFC 0019 §4.1), and a step run on such a scheduler stays `agent.determinism = "nondeterministic"` (RFC 0019 §3.3). Value-deterministic work (§7.1) is attested by the hash of its output bytes, which a verifier recomputes under any scheduler and any N; naming a field for that hash is an RFC 0019 amendment. | `mindc test --scheduler replay` passes on the existing test suite; any non-determinism in a test is detected. A step run under `WorkStealingScheduler` leaves `evidence_chain.trace_hash` and `agent.replay_trace` untouched and records `agent.determinism = "nondeterministic"`. |

Phase A is the prerequisite for all subsequent phases. Phase B is the highest
priority after Phase A because `ReplayScheduler` is the governance-substrate
primitive that justifies the model's design. Phases C, D, and E are
independent of each other after Phase B and may be implemented in any order.
Phase F gates on Phase B (for `trace_hash`) and Phase E (for `mindc test`
async test runner).

---

## 7. Determinism contract

A `ReplayScheduler` SHALL produce byte-identical execution traces given
identical inputs across independent runs, across hosts, and across
microarchitectures. The determinism property holds for the full execution trace:
task submission order, yield-point sequence, completion order, and all values
produced by tasks.

The `trace_hash` value — a SHA-256 over the event log — is the machine-verifiable
commitment to this property. It is the governance-substrate artefact: an audit
that records `trace_hash = <hex>` for a given input can be replayed against the
same input on any machine and the hash must match.

**What is not guaranteed:** execution *performance* under `ReplayScheduler` may
differ across machines. The trace is deterministic; the wall-clock time is not.
Performance measurement must use a non-replay scheduler.

**What the non-replay schedulers guarantee:** `SingleThreadScheduler` and
`WorkStealingScheduler` are not deterministic under the replay definition above.
They are correct — tasks always complete, results are always computed, panics
always propagate — but the task interleaving order is not recorded and cannot
be replayed. Code that requires the determinism contract must use
`ReplayScheduler` or a scheduler that wraps it.

### 7.1 Value determinism under parallel schedulers

§7 defines **trace determinism**: the same event log, so the same trace hash.
Only `ReplayScheduler` provides it. A second, weaker property is what most
parallel numeric code needs:

**Value determinism.** Every value a computation produces is bit-identical
whatever the scheduler, the worker count N, the steal order and the timing.
The trace may differ between runs; the values may not.

`par_for` and `par_fold` (§5.4) provide value determinism on every scheduler,
including `WorkStealingScheduler`, by five rules:

1. **Decomposition is a pure function of the problem.** Fold leaves are fixed-size (`PINNED_FOLD_ELEM_CAP`, 4096 elements, the last leaf ragged); leaf boundaries depend only on the input shape, never on N, load or which worker is free. The scheduling grain (how many whole leaves a steal moves) is tuning and may vary; the leaf size is semantics and may not.
2. **Combination order is a pure function of the decomposition.** Each leaf writes its partial to the slot for its leaf index. Partials combine by the fixed radix-4096 left-to-right hierarchical fold specified in draft RFC `docs/rfcs/DRAFT-deterministic-format-frontend.md` §2e: a tree keyed by leaf index, never by completion order. A worker holding several leaves evaluates exactly the subtrees the tree defines over them, not a flat loop. `SyncScheduler` evaluates the identical tree, so the single-thread value is the reference. There is one fold tree, not two.
3. **No shared mutable accumulator.** No atomic add on a float, no lock-protected running sum, no "first to finish writes first". `par_for` writes only to slots derived from the leaf index; overlapping writes are outside the contract.
4. **Work stealing chooses only who and when.** A steal moves a leaf to another worker; it cannot change the leaf's inputs, its boundaries or its place in the tree.
5. **Leaf bodies are pure in everything but the leaf.** A body's inputs are the leaf index and immutable data. Worker, thread and task identity, allocation addresses and generation counters, clocks, scratch contents at leaf entry, and any random state not derived from the leaf index are not inputs. Every worker thread runs with the floating-point control state of the single-thread path; floating-point status flags are not observable.

Consequences:

- `race` and `select` (§5.1) are first-winner and are **not** value-deterministic unless every candidate yields the same value; they may not appear on a value-deterministic result path. `par_for` has no early exit; a search is a `par_fold` whose combine takes the lowest leaf index, and every leaf completes.
- Once Phase E ships panic propagation (§5.2), a panic in a leaf follows it: all leaves complete, and the parent observes the panics ordered by leaf index. Phase E's gate covers this ordering.
- The precedent exists in the multithreaded GEMM canary and the multi-thread == single-thread rule of `docs/rfcs/DRAFT-deterministic-format-frontend.md` (§8, Phase 3); this section makes it the scheduler contract.
- Changing N, the scheduling grain or the steal policy may change wall-clock time only; changing the leaf size or the tree changes values and is a versioned semantic change, not tuning.

**Evidence.** Value-deterministic work is attested by the hash of its output
bytes, never by a non-replay scheduler's trace hash, which varies run to run
(Phase F, §6).

---

## 8. Open questions

### Cancellation: cooperative vs forced

Cooperative cancellation (the Phase 1 model) requires tasks to call
`yield_now` to observe a cancellation signal. A task that does not yield does
not get cancelled.

Forced cancellation — interrupting a task at any yield point regardless of
whether it called `yield_now` — is more powerful but requires every suspension
point in the task to be a potential cancellation point, which imposes overhead
and complicates the correctness model.

**Proposed: cooperative only in Phase 1.** Forced cancellation via an explicit
`task.kill()` method is deferred to Phase 2, marked as unsafe-equivalent, and
requires a `# safety:` justification. The `race` combinator (§5.1) achieves
the practical effect of cancellation without the correctness complexity.

### I/O integration: platform event loop abstraction

Phase C introduces `SingleThreadScheduler` with asynchronous I/O. The I/O
abstraction must work on Linux (epoll), macOS (kqueue), and Windows (IOCP)
without the scheduler implementation containing platform-specific branches.

**Proposed: `mind.io` abstraction module.** The `mind.io` standard module
exposes a platform-agnostic I/O readiness interface. Each scheduler that
performs I/O depends on `mind.io` rather than on platform primitives directly.
The `mind.io` module is the single place where platform-specific code lives.
The design of `mind.io` is deferred to Phase C and will be specified in a
sub-RFC.

### `Future<T>` memory: stack vs heap

A `Future<T>` could be allocated on the stack (zero allocation cost for small
futures) or on the heap as a `GenRef<Future<T>>` (consistent with Tier 3,
uniform across all future sizes).

**Proposed: heap, `GenRef<Future<T>>`** — consistent with RFC 0010 §3.3 and
the structured-concurrency tree. A stack-allocated future cannot be moved into
a `TaskGroup` without copying; heap allocation avoids the copy. The generation
check is the cross-task lifetime guarantee: a parent task that holds a
`GenRef<Future<T>>` to a child will panic cleanly if the child was freed out
of order, rather than observing undefined behaviour. Small-future stack
optimisation is a Phase 2 compiler optimisation, not a Phase 1 language rule.

---

## 9. Relation to other RFCs

**RFC 0010 (memory safety + C ABI)** — `Future<T>` is a Tier 3 region-exterior
allocation (RFC 0010 §3.3), managed via `GenRef<Future<T>>`. The generation
check is the cross-task lifetime mechanism. RFC 0011 Phase A depends on RFC 0010
Phase J being shipped (or at least being in the same release) because `GenRef`
is the heap allocation primitive for futures. Phase A can prototype with a
temporary heap primitive; Phase J closes the loop to the normative model.

**RFC 0008 (mindc build + test)** — `mindc test` (RFC 0008 §5) gains a
`--scheduler replay` flag in RFC 0011 Phase F. This extends the existing test
runner without modifying the test discovery or result-reporting model. The
bench-gate discipline (RFC 0008 §4.7) applies: the `ReplayScheduler` and test
runner overhead must not appear in the default non-test build path.

**RFC 0009 (federation package layer)** — the mirror fallback in `mindc fetch`
(RFC 0009 §5.2) is currently sequential and synchronous. When RFC 0011 Phase D
ships, a future RFC may specify parallel mirror fetches with a first-winner
model using `race` (§5.1). RFC 0011 is not a prerequisite for RFC 0009.

**Deterministic format front-end (draft RFC `docs/rfcs/DRAFT-deterministic-format-frontend.md`) and RFC 0012** — §7.1 is defined in terms of the radix-4096 tiled fold specified in that draft's §2e, itself recursion on the shipped pinned fold (`emit_tensor_reduce_pinned`, the strict lowering behind RFC 0012 §5.1 `#[deterministic]`). The tree fixes the order in which leaf results combine; the order inside a leaf is the leaf body's. When every leaf reduces its elements with the pinned fold, thread-level `par_fold` and the single-core SIMD fold evaluate one tree and cannot disagree; for any other leaf body, §7.1 guarantees identical values across schedulers and N, and cross-substrate identity additionally needs the leaf body itself to be substrate-identical. RFC 0006 dense f32 reductions via `vector.reduction` are explicitly not bit-exact and are not covered by §7.1.

---

## 10. Decision points

This section records the design choices made for this RFC. These decisions are
final for the specified phases; superseding a decision requires a new RFC or an
explicit backward-compatibility annotation.

### Naming: `Sender<T>` vs `Promise<T>` vs `Producer<T>`

- `Promise<T>` — implies a one-shot contract between two parties; the name is
  correct for the semantics but has a strong association with callback-chaining
  style from other ecosystems, which is not MIND's model.
- `Producer<T>` — accurate but asymmetric with the consumer side; the pair
  would be `Producer<T>` / `Consumer<T>`, which is longer and less distinctive
  than `Sender` / `Receiver`.
- `Sender<T>` / `Receiver<T>` — short, symmetric, and carries the correct
  directionality semantics. A `Sender` sends; a `Receiver` receives. The algebra
  methods (`then`, `map`, `recover`) are defined on `Sender<T>` because
  transformations are applied to the producing side before it is submitted to
  a scheduler.

**Decision: `Sender<T>` / `Receiver<T>`.**

### Default scheduler: implicit thread-local vs explicit pass-through

- Implicit thread-local scheduler — convenient; most call sites do not need to
  name the scheduler. The cost: functions that perform async work do not
  advertise that fact in their signature. A function that silently reads the
  thread-local scheduler is indistinguishable, from its type signature alone,
  from a purely synchronous function.
- Explicit pass-through — every function that performs async work declares it
  via the `&Scheduler` argument. Call sites must name the scheduler they want.
  The cost: more verbose. The benefit: the type signature is the audit trail;
  no hidden state, no implicit non-determinism.

**Decision: explicit pass-through. No implicit scheduler state.**

This decision is load-bearing for the governance-substrate use case. An audit
that traces a MIND program's execution must be able to identify every async
call site from the source alone. An implicit scheduler makes this impossible.

### Await syntax: `.await_on(&s)` vs `await(future, scheduler)` vs bare `.await`

- Bare `.await` — maximally ergonomic; requires a language keyword and an
  implicit scheduler. Incompatible with the explicit-pass-through decision above.
- `await(future, scheduler)` — function-call syntax; no new keyword; scheduler
  is explicit. Less idiomatic as MIND's surface syntax is method-call-dominant.
- `.await_on(&s)` — method-call form; no new keyword; scheduler is explicit;
  consistent with the method-call surface used throughout MIND.

**Decision: method form `.await_on(&s)`.**

The method form keeps the scheduler visible at the call site and requires no
new language keyword. The `_on` suffix distinguishes it from any future
bare-keyword form if one is ever introduced, and makes the scheduler argument
legible in pipeline chains:

```mind
a.then(|x| b(x)).await_on(&replay_scheduler)
```

---

## 11. References

- RFC 0010 — memory safety + C ABI (`Future<T>` as `GenRef`-managed Tier 3
  allocation; RFC 0011 Phase A depends on RFC 0010 Phase J).
- RFC 0008 — mindc build + test (`mindc test --scheduler replay` in Phase F).
- RFC 0009 — federation package layer (mirror fallback; future parallel-fetch
  extension via `race`).
- The MIND governance-substrate and evidence chain (the properties that make
  `ReplayScheduler` and `trace_hash` load-bearing, not optional).
