# RFC 0011: Async + Structured Concurrency Model

| Field | Value |
|---|---|
| RFC | 0011 |
| Title | Async + structured concurrency model |
| Status | **Phase A shipped** — `std/async.mind` (sync + ReplayScheduler, pure MIND); Phases B–F deferred |
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

---

## 6. Phasing

| Phase | Deliverable | Gate |
|---|---|---|
| A | **SHIPPED** `std/async.mind` — `SyncScheduler` + `ReplayScheduler` (heap-record i64 ABI, pure MIND, no threads). `Sender` / `Receiver` pipeline algebra (`submit`, `then`, `run`). `trace_hash` — FNV-1a rolling hash over the event log (governance-substrate determinism primitive, RFC 0011 §7). Thunk model: integer-addend pipeline (Phase A honest: sync = immediate evaluation). | `std_surface_async` test suite: parse + lower, cross-module resolver, IR structural checks. Phase A gates: `cargo test --features "std-surface cross-module-imports" --test std_surface_async` — all pass. `mlir-build` functional tests (determinism contract, native .so) gated for MLIR environments. |
| B | `ReplayScheduler` — records all scheduler events (submit order, yield points, completion order) into an append-only event log. Replay mode reads from the log and re-executes in recorded order. `trace_hash` is the SHA-256 of the event log. | Byte-identical execution traces across two independent runs on the same input; `trace_hash` values match. |
| C | `SingleThreadScheduler` — asynchronous I/O on one thread using the platform event loop abstraction (`mind.io`). No thread pool. Suitable for I/O-bound workloads on a single core. | Integration test: concurrent HTTP fetches complete without blocking; CPU does not spin. |
| D | `WorkStealingScheduler` — parallel execution across N threads using work-stealing queues. Explicit N parameter; defaults to the number of logical processors. | Throughput test: N independent CPU-bound tasks complete in approximately 1/N wall time vs sequential. |
| E | Structured concurrency primitives: `TaskGroup` (§5), `select` (§5.1), `race` (§5.1). Task panic propagation (§5.2). | `TaskGroup` join semantics verified: all children complete before join returns; panics propagate correctly. |
| F | Governance-substrate guarantees: every scheduler's `trace_hash` is part of the evidence chain. `mindc test` (RFC 0008) gains a `--scheduler replay` flag that runs all async tests under `ReplayScheduler` and fails if two runs produce different `trace_hash` values. | `mindc test --scheduler replay` passes on the existing test suite; any non-determinism in a test is detected. |

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
