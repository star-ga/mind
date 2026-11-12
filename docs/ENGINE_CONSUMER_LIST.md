<!--
Copyright 2025 STARGA Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    http://www.apache.org/licenses/LICENSE-2.0
Part of the MIND project (Machine Intelligence Native Design).
-->

# Engine consumer list — the evidence pack for every hand-written semantics engine

**Status: evidence only. This document changes no code and authorises no
deletion.** It exists because the architectural question "how many hand-written
implementations of MIND's semantics does this tree carry, and which of them may
be retired?" has so far been answered with summaries. A summary is not the
artefact: an approver who rules on a list of module names and line counts has
approved the *summary*, not the modules. Everything below is a measured fact
with the command that produced it, so the ruling is made on the thing itself.

Measured at commit `03368cc7`. Every count, line number and program output in
this file was produced by running the quoted command against that tree.

## What this document is for, and the rule it serves

The plan of record allows **two** hand-written implementations of any construct:
the Rust reference (the bootstrap oracle) and the self-host MIND implementation.
No construct gets a third. Of the nine engines below, **five are not surplus** —
the Rust reference emitter, the two self-host backends, the shipped interpreter,
and the differential fuzzer's oracle pair (a permitted testing tool the rule does
not govern). **Four are surplus:** the second Rust MLIR emitter, the self-host
MLIR-text emitter, the IR preview evaluator, and the C shadow of the std
surface.

The disposition vocabulary is the deletion ladder, and the order is absolute:

| Rung | Meaning |
|------|---------|
| **WIRE** | Give it the caller it never had. |
| **SUBSTITUTE** | Something else does the job — migrate the consumers to the survivor, then retire. |
| **RELOCATE** | Keep it, but out of the shipped artefact. |
| **GATE** | Keep and use it, but fence what it is allowed to claim. |
| **DELETE** | Only when the capability is genuinely not wanted. |

Two standing constraints govern every row:

* *"Nothing calls it"* is evidence about **wiring**, never about **worth**. An
  unreachable finished feature is unreachable AND valuable.
* A capability that produces **correct** results today may not be removed to
  make a gate green. Refusals are only for paths that produce a **wrong or
  unattested** result.

## Summary table

| # | Engine | What it is | Live consumers (measured) | Finished / tested | Surplus under the two-engine rule? | Disposition |
|---|--------|-----------|---------------------------|-------------------|------------------------------------|-------------|
| 1 | `src/mlir/lowering.rs::lower_ir_to_mlir` | Canonical Rust IR→MLIR emitter | `src/pipeline.rs` (i.e. `mindc`), 1 bench, 10 integration tests | Yes; 32 of 43 `Instr` variants, guarded arithmetic | **No** — this IS the Rust reference leg | **KEEP.** It is the substitution target for #2. |
| 2 | `src/eval/mlir_export.rs::emit_mlir_with_opts` | Second Rust IR→MLIR emitter, in the `mind` binary | `src/main.rs:434,479`; `src/eval/mod.rs:669,689,735`; 9 test files | Partly; 20 variants, **type-blind**, unguarded, bare `_ => {}` | **Yes** (third emitter of the same construct) | **GATE now, SUBSTITUTE next.** Not delete: it is the sole emitter of 10 tensor/shape ops. |
| 3 | `main.mind` mic@3 emitter (SECTION 4, ~147 `emit_*` fns) | The canonical self-host emitter | keystone, `mic3_flip_smoke`, `mic3_primitives_smoke`, self-host loop | Yes; gate-locked byte-for-byte | **No** — this IS the self-host leg | **KEEP.** |
| 4 | `main.mind` native-ELF emitter (SECTION 4c, 580 `nb_*` fns) | Rust-free x86-64 backend; produces the frozen bootstrap ELF | `self_host_loop_smoke` (non-skippable), ~60 `self_host_native_*` gates | Yes; frozen `stage1.elf` reproduces itself | **No** — this is the Rust-independence deliverable | **KEEP.** |
| 5 | `main.mind` real-MLIR-text emitter (SECTION 4b, 64 fns) | Byte-parity port of #1 into MIND | `selftest_emit_mlir` → `self_host_mlir_smoke` (`ci,keystone`), `mod_operator_smoke` | Yes for its stated scalar subset | **Yes** — a further hand-written meaning | **OPERATOR DECISION (D4).** Gate-owned; retiring it retires a keystone gate. |
| 6 | `src/eval/mod.rs` AST interpreter | The shipped execution engine | `mind eval`, `mind repl`, `mindc test` pass 1, `mindc conformance` value oracle, tensor stdlib | Yes, but **diverges from both code generators on the defined division edges** | **No** — it is the product's interpreter | **WIRE + fix the edge contract.** |
| 7 | `src/eval/ir_interp.rs::eval_ir` | IR "preview" evaluator | `src/main.rs:530`; `tests/ir_lower.rs:23,33` — and **no longer** the conformance oracle | No; 20 of 43 variants modelled, `Index` is identity | **Yes** | **SUBSTITUTE.** The blocker cleared; the capability is already duplicated. |
| 8 | `tests/mindfuzz_cross_substrate.rs` AST oracle + `mod mic3vm` | Two independent differential oracles | The `mindfuzz_cross_substrate` test target itself | Yes | **No** — permitted migration tool, not a compiler engine | **KEEP + de-duplicate one table.** |
| 9 | `runtime-support/mind_intrinsics.c` weak `vec_*`/`map_*`/`string_*` | C fallbacks that shadow the std surface | Every `--emit-shared` link (unconditional) | Yes, and load-bearing — but **no gate asserts it agrees with `std/*.mind`** | **Yes** (a second implementation per symbol) | **GATE.** Name one authority per symbol and prove agreement. |
| 10 | Supplementary: four more scalar-semantics tables | See §10 | mixed | mixed | Yes | **Recorded, not tasked.** |

---

## 1. `src/mlir/lowering.rs::lower_ir_to_mlir`

**Capability.** The canonical IR→MLIR text emitter: the one `mindc` uses, and
the one that spends real code closing cross-substrate divergences.

**Finished and tested.** Yes. It handles 32 of 43 `Instr` variants
(`grep -oE 'Instr::[A-Za-z0-9]+' src/mlir/lowering.rs | sort -u | wc -l` → 32),
including all control flow, calls, params and the vector intrinsics.

Its arithmetic is *guarded*. `mindc --emit-mlir` on `fn d(a: i64, b: i64) -> i64 { return a / b; }`:

```
%dism2 = arith.cmpi "eq", %0, %dmin2 : i64      ; lhs == INT_MIN
%disn2 = arith.cmpi "eq", %1, %dn12 : i64       ; rhs == -1
%dovf2 = arith.andi %dism2, %disn2 : i1
%disz2 = arith.cmpi "eq", %1, %dzc2 : i64       ; rhs == 0
%dsub2 = arith.ori %dovf2, %disz2 : i1
%dsf2  = arith.select %dsub2, %done2, %1 : i64
%ddt2  = arith.divsi %0, %dsf2 : i64
%2     = arith.select %disz2, %ddz2, %ddt2 : i64
```

i.e. MIND's defined edges as they stood then (`x/0 = 0`, `INT_MIN/-1 = INT_MIN`); `mindc build` now traps on both with exit status 79 (docs/errors.md, "Integer division trap").

**Reachability blind spots, all seven checked.**

| Blind spot | Result |
|---|---|
| Facade re-export hiding the real module | `src/mlir/mod.rs:32-33` re-exports it; consumers reach it as `mlir::lower_ir_to_mlir`. Found. |
| Invocation by path from CI or a script | None; it is a library symbol. |
| String dispatch | None. |
| Name-computed / C-ABI lookup | None. |
| Consumer trees outside `src/` | `benches/std_surface.rs:15,67,95`; ten `tests/*.rs` files (`extern_c_phase_{a,b,c}`, `std_surface_{fndef_lowering,bitwise_binops,while_statement,if_statement,bool_return,call_lowering,intrinsics}`). |
| Registries / entry points | Reached through the `mindc` `[[bin]]` target via `src/pipeline.rs:503,510,553`. |
| Feature-gated call sites | `benches/std_surface.rs` is `required-features = ["std-surface","mlir-lowering"]` — a `--no-default-features` reachability scan under-reports it. |

**Wiring cost / what a user loses.** Not applicable: it is wired and it is the
product path.

**Recommendation — KEEP.** It is the Rust reference leg the plan of record
names. It is also the *substitution target* for engine 2, which makes engine 2's
disposition depend on this file gaining ten tensor/shape arms (see §2).

---

## 2. `src/eval/mlir_export.rs::emit_mlir_with_opts`

**Capability.** A complete second IR→MLIR emitter, wired into the `mind` binary:
`mind eval --emit-mlir`, the AOT emit path, and the three `ExecMode::Mlir*`
execution modes. It is also the **only** emitter in the tree for ten
tensor/shape instructions.

**Finished and tested — partly, and wrong where it overlaps engine 1.**

Measured divergences, reproduced at this commit:

```
$ mind eval $'let a = 10\nlet b = 3\nlet c = a / b\nc' --emit-mlir
    %2 = arith.divsi %0, %1 : i64        <- bare; engine 1 emits the guarded
                                            sequence quoted in §1

$ mind eval $'let a = 1.5\nlet b = 2.25\nlet c = a + b\nc' --emit-mlir
    // result: %0                        <- ConstF64 hits the `_ => {}` arm,
    // result: %1                           so %0 and %1 are NEVER defined
    %2 = arith.addi %0, %1 : i64         <- FLOAT add typed i64
```

The cause is structural, not a missed case:

* `src/eval/mlir_export.rs:260` routes `Instr::BinOp` to `emit_int_binop`
  **unconditionally, with no type dispatch**;
* `emit_int_binop` (`:379-405`) formats every arm `... : i64` with no shift-count
  mask and no div/rem guard;
* `src/eval/mlir_export.rs:340` is a bare `_ => {}` over `Instr` — every variant
  outside its 20-arm list is **silently dropped** from the emitted module, which
  is why `ConstF64` produced no definition for `%0`/`%1` above.

**Reachability blind spots, all seven checked.**

| Blind spot | Result |
|---|---|
| Facade re-export | `src/eval/mod.rs:59` (`pub mod mlir_export`) plus five `pub use` lines at `:239-243`, and the wrapper `emit_mlir_string` at `:252`. Consumers call the wrapper, not the module — a grep for the module name alone under-reports them. |
| Invocation by path | None. |
| String dispatch | `apply_lowering(mlir, preset: &str)` at `:197` is a **string-keyed** preset dispatcher, reached from `src/eval/mlir_build.rs:146`. Invisible to a symbol-level scan. |
| Name-computed / C-ABI | None. |
| Consumers outside `src/` | Nine test files call `to_mlir*`/`emit_mlir_string`: `mlir_export{,_indexing,_linalg,_reductions,_shape}.rs`, `mlir_broadcast.rs`, `mlir_build.rs`, `mlir_lowering.rs`, `mlir_opt.rs`. |
| Registries / entry points | Reached from the `mind` `[[bin]]` target (`src/main.rs:434,479`) — a different binary from `mindc`. Pinned in `tests/module_size_ratchet.rs:60` at 1447 lines. |
| Feature-gated call sites | `src/main.rs:434` is inside `#[cfg(feature = "mlir-build")]`; `src/eval/mod.rs:669,689,735` sit behind `mlir-exec`, `mlir-jit` and `mlir-gpu`. A default-feature reachability scan reports most of this file dead. |

**What wiring/substitution would cost — the decisive measurement.** Neither
emitter is a superset of the other:

* Handled only by **engine 2** (10): `Dot`, `Transpose`, `Reshape`, `Squeeze`,
  `ExpandDims`, `Index`, `Slice`, `Gather`, `Conv2dGradInput`, `Conv2dGradFilter`.
* Handled only by **engine 1** (21): `FnDef`, `Call`, `Return`, `Param`, `If`,
  `While`, `Break`, `Continue`, `Region`, `ConstF64`, `ConstDenseTensor`,
  `ConstArray`, `ArrayLoad`, `ArrayStore`, `ExternFnDecl`, and the six `Vec*`
  intrinsics.

So "delete engine 2, delegate to engine 1" **loses the tensor/shape and
convolution-gradient lowering outright**, plus the nine test files that pin it.
The honest cost is: port ten `Instr` arms into `src/mlir/lowering.rs`, move nine
test files onto it, then retire this one.

**What a user loses without it.** Today: `mind eval --emit-mlir`,
`mind eval --emit-obj`, `--build`, and every `ExecMode::Mlir*` path, plus all
tensor/shape MLIR emission.

**Recommendation — GATE now, SUBSTITUTE next; do not delete.**

1. *Gate (small, and it is the fail-closed half):* replace `:340`'s `_ => {}`
   with an explicit exhaustive list plus a refusal for the variants it cannot
   emit, and make `Instr::BinOp` refuse on a float or narrow operand rather than
   mistype it `i64`. This removes the wrong-bytes path without touching the ten
   correct tensor arms — no capability loss.
2. *Substitute:* port the ten tensor/shape arms into engine 1, migrate the nine
   test files, then retire this file. That is the move that actually reduces the
   engine count.

---

## 3. `examples/mindc_mind/main.mind` — mic@3 emitter (SECTION 4)

**Capability.** The canonical self-host emitter: MIND source → mic@1 IR text and
the mic@3 binary artefact that anchors `trace_hash`.

**Census correction.** The 190 `emit_*` functions are not all this engine. Of
them, **43 live inside SECTION 4b** (the MLIR-text emitter, §5), so SECTION 4's
share is ~147:

```
$ sed -n '23422,25368p' examples/mindc_mind/main.mind \
    | grep -oE '^(pub )?fn [a-z0-9]+_' | sed 's/^pub //;s/^fn //' | sort | uniq -c
     43 emit_
     21 mlir_
```

**Finished and tested.** Yes, and locked byte-for-byte: the keystone
(`tests/phase_g_keystone_bootstrap.rs`, 7 cases), `mic3_flip_smoke` and
`mic3_primitives_smoke` (all three wired `ci,keystone,preflight` in
`examples/mindc_mind/SMOKE_WIRING.tsv`), plus the 23-arm per-arm oracle-parity
linter.

**Reachability blind spots.** All consumers are of the *C-ABI export* kind that
symbol scans miss: the harnesses `dlopen` the built `libmindc_mind.so` and bind
symbols by string (`lib.<name>` in `examples/mindc_mind/*.py`). There is a
declared registry — `SMOKE_WIRING.tsv`, itself re-derived and enforced by
`smoke_wiring_lint.py` (`ci,keystone,preflight`), so a gate cannot silently
un-wire.

**Recommendation — KEEP.** This is the self-host leg the two-engine rule names.
It is not surplus.

---

## 4. `examples/mindc_mind/main.mind` — native-ELF emitter (SECTION 4c)

**Capability.** A Rust-free, LLVM-free x86-64 ELF backend written in MIND. It is
what produces `examples/mindc_mind/testdata/selfhost_loop/stage1.elf`.

**Census.** SECTION 4c spans `28234`–EOF (24,086 lines) and holds 797 functions,
580 of them `nb_*`:

```
$ sed -n '28234,52319p' examples/mindc_mind/main.mind | grep -cE '^(pub )?fn '
797
```

**Finished and tested.** Yes, and it is the strongest-gated engine in the tree.
`testdata/selfhost_loop/MANIFEST.txt` records the frozen stage0 compiler
(`stage1.elf`, 2,219,218 bytes, sha256 `390fa112…`) and says plainly:
*"the checked-in pure-MIND stage0 compiler that reproduces itself
byte-identically (stage1==stage2==stage3) … NOT skippable in CI."* Roughly sixty
`self_host_native_*` smokes cover its per-construct behaviour.

**Reachability blind spots.** Same C-ABI/`dlopen` shape as §3, plus one that no
code scan can see at all: the **artefact itself** is a consumer — the frozen ELF
is executed by `self_host_loop_smoke.py`, which runs `stage1 → stage2 → stage3`
with "zero rustc, zero LLVM, zero clang, zero .so" in the chain.

**What a user loses without it.** Reproduction independence — the entire
Rust-independence claim.

**Recommendation — KEEP.** Not surplus. It is the product this programme is for.

---

## 5. `examples/mindc_mind/main.mind` — real-MLIR-text emitter (SECTION 4b) — **operator decision D4**

**Capability.** A byte-for-byte MIND port of engine 1's output for the scalar
`i64`/`f64` subset. Its own header (`main.mind:23422-23457`) states the scope:
`func.func` wrapper, `ConstI64`, `FloatLit`, `BinOp add/sub/mul/div`, ident,
let-binding, `Return`, synthetic `@main`; explicitly out of scope are tensors,
narrow ints, f64 control flow, calls and comparisons.

**Census.** 1,947 lines, 72 functions (43 `emit_*` + 21 `mlir_*` + 8 other).

**Correction to the record.** The figure "49 `arith.`/`func.func`/`scf.` sites"
counts **comments only**:

```
$ grep -n 'arith\.\|func\.func\|scf\.' examples/mindc_mind/main.mind \
    | grep -vE ':[0-9]+:\s*//'
(no output)
```

The real emitter is `emit_mlir_arith_op` (`:23510`), which pushes the mnemonic
byte by byte (`es_push(s, 97) /* 'a' */ …`). That function is also the
Article-X three-backend parity site — a new operator must be added there, not
only to the mic@3 byte table.

**Reachability — and the part that makes this a decision, not a cleanup.**

* In-file: `emit_mlir_module` (`:25159`) has exactly **one** call site,
  `:25191`, inside `selftest_emit_mlir` (`:25188`).
* Outside the file, it is **gate-owned**: `SMOKE_WIRING.tsv` classifies
  `self_host_mlir_smoke` as `ci,keystone` / `gate` with no deferred marker;
  `.github/workflows/ci.yml:754-762` runs it as *"Self-host MLIR-text f64+i64
  emit gate"*; `mod_operator_smoke.py` also drives `selftest_emit_mlir` by
  C-ABI name (`:139-144`).

**Method caveat that this row proves out.** A symbol-index caller query
**over-reports** for this file. Asking for the callers of `emit_mlir_module`
returns six, of which exactly one is real; the other five cite function
*definition* lines, not call sites:

```
reported callers of emit_mlir_module   |  what is actually at that line
  parse_fn_def          :5435          |  pub fn parse_fn_def(...) -> ParseResult {
  parse_if              :3988          |  pub fn parse_if(...) -> ParseResult {
  parse_primary_ns      :2065          |  pub fn parse_primary_ns(...) -> ParseResult {
  selftest_emit_mlir    :25188         |  the ONE real call site (body at :25191)
  slit_desugar_body     :17516         |  pub fn slit_desugar_body(...) -> i64 {
  slit_find_ctor        :9150          |  pub fn slit_find_ctor(...) -> i64 {

$ grep -n 'emit_mlir_module(' examples/mindc_mind/main.mind
25159:pub fn emit_mlir_module(ast_addr: i64, buf: i64) -> EmitState {
25191:    emit_mlir_module(ast_root, src_addr)          <- the only call
```

The identical spurious set comes back for `selftest_emit_mlir`, whose *real*
consumers (two harnesses and a CI step, all binding it by C-ABI string) the
index does not list at all. **Consequence for this pack:** an index caller list
is neither an upper nor a lower bound — it under-reports C-ABI consumers and
over-reports within-file ones. Every `.mind` reachability claim in this document
was cross-checked with `grep` and against `SMOKE_WIRING.tsv`; none rests on an
index query alone.

**Wiring cost / what is lost.** Nothing is unwired, so there is nothing to wire.
Retiring it means retiring `self_host_mlir_smoke` — a keystone-class gate — and
losing the only differential proof that a MIND program can reproduce
`mindc --emit-mlir` byte-for-byte.

**Recommendation — OPERATOR DECISION (D4), with the evidence stated.**
It is genuinely a further hand-written meaning of `+`, and the MLIR backend has
been demoted from the self-host path — so the rule says it should go. But it is
not unowned code: deleting it deletes a green keystone gate, which is a
plan-of-record call. The two coherent answers are:

* **GATE (recommended if the MLIR backend keeps any product role):** record in
  the section header *why* a self-host MLIR emitter is still wanted and which
  gate consumes it, so it stops being an unattributed fifth implementation.
* **RETIRE (only as an explicit plan-of-record act):** retire SECTION 4b and
  `self_host_mlir_smoke` together, in one change, with the commit body naming
  the gate being removed.

What must **not** happen is a silent drift into "nobody knows why this is here".

---

## 6. `src/eval/mod.rs` — the AST interpreter

**Capability.** The shipped execution engine: `mind eval` (`src/main.rs:397`),
`mind repl` (`:722`), `mindc test` pass 1 (`src/test/mod.rs:544`), the tensor
stdlib, and — **new at this commit** — the conformance suite's runtime-value
oracle (`src/conformance.rs:384`).

**Correction to the record.** It does **not** serve comptime folding. Compile-time
constant evaluation is a separate engine, `src/opt/comptime.rs`, consumed by
`src/opt/scev.rs:518` and `src/opt/collapse.rs:55` (see §10).

**Finished and tested — but it disagrees with both code generators on MIND's
defined arithmetic edges.** When this audit was written both generators
defined `x/0 = 0` and `INT_MIN/-1 = INT_MIN` (engine 1's guard is quoted in §1).
That contract is superseded: `mindc build` (MLIR and native) now traps on `x/0`,
`x%0` and i64 `INT_MIN/-1` with exit status 79 (docs/errors.md, "Integer
division trap"). The interpreter does neither:

```
$ mind eval '7 / 0'
Evaluation error: division by zero                 <- generators trap (exit 79)

$ mind eval '(0 - 9223372036854775807 - 1) / (0 - 1)'
thread 'main' panicked at src/eval/mod.rs:2712:13:
attempt to divide with overflow                    <- RELEASE build, exit 101
```

`apply_int_op` (`:2699`) is deliberately `wrapping_*` for `+ - *`, with the
reason written down — *"so debug and release mindc agree with the shipped .so
and never panic on overflow"* — and then uses a bare `left / right` for `Div`
(`:2712`) and `left % right` for `Mod` (`:2718`), which is exactly the panic the
comment exists to prevent.

**Second defect, structural — CLOSED.** `Node::Assert { .. } => Ok(Value::Int(0))`
made asserts a no-op, which is precisely why a *separate* statement evaluator
had to exist in `src/test/mod.rs` (`eval_asserts_in_stmts`, with its own env and
its own `Let/Assign/If/While` walk). Recommendation 1 below has landed: the
`Assert` arm evaluates its condition under a private thread-local guard that only
`mindc test` holds, the runner calls the test fn as ONE ordered interpreter pass,
and the surplus walker is deleted
(issues #240 / #241 / #243).

**Reachability blind spots.**

| Blind spot | Result |
|---|---|
| Facade re-export | `eval_module_value_with_env` / `_mode` are the public entries; both are re-exported and both are called. |
| Invocation by path | Reached indirectly by every CI step that runs `mindc test`. |
| String dispatch | None. |
| Name-computed / C-ABI | None. |
| Consumers outside `src/` | `tests/transpose_preview.rs:24`, `tests/index_slice_grad.rs:24`, plus ~20 in-module tests. |
| Registries / entry points | Both `[[bin]]` targets. |
| **Consumers outside this repository** | **Yes — `mindc test` is a published CLI surface invoked by downstream STARGA projects' CI.** Those consumers are recorded in the internal audit register rather than here. The operative fact for this pack: **deletion of this execution path is not available without substitution.** |

**Recommendation — WIRE, and fix the edge contract at the layer it belongs to.**

1. **Done.** `Node::Assert` has a real arm under a test mode and
   `src/test/mod.rs::eval_asserts_in_stmts` is retired — a whole surplus
   evaluator removed by *wiring*, the top rung of the ladder.
2. Route every interpreter `Div`/`Mod` through one shared helper implementing
   the division trap of docs/errors.md (`x/0`, `x%0`, i64 `INT_MIN/-1` stop the
   program with exit status 79), matching both generators, and add a
   conformance cell per edge. This is a capability *gain*: a program on a
   defined edge stops deterministically instead of crashing the process.

---

## 7. `src/eval/ir_interp.rs::eval_ir`

**Capability.** A constant-fill "preview" evaluator over lowered IR. It prints
the `--- Result ---` line under `mind eval`.

**Two facts that changed since this engine was last written up.**

* **Its conformance consumer is gone.** `src/conformance.rs` no longer calls it;
  `:328` records the substitution in prose: *"It replaces the IR preview
  evaluator (`eval::ir_interp::eval_ir`), which models 20 of the 35 `Instr`
  variants and silently ignores the rest."* The remaining consumers are
  `src/main.rs:530` and `tests/ir_lower.rs:23,33`. `grep -rn 'ir_interp' src tests --include=*.rs`
  now returns only that doc comment plus `src/eval/mod.rs:47,225`.
* **The bare `_ => {}` is gone.** The match is exhaustive; the 23 unmodelled
  variants are listed explicitly so a new `Instr` fails to *compile* here rather
  than being ignored at runtime (`src/eval/ir_interp.rs:226-256`). The
  *modelling* gap remains — 20 of 43 variants are modelled, `Index`/`Slice` are
  the identity (`:206`), `Sum`/`Mean` return the input's `fill` (`:47,:56`).

**The live defect that remains.** Its integer arithmetic is bare Rust and
therefore build-profile-dependent, while its sibling engine documents the
opposite policy:

```
$ /…/release/mind eval '9223372036854775807 + 1'
-9223372036854775808          <- AST interpreter (wraps)
--- Result ---
-9223372036854775808          <- ir_interp (wraps in release)

$ /…/debug/mind eval '9223372036854775807 + 1'
-9223372036854775808
thread 'main' panicked at src/eval/ir_interp.rs:268:27:
attempt to add with overflow                        <- exit 101
```

`:268-271` is `BinOp::Add => a + b … BinOp::Div => a / b`, and `:295` is
`BinOp::Shr => a >> b` while `:293` is `BinOp::Shl => a.wrapping_shl(b as u32)` —
asymmetric inside one table.

**Reachability blind spots.**

| Blind spot | Result |
|---|---|
| Facade re-export | `src/eval/mod.rs:47` (`pub mod`) + `:225` (`pub use ir_interp::eval_ir`). Consumers say `eval::eval_ir`. |
| Invocation by path | None. |
| String dispatch | None. |
| Name-computed / C-ABI | None. |
| Consumers outside `src/` | `tests/ir_lower.rs:23,33` only. |
| Registries / entry points | The `mind` `[[bin]]` target. |
| Feature-gated call sites | The 16 std-surface variants are `#[cfg(feature = "std-surface")]`; the call sites are not gated. |

**What substitution costs — measured, not estimated.** The capability is
*already duplicated on the same output*. `mind eval` prints the value twice, once
from each engine, and they agree:

```
$ mind eval $'let x: Tensor[f32,(2,3)] = 0\nx + 1'
Tensor[F32,(2,3)] fill=1        <- AST interpreter (src/main.rs:397)
--- Lowered IR --- …
--- Result ---
Tensor[F32,(2,3)] fill=1        <- ir_interp (src/main.rs:530)
```

Both `tests/ir_lower.rs` assertions — `rendered == "7"` and
`rendered.contains("Tensor[")` — are satisfied by the first line, which the AST
interpreter produced. So substitution is: print the already-computed AST value
under the `--- Result ---` header, and re-point the two tests. **Zero capability
loss, and the debug-build crash goes with it.**

**What a user loses without it.** Nothing observable: the same two values are
already printed by the surviving engine.

**Recommendation — SUBSTITUTE.** This is the clearest row in the pack: the
blocker (a live conformance consumer) has cleared, the capability is duplicated,
the replacement is the engine the plan of record keeps, and the engine count
goes down by one. It should still not be *deleted on a grep* — the change is
"re-point two consumers at the survivor, then remove", and the commit should
carry the mutation proof that the debug-build panic no longer reproduces.

---

## 8. `tests/mindfuzz_cross_substrate.rs` — the AST oracle and `mod mic3vm`

**Capability.** Three mutually independent oracles for the differential
determinism fuzzer: a native-ELF run, an in-test AST interpreter
(`apply_op:595`, `eval_expr:621`, `eval_program:654`), and a mic@3 bytecode VM
(`mod mic3vm:718`, `eval_fn:742`, `exec_seq:803`, `apply:974`) that decodes the
emitted artefact through the compiler's own `parse_mic3`.

**This is a permitted migration tool, not a rule violation.** The claim that
these are surplus compiler engines was **refuted**: independence is the entire
mechanism of differential testing, and the canonical self-host emitter *is*
inside a differential loop — a different one, `examples/mindc_mind/mindfuzz_self_host.py`,
whose two oracles are the pure-MIND front-end and the Rust reference derivation,
compared by mic@3 byte-identity. The scope split is documented in both files.
Recorded here as a permitted tool so the census does not double-count it.

**The residual, which is real.** The two tables in this one file are a single
definition transcribed twice:

```
:595  Op::Div => lv.wrapping_div(rv),  Op::Mod => lv.wrapping_rem(rv),
      Op::Shl => lv.wrapping_shl((rv as u64 & 63) as u32),
:974  BinOp::Div => l.wrapping_div(r), BinOp::Mod => l.wrapping_rem(r),
      BinOp::Shl => l.wrapping_shl((r as u64 & 63) as u32),
```

Note also that this convention is a **third** answer on the defined edges:
`wrapping_div` gives `INT_MIN/-1 = INT_MIN` (the generators' answer before the
division trap; they now exit 79) and *panics* on `x/0` (agreeing with neither). The fuzzer never sees it because the
generator forces divisors non-zero with `x | 1` — a scope choice, not agreement.

**Reachability blind spots.**

| Blind spot | Result |
|---|---|
| Facade re-export | N/a (test-local). |
| Invocation by path | `.github/workflows/ci.yml:412` runs `--test mindfuzz_cross_substrate`, and `:457` reads the digest file it writes; the `mindfuzz_cross_runner_identity` job asserts avx2 == neon. |
| String dispatch | The compiled program is `dlopen`ed and the generated `f` bound by name. |
| Name-computed / C-ABI | As above. |
| Consumers outside the file | The CI digest artefact — a consumer no code scan sees. |
| Registries / entry points | `Cargo.toml` declares `[[test]] name = "mindfuzz_cross_substrate", required-features = ["mlir-build"]`, so a default-feature scan reports the whole file dead. |

**Recommendation — KEEP both oracles; de-duplicate the one table.** Factor the
scalar op table into a single shared `const fn` used by both. The oracles stay
independent where independence matters — one walks an AST, the other decodes
mic@3 bytes, the third is a native ELF — while the *convention* is stated once.
This reduces the transcription count rather than adding to it.

**Not actionable as stated:** "cross-check that table against the conformance
grid". The grid is golden IR/MLIR text plus a pinned value per case; it contains
no operator-semantics table to check against. Building one is a prerequisite,
not part of this change.

---

## 9. `runtime-support/mind_intrinsics.c` — the weak `vec_*` / `map_*` / `string_*` fallbacks

**Capability.** C implementations of 40 std-surface functions across
`vec_*` (9), `map_*` (14) and `string_*` (17), each declared
`MIND_EXPORT_WEAK` (`__attribute__((weak))` on ELF/Mach-O). The
`--emit-shared` link path links this object into **every** cdylib
unconditionally, so a consumer `.so` is self-contained.

**Why the weakness is load-bearing — and why it is not the whole answer.** The
file's own header (`:71-100`) explains the design honestly: a strong MIND
definition overrides the weak C fallback (so `std/vec.mind` can be compiled to a
`.so` without a `multiple definition` link error), while a consumer that does not
define the symbol still gets the fallback. `tests/std_surface_self_emit_shared.rs`
gates exactly that, for `string`/`map`/`vec`, with `json` as the
non-overlapping control.

The residual is the one that makes the source of truth ambiguous: **which
definition executes depends on the link**, and *no gate asserts that the two
implementations agree*. Reproduced at this commit — a `std` fork whose
`vec_cap` body returns `777`, consumed by an ordinary program:

```
$ MIND_STDLIB_PATH=<fork> mindc build m.mind --no-cache   # m.mind: vec_cap(vec_new())
   Finished cpu [binary] …/m      Artifact: 37376 bytes
$ ./m
0                                    <- NOT 777: the MIND body never ran

$ nm ./m | grep ' vec_cap'
0000000000002a40 W vec_cap           <- weak binding, from the C object

$ objdump -d ./m | sed -n '/<vec_cap>:/,/ret/p'
0000000000002a40 <vec_cap>:
    2a40: 48 8b 47 10   mov 0x10(%rdi),%rax
    2a44: c3            ret          <- byte-for-byte runtime-support/mind_intrinsics.c:452,
                                        `return __mind_load_i64(v + 16);`
```

So for an ordinary consumer the `.mind` text is inert: editing it changes no
emitted byte, and nothing goes red.

**The registry that exists — and the one that does not.**
`src/intrinsics.rs` carries a real structural gate,
`classification_covers_every_runtime_support_intrinsic`, which scans this C file
and fails until every exported intrinsic is classified — with an anti-vacuity
floor (`exported.len() >= 25`). But it keys on `line.find("__mind_")`, so it
covers only the `__mind_*` intrinsics. **The 40 `vec_*`/`map_*`/`string_*`
surface symbols are outside it.**

**Reachability blind spots.**

| Blind spot | Result |
|---|---|
| Facade re-export | N/a (C). |
| Invocation by path | The file is `include_str!`d into the binary (`src/eval/mlir_build.rs:40`) and written out under a fixed basename (`:361`) with `-ffile-prefix-map`, so build paths never leak into `.symtab`. |
| String dispatch | Symbols are resolved **by name at link time** — invisible to every Rust reachability tool. `grep -rn vec_cap --include=*.rs src/` finds nothing, yet `vec_cap` ships. |
| Name-computed / C-ABI | The entire surface is this case. |
| Consumers outside `src/` | `src/project/mod.rs:2949` (the project build path) and every emitted `.so`. |
| Registries | `src/intrinsics.rs:602` for `__mind_*` only, as above. |

**What a user loses without it.** Self-contained cdylibs. Removing the C
definitions breaks every consumer `.so` that does not itself compile the std
surface; removing the *weakness* re-breaks the self-compiling std path that
`tests/std_surface_self_emit_shared.rs` was written for. Neither is available.

**Recommendation — GATE.** Name one authority per symbol and prove agreement:

1. Extend the existing registry gate in `src/intrinsics.rs` to cover
   `MIND_EXPORT_WEAK` surface symbols, so a new C fallback must be declared
   alongside its `std/*.mind` counterpart instead of appearing silently.
2. Add a differential gate that runs each surface function's MIND body against
   the C fallback over a fixed corpus. Until then the ambiguity is documented
   rather than resolved — which is the point of writing it down here.

Deletion is not on the ladder for this row.

---

## 10. Supplementary — four further scalar-semantics tables found while tracing the nine

Recorded so the unification scope is not undercounted. **Not tasked here.**

| Site | What it is | Wiring status | Note |
|---|---|---|---|
| `src/opt/comptime.rs:52,160` | Two compile-time const evaluators | Wired: `src/opt/scev.rs:518`, `src/opt/collapse.rs:55` | The only tables that get the edges *right*: `wrapping_div`/`wrapping_rem` with an explicit `b == 0` refusal, and a comment saying why. |
| `src/opt/ir_canonical.rs:253` | IR-level const folder | Wired (backend prep) | Correctly refuses to fold both `r == 0` and `l == i64::MIN && r == -1`. |
| `src/opt/fold.rs:109` | AST const folder | **Not pipeline-wired.** Sole consumer `tests/const_folding.rs:23`; `src/opt/comptime.rs:19` says so in-source: *"`opt::fold` is leaf-only and not pipeline-wired"* | Refuses `b == 0` but uses a bare `a / b`, so it would panic at `INT_MIN/-1`. An unwired engine is a **question** — why was it built and never connected? — not a deletion candidate. |
| `src/test/mod.rs` (was `eval_asserts_in_stmts`) | a second statement evaluator, now RETIRED | — | Existed solely because `Node::Assert` was a no-op in engine 6. Retired by *wiring* engine 6 (§6): `mindc test` now runs the body as one interpreter call under its scoped assertion-checking mode. |

## The cross-cutting fact: `x/0` and `INT_MIN/-1` have eight answers

Every row above touches the same divergence. Collected once, because it is the
strongest single argument for unification:

| Implementation | `x / 0` | `INT_MIN / -1` |
|---|---|---|
| `src/mlir/lowering.rs` (engine 1) | trap, exit 79 | trap, exit 79 |
| `main.mind` native emitter (engine 4) | trap, exit 79 | trap, exit 79 |
| `src/eval/mlir_export.rs` (engine 2) | bare `arith.divsi` — **target-defined** | bare `arith.divsi` — **target-defined** |
| `src/eval/mod.rs` (engine 6) | `Err(DivZero)` | **process panic** (release) |
| `src/eval/ir_interp.rs` (engine 7) | **process panic** | **process panic** |
| `tests/mindfuzz_*` (engine 8) | **panic** (`wrapping_div`) | `INT_MIN` |
| `src/opt/comptime.rs` | refuse (`None`/`Err`) | `INT_MIN` |
| `src/opt/ir_canonical.rs` | refuse to fold | refuse to fold |
| `src/opt/fold.rs` (unwired) | refuse to fold | **would panic** |

Nine implementations, **eight distinct answer-pairs**. Exactly one pair —
trap with exit status 79 for both (docs/errors.md, "Integer division trap";
it superseded the earlier `(0, INT_MIN)` contract) — is the sanctioned
contract, and only the two code generators hold it. One is target-defined, which is a determinism leak by itself. Two of
them (engines 6 and 7) crash the **shipped compiler process** on an operation
MIND defines as total; a third crash sits in the unwired folder and a fourth in
the fuzzer's oracle, hidden only because its generator forces divisors non-zero.

That spread is not a set of bugs to be fixed one at a time — it is the predicted
cost of nine hand-written tables, and it is what the two-engine rule exists to
prevent. It is also the cheapest available proof that the rule is worth
enforcing: no single one of these sites looks wrong on its own.

## How to re-derive everything in this file

```
grep -rn 'ir_interp' src tests --include=*.rs
grep -rn 'emit_mlir_with_opts\|emit_mlir_string' src --include=*.rs
grep -oE 'Instr::[A-Za-z0-9]+' src/mlir/lowering.rs   | sort -u | wc -l   # 32
grep -oE 'Instr::[A-Za-z0-9]+' src/eval/mlir_export.rs | sort -u | wc -l  # 20
grep -oE 'Instr::[A-Za-z0-9]+' src/eval/ir_interp.rs   | sort -u | wc -l  # 43 named, 20 modelled
awk '/^pub enum Instr/,/^\}/' src/ir/mod.rs | grep -cE '^    [A-Z]'       # 43
sed -n '23422,25368p' examples/mindc_mind/main.mind | grep -cE '^(pub )?fn '   # 72 (SECTION 4b)
sed -n '28234,52319p' examples/mindc_mind/main.mind | grep -cE '^(pub )?fn '   # 797 (SECTION 4c)
grep -rnE 'BinOp::(Div|Mod) =>' src/ --include=*.rs
mind eval '7 / 0'
mind eval '(0 - 9223372036854775807 - 1) / (0 - 1)'
mindc --emit-mlir F   # engine 1, guarded;  F = fn d(a: i64, b: i64) -> i64 { return a / b; }
mind  eval $'let a = 10\nlet b = 3\nlet c = a / b\nc' --emit-mlir   # engine 2, unguarded
```

A symbol-index caller list for `main.mind` both over-reports and under-reports
(see §5): cross-check it with `grep` and with
`examples/mindc_mind/SMOKE_WIRING.tsv` before relying on it.
