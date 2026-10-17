# Early scoped fixed-call source review

Author: STARGA Inc.
Verdict: PASS_WITH_NOTES for the proposed design; implementation not frozen or
executed. This is not final code approval.

Reviewed contract: `PRODUCTION_FIXED_CALL_INTEGRATION_SLICE.md`, SHA-256
`6ebbdf720e406dcc724c880e70548a8e21103ec256f75905b9d50d7d8c5ff711`.
Reviewed source base: `17ea2dffec714cfa2532250a5ad13f56b5f1f7fa`.
`CAPTURE.json` binds copies of every source cited below. The current author's
work-in-progress main was not substituted for this base during review.

## Findings and required implementation checks

### High: owning the C declaration alone does not validate the local call edge

`src/build/native_scope.rs:24–33` explicitly documents that flat text promotes
private sibling functions and checks visibility before flattening. The new
prefix-owned declaration map must not bypass that boundary. A concrete source
counterexample is user `fn main()->i64 { proc_errno() }`: `std.process` has a
private helper of that name with its own valid `__errno_location` declaration.
An implementation checking only the final foreign call's owner would let an
undeclared user reach it indirectly through the private helper.

This is a source-level counterexample, not a reproduced candidate failure. The
author accepted the requirement and is adding owner-local-first lookup, unique
public lookup from an explicit canonical import, and equality between that
resolved function and the actual flat-retained target AST. R15 freezes the
private-helper refusal. P11 uses the legitimate exported `tcp_close` in
`std/net.mind:211`; it is paired with the no-import refusal R17. Do not treat
this requirement as closed until those real edges are measured.

### High: source metadata cannot authenticate itself

`src/build/native_image.rs:42` adds one newline to each actual std module;
`:64` composes user source bytes; `:77` currently stores only bytes, `user_lo`
and length. `src/build/native_scope.rs:74–98` retains real module ownership and
entry identity before that composition. A region label, supplied digest or
user_lo alone therefore cannot establish original ownership or visibility.

The announced design satisfies this at the interface level: `nsf_compile`
constructs ownership internally after matching the exact build-bound canonical
prefix. Q01–Q05 fix altered comments/joiners/order and moved boundary inputs,
including their true recomputed source hashes. The versioned input must remain
separate from v1. A standalone suffix has the semantics of one translation
unit; bytes cannot prove whether a caller formerly concatenated multiple files.
The production bridge must require `SingleFile` or a legitimately resolved
one-source project, not a caller-supplied count. General project source sets
remain a separate unresolved integration obligation.

### High: use the actual final frame, and discard speculative foreign state

At `main.mind:48539` HOME lowers the body, and only at `:48560` patches its
actual frame from final `next_id`. `native_c_call_emit.mind:23–43` validates
every argument/result displacement against an actual aligned frame. Passing a
large guessed frame to NCE is not equivalent to reserving that runtime frame.
At `main.mind:48982–48990`, failed SSA lowers the same function again through
HOME. Any new foreign records/import uses from the failed attempt must be
rolled back, not retained as stale fixups.

Use deterministic staged regions followed by exact-frame NCE validation, or a
fresh checked two-pass scheme. Equal reserved/replacement length and exact
foreign offset ownership are required. Keep local E8 and foreign FF15 records
separate. P09 exercises repeated calls and later slots; P10 checks nested
argument evaluation once. Neither alone proves a deliberately invalid-frame
seam or rollback, so PLAN.md preserves those explicit mutation obligations.

### High: deduplicate loader identities only after binding each source prototype

`src/project/active_module_table.rs:103–105` and `:232–239` track file-local
extern hiding; `src/project/substrate_link.rs:241–245` prevents user symbols
from interposing during std compilation. `src/ir/native_c_abi.rs:234–235`
rejects duplicate declaration names inside one AST. Therefore the four prefix
close declarations cannot authorize an undeclared user close, nor can a correct
prefix prototype repair a wrong user prototype. Keep each original declaration,
owner, exact pointer type, compatibility action and use after import merging.

R01–R07 cover the distinguishing boundary. P11 uses public std.net close and
the user's own independently declared close; it must yield one loader import.
This is not permission to ignore same-unit duplicate declarations or choose a
first/last source prototype.

### Medium: real source errno and pointer results must close the missing join

The binding table gives `close: i32(i32)` and
`__errno_location: *mut i32()` with borrowed thread-local result role13.
`native_c_call_emit.mind:106` normalizes i32 return bits. The pointer source
type must remain exact until the explicit address conversion; a coarse INT
slot or ABI pointer tag is not a substitute for its original qualifier/depth.
P01/P02 test signed -1 and actual TLS updates. P03–P08 combine the new call
path with existing source-lowered NIO and owned descriptor effects. There is
no extra test-only errno builtin or metadata flag standing in for libc.

### Medium: preserve old deterministic outputs and keep status narrow

`native_source_image.mind:107/130` currently creates startup-only or fixed NIO
import/fixup vectors; `main.mind:52323` owns parts cleanup. Any extension needs
bounded owned storage and updated producer/consumer/cleanup paths, not writes
past the current scratch extent. No-C C01/C02 require exact same-source old/new
images. Both source-built backends and repeat requests must agree exactly.
Successful scoped diagnostic images do not remove the shipping native frontend
extern refusal, qualify a seed or satisfy the frozen consumer ten/35 contract.

## Outcome and evidence limit

The design accounts for the real joins needed to turn source calls into libc
calls. Its strongest early improvement is the explicit source-scope boundary;
the private-helper example shows why it must cover local edges as well as
extern rows. All findings were sent to the author and root before execution.
No compiler, generated program or test ran in this review. Thirty source
fixtures plus five scope perturbations are preregistered separately; later
observations must preserve their original expected outcomes and any failures.
