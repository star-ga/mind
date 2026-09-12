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

//! `mindc check` must reject what `mindc lower` refuses to emit.
//!
//! THE DIVERGENCE THIS CLOSES. A nine-line file declaring a struct field of an unresolved
//! module-qualified type and calling a method on it passed `mindc check` CLEAN and PANICKED
//! `mindc build`:
//!
//! ```text
//! thread 'main' panicked at src/eval/lower.rs:
//!   method call `<expr>.fetch_add(...)` could not be resolved: the receiver's struct type
//!   is unknown, so it cannot desugar to a `<type>_fetch_add` free function.
//! ```
//!
//! A gate more permissive than the compiler it gates tells the user their source is fine
//! right up until it is not. The panic itself is correct as a last line of defence — emitting
//! a const-0 placeholder there would be a silent miscompile — but it is the wrong MECHANISM
//! for a source error, and it arrives with no `.mind` location.
//!
//! SCOPE, and why it is exactly this narrow: only a method call WITH ARGUMENTS is rejected.
//! A zero-arg unresolved call still takes lowering's historical const-0 placeholder path, so
//! flagging it here would reject code that builds today.
//!
//! MEASURED BLAST RADIUS, because a diagnostic that over-fires is worse than the panic:
//! all 35 previously-check-clean `std/*.mind` files stay clean; the 3 files in 512-mind that
//! this newly reports (`crosschain`, `cross_platform`, `wob_harness`) were each verified to
//! build-panic on this exact condition, so they are true positives, not collateral.

use libmind::parser::parse;
use libmind::type_checker::check_module_types_in_file;

fn errors_for(src: &str) -> Vec<String> {
    let module = parse(src).expect("fixture must parse");
    check_module_types_in_file(&module, src, Some("fixture.mind"), &Default::default())
        .iter()
        .map(|e| format!("{e:?}"))
        .collect()
}

const UNRESOLVED: &str = "unresolved receiver type";

#[test]
fn a_method_call_with_args_on_an_unresolved_receiver_is_rejected() {
    let errs = errors_for(
        "use mind.core.atomic\n\
         \n\
         struct S {\n\
         \x20   c: atomic.U64,\n\
         }\n\
         \n\
         pub fn f(s: &S) -> i64 {\n\
         \x20   s.c.fetch_add(1)\n\
         }\n",
    );
    assert!(
        errs.iter().any(|e| e.contains(UNRESOLVED)),
        "a method call with args on an unresolved receiver must be reported by check, \
         because lowering PANICS on it. Got: {errs:?}"
    );
}

/// POSITIVE CONTROL: a RESOLVED receiver must not be flagged.
///
/// Without this the assertion above would also pass if the check fired on every method call,
/// which would reject most of the stdlib.
#[test]
fn a_resolved_receiver_is_not_flagged() {
    let errs = errors_for(
        "struct Counter {\n\
         \x20   n: i64,\n\
         }\n\
         \n\
         fn counter_bump(c: &Counter, by: i64) -> i64 {\n\
         \x20   c.n + by\n\
         }\n\
         \n\
         pub fn f(c: &Counter) -> i64 {\n\
         \x20   c.bump(1)\n\
         }\n",
    );
    assert!(
        !errs.iter().any(|e| e.contains(UNRESOLVED)),
        "a receiver whose type resolves to a struct with a matching `<type>_<method>` free \
         function must NOT be flagged. Got: {errs:?}"
    );
}

/// A ZERO-ARG unresolved call must still be accepted.
///
/// Lowering keeps a const-0 placeholder for that shape, so rejecting it here would break
/// source that builds today. This pins the narrowness of the rule rather than trusting it.
#[test]
fn a_zero_arg_unresolved_call_is_still_accepted() {
    let errs = errors_for(
        "use mind.core.atomic\n\
         \n\
         struct S {\n\
         \x20   c: atomic.U64,\n\
         }\n\
         \n\
         pub fn f(s: &S) -> i64 {\n\
         \x20   s.c.load()\n\
         }\n",
    );
    assert!(
        !errs.iter().any(|e| e.contains(UNRESOLVED)),
        "a zero-arg unresolved call takes lowering's placeholder path and must not be \
         rejected — doing so would break code that builds. Got: {errs:?}"
    );
}
