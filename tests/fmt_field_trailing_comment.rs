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

//! A comment inside a struct body must stay inside that body.
//!
//! KNOWN DEFECT, recorded as an ignored test rather than as prose. `mindc fmt` moves EVERY
//! comment that appears inside a struct's braces out past the closing brace:
//!
//! ```text
//! struct S {                        struct S {
//!     // lead-a               ->        a: i64,
//!     a: i64, // trail-a            }
//! }                                 // lead-a
//!                                   // trail-a
//! ```
//!
//! Both positions are affected — leading and trailing — and the comments keep their relative
//! order, so nothing is deleted. What is lost is the ATTACHMENT: neither line documents `a`
//! any more, and a reader of the formatted file cannot recover which field either belonged to.
//! On a wire-format record — `request_id: bytes, // 8 bytes from request_ns` in MindLLM's
//! `governance_bridge.mind` — that comment is the only statement of the field's encoding.
//!
//! MEASURED SCOPE (2026-09-12), because the first version of this file got it wrong. It
//! described only the TRAILING position and asserted, as a positive control, that a LEADING
//! field comment was handled correctly. That control FAILED, which is how the real scope was
//! found: a leading comment is displaced identically. The genuine control is a comment ABOVE
//! the struct, which `fmt` does preserve in place — so the printer's comment machinery works,
//! and this is specifically about the region between the braces.
//!
//! WHY THE EXISTING GATES CANNOT SEE IT. `fmt` stays IDEMPOTENT across this transform (a
//! second pass moves nothing further) and the output still parses, so both halves of the
//! reparse gate pass. That gate was built to catch a lossy re-emit and is the right instrument
//! for one; a comment emitted in the wrong PLACE needs an assertion about position instead of
//! about survival. Same blind spot that let the bool-literal desugar ship: `0` formats to `0`
//! forever.
//!
//! THE CAUSE is in `src/fmt/printer.rs`. `Trivia::comments_at` is indexed by original source
//! LINE and consumed only by `emit_leading` before a TOP-LEVEL item. Struct fields are not
//! top-level items and never consume their own lines, so every comment between the braces sits
//! in the map until the next top-level item flushes it — which places it after the struct.
//!
//! THE FIX is additive and needs no new AST variant: `ast::Field` already carries a `span`, so
//! `emit_struct_def` can consume the comments at each field's line and emit them in place.
//! Un-ignore the first test as the acceptance criterion. Enum variants (`EnumVariant`) and
//! match arms have the same shape and the same bug.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default())
        .unwrap_or_else(|e| panic!("format_source failed on:\n{src}\nerror: {e:?}"))
}

/// `src` with a comment in each in-body position.
const SRC: &str = "struct S {\n    // lead-a\n    a: i64, // trail-a\n}\n\npub fn f(s: &S) -> i64 {\n    s.a\n}\n";

/// Everything between the braces must stay between the braces.
///
/// Remove `#[ignore]` when `emit_struct_def` consumes its fields' comment lines.
#[test]
#[ignore = "known defect: fmt moves every in-body struct comment out past the closing brace"]
fn comments_inside_a_struct_body_stay_inside_it() {
    let out = fmt(SRC);
    let close = out.find("\n}").expect("struct must close");

    for c in ["lead-a", "trail-a"] {
        assert!(
            !out[close..].contains(c),
            "`// {c}` was emitted after the struct's closing brace.\n--- out ---\n{out}"
        );
    }
    // The trailing one must land on its field's line, not merely somewhere inside.
    assert!(
        out.lines().any(|l| l.contains("a: i64") && l.contains("trail-a")),
        "`// trail-a` must stay on the `a: i64` line.\n--- out ---\n{out}"
    );
}

/// The defect is REAL, and this pins its exact current shape.
///
/// Runs unignored. If it ever fails, the defect has been fixed or has changed scope — un-ignore
/// the test above rather than trusting this one.
#[test]
fn the_defect_is_currently_present_and_this_is_its_exact_shape() {
    let out = fmt(SRC);
    let close = out.find("\n}").expect("struct must close");
    let tail = &out[close..];

    assert!(
        tail.contains("lead-a") && tail.contains("trail-a"),
        "the recorded defect did not reproduce — fmt no longer displaces in-body struct \
         comments. If it was FIXED, remove the #[ignore] above and delete this control.\n\
         --- out ---\n{out}"
    );

    // Nothing is deleted and relative order holds: this is misplacement, not loss.
    let il = out.find("lead-a").expect("lead-a must survive");
    let it = out.find("trail-a").expect("trail-a must survive");
    assert!(il < it, "comment order must be preserved.\n{out}");

    // POSITIVE CONTROL, and the corrected one. A comment ABOVE the struct is preserved in
    // place, so the printer's comment machinery works and the defect is scoped to the region
    // between the braces. The first version of this file used a LEADING FIELD comment here and
    // it failed — which is what revealed the defect covers both in-body positions.
    let outside = fmt("// about S\nstruct S {\n    a: i64,\n}\n\npub fn f(s: &S) -> i64 {\n    s.a\n}\n");
    assert!(
        outside.starts_with("// about S"),
        "a comment ABOVE a struct must stay above it — if this fails, the displacement is \
         broader than the struct body and this file understates it.\n--- out ---\n{outside}"
    );
}
