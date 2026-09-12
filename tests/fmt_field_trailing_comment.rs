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

//! A trailing comment on a struct field must stay on that field.
//!
//! KNOWN DEFECT, recorded as an ignored test rather than as prose. `mindc fmt` strips a
//! comment that sits on the same line as a struct field and re-emits it AFTER the closing
//! brace, so:
//!
//! ```text
//! struct S {                          struct S {
//!     a: i64,  // describes a    ->       a: i64,
//!     b: i64,  // describes b             b: i64,
//! }                                   }
//!                                     // describes a
//!                                     // describes b
//! ```
//!
//! Both comments survive as text, and they keep their relative order, so nothing is deleted.
//! What is lost is the ATTACHMENT: "describes a" no longer documents `a`, and a reader of the
//! formatted file cannot recover which field either line belonged to. On a wire-format record
//! — `request_id: bytes, // 8 bytes from request_ns` in MindLLM's `governance_bridge.mind` —
//! that comment is the only statement of the field's encoding.
//!
//! WHY THE EXISTING GATES CANNOT SEE IT, which is the part worth keeping. `fmt` stays
//! IDEMPOTENT across this transform (the second pass moves nothing further) and the output
//! still parses, so both halves of the reparse gate pass. That gate was built to catch a lossy
//! re-emit and it is the right instrument for one; a comment emitted in the wrong PLACE is a
//! different failure, and it needs an assertion about position rather than about survival.
//! Same blind spot that let the bool-literal desugar ship: `0` formats to `0` forever.
//!
//! THE CAUSE is in `src/fmt/printer.rs`. `Trivia::comments_at` is indexed by original source
//! LINE, and it is consumed only by `emit_leading` before a top-level item. Struct fields are
//! not top-level items and never consume their own line, so a field's comment sits in the map
//! until the next item flushes it — which places it after the struct.
//!
//! THE FIX is additive and needs no new AST variant: `ast::Field` already carries a `span`, so
//! `emit_struct_def` can take the comments at each field's line and emit them as a trailing
//! comment on that field. Un-ignore this test as the acceptance criterion. The same treatment
//! is owed to enum variants (`EnumVariant`) and to match arms, which have the same shape.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default())
        .unwrap_or_else(|e| panic!("format_source failed on:\n{src}\nerror: {e:?}"))
}

const SRC: &str = "struct S {\n    a: i64, // describes a\n    b: i64, // describes b\n}\n\npub fn f(s: &S) -> i64 {\n    s.a\n}\n";

/// The defect. Remove `#[ignore]` when `emit_struct_def` attaches field comments.
#[test]
#[ignore = "known defect: fmt detaches a field's trailing comment and emits it after the struct"]
fn a_field_trailing_comment_stays_on_its_field() {
    let out = fmt(SRC);

    // The comment must appear on the same output line as the field it documents.
    for (field, comment) in [("a: i64", "describes a"), ("b: i64", "describes b")] {
        let on_same_line = out
            .lines()
            .any(|l| l.contains(field) && l.contains(comment));
        assert!(
            on_same_line,
            "`// {comment}` must stay on the `{field}` line, not float free of it.\n\
             --- in ---\n{SRC}\n--- out ---\n{out}"
        );
    }

    // And it must NOT have been pushed outside the struct body.
    let close = out.find("\n}").expect("struct must close");
    assert!(
        !out[close..].contains("describes a"),
        "a field comment was emitted after the struct's closing brace.\n--- out ---\n{out}"
    );
}

/// The defect is REAL and this file is not testing a straw man.
///
/// Runs unignored, and asserts the CURRENT broken behaviour: both comments survive, in order,
/// but land after the closing brace. If this ever fails, the defect has been fixed (or has
/// changed shape) and the ignored test above should be re-run rather than trusted.
#[test]
fn the_defect_is_currently_present_and_this_is_its_exact_shape() {
    let out = fmt(SRC);
    let close = out.find("\n}").expect("struct must close");
    let tail = &out[close..];

    assert!(
        tail.contains("describes a") && tail.contains("describes b"),
        "the recorded defect did not reproduce — fmt no longer emits field comments after the \
         struct. If this is because it was FIXED, remove the #[ignore] from the test above and \
         delete this control.\n--- out ---\n{out}"
    );

    // Nothing is deleted, and relative order holds — this is misplacement, not loss.
    let ia = out.find("describes a").expect("comment a must survive");
    let ib = out.find("describes b").expect("comment b must survive");
    assert!(ia < ib, "comment order must be preserved.\n{out}");

    // Positive control for the assertions above: a LEADING comment on a field is handled
    // correctly today, so the printer's comment machinery works and this is specifically
    // about the trailing position.
    let leading = fmt("struct S {\n    // describes a\n    a: i64,\n}\n\npub fn f(s: &S) -> i64 {\n    s.a\n}\n");
    let close2 = leading.find("\n}").expect("struct must close");
    assert!(
        !leading[close2..].contains("describes a"),
        "a LEADING field comment was also displaced, so this is broader than the trailing \
         position and the report above understates it.\n--- out ---\n{leading}"
    );
}
