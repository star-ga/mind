// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 6.5 Stage 1b — the runtime-support C stub as mindc bundles it.
//!
//! Bundled at compile time (like std/*.mind in Phase C) so the --emit-shared
//! cdylib path needs no external file at build time. The text is compiled to a
//! temporary .o by `compile_runtime_support_obj` and statically linked into
//! every --emit-shared output, making the .so self-contained.
//!
//! `runtime-support/mind_intrinsics.c` keeps focused parts of itself in sibling
//! files that it `#include`s by quoted name. A compile of the file on disk finds
//! them beside it; the bundle has no directory, so each one is spliced in at
//! its own include line. The stub is therefore still written and compiled as
//! ONE source under the fixed `mind_intrinsics.c` basename (RFC 0015
//! bit-identity), and the object cache key, which hashes this text, covers
//! every byte the compiler reads.

use std::sync::LazyLock;

const MIND_INTRINSICS_C: &str = include_str!("../../runtime-support/mind_intrinsics.c");

/// Every file `mind_intrinsics.c` includes by quoted name, with its text.
const QUOTED_INCLUDES: &[(&str, &str)] = &[(
    "mind_fs_read.h",
    include_str!("../../runtime-support/mind_fs_read.h"),
)];

/// The stub with each quoted include spliced in at its include line.
pub(super) static MIND_RUNTIME_SUPPORT_C: LazyLock<String> =
    LazyLock::new(|| splice_includes(MIND_INTRINSICS_C, QUOTED_INCLUDES));

/// The function names the runtime support object defines: every
/// `MIND_EXPORT` / `MIND_EXPORT_WEAK` definition of the whole bundle. A MIND
/// function with one of these names keeps it as its object symbol (see
/// `mlir::nobuiltin`), so a call binds as it always did.
pub(crate) fn exported_names() -> impl Iterator<Item = &'static str> {
    MIND_RUNTIME_SUPPORT_C.lines().filter_map(|line| {
        // The `#  define MIND_EXPORT ...` lines start with `#` and never match.
        let rest = line.trim_start().strip_prefix("MIND_EXPORT")?;
        let rest = rest.strip_prefix("_WEAK").unwrap_or(rest);
        let name = rest[..rest.find('(')?]
            .trim()
            .rsplit(char::is_whitespace)
            .next()?;
        Some(name.trim_start_matches('*')).filter(|name| !name.is_empty())
    })
}

/// `src` with every line that is exactly `#include "<name>"` for a listed
/// `name` replaced by that file's text. Any other line is kept as written, so
/// an include missing from the list stays visible (and fails the compile).
fn splice_includes(src: &str, includes: &[(&str, &str)]) -> String {
    let directives: Vec<(String, &str)> = includes
        .iter()
        .map(|(name, text)| (format!("#include \"{name}\""), *text))
        .collect();
    let spliced: usize = includes.iter().map(|(_, text)| text.len()).sum();
    let mut out = String::with_capacity(src.len() + spliced);
    for line in src.split_inclusive('\n') {
        let directive = line.trim_end_matches(['\n', '\r']);
        match directives.iter().find(|(d, _)| d == directive) {
            Some((_, text)) => out.push_str(text),
            None => out.push_str(line),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quoted_include_lines(text: &str) -> Vec<&str> {
        text.lines()
            .filter(|l| l.trim_start().starts_with("#include \""))
            .collect()
    }

    #[test]
    fn every_quoted_include_of_the_stub_is_spliced() {
        // Positive control: the file on disk does include a sibling, so the
        // assertion below cannot pass because there was nothing to splice.
        assert!(
            quoted_include_lines(MIND_INTRINSICS_C).contains(&"#include \"mind_fs_read.h\""),
            "mind_intrinsics.c no longer includes mind_fs_read.h; update QUOTED_INCLUDES"
        );
        let left = quoted_include_lines(&MIND_RUNTIME_SUPPORT_C);
        assert!(
            left.is_empty(),
            "the bundled stub still has quoted includes {left:?}: the bundle is \
             compiled alone in a temp dir, so each must be listed in QUOTED_INCLUDES"
        );
    }

    #[test]
    fn the_bundle_defines_the_reader_exactly_once() {
        let bundle: &str = &MIND_RUNTIME_SUPPORT_C;
        for (name, text) in QUOTED_INCLUDES {
            assert!(text.ends_with('\n'), "{name} must end with a newline");
            assert_eq!(bundle.matches(text).count(), 1, "{name} spliced once");
        }
        for def in [
            "MIND_EXPORT int64_t is_dir_fd(int64_t fd) {",
            "MIND_EXPORT_WEAK int64_t read_to_string(int64_t path_addr) {",
        ] {
            assert_eq!(bundle.matches(def).count(), 1, "{def}");
        }
    }

    #[test]
    fn exported_names_cover_the_stub_and_its_spliced_includes() {
        let names: Vec<&str> = exported_names().collect();
        for name in [
            "vec_new",
            "vec_push",
            "__mind_write",
            "is_dir_fd",
            "read_to_string",
        ] {
            assert!(names.contains(&name), "{name} is a runtime-support export");
        }
        // A libc name is never a runtime-support export, and the parse reads
        // names, not declarator punctuation.
        for name in ["write", "memcpy", "malloc", "pow"] {
            assert!(!names.contains(&name), "{name} must not be an export");
        }
        assert!(names.len() > 100, "only {} exports parsed", names.len());
        assert!(
            names
                .iter()
                .all(|n| n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
        );
    }

    #[test]
    fn splice_replaces_only_a_listed_include_line() {
        let src = "a\n#include \"x.h\"\r\n#include \"y.h\"\n#include <z.h>\nb";
        let got = splice_includes(src, &[("x.h", "X1\nX2\n")]);
        assert_eq!(got, "a\nX1\nX2\n#include \"y.h\"\n#include <z.h>\nb");
    }
}
