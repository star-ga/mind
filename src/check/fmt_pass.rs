// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The `fmt` phase of `mindc check`: a file whose layout differs from what
//! `mindc fmt` would write fails with `fmt::drift`, and a file the formatter
//! refuses to lay out (it would drop source) is reported, never passed unchecked.

use std::path::Path;

use super::{CheckDiagnostic, CheckPhase, CheckSeverity, offset_to_line_col};
use crate::fmt::format_source;
use crate::project::MindcraftConfig;

/// Format-check: parse + format; if output differs from source, emit a
/// `fmt::drift` diagnostic pointing at the first differing line.
pub(super) fn check_fmt(
    path: &Path,
    source: &str,
    config: &MindcraftConfig,
    out: &mut Vec<CheckDiagnostic>,
) {
    let formatted = match format_source(source, &config.format) {
        Ok(f) => f,
        // A formatter gap, not the user's: say so rather than pass the file unchecked.
        Err(crate::fmt::FmtError::LossyFormat(lost)) => {
            let names: Vec<&str> = lost.iter().map(|(name, _)| name.as_str()).collect();
            out.push(CheckDiagnostic {
                file: path.to_path_buf(),
                line: 1,
                col: 1,
                severity: CheckSeverity::Warn,
                message: format!(
                    "the formatter cannot lay out this file without dropping `{}`, so its \
                     formatting was not checked",
                    names.join("`, `")
                ),
                rule_id: "fmt::unformattable".to_string(),
                phase: CheckPhase::Fmt,
                help: None,
                auto_fix: None,
            });
            return;
        }
        Err(_) => return, // parse error — let type-check surface it
    };

    if formatted == source {
        return;
    }

    // Find the first differing line for a precise location.
    let (line, col) = first_diff_position(source, &formatted);

    out.push(CheckDiagnostic {
        file: path.to_path_buf(),
        line,
        col,
        severity: CheckSeverity::Error,
        message: "file is not formatted; run `mindc fmt` to fix".to_string(),
        rule_id: "fmt::drift".to_string(),
        phase: CheckPhase::Fmt,
        help: Some("run `mindc fmt <file>` to auto-format".to_string()),
        // fmt::drift is fixed by rewriting the whole file; no byte-range fix.
        auto_fix: None,
    });
}

/// Find the (1-based line, 1-based col) of the first byte where `a` and `b`
/// differ. Falls back to (1, 1) if identical (should not occur when called
/// after a drift check).
fn first_diff_position(a: &str, b: &str) -> (usize, usize) {
    let diff_offset = a
        .bytes()
        .zip(b.bytes())
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| a.len().min(b.len()));
    offset_to_line_col(a, diff_offset)
}
