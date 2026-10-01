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

//! Failures that stop `mindc check` before any file is checked.
//!
//! A path that does not resolve, a source that cannot be read, or a project scope that cannot
//! be installed (a missing `Mind.toml`, a sibling module that does not parse, a declared source
//! that does not resolve) used to print only to stderr and exit 1 with NOTHING on stdout. A CI
//! step that parses `--reporter json` output then saw an empty result next to a failing status,
//! and a pipeline that reads only stdout reported the file as clean.

use std::path::{Path, PathBuf};

use super::{CheckDiagnostic, CheckPhase, CheckSeverity, ReporterKind, emit_lsp_diagnostics};

/// Report `message` through `reporter` and return exit code 1.
///
/// The human reporter keeps printing it to stderr. The JSON and LSP reporters emit it as a
/// single error diagnostic on stdout, attributed to `file` (or `.` when there is none).
pub(super) fn report(reporter: ReporterKind, file: Option<&Path>, message: &str) -> i32 {
    let diag = diagnostic(file, message);
    let rendered = match reporter {
        ReporterKind::Human => {
            eprintln!("{message}");
            return 1;
        }
        ReporterKind::Json => serde_json::to_string_pretty(std::slice::from_ref(&diag)),
        ReporterKind::Lsp => emit_lsp_diagnostics(std::slice::from_ref(&diag)),
    };
    match rendered {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("{message}");
            eprintln!("error[check]: diagnostic serialisation failed: {e}");
        }
    }
    1
}

/// The diagnostic for [`report`]. The rule id carries the error's own code when the message
/// names one (`error[type-check][E2003]: ...` becomes `type_check::E2003`) and `check::project`
/// otherwise. The leading `error[...]: ` prefix is dropped from the message, because the
/// severity and rule id already say it.
fn diagnostic(file: Option<&Path>, message: &str) -> CheckDiagnostic {
    let code = message.find("][E").and_then(|start| {
        let rest = &message[start + 2..];
        let code = &rest[..rest.find(']')?];
        (code.len() > 1 && code[1..].bytes().all(|b| b.is_ascii_digit())).then_some(code)
    });
    let rule_id = match code {
        Some(code) => format!("type_check::{code}"),
        None => "check::project".to_string(),
    };
    let body = match message.strip_prefix("error[") {
        Some(_) => message.find("]: ").map_or(message, |i| &message[i + 3..]),
        None => message,
    };
    CheckDiagnostic {
        file: file.map_or_else(|| PathBuf::from("."), Path::to_path_buf),
        line: 1,
        col: 1,
        severity: CheckSeverity::Error,
        message: body.to_string(),
        rule_id,
        phase: CheckPhase::TypeCheck,
        help: None,
        auto_fix: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_coded_message_keeps_its_code_and_loses_its_prefix() {
        let d = diagnostic(
            Some(Path::new("main.mind")),
            "error[type-check][E2003]: local import(s) `sibling` require a project",
        );
        assert_eq!(d.rule_id, "type_check::E2003");
        assert_eq!(d.message, "local import(s) `sibling` require a project");
        assert_eq!(d.file, PathBuf::from("main.mind"));
        assert_eq!(d.severity, CheckSeverity::Error);
    }

    #[test]
    fn an_uncoded_message_is_a_project_failure() {
        let d = diagnostic(None, "error[check]: path not found: x.mind");
        assert_eq!(d.rule_id, "check::project");
        assert_eq!(d.message, "path not found: x.mind");
        assert_eq!(d.file, PathBuf::from("."));
    }

    #[test]
    fn a_bracket_that_is_not_a_code_is_not_taken_for_one() {
        // `][E` followed by something other than digits must not become a rule id.
        let d = diagnostic(None, "error[a][Ex]: m");
        assert_eq!(d.rule_id, "check::project");
        assert_eq!(d.message, "m");
    }
}
