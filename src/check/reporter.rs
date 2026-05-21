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

//! Diagnostic reporter trait for `mindc check`.
//!
//! The trait is currently thin — Phase 6 will add streaming variants.

use super::CheckDiagnostic;

/// Trait for emitting check diagnostics.
pub trait Reporter {
    /// Emit all diagnostics.
    fn emit(&self, diags: &[CheckDiagnostic]);
}

/// Human-readable reporter: one line per diagnostic.
pub struct HumanReporter;

impl Reporter for HumanReporter {
    fn emit(&self, diags: &[CheckDiagnostic]) {
        for d in diags {
            println!("{}", d.human_line());
        }
    }
}

/// JSON reporter: emit a JSON array of diagnostic objects.
pub struct JsonReporter;

impl Reporter for JsonReporter {
    fn emit(&self, diags: &[CheckDiagnostic]) {
        match serde_json::to_string_pretty(diags) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("error[check]: JSON serialisation failed: {e}"),
        }
    }
}
