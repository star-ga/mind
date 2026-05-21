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

//! Run all Phase 4 lint rules against `std/vec.mind` and print diagnostics.
//!
//! This test always passes — it is a reporting fixture, not a gate.
//! The test prints the diagnostic list so the caller can inspect what fires.

use std::path::Path;

use libmind::lint::check_source;
use libmind::lint::rules::register_defaults;
use libmind::lint::rule::RuleRegistry;
use libmind::project::MindcraftConfig;

#[test]
fn lint_std_vec_mind() {
    let vec_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("std/vec.mind");
    let source = std::fs::read_to_string(&vec_path)
        .expect("std/vec.mind must exist");

    let mut registry = RuleRegistry::new();
    register_defaults(&mut registry);

    let config = MindcraftConfig::default();
    let diags = check_source(&source, &vec_path, &config, &registry);

    println!("--- lint diagnostics for std/vec.mind ({} total) ---", diags.len());
    for d in &diags {
        println!(
            "[{}] {} | span={}..{} | {}",
            d.rule_id, d.message, d.span.start, d.span.end,
            d.help.as_deref().unwrap_or("")
        );
    }

    // Group by rule_id for summary.
    let mut by_rule: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for d in &diags {
        *by_rule.entry(d.rule_id.as_str()).or_default() += 1;
    }
    println!("--- summary ---");
    for (rule, count) in &by_rule {
        println!("  {rule}: {count}");
    }

    // This test always passes — it is a reporting fixture.
    // No assertion: we just want to see what fires.
}
