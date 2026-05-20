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

//! Built-in lint rules.
//!
//! Phase 3 ships one proof-of-life rule (`trailing_whitespace`) to validate
//! the infrastructure end-to-end.  Phase 4 adds the five named production
//! rules (`q16_overflow`, `unused_import`, `naming_convention`, `shadowing`,
//! and one additional rule).

pub mod trailing_whitespace;

pub use trailing_whitespace::TrailingWhitespace;

use super::rule::RuleRegistry;

/// Register all built-in Phase 3 rules into `registry`.
///
/// Call this once at startup before invoking [`super::run_lint`].
pub fn register_defaults(registry: &mut RuleRegistry) {
    registry.register(TrailingWhitespace);
}
