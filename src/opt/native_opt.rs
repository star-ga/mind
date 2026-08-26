// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Opt-in native `mic@3 → mic@3` optimizer (roadmap C6).
//!
//! The load-bearing architectural decision (see `docs/INDEPENDENCE_ROADMAP.md` §C6):
//! this optimizer is a **canonical IR transform sitting UPSTREAM of emitter
//! divergence**, so both the Rust `lower.rs`/mic@3 oracle and the self-host `nb_*`
//! emitter consume identical *optimized* bytes — each pass is written once and stays
//! oracle-parity-safe by construction, never twice-to-byte-parity.
//!
//! **DEFAULT OFF.** With [`OptLevel::Off`] (the default, when `MIND_NATIVE_OPT` is
//! unset) [`optimize_mic3`] is a no-op and every emitted byte is byte-identical to the
//! un-optimized canonical IR — so keystone / cross-substrate canaries / frozen self-host
//! seeds are untouched until a deliberate opt-in. Enabling the optimizer changes the
//! emitted bytes and is therefore a whole-corpus reseed event (one per landing).
//!
//! **Determinism contract.** Every pass is a *pure function with a pinned schedule*
//! (fixed pass order, fixed iteration count or a provably order-independent fixpoint —
//! never "loop until quiet"), so the optimized `mic@3` — which `trace_hash` anchors,
//! oracle-parity compares, and the canaries pin — is a deterministic function of the
//! input IR, stable run-to-run and across hosts.
//!
//! Planned pass schedule (roadmap order; MIND's `If`/`While` region IR makes these
//! structural rewrites — dominance == region nesting, LICM preheader == the slot before
//! the `While`, dead-arm pruning rewrites the region to its surviving child, no φ):
//! SCCP-lite → strength-reduction (`(op, operand_type)`-keyed) → region-scoped CSE/GVN
//! (state edge in the key) → LICM-on-`While` → linear-scan register allocation.

use crate::ir::IRModule;

/// Native-optimizer level. `Off` (default) is a strict no-op — byte-identical output.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OptLevel {
    /// No native passes; the emitted `mic@3` is byte-identical to the un-optimized
    /// canonical IR. This is the shipped default until the opt-in passes land + reseed.
    #[default]
    Off,
    // Future (each landing = one whole-corpus reseed):
    //   Basic — SCCP-lite + strength-reduction + region-scoped CSE/GVN
    //   Full  — Basic + LICM-on-While + linear-scan RA
}

/// Resolve the opt level from the environment. Deterministic (a single env read, no
/// host-dependent heuristic). Any unrecognized value is `Off` (fail-safe): an unknown
/// flag never silently enables a byte-changing transform.
pub fn opt_level_from_env() -> OptLevel {
    match std::env::var("MIND_NATIVE_OPT").ok().as_deref() {
        // No byte-changing levels are wired yet; every value maps to Off until a pass
        // ships with its reseed. (Kept explicit so adding a level is a one-line change.)
        _ => OptLevel::Off,
    }
}

/// Run the opt-in native `mic@3 → mic@3` optimizer on the canonical IR, in place.
///
/// With [`OptLevel::Off`] this returns immediately, leaving the module — and therefore
/// every emitted byte — unchanged. Future levels dispatch a *pinned* pass schedule here.
pub fn optimize_mic3(module: &mut IRModule, level: OptLevel) {
    match level {
        OptLevel::Off => {
            // Strict no-op: do not touch `module`. Byte-identity is the contract.
            let _ = module;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_the_default() {
        assert_eq!(OptLevel::default(), OptLevel::Off);
    }

    #[test]
    fn env_defaults_to_off() {
        // With no level wired, any environment resolves to Off (fail-safe).
        assert_eq!(opt_level_from_env(), OptLevel::Off);
    }
}
