// Copyright 2025-2026 STARGA Inc.
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

//! RSASSA-PSS salt-length bound of the x509 chain verifier.
//!
//! RFC 8017 §9.1.2 allows `sLen <= emLen - hLen - 2`, which depends on the
//! issuer key: 222 for RSA-2048, 350 for RSA-3072, 478 for RSA-4096 (SHA-256).
//! The verifier derives the bound from the issuer's modulus, so a long-salt
//! signature under a larger key verifies and one past that key's bound is
//! refused before any signature arithmetic.
//!
//! Fixtures: `tests/x509/testdata/adv/pss{3072,4096}_long_*` (see the README).
//!
//! Gate: `cargo test --features "mlir-build std-surface cross-module-imports"
//!        --test x509_pss_salt`.

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#![cfg(not(windows))]

mod common;
use common::x509_so::{Lib, now_unix, run};

const HOST: &str = "adv.example.com";

/// A leaf signed by an RSA-3072 root with the largest salt that key allows
/// (350) verifies; so does RSA-4096 with its largest (478).
#[test]
fn x509_pss_long_salt_under_large_issuer_verifies() {
    let Some(lib) = Lib::get() else { return };
    let now = now_unix();
    assert_eq!(
        run(
            lib,
            &["adv/pss3072_long_leaf"],
            &["adv/pss3072_long_root"],
            HOST,
            now
        ),
        0,
        "salt 350 under RSA-3072"
    );
    assert_eq!(
        run(
            lib,
            &["adv/pss4096_long_leaf"],
            &["adv/pss4096_long_root"],
            HOST,
            now
        ),
        0,
        "salt 478 under RSA-4096"
    );
}

/// The same RSA-3072 leaf with saltLength 351 (one past the key's bound, both
/// copies of the AlgorithmIdentifier patched) is refused as an unsupported
/// parameter (-2), not as a bad signature (-3): the bound check comes first.
/// The unpatched leaf above is the positive control.
#[test]
fn x509_pss_salt_past_the_issuer_key_bound_is_refused() {
    let Some(lib) = Lib::get() else { return };
    let now = now_unix();
    let got = run(
        lib,
        &["adv/pss3072_long_leaf_salt351"],
        &["adv/pss3072_long_root"],
        HOST,
        now,
    );
    assert_eq!(got, WANT_OVER_BOUND, "salt 351 under RSA-3072");
    // A 2048-bit issuer with a salt of 32 (the committed pss4096 chain) is
    // still accepted.
    assert_eq!(
        run(
            lib,
            &["adv/pss4096_leaf", "adv/pss4096_inter2048"],
            &["adv/pss4096_root"],
            HOST,
            now
        ),
        0
    );
}

const WANT_OVER_BOUND: i64 = -2;
