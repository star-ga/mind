// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! The quant byte-identity gates, one per example.

use super::oracle::*;
use super::*;

pub(crate) const QUANT_BLACK_SCHOLES_GRID: [(f64, f64, f64, f64, f64, f64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
    (120.0, 100.0, 0.05, 0.0, 0.2, 1.0),
    (80.0, 100.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 100.0, 0.05, 0.03, 0.25, 2.0),
    (50.0, 100.0, 0.05, 0.0, 0.2, 0.25),
    (100.0, 100.0, 0.01, 0.0, 0.05, 0.5),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.0027397260273972603),
    (100.0, 80.0, 0.05, 0.0, 0.3, 1.5),
];

#[test]
fn quant_black_scholes_reproducibility_gate() {
    let id = "quant-black-scholes";
    let Some(so) = build_quant_so("black_scholes") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-black-scholes .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"bs_call").expect("bs_call symbol") };

    let mut out = Vec::with_capacity(QUANT_BLACK_SCHOLES_GRID.len());
    for &(s, k, r, q, sigma, t) in QUANT_BLACK_SCHOLES_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t) };
        let oracle = ref_bs_call(s, k, r, q, sigma, t);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: bs_call({s},{k},{r},{q},{sigma},{t}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

pub(crate) const QUANT_BOND_CURVE_GRID: [(f64, f64, f64, i64, i64); 8] = [
    (100.0, 0.05, 0.05, 2, 10),
    (100.0, 0.04, 0.05, 2, 10),
    (1000.0, 0.03, 0.02, 1, 5),
    (100.0, 0.0, 0.05, 2, 10),
    (100.0, 0.06, 0.06, 4, 20),
    (100.0, 0.05, -0.01, 2, 6),
    (100.0, 0.05, 0.05, 12, 3),
    (500.0, 0.045, 0.03, 2, 8),
];

#[test]
fn quant_bond_curve_reproducibility_gate() {
    let id = "quant-bond-curve";
    let Some(so) = build_quant_so("bond_curve") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-bond-curve .so") };
    let f: Symbol<QuantBondPriceFn> = unsafe { lib.get(b"bond_price").expect("bond_price symbol") };

    let mut out = Vec::with_capacity(QUANT_BOND_CURVE_GRID.len());
    for &(face, cr, ytm, freq, n) in QUANT_BOND_CURVE_GRID.iter() {
        let got = unsafe { f(face, cr, ytm, freq, n) };
        let oracle = ref_bond_price(face, cr, ytm, freq, n);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: bond_price({face},{cr},{ytm},{freq},{n}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

#[test]
fn quant_greeks_higher_order_reproducibility_gate() {
    let id = "quant-greeks-higher-order";
    let Some(so) = build_quant_so("greeks_higher_order") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-greeks-higher-order .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"hog_vanna").expect("hog_vanna symbol") };

    let mut out = Vec::with_capacity(QUANT_BLACK_SCHOLES_GRID.len());
    for &(s, k, r, q, sigma, t) in QUANT_BLACK_SCHOLES_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t) };
        let oracle = ref_hog_vanna(s, k, r, q, sigma, t);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: hog_vanna({s},{k},{r},{q},{sigma},{t}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

pub(crate) const QUANT_SVI_GRID: [(f64, f64, f64, f64, f64, f64); 8] = [
    (0.0, 0.04, 0.4, -0.3, 0.0, 0.2),
    (0.1, 0.04, 0.4, -0.3, 0.0, 0.2),
    (-0.1, 0.04, 0.4, -0.3, 0.0, 0.2),
    (0.2, 0.05, 0.3, -0.5, 0.05, 0.15),
    (-0.2, 0.05, 0.3, -0.5, 0.05, 0.15),
    (0.0, 0.02, 0.2, 0.0, 0.0, 0.3),
    (0.05, 0.03, 0.25, -0.2, -0.02, 0.1),
    (0.3, 0.04, 0.4, -0.3, 0.0, 0.2),
];

#[test]
fn quant_implied_vol_surface_reproducibility_gate() {
    let id = "quant-implied-vol-surface";
    let Some(so) = build_quant_so("implied_vol_surface") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol-surface .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe {
        lib.get(b"svi_total_variance")
            .expect("svi_total_variance symbol")
    };

    let mut out = Vec::with_capacity(QUANT_SVI_GRID.len());
    for &(k, a, b, rho, m, p_sigma) in QUANT_SVI_GRID.iter() {
        let got = unsafe { f(k, a, b, rho, m, p_sigma) };
        let oracle = ref_svi_total_variance(k, a, b, rho, m, p_sigma);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: svi_total_variance({k},{a},{b},{rho},{m},{p_sigma}) diverged from the \
             Rust oracle within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

pub(crate) const QUANT_PORTFOLIO_RISK_GRID: [(f64, f64, f64, f64, f64); 8] = [
    (0.5, 0.5, 0.2, 0.3, 0.4),
    (0.7, 0.3, 0.15, 0.25, -0.2),
    (0.3, 0.7, 0.2, 0.2, 1.0),
    (0.5, 0.5, 0.2, 0.3, -1.0),
    (1.0, 0.0, 0.2, 0.3, 0.0),
    (0.6, 0.4, 0.1, 0.5, 0.6),
    (0.25, 0.75, 0.3, 0.1, -0.5),
    (0.9, 0.1, 0.18, 0.35, 0.1),
];

#[test]
fn quant_portfolio_risk_reproducibility_gate() {
    let id = "quant-portfolio-risk";
    let Some(so) = build_quant_so("portfolio_risk") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-portfolio-risk .so") };
    let f: Symbol<QuantF64x5Fn> =
        unsafe { lib.get(b"two_asset_vol").expect("two_asset_vol symbol") };

    let mut out = Vec::with_capacity(QUANT_PORTFOLIO_RISK_GRID.len());
    for &(w1, w2, s1, s2, rho) in QUANT_PORTFOLIO_RISK_GRID.iter() {
        let got = unsafe { f(w1, w2, s1, s2, rho) };
        let oracle = ref_two_asset_vol(w1, w2, s1, s2, rho);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: two_asset_vol({w1},{w2},{s1},{s2},{rho}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-american-binomial: NO within-run Rust oracle (see manifest) -----
pub(crate) const QUANT_AMERICAN_BINOMIAL_GRID: [(f64, f64, f64, f64, f64, f64, i64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100),
    (90.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100),
    (110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100),
    (100.0, 100.0, 0.1, 0.02, 0.3, 0.5, 50),
    (100.0, 100.0, 0.05, 0.0, 0.4, 2.0, 200),
    (80.0, 100.0, 0.08, 0.01, 0.25, 1.0, 100),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.1, 30),
    (120.0, 100.0, 0.06, 0.02, 0.35, 1.5, 150),
];

#[test]
fn quant_american_binomial_reproducibility_gate() {
    let id = "quant-american-binomial";
    let Some(so) = build_quant_so("american_binomial") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-american-binomial .so") };
    let f: Symbol<QuantF64x6I64Fn> = unsafe {
        lib.get(b"bopm_american_put")
            .expect("bopm_american_put symbol")
    };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (lattice backward-induction buffer walk). Every result must still be
    // FINITE (not NaN/inf), which at least proves the lattice ran to
    // completion rather than degenerating.
    let mut out = Vec::with_capacity(QUANT_AMERICAN_BINOMIAL_GRID.len());
    for &(s, k, r, q, sigma, t, n) in QUANT_AMERICAN_BINOMIAL_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t, n) };
        assert!(
            got.is_finite(),
            "{id}: bopm_american_put({s},{k},{r},{q},{sigma},{t},{n}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-barrier-analytic: NO within-run Rust oracle (see manifest) ------
pub(crate) const QUANT_BARRIER_ANALYTIC_GRID: [(f64, f64, f64, f64, f64, f64, f64); 8] = [
    (100.0, 100.0, 90.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 90.0, 85.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 110.0, 95.0, 0.05, 0.0, 0.25, 0.5),
    (100.0, 100.0, 95.0, 0.05, 0.02, 0.3, 2.0),
    (100.0, 100.0, 99.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 105.0, 90.0, 0.03, 0.0, 0.2, 1.0),
    (100.0, 100.0, 80.0, 0.05, 0.0, 0.2, 0.25),
    (100.0, 95.0, 92.0, 0.05, 0.01, 0.22, 1.0),
];

#[test]
fn quant_barrier_analytic_reproducibility_gate() {
    let id = "quant-barrier-analytic";
    let Some(so) = build_quant_so("barrier_analytic") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-barrier-analytic .so") };
    let f: Symbol<QuantF64x7Fn> = unsafe { lib.get(b"bar_doc").expect("bar_doc symbol") };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (large Reiner-Rubinstein block transcription).
    let mut out = Vec::with_capacity(QUANT_BARRIER_ANALYTIC_GRID.len());
    for &(s, k, h, r, q, sigma, t) in QUANT_BARRIER_ANALYTIC_GRID.iter() {
        let got = unsafe { f(s, k, h, r, q, sigma, t) };
        assert!(
            got.is_finite(),
            "{id}: bar_doc({s},{k},{h},{r},{q},{sigma},{t}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-crr-binomial: NO within-run Rust oracle (see manifest) ---------
#[allow(clippy::type_complexity)]
pub(crate) const QUANT_CRR_BINOMIAL_GRID: [(f64, f64, f64, f64, f64, f64, i64, i64, i64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, 1, 0),
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, 0, 1),
    (90.0, 100.0, 0.05, 0.0, 0.2, 1.0, 200, 1, 0),
    (110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 200, 0, 1),
    (100.0, 100.0, 0.1, 0.02, 0.3, 0.5, 50, 1, 1),
    (80.0, 100.0, 0.08, 0.01, 0.25, 1.0, 100, 0, 0),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.1, 30, 1, 0),
    (120.0, 100.0, 0.06, 0.02, 0.35, 1.5, 150, 0, 1),
];

#[test]
fn quant_crr_binomial_reproducibility_gate() {
    let id = "quant-crr-binomial";
    let Some(so) = build_quant_so("crr_binomial") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-crr-binomial .so") };
    let f: Symbol<QuantCrrPriceFn> = unsafe { lib.get(b"crr_price").expect("crr_price symbol") };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (lattice backward-induction buffer walk, distinct implementation from
    // american_binomial).
    let mut out = Vec::with_capacity(QUANT_CRR_BINOMIAL_GRID.len());
    for &(s, k, r, q, sigma, t, n, is_call, american) in QUANT_CRR_BINOMIAL_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t, n, is_call, american) };
        assert!(
            got.is_finite(),
            "{id}: crr_price({s},{k},{r},{q},{sigma},{t},{n},{is_call},{american}) = {got} \
             is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-implied-vol: NO within-run Rust oracle (see manifest) ----------
pub(crate) const QUANT_IMPLIED_VOL_GRID: [(f64, f64, f64, f64, f64, f64); 8] = [
    (10.450583572185568, 100.0, 100.0, 0.05, 0.0, 1.0),
    (26.169043946847296, 120.0, 100.0, 0.05, 0.0, 1.0),
    (1.859419572812186, 80.0, 100.0, 0.05, 0.0, 1.0),
    (14.883718105064894, 100.0, 100.0, 0.05, 0.03, 2.0),
    (
        0.42448595543281803,
        100.0,
        100.0,
        0.05,
        0.0,
        0.0027397260273972603,
    ),
    (5.0, 100.0, 100.0, 0.05, 0.0, 1.0),
    (20.0, 100.0, 100.0, 0.05, 0.0, 1.0),
    (2.5, 90.0, 100.0, 0.03, 0.0, 0.5),
];

#[test]
fn quant_implied_vol_reproducibility_gate() {
    let id = "quant-implied-vol";
    let Some(so) = build_quant_so("implied_vol") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe {
        lib.get(b"implied_vol_call")
            .expect("implied_vol_call symbol")
    };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (input-dependent Newton/bisection branch sequence over a fixed
    // iteration count).
    let mut out = Vec::with_capacity(QUANT_IMPLIED_VOL_GRID.len());
    for &(price, s, k, r, q, t) in QUANT_IMPLIED_VOL_GRID.iter() {
        let got = unsafe { f(price, s, k, r, q, t) };
        assert!(
            got.is_finite(),
            "{id}: implied_vol_call({price},{s},{k},{r},{q},{t}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-monte-carlo-gbm: NO within-run Rust oracle (see manifest) ------
#[allow(clippy::type_complexity)]
pub(crate) const QUANT_MONTE_CARLO_GBM_GRID: [(f64, f64, f64, f64, f64, f64, i64, i64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 12345),
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 7),
    (90.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 42),
    (110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 99),
    (100.0, 100.0, 0.1, 0.02, 0.3, 0.5, 500, 2024),
    (80.0, 100.0, 0.08, 0.01, 0.25, 1.0, 500, 555),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.1, 500, 1),
    (120.0, 100.0, 0.06, 0.02, 0.35, 1.5, 500, 8675309),
];

#[test]
fn quant_monte_carlo_gbm_reproducibility_gate() {
    let id = "quant-monte-carlo-gbm";
    let Some(so) = build_quant_so("monte_carlo_gbm") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-monte-carlo-gbm .so") };
    let f: Symbol<QuantMcCallFn> = unsafe { lib.get(b"mc_call").expect("mc_call symbol") };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (in-kernel deterministic LCG path generator, not the host RNG).
    let mut out = Vec::with_capacity(QUANT_MONTE_CARLO_GBM_GRID.len());
    for &(s, k, r, q, sigma, t, npath, seed) in QUANT_MONTE_CARLO_GBM_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t, npath, seed) };
        assert!(
            got.is_finite(),
            "{id}: mc_call({s},{k},{r},{q},{sigma},{t},{npath},{seed}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

/// `pin_or_bless` (the i64-result helper above) takes an `i64 result` for its
/// error message; the quant fixtures have no single scalar result (they fold
/// an 8-point grid), so this is the grid-shaped sibling: same bless/pin
/// mechanism, no `result_i64` in the panic text.
pub(crate) fn pin_or_bless_hash(id: &str, computed: &str) {
    let substrate = host_substrate();
    if common::gate::bless_mode() {
        emit_bless(id, substrate, computed);
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n\
             Re-bless with MIND_BENCH_BLESS=1 only on an intentional lowering change (RFC 0020 §13)."
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}'. Computed {computed}; \
             bless it with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
    common::xsi_gate::record_measured(id);
}
