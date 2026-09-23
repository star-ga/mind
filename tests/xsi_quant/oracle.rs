// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Rust oracles for the quant kernels, transcribed from the .mind sources.

// =============================================================================
// examples/quant/* — cross-substrate byte-identity fixtures (quant-*).
//
// Ten deterministic scalar f64 pricing kernels, one per examples/quant/<name>
// package, each built to its OWN `.so` via `build_quant_so` (the generic
// examples/-tree builder above) because mindc 0.10.2 cannot carry an f64
// across a module boundary (roadmap 17.1) — every quant example is therefore
// fully self-contained, exactly like `examples/quant/*/main.mind` documents.
//
// Every fixture evaluates its kernel over a SMALL (8-point) fixed deterministic
// input grid of exact-representable f64 constants, folds the grid to one hash
// via `canonical_hash_f64_grid` (manifest `output_encoding =
// "f64_bits_i64_le_grid"`), and pins that hash to the committed per-substrate
// reference — same `pin_or_bless` mechanism every other fixture in this file
// uses. See each `tests/cross_substrate_identity/quant-*/manifest.toml` for
// the exact grid, the per-fixture avx2==neon argument, and — for the five
// fixtures with no within-run Rust oracle — the explicit statement of why.
//
// WITHIN-RUN ORACLE COVERAGE. Five of the ten fixtures (black_scholes,
// bond_curve, greeks_higher_order, implied_vol_surface, portfolio_risk) are
// closed-form / fixed-loop-order scalar chains with a straightforward,
// faithful Rust port below (`ref_*`), checked byte-exact against the kernel
// EVERY run, not merely at bless time. The other five (american_binomial,
// barrier_analytic, crr_binomial, implied_vol, monte_carlo_gbm) have NO Rust
// oracle here — each manifest states exactly why (a lattice buffer-walk, a
// large closed-form transcription, or an input-dependent root-find/RNG stream
// whose faithful port would be a large surface of its own) — and rely on the
// pinned committed hash alone, as the remaining gate.
//
// --- shared math kernel oracle, ported verbatim from
// examples/quant/black_scholes/main.mind's KERNEL BEGIN..END block (every
// quant example vendors a byte-identical copy of this block — see
// quant_shared_kernels_byte_identical_across_examples, which enforces that byte-identity
// across the ten .mind sources; this Rust port targets that one shared
// definition, not ten separate ones). Every step below reproduces the MIND
// kernel's exact op order and constants; `sqrt` is MIND's `llvm.intr.sqrt`
// (IEEE-754 correctly-rounded), matching Rust's `f64::sqrt`.

// Ported for parity with the MIND KERNEL BEGIN..END block, though none of the
// oracles below happen to call it directly (each oracle's own branch/qNaN
// checks make absf unnecessary at the call sites used here).
#[allow(dead_code)]
pub(crate) fn ref_absf(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

pub(crate) fn ref_inf() -> f64 {
    let mut v: f64 = 1.0;
    let mut i: i64 = 0;
    while i < 350 {
        v *= 10.0;
        i += 1;
    }
    v
}

pub(crate) fn ref_canonical_qnan() -> f64 {
    f64::from_bits(9221120237041090560u64)
}

pub(crate) fn ref_rnd_to_i64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        0 - ((0.5 - x) as i64)
    }
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_det_exp(x: f64) -> f64 {
    let ln2_hi: f64 = 0.6931467056274414;
    let ln2_lo: f64 = 0.00000047493250390941725;
    let inv_ln2: f64 = 1.4426950408889634;
    let k: i64 = ref_rnd_to_i64(x * inv_ln2);
    let kf: f64 = k as f64;
    let r: f64 = (x - kf * ln2_hi) - kf * ln2_lo;
    let mut t: f64 = 0.0000000000007647163731819816;
    t = 0.00000000016059043836821613 + r * t;
    t = 0.00000000208767569878681 + r * t;
    t = 0.00000002505210838544172 + r * t;
    t = 0.0000002755731922398589 + r * t;
    t = 0.0000027557319223985893 + r * t;
    t = 0.0000248015873015873 + r * t;
    t = 0.0001984126984126984 + r * t;
    t = 0.001388888888888889 + r * t;
    t = 0.008333333333333333 + r * t;
    t = 0.041666666666666664 + r * t;
    t = 0.16666666666666666 + r * t;
    t = 0.5 + r * t;
    t = 1.0 + r * t;
    t = 1.0 + r * t;
    let mut y: f64 = t * 1.0;
    let mut j: i64 = 0;
    if k >= 0 {
        while j < k {
            y *= 2.0;
            j += 1;
        }
    } else {
        while j < (0 - k) {
            y /= 2.0;
            j += 1;
        }
    }
    y
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_dm_exp(x: f64) -> f64 {
    if x != x {
        return x;
    }
    if x >= 709.7827128933841 {
        return ref_inf();
    }
    if x <= 0.0 - 745.1332191019412 {
        return 0.0;
    }
    ref_det_exp(x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_det_log_core(x: f64) -> f64 {
    let ln2: f64 = 0.6931471805599453;
    let mut k: i64 = 0;
    let mut m: f64 = x * 1.0;
    while m >= 1.4142135623730951 {
        m /= 2.0;
        k += 1;
    }
    while m < 0.7071067811865476 {
        m *= 2.0;
        k -= 1;
    }
    let s: f64 = (m - 1.0) / (m + 1.0);
    let s2: f64 = s * s;
    let mut acc: f64 = 0.0;
    let mut n: i64 = 29;
    while n >= 1 {
        acc = 1.0 / (n as f64) + s2 * acc;
        n -= 2;
    }
    (k as f64) * ln2 + 2.0 * s * acc
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_dm_log(x: f64) -> f64 {
    if x != x {
        return x;
    }
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0 - ref_inf();
    }
    if x >= ref_inf() {
        return ref_inf();
    }
    ref_det_log_core(x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_dm_sqrt(x: f64) -> f64 {
    if x != x {
        return ref_canonical_qnan();
    }
    if x < 0.0 {
        return ref_canonical_qnan();
    }
    x.sqrt()
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_erf_series(x: f64) -> f64 {
    let two_over_sqrtpi: f64 = 1.1283791670955126;
    let x2: f64 = x * x;
    let mut t: f64 = x * 1.0;
    let mut s: f64 = x * 1.0;
    let mut n: i64 = 1;
    while n < 60 {
        t = t * (0.0 - x2) / (n as f64);
        s += t / ((2 * n + 1) as f64);
        n += 1;
    }
    two_over_sqrtpi * s
}

pub(crate) fn ref_erfc_cf(x: f64) -> f64 {
    let sqrtpi: f64 = 1.7724538509055159;
    let mut acc: f64 = 0.0;
    let mut n: i64 = 200;
    while n >= 1 {
        acc = ((n as f64) * 0.5) / (x + acc);
        n -= 1;
    }
    ref_det_exp(0.0 - x * x) / sqrtpi / (x + acc)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_dm_erfc(x: f64) -> f64 {
    if x != x {
        return x;
    }
    if x >= ref_inf() {
        return 0.0;
    }
    if x <= 0.0 - ref_inf() {
        return 2.0;
    }
    if x < 0.0 {
        return 2.0 - ref_dm_erfc(0.0 - x);
    }
    if x < 1.0 {
        return 1.0 - ref_erf_series(x);
    }
    ref_erfc_cf(x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_dm_norm_cdf(x: f64) -> f64 {
    if x != x {
        return x;
    }
    let inv_sqrt2: f64 = 0.7071067811865476;
    0.5 * ref_dm_erfc(0.0 - x * inv_sqrt2)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_dm_norm_pdf(x: f64) -> f64 {
    if x != x {
        return x;
    }
    let inv_sqrt_2pi: f64 = 0.3989422804014327;
    inv_sqrt_2pi * ref_det_exp(0.0 - 0.5 * x * x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_bs_valid(s: f64, k: f64, sigma: f64, t: f64) -> bool {
    !(s != s
        || k != k
        || sigma != sigma
        || t != t
        || s <= 0.0
        || k <= 0.0
        || sigma <= 0.0
        || t <= 0.0)
}

pub(crate) fn ref_bs_d1(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sq: f64 = sigma * ref_dm_sqrt(t);
    (ref_dm_log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / sq
}

// --- quant-black-scholes: ref_bs_call ---------------------------------------
pub(crate) fn ref_bs_call(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if !ref_bs_valid(s, k, sigma, t) {
        return ref_canonical_qnan();
    }
    let d1: f64 = ref_bs_d1(s, k, r, q, sigma, t);
    let d2: f64 = d1 - sigma * ref_dm_sqrt(t);
    let df_r: f64 = ref_dm_exp(0.0 - r * t);
    let df_q: f64 = ref_dm_exp(0.0 - q * t);
    s * df_q * ref_dm_norm_cdf(d1) - k * df_r * ref_dm_norm_cdf(d2)
}

// --- quant-bond-curve: ref_bond_price ---------------------------------------
// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_bond_valid(
    face: f64,
    coupon_rate: f64,
    ytm: f64,
    freq: i64,
    n_periods: i64,
) -> bool {
    if face != face || coupon_rate != coupon_rate || ytm != ytm {
        return false;
    }
    if freq <= 0 || n_periods <= 0 || face <= 0.0 || coupon_rate < 0.0 {
        return false;
    }
    let f: f64 = freq as f64;
    !(1.0 + ytm / f <= 0.0)
}

pub(crate) fn ref_df_single(ytm: f64, freq: f64) -> f64 {
    1.0 / (1.0 + ytm / freq)
}

pub(crate) fn ref_df_pow(v: f64, i: i64) -> f64 {
    let mut acc: f64 = 1.0 * 1.0;
    let mut j: i64 = 0;
    while j < i {
        acc *= v;
        j += 1;
    }
    acc
}

pub(crate) fn ref_bond_price(
    face: f64,
    coupon_rate: f64,
    ytm: f64,
    freq: i64,
    n_periods: i64,
) -> f64 {
    if !ref_bond_valid(face, coupon_rate, ytm, freq, n_periods) {
        return ref_canonical_qnan();
    }
    let f: f64 = freq as f64;
    let v: f64 = ref_df_single(ytm, f);
    let cpn: f64 = face * coupon_rate / f;
    let mut acc: f64 = 0.0 * 1.0;
    let mut i: i64 = 1;
    while i <= n_periods {
        acc += cpn * ref_df_pow(v, i);
        i += 1;
    }
    acc += face * ref_df_pow(v, n_periods);
    acc
}

// --- quant-greeks-higher-order: ref_hog_vanna -------------------------------
pub(crate) fn ref_hog_vanna(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if !ref_bs_valid(s, k, sigma, t) {
        return ref_canonical_qnan();
    }
    let d1: f64 = ref_bs_d1(s, k, r, q, sigma, t);
    let d2: f64 = d1 - sigma * ref_dm_sqrt(t);
    0.0 - ref_dm_exp(0.0 - q * t) * ref_dm_norm_pdf(d1) * d2 / sigma
}

// --- quant-implied-vol-surface: ref_svi_total_variance ----------------------
pub(crate) fn ref_svi_w_min_raw(a: f64, b: f64, rho: f64, p_sigma: f64) -> f64 {
    a + b * p_sigma * ref_dm_sqrt(1.0 - rho * rho)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_svi_params_valid(a: f64, b: f64, rho: f64, m: f64, p_sigma: f64) -> bool {
    if a != a || b != b || rho != rho || m != m || p_sigma != p_sigma {
        return false;
    }
    if b < 0.0 || rho <= 0.0 - 1.0 || rho >= 1.0 || p_sigma <= 0.0 {
        return false;
    }
    !(ref_svi_w_min_raw(a, b, rho, p_sigma) < 0.0)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_svi_total_variance(
    k: f64,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    p_sigma: f64,
) -> f64 {
    if !ref_svi_params_valid(a, b, rho, m, p_sigma) {
        return ref_canonical_qnan();
    }
    if k != k {
        return ref_canonical_qnan();
    }
    let km: f64 = k - m;
    a + b * (rho * km + ref_dm_sqrt(km * km + p_sigma * p_sigma))
}

// --- quant-portfolio-risk: ref_two_asset_vol ---------------------------------
// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
pub(crate) fn ref_two_asset_vol(w1: f64, w2: f64, s1: f64, s2: f64, rho: f64) -> f64 {
    if s1 != s1 || s2 != s2 || rho != rho {
        return ref_canonical_qnan();
    }
    if s1 < 0.0 || s2 < 0.0 {
        return ref_canonical_qnan();
    }
    if rho < 0.0 - 1.0 || rho > 1.0 {
        return ref_canonical_qnan();
    }
    let v: f64 = w1 * w1 * s1 * s1 + w2 * w2 * s2 * s2 + 2.0 * w1 * w2 * rho * s1 * s2;
    ref_dm_sqrt(v)
}
