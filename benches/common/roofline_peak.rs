// Shared roofline denominator for the deterministic GEMM / int-dot benches.
//
// The `%-of-ISA-peak` axis printed by det_matmul_{q16,i8,i16} and det_gemv_q16_mt
// needs an HONEST denominator: the single-core integer multiply-accumulate ceiling
// the emitted microkernel can actually retire on THIS host, not a best-case number
// borrowed from a newer microarchitecture.
//
// The ceiling is `f_GHz * macs_per_uop * issue_ports`:
//   * vpmaddwd (int8/int16 path): 16 i16-MACs per uop.
//   * vpmuldq  (Q16.16 path):      4 i64-MACs per uop.
//   * issue_ports: Haswell and Broadwell issue these on ONE integer-vector port;
//     Skylake+ and Zen2+ issue on TWO. Applying the two-port (Skylake) number to a
//     one-port Haswell doubles the denominator and halves the reported %-of-peak —
//     the exact over-statement this module removes.
//
// When the host microarchitecture is not recognized we do NOT guess a port count:
// we report BOTH the one-port and two-port ceilings so the reader can bracket the
// true %-of-peak instead of trusting a coin flip.
//
// This is a pure bench-side denominator. It emits no compiled kernel bytes, is on
// no artifact/emitter path, and cannot move a canary hash.

/// int-MAC micro-op width for the int8 / int16 `vpmaddwd` path: 16×i16 -> 8×i32.
pub const MACS_PER_UOP_VPMADDWD: f64 = 16.0;
/// int-MAC micro-op width for the Q16.16 `vpmuldq` path: 4×i64 widen-multiply.
pub const MACS_PER_UOP_VPMULDQ: f64 = 4.0;

/// Reference clock used when a perf-counter measurement is unavailable (see
/// `host_clock_ghz`). Nominal, labelled as such wherever it is printed.
pub const REF_GHZ_X86: f64 = 3.5;
/// Conservative aarch64 reference clock (kept from the prior per-bench estimates;
/// no ARM host is available here to re-derive it).
pub const REF_GHZ_NEON: f64 = 3.0;

/// Which emitted int-MAC micro-op a bench's kernel retires — selects the MAC width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntUop {
    /// int8 / int16 `_mm256_madd_epi16` path.
    Vpmaddwd,
    /// Q16.16 `vpmuldq` outer-product path.
    Vpmuldq,
}

impl IntUop {
    fn macs_per_uop(self) -> f64 {
        match self {
            IntUop::Vpmaddwd => MACS_PER_UOP_VPMADDWD,
            IntUop::Vpmuldq => MACS_PER_UOP_VPMULDQ,
        }
    }
    /// Conservative aarch64 single-core ceiling (GMAC/s), preserved from the prior
    /// per-bench constants (SDOT ~48, 4×i32 widen-MAC ~24).
    fn neon_peak(self) -> f64 {
        match self {
            IntUop::Vpmaddwd => 48.0,
            IntUop::Vpmuldq => 24.0,
        }
    }
}

/// Single-core integer-MAC ceiling for a given microarchitecture and clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IsaPeak {
    /// Port count is known -> one definite ceiling (GMAC/s).
    Known(f64),
    /// Microarchitecture unclassified -> report both ceilings and let the reader
    /// bracket the truth. `one_port <= two_port`.
    Unknown { one_port: f64, two_port: f64 },
    /// Neither x86_64 nor aarch64: no documented estimate.
    Unavailable,
}

impl IsaPeak {
    /// Scale the ceiling(s) by an all-core factor (used by the MT GEMV bench).
    pub fn scale(self, factor: f64) -> IsaPeak {
        match self {
            IsaPeak::Known(p) => IsaPeak::Known(p * factor),
            IsaPeak::Unknown { one_port, two_port } => IsaPeak::Unknown {
                one_port: one_port * factor,
                two_port: two_port * factor,
            },
            IsaPeak::Unavailable => IsaPeak::Unavailable,
        }
    }
}

/// Number of integer-vector issue ports for `vpmaddwd` / `vpmuldq` on the given
/// x86 microarchitecture (`family`/`model` are the CPUID display values).
///
/// `Some(1)` Haswell/Broadwell, `Some(2)` Skylake+/Zen2+, `None` unclassified.
pub fn x86_madd_ports(family: u32, model: u32) -> Option<u8> {
    match (family, model) {
        // Intel Haswell
        (6, 0x3C) | (6, 0x3F) | (6, 0x45) | (6, 0x46) => Some(1),
        // Intel Broadwell
        (6, 0x3D) | (6, 0x47) | (6, 0x4F) | (6, 0x56) => Some(1),
        // Intel Skylake / Kaby / Coffee / Whiskey / Comet / Amber (client)
        (6, 0x4E) | (6, 0x5E) | (6, 0x8E) | (6, 0x9E) | (6, 0xA5) | (6, 0xA6) => Some(2),
        // Intel Skylake-SP / Cascade Lake / Cooper Lake (server)
        (6, 0x55) => Some(2),
        // Intel Ice / Tiger / Rocket / Alder / Raptor / Sapphire (representative)
        (6, 0x6A) | (6, 0x6C) | (6, 0x7D) | (6, 0x7E) | (6, 0x8C) | (6, 0x8D) | (6, 0x97)
        | (6, 0x9A) | (6, 0xA7) | (6, 0xB7) | (6, 0xBA) | (6, 0xBF) | (6, 0xCF) => Some(2),
        // AMD Zen2 (fam 0x17, model >= 0x30), Zen3/Zen4 (0x19), Zen5 (0x1A)
        (0x17, m) if m >= 0x30 => Some(2),
        (0x19, _) | (0x1A, _) => Some(2),
        _ => None,
    }
}

/// Pure, clock-parameterized single-core ceiling. This is the correctness core the
/// unit test drives with CPUID stubs — no host state, no hidden clock.
pub fn isa_peak_gmacs(family: u32, model: u32, ghz: f64, macs_per_uop: f64) -> IsaPeak {
    match x86_madd_ports(family, model) {
        Some(ports) => IsaPeak::Known(ghz * macs_per_uop * f64::from(ports)),
        None => IsaPeak::Unknown {
            one_port: ghz * macs_per_uop,
            two_port: ghz * macs_per_uop * 2.0,
        },
    }
}

/// CPUID display (family, model) of the running x86 host.
#[cfg(target_arch = "x86_64")]
pub fn host_family_model() -> (u32, u32) {
    // CPUID leaf 1 is architectural on every x86_64 CPU; `__cpuid` is safe here.
    let r = std::arch::x86_64::__cpuid(1);
    let eax = r.eax;
    let base_family = (eax >> 8) & 0xF;
    let base_model = (eax >> 4) & 0xF;
    let ext_model = (eax >> 16) & 0xF;
    let ext_family = (eax >> 20) & 0xFF;
    let family = if base_family == 0xF {
        base_family + ext_family
    } else {
        base_family
    };
    let model = if base_family == 0x6 || base_family == 0xF {
        (ext_model << 4) | base_model
    } else {
        base_model
    };
    (family, model)
}

/// Host clock in GHz plus whether it was MEASURED (perf counters) or NOMINAL.
///
// deferred: prefer measured core cycles via perf_event_open(PERF_COUNT_HW_CPU_CYCLES)
// over a calibrated CLOCK_MONOTONIC window, because nominal undercounts turbo and
// overcounts throttling — stubbed because it needs libc + CAP_PERFMON (perf_event
// paranoid<=1) which is unavailable on the CI/U1 box; upgrade path: gate the syscall
// behind a bench-only `roofline-perf` feature and fall back to nominal (as now) on
// EACCES. Until then we report NOMINAL and say so at every print site.
pub fn host_clock_ghz() -> (f64, bool) {
    (REF_GHZ_X86, false)
}

/// Host single-core int-MAC ceiling for the given emitted micro-op.
pub fn host_isa_peak(uop: IntUop) -> IsaPeak {
    #[cfg(target_arch = "x86_64")]
    {
        let (family, model) = host_family_model();
        let (ghz, _measured) = host_clock_ghz();
        isa_peak_gmacs(family, model, ghz, uop.macs_per_uop())
    }
    #[cfg(all(target_arch = "aarch64", not(target_arch = "x86_64")))]
    {
        IsaPeak::Known(uop.neon_peak())
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = uop;
        IsaPeak::Unavailable
    }
}

/// Human-readable clock label for the print sites.
fn clock_label() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let (ghz, measured) = host_clock_ghz();
        if measured {
            format!("measured {ghz:.2} GHz")
        } else {
            format!("nominal {ghz:.1} GHz")
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        format!("nominal {REF_GHZ_NEON:.1} GHz")
    }
}

/// Format the roofline bracket for `gmacs` against a `peak` (single- or all-core).
pub fn roofline_pct(gmacs: f64, peak: IsaPeak) -> String {
    let clk = clock_label();
    match peak {
        IsaPeak::Known(p) => format!(
            "{:.1}% of ISA peak (~{p:.0} GMAC/s est., {clk})",
            gmacs / p * 100.0
        ),
        IsaPeak::Unknown { one_port, two_port } => format!(
            "{:.1}-{:.1}% of ISA peak (uarch unclassified; ~{one_port:.0} GMAC/s 1-port / \
             ~{two_port:.0} GMAC/s 2-port, {clk})",
            gmacs / two_port * 100.0,
            gmacs / one_port * 100.0
        ),
        IsaPeak::Unavailable => "ISA peak unknown".to_string(),
    }
}

/// Single-core roofline bracket for `gmacs` running the given emitted micro-op.
pub fn roofline_pct_single(gmacs: f64, uop: IntUop) -> String {
    roofline_pct(gmacs, host_isa_peak(uop))
}

/// All-core roofline bracket: the single-core ceiling scaled by `cores`.
pub fn roofline_pct_all_core(gmacs: f64, uop: IntUop, cores: f64) -> String {
    let clk = clock_label();
    match host_isa_peak(uop).scale(cores) {
        IsaPeak::Known(p) => format!(
            "{:.1}% of all-core ISA peak (~{p:.0} GMAC/s est., {cores:.0}c, {clk})",
            gmacs / p * 100.0
        ),
        IsaPeak::Unknown { one_port, two_port } => format!(
            "{:.1}-{:.1}% of all-core ISA peak (uarch unclassified; ~{one_port:.0} GMAC/s 1-port /              ~{two_port:.0} GMAC/s 2-port, {cores:.0}c, {clk})",
            gmacs / two_port * 100.0,
            gmacs / one_port * 100.0
        ),
        IsaPeak::Unavailable => "ISA peak unknown".to_string(),
    }
}
