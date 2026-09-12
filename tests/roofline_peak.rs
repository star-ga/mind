//! Correctness proof for the CPUID-gated roofline denominator (B4).
//!
//! Drives the pure, clock-parameterized `isa_peak_gmacs` with CPUID stubs so the
//! `%-of-ISA-peak` axis can never again apply a Skylake two-port ceiling to a
//! one-port Haswell. Pure math over the shared bench module — no host state, no
//! emitted bytes, not on any compile/canary path.

#[allow(dead_code)]
mod roofline {
    include!("../benches/common/roofline_peak.rs");
}

use roofline::{
    IsaPeak, MACS_PER_UOP_VPMADDWD, MACS_PER_UOP_VPMULDQ, isa_peak_gmacs, x86_madd_ports,
};

const GHZ: f64 = 3.5;

// Representative CPUID (family, model) stubs.
const HASWELL: (u32, u32) = (6, 0x3C);
const BROADWELL: (u32, u32) = (6, 0x4F);
const SKYLAKE: (u32, u32) = (6, 0x5E);
const SKYLAKE_SP: (u32, u32) = (6, 0x55);
const ZEN3: (u32, u32) = (0x19, 0x21);
const UNKNOWN: (u32, u32) = (6, 0xFF);

#[test]
fn haswell_is_one_port_56_vpmaddwd_14_vpmuldq() {
    let (f, m) = HASWELL;
    assert_eq!(x86_madd_ports(f, m), Some(1), "Haswell issues on one port");
    assert_eq!(
        isa_peak_gmacs(f, m, GHZ, MACS_PER_UOP_VPMADDWD),
        IsaPeak::Known(56.0),
        "Haswell vpmaddwd ceiling = 3.5 GHz x 16 MAC x 1 port"
    );
    assert_eq!(
        isa_peak_gmacs(f, m, GHZ, MACS_PER_UOP_VPMULDQ),
        IsaPeak::Known(14.0),
        "Haswell vpmuldq ceiling = 3.5 GHz x 4 MAC x 1 port"
    );
}

#[test]
fn broadwell_is_one_port() {
    let (f, m) = BROADWELL;
    assert_eq!(x86_madd_ports(f, m), Some(1));
    assert_eq!(
        isa_peak_gmacs(f, m, GHZ, MACS_PER_UOP_VPMADDWD),
        IsaPeak::Known(56.0)
    );
}

#[test]
fn skylake_is_two_port_112() {
    for (f, m) in [SKYLAKE, SKYLAKE_SP, ZEN3] {
        assert_eq!(
            x86_madd_ports(f, m),
            Some(2),
            "{f:#x}/{m:#x} issues on two ports"
        );
        assert_eq!(
            isa_peak_gmacs(f, m, GHZ, MACS_PER_UOP_VPMADDWD),
            IsaPeak::Known(112.0),
            "two-port vpmaddwd ceiling = 3.5 GHz x 16 MAC x 2 ports"
        );
        assert_eq!(
            isa_peak_gmacs(f, m, GHZ, MACS_PER_UOP_VPMULDQ),
            IsaPeak::Known(28.0)
        );
    }
}

#[test]
fn unknown_model_reports_both_ceilings() {
    let (f, m) = UNKNOWN;
    assert_eq!(
        x86_madd_ports(f, m),
        None,
        "unclassified uarch => no port guess"
    );
    assert_eq!(
        isa_peak_gmacs(f, m, GHZ, MACS_PER_UOP_VPMADDWD),
        IsaPeak::Unknown {
            one_port: 56.0,
            two_port: 112.0
        },
        "unknown host must print BOTH the 1-port and 2-port ceilings, not guess"
    );
    assert_eq!(
        isa_peak_gmacs(f, m, GHZ, MACS_PER_UOP_VPMULDQ),
        IsaPeak::Unknown {
            one_port: 14.0,
            two_port: 28.0
        }
    );
}
