use mind::conformance::{run_conformance, ConformanceOptions, ConformanceProfile};

#[cfg(not(debug_assertions))]
#[ignore]
#[test]
fn _ignore_in_release_mode() {}

#[test]
fn cpu_conformance_profile_passes() {
    run_conformance(ConformanceOptions {
        profile: ConformanceProfile::CpuBaseline,
    })
    .expect("CPU conformance suite should pass");
}

#[cfg(feature = "mlir-gpu")]
#[test]
fn gpu_profile_runs_when_enabled() {
    run_conformance(ConformanceOptions {
        profile: ConformanceProfile::CpuAndGpu,
    })
    .expect("GPU conformance profile should be executed");
}
