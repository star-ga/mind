#[cfg(feature = "cpu-exec")]
pub mod cpu;

#[cfg(not(feature = "cpu-exec"))]
mod cpu_disabled {
    #[allow(dead_code)]
    pub struct CpuExecUnavailable;
}
