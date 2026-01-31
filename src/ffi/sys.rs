// Copyright 2025-2026 STARGA Inc.
// System-level FFI for anti-debugging and protection
// Cross-platform: Linux, Windows, macOS

use std::os::raw::{c_int, c_long, c_void};

// ============================================================================
// LINUX FFI
// ============================================================================

#[cfg(target_os = "linux")]
pub mod linux {
    use super::*;

    // ptrace constants
    pub const PTRACE_TRACEME: c_int = 0;
    pub const PTRACE_DETACH: c_int = 17;

    extern "C" {
        pub fn ptrace(request: c_int, pid: i32, addr: *mut c_void, data: *mut c_void) -> c_long;
    }

    /// Check if being traced via ptrace
    pub fn is_traced() -> bool {
        unsafe {
            // Try to trace ourselves - fails if already being traced
            let result = ptrace(
                PTRACE_TRACEME,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if result == -1 {
                return true; // Already being traced
            }
            // Detach from self
            ptrace(PTRACE_DETACH, 0, std::ptr::null_mut(), std::ptr::null_mut());
            false
        }
    }

    /// Check TracerPid in /proc/self/status
    pub fn check_tracer_pid() -> bool {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("TracerPid:") {
                    if let Some(pid_str) = line.split_whitespace().nth(1) {
                        if let Ok(pid) = pid_str.parse::<i32>() {
                            return pid != 0;
                        }
                    }
                }
            }
        }
        false
    }

    /// Combined debugger check for Linux
    pub fn debugger_present() -> bool {
        is_traced() || check_tracer_pid()
    }
}

// ============================================================================
// WINDOWS FFI
// ============================================================================

#[cfg(target_os = "windows")]
pub mod windows {
    use super::*;

    #[link(name = "kernel32")]
    extern "system" {
        pub fn IsDebuggerPresent() -> c_int;
        pub fn CheckRemoteDebuggerPresent(
            hProcess: *mut c_void,
            pbDebuggerPresent: *mut c_int,
        ) -> c_int;
        pub fn GetCurrentProcess() -> *mut c_void;
    }

    /// Check if a debugger is attached (local or remote)
    pub fn debugger_present() -> bool {
        unsafe {
            // Check local debugger
            if IsDebuggerPresent() != 0 {
                return true;
            }

            // Check remote debugger
            let mut is_remote: c_int = 0;
            let process = GetCurrentProcess();
            if CheckRemoteDebuggerPresent(process, &mut is_remote) != 0 {
                if is_remote != 0 {
                    return true;
                }
            }

            false
        }
    }
}

// ============================================================================
// MACOS FFI
// ============================================================================

#[cfg(target_os = "macos")]
pub mod macos {
    use super::*;

    // sysctl MIB for process info
    const CTL_KERN: c_int = 1;
    const KERN_PROC: c_int = 14;
    const KERN_PROC_PID: c_int = 1;

    // Process flags
    const P_TRACED: i32 = 0x00000800;

    #[repr(C)]
    struct KinfoProc {
        // Simplified - only need kp_proc.p_flag
        _padding: [u8; 32],
        p_flag: i32,
        _rest: [u8; 616], // Total struct is ~648 bytes
    }

    extern "C" {
        fn sysctl(
            name: *const c_int,
            namelen: u32,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// Check if being debugged via sysctl
    pub fn debugger_present() -> bool {
        unsafe {
            let pid = std::process::id() as c_int;
            let mib: [c_int; 4] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid];

            let mut info: KinfoProc = std::mem::zeroed();
            let mut size = std::mem::size_of::<KinfoProc>();

            let result = sysctl(
                mib.as_ptr(),
                4,
                &mut info as *mut _ as *mut c_void,
                &mut size,
                std::ptr::null(),
                0,
            );

            if result == 0 {
                return (info.p_flag & P_TRACED) != 0;
            }

            false
        }
    }
}

// ============================================================================
// CROSS-PLATFORM API
// ============================================================================

/// Check if a debugger is present (any platform)
pub fn is_debugger_present() -> bool {
    #[cfg(target_os = "linux")]
    {
        linux::debugger_present()
    }

    #[cfg(target_os = "windows")]
    {
        windows::debugger_present()
    }

    #[cfg(target_os = "macos")]
    {
        macos::debugger_present()
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false // Unknown platform - assume no debugger
    }
}

/// Get current platform name
pub fn platform_name() -> &'static str {
    #[cfg(target_os = "linux")]
    {
        "linux"
    }
    #[cfg(target_os = "windows")]
    {
        "windows"
    }
    #[cfg(target_os = "macos")]
    {
        "macos"
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        "unknown"
    }
}

// ============================================================================
// MIND LANGUAGE BINDINGS
// ============================================================================

/// Expose to Mind language as `sys.is_debugger_present()`
#[no_mangle]
pub extern "C" fn mind_sys_is_debugger_present() -> c_int {
    if is_debugger_present() {
        1
    } else {
        0
    }
}

/// Expose to Mind language as `sys.platform()`
#[no_mangle]
pub extern "C" fn mind_sys_platform() -> *const u8 {
    platform_name().as_ptr()
}

/// Timing-based debugger detection
/// Returns 1 if timing anomaly detected (likely debugger stepping)
#[no_mangle]
pub extern "C" fn mind_sys_timing_check() -> c_int {
    let start = std::time::Instant::now();

    // Known-duration operation
    let mut x: u64 = 0;
    for i in 0..10000u64 {
        x = x.wrapping_add(i);
    }
    std::hint::black_box(x);

    let elapsed = start.elapsed();

    // More than 50ms = suspicious
    if elapsed.as_millis() > 50 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_check_runs() {
        // Should not panic, may return true or false depending on test environment
        let _ = is_debugger_present();
    }

    #[test]
    fn test_timing_check() {
        // In normal execution, should return 0 (no anomaly)
        assert_eq!(mind_sys_timing_check(), 0);
    }
}
