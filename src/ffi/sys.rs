// Copyright 2025-2026 STARGA Inc.
// System-level FFI: debugger detection and platform identification.
//
// Scope note: this module is a small, side-effect-free reporting surface only.
// It is deliberately NOT an anti-tamper mechanism -- the protection subsystem
// (VM bytecode interpreter, encrypted strings, redundant state, watchdog) lives
// in the runtime, not here. Everything below must be safe to call from any
// process at any time, including under a test harness or a CI runner.
//
// deferred: this module currently has no in-repo consumers (the `mind_sys_*`
// symbols are exported from the cdylib for embedders, and nothing in this
// repository calls them). It is compiled only under the `ffi-c` feature, which
// `full` enables. If it is still unused when the embedder API is finalised,
// delete it rather than growing it -- upgrade path: either wire it into the
// documented embedder surface (docs/rfcs/0010-memory-safety-and-c-abi.md) with
// tests that a consumer actually exercises, or remove the module and its
// `pub mod sys;` line in src/ffi/mod.rs.

use std::os::raw::{c_char, c_int};

// ============================================================================
// LINUX
// ============================================================================

#[cfg(target_os = "linux")]
pub mod linux {
    /// Read `TracerPid` from `/proc/self/status`.
    ///
    /// This is the whole Linux check, on purpose. A previous revision first
    /// called `ptrace(PTRACE_TRACEME)` on the assumption that the call fails
    /// when a debugger is already attached. That was wrong twice over:
    ///
    ///  1. On an untraced process `PTRACE_TRACEME` SUCCEEDS, so the check
    ///     reported "no debugger" and detected nothing that the `TracerPid`
    ///     read below does not already detect.
    ///  2. It made the caller a tracee of its own parent, permanently: the
    ///     paired `ptrace(PTRACE_DETACH, 0, ...)` fails with ESRCH (pid 0 is
    ///     not a valid ptrace target and a tracee cannot detach itself), so
    ///     the process stayed traced. The next `execve` then stopped with
    ///     SIGTRAP and never resumed, and a test binary that did this exited
    ///     into a zombie its parent could not reap -- `cargo test` printed a
    ///     green result and then hung forever.
    ///
    /// Do not reintroduce a self-`PTRACE_TRACEME` probe here. It buys no
    ///  detection and it corrupts the process state of every caller.
    /// `debugger_check_leaves_ptrace_state_untouched` below is the guard.
    pub fn check_tracer_pid() -> bool {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("TracerPid:") {
                    if let Ok(pid) = rest.trim().parse::<i32>() {
                        return pid != 0;
                    }
                }
            }
        }
        false
    }

    /// Combined debugger check for Linux.
    pub fn debugger_present() -> bool {
        check_tracer_pid()
    }
}

// ============================================================================
// WINDOWS
// ============================================================================

#[cfg(target_os = "windows")]
pub mod windows {
    use std::os::raw::{c_int, c_void};

    // `unsafe extern` is required from edition 2024 on; a bare `extern "system"`
    // block does not compile ("extern blocks must be unsafe") and this leg was
    // never built, so the error sat here undetected.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn IsDebuggerPresent() -> c_int;
        pub fn CheckRemoteDebuggerPresent(
            hProcess: *mut c_void,
            pbDebuggerPresent: *mut c_int,
        ) -> c_int;
        pub fn GetCurrentProcess() -> *mut c_void;
    }

    /// Check if a debugger is attached (local or remote).
    pub fn debugger_present() -> bool {
        unsafe {
            if IsDebuggerPresent() != 0 {
                return true;
            }

            let mut is_remote: c_int = 0;
            let process = GetCurrentProcess();
            if CheckRemoteDebuggerPresent(process, &mut is_remote) != 0 && is_remote != 0 {
                return true;
            }

            false
        }
    }
}

// ============================================================================
// MACOS
// ============================================================================

#[cfg(target_os = "macos")]
pub mod macos {
    use std::os::raw::{c_int, c_void};

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

    unsafe extern "C" {
        fn sysctl(
            name: *const c_int,
            namelen: u32,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// Check if being debugged via sysctl.
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

/// Check if a debugger is present (any platform).
///
/// Observational only: calling this never changes the state of the process.
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

/// Current platform name, as a NUL-terminated C string.
///
/// This is the authority for [`platform_name`] and for `mind_sys_platform`.
/// It must stay a `CStr`: handing a C caller `str::as_ptr()` hands it a
/// pointer with no terminator, and the caller's `strlen` then runs off the end
/// of the literal into whatever `.rodata` the linker happened to place next.
pub fn platform_cstr() -> &'static std::ffi::CStr {
    #[cfg(target_os = "linux")]
    {
        c"linux"
    }
    #[cfg(target_os = "windows")]
    {
        c"windows"
    }
    #[cfg(target_os = "macos")]
    {
        c"macos"
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        c"unknown"
    }
}

/// Get current platform name.
pub fn platform_name() -> &'static str {
    // Every arm of `platform_cstr` is ASCII, so this cannot fail.
    match platform_cstr().to_str() {
        Ok(s) => s,
        Err(_) => "unknown",
    }
}

// ============================================================================
// MIND LANGUAGE BINDINGS
// ============================================================================

/// Expose to Mind language as `sys.is_debugger_present()`.
#[unsafe(no_mangle)]
pub extern "C" fn mind_sys_is_debugger_present() -> c_int {
    if is_debugger_present() { 1 } else { 0 }
}

/// Expose to Mind language as `sys.platform()`.
///
/// Returns a static, NUL-terminated string owned by this library. The caller
/// must not free it.
#[unsafe(no_mangle)]
pub extern "C" fn mind_sys_platform() -> *const c_char {
    platform_cstr().as_ptr()
}

/// Timing-based debugger detection.
/// Returns 1 if timing anomaly detected (likely debugger stepping).
///
/// deferred: this is a coarse wall-clock heuristic, not a reliable signal -- a
/// loaded or preempted machine can exceed the threshold with no debugger
/// attached, and any debugger that does not single-step this exact loop will
/// not trip it. Kept only because it is side-effect free. Upgrade path: drop it
/// in favour of the runtime protection subsystem's checks, or delete it with
/// the rest of the module (see the module-level `deferred:` note).
#[unsafe(no_mangle)]
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
    if elapsed.as_millis() > 50 { 1 } else { 0 }
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

    /// Regression guard for the self-`PTRACE_TRACEME` defect.
    ///
    /// The old `is_traced()` left the process permanently traced by its parent,
    /// which wedged the very next `execve` with SIGTRAP and turned this test
    /// binary into an unreapable zombie -- `cargo test` reported success and
    /// then hung. `TracerPid` is the observable that moved (0 -> parent pid),
    /// so pin it: the check must not change it.
    #[cfg(target_os = "linux")]
    #[test]
    fn debugger_check_leaves_ptrace_state_untouched() {
        // `/proc/thread-self`, NOT `/proc/self`. `/proc/self/status` is the
        // THREAD-GROUP (TGID) view, but `PTRACE_TRACEME` binds the tracer to the
        // CALLING THREAD's TID — and libtest runs every `#[test]` on a spawned
        // thread, not on the main thread. Reading the TGID view here would let a
        // reintroduced self-TRACEME probe pass this guard unnoticed: the thread
        // would be traced, the process-level TracerPid would still read 0, the
        // assert would pass, and the hang this test exists to prevent would ship.
        // `/proc/thread-self` is the calling thread's own directory (Linux 3.17+),
        // so it observes exactly the state PTRACE_TRACEME actually changes.
        fn tracer_pid() -> i32 {
            let status = std::fs::read_to_string("/proc/thread-self/status")
                .expect("/proc/thread-self/status must be readable");
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("TracerPid:") {
                    return rest
                        .trim()
                        .parse::<i32>()
                        .expect("TracerPid must be an int");
                }
            }
            panic!("no TracerPid line in /proc/thread-self/status");
        }

        let before = tracer_pid();
        for _ in 0..8 {
            let _ = is_debugger_present();
            let _ = mind_sys_is_debugger_present();
        }
        let after = tracer_pid();

        assert_eq!(
            before, after,
            "debugger check changed the process ptrace state (TracerPid {before} -> {after}); \
             a self-PTRACE_TRACEME probe has been reintroduced"
        );
    }

    /// `mind_sys_platform` hands its pointer straight to C, so the byte after
    /// the last character must be NUL. Returning `str::as_ptr()` here caused a
    /// 204-byte over-read of neighbouring `.rodata` in the built cdylib.
    #[test]
    fn platform_string_is_nul_terminated_for_c_callers() {
        let name = platform_name();
        let ptr = mind_sys_platform();
        assert!(!ptr.is_null());

        // Walk exactly as a C `strlen` would, with a hard bound so a missing
        // terminator fails the test instead of running away.
        let mut len = 0usize;
        while len < 4096 {
            // SAFETY: the contract under test is that these bytes are readable
            // up to and including a NUL within the returned static string.
            let b = unsafe { *ptr.add(len) };
            if b == 0 {
                break;
            }
            len += 1;
        }

        assert_eq!(
            len,
            name.len(),
            "C strlen({len}) disagrees with platform_name().len() ({}) - the returned \
             pointer is not NUL-terminated and a C caller over-reads past the literal",
            name.len()
        );

        // SAFETY: `len` bytes before the NUL were just validated as readable.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        assert_eq!(
            bytes,
            name.as_bytes(),
            "C string content must match platform_name()"
        );
    }
}
