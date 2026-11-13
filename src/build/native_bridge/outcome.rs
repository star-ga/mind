// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The verdict of one supervised compiler run: the artifact bytes, or the
//! printed outcome and the exit code. Split out of `native_bridge.rs` so the
//! bridge stays under the module-size ceiling; the wording is unchanged.

use super::child;

/// Turn a supervised compiler run into the artifact bytes, or print the real
/// outcome and return the exit code. Fail-closed: no artifact on any failure,
/// and never an MLIR fallback.
///
/// Two kinds of failure, told apart because they mean different things:
/// * the compiler read the whole image and exited nonzero -- it REJECTED the
///   PROGRAM; that verdict keeps its own uncoded wording and exit status;
/// * anything else -- a pipe to the child failed, it ended by a signal, or it
///   could not be reaped -- is a failure of the compiler PROCESS, coded
///   [`child::CHILD_FAILURE_CODE`] and reported with the reaped outcome (exit
///   status, or signal name and number) and the head and tail of its stderr.
pub(super) fn finish_compiler_run(run: child::ChildRun) -> Result<Vec<u8>, i32> {
    let ok_status = matches!(&run.reaped, child::Reaped::Exited(s) if s.success());
    if run.failure.is_none() && ok_status {
        // Both pipes reached EOF, so stdout is complete.
        return Ok(run.stdout.unwrap_or_default());
    }
    let code = match &run.reaped {
        child::Reaped::Exited(s) | child::Reaped::KilledAfterGrace { status: s, .. } => {
            // A stream failure is the bridge's own error when the child itself
            // reported success: its input was incomplete.
            child::exit_code_for(s, if run.failure.is_some() { 2 } else { 1 })
        }
        child::Reaped::WaitFailed(_) => 2,
    };
    let rejected_program = run.failure.is_none()
        && matches!(&run.reaped, child::Reaped::Exited(s) if s.code().is_some());
    if rejected_program {
        print_compiler_stderr(run.stderr.as_ref(), false);
        eprintln!(
            "error[backend-native]: pure-MIND compiler rejected the program (exit {code}); \
             no artifact written (native backend is fail-closed — no MLIR fallback)."
        );
        return Err(code);
    }
    let coded = format!("error[backend-native][{}]", child::CHILD_FAILURE_CODE);
    match &run.failure {
        Some(failure) => {
            eprintln!("{coded}: {failure}");
            eprintln!(
                "  compiler process: {}",
                child::describe_reaped(&run.reaped)
            );
        }
        None => eprintln!(
            "{coded}: pure-MIND compiler {}",
            child::describe_reaped(&run.reaped)
        ),
    }
    print_compiler_stderr(run.stderr.as_ref(), true);
    eprintln!(
        "error[backend-native]: no artifact written (native backend is fail-closed — \
         no MLIR fallback)."
    );
    Err(code)
}

/// Re-emit the child's captured stderr -- its head, then its tail -- stating
/// the elided middle or unavailability.
fn print_compiler_stderr(capture: Option<&child::BoundedCapture>, after_failure: bool) {
    use std::io::Write as _;
    let Some(cap) = capture else {
        eprintln!(
            "  compiler stderr: unavailable (its pipe was still held open after the \
             process was reaped)"
        );
        return;
    };
    if cap.is_empty() {
        if after_failure {
            eprintln!("  compiler stderr: (empty)");
        }
        return;
    }
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(&cap.head);
    if cap.dropped > 0 {
        if !cap.head.ends_with(b"\n") {
            let _ = writeln!(err);
        }
        let _ = writeln!(
            err,
            "  [{} bytes of compiler stderr truncated between the first {} and the last {}]",
            cap.dropped,
            cap.head.len(),
            cap.tail.len()
        );
    }
    let _ = err.write_all(&cap.tail);
    if cap.last_byte() != Some(b'\n') {
        let _ = writeln!(err);
    }
}
