// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! `mlir-opt | mlir-translate`, run concurrently instead of one after the other.
//!
//! `mlir-translate` is spawned first, so its process startup (dynamic linking
//! and dialect registration) overlaps `mlir-opt`'s run instead of following it.
//! A tee thread copies `mlir-opt`'s stdout into a buffer (which becomes
//! `optimized_mlir`) and into `mlir-translate`'s stdin.
//!
//! Byte-identity: the serial path hands `mlir-translate` the TRIMMED decoded
//! text of `mlir-opt`'s stdout (`decode_to_string`). The tee forwards exactly
//! the ASCII-whitespace-trimmed byte range of the raw stream, and after both
//! children finish that forwarded range is compared with `optimized_mlir`. If
//! they differ (non-UTF-8 output, a non-ASCII whitespace edge, or a forwarding
//! write that failed) the concurrent result is discarded and `mlir-translate`
//! is re-run serially on `optimized_mlir`. So `llvm_ir` is always what the
//! serial path produces for the same `optimized_mlir`; the fallback only costs
//! time, never bytes.
//!
//! Error attribution matches the serial path: an `mlir-opt` failure, timeout
//! or spawn error is reported as `mlir-opt` (with the bufferize hint) and the
//! concurrent `mlir-translate` is killed; only after `mlir-opt` succeeds is
//! `mlir-translate`'s own outcome surfaced.

use std::io::{Read, Write};
use std::ops::Range;
use std::process::{ChildStdin, ChildStdout, Stdio};
use std::thread::JoinHandle;
use std::time::Instant;

use super::{
    BuildError, BuildTools, decode_to_string, drain_pipe, join_drain, kill_and_reap, run_mlir_opt,
    run_mlir_translate, spawn_tool, wait_with_timeout,
};

/// What the opt→translate stage produced. `llvm_ir` is kept as a `Result` so
/// the caller can write `optimized_mlir` before surfacing a translate failure,
/// in the same order as the serial path.
pub(super) struct OptTranslateOutput {
    pub(super) optimized_mlir: String,
    pub(super) llvm_ir: Result<String, BuildError>,
    /// True when `llvm_ir` came from the concurrently-run `mlir-translate`,
    /// false when it came from a serial run (spawn failure or fallback).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) pipelined: bool,
}

/// Bytes `mlir-opt` wrote, and which of them reached `mlir-translate`.
struct TeeResult {
    raw: Vec<u8>,
    forwarded: Range<usize>,
    write_ok: bool,
}

/// Run `mlir-opt --pass-pipeline=builtin.module(<pipeline>)` on `input` with
/// its output streamed into a concurrently running `mlir-translate`.
///
/// Timeouts: `mlir-opt` gets `tools.timeout` from the end of its stdin write
/// (as in `run_command`); `mlir-translate` gets `tools.timeout` from the moment
/// `mlir-opt` exits, i.e. never less time with its complete input than the
/// serial path gives it.
pub(super) fn run_mlir_opt_into_translate(
    input: &str,
    pipeline: &str,
    tools: &BuildTools,
) -> Result<OptTranslateOutput, BuildError> {
    let translate_args = vec![String::from("--mlir-to-llvmir")];
    let mut translate = match spawn_tool(
        &tools.mlir_translate,
        &translate_args,
        Stdio::piped(),
        "mlir-translate",
    ) {
        Ok(child) => child,
        // Serial path: an mlir-opt failure outranks a missing mlir-translate.
        Err(_) => return serial(input, pipeline, tools),
    };
    let translate_stdout = drain_pipe(translate.stdout.take());
    let translate_stderr = drain_pipe(translate.stderr.take());
    let Some(translate_stdin) = translate.stdin.take() else {
        kill_and_reap(&mut translate);
        return serial(input, pipeline, tools);
    };

    let opt_args = vec![format!("--pass-pipeline=builtin.module({pipeline})")];
    let mut opt = match spawn_tool(&tools.mlir_opt, &opt_args, Stdio::piped(), "mlir-opt") {
        Ok(child) => child,
        Err(err) => {
            kill_and_reap(&mut translate);
            return Err(err);
        }
    };
    let opt_stderr = drain_pipe(opt.stderr.take());
    let tee = opt
        .stdout
        .take()
        .map(|out| tee_thread(out, translate_stdin));

    if let Some(mut stdin) = opt.stdin.take() {
        if let Err(err) = stdin.write_all(input.as_bytes()) {
            kill_and_reap(&mut opt);
            kill_and_reap(&mut translate);
            return Err(BuildError::Io(err));
        }
        // Dropping stdin signals EOF to mlir-opt.
    }

    let opt_status = match wait_with_timeout(&mut opt, Instant::now(), tools.timeout, "mlir-opt") {
        Ok(status) => status,
        Err(err) => {
            kill_and_reap(&mut translate);
            return Err(err);
        }
    };
    if !opt_status.success() {
        kill_and_reap(&mut translate);
        let stderr = decode_to_string(&join_drain(opt_stderr));
        return Err(BuildError::Subprocess {
            tool: "mlir-opt",
            stderr: crate::diagnostics::toolchain::with_bufferize_hint(&tools.mlir_opt, stderr),
        });
    }

    // mlir-opt succeeded: wait for mlir-translate BEFORE joining the tee, so a
    // translate that stops reading is still bounded by the timeout (the kill
    // unblocks the tee's write).
    let translate_status = wait_with_timeout(
        &mut translate,
        Instant::now(),
        tools.timeout,
        "mlir-translate",
    );
    let tee = join_tee(tee);
    let optimized_mlir = decode_to_string(&tee.raw);

    let status = match translate_status {
        Ok(status) => status,
        // Timed out (or could not be polled) with its input available: the
        // serial path would hit the same wall, so surface it as translate's.
        Err(err) => {
            return Ok(OptTranslateOutput {
                optimized_mlir,
                llvm_ir: Err(err),
                pipelined: true,
            });
        }
    };

    let fed_identical =
        tee.write_ok && tee.raw.get(tee.forwarded.clone()) == Some(optimized_mlir.as_bytes());
    if !fed_identical {
        let llvm_ir = run_mlir_translate(&optimized_mlir, tools);
        return Ok(OptTranslateOutput {
            optimized_mlir,
            llvm_ir,
            pipelined: false,
        });
    }

    let stdout = join_drain(translate_stdout);
    let stderr = join_drain(translate_stderr);
    let llvm_ir = if status.success() {
        Ok(decode_to_string(&stdout))
    } else {
        Err(BuildError::Subprocess {
            tool: "mlir-translate",
            stderr: decode_to_string(&stderr),
        })
    };
    Ok(OptTranslateOutput {
        optimized_mlir,
        llvm_ir,
        pipelined: true,
    })
}

/// The serial opt-then-translate path, used when mlir-translate cannot be
/// spawned up front.
fn serial(
    input: &str,
    pipeline: &str,
    tools: &BuildTools,
) -> Result<OptTranslateOutput, BuildError> {
    let optimized_mlir = run_mlir_opt(input, pipeline, tools)?;
    let llvm_ir = run_mlir_translate(&optimized_mlir, tools);
    Ok(OptTranslateOutput {
        optimized_mlir,
        llvm_ir,
        pipelined: false,
    })
}

fn tee_thread(out: ChildStdout, sink: ChildStdin) -> JoinHandle<TeeResult> {
    std::thread::spawn(move || tee_trimmed(out, sink))
}

fn join_tee(tee: Option<JoinHandle<TeeResult>>) -> TeeResult {
    tee.and_then(|t| t.join().ok()).unwrap_or(TeeResult {
        raw: Vec::new(),
        forwarded: 0..0,
        // No record of what was forwarded: force the serial fallback.
        write_ok: false,
    })
}

/// The whitespace `str::trim` strips, restricted to ASCII. A non-ASCII
/// whitespace edge makes `forwarded` differ from the trimmed text, which the
/// caller detects and answers with the serial fallback.
fn is_trim_ascii(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

/// Copy `src` to EOF into a buffer, forwarding to `sink` exactly the bytes
/// between the first and last non-whitespace byte. Trailing whitespace is held
/// back until a later non-whitespace byte proves it interior, so the stream
/// can be forwarded as it arrives. After a failed write the rest of `src` is
/// still drained (so mlir-opt never blocks) but no longer forwarded.
fn tee_trimmed<R: Read, W: Write>(mut src: R, sink: W) -> TeeResult {
    let mut raw = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    let mut sink = Some(sink);
    let mut write_ok = true;
    let mut start: Option<usize> = None;
    let mut sent = 0usize;
    loop {
        let n = match src.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        let base = raw.len();
        raw.extend_from_slice(&chunk[..n]);
        if start.is_none() {
            if let Some(i) = raw[base..].iter().position(|&b| !is_trim_ascii(b)) {
                start = Some(base + i);
                sent = base + i;
            } else {
                continue;
            }
        }
        if let Some(last) = raw[sent..].iter().rposition(|&b| !is_trim_ascii(b)) {
            let end = sent + last + 1;
            if let Some(w) = sink.as_mut() {
                if w.write_all(&raw[sent..end]).is_err() {
                    write_ok = false;
                    sink = None;
                }
            }
            sent = end;
        }
    }
    drop(sink);
    let forwarded = match start {
        Some(s) => s..sent,
        None => 0..0,
    };
    TeeResult {
        raw,
        forwarded,
        write_ok,
    }
}

#[cfg(test)]
pub(super) fn tee_trimmed_for_test(chunks: &[&[u8]]) -> (Vec<u8>, Vec<u8>) {
    struct Chunked<'a> {
        chunks: std::slice::Iter<'a, &'a [u8]>,
    }
    impl Read for Chunked<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            match self.chunks.next() {
                Some(c) => {
                    buf[..c.len()].copy_from_slice(c);
                    Ok(c.len())
                }
                None => Ok(0),
            }
        }
    }
    let mut sink = Vec::new();
    let res = tee_trimmed(
        Chunked {
            chunks: chunks.iter(),
        },
        &mut sink,
    );
    assert!(res.write_ok);
    assert_eq!(&res.raw[res.forwarded.clone()], sink.as_slice());
    (res.raw, sink)
}
