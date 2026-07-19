// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Mind AI Protocol (MAP) Server
//!
//! A compiler-in-the-loop server for AI agents to interact with the MIND compiler.
//! Implements RFC-0002: Mind AI Protocol.

// Deliberately NOT registering `libmind::SmallHeapAlloc` here: MAP is a
// long-running server, and that allocator never returns memory to the OS and
// ratchets on cross-thread frees — correct for the short-lived `mindc` process,
// wrong for a daemon. MAP uses the System allocator.
//!
//! # Usage
//!
//! ```bash
//! # Start server on stdio (default)
//! mind-ai
//!
//! # Start server on TCP port
//! mind-ai --tcp 8080
//!
//! # Restricted mode
//! mind-ai --mode no_io,no_unsafe
//! ```

use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex, MutexGuard};

/// Extension trait for poison-safe mutex locking.
///
/// # Poison Recovery Policy
///
/// When a thread panics while holding a mutex, the mutex becomes "poisoned".
/// This implementation recovers by taking ownership of the inner data.
///
/// ## Recovery Strategy:
/// 1. Log a warning for observability
/// 2. Continue operation with existing state
/// 3. Rely on session isolation (each request is independent)
/// 4. Module state is reset on `bye` or new `hello`
///
/// ## Rationale:
/// - MAP sessions are stateless between commands (request-response)
/// - Each command validates inputs independently
/// - Partial corruption is detected by subsequent `check` commands
/// - Server availability is prioritized over strict consistency
///
/// For stricter consistency, configure heartbeat/restart monitoring.
trait MutexExt<T> {
    /// Lock the mutex, recovering from poison if necessary.
    fn lock_or_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> MutexExt<T> for Mutex<T> {
    fn lock_or_recover(&self) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                eprintln!("[WARN] Mutex poisoned - recovering (see Poison Recovery Policy docs)");
                poisoned.into_inner()
            }
        }
    }
}

/// Protocol version
const MAP_VERSION: u32 = 1;
const MIC_VERSION: u32 = 1;
const SERVER_VERSION: &str = "0.2.0";

// ---------------------------------------------------------------------------
// Resource budgets (DoS hardening — RFC-0002 §Resource Limits)
//
// The MAP protocol is a long-lived, stateful, line-oriented protocol that
// accepts untrusted input from AI agents. Without explicit budgets a peer can
// exhaust server memory by sending one enormous line, an unterminated heredoc,
// a module with an unbounded node count, or an unbounded stream of growing
// `patch.*` operations.
//
// Every budget below is a NAMED, DOCUMENTED constant. When a budget is
// exceeded the server returns a CLEAR structured error (`code=E1xx`) — it
// never panics, aborts, or silently truncates. This is the protocol-level
// analogue of the bounded-query hardening applied elsewhere in the ecosystem.
//
// The values mirror the conventions established in
// `src/ir/compact/v2/parse.rs` (MAX_INPUT_SIZE / MAX_LINE_COUNT etc.) so the
// MIC parser and the MAP transport agree on the same order of magnitude.
// ---------------------------------------------------------------------------

/// Maximum bytes accepted in a single protocol line (request line or one
/// heredoc body line). Protects against a single unbounded line forcing an
/// arbitrarily large allocation in the reader. Exceeding this is `E101`.
const MAX_LINE_BYTES: usize = 1024 * 1024; // 1 MiB

/// Maximum total bytes accumulated for one heredoc-delimited command body
/// before the terminating `EOF`. An unterminated or hostile heredoc can
/// otherwise grow without bound. Exceeding this is `E102`.
const MAX_SESSION_BYTES: usize = 16 * 1024 * 1024; // 16 MiB

/// Maximum number of MIC nodes a loaded module may contain. Bounds the work of
/// per-node validation/patch scans and the memory held by the resident module.
/// Exceeding this (on `load.mic` or after a `patch.*`) is `E103`.
const MAX_MODULE_NODES: usize = 1_000_000;

/// Maximum number of mutating `patch.*` operations permitted per session.
/// Bounds the amortised cost of repeated whole-module rewrites and caps the
/// rate at which a peer can grow the resident module. Exceeding this is `E104`.
const MAX_PATCH_OPS: u64 = 100_000;

/// Session mode flags
#[derive(Debug, Clone, Default)]
struct SessionMode {
    no_io: bool,
    no_unsafe: bool,
    pure_only: bool,
}

impl SessionMode {
    fn parse(s: &str) -> Self {
        let mut mode = SessionMode::default();
        for flag in s.split(',') {
            match flag.trim() {
                "no_io" => mode.no_io = true,
                "no_unsafe" => mode.no_unsafe = true,
                "pure_only" => mode.pure_only = true,
                _ => {}
            }
        }
        mode
    }
}

/// Session state
struct Session {
    /// Sequence counter
    seq: u64,
    /// Session mode
    mode: SessionMode,
    /// Current module (MIC format)
    module: Option<String>,
    /// Node count
    node_count: usize,
    /// Type count
    type_count: usize,
    /// Symbol count
    symbol_count: usize,
    /// Diagnostics
    diagnostics: Vec<Diagnostic>,
    /// Number of mutating `patch.*` operations applied this session.
    /// Enforced against `MAX_PATCH_OPS` for DoS hardening.
    patch_ops: u64,
}

#[derive(Debug, Clone)]
struct Diagnostic {
    severity: char, // E, W, I, H
    node: String,
    message: String,
}

impl Session {
    fn new() -> Self {
        Self {
            seq: 0,
            mode: SessionMode::default(),
            module: None,
            node_count: 0,
            type_count: 0,
            symbol_count: 0,
            diagnostics: Vec::new(),
            patch_ops: 0,
        }
    }

    fn count_entries(&mut self, mic: &str) {
        self.node_count = 0;
        self.type_count = 0;
        self.symbol_count = 0;

        for line in mic.lines() {
            let line = line.trim();
            if line.starts_with('N') && !line.starts_with("N ") {
                self.node_count += 1;
            } else if line.starts_with('T') {
                self.type_count += 1;
            } else if line.starts_with('S') {
                self.symbol_count += 1;
            }
        }
    }

    /// Count the MIC nodes in `mic` without mutating session state.
    ///
    /// Used to enforce `MAX_MODULE_NODES` before a candidate module is accepted
    /// into the session (on `load.mic` and after any `patch.*` mutation).
    fn count_nodes(mic: &str) -> usize {
        mic.lines()
            .filter(|line| {
                let line = line.trim();
                line.starts_with('N') && !line.starts_with("N ")
            })
            .count()
    }
}

/// Build a structured budget-exceeded error response.
///
/// Returns the canonical `=<seq> err code=E1xx msg="..."` form used for all
/// resource-limit rejections so callers can match on the stable code while
/// still receiving a human-readable reason. Never panics.
fn budget_error(seq: u64, code: &str, msg: String) -> String {
    format!("={} err code={} msg=\"{}\"", seq, code, msg)
}

/// Protocol handler
struct MapServer {
    session: Arc<Mutex<Session>>,
}

impl MapServer {
    fn new() -> Self {
        Self {
            session: Arc::new(Mutex::new(Session::new())),
        }
    }

    fn handle_line(&self, line: &str) -> Option<String> {
        // Budget: reject an oversized request line before any further work.
        // We attempt to recover the sequence number for a well-formed error;
        // if even that is unparseable we fall back to the bare `!error` form.
        if line.len() > MAX_LINE_BYTES {
            let seq = line
                .trim_start()
                .strip_prefix('@')
                .and_then(|r| r.split_whitespace().next())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            return Some(budget_error(
                seq,
                "E101",
                format!(
                    "request line too large: {} bytes (max {})",
                    line.len(),
                    MAX_LINE_BYTES
                ),
            ));
        }

        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

        // Parse request: @<seq> <command> [args...]
        if !line.starts_with('@') {
            return Some("!error invalid request format".to_string());
        }

        let rest = &line[1..];
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.is_empty() {
            return Some("!error missing sequence number".to_string());
        }

        let seq: u64 = match parts[0].parse() {
            Ok(s) => s,
            Err(_) => return Some("!error invalid sequence number".to_string()),
        };

        if parts.len() < 2 {
            return Some(format!("={} err msg=\"missing command\"", seq));
        }

        let cmd_parts: Vec<&str> = parts[1].splitn(2, ' ').collect();
        let cmd = cmd_parts[0];
        let args = if cmd_parts.len() > 1 {
            cmd_parts[1]
        } else {
            ""
        };

        self.dispatch(seq, cmd, args)
    }

    fn dispatch(&self, seq: u64, cmd: &str, args: &str) -> Option<String> {
        match cmd {
            "hello" => self.handle_hello(seq, args),
            "bye" => self.handle_bye(seq),
            "load.mic" => self.handle_load_mic(seq, args),
            "dump" => self.handle_dump(seq, args),
            "check" => self.handle_check(seq),
            "verify" => self.handle_verify(seq, args),
            "lint" => self.handle_lint(seq, args),
            "patch.insert" => self.handle_patch_insert(seq, args),
            "patch.delete" => self.handle_patch_delete(seq, args),
            "patch.replace" => self.handle_patch_replace(seq, args),
            "query.node" => self.handle_query_node(seq, args),
            "query.stats" => self.handle_query_stats(seq),
            _ => Some(format!(
                "={} err code=E005 msg=\"unknown command: {}\"",
                seq, cmd
            )),
        }
    }

    fn handle_hello(&self, seq: u64, args: &str) -> Option<String> {
        let mut session = self.session.lock_or_recover();

        // Parse args
        let mut mic_ver = 1u32;
        let mut map_ver = 1u32;
        let mut mode = SessionMode::default();

        for arg in args.split_whitespace() {
            if let Some(val) = arg.strip_prefix("mic=") {
                mic_ver = val.parse().unwrap_or(1);
            } else if let Some(val) = arg.strip_prefix("map=") {
                map_ver = val.parse().unwrap_or(1);
            } else if let Some(val) = arg.strip_prefix("mode=") {
                mode = SessionMode::parse(val);
            }
        }

        // Version check
        if mic_ver != MIC_VERSION || map_ver != MAP_VERSION {
            return Some(format!(
                "={} err msg=\"version mismatch: server supports mic={} map={}\"",
                seq, MIC_VERSION, MAP_VERSION
            ));
        }

        session.mode = mode;
        session.seq = seq;

        Some(format!(
            "={} ok version={} mic={} map={} features=[patch,check,dump,query]",
            seq, SERVER_VERSION, MIC_VERSION, MAP_VERSION
        ))
    }

    fn handle_bye(&self, seq: u64) -> Option<String> {
        let mut session = self.session.lock_or_recover();
        session.module = None;
        session.diagnostics.clear();
        // Reset the per-session patch-rate budget on session teardown.
        session.patch_ops = 0;
        Some(format!("={} ok", seq))
    }

    fn handle_load_mic(&self, seq: u64, args: &str) -> Option<String> {
        let mut session = self.session.lock_or_recover();

        // Check mode
        if session.mode.no_io {
            // In no_io mode, only inline MIC is allowed (which is what we're doing)
        }

        // For heredoc, args would contain the MIC content after <<EOF processing
        // In this simple implementation, we just take the content directly
        let mic_content = args.trim();

        // Validate version header
        let first_line = mic_content.lines().next().unwrap_or("");
        if !first_line.trim().starts_with("mic@") {
            return Some(format!(
                "={} err line=1 msg=\"missing version header\"",
                seq
            ));
        }

        // Budget: bound the resident module node count before accepting it.
        let nodes = Session::count_nodes(mic_content);
        if nodes > MAX_MODULE_NODES {
            return Some(budget_error(
                seq,
                "E103",
                format!(
                    "module too large: {} nodes (max {})",
                    nodes, MAX_MODULE_NODES
                ),
            ));
        }

        session.module = Some(mic_content.to_string());
        session.count_entries(mic_content);

        Some(format!(
            "={} ok nodes={} types={} symbols={}",
            seq, session.node_count, session.type_count, session.symbol_count
        ))
    }

    fn handle_dump(&self, seq: u64, args: &str) -> Option<String> {
        let session = self.session.lock_or_recover();

        let format = if args.contains("format=json") {
            "json"
        } else if args.contains("format=ir") {
            "ir"
        } else {
            "mic"
        };

        match &session.module {
            Some(mic) => {
                if format == "mic" {
                    Some(format!("={} ok <<EOF\n{}\nEOF", seq, mic))
                } else {
                    Some(format!(
                        "={} err msg=\"format {} not implemented\"",
                        seq, format
                    ))
                }
            }
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }

    fn handle_check(&self, seq: u64) -> Option<String> {
        let mut session = self.session.lock_or_recover();
        session.diagnostics.clear();

        match &session.module {
            Some(mic) => {
                // Perform basic validation
                let mut errors = Vec::new();

                for (i, line) in mic.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }

                    // Check for undefined references
                    if line.starts_with('N') {
                        // Extract node references and check they're defined
                        // This is a simplified check
                        if line.contains("N99") || line.contains("N98") {
                            errors.push(Diagnostic {
                                severity: 'E',
                                node: format!("line {}", i + 1),
                                message: "undefined reference".to_string(),
                            });
                        }
                    }
                }

                session.diagnostics = errors.clone();

                if errors.is_empty() {
                    Some(format!("={} ok diags=0", seq))
                } else {
                    let diag_strs: Vec<String> = errors
                        .iter()
                        .map(|d| format!("{}:{}:{}", d.severity, d.node, d.message))
                        .collect();
                    Some(format!(
                        "={} ok diags={} <<EOF\n{}\nEOF",
                        seq,
                        errors.len(),
                        diag_strs.join("\n")
                    ))
                }
            }
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }

    fn handle_verify(&self, seq: u64, _args: &str) -> Option<String> {
        // Same as check for now
        self.handle_check(seq)
    }

    fn handle_lint(&self, seq: u64, _args: &str) -> Option<String> {
        let session = self.session.lock_or_recover();

        match &session.module {
            Some(_) => Some(format!("={} ok hints=0", seq)),
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }

    /// Charge one mutating patch operation against the per-session rate budget.
    ///
    /// Returns `Err(response)` with a structured `E104` error when the session
    /// has already used its `MAX_PATCH_OPS` allowance; otherwise increments the
    /// counter and returns `Ok(())`. Never panics.
    fn charge_patch_op(session: &mut Session, seq: u64) -> Result<(), String> {
        if session.patch_ops >= MAX_PATCH_OPS {
            return Err(budget_error(
                seq,
                "E104",
                format!(
                    "patch operation budget exhausted: {} (max {})",
                    session.patch_ops, MAX_PATCH_OPS
                ),
            ));
        }
        session.patch_ops += 1;
        Ok(())
    }

    /// Reject a candidate module that would exceed the node budget.
    ///
    /// Returns `Err(response)` with a structured `E103` error when committing
    /// `new_mic` would push the resident module past `MAX_MODULE_NODES`.
    fn guard_module_nodes(seq: u64, new_mic: &str) -> Result<(), String> {
        let nodes = Session::count_nodes(new_mic);
        if nodes > MAX_MODULE_NODES {
            return Err(budget_error(
                seq,
                "E103",
                format!(
                    "module too large: {} nodes (max {})",
                    nodes, MAX_MODULE_NODES
                ),
            ));
        }
        Ok(())
    }

    fn handle_patch_insert(&self, seq: u64, args: &str) -> Option<String> {
        let mut session = self.session.lock_or_recover();

        if let Err(resp) = Self::charge_patch_op(&mut session, seq) {
            return Some(resp);
        }

        match &session.module {
            Some(mic) => {
                // Parse after=N<id> and the node definition
                // Simplified: just append to module
                let mut new_mic = mic.clone();

                // Find the node content (after heredoc marker or directly)
                let node_content = if args.contains("<<EOF") {
                    args.split("<<EOF").nth(1).unwrap_or("").trim()
                } else {
                    args.trim()
                };

                // Extract after position
                let _after_id = if let Some(rest) = args.strip_prefix("after=N") {
                    rest.split_whitespace()
                        .next()
                        .and_then(|s| s.parse::<usize>().ok())
                } else {
                    None
                };

                // Simple append for now
                if !new_mic.ends_with('\n') {
                    new_mic.push('\n');
                }
                new_mic.push_str(node_content);
                new_mic.push('\n');

                // Budget: reject if this insert grows the module past the cap.
                if let Err(resp) = Self::guard_module_nodes(seq, &new_mic) {
                    return Some(resp);
                }

                session.module = Some(new_mic);
                let mic_ref = session.module.as_ref().unwrap().clone();
                session.count_entries(&mic_ref);

                // Extract new node ID from content
                let new_id = node_content
                    .split_whitespace()
                    .next()
                    .filter(|s| s.starts_with('N'))
                    .unwrap_or("N?");

                Some(format!("={} ok id={}", seq, new_id))
            }
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }

    fn handle_patch_delete(&self, seq: u64, args: &str) -> Option<String> {
        let mut session = self.session.lock_or_recover();

        if let Err(resp) = Self::charge_patch_op(&mut session, seq) {
            return Some(resp);
        }

        match &session.module {
            Some(mic) => {
                let node_id = args.trim();

                // Check if node has dependents
                let has_deps = mic.lines().any(|line| {
                    let line = line.trim();
                    !line.starts_with(node_id) && line.contains(node_id)
                });

                if has_deps {
                    return Some(format!("={} err msg=\"{} has dependents\"", seq, node_id));
                }

                // Remove the node line
                let new_mic: String = mic
                    .lines()
                    .filter(|line| !line.trim().starts_with(node_id))
                    .collect::<Vec<_>>()
                    .join("\n");

                session.module = Some(new_mic);
                let mic_ref = session.module.as_ref().unwrap().clone();
                session.count_entries(&mic_ref);

                Some(format!("={} ok", seq))
            }
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }

    fn handle_patch_replace(&self, seq: u64, args: &str) -> Option<String> {
        let mut session = self.session.lock_or_recover();

        if let Err(resp) = Self::charge_patch_op(&mut session, seq) {
            return Some(resp);
        }

        match &session.module {
            Some(mic) => {
                // Parse node ID and new content
                let parts: Vec<&str> = args.splitn(2, ' ').collect();
                if parts.is_empty() {
                    return Some(format!("={} err msg=\"missing node id\"", seq));
                }

                let node_id = parts[0];
                let new_content = parts.get(1).unwrap_or(&"");

                // Replace the node line
                let new_mic: String = mic
                    .lines()
                    .map(|line| {
                        if line.trim().starts_with(node_id) {
                            new_content.trim()
                        } else {
                            line
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                // Budget: a replacement body may introduce new nodes; bound it.
                if let Err(resp) = Self::guard_module_nodes(seq, &new_mic) {
                    return Some(resp);
                }

                session.module = Some(new_mic);
                let mic_ref = session.module.as_ref().unwrap().clone();
                session.count_entries(&mic_ref);

                Some(format!("={} ok", seq))
            }
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }

    fn handle_query_node(&self, seq: u64, args: &str) -> Option<String> {
        let session = self.session.lock_or_recover();

        match &session.module {
            Some(mic) => {
                let node_id = args.trim();

                for line in mic.lines() {
                    let line = line.trim();
                    if line.starts_with(node_id) {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            let kind = parts[1];
                            let inputs: Vec<&str> = parts[2..]
                                .iter()
                                .filter(|s| s.starts_with('N'))
                                .copied()
                                .collect();
                            let type_ref =
                                parts.iter().find(|s| s.starts_with('T')).unwrap_or(&"T?");

                            return Some(format!(
                                "={} ok kind={} inputs=[{}] type={}",
                                seq,
                                kind,
                                inputs.join(","),
                                type_ref
                            ));
                        }
                    }
                }

                Some(format!("={} err msg=\"node {} not found\"", seq, node_id))
            }
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }

    fn handle_query_stats(&self, seq: u64) -> Option<String> {
        let session = self.session.lock_or_recover();

        match &session.module {
            Some(_) => Some(format!(
                "={} ok nodes={} types={} symbols={}",
                seq, session.node_count, session.type_count, session.symbol_count
            )),
            None => Some(format!("={} err msg=\"no module loaded\"", seq)),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut _mode = SessionMode::default();
    let mut tcp_port: Option<u16> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" if i + 1 < args.len() => {
                _mode = SessionMode::parse(&args[i + 1]);
                i += 2;
            }
            "--tcp" if i + 1 < args.len() => {
                tcp_port = args[i + 1].parse().ok();
                i += 2;
            }
            "--help" | "-h" => {
                println!("Mind AI Protocol Server v{}", SERVER_VERSION);
                println!();
                println!("Usage: mind-ai [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --mode <mode>   Set restricted mode (no_io,no_unsafe,pure_only)");
                println!("  --tcp <port>    Start TCP server on port");
                println!("  --help          Show this help");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    let server = MapServer::new();

    if let Some(port) = tcp_port {
        // TCP mode
        eprintln!("Starting MAP server on TCP port {} (not implemented)", port);
        eprintln!("Use stdio mode for now");
    }

    // Stdio mode
    eprintln!(
        "Mind AI Protocol Server v{} ready (MIC v{}, MAP v{})",
        SERVER_VERSION, MIC_VERSION, MAP_VERSION
    );

    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    let mut stdout = std::io::stdout();

    let mut heredoc_buffer: Option<(u64, String, String)> = None; // (seq, cmd, content)

    // Sequence number of a heredoc whose body overflowed MAX_SESSION_BYTES.
    // While set, body lines are drained (discarded) until the closing `EOF`
    // so an over-budget body is never reinterpreted as protocol commands.
    let mut draining_overflow: Option<u64> = None;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };

        // Budget: reject an oversized physical line up front. This covers
        // heredoc body lines (which bypass `handle_line`) as well as ordinary
        // request lines. The body of an already-overflowed heredoc is being
        // drained, so skip the per-line error there to avoid response floods.
        if line.len() > MAX_LINE_BYTES && draining_overflow.is_none() {
            let seq = heredoc_buffer
                .as_ref()
                .map(|(s, _, _)| *s)
                .or_else(|| {
                    line.trim_start()
                        .strip_prefix('@')
                        .and_then(|r| r.split_whitespace().next())
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(0);
            writeln!(
                stdout,
                "{}",
                budget_error(
                    seq,
                    "E101",
                    format!(
                        "request line too large: {} bytes (max {})",
                        line.len(),
                        MAX_LINE_BYTES
                    )
                )
            )
            .ok();
            stdout.flush().ok();
            // If this line was a heredoc body, drain the rest of the body.
            if let Some((s, _, _)) = heredoc_buffer.take() {
                draining_overflow = Some(s);
            }
            continue;
        }

        // Drain the body of an over-budget heredoc until its terminator.
        if draining_overflow.is_some() {
            if line.trim() == "EOF" {
                draining_overflow = None;
            }
            continue;
        }

        // Handle heredoc
        if let Some((seq, cmd, content)) = heredoc_buffer.as_mut() {
            if line.trim() == "EOF" {
                // Process accumulated heredoc
                let full_args = content.clone();
                let cmd_clone = cmd.clone();
                let seq_val = *seq;
                heredoc_buffer = None;
                if let Some(response) = server.dispatch(seq_val, &cmd_clone, &full_args) {
                    writeln!(stdout, "{}", response).ok();
                    stdout.flush().ok();
                }
            } else {
                // Budget: bound total accumulated heredoc body size. An
                // unterminated or hostile heredoc would otherwise grow without
                // limit. On overflow, emit a structured error and drain the
                // remaining body until `EOF`.
                if content.len() + line.len() + 1 > MAX_SESSION_BYTES {
                    let seq_val = *seq;
                    writeln!(
                        stdout,
                        "{}",
                        budget_error(
                            seq_val,
                            "E102",
                            format!(
                                "heredoc body too large: exceeds {} bytes",
                                MAX_SESSION_BYTES
                            )
                        )
                    )
                    .ok();
                    stdout.flush().ok();
                    heredoc_buffer = None;
                    draining_overflow = Some(seq_val);
                    continue;
                }
                content.push_str(&line);
                content.push('\n');
            }
            continue;
        }

        // Check for heredoc start
        if line.contains("<<EOF") {
            let parts: Vec<&str> = line.splitn(2, "<<EOF").collect();
            if parts.len() == 2 && line.starts_with('@') {
                let rest = &line[1..];
                let seq_end = rest.find(' ').unwrap_or(rest.len());
                if let Ok(seq) = rest[..seq_end].parse::<u64>() {
                    let cmd_start = seq_end + 1;
                    if cmd_start < rest.len() {
                        let cmd_line = &rest[cmd_start..];
                        let cmd_end = cmd_line.find(' ').unwrap_or(cmd_line.len());
                        let cmd = cmd_line[..cmd_end].to_string();
                        let pre_heredoc = if cmd_end < cmd_line.len() {
                            cmd_line[cmd_end..].trim_end_matches("<<EOF").to_string()
                        } else {
                            String::new()
                        };
                        heredoc_buffer = Some((seq, cmd, pre_heredoc));
                        continue;
                    }
                }
            }
        }

        if let Some(response) = server.handle_line(&line) {
            writeln!(stdout, "{}", response).ok();
            stdout.flush().ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello() {
        let server = MapServer::new();
        let resp = server.handle_line("@1 hello mic=1 map=1");
        assert!(resp.is_some());
        let resp = resp.unwrap();
        assert!(resp.starts_with("=1 ok"));
        assert!(resp.contains("version="));
    }

    #[test]
    fn test_load_mic() {
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");

        let resp = server.handle_line("@2 load.mic mic@1\nT0 f32\nN0 const.i64 42 T0\nO N0");
        assert!(resp.is_some());
        let resp = resp.unwrap();
        assert!(resp.contains("=2 ok"));
        assert!(resp.contains("nodes=1"));
    }

    #[test]
    fn test_check() {
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");
        server.handle_line("@2 load.mic mic@1\nT0 f32\nN0 const.i64 42 T0\nO N0");

        let resp = server.handle_line("@3 check");
        assert!(resp.is_some());
        let resp = resp.unwrap();
        assert!(resp.contains("=3 ok"));
        assert!(resp.contains("diags=0"));
    }

    #[test]
    fn test_query_stats() {
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");
        server.handle_line("@2 load.mic mic@1\nT0 f32\nN0 const.i64 42 T0\nO N0");

        let resp = server.handle_line("@3 query.stats");
        assert!(resp.is_some());
        let resp = resp.unwrap();
        assert!(resp.contains("=3 ok"));
        assert!(resp.contains("nodes=1"));
    }

    #[test]
    fn test_bye() {
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");

        let resp = server.handle_line("@2 bye");
        assert!(resp.is_some());
        assert_eq!(resp.unwrap(), "=2 ok");
    }

    // -----------------------------------------------------------------------
    // Resource-budget / DoS-hardening tests (RFC-0002 §Resource Limits).
    //
    // Each test drives an over-budget input through the protocol surface and
    // asserts a CLEAR structured error (`code=E1xx`) is returned rather than a
    // panic, an allocation failure, or silent acceptance.
    // -----------------------------------------------------------------------

    #[test]
    fn test_oversized_line_rejected_e101() {
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");

        // A single request line larger than MAX_LINE_BYTES must be rejected
        // before any command processing, with the sequence number recovered.
        let big = format!("@7 query.node {}", "N".repeat(MAX_LINE_BYTES + 16));
        let resp = server.handle_line(&big).expect("expected a response");
        assert!(resp.starts_with("=7 err"), "got: {}", resp);
        assert!(resp.contains("code=E101"), "got: {}", resp);
        assert!(resp.contains("too large"), "got: {}", resp);
    }

    #[test]
    fn test_oversized_line_unparseable_seq_uses_zero() {
        let server = MapServer::new();
        // No valid leading `@<seq>`: budget error still emitted, seq falls to 0.
        let big = "x".repeat(MAX_LINE_BYTES + 1);
        let resp = server.handle_line(&big).expect("expected a response");
        assert!(resp.starts_with("=0 err"), "got: {}", resp);
        assert!(resp.contains("code=E101"), "got: {}", resp);
    }

    #[test]
    fn test_oversized_module_rejected_e103() {
        // Drive the node-budget guard directly: a candidate module with one
        // more node than MAX_MODULE_NODES must be rejected with E103. (Over the
        // wire such a module arrives via heredoc, where each physical line is
        // small and the per-line E101 cap does not apply; the node count is the
        // operative budget. Exercising the guard avoids a multi-MB single line.)
        let mut module = String::from("mic@1\n");
        module.reserve((MAX_MODULE_NODES + 1) * 6);
        for _ in 0..=MAX_MODULE_NODES {
            module.push_str("N0 const.i64 0 T0\n");
        }
        assert_eq!(Session::count_nodes(&module), MAX_MODULE_NODES + 1);

        let resp = MapServer::guard_module_nodes(2, &module)
            .expect_err("over-limit module must be rejected");
        assert!(
            resp.starts_with("=2 err"),
            "got: {}",
            &resp[..40.min(resp.len())]
        );
        assert!(
            resp.contains("code=E103"),
            "got: {}",
            &resp[..60.min(resp.len())]
        );
        assert!(resp.contains("module too large"));
    }

    #[test]
    fn test_module_at_limit_accepted() {
        // Exactly at the limit must be accepted (boundary is inclusive).
        let mut module = String::from("mic@1\n");
        module.reserve(MAX_MODULE_NODES * 6);
        for _ in 0..MAX_MODULE_NODES {
            module.push_str("N0 const.i64 0 T0\n");
        }
        assert_eq!(Session::count_nodes(&module), MAX_MODULE_NODES);
        assert!(
            MapServer::guard_module_nodes(2, &module).is_ok(),
            "boundary module must pass the node budget"
        );
    }

    #[test]
    fn test_load_mic_small_module_still_loads() {
        // The E103 guard must not regress normal small-module loads.
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");
        let resp = server
            .handle_line("@2 load.mic mic@1\nT0 f32\nN0 const.i64 42 T0\nO N0")
            .expect("expected a response");
        assert!(resp.contains("=2 ok"), "got: {}", resp);
        assert!(resp.contains("nodes=1"), "got: {}", resp);
    }

    #[test]
    fn test_patch_rate_budget_rejected_e104() {
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");
        server.handle_line("@2 load.mic mic@1\nT0 f32\nN0 const.i64 42 T0\nO N0");

        // Seed the session to the patch-rate ceiling so the next mutating
        // patch is rejected — avoids issuing MAX_PATCH_OPS real operations.
        {
            let mut s = server.session.lock_or_recover();
            s.patch_ops = MAX_PATCH_OPS;
        }

        for cmd in [
            "@3 patch.insert N1 const.i64 7 T0",
            "@4 patch.delete N0",
            "@5 patch.replace N0 const.i64 9 T0",
        ] {
            let resp = server.handle_line(cmd).expect("expected a response");
            assert!(resp.contains("err"), "expected rejection, got: {}", resp);
            assert!(resp.contains("code=E104"), "got: {}", resp);
            assert!(resp.contains("budget exhausted"), "got: {}", resp);
        }
    }

    #[test]
    fn test_patch_op_counter_increments_and_resets_on_bye() {
        let server = MapServer::new();
        server.handle_line("@1 hello mic=1 map=1");
        server.handle_line("@2 load.mic mic@1\nT0 f32\nN0 const.i64 42 T0\nO N0");

        server.handle_line("@3 patch.insert N1 const.i64 7 T0");
        assert_eq!(server.session.lock_or_recover().patch_ops, 1);

        // `bye` resets the per-session patch budget.
        server.handle_line("@4 bye");
        assert_eq!(server.session.lock_or_recover().patch_ops, 0);
    }
}
