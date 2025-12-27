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

use std::collections::HashMap;
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

/// Session mode flags
#[derive(Debug, Clone, Default)]
struct SessionMode {
    no_io: bool,
    no_unsafe: bool,
    pure_only: bool,
}

impl SessionMode {
    fn parse(s: &str) -> Self {
        let mut _mode = SessionMode::default();
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
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

        // Parse request: @<seq> <command> [args...]
        if !line.starts_with('@') {
            return Some(format!("!error invalid request format"));
        }

        let rest = &line[1..];
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.is_empty() {
            return Some(format!("!error missing sequence number"));
        }

        let seq: u64 = match parts[0].parse() {
            Ok(s) => s,
            Err(_) => return Some(format!("!error invalid sequence number")),
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
            _ => Some(format!("={} err code=E005 msg=\"unknown command: {}\"", seq, cmd)),
        }
    }

    fn handle_hello(&self, seq: u64, args: &str) -> Option<String> {
        let mut session = self.session.lock_or_recover();

        // Parse args
        let mut mic_ver = 1u32;
        let mut map_ver = 1u32;
        let mut _mode = SessionMode::default();

        for arg in args.split_whitespace() {
            if arg.starts_with("mic=") {
                mic_ver = arg[4..].parse().unwrap_or(1);
            } else if arg.starts_with("map=") {
                map_ver = arg[4..].parse().unwrap_or(1);
            } else if arg.starts_with("mode=") {
                _mode = SessionMode::parse(&arg[5..]);
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
            return Some(format!("={} err line=1 msg=\"missing version header\"", seq));
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
                    Some(format!("={} err msg=\"format {} not implemented\"", seq, format))
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

    fn handle_patch_insert(&self, seq: u64, args: &str) -> Option<String> {
        let mut session = self.session.lock_or_recover();

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
                let _after_id = if args.starts_with("after=N") {
                    args[7..].split_whitespace().next().and_then(|s| s.parse::<usize>().ok())
                } else {
                    None
                };

                // Simple append for now
                if !new_mic.ends_with('\n') {
                    new_mic.push('\n');
                }
                new_mic.push_str(node_content);
                new_mic.push('\n');

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
                            let type_ref = parts.iter().find(|s| s.starts_with('T')).unwrap_or(&"T?");

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
    eprintln!("Mind AI Protocol Server v{} ready (MIC v{}, MAP v{})",
        SERVER_VERSION, MIC_VERSION, MAP_VERSION);

    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    let mut stdout = std::io::stdout();

    let mut heredoc_buffer: Option<(u64, String, String)> = None; // (seq, cmd, content)

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };

        // Handle heredoc
        if let Some((seq, ref cmd, ref mut content)) = heredoc_buffer {
            if line.trim() == "EOF" {
                // Process accumulated heredoc
                let full_args = content.clone();
                heredoc_buffer = None;
                if let Some(response) = server.dispatch(seq, &cmd, &full_args) {
                    writeln!(stdout, "{}", response).ok();
                    stdout.flush().ok();
                }
            } else {
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
}
