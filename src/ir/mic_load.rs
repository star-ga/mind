// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! mic@1 / MIC@3 load and save for an [`IRModule`].

use super::{IRModule, compact};
use std::fmt;

/// Errors surfaced by [`load`].
#[derive(Debug)]
pub enum LoadError {
    /// Input bytes are not valid UTF-8 (mic@1/mic@2 are text formats).
    InvalidUtf8(std::str::Utf8Error),
    /// Input did not match a recognised MIC format header.
    UnknownFormat,
    /// mic@1 parser failed.
    Mic1(compact::MicParseError),
    /// mic@2 was detected; use `compact::v2::parse_mic2` directly — that
    /// format produces `Graph`, not [`IRModule`].
    Mic2NotSupportedByLoad,
    /// Binary MIC-B was detected; use `compact::v2::parse_micb` directly.
    MicBNotSupportedByLoad,
    /// MIC@3 binary parse error (RFC 0021).
    Mic3(compact::Mic3Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::InvalidUtf8(e) => write!(f, "invalid UTF-8: {}", e),
            LoadError::UnknownFormat => f.write_str("unrecognised MIC format header"),
            LoadError::Mic1(e) => write!(f, "mic@1 parse error: {}", e.message),
            LoadError::Mic2NotSupportedByLoad => {
                f.write_str("mic@2 detected — use compact::v2::parse_mic2 directly")
            }
            LoadError::MicBNotSupportedByLoad => {
                f.write_str("MIC-B detected — use compact::v2::parse_micb directly")
            }
            LoadError::Mic3(e) => write!(f, "mic@3 parse error: {}", e),
        }
    }
}

impl std::error::Error for LoadError {}

/// Load an [`IRModule`] from mic@1 text or a MIC@3 artifact.
///
/// This is the stable runtime-facing entry point used by `mind-runtime` and
/// other backends to consume pre-compiled IR without re-running the surface
/// parser. MIC@3 accepts a body with an optional well-formed MAP epilogue;
/// signature and evidence verification remain separate operations.
///
/// `mic@2` and `MIC-B` are also detected, but those formats produce a
/// different [`Graph`](compact::v2::Graph) type and must be loaded through
/// [`compact::v2::parse_mic2`] / [`compact::v2::parse_micb`] directly.
///
/// # Stability
/// The mic@1 textual form is part of the v0.2.x stability surface — see
/// `docs/versioning.md` and `docs/ir-stability.md`.
pub fn load(data: &[u8]) -> Result<IRModule, LoadError> {
    match compact::detect_format(data) {
        compact::MicFormat::Mic1 => {
            let text = std::str::from_utf8(data).map_err(LoadError::InvalidUtf8)?;
            compact::parse_mic(text).map_err(LoadError::Mic1)
        }
        compact::MicFormat::Mic2 => Err(LoadError::Mic2NotSupportedByLoad),
        compact::MicFormat::MicB => Err(LoadError::MicBNotSupportedByLoad),
        compact::MicFormat::Mic3 => compact::parse_mic3_envelope(data).map_err(|error| {
            LoadError::Mic3(compact::Mic3Error {
                message: error.to_string(),
            })
        }),
        compact::MicFormat::Unknown => {
            // Not a recognised binary magic (mic@3 / MIC-B). If the bytes are
            // not valid UTF-8 either, the input is malformed binary, not a
            // text format we failed to recognise — surface that as
            // `InvalidUtf8` (the text formats mic@1/mic@2 are UTF-8). Valid
            // UTF-8 with an unknown header is a genuine `UnknownFormat`.
            match std::str::from_utf8(data) {
                Err(e) => Err(LoadError::InvalidUtf8(e)),
                Ok(_) => Err(LoadError::UnknownFormat),
            }
        }
    }
}

/// Serialize an [`IRModule`] to mic@1 text.
///
/// This is the stable counterpart to [`load`]. Output is deterministic
/// (RFC-0001): the same module always produces byte-identical text.
pub fn save(module: &IRModule) -> String {
    compact::emit_mic(module)
}
