// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Content-addressed fingerprint used as the cache key.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ProfileTag {
    #[default]
    Default,
    Systems,
    Embedded,
}

impl ProfileTag {
    /// Parse a profile name from CLI / Mind.toml. Unknown names map to
    /// `Default` — the strict-validation path lives at the clap layer.
    pub fn parse(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "systems" => ProfileTag::Systems,
            "embedded" => ProfileTag::Embedded,
            _ => ProfileTag::Default,
        }
    }
}

impl ProfileTag {
    /// String form used in cache-key fingerprints and Display.
    pub fn as_str(self) -> &'static str {
        match self {
            ProfileTag::Default => "default",
            ProfileTag::Systems => "systems",
            ProfileTag::Embedded => "embedded",
        }
    }
}

impl fmt::Display for ProfileTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// SHA-256 fingerprint rendered as a 64-character lowercase hex string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fingerprint(pub String);

impl Fingerprint {
    pub fn new(hex: impl Into<String>) -> Self {
        Self(hex.into())
    }

    pub fn shard_prefix(&self) -> &str {
        if self.0.len() >= 2 { &self.0[..2] } else { "" }
    }

    pub fn shard_tail(&self) -> &str {
        if self.0.len() > 2 { &self.0[2..] } else { "" }
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_tag_renders() {
        assert_eq!(ProfileTag::Default.to_string(), "default");
        assert_eq!(ProfileTag::Systems.to_string(), "systems");
        assert_eq!(ProfileTag::Embedded.to_string(), "embedded");
    }

    #[test]
    fn fingerprint_shards() {
        let fp = Fingerprint::new("deadbeef".to_string());
        assert_eq!(fp.shard_prefix(), "de");
        assert_eq!(fp.shard_tail(), "adbeef");
    }

    #[test]
    fn empty_fingerprint_is_safe() {
        let fp = Fingerprint::new("".to_string());
        assert_eq!(fp.shard_prefix(), "");
        assert_eq!(fp.shard_tail(), "");
    }
}
