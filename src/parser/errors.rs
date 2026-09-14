// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

#[derive(Debug, Clone)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
    /// Stable cause code for diagnostics whose parser boundary has a
    /// feature-specific contract. Ordinary parse errors remain `None`.
    pub cause_code: Option<&'static str>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at offset {}: {}", self.offset, self.message)
    }
}
