// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

/// Byte-offset-to-line-number index built from a source string.
pub(super) struct LineIndex {
    /// Byte offsets of each line's first character.
    pub(super) starts: Vec<usize>,
}

impl LineIndex {
    pub(super) fn build(src: &str) -> Self {
        let mut starts = vec![0usize];
        for (i, &b) in src.as_bytes().iter().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        Self { starts }
    }

    /// Zero-based line number for `byte_offset`.
    pub(super) fn line_of(&self, byte_offset: usize) -> usize {
        match self.starts.binary_search(&byte_offset) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }
}
