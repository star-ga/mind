#![cfg(feature = "ffi-c")]

// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

#[test]
fn header_contains_expected_symbols() {
    let header = mind::ffi::header::generate_header();
    assert!(header.contains("MindTensor"));
    assert!(header.contains("mind_infer"));
}
