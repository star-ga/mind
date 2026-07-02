// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! Regression test: the MIC-B parser must reject blobs that ULEB128-encode an
//! astronomically large element count rather than hanging / OOM-ing.
//!
//! This is the DoS class fixed in mic@3 via `read_count` and now ported to
//! mic-b via `read_bounded_count` / `MAX_MICB_ELEMENTS`.  Each sub-test splices
//! the same huge-count varint into a different structural position and verifies
//! that `parse_micb` returns `Err` immediately instead of looping for billions
//! of iterations.
//!
//! ADDITIVE property: these crafted blobs always exceed `MAX_MICB_ELEMENTS`
//! (16 000 000), so the rejection path is entirely separate from valid input.
//! Every existing round-trip test continues to pass byte-identically.

use libmind::ir::compact::v2::parse_micb;
use std::io::Cursor;

/// ULEB128 encoding of 0xFF_FF_FF_FF (4 294 967 295) — fits in 5 bytes and
/// exceeds `MAX_MICB_ELEMENTS` (16 000 000) by ~268×, guaranteeing the guard
/// fires on every platform without any platform-specific integer-size concerns.
///
/// Encoding: 0xFF 0xFF 0xFF 0xFF 0x0F
const HUGE_ULEB: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF, 0x0F];

/// Valid MIC-B header: magic "MICB" + version 0x02.
const HEADER: &[u8] = &[0x4D, 0x49, 0x43, 0x42, 0x02];

/// Build a blob from header + HUGE count as the very first table field
/// (string-table count).  The parser must reject it at the string-table-count
/// site before any allocation.
fn blob_huge_string_count() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.extend_from_slice(HUGE_ULEB); // n_strings = 4 294 967 295
    v
}

/// Build a blob with a valid string-table count of 1, then splice HUGE into
/// the per-string byte-length field.
fn blob_huge_string_len() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x01); // n_strings = 1 (ULEB: 1 byte)
    v.extend_from_slice(HUGE_ULEB); // string[0].len = 4 294 967 295
    v
}

/// Build a blob with a valid string table (0 strings), then splice HUGE into
/// the symbol-table count.
fn blob_huge_symbol_count() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00); // n_strings = 0
    v.extend_from_slice(HUGE_ULEB); // n_symbols = huge
    v
}

/// Build a blob with empty string + symbol tables, then splice HUGE into the
/// type-table count.
fn blob_huge_type_count() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00); // n_strings = 0
    v.push(0x00); // n_symbols = 0
    v.extend_from_slice(HUGE_ULEB); // n_types = huge
    v
}

/// Build a blob with a single type whose shape rank is HUGE.
fn blob_huge_shape_rank() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00); // n_strings = 0
    v.push(0x00); // n_symbols = 0
    v.push(0x01); // n_types = 1
    v.push(0x00); // dtype byte = 0 (F32)
    v.extend_from_slice(HUGE_ULEB); // rank = huge
    v
}

/// Build a blob with empty string/symbol/type tables, then splice HUGE into
/// the value-table count.
fn blob_huge_value_count() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00); // n_strings = 0
    v.push(0x00); // n_symbols = 0
    v.push(0x00); // n_types = 0
    v.extend_from_slice(HUGE_ULEB); // n_values = huge
    v
}

/// Build a blob where all outer tables are empty and the MAP section
/// appears (0x4D sentinel) with a huge entry count.
fn blob_huge_map_count() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00); // n_strings = 0
    v.push(0x00); // n_symbols = 0
    v.push(0x00); // n_types = 0
    v.push(0x00); // n_values = 0
    v.push(0x00); // output = 0 (ULEB)
    v.push(0x4D); // MAP sentinel
    v.extend_from_slice(HUGE_ULEB); // MAP entry count = huge
    v
}

/// Build a blob where a MAP Bytes value carries a huge byte-length.
fn blob_huge_map_bytes_len() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    // String table: 1 entry "k" (key)
    v.push(0x01); // n_strings = 1
    v.push(0x01); // strings[0].len = 1
    v.push(b'k'); // "k"
    v.push(0x00); // n_symbols = 0
    v.push(0x00); // n_types = 0
    v.push(0x00); // n_values = 0
    v.push(0x00); // output = 0 (ULEB)
    v.push(0x4D); // MAP sentinel
    v.push(0x01); // MAP entry count = 1
    v.push(0x00); // key string index = 0 ("k")
    v.push(0x02); // MapValue tag = 2 (Bytes)
    v.extend_from_slice(HUGE_ULEB); // bytes length = huge
    v
}

/// Build a blob with a Node value whose input count is HUGE.
/// We need: 1 string ("x"), 1 type (F32 scalar, rank 0), 2 values
/// (Arg "x" of type 0, then Node tag=2 with opcode Matmul=0 and huge inputs).
fn blob_huge_node_input_count() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    // String table: ["x"]
    v.push(0x01); // n_strings = 1
    v.push(0x01); // len("x") = 1
    v.push(b'x');
    // Symbol table: empty
    v.push(0x00);
    // Type table: 1 type, F32, rank 0
    v.push(0x01); // n_types = 1
    v.push(0x00); // dtype = F32
    v.push(0x00); // rank = 0
    // Value table: 2 values
    v.push(0x02); // n_values = 2
    //   value[0]: Arg "x" type 0
    v.push(0x00); // tag = Arg
    v.push(0x00); // name_idx = 0
    v.push(0x00); // type_idx = 0
    //   value[1]: Node, opcode Matmul (byte 0), HUGE inputs
    v.push(0x02); // tag = Node
    v.push(0x00); // opcode byte = Matmul
    v.extend_from_slice(HUGE_ULEB); // n_inputs = huge
    v
}

/// Build a blob with a Transpose opcode whose perm-length is HUGE.
fn blob_huge_transpose_perm() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00); // n_strings = 0
    v.push(0x00); // n_symbols = 0
    v.push(0x00); // n_types = 0
    v.push(0x01); // n_values = 1
    //   value[0]: Node, opcode Transpose (byte 11), huge perm
    v.push(0x02); // tag = Node
    v.push(0x0B); // opcode byte = 11 = Transpose
    v.extend_from_slice(HUGE_ULEB); // perm count = huge
    v
}

/// Build a blob with a Sum opcode whose axes-length is HUGE.
fn blob_huge_sum_axes() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00); // n_strings = 0
    v.push(0x00); // n_symbols = 0
    v.push(0x00); // n_types = 0
    v.push(0x01); // n_values = 1
    v.push(0x02); // tag = Node
    v.push(0x0D); // opcode byte = 13 = Sum
    v.extend_from_slice(HUGE_ULEB); // axes count = huge
    v
}

/// Build a blob with a Mean opcode whose axes-length is HUGE.
fn blob_huge_mean_axes() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00);
    v.push(0x00);
    v.push(0x00);
    v.push(0x01);
    v.push(0x02);
    v.push(0x0E); // opcode byte = 14 = Mean
    v.extend_from_slice(HUGE_ULEB);
    v
}

/// Build a blob with a Max opcode whose axes-length is HUGE.
fn blob_huge_max_axes() -> Vec<u8> {
    let mut v = HEADER.to_vec();
    v.push(0x00);
    v.push(0x00);
    v.push(0x00);
    v.push(0x01);
    v.push(0x02);
    v.push(0x0F); // opcode byte = 15 = Max
    v.extend_from_slice(HUGE_ULEB);
    v
}

macro_rules! assert_dos_rejected {
    ($name:expr, $blob:expr) => {{
        let blob = $blob;
        let mut cursor = Cursor::new(&blob);
        let result = parse_micb(&mut cursor);
        assert!(
            result.is_err(),
            "parse_micb should have rejected the DoS blob for '{}', but returned Ok",
            $name
        );
        // The error message must mention the huge count to confirm the right
        // guard fired (not an unrelated early error such as a bad magic check).
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("MAX_MICB_ELEMENTS")
                || msg.contains("out of bounds")
                || msg.contains("exceeds"),
            "rejection message for '{}' should indicate a count/size overflow, got: {}",
            $name,
            msg
        );
    }};
}

#[test]
fn micb_dos_huge_string_count_rejected() {
    assert_dos_rejected!("string-table count", blob_huge_string_count());
}

#[test]
fn micb_dos_huge_string_len_rejected() {
    assert_dos_rejected!("per-string byte-length", blob_huge_string_len());
}

#[test]
fn micb_dos_huge_symbol_count_rejected() {
    assert_dos_rejected!("symbol-table count", blob_huge_symbol_count());
}

#[test]
fn micb_dos_huge_type_count_rejected() {
    assert_dos_rejected!("type-table count", blob_huge_type_count());
}

#[test]
fn micb_dos_huge_shape_rank_rejected() {
    assert_dos_rejected!("shape rank", blob_huge_shape_rank());
}

#[test]
fn micb_dos_huge_value_count_rejected() {
    assert_dos_rejected!("value-table count", blob_huge_value_count());
}

#[test]
fn micb_dos_huge_map_count_rejected() {
    assert_dos_rejected!("MAP entry count", blob_huge_map_count());
}

#[test]
fn micb_dos_huge_map_bytes_len_rejected() {
    assert_dos_rejected!("MAP Bytes value length", blob_huge_map_bytes_len());
}

#[test]
fn micb_dos_huge_node_input_count_rejected() {
    assert_dos_rejected!("node input count", blob_huge_node_input_count());
}

#[test]
fn micb_dos_huge_transpose_perm_rejected() {
    assert_dos_rejected!("Transpose perm count", blob_huge_transpose_perm());
}

#[test]
fn micb_dos_huge_sum_axes_rejected() {
    assert_dos_rejected!("Sum axes count", blob_huge_sum_axes());
}

#[test]
fn micb_dos_huge_mean_axes_rejected() {
    assert_dos_rejected!("Mean axes count", blob_huge_mean_axes());
}

#[test]
fn micb_dos_huge_max_axes_rejected() {
    assert_dos_rejected!("Max axes count", blob_huge_max_axes());
}
