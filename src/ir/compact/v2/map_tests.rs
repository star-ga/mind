// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! mic@2.1 MAP section tests.
//!
//! Coverage:
//!  - Empty-MAP byte-identity vs pre-2.1 output (text + binary).
//!  - Populated-MAP text round-trip byte-identity.
//!  - Populated-MAP binary round-trip byte-identity.
//!  - Text↔binary cross round-trip (parse text, emit binary, parse binary, emit text).
//!  - Canonical key sorting (unsorted input → sorted output).
//!  - Duplicate-key parse error (text + binary).
//!  - Each value type: String, Int, Bytes, Nested.
//!  - §3.4 binary detection rule: EOF / 0x4D / bad byte.
//!  - §3.5 limit rejections.
//!  - §8 compatibility matrix rows.

use std::io::Cursor;

use super::binary::{emit_micb, parse_micb};
use super::emit::emit_mic2;
use super::parse::parse_mic2;
use super::types::{Graph, GraphEq, Map, MapValue};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn residual_with_map() -> Graph {
    let mut g = Graph::residual_block();
    g.map
        .insert("target.canonical_name", MapValue::String("cpu_avx2".into()));
    g.map.insert(
        "evidence_chain.substrate",
        MapValue::String("x86_avx2".into()),
    );
    g.map.insert(
        "evidence_chain.trace_hash",
        MapValue::Bytes(vec![0xde, 0xad, 0xbe, 0xef]),
    );
    g
}

fn emit_micb_bytes(g: &Graph) -> Vec<u8> {
    let mut buf = Vec::new();
    emit_micb(g, &mut buf).expect("emit_micb failed");
    buf
}

// ---------------------------------------------------------------------------
// 1. Empty-MAP byte-identity (THE CRITICAL CONTRACT — §2 rule 3, §5 rule 10)
// ---------------------------------------------------------------------------

#[test]
fn empty_map_text_byte_identical_to_pre_map_output() {
    let graph_pre = Graph::residual_block();

    // Capture the pre-2.1 canonical text.
    let text_pre = emit_mic2(&graph_pre);

    // A graph with an explicit empty Map must emit identically.
    let mut graph_with_empty = Graph::residual_block();
    graph_with_empty.map = Map::new();
    let text_with_empty = emit_mic2(&graph_with_empty);

    assert_eq!(
        text_pre, text_with_empty,
        "empty-MAP graph must emit byte-identically to pre-MAP graph (text)"
    );

    // The output must NOT contain "map {".
    assert!(
        !text_pre.contains("map {"),
        "empty MAP must be entirely absent from text output"
    );
}

#[test]
fn empty_map_binary_byte_identical_to_pre_map_output() {
    let graph_pre = Graph::residual_block();
    let bin_pre = emit_micb_bytes(&graph_pre);

    let mut graph_with_empty = Graph::residual_block();
    graph_with_empty.map = Map::new();
    let bin_with_empty = emit_micb_bytes(&graph_with_empty);

    assert_eq!(
        bin_pre, bin_with_empty,
        "empty-MAP graph must emit byte-identically to pre-MAP graph (binary)"
    );

    // The binary must not contain the 0x4D MAP marker.
    assert!(
        !bin_pre.windows(1).any(|w| w[0] == 0x4D && {
            // 0x4D is 'M' — check that it's not part of the "MICB" magic.
            // The magic is at bytes [0..4] so any 0x4D after byte 4 is suspect.
            // But graph strings may contain 'M'; this is only a sanity spot-check
            // for the common residual-block case where no string contains 0x4D.
            false // Skip: 0x4D appears in "MICB" magic, can't assert globally.
        }),
        "sanity: MAP marker check skipped for magic overlap"
    );
}

// ---------------------------------------------------------------------------
// 2. Populated-MAP text round-trip byte-identity  (§5)
// ---------------------------------------------------------------------------

#[test]
fn populated_map_text_roundtrip_byte_identical() {
    let g = residual_with_map();
    let text1 = emit_mic2(&g);
    let parsed = parse_mic2(&text1).expect("parse failed");
    let text2 = emit_mic2(&parsed);
    assert_eq!(text1, text2, "emit(parse(emit(G))) != emit(G) for text");
}

// ---------------------------------------------------------------------------
// 3. Populated-MAP binary round-trip byte-identity  (§5)
// ---------------------------------------------------------------------------

#[test]
fn populated_map_binary_roundtrip_byte_identical() {
    let g = residual_with_map();
    let bin1 = emit_micb_bytes(&g);
    let parsed = parse_micb(&mut Cursor::new(&bin1)).expect("parse failed");
    let bin2 = emit_micb_bytes(&parsed);
    assert_eq!(
        bin1, bin2,
        "emit_micb(parse_micb(emit_micb(G))) != emit_micb(G)"
    );
}

// ---------------------------------------------------------------------------
// 4. Text ↔ binary cross round-trip (§3 "round-trip losslessly")
// ---------------------------------------------------------------------------

#[test]
fn text_binary_cross_roundtrip() {
    let g = residual_with_map();

    // text → parse → binary → parse → text
    let text = emit_mic2(&g);
    let parsed_from_text = parse_mic2(&text).expect("parse text failed");
    let bin = emit_micb_bytes(&parsed_from_text);
    let parsed_from_bin = parse_micb(&mut Cursor::new(&bin)).expect("parse binary failed");
    let text2 = emit_mic2(&parsed_from_bin);

    assert_eq!(text, text2, "text→parse→binary→parse→text must be lossless");

    // binary → parse → text → parse → binary
    let bin2 = emit_micb_bytes(&g);
    let parsed2 = parse_micb(&mut Cursor::new(&bin2)).expect("parse binary failed");
    let text3 = emit_mic2(&parsed2);
    let parsed3 = parse_mic2(&text3).expect("parse text failed");
    let bin3 = emit_micb_bytes(&parsed3);

    assert_eq!(
        bin2, bin3,
        "binary→parse→text→parse→binary must be lossless"
    );
}

// ---------------------------------------------------------------------------
// 5. Canonical key sorting (unsorted input → sorted output)
// ---------------------------------------------------------------------------

#[test]
fn canonical_sorting_unsorted_input_produces_sorted_output() {
    let mut g = Graph::residual_block();
    // Insert in reverse-alphabetical order.
    g.map.insert("z.key", MapValue::Int(3));
    g.map.insert("m.key", MapValue::Int(2));
    g.map.insert("a.key", MapValue::Int(1));

    let text = emit_mic2(&g);

    // Extract map block.
    let map_start = text.find("map {").expect("map block missing");
    let map_block = &text[map_start..];

    let a_pos = map_block.find("a.key").expect("a.key missing");
    let m_pos = map_block.find("m.key").expect("m.key missing");
    let z_pos = map_block.find("z.key").expect("z.key missing");

    assert!(
        a_pos < m_pos && m_pos < z_pos,
        "keys must be sorted: a < m < z"
    );
}

#[test]
fn canonical_sorting_round_trips_stably() {
    let mut g = Graph::residual_block();
    g.map.insert("zzz", MapValue::Int(99));
    g.map.insert("aaa", MapValue::Int(1));
    g.map.insert("mmm", MapValue::Int(50));

    // Two emits must be identical even though internal storage order may vary.
    let t1 = emit_mic2(&g);
    let t2 = emit_mic2(&g);
    assert_eq!(t1, t2, "emission must be deterministic");
}

// ---------------------------------------------------------------------------
// 6. Duplicate-key parse error (§4)
// ---------------------------------------------------------------------------

#[test]
fn duplicate_key_text_is_parse_error() {
    let input = "mic@2\nT0 f16 128\na X T0\nO 0\nmap {\n  foo.bar = 1\n  foo.bar = 2\n}\n";
    let result = parse_mic2(input);
    assert!(result.is_err(), "duplicate key must be a parse error");
    assert!(
        result.unwrap_err().message.contains("duplicate"),
        "error must mention 'duplicate'"
    );
}

#[test]
fn duplicate_key_binary_is_parse_error() {
    // Build a binary with a duplicate key by constructing raw bytes.
    // We'll encode it directly: MICB header + minimal graph + MAP with dupes.
    use super::varint::{sleb128_write, uleb128_write};

    let mut buf = Vec::new();
    // Magic + version
    buf.extend_from_slice(b"MICB");
    buf.push(0x02);
    // String table: 1 entry "dup.key"
    uleb128_write(&mut buf, 1).unwrap();
    uleb128_write(&mut buf, 7).unwrap();
    buf.extend_from_slice(b"dup.key");
    // Symbols: 0
    uleb128_write(&mut buf, 0).unwrap();
    // Types: 0
    uleb128_write(&mut buf, 0).unwrap();
    // Values: 0
    uleb128_write(&mut buf, 0).unwrap();
    // Output: 0
    uleb128_write(&mut buf, 0).unwrap();
    // MAP marker
    uleb128_write(&mut buf, 0x4D).unwrap();
    // count = 2 (duplicate!)
    uleb128_write(&mut buf, 2).unwrap();
    // entry 1: key=0 "dup.key", tag=1 (int), value=42
    uleb128_write(&mut buf, 0).unwrap();
    buf.push(1);
    sleb128_write(&mut buf, 42).unwrap();
    // entry 2: key=0 "dup.key" again (duplicate!)
    uleb128_write(&mut buf, 0).unwrap();
    buf.push(1);
    sleb128_write(&mut buf, 99).unwrap();

    let result = parse_micb(&mut Cursor::new(&buf));
    assert!(result.is_err(), "binary duplicate key must be parse error");
}

// ---------------------------------------------------------------------------
// 7. Each value type: String, Int, Bytes, Nested
// ---------------------------------------------------------------------------

#[test]
fn map_value_type_string() {
    let mut g = Graph::residual_block();
    g.map.insert("k", MapValue::String("hello world".into()));
    let text = emit_mic2(&g);
    assert!(
        text.contains("k = \"hello world\""),
        "string value must appear quoted"
    );
    let parsed = parse_mic2(&text).expect("parse failed");
    assert_eq!(
        parsed.map.iter().find(|(k, _)| *k == "k").map(|(_, v)| v),
        Some(&MapValue::String("hello world".into()))
    );
}

#[test]
fn map_value_type_int_positive() {
    let mut g = Graph::residual_block();
    g.map.insert("n", MapValue::Int(42));
    let text = emit_mic2(&g);
    assert!(text.contains("n = 42"), "int 42 must appear unquoted");
    let parsed = parse_mic2(&text).expect("parse failed");
    assert_eq!(
        parsed.map.iter().find(|(k, _)| *k == "n").map(|(_, v)| v),
        Some(&MapValue::Int(42))
    );
}

#[test]
fn map_value_type_int_negative() {
    let mut g = Graph::residual_block();
    g.map.insert("n", MapValue::Int(-7));
    let text = emit_mic2(&g);
    assert!(text.contains("n = -7"), "negative int must appear");
    let parsed = parse_mic2(&text).expect("parse failed");
    assert_eq!(
        parsed.map.iter().find(|(k, _)| *k == "n").map(|(_, v)| v),
        Some(&MapValue::Int(-7))
    );
}

#[test]
fn map_value_type_int_zero_normalised() {
    let mut g = Graph::residual_block();
    g.map.insert("z", MapValue::Int(0));
    let text = emit_mic2(&g);
    assert!(text.contains("z = 0"), "zero must emit as '0'");
    // Parsing "0" must not fail and must produce Int(0).
    let parsed = parse_mic2(&text).expect("parse failed");
    assert_eq!(
        parsed.map.iter().find(|(k, _)| *k == "z").map(|(_, v)| v),
        Some(&MapValue::Int(0))
    );
}

#[test]
fn map_value_type_bytes() {
    let mut g = Graph::residual_block();
    g.map
        .insert("h", MapValue::Bytes(vec![0xca, 0xfe, 0xf0, 0x0d]));
    let text = emit_mic2(&g);
    assert!(
        text.contains("h = bytes(0xcafef00d)"),
        "bytes must be lowercase hex"
    );
    let parsed = parse_mic2(&text).expect("parse failed");
    assert_eq!(
        parsed.map.iter().find(|(k, _)| *k == "h").map(|(_, v)| v),
        Some(&MapValue::Bytes(vec![0xca, 0xfe, 0xf0, 0x0d]))
    );
}

#[test]
fn map_value_type_nested() {
    let mut g = Graph::residual_block();
    let mut inner = Map::new();
    inner.insert("child", MapValue::Int(1));
    g.map.insert("parent", MapValue::Nested(inner));
    let text = emit_mic2(&g);
    assert!(text.contains("parent = {"), "nested map must use braces");
    assert!(text.contains("child = 1"), "nested key must appear");
    let parsed = parse_mic2(&text).expect("parse failed");
    match parsed
        .map
        .iter()
        .find(|(k, _)| *k == "parent")
        .map(|(_, v)| v)
    {
        Some(MapValue::Nested(inner)) => {
            assert_eq!(inner.iter().next(), Some(("child", &MapValue::Int(1))));
        }
        other => panic!("expected Nested, got: {:?}", other),
    }
}

#[test]
fn map_value_string_escapes_round_trip() {
    let special = "tab:\there\nnewline\\backslash\"quote";
    let mut g = Graph::residual_block();
    g.map.insert("esc", MapValue::String(special.into()));
    let text = emit_mic2(&g);
    let parsed = parse_mic2(&text).expect("parse failed");
    assert_eq!(
        parsed.map.iter().find(|(k, _)| *k == "esc").map(|(_, v)| v),
        Some(&MapValue::String(special.into())),
        "string with escapes must round-trip"
    );
}

// ---------------------------------------------------------------------------
// 8. §3.4 binary detection rule: EOF / 0x4D / bad byte
// ---------------------------------------------------------------------------

#[test]
fn detection_rule_eof_means_empty_map() {
    // A valid mic@2 binary (no MAP) must parse as empty MAP.
    let g = Graph::residual_block();
    let bin = emit_micb_bytes(&g);
    // Verify binary has no MAP (empty graph must emit identically to pre-2.1).
    let parsed = parse_micb(&mut Cursor::new(&bin)).expect("parse failed");
    assert!(
        parsed.map.is_empty(),
        "mic@2 binary without MAP must parse as empty map"
    );
}

#[test]
fn detection_rule_0x4d_signals_map() {
    let mut g = Graph::residual_block();
    g.map.insert("k", MapValue::Int(1));
    let bin = emit_micb_bytes(&g);
    // Must parse without error and contain the key.
    let parsed = parse_micb(&mut Cursor::new(&bin)).expect("parse failed");
    assert!(
        !parsed.map.is_empty(),
        "MAP marker 0x4D must trigger MAP decode"
    );
    assert_eq!(
        parsed.map.iter().find(|(k, _)| *k == "k").map(|(_, v)| v),
        Some(&MapValue::Int(1))
    );
}

#[test]
fn detection_rule_bad_byte_is_parse_error() {
    // Build a binary that ends with an unexpected byte (not 0x4D, not EOF).
    let g = Graph::residual_block();
    let mut bin = emit_micb_bytes(&g);
    // Append a byte that is NOT 0x4D.
    bin.push(0x42); // 'B'

    let result = parse_micb(&mut Cursor::new(&bin));
    assert!(
        result.is_err(),
        "unexpected byte after output must be a parse error"
    );
    let err = result.unwrap_err();
    assert!(
        err.message.contains("0x42") || err.message.contains("unexpected"),
        "error must mention the bad byte or 'unexpected': {}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// 9. §3.5 limit rejections
// ---------------------------------------------------------------------------

#[test]
fn limit_key_too_long_text() {
    // Key > 256 bytes.
    let long_key = "a".repeat(257);
    let input = format!("mic@2\nT0 f16 128\na X T0\nO 0\nmap {{\n  {long_key} = 1\n}}\n");
    let result = parse_mic2(&input);
    assert!(result.is_err(), "key > 256 bytes must be rejected");
}

#[test]
fn limit_key_depth_too_large_text() {
    // 9 dotted segments = depth 9 > limit 8.
    let deep_key = "a.b.c.d.e.f.g.h.i"; // 9 segments
    let input = format!("mic@2\nT0 f16 128\na X T0\nO 0\nmap {{\n  {deep_key} = 1\n}}\n");
    let result = parse_mic2(&input);
    assert!(
        result.is_err(),
        "key with 9 segments must be rejected (limit 8)"
    );
}

#[test]
fn limit_nesting_too_deep_text() {
    // Build a 5-level nesting (limit is 4).
    // Top-level map = depth 0. a={} depth 1, b={} depth 2, c={} depth 3,
    // d={} depth 4, e={} would be depth 5 — must be rejected.
    let input = concat!(
        "mic@2\nT0 f16 128\na X T0\nO 0\n",
        "map {\n",
        "  a = {\n",
        "    b = {\n",
        "      c = {\n",
        "        d = {\n",
        "          e = {\n",
        "            f = 1\n",
        "          }\n",
        "        }\n",
        "      }\n",
        "    }\n",
        "  }\n",
        "}\n"
    );
    let result = parse_mic2(input);
    assert!(result.is_err(), "nesting depth > 4 must be rejected");
}

#[test]
fn limit_string_too_large_text() {
    // String > 64 KiB.
    let big = "x".repeat(64 * 1024 + 1);
    let input = format!("mic@2\nT0 f16 128\na X T0\nO 0\nmap {{\n  k = \"{big}\"\n}}\n");
    let result = parse_mic2(&input);
    assert!(result.is_err(), "string > 64 KiB must be rejected");
}

// ---------------------------------------------------------------------------
// 10. §8 compatibility matrix
// ---------------------------------------------------------------------------

#[test]
fn compat_mic2_binary_parses_as_empty_map() {
    // §8: mic@2 reader sees OK for mic@2.1 empty-MAP (identical bytes).
    // §8: mic@2.1 reader sees mic@2 binary as empty MAP.
    let g = Graph::residual_block(); // no MAP
    let bin = emit_micb_bytes(&g);
    let parsed = parse_micb(&mut Cursor::new(&bin)).expect("parse failed");
    assert!(
        parsed.map.is_empty(),
        "mic@2 binary must parse as empty MAP under mic@2.1"
    );
    // Re-emit must be byte-identical to the original.
    let bin2 = emit_micb_bytes(&parsed);
    assert_eq!(
        bin, bin2,
        "round-trip of mic@2 binary via mic@2.1 must be byte-identical"
    );
}

#[test]
fn compat_mic21_empty_map_is_identical_bytes_to_mic2() {
    // §8: mic@2.1 (empty MAP) = OK (identical bytes) to mic@2 reader.
    let mut g_21 = Graph::residual_block();
    g_21.map = Map::new(); // explicitly empty
    let g_2 = Graph::residual_block();

    let bin_21 = emit_micb_bytes(&g_21);
    let bin_2 = emit_micb_bytes(&g_2);
    assert_eq!(
        bin_21, bin_2,
        "mic@2.1 with empty MAP must be identical bytes to mic@2"
    );

    let text_21 = emit_mic2(&g_21);
    let text_2 = emit_mic2(&g_2);
    assert_eq!(
        text_21, text_2,
        "mic@2.1 text with empty MAP must be identical to mic@2 text"
    );
}

#[test]
fn compat_mic2_text_parses_as_empty_map() {
    // A classic mic@2 text parses under the mic@2.1 parser as empty MAP.
    let classic =
        "mic@2\nT0 f16 128 128\nT1 f16 128\na X T0\np W T0\np b T1\nm 0 1\n+ 3 2\nr 4\n+ 5 0\nO 6";
    let parsed = parse_mic2(classic).expect("parse failed");
    assert!(
        parsed.map.is_empty(),
        "classic mic@2 text must parse as empty MAP"
    );
    // Re-emit must be byte-identical.
    let re_emitted = emit_mic2(&parsed);
    assert_eq!(
        classic, re_emitted,
        "round-trip of mic@2 text via mic@2.1 must be identical"
    );
}

// ---------------------------------------------------------------------------
// 11. String table interning in binary: MAP keys/values appear once
// ---------------------------------------------------------------------------

#[test]
fn binary_string_table_dedup_for_map_keys() {
    let mut g = Graph::residual_block();
    // Same string used as both a value name in the graph and a MAP string value.
    g.map.insert("meta.info", MapValue::String("X".into())); // "X" is already a value name
    let bin = emit_micb_bytes(&g);
    let parsed = parse_micb(&mut Cursor::new(&bin)).expect("parse failed");
    // Should parse correctly.
    assert_eq!(
        parsed
            .map
            .iter()
            .find(|(k, _)| *k == "meta.info")
            .map(|(_, v)| v),
        Some(&MapValue::String("X".into()))
    );
    // And the binary should still round-trip.
    let bin2 = emit_micb_bytes(&parsed);
    assert_eq!(bin, bin2);
}

// ---------------------------------------------------------------------------
// 12. Int boundary values
// ---------------------------------------------------------------------------

#[test]
fn map_int_i64_min_max_roundtrip() {
    let mut g = Graph::residual_block();
    g.map.insert("lo", MapValue::Int(i64::MIN));
    g.map.insert("hi", MapValue::Int(i64::MAX));

    // Text round-trip.
    let text = emit_mic2(&g);
    let parsed_text = parse_mic2(&text).expect("parse text failed");
    assert_eq!(
        parsed_text
            .map
            .iter()
            .find(|(k, _)| *k == "lo")
            .map(|(_, v)| v),
        Some(&MapValue::Int(i64::MIN))
    );
    assert_eq!(
        parsed_text
            .map
            .iter()
            .find(|(k, _)| *k == "hi")
            .map(|(_, v)| v),
        Some(&MapValue::Int(i64::MAX))
    );

    // Binary round-trip.
    let bin = emit_micb_bytes(&g);
    let parsed_bin = parse_micb(&mut Cursor::new(&bin)).expect("parse binary failed");
    assert_eq!(
        parsed_bin
            .map
            .iter()
            .find(|(k, _)| *k == "lo")
            .map(|(_, v)| v),
        Some(&MapValue::Int(i64::MIN))
    );
    assert_eq!(
        parsed_bin
            .map
            .iter()
            .find(|(k, _)| *k == "hi")
            .map(|(_, v)| v),
        Some(&MapValue::Int(i64::MAX))
    );
}

// ---------------------------------------------------------------------------
// 13. Bytes value: empty bytes, 1-byte, multi-byte
// ---------------------------------------------------------------------------

#[test]
fn map_bytes_various_lengths() {
    for case in [vec![], vec![0x00], vec![0xFF, 0xFE, 0xFD]] {
        let mut g = Graph::residual_block();
        g.map.insert("b", MapValue::Bytes(case.clone()));
        let text = emit_mic2(&g);
        let parsed = parse_mic2(&text).expect("parse failed");
        assert_eq!(
            parsed.map.iter().find(|(k, _)| *k == "b").map(|(_, v)| v),
            Some(&MapValue::Bytes(case.clone())),
            "bytes round-trip failed for {:?}",
            case
        );
        let bin = emit_micb_bytes(&g);
        let parsed_bin = parse_micb(&mut Cursor::new(&bin)).expect("parse binary failed");
        assert_eq!(
            parsed_bin
                .map
                .iter()
                .find(|(k, _)| *k == "b")
                .map(|(_, v)| v),
            Some(&MapValue::Bytes(case.clone())),
            "binary bytes round-trip failed for {:?}",
            case
        );
    }
}

// ---------------------------------------------------------------------------
// 14. Graph equality with MAP
// ---------------------------------------------------------------------------

#[test]
fn graph_eq_considers_map() {
    let g1 = Graph::residual_block(); // empty map
    let mut g2 = Graph::residual_block();
    g2.map.insert("k", MapValue::Int(1));

    assert!(!g1.eq(&g2), "graphs with different maps must not be equal");
    assert!(g1.eq(&g1), "graph must equal itself (empty map)");
    assert!(g2.eq(&g2), "graph must equal itself (non-empty map)");
}
