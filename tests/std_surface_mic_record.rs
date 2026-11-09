// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/mic_record.mind` + `std/mic_results.mind`: a MIND program writes and
//! reads its results as mic@3 MAP records.
//!
//! Verifies:
//!  1. both modules parse and lower with their public API;
//!  2. the reference encoder (the compiler's own mic@3 body, ULEB128 and
//!     zigzag) produces the documented wire bytes, and the compiler's readers
//!     accept them;
//!  3. (MLIR) a realistic results record written by the MIND writer is
//!     byte-equal to the reference, canonical for `mic3_canonical_check`, and
//!     decodes back to every field; re-encoding gives the same bytes;
//!  4. (MLIR) the bytes are independent of insertion order and repeat exactly;
//!  5. (MLIR) the writer and the results helpers refuse invalid input with the
//!     named error and leave the record unchanged;
//!  6. (MLIR) `import std.mic_record` / `std.mic_results` link through the
//!     bundled stdlib and store a record through std.fs.
//!
//! Refusals of malformed input are in `std_surface_mic_record_refusals.rs`.
//!
//! Gate: `cargo test --no-default-features --features
//!        "mlir-build std-surface cross-module-imports" --test std_surface_mic_record`

#![cfg(feature = "std-surface")]

mod common;

use common::mic_record::{Val, reference};
use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
use libmind::ir::compact::v3::mic3_app_metadata;
use libmind::ir::compact::v3::{mic3_canonical_check, parse_mic3_envelope};
use libmind::parser;

fn fn_names(src: &str) -> Vec<String> {
    let module = parser::parse(src).expect("module must parse");
    let ir = lower_to_ir(&module).expect("module must lower");
    ir.instrs
        .iter()
        .filter_map(|i| match i {
            Instr::FnDef { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn both_modules_parse_and_lower_with_their_api() {
    let codec = fn_names(include_str!("../std/mic_record.mind"));
    for f in [
        "mrec_new",
        "mrec_put_int",
        "mrec_put_str",
        "mrec_put_bytes",
        "mrec_remove",
        "mrec_encode",
        "mrec_decode",
        "mrec_find",
        "mrec_get_int",
        "mrec_get_val",
    ] {
        assert!(
            codec.iter().any(|n| n == f),
            "std.mic_record must define {f}"
        );
    }
    let results = fn_names(&common::mic_record::combined_source());
    for f in [
        "mres_new",
        "mres_set_score",
        "mres_get_score",
        "mres_set_level_actions",
        "mres_get_level_actions",
    ] {
        assert!(
            results.iter().any(|n| n == f),
            "std.mic_results must define {f}"
        );
    }
}

#[test]
fn the_reference_encoding_is_the_documented_wire_layout() {
    let got = reference(&[
        ("app.s", Val::Str("hi")),
        ("app.n", Val::Int(-1)),
        ("app.b", Val::Bytes(&[0xAB])),
    ]);
    let want: Vec<u8> = [
        &b"MIC3"[..],
        &[0x02, 0, 0, 0, 0, 0, 0, 0, 0x4D, 0x03],
        &[0x05],
        b"app.b",
        &[0x02, 0x01, 0xAB, 0x05],
        b"app.n",
        &[0x01, 0x01, 0x05],
        b"app.s",
        &[0x00, 0x02],
        b"hi",
    ]
    .concat();
    assert_eq!(
        got, want,
        "documented layout: body, sentinel, sorted entries"
    );
    mic3_canonical_check(&got).expect("the compiler finds the layout canonical");
    let module = parse_mic3_envelope(&got).expect("the compiler decodes the layout");
    assert!(
        module.instrs.is_empty(),
        "a record carries the empty module"
    );
}

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod functional {
    use super::*;
    use common::mic_record::{Lib, bytes_at, load, ptr};
    use std::sync::OnceLock;

    const TARGET: &str = "std_surface_mic_record";

    const SCORECARD: &str = "8f2c1d7e-3b4a-4c5d-9e6f-a1b2c3d4e5f6";
    const WEIGHTS: [u8; 32] = [
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55,
    ];
    /// (game id, total actions, per-level (level, actions)).
    type Game = (&'static str, i64, &'static [(i64, i64)]);
    /// Level 10 sorts before level 2.
    const GAMES: [Game; 2] = [
        ("ls20-cb3b57cc", 214, &[(0, 17), (1, 42), (2, 9), (10, 146)]),
        ("ft09-b8377d4b", 88, &[(0, 31), (1, 57)]),
    ];

    fn game_key(gid: &str, level: Option<i64>) -> String {
        match level {
            Some(l) => format!("results.game.{gid}.level.{l}.actions"),
            None => format!("results.game.{gid}.actions"),
        }
    }

    /// The realistic record as reference entries (keys owned by the caller).
    fn realistic_keys() -> Vec<(String, i64)> {
        let mut out = Vec::new();
        for (gid, total, levels) in GAMES {
            out.push((game_key(gid, None), total));
            for &(l, a) in levels {
                out.push((game_key(gid, Some(l)), a));
            }
        }
        out
    }

    fn realistic_reference() -> Vec<u8> {
        let keys = realistic_keys();
        let mut entries: Vec<(&str, Val<'_>)> = vec![
            ("results.schema", Val::Int(1)),
            ("results.scorecard_id", Val::Str(SCORECARD)),
            ("results.score", Val::Int(121)),
            ("results.score_scale", Val::Int(10000)),
            ("results.weights_sha256", Val::Bytes(&WEIGHTS)),
        ];
        entries.extend(keys.iter().map(|(k, v)| (k.as_str(), Val::Int(*v))));
        reference(&entries)
    }

    fn lib() -> Option<&'static Lib> {
        static LIB: OnceLock<Option<Lib>> = OnceLock::new();
        LIB.get_or_init(|| load(TARGET, &common::mic_record::combined_source()))
            .as_ref()
    }

    /// Write the realistic record through the results helpers, in source
    /// order or reversed.
    fn write_realistic(lib: &Lib, reversed: bool) -> i64 {
        let h = lib.call("mres_new", &[]);
        assert!(h > 0, "mres_new must return a handle");
        let mut steps: Vec<Box<dyn Fn() -> i64 + '_>> = vec![
            Box::new(move || {
                let s = SCORECARD.as_bytes();
                lib.call("mres_set_scorecard", &[h, ptr(s), s.len() as i64])
            }),
            Box::new(move || lib.call("mres_set_score", &[h, 121, 10000])),
            Box::new(move || lib.call("mres_set_weights_sha256", &[h, ptr(&WEIGHTS)])),
        ];
        for (gid, total, levels) in GAMES {
            let g = gid.as_bytes();
            steps.push(Box::new(move || {
                lib.call("mres_set_game_actions", &[h, ptr(g), g.len() as i64, total])
            }));
            for &(l, a) in levels {
                steps.push(Box::new(move || {
                    let args = [h, ptr(g), g.len() as i64, l, a];
                    lib.call("mres_set_level_actions", &args)
                }));
            }
        }
        if reversed {
            steps.reverse();
        }
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(step(), 0, "results write {i} must succeed");
        }
        h
    }

    #[test]
    fn a_realistic_results_record_is_the_reference_mic3_artifact() {
        let Some(lib) = lib() else { return };
        let bytes = lib.encode(write_realistic(lib, false));
        assert_eq!(
            bytes,
            realistic_reference(),
            "MIND writer == reference encoder"
        );
        mic3_canonical_check(&bytes).expect("the compiler finds the record canonical");
        let module = parse_mic3_envelope(&bytes).expect("the compiler decodes the record");
        assert!(module.instrs.is_empty());
        let app = mic3_app_metadata(&bytes).expect("the compiler reads the MAP");
        assert_eq!(
            app,
            vec![("results.scorecard_id".to_string(), SCORECARD.to_string())],
            "the compiler surfaces the String entry"
        );
    }

    #[test]
    fn decoding_returns_every_field_and_reencodes_the_same_bytes() {
        let Some(lib) = lib() else { return };
        let bytes = lib.encode(write_realistic(lib, false));
        let r = lib.decode(&bytes);
        assert!(r > 0, "decode refused its own record: {r}");
        let mut out = [0i64; 2];
        let o = out.as_mut_ptr() as i64;
        assert_eq!(lib.call("mres_get_schema", &[r, o]), 0);
        assert_eq!(out[0], 1);
        assert_eq!(lib.call("mres_get_scorecard", &[r, o]), 0);
        assert_eq!(bytes_at(out[0], out[1]), SCORECARD.as_bytes());
        assert_eq!(lib.call("mres_get_score", &[r, o]), 0);
        assert_eq!(out, [121, 10000]);
        assert_eq!(lib.call("mres_get_weights_sha256", &[r, o]), 0);
        assert_eq!(bytes_at(out[0], out[1]), WEIGHTS);
        for (gid, total, levels) in GAMES {
            let g = gid.as_bytes();
            let args = [r, ptr(g), g.len() as i64, o];
            assert_eq!(lib.call("mres_get_game_actions", &args), 0);
            assert_eq!(out[0], total, "{gid} total");
            for &(l, a) in levels {
                let args = [r, ptr(g), g.len() as i64, l, o];
                assert_eq!(lib.call("mres_get_level_actions", &args), 0);
                assert_eq!(out[0], a, "{gid} level {l}");
            }
        }
        let n = lib.count(r);
        assert_eq!(n as usize, 5 + realistic_keys().len());
        let mut keys = Vec::new();
        for i in 0..n {
            let ka = lib.call("mrec_key_addr_at", &[r, i]);
            keys.push(bytes_at(ka, lib.call("mrec_key_len_at", &[r, i])));
        }
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "entries iterate in key-byte order");
        assert_eq!(lib.encode(r), bytes, "decode -> encode is the identity");
    }

    #[test]
    fn the_bytes_repeat_exactly_and_ignore_insertion_order() {
        let Some(lib) = lib() else { return };
        let h = write_realistic(lib, false);
        let first = lib.encode(h);
        assert_eq!(lib.encode(h), first, "a second encode of one record");
        let other = lib.encode(write_realistic(lib, true));
        assert_eq!(other, first, "reversed insertion order");
        assert_eq!(first, realistic_reference(), "a fixed byte image");
        let kat = lib.new_record();
        assert_eq!(lib.put_str(kat, b"app.s", b"hi"), 0);
        assert_eq!(lib.put_int(kat, b"app.n", -1), 0);
        assert_eq!(lib.put_bytes(kat, b"app.b", &[0xAB]), 0);
        let want = reference(&[
            ("app.b", Val::Bytes(&[0xAB])),
            ("app.n", Val::Int(-1)),
            ("app.s", Val::Str("hi")),
        ]);
        assert_eq!(
            lib.encode(kat),
            want,
            "the documented layout from the writer"
        );
    }

    #[test]
    fn int_values_round_trip_at_the_zigzag_edges() {
        let Some(lib) = lib() else { return };
        let values = [
            i64::MIN,
            i64::MIN + 1,
            -8193,
            -8192,
            -65,
            -64,
            -1,
            0,
            1,
            63,
            64,
            8191,
            8192,
            i64::MAX,
        ];
        let keys: Vec<String> = (0..values.len()).map(|i| format!("int.v{i:02}")).collect();
        let h = lib.new_record();
        for (k, v) in keys.iter().zip(values) {
            assert_eq!(lib.put_int(h, k.as_bytes(), v), 0);
        }
        let entries: Vec<(&str, Val<'_>)> = keys
            .iter()
            .zip(values)
            .map(|(k, v)| (k.as_str(), Val::Int(v)))
            .collect();
        let bytes = lib.encode(h);
        assert_eq!(bytes, reference(&entries));
        let r = lib.decode(&bytes);
        assert!(r > 0, "decode: {r}");
        for (k, v) in keys.iter().zip(values) {
            assert_eq!(lib.get_int(r, k.as_bytes()), (0, v), "{k}");
        }
    }

    #[test]
    fn the_writer_refuses_invalid_entries_with_the_named_error() {
        let Some(lib) = lib() else { return };
        let e = |n: &str| lib.err(n);
        let h = lib.new_record();
        assert_eq!(lib.put_int(0, b"a.b", 1), e("handle"));
        assert_eq!(lib.put_int(h, b"", 1), e("key"));
        assert_eq!(lib.put_int(h, b"nodot", 1), e("key"));
        assert_eq!(lib.put_int(h, b"a.\xFF", 1), e("utf8"));
        assert_eq!(lib.put_int(h, b"evidence_chain.x", 1), e("reserved"));
        assert_eq!(lib.put_int(h, b"signature.x", 1), e("reserved"));
        // The bare prefixes too: the compiler's MAP reader refuses them.
        assert_eq!(lib.put_int(h, b"evidence_chain.", 1), e("reserved"));
        assert_eq!(lib.put_int(h, b"signature.", 1), e("reserved"));
        let long = [b"a.".as_slice(), &[b'k'; 255]].concat();
        assert_eq!(lib.put_int(h, &long, 1), e("oversized"), "257-byte key");
        let neg = [h, ptr(b"a.b"), -1, 1];
        assert_eq!(
            lib.call("mrec_put_int", &neg),
            e("value"),
            "negative key length"
        );
        let neg = [h, ptr(b"a.b"), 3, ptr(b"x"), -1];
        assert_eq!(
            lib.call("mrec_put_bytes", &neg),
            e("value"),
            "negative value length"
        );
        let huge = [h, ptr(b"a.b"), 3, ptr(b"x"), i64::MAX];
        assert_eq!(
            lib.call("mrec_put_bytes", &huge),
            e("oversized"),
            "i64::MAX value"
        );
        for bad in [
            &b"\xC0\x80"[..],
            b"\xED\xA0\x80",
            b"\xF4\x90\x80\x80",
            b"\xE2\x82",
            b"\x80",
            b"\xF5\x80\x80\x80",
            b"\xE0\x9F\xBF",
            b"\xF0\x8F\xBF\xBF",
            b"\xE2\x82\x41",
            b"\xF0\x90\x80\x41",
        ] {
            assert_eq!(lib.put_str(h, b"s.bad", bad), e("utf8"), "{bad:02X?}");
        }
        // A sequence cut by the length, though the next byte in memory would
        // complete it ("\u{20AC}").
        let euro = b"\xE2\x82\xAC";
        let cut = [h, ptr(b"s.bad"), 5, ptr(euro), 2];
        assert_eq!(lib.call("mrec_put_str", &cut), e("utf8"), "cut sequence");
        assert_eq!(lib.count(h), 0, "no refused entry was stored");
        // Positive controls: the same shapes that sit just inside each rule.
        assert_eq!(lib.put_int(h, &long[..256], 1), 0, "256-byte key");
        assert_eq!(lib.put_int(h, b"signatures.x", 1), 0);
        assert_eq!(lib.put_int(h, b"evidence_chainx.y", 1), 0);
        for (i, ok) in [
            &b"\xC2\x80"[..],
            b"\xE0\xA0\x80",
            b"\xED\x9F\xBF",
            b"\xF0\x90\x80\x80",
            b"\xF4\x8F\xBF\xBF",
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(lib.put_str(h, format!("s.ok{i}").as_bytes(), ok), 0);
        }
        assert_eq!(
            lib.put_bytes(h, b"b.raw", b"\xFF\xC0"),
            0,
            "Bytes are not UTF-8"
        );
        assert_eq!(lib.put_int(h, b"s.ok0", 9), e("duplicate"));
        assert_eq!(lib.count(h), 9);
    }

    #[test]
    fn the_writer_enforces_the_entry_and_size_limits_exactly() {
        let Some(lib) = lib() else { return };
        let max_entries = lib.call("mrec_max_entries", &[]);
        let h = lib.new_record();
        for i in 0..max_entries {
            assert_eq!(lib.put_int(h, format!("k.{i:05}").as_bytes(), i), 0);
        }
        assert_eq!(lib.put_int(h, b"k.over", 0), lib.err("oversized"));
        assert_eq!(lib.count(h), max_entries);

        // total = body 12 + sentinel 1 + count 1 + key len 1 + key 5 + tag 1
        // + 4-byte length + value
        let max = lib.call("mrec_max_record_bytes", &[]) as usize;
        let value = vec![0x5Au8; max - 25];
        let fits = lib.new_record();
        assert_eq!(lib.put_bytes(fits, b"big.v", &value), 0);
        let bytes = lib.encode(fits);
        assert_eq!(bytes.len(), max, "a record of exactly the limit");
        assert!(lib.decode(&bytes) > 0, "the reader takes exactly the limit");
        let over = lib.new_record();
        let value = vec![0x5Au8; max - 24];
        assert_eq!(lib.put_bytes(over, b"big.v", &value), lib.err("oversized"));
        assert_eq!(lib.count(over), 0);
    }

    #[test]
    fn remove_and_lookup_keep_the_table_canonical() {
        let Some(lib) = lib() else { return };
        let h = lib.new_record();
        assert_eq!(lib.put_int(h, b"k.a", 1), 0);
        assert_eq!(lib.put_str(h, b"k.b", b"two"), 0);
        assert_eq!(lib.put_bytes(h, b"k.c", b"\x03"), 0);
        assert_eq!(lib.call("mrec_remove", &[h, ptr(b"k.b"), 3]), 0);
        let rest = [("k.a", Val::Int(1)), ("k.c", Val::Bytes(b"\x03"))];
        assert_eq!(lib.encode(h), reference(&rest), "remove updates the size");
        assert_eq!(
            lib.call("mrec_remove", &[h, ptr(b"k.b"), 3]),
            lib.err("not_found")
        );
        assert_eq!(lib.find(h, b"k.c"), 1);
        assert_eq!(lib.find(h, b"k.z"), lib.err("not_found"));
        assert_eq!(lib.get_int(h, b"k.c").0, lib.err("type"));
        let mut out = [0i64; 2];
        let args = [h, ptr(b"k.a"), 3, 2, out.as_mut_ptr() as i64];
        assert_eq!(lib.call("mrec_get_val", &args), lib.err("type"));
        let args = [h, ptr(b"k.c"), 3, 2, out.as_mut_ptr() as i64];
        assert_eq!(lib.call("mrec_get_val", &args), 0);
        assert_eq!(bytes_at(out[0], out[1]), b"\x03");
        assert_eq!(lib.call("mrec_tag_at", &[h, 2]), lib.err("not_found"));
        assert_eq!(lib.call("mrec_tag_at", &[h, -1]), lib.err("not_found"));
        assert_eq!(
            lib.call("mrec_int_at", &[h, 1]),
            0,
            "a Bytes entry has no Int"
        );
        assert_eq!(lib.call("mrec_int_at", &[h, 0]), 1);
        assert_eq!(
            lib.call("mrec_int_at", &[h, 9]),
            0,
            "out of range has no Int"
        );
        let e = |n: &str| lib.err(n);
        assert_eq!(lib.call("mrec_find", &[h, ptr(b"k.a"), -1]), e("value"));
        assert_eq!(lib.call("mrec_find", &[0, ptr(b"k.a"), 3]), e("handle"));
        assert_eq!(lib.call("mrec_tag_at", &[0, 0]), e("handle"));
        assert_eq!(lib.call("mrec_encode", &[0]), 0, "no buffer for a 0 handle");
        assert_eq!(lib.count(0), e("handle"));
    }

    #[test]
    fn the_results_helpers_refuse_invalid_arguments_and_write_nothing() {
        let Some(lib) = lib() else { return };
        let e = |n: &str| lib.err(n);
        let h = lib.call("mres_new", &[]);
        let g = b"ls20".as_slice();
        let set_level = |gid: &[u8], len: i64, l: i64, a: i64| {
            lib.call("mres_set_level_actions", &[h, ptr(gid), len, l, a])
        };
        assert_eq!(set_level(b"ls.20", 5, 0, 1), e("value"), "'.' in a game id");
        assert_eq!(set_level(g, 0, 0, 1), e("value"), "empty game id");
        assert_eq!(
            set_level(g, i64::MAX, 0, 1),
            e("oversized"),
            "game id length"
        );
        assert_eq!(set_level(g, 4, -1, 1), e("value"), "negative level");
        assert_eq!(set_level(g, 4, 0, -1), e("value"), "negative actions");
        let args = [h, ptr(g), 4, -1];
        assert_eq!(lib.call("mres_set_game_actions", &args), e("value"));
        assert_eq!(
            lib.call("mres_set_score", &[h, 1, 0]),
            e("value"),
            "scale 0"
        );
        assert_eq!(
            lib.call("mres_set_score", &[h, 1, -5]),
            e("value"),
            "scale < 0"
        );
        assert_eq!(lib.count(h), 1, "only results.schema");
        let mut out = [0i64; 2];
        let o = out.as_mut_ptr() as i64;
        let args = [h, ptr(g), 4, -1, o];
        assert_eq!(lib.call("mres_get_level_actions", &args), e("value"));
        assert_eq!(lib.call("mres_get_score", &[h, o]), e("not_found"));

        // Both score keys or neither, when the second write is refused.
        assert_eq!(lib.put_int(h, b"results.score_scale", 7), 0);
        assert_eq!(lib.call("mres_set_score", &[h, 3, 4]), e("duplicate"));
        assert_eq!(lib.find(h, b"results.score"), e("not_found"), "rolled back");
        let full = lib.call("mres_new", &[]);
        for i in 1..lib.call("mrec_max_entries", &[]) - 1 {
            assert_eq!(lib.put_int(full, format!("k.{i:05}").as_bytes(), i), 0);
        }
        assert_eq!(lib.call("mres_set_score", &[full, 3, 4]), e("oversized"));
        assert_eq!(
            lib.find(full, b"results.score"),
            e("not_found"),
            "rolled back"
        );
        // ... and when the first write is refused: the old score stays and no
        // scale is stored next to it.
        let s = lib.new_record();
        assert_eq!(lib.put_int(s, b"results.score", 9), 0);
        assert_eq!(lib.call("mres_set_score", &[s, 3, 4]), e("duplicate"));
        assert_eq!(
            lib.find(s, b"results.score_scale"),
            e("not_found"),
            "no scale written"
        );
        assert_eq!(lib.get_int(s, b"results.score"), (0, 9), "old score kept");

        // Stored values a reader must not trust.
        let r = lib.new_record();
        assert_eq!(lib.put_int(r, b"results.score", 5), 0);
        assert_eq!(lib.put_int(r, b"results.score_scale", 0), 0);
        assert_eq!(
            lib.call("mres_get_score", &[r, o]),
            e("value"),
            "stored scale 0"
        );
        assert_eq!(lib.put_bytes(r, b"results.weights_sha256", &[0; 31]), 0);
        assert_eq!(lib.call("mres_get_weights_sha256", &[r, o]), e("value"));
        assert_eq!(lib.call("mres_get_schema", &[r, o]), e("not_found"));

        // A scale without a score is not a score.
        let c = lib.new_record();
        assert_eq!(lib.put_int(c, b"results.score_scale", 4), 0);
        let mut score = [-1i64; 2];
        let so = score.as_mut_ptr() as i64;
        assert_eq!(lib.call("mres_get_score", &[c, so]), e("not_found"));
        assert_eq!(lib.put_int(c, b"results.score", 3), 0);
        assert_eq!(lib.call("mres_get_score", &[c, so]), 0, "positive control");
        assert_eq!(score, [3, 4]);

        // A score without a scale is not a score either. The missing scale
        // must be refused, never passed back from the caller's stale out + 8.
        let m = lib.new_record();
        assert_eq!(lib.put_int(m, b"results.score", 5), 0);
        let mut stale = [0i64, 7];
        let mo = stale.as_mut_ptr() as i64;
        assert_eq!(
            lib.call("mres_get_score", &[m, mo]),
            e("not_found"),
            "score without a scale"
        );

        // No digest stored: not_found, even when out + 8 already holds 32
        // from an earlier call.
        let w = lib.new_record();
        let mut digest = [0i64, 32];
        let wo = digest.as_mut_ptr() as i64;
        assert_eq!(
            lib.call("mres_get_weights_sha256", &[w, wo]),
            e("not_found")
        );
        assert_eq!(digest, [0, 32], "nothing written");
        let sha = [0xA5u8; 32];
        assert_eq!(lib.put_bytes(w, b"results.weights_sha256", &sha), 0);
        assert_eq!(
            lib.call("mres_get_weights_sha256", &[w, wo]),
            0,
            "positive control"
        );
        assert_eq!(bytes_at(digest[0], digest[1]), sha);
    }

    /// The real `import` path: the bundled modules, linked through the
    /// substrate archive, store a record through std.fs and load it back.
    #[test]
    fn imported_modules_store_and_load_a_record_through_std_fs() {
        let driver = "import std.fs;\nimport std.mic_record;\nimport std.mic_results;\n\n\
             pub fn smoke_write(path: i64, gid: i64, gn: i64) -> i64 {\n\
             \x20   let h: i64 = mic_results.mres_new();\n\
             \x20   let _ = mic_results.mres_set_score(h, 121, 10000);\n\
             \x20   let _ = mic_results.mres_set_level_actions(h, gid, gn, 0, 17);\n\
             \x20   let b: i64 = mic_record.mrec_encode(h);\n\
             \x20   fs.write_string(path, mic_record.mrec_buf_addr(b), mic_record.mrec_buf_len(b))\n\
             }\n\n\
             pub fn smoke_read(path: i64, gid: i64, gn: i64) -> i64 {\n\
             \x20   let s: i64 = fs.read_to_string(path);\n\
             \x20   let r: i64 = mic_record.mrec_decode(__mind_load_i64(s + 0), __mind_load_i64(s + 8));\n\
             \x20   if r < 0 {\n\
             \x20       return r;\n\
             \x20   }\n\
             \x20   let out: i64 = __mind_alloc(8);\n\
             \x20   let rc: i64 = mic_results.mres_get_level_actions(r, gid, gn, 0, out);\n\
             \x20   if rc != 0 {\n\
             \x20       return rc;\n\
             \x20   }\n\
             \x20   __mind_load_i64(out)\n\
             }\n";
        let target = "std_surface_mic_record_import";
        let Some(lib) = load(target, driver) else {
            return;
        };
        let dir = common::scratch_dir(target);
        let g = b"ls20".as_slice();
        let path = |name: &str| {
            let p = dir.join(name);
            std::ffi::CString::new(p.to_str().unwrap()).unwrap()
        };
        let (a, b) = (path("a.mic3"), path("b.mic3"));
        for p in [&a, &b] {
            let args = [p.as_ptr() as i64, ptr(g), 4];
            assert_eq!(lib.call("smoke_write", &args), 1, "fs.write_string");
        }
        let fa = std::fs::read(dir.join("a.mic3")).unwrap();
        assert_eq!(fa, std::fs::read(dir.join("b.mic3")).unwrap(), "two writes");
        let want = reference(&[
            ("results.game.ls20.level.0.actions", Val::Int(17)),
            ("results.schema", Val::Int(1)),
            ("results.score", Val::Int(121)),
            ("results.score_scale", Val::Int(10000)),
        ]);
        assert_eq!(fa, want, "the stored file is the reference record");
        let args = [a.as_ptr() as i64, ptr(g), 4];
        assert_eq!(lib.call("smoke_read", &args), 17);
        std::fs::write(dir.join("a.mic3"), &fa[..fa.len() - 1]).unwrap();
        let truncated = lib.call("smoke_read", &args);
        let codec = self::lib().expect("the codec gate built");
        assert_eq!(truncated, codec.err("truncated"), "a cut file");
    }
}
