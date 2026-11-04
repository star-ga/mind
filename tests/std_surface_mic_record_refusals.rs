// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/mic_record.mind` reader refusals: every malformed record is refused
//! with its named error, and no declared length is trusted before it is
//! checked against the fixed limits and the bytes that remain.
//!
//! The inputs are built from the reference encoder in `common::mic_record`,
//! then cut or patched, so each case differs from a valid record in one
//! place.
//!
//! Gate: `cargo test --no-default-features --features
//!        "mlir-build std-surface cross-module-imports" --test std_surface_mic_record_refusals`

#![cfg(all(
    feature = "std-surface",
    feature = "mlir-build",
    feature = "cross-module-imports"
))]

mod common;

use common::mic_record::{Lib, Val, head, load, reference, uleb};
use std::sync::OnceLock;

const TARGET: &str = "std_surface_mic_record_refusals";

fn lib() -> Option<&'static Lib> {
    static LIB: OnceLock<Option<Lib>> = OnceLock::new();
    LIB.get_or_init(|| load(TARGET, &common::mic_record::combined_source()))
        .as_ref()
}

/// A valid two-entry record: a String and an Int.
fn valid() -> Vec<u8> {
    reference(&[("app.name", Val::Str("run")), ("app.score", Val::Int(300))])
}

/// One entry with a raw key and a raw tagged value, after a count of 1.
fn one_entry(key: &[u8], tag_and_value: &[u8]) -> Vec<u8> {
    [
        head(&[1]),
        uleb(key.len() as u64),
        key.to_vec(),
        tag_and_value.to_vec(),
    ]
    .concat()
}

#[test]
fn the_valid_baseline_decodes() {
    let Some(lib) = lib() else { return };
    let r = lib.decode(&valid());
    assert!(r > 0, "the positive control must decode: {r}");
    assert_eq!(lib.count(r), 2);
    assert_eq!(lib.get_int(r, b"app.score"), (0, 300));
}

#[test]
fn every_truncation_of_a_record_is_refused() {
    let Some(lib) = lib() else { return };
    let rec = valid();
    for n in 0..rec.len() {
        let rc = lib.decode(&rec[..n]);
        assert!(rc < 0, "a {n}-byte prefix of {} was accepted", rec.len());
        let want = if n == 12 { "no_map" } else { "truncated" };
        assert_eq!(rc, lib.err(want), "prefix of {n} bytes");
    }
}

#[test]
fn oversized_counts_and_lengths_are_refused_before_any_allocation() {
    let Some(lib) = lib() else { return };
    let e = |n: &str| lib.err(n);
    // A count with bit 63 set: ten ULEB bytes, the last 0x01.
    let bit63 = [0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01];
    assert_eq!(lib.decode(&head(&bit63)), e("oversized"), "count 2^63");
    // A count far above the entries present runs out of input.
    assert_eq!(
        lib.decode(&head(&uleb(1 << 40))),
        e("truncated"),
        "count 2^40"
    );
    let mut over = head(&uleb(4097));
    for i in 0..4097 {
        over.extend(uleb(6));
        over.extend(format!("k.{i:04}").as_bytes());
        over.extend([0x01, 0x00]);
    }
    assert_eq!(lib.decode(&over), e("oversized"), "4097 entries");
    // Key lengths: over the 256-byte limit, then inside it but past the input.
    let long_key = [head(&[1]), uleb(300), vec![b'k'; 300], vec![0x01, 0x00]].concat();
    assert_eq!(lib.decode(&long_key), e("oversized"), "300-byte key");
    let short = [head(&[1]), uleb(300), b"a.b".to_vec()].concat();
    assert_eq!(
        lib.decode(&short),
        e("oversized"),
        "300-byte key, 3 bytes left"
    );
    let short = [head(&[1]), uleb(200), b"a.b".to_vec()].concat();
    assert_eq!(
        lib.decode(&short),
        e("truncated"),
        "200-byte key, 3 bytes left"
    );
    // Value lengths: 2^62, bit 63, 11 MiB, and 1 MiB with 2 bytes present.
    let s62 = one_entry(b"a.b", &[&[0x00][..], &uleb(1 << 62)].concat());
    assert_eq!(lib.decode(&s62), e("oversized"), "2^62-byte String");
    let b63 = one_entry(b"a.b", &[&[0x02][..], &bit63].concat());
    assert_eq!(lib.decode(&b63), e("oversized"), "2^63-byte Bytes");
    let b11 = one_entry(b"a.b", &[&[0x02][..], &uleb(11 << 20), b"xy"].concat());
    assert_eq!(lib.decode(&b11), e("oversized"), "11 MiB Bytes");
    let b1 = one_entry(b"a.b", &[&[0x02][..], &uleb(1 << 20), b"xy"].concat());
    assert_eq!(lib.decode(&b1), e("truncated"), "1 MiB Bytes, 2 bytes left");
    // The whole input over the record limit, checked before it is read.
    let max = lib.call("mrec_max_record_bytes", &[]) as usize;
    let mut big = valid();
    big.resize(max + 1, 0);
    assert_eq!(lib.decode(&big), e("oversized"), "input of limit + 1 bytes");
    let neg = [big.as_ptr() as i64, -1];
    assert_eq!(lib.call("mrec_decode", &neg), e("value"), "negative length");
}

#[test]
fn a_wrong_version_magic_or_body_is_refused() {
    let Some(lib) = lib() else { return };
    let e = |n: &str| lib.err(n);
    for version in [0x00u8, 0x01, 0x03, 0x04, 0xFF] {
        let mut rec = valid();
        rec[4] = version;
        assert_eq!(lib.decode(&rec), e("version"), "version 0x{version:02X}");
    }
    let mut rec = valid();
    rec[3] = b'2';
    assert_eq!(lib.decode(&rec), e("magic"), "MIC2");
    for at in 5..12 {
        let mut rec = valid();
        rec[at] = 0x01;
        assert_eq!(lib.decode(&rec), e("body"), "non-empty module at byte {at}");
    }
    let mut rec = valid();
    rec[12] = 0x00;
    assert_eq!(lib.decode(&rec), e("no_map"), "no sentinel");
    // A short length is honoured even when the bytes after it are readable.
    let after = b"MIC3\x03";
    assert_eq!(
        lib.call("mrec_decode", &[after.as_ptr() as i64, 4]),
        e("truncated")
    );
    let mut rec = valid();
    rec[9] = 0x01;
    assert_eq!(
        lib.call("mrec_decode", &[rec.as_ptr() as i64, 8]),
        e("truncated")
    );
}

#[test]
fn non_canonical_maps_are_refused() {
    let Some(lib) = lib() else { return };
    let e = |n: &str| lib.err(n);
    let int0 = [0x01u8, 0x00];
    // Entry order and duplicates.
    let unsorted = [
        head(&[2]),
        uleb(3),
        b"b.k".to_vec(),
        int0.to_vec(),
        uleb(3),
        b"a.k".to_vec(),
        int0.to_vec(),
    ]
    .concat();
    assert_eq!(lib.decode(&unsorted), e("order"));
    let entry = [uleb(3), b"a.k".to_vec(), int0.to_vec()].concat();
    let dup = [head(&[2]), entry.clone(), entry].concat();
    assert_eq!(lib.decode(&dup), e("duplicate"));
    // ULEB128 forms: a padded count, a padded Int, 11 bytes, bit 64.
    assert_eq!(
        lib.decode(&head(&[0x80, 0x00])),
        e("varint"),
        "padded count 0"
    );
    let padded = one_entry(b"a.b", &[0x01, 0x81, 0x00]);
    assert_eq!(lib.decode(&padded), e("varint"), "padded Int");
    let eleven = [&[0x01u8][..], &[0x80; 10], &[0x00]].concat();
    assert_eq!(
        lib.decode(&one_entry(b"a.b", &eleven)),
        e("varint"),
        "11 bytes"
    );
    let bit64 = [&[0x01u8][..], &[0xFF; 9], &[0x02]].concat();
    assert_eq!(
        lib.decode(&one_entry(b"a.b", &bit64)),
        e("varint"),
        "past 64 bits"
    );
    let max = [&[0x01u8][..], &[0xFF; 9], &[0x01]].concat();
    let r = lib.decode(&one_entry(b"a.b", &max));
    assert!(
        r > 0,
        "the largest ULEB128 (zigzag of i64::MIN) is valid: {r}"
    );
    assert_eq!(lib.get_int(r, b"a.b"), (0, i64::MIN));
    // Tags, keys and UTF-8.
    assert_eq!(lib.decode(&one_entry(b"a.b", &[0x03, 0x00])), e("tag"));
    let tagged = one_entry(b"a.b", &[0x03]);
    let no_tag = [tagged.as_ptr() as i64, tagged.len() as i64 - 1];
    assert_eq!(
        lib.call("mrec_decode", &no_tag),
        e("truncated"),
        "no tag byte"
    );
    assert_eq!(lib.decode(&one_entry(b"a.\xFF", &int0)), e("utf8"), "key");
    let bad = one_entry(b"a.b", &[0x00, 0x03, 0xED, 0xA0, 0x80]);
    assert_eq!(lib.decode(&bad), e("utf8"), "surrogate in a String");
    let raw = one_entry(b"a.b", &[0x02, 0x03, 0xED, 0xA0, 0x80]);
    assert!(lib.decode(&raw) > 0, "the same bytes as Bytes are valid");
    assert_eq!(lib.decode(&one_entry(b"nodot", &int0)), e("key"));
    assert_eq!(lib.decode(&one_entry(b"", &int0)), e("key"), "empty key");
    assert_eq!(
        lib.decode(&one_entry(b"signature.ed", &int0)),
        e("reserved")
    );
    assert_eq!(
        lib.decode(&one_entry(b"evidence_chain.x", &int0)),
        e("reserved")
    );
    // Bytes after the last entry.
    let trailing = [valid(), vec![0x00]].concat();
    assert_eq!(lib.decode(&trailing), e("trailing"));
    let empty = head(&[0]);
    assert_eq!(lib.count(lib.decode(&empty)), 0, "an empty MAP is a record");
    assert_eq!(lib.decode(&[empty, vec![0x4D]].concat()), e("trailing"));
}
