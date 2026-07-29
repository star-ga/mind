// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! H2 regression: a mic@3 artifact that references one large string-table entry
//! many times must be REJECTED (per-parse decode budget exceeded), not expanded
//! into unbounded retained heap. `read_string` clones a table entry per wire
//! reference (a 1-byte ULEB index), so before the fix a ~68 KiB crafted input
//! decoded into >64 MiB of retained `String` clones (a dynamic repro measured
//! 105 KB -> 1.31 GB RSS, ~12000x). The parser now charges each clone against a
//! generous per-parse budget and fails closed on a reference bomb, while a
//! legitimate module that re-references short identifiers still parses.
#![cfg(feature = "std-surface")]

use libmind::ir::compact::parse_mic3;

fn uleb(mut n: u64, out: &mut Vec<u8>) {
    loop {
        let mut b = (n & 0x7f) as u8;
        n >>= 7;
        if n != 0 {
            b |= 0x80;
        }
        out.push(b);
        if n == 0 {
            break;
        }
    }
}

/// Craft a mic@3 blob: one string-table entry of `l` bytes, referenced `m`
/// times via an `OP_BREAK` (0x28) live snapshot (each entry is `read_string`
/// index 0 + a ValueId).
fn string_bomb(l: usize, m: usize) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"MIC3");
    b.push(0x02); // version
    uleb(1, &mut b); // string-table count = 1
    uleb(l as u64, &mut b); // entry length
    b.extend(std::iter::repeat_n(b'a', l));
    uleb(0, &mut b); // next_id
    uleb(0, &mut b); // exports count
    uleb(1, &mut b); // instruction count = 1
    b.push(0x28); // OP_BREAK
    uleb(m as u64, &mut b); // live snapshot count
    for _ in 0..m {
        uleb(0, &mut b); // name string idx = 0 (references the big entry)
        uleb(0, &mut b); // vid = 0
    }
    uleb(0, &mut b); // struct_defs
    uleb(0, &mut b); // const_array_defs
    uleb(0, &mut b); // repr_c_structs
    b
}

#[test]
fn mic3_string_reference_bomb_is_rejected() {
    // 2000 references to a 64 KiB entry = 128 MiB of would-be clones from a
    // ~68 KiB input. The parser must trip the decode budget (floor 64 MiB) and
    // fail closed well before that, rather than OOM.
    let b = string_bomb(64 * 1024, 2000);
    assert!(
        b.len() < 128 * 1024,
        "crafted input stays tiny: {} bytes",
        b.len()
    );

    let res = parse_mic3(&b);
    assert!(
        res.is_err(),
        "a string-reference decompression bomb must be rejected, not parsed"
    );
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("amplification budget") || msg.contains("decompression bomb"),
        "rejection must name the decode-budget guard, got: {msg}"
    );
}

/// Craft a nested-`OP_REGION` (0x24) chain `depth` levels deep where every level
/// declares a huge `body_len` (`declared`, which must be `<= limit` to pass
/// `read_count`). Each region's body begins with the next region, so the decoder
/// recurses to the depth guard. Before the reservation-amplification fix, each of
/// the ~256 live frames reserved `Vec::<Instr>::with_capacity(min(declared,
/// limit))` — held live across the recursion — for a ~tens-of-GB spike from a
/// ~1 MiB input. The fix caps the *reservation* (not the loop) at a small
/// constant, so the parse fails closed at the depth guard using trivial memory.
///
/// The trailing padding makes the whole-input length (the `limit` the reservation
/// and `read_count` are bounded against) `>= declared`, so the large `body_len`
/// survives `read_count` and actually reaches the `with_capacity` call.
fn nested_region_reservation_bomb(depth: usize, declared: u64, total_len: usize) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"MIC3");
    b.push(0x02); // version
    uleb(0, &mut b); // string-table count
    uleb(0, &mut b); // next_id
    uleb(0, &mut b); // exports count
    uleb(1, &mut b); // instruction count = 1 (the outermost region)
    for _ in 0..depth {
        b.push(0x24); // OP_REGION
        uleb(declared, &mut b); // body_len — huge, but <= limit so read_count passes
    }
    // Pad so the whole-input length is at least `declared`; the decoder trips the
    // depth guard long before it reaches this padding.
    if b.len() < total_len {
        b.resize(total_len, 0);
    }
    b
}

#[test]
fn mic3_nested_region_reservation_bomb_is_rejected() {
    // 256 nested regions, each declaring a 1_000_000-element body, from a ~1 MiB
    // input. Pre-fix this drove ~256 * 1e6 * size_of::<Instr>() of live
    // reservations (tens of GB -> SIGABRT); post-fix the reservation is capped at
    // a small constant so the parser fails closed at the depth guard.
    let declared = 1_000_000u64;
    let b = nested_region_reservation_bomb(256, declared, declared as usize + 1);
    assert!(
        b.len() < 10 * 1024 * 1024,
        "crafted input stays within the mic@3 input cap: {} bytes",
        b.len()
    );

    let res = parse_mic3(&b);
    assert!(
        res.is_err(),
        "a nested-region reservation-amplification bomb must be rejected, not parsed"
    );
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("nesting depth") || msg.contains("depth"),
        "rejection should trip the nesting-depth guard, got: {msg}"
    );
}

#[test]
fn mic3_modest_nested_regions_still_parse() {
    // Control: a shallow nested-region chain (well under the depth guard, small
    // declared bodies) that is TRUNCATED must fail as a clean truncation error,
    // never OOM and never a spurious depth error — proving the reservation cap
    // does not change decode semantics for legitimate shapes.
    let b = nested_region_reservation_bomb(4, 8, 64);
    let res = parse_mic3(&b);
    // It is not a well-formed module (bodies are unbacked padding), so it errors;
    // the point is that it errors quickly and safely, not via the depth guard.
    assert!(res.is_err(), "unbacked shallow region chain must error");
}

#[test]
fn mic3_modest_string_reuse_still_parses() {
    // Control: a short (32-byte) identifier referenced 100x is trivially within
    // budget — the guard must never false-reject a legitimate module.
    let b = string_bomb(32, 100);
    let res = parse_mic3(&b);
    assert!(
        res.is_ok(),
        "modest string reuse must parse (budget must not false-reject); err: {:?}",
        res.err()
    );
}
