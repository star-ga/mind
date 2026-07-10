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
    b.extend(std::iter::repeat(b'a').take(l));
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
