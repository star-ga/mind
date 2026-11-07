// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The literal-parser and anchor-bundle program of
//! `tests/std_tls_x509_import.rs`: `x509_parse_ipv4` and `x509_parse_ipv6` on
//! IPv4 and IPv6 literals and on a DNS name, and `x509_load_anchors` on the DER
//! and PEM bundles (the same eight certificates) and on malformed PEM.

use super::{DIFFERS, TD, program, stores};

const CHECKS: &str = r#"fn check_v4() -> i64 {
    let h: i64 = __mind_alloc(16);
    let n: i64 = host_v4(h);
    let w: i64 = __mind_alloc(8);
    want_v4(w);
    let o: i64 = __mind_alloc(8);
    if x509_parse_ipv4(h, n, o) != 1 {
        return 1;
    }
    return differs(o, w, 4);
}

fn check_dns_name() -> i64 {
    let h: i64 = __mind_alloc(16);
    let n: i64 = host_dns(h);
    let o: i64 = __mind_alloc(16);
    if x509_parse_ipv4(h, n, o) != 0 {
        return 1;
    }
    if x509_parse_ipv6(h, n, o) != 0 {
        return 1;
    }
    return 0;
}

fn check_v6(h: i64, n: i64, w: i64) -> i64 {
    let o: i64 = __mind_alloc(16);
    if x509_parse_ipv6(h, n, o) != 1 {
        return 1;
    }
    return differs(o, w, 16);
}

fn check_bundle(blob: i64, count: i64, der: i64, der_len: i64) -> i64 {
    if blob <= 0 {
        return 1;
    }
    if __mind_load_i64(blob) != count {
        return 1;
    }
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < count {
        total = total + __mind_load_i64(blob + 16 + 16 * i);
        i = i + 1;
    }
    if total != der_len {
        return 1;
    }
    return differs(blob + 8 + 16 * count, der, der_len);
}

fn checks() -> i64 {
    let mut n: i64 = check_v4();
    n = n + check_dns_name();
    let h6: i64 = __mind_alloc(48);
    let w6: i64 = __mind_alloc(16);
    want_v6(w6);
    n = n + check_v6(h6, host_v6(h6), w6);
    let h6l: i64 = __mind_alloc(48);
    let w6l: i64 = __mind_alloc(16);
    want_v6_loop(w6l);
    n = n + check_v6(h6l, host_v6_loop(h6l), w6l);
    let der: i64 = __mind_alloc(8192);
    let der_len: i64 = der_bundle(der);
    let pem: i64 = __mind_alloc(8192);
    let pem_len: i64 = pem_bundle(pem);
    n = n + check_bundle(x509_load_anchors(pem, pem_len), want_count(), der, der_len);
    n = n + check_bundle(x509_load_anchors(der, der_len), 8, der, der_len);
    let bad: i64 = __mind_alloc(1024);
    let bad_len: i64 = bad_bundle(bad);
    if x509_load_anchors(bad, bad_len) != 0 - 1 {
        n = n + 1;
    }
    return n;
}

"#;

/// The program source; `mutate` expects nine anchors in the PEM bundle
/// instead of eight.
pub fn source(mutate: bool) -> String {
    let read = |f: &str| {
        let path = format!("{TD}/{f}");
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    };
    let mut v6 = [0u8; 16];
    v6[..4].copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8]);
    v6[15] = 1;
    let mut v6_loop = [0u8; 16];
    v6_loop[15] = 1;
    let mut body = String::new();
    for (name, data) in [
        ("der_bundle", read("bundle_anchors.der")),
        ("pem_bundle", read("bundle_anchors.pem")),
        ("bad_bundle", read("bundle_bad.pem")),
        ("host_v4", b"192.168.1.1".to_vec()),
        ("want_v4", vec![192, 168, 1, 1]),
        ("host_dns", b"example.com".to_vec()),
        ("host_v6", b"2001:db8::1".to_vec()),
        ("want_v6", v6.to_vec()),
        ("host_v6_loop", b"::1".to_vec()),
        ("want_v6_loop", v6_loop.to_vec()),
    ] {
        body.push_str(&stores(name, &data));
    }
    let count = if mutate { 9 } else { 8 };
    body.push_str(&format!(
        "fn want_count() -> i64 {{\n    return {count};\n}}\n\n"
    ));
    body.push_str(DIFFERS);
    body.push_str(CHECKS);
    program(&["x509"], &body)
}
