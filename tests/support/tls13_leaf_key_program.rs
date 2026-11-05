// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The leaf-key programs of `tests/std_tls_x509_import.rs`.
//!
//! `x509_parse` records an RSA key only from a subjectPublicKeyInfo that
//! decodes strictly as rsaEncryption (NULL or absent parameters, a BIT STRING
//! with 0 unused bits holding exactly SEQUENCE { INTEGER n, INTEGER e }, both
//! positive and minimal). An ECDSA or other key leaves the key slots zero, and
//! a malformed rsaEncryption key rejects the certificate. The TLS client takes
//! its CertificateVerify key from those slots, so an ECDSA leaf is refused at
//! CertificateVerify (E_CV) and a malformed RSA leaf at Certificate (E_CHAIN).
//! The malformed keys are derived from `rsa_leaf` here, re-encoding every
//! enclosing length, so each one differs from the valid key in one rule.

use super::{TD, program, stores};

/// `(tag, header length, content length)` of the TLV at `at`.
fn header(der: &[u8], at: usize) -> (u8, usize, usize) {
    let l0 = usize::from(der[at + 1]);
    if l0 < 0x80 {
        return (der[at], 2, l0);
    }
    let nb = l0 & 0x7f;
    let len = der[at + 2..at + 2 + nb]
        .iter()
        .fold(0, |acc, &b| (acc << 8) | usize::from(b));
    (der[at], 2 + nb, len)
}

/// The TLVs in the content of the TLV `der`.
fn children(der: &[u8]) -> Vec<&[u8]> {
    let (_, h, len) = header(der, 0);
    let mut out = Vec::new();
    let mut at = h;
    while at < h + len {
        let (_, ch, cl) = header(der, at);
        out.push(&der[at..at + ch + cl]);
        at += ch + cl;
    }
    out
}

fn content(der: &[u8]) -> &[u8] {
    let (_, h, len) = header(der, 0);
    &der[h..h + len]
}

fn tlv(tag: u8, body: &[u8]) -> Vec<u8> {
    let n = body.len();
    let mut out = vec![tag];
    match n {
        0..=0x7f => out.push(n as u8),
        0x80..=0xff => out.extend([0x81, n as u8]),
        _ => out.extend([0x82, (n >> 8) as u8, n as u8]),
    }
    out.extend_from_slice(body);
    out
}

fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

/// `der` with the TLV at child-index `path` replaced by `f(old)`; every
/// enclosing length is re-encoded.
fn rebuild(der: &[u8], path: &[usize], f: &dyn Fn(&[u8]) -> Vec<u8>) -> Vec<u8> {
    let Some((&i, rest)) = path.split_first() else {
        return f(der);
    };
    let mut body = Vec::new();
    for (k, kid) in children(der).into_iter().enumerate() {
        if k == i {
            body.extend(rebuild(kid, rest, f));
        } else {
            body.extend_from_slice(kid);
        }
    }
    tlv(der[0], &body)
}

/// The parts of `rsa_leaf`'s rsaEncryption subjectPublicKeyInfo.
struct Parts {
    oid: Vec<u8>,
    null: Vec<u8>,
    n: Vec<u8>,
    e: Vec<u8>,
}

/// SEQUENCE { SEQUENCE { alg }, BIT STRING { unused, key } }.
fn spki(alg: &[u8], unused: u8, key: &[u8]) -> Vec<u8> {
    let bits = cat(&[&[unused], key]);
    tlv(0x30, &cat(&[&tlv(0x30, alg), &tlv(0x03, &bits)]))
}

fn rsa_key(p: &Parts) -> Vec<u8> {
    tlv(0x30, &cat(&[&p.n, &p.e]))
}

type Variant = (&'static str, fn(&Parts) -> Vec<u8>);

/// One rule broken per variant; `oid_not_rsa` names another algorithm (no
/// RSA key) and `params_absent` is the RFC 8017 form that stays valid.
#[rustfmt::skip]
const VARIANTS: &[Variant] = &[
    ("params_absent", |p| spki(&p.oid, 0, &rsa_key(p))),
    ("oid_not_rsa", |p| { let mut o = p.oid.clone(); *o.last_mut().unwrap() = 0x0b; spki(&cat(&[&o, &p.null]), 0, &rsa_key(p)) }),
    ("key_not_sequence", |p| spki(&cat(&[&p.oid, &p.null]), 0, &tlv(0x31, &cat(&[&p.n, &p.e])))),
    ("modulus_not_integer", |p| spki(&cat(&[&p.oid, &p.null]), 0, &tlv(0x30, &cat(&[&tlv(0x04, content(&p.n)), &p.e])))),
    ("third_integer", |p| spki(&cat(&[&p.oid, &p.null]), 0, &tlv(0x30, &cat(&[&p.n, &p.e, &[2, 1, 1]])))),
    ("trailing_bytes", |p| spki(&cat(&[&p.oid, &p.null]), 0, &cat(&[&rsa_key(p), &[0]]))),
    ("unused_bits", |p| spki(&cat(&[&p.oid, &p.null]), 1, &rsa_key(p))),
    ("params_not_null", |p| spki(&cat(&[&p.oid, &[4, 0]]), 0, &rsa_key(p))),
    ("negative_modulus", |p| spki(&cat(&[&p.oid, &p.null]), 0, &tlv(0x30, &cat(&[&tlv(2, &content(&p.n)[1..]), &p.e])))),
    ("modulus_extra_zero", |p| spki(&cat(&[&p.oid, &p.null]), 0, &tlv(0x30, &cat(&[&tlv(2, &cat(&[&[0], content(&p.n)])), &p.e])))),
];

fn read(name: &str) -> Vec<u8> {
    let path = format!("{TD}/{name}.der");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// A fixture by name; `rsa_leaf#<variant>` is `rsa_leaf` with that SPKI.
fn cert(name: &str) -> Vec<u8> {
    let Some((base, tweak)) = name.split_once('#') else {
        return read(name);
    };
    let der = read(base);
    // Certificate -> tbsCertificate (0) -> [0] version, serial, signature,
    // issuer, validity, subject, subjectPublicKeyInfo (6).
    let tbs = children(&der)[0];
    let old = children(tbs)[6];
    let alg = children(children(old)[0]);
    assert_eq!(
        alg[0],
        tlv(0x06, &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 1, 1, 1]),
        "{base}: rsaEncryption"
    );
    let key = children(old)[1];
    assert_eq!(content(key)[0], 0, "{base}: 0 unused bits");
    let rsa = &content(key)[1..];
    let ne = children(rsa);
    assert_eq!(
        content(ne[0])[0],
        0,
        "{base}: the modulus carries a sign byte"
    );
    let parts = Parts {
        oid: alg[0].to_vec(),
        null: alg[1].to_vec(),
        n: ne[0].to_vec(),
        e: ne[1].to_vec(),
    };
    let (_, make) = VARIANTS
        .iter()
        .find(|(v, _)| *v == tweak)
        .unwrap_or_else(|| panic!("unknown SPKI variant `{tweak}`"));
    let out = rebuild(&der, &[0, 6], &|_| make(&parts));
    assert_ne!(out, der, "{name} must change the certificate");
    out
}

fn fname(name: &str) -> String {
    format!("der_{}", name.replace('#', "_"))
}

fn be24(n: usize) -> [u8; 3] {
    [(n >> 16) as u8, (n >> 8) as u8, n as u8]
}

/// A TLS 1.3 Certificate handshake message carrying `names`, leaf first.
fn certificate_msg(names: &[&str]) -> Vec<u8> {
    let mut list = Vec::new();
    for n in names {
        let der = cert(n);
        list.extend(be24(der.len()));
        list.extend(der);
        list.extend([0, 0]);
    }
    let body = cat(&[&[0], &be24(list.len()), &list]);
    cat(&[&[11], &be24(body.len()), &body])
}

/// CertificateVerify, rsa_pss_rsae_sha256, a 256-byte signature.
fn cv_msg() -> Vec<u8> {
    cat(&[&[15], &be24(260), &[8, 4, 1, 0], &[1; 256]])
}

/// The tls13c_init anchor input: u32be count, then u32be length and DER.
fn anchors(names: &[&str]) -> Vec<u8> {
    let mut out = (names.len() as u32).to_be_bytes().to_vec();
    for n in names {
        let der = cert(n);
        out.extend((der.len() as u32).to_be_bytes());
        out.extend(der);
    }
    out
}

/// `(certificate, x509_parse rc, n_len, e_len)`.
#[rustfmt::skip]
const PARSE: &[(&str, i64, i64, i64)] = &[
    ("rsa_leaf", 0, 256, 3),
    ("rsa_leaf#params_absent", 0, 256, 3),
    ("ec_leaf", 0, 0, 0),
    ("p384_leaf", 0, 0, 0),
    ("rsa_leaf#oid_not_rsa", 0, 0, 0),
    ("rsa_leaf#key_not_sequence", 1, 0, 0),
    ("rsa_leaf#modulus_not_integer", 1, 0, 0),
    ("rsa_leaf#third_integer", 1, 0, 0),
    ("rsa_leaf#trailing_bytes", 1, 0, 0),
    ("rsa_leaf#unused_bits", 1, 0, 0),
    ("rsa_leaf#params_not_null", 1, 0, 0),
    ("rsa_leaf#negative_modulus", 1, 0, 0),
    ("rsa_leaf#modulus_extra_zero", 1, 0, 0),
];

/// `(Certificate chain, tls13c_parse_certificate rc, RSA len, exponent)`.
/// With RSA len 0 a CertificateVerify must then fail with E_CV.
#[rustfmt::skip]
const TLS: &[(&[&str], &str, i64, i64)] = &[
    (&["ec_leaf", "ec_inter"], "0", 0, 0),
    (&["rsa_leaf#oid_not_rsa"], "0", 0, 0),
    (&["rsa_leaf"], "0", 256, 65537),
    (&["rsa_leaf#third_integer"], "tls13c_err_chain()", 0, 0),
    (&["rsa_leaf#key_not_sequence"], "tls13c_err_chain()", 0, 0),
    (&["rsa_leaf#trailing_bytes"], "tls13c_err_chain()", 0, 0),
];

const HELPERS: &str = r#"fn leaf_state() -> i64 {
    let s: i64 = __mind_alloc(tls13c_state_size());
    let e: i64 = __mind_alloc(192);
    tls13c_zero(e, 192);
    let a: i64 = __mind_alloc(4);
    tls13c_zero(a, 4);
    let d: i64 = __mind_alloc(3);
    __mind_store_i8(d, 97);
    __mind_store_i8(d + 1, 46);
    __mind_store_i8(d + 2, 98);
    tls13c_init(s, e, 192, a, 4, d, 3, 1700000000, 10, 0);
    return s;
}

fn cv_refused(s: i64) -> i64 {
    let m: i64 = __mind_alloc(264);
    let n: i64 = cv_msg(m);
    if tls13c_parse_and_verify_cv(s, m, n) != tls13c_err_cv() {
        return 1;
    }
    return 0;
}

"#;

/// The full handshake state machine from WAIT_CERT with the real chain
/// verifier: the ECDSA chain verifies (verdict 0) and is refused at
/// CertificateVerify; the RSA chain verifies and records its key.
fn flow(now: i64, mutate: bool) -> String {
    let cv_want = if mutate {
        "tls13c_err_chain()"
    } else {
        "tls13c_err_cv()"
    };
    let mut s = stores("anchors_ec", &anchors(&["ec_root"]));
    s.push_str(&stores("anchors_rsa", &anchors(&["rsa_root"])));
    s.push_str(&stores("host_ec", b"www.example.com"));
    s.push_str(&stores("host_rsa", b"www.rsa-test.example.com"));
    s.push_str(&format!(
        "fn flow_state(rsa: i64) -> i64 {{\n    let s: i64 = __mind_alloc(tls13c_state_size());\n\
         \x20   let e: i64 = __mind_alloc(192);\n    tls13c_zero(e, 192);\n\
         \x20   let a: i64 = __mind_alloc(4096);\n    let d: i64 = __mind_alloc(32);\n\
         \x20   let mut an: i64 = 0;\n    let mut dn: i64 = 0;\n\
         \x20   if rsa == 1 {{\n        an = anchors_rsa(a);\n        dn = host_rsa(d);\n    }} else {{\n\
         \x20       an = anchors_ec(a);\n        dn = host_ec(d);\n    }}\n\
         \x20   if tls13c_init(s, e, 192, a, an, d, dn, {now}, 10, 0) != 0 {{\n        return 0;\n    }}\n\
         \x20   tls13c_set_status(s, tls13c_st_wait_cert());\n    return s;\n}}\n\n\
         fn flow_cert(s: i64, rsa: i64, want_n: i64) -> i64 {{\n    if s == 0 {{\n        return 1;\n    }}\n\
         \x20   let m: i64 = __mind_alloc(4096);\n    let mut n: i64 = 0;\n\
         \x20   if rsa == 1 {{\n        n = tls_msg_2(m);\n    }} else {{\n        n = tls_msg_0(m);\n    }}\n\
         \x20   if tls13c_consume_hs_messages(s, m, n) != 0 {{\n        return 1;\n    }}\n\
         \x20   if tls13c_verdict(s) != 0 {{\n        return 1;\n    }}\n\
         \x20   if tls13c_status(s) != tls13c_st_wait_cv() {{\n        return 1;\n    }}\n\
         \x20   if tls13c_load_i64(s + 4752) != want_n {{\n        return 1;\n    }}\n    return 0;\n}}\n\n\
         fn flow_ec() -> i64 {{\n    let s: i64 = flow_state(0);\n\
         \x20   if flow_cert(s, 0, 0) != 0 {{\n        return 1;\n    }}\n\
         \x20   let m: i64 = __mind_alloc(264);\n    let n: i64 = cv_msg(m);\n\
         \x20   if tls13c_consume_hs_messages(s, m, n) != {cv_want} {{\n        return 1;\n    }}\n\
         \x20   if tls13c_status(s) != tls13c_st_error() {{\n        return 1;\n    }}\n\
         \x20   if tls13c_error(s) != tls13c_err_cv() {{\n        return 1;\n    }}\n    return 0;\n}}\n\n\
         fn flow_rsa() -> i64 {{\n    return flow_cert(flow_state(1), 1, 256);\n}}\n\n"
    ));
    s
}

/// The leaf-key program. `full` adds the state-machine flow (ECDSA chain
/// verification: the executable only). `mutate` breaks one expectation: the
/// first x509_parse case, or with `full` the flow's CertificateVerify verdict.
pub fn source(full: bool, mutate: bool) -> String {
    let mut body = stores("cv_msg", &cv_msg());
    let mut calls = Vec::new();
    for (i, (name, rc, n, e)) in PARSE.iter().enumerate() {
        let der = cert(name);
        let rc = rc + if mutate && !full && i == 0 { 100 } else { 0 };
        body.push_str(&stores(&fname(name), &der));
        body.push_str(&format!(
            "// {name}\nfn parse_{i}() -> i64 {{\n    let c: i64 = __mind_alloc({});\n\
             \x20   let n: i64 = {}(c);\n    let f: i64 = __mind_alloc(176);\n\
             \x20   let rc: i64 = x509_parse(c, n, f);\n    if rc != {rc} {{\n        return 1;\n    }}\n\
             \x20   if rc == 0 {{\n        if __mind_load_i64(f + 104) != {n} {{\n            return 1;\n        }}\n\
             \x20       if __mind_load_i64(f + 120) != {e} {{\n            return 1;\n        }}\n    }}\n\
             \x20   return 0;\n}}\n\n",
            der.len(),
            fname(name)
        ));
        calls.push(format!("parse_{i}()"));
    }
    body.push_str(HELPERS);
    for (i, (chain, rc, n, e)) in TLS.iter().enumerate() {
        let msg = certificate_msg(chain);
        body.push_str(&stores(&format!("tls_msg_{i}"), &msg));
        let after = if *n == 0 {
            "    return cv_refused(s);\n".to_string()
        } else {
            format!(
                "    if tls13c_load_i64(s + 4784) != {e} {{\n        return 1;\n    }}\n    return 0;\n"
            )
        };
        body.push_str(&format!(
            "// {chain:?}\nfn tls_{i}() -> i64 {{\n    let s: i64 = leaf_state();\n\
             \x20   let m: i64 = __mind_alloc({});\n    let n: i64 = tls_msg_{i}(m);\n\
             \x20   let rc: i64 = tls13c_parse_certificate(s, m, n);\n\
             \x20   if rc != {rc} {{\n        return 1;\n    }}\n    if rc != 0 {{\n        return 0;\n    }}\n\
             \x20   if tls13c_load_i64(s + 4752) != {n} {{\n        return 1;\n    }}\n{after}}}\n\n",
            msg.len()
        ));
        calls.push(format!("tls_{i}()"));
    }
    if full {
        let now: i64 = std::fs::read_to_string(format!("{TD}/now.txt"))
            .expect("read now.txt")
            .trim()
            .parse()
            .expect("now.txt holds unix seconds");
        body.push_str(&flow(now, mutate));
        calls.push("flow_ec()".into());
        calls.push("flow_rsa()".into());
    }
    body.push_str(&format!(
        "fn checks() -> i64 {{\n    return {};\n}}\n\n",
        calls.join(" + ")
    ));
    program(
        &["x509_der", "tls13_client_core", "tls13_client_api", "https"],
        &body,
    )
}
