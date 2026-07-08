// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 8032 Ed25519 — deterministic signing/verification, self-contained.
//!
//! # Why a self-contained port (RFC 0021 §6 signing layer)
//!
//! The evidence chain's signing layer must not pull an external crypto crate
//! (the compiler ships with only `clap` as a runtime dependency) and must reuse
//! the same FIPS-style, byte-for-byte deterministic discipline as the rest of
//! RFC 0016/0021 — so the digest a Rust bootstrap produces is bit-identical to a
//! future pure-MIND `std.ed25519` over the same bytes.  Ed25519 is *deterministic*
//! by construction (RFC 8032 §5.1.6: the per-signature nonce is derived by hashing
//! the private prefix with the message, no RNG), which is exactly the property the
//! MIND determinism gate requires: the same key + message always yields the same
//! signature bytes on every substrate.
//!
//! # Provenance
//!
//! The Edwards-curve field/point arithmetic is a direct port of TweetNaCl
//! (D. J. Bernstein, Bernard van Gogh, et al., public domain), chosen for the
//! same reason `std/x25519.mind` uses the TweetNaCl `gf` limb representation:
//! every limb product stays inside a single `i64` with no 128-bit intermediate,
//! making the arithmetic trivially auditable against the RFC 8032 test vectors.
//! SHA-512 is a standalone FIPS-180-4 implementation.
//!
//! # Scope
//!
//! Sign/verify over an arbitrary message (the evidence layer signs the 32-byte
//! `trace_hash`).  Constant-time discipline is inherited from TweetNaCl's
//! branch-free field ops; this is a provenance/attestation primitive, not a
//! high-throughput one.

// ─── SHA-512 (FIPS 180-4) ──────────────────────────────────────────────────────

const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// SHA-512 of `data` (FIPS 180-4), returning the 64-byte digest.
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h: [u64; 8] = [
        0x6a09e667f3bcc908,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
        0x510e527fade682d1,
        0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b,
        0x5be0cd19137e2179,
    ];

    // Padding: append 0x80, then zeros, then 128-bit big-endian bit length.
    let bitlen = (data.len() as u128) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 128 != 112 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    let mut w = [0u64; 80];
    for block in msg.chunks_exact(128) {
        for (i, wi) in w.iter_mut().enumerate().take(16) {
            let mut b = [0u8; 8];
            b.copy_from_slice(&block[i * 8..i * 8 + 8]);
            *wi = u64::from_be_bytes(b);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..80 {
            let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA512_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 64];
    for (i, hi) in h.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&hi.to_be_bytes());
    }
    out
}

// ─── Ed25519 field arithmetic (TweetNaCl `gf` = 16 × i64 limbs, radix 2^16) ──────

type Gf = [i64; 16];

const GF0: Gf = [0; 16];
const GF1: Gf = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// d2 = 2*d (mod p), the Edwards curve constant used in point addition.
const D2: Gf = [
    0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0, 0xd130, 0xeef3, 0x80f2, 0x198e,
    0xfce7, 0x56df, 0xd9dc, 0x2406,
];
// d, the Edwards curve constant used in point decompression.
const D: Gf = [
    0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070, 0xe898, 0x7779, 0x4079, 0x8cc7,
    0xfe73, 0x2b6f, 0x6cee, 0x5203,
];
// Base point coordinates.
const X: Gf = [
    0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c, 0xdc5c, 0xfdd6, 0xe231, 0xc0a4,
    0x53fe, 0xcd6e, 0x36d3, 0x2169,
];
const Y: Gf = [
    0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666,
    0x6666, 0x6666, 0x6666, 0x6666,
];
// sqrt(-1) mod p.
const I: Gf = [
    0xa0b0, 0x4a0e, 0x1b27, 0xc4ee, 0xe478, 0xad2f, 0x1806, 0x2f43, 0xd7a7, 0x3dfb, 0x0099, 0x2b4d,
    0xdf0b, 0x4fc1, 0x2480, 0x2b83,
];

// Group order L (little-endian bytes): 2^252 + 27742317777372353535851937790883648493.
const L: [i64; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
];

fn car25519(o: &mut Gf) {
    for i in 0..16 {
        o[i] += 1i64 << 16;
        let c = o[i] >> 16;
        // o[(i+1) if i<15 else 0] += c - 1 + 37*(c-1)*(i==15)
        let target = if i < 15 { i + 1 } else { 0 };
        o[target] += c - 1 + 37 * (c - 1) * ((i == 15) as i64);
        o[i] -= c << 16;
    }
}

fn sel25519(p: &mut Gf, q: &mut Gf, b: i64) {
    let c = !(b - 1);
    for i in 0..16 {
        let t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

fn pack25519(n: &Gf) -> [u8; 32] {
    let mut t = *n;
    car25519(&mut t);
    car25519(&mut t);
    car25519(&mut t);
    for _ in 0..2 {
        let mut m: Gf = GF0;
        m[0] = t[0] - 0xffed;
        for i in 1..15 {
            m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
            m[i - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        let b = (m[15] >> 16) & 1;
        m[14] &= 0xffff;
        sel25519(&mut t, &mut m, 1 - b);
    }
    let mut o = [0u8; 32];
    for i in 0..16 {
        o[2 * i] = (t[i] & 0xff) as u8;
        o[2 * i + 1] = (t[i] >> 8) as u8;
    }
    o
}

/// Constant-time equality of two field elements (packs both, compares bytes).
fn neq25519(a: &Gf, b: &Gf) -> bool {
    pack25519(a) != pack25519(b)
}

fn par25519(a: &Gf) -> u8 {
    pack25519(a)[0] & 1
}

fn unpack25519(n: &[u8; 32]) -> Gf {
    let mut o: Gf = GF0;
    for i in 0..16 {
        o[i] = n[2 * i] as i64 + ((n[2 * i + 1] as i64) << 8);
    }
    o[15] &= 0x7fff;
    o
}

fn add_gf(a: &Gf, b: &Gf) -> Gf {
    let mut o: Gf = GF0;
    for i in 0..16 {
        o[i] = a[i] + b[i];
    }
    o
}

fn sub_gf(a: &Gf, b: &Gf) -> Gf {
    let mut o: Gf = GF0;
    for i in 0..16 {
        o[i] = a[i] - b[i];
    }
    o
}

fn mul_gf(a: &Gf, b: &Gf) -> Gf {
    let mut t = [0i64; 31];
    for i in 0..16 {
        for j in 0..16 {
            t[i + j] += a[i] * b[j];
        }
    }
    for i in 0..15 {
        t[i] += 38 * t[i + 16];
    }
    let mut o: Gf = GF0;
    o[..16].copy_from_slice(&t[..16]);
    car25519(&mut o);
    car25519(&mut o);
    o
}

fn sq_gf(a: &Gf) -> Gf {
    mul_gf(a, a)
}

fn inv25519(i: &Gf) -> Gf {
    let mut c = *i;
    for a in (0..=253).rev() {
        c = sq_gf(&c);
        if a != 2 && a != 4 {
            c = mul_gf(&c, i);
        }
    }
    c
}

fn pow2523(i: &Gf) -> Gf {
    let mut c = *i;
    for a in (0..=250).rev() {
        c = sq_gf(&c);
        if a != 1 {
            c = mul_gf(&c, i);
        }
    }
    c
}

// ─── Edwards points (extended coords P = [x, y, z, t]) ───────────────────────────

type Point = [Gf; 4];

fn point_add(p: &mut Point, q: &Point) {
    let a = mul_gf(&sub_gf(&p[1], &p[0]), &sub_gf(&q[1], &q[0]));
    let b = mul_gf(&add_gf(&p[0], &p[1]), &add_gf(&q[0], &q[1]));
    let mut c = mul_gf(&p[3], &q[3]);
    c = mul_gf(&c, &D2);
    let mut d = mul_gf(&p[2], &q[2]);
    d = add_gf(&d, &d);
    let e = sub_gf(&b, &a);
    let f = sub_gf(&d, &c);
    let g = add_gf(&d, &c);
    let h = add_gf(&b, &a);
    p[0] = mul_gf(&e, &f);
    p[1] = mul_gf(&h, &g);
    p[2] = mul_gf(&g, &f);
    p[3] = mul_gf(&e, &h);
}

fn cswap(p: &mut Point, q: &mut Point, b: u8) {
    for i in 0..4 {
        sel25519(&mut p[i], &mut q[i], b as i64);
    }
}

fn pack_point(p: &Point) -> [u8; 32] {
    let zi = inv25519(&p[2]);
    let tx = mul_gf(&p[0], &zi);
    let ty = mul_gf(&p[1], &zi);
    let mut r = pack25519(&ty);
    r[31] ^= par25519(&tx) << 7;
    r
}

fn scalarmult(q: &Point, s: &[u8; 32]) -> Point {
    let mut p: Point = [GF0, GF1, GF1, GF0];
    let mut q = *q;
    for i in (0..=255).rev() {
        let b = (s[i >> 3] >> (i & 7)) & 1;
        cswap(&mut p, &mut q, b);
        point_add(&mut q, &p);
        let p_copy = p;
        point_add(&mut p, &p_copy);
        cswap(&mut p, &mut q, b);
    }
    p
}

fn scalarbase(s: &[u8; 32]) -> Point {
    let q: Point = [X, Y, GF1, mul_gf(&X, &Y)];
    scalarmult(&q, s)
}

/// Decode a compressed point into `-P` (TweetNaCl `unpackneg`), returning `None`
/// if the encoding is not a valid curve point.
fn unpack_neg(p: &[u8; 32]) -> Option<Point> {
    let mut r: Point = [GF0, GF0, GF1, GF0];
    r[1] = unpack25519(p);
    let num = sub_gf(&sq_gf(&r[1]), &r[2]); // y^2 - 1
    let den = add_gf(&r[2], &mul_gf(&sq_gf(&r[1]), &D)); // 1 + d*y^2

    let den2 = sq_gf(&den);
    let den4 = sq_gf(&den2);
    let den6 = mul_gf(&den4, &den2);
    let mut t = mul_gf(&den6, &num);
    t = mul_gf(&t, &den);

    t = pow2523(&t);
    t = mul_gf(&t, &num);
    t = mul_gf(&t, &den);
    t = mul_gf(&t, &den);
    r[0] = mul_gf(&t, &den);

    let mut chk = mul_gf(&sq_gf(&r[0]), &den);
    if neq25519(&chk, &num) {
        r[0] = mul_gf(&r[0], &I);
    }
    chk = mul_gf(&sq_gf(&r[0]), &den);
    if neq25519(&chk, &num) {
        return None;
    }

    if par25519(&r[0]) == (p[31] >> 7) {
        r[0] = sub_gf(&GF0, &r[0]);
    }
    r[3] = mul_gf(&r[0], &r[1]);
    Some(r)
}

// ─── Scalar reduction mod L ──────────────────────────────────────────────────────

fn mod_l(x: &mut [i64; 64]) -> [u8; 32] {
    let mut carry: i64;
    for i in (32..64).rev() {
        carry = 0;
        let mut j = i - 32;
        while j < i - 12 {
            x[j] += carry - 16 * x[i] * L[j - (i - 32)];
            carry = (x[j] + 128) >> 8;
            x[j] -= carry << 8;
            j += 1;
        }
        x[j] += carry;
        x[i] = 0;
    }
    carry = 0;
    for j in 0..32 {
        x[j] += carry - (x[31] >> 4) * L[j];
        carry = x[j] >> 8;
        x[j] &= 255;
    }
    for j in 0..32 {
        x[j] -= carry * L[j];
    }
    let mut r = [0u8; 32];
    for i in 0..32 {
        x[i + 1] += x[i] >> 8;
        r[i] = (x[i] & 255) as u8;
    }
    r
}

/// Reduce a 64-byte little-endian value mod L to 32 bytes.
fn reduce(r: &[u8; 64]) -> [u8; 32] {
    let mut x = [0i64; 64];
    for i in 0..64 {
        x[i] = r[i] as i64;
    }
    mod_l(&mut x)
}

// ─── Public API ─────────────────────────────────────────────────────────────────

/// Derive the 32-byte Ed25519 public key from a 32-byte seed (RFC 8032 §5.1.5).
pub fn public_key(seed: &[u8; 32]) -> [u8; 32] {
    let h = sha512(seed);
    let mut a = [0u8; 32];
    a.copy_from_slice(&h[..32]);
    a[0] &= 248;
    a[31] &= 127;
    a[31] |= 64;
    pack_point(&scalarbase(&a))
}

/// Sign `msg` with the 32-byte `seed` (RFC 8032 §5.1.6), returning the 64-byte
/// signature `R || S`.  Deterministic: identical `(seed, msg)` ⇒ identical bytes.
pub fn sign(seed: &[u8; 32], msg: &[u8]) -> [u8; 64] {
    let h = sha512(seed);
    let mut a = [0u8; 32]; // clamped secret scalar
    a.copy_from_slice(&h[..32]);
    a[0] &= 248;
    a[31] &= 127;
    a[31] |= 64;
    let prefix = &h[32..64];
    let pk = pack_point(&scalarbase(&a));

    // r = SHA-512(prefix || msg) mod L
    let mut r_input = Vec::with_capacity(32 + msg.len());
    r_input.extend_from_slice(prefix);
    r_input.extend_from_slice(msg);
    let r = reduce(&sha512(&r_input));
    let big_r = pack_point(&scalarbase(&r));

    // k = SHA-512(R || A || msg) mod L
    let mut k_input = Vec::with_capacity(64 + msg.len());
    k_input.extend_from_slice(&big_r);
    k_input.extend_from_slice(&pk);
    k_input.extend_from_slice(msg);
    let k = reduce(&sha512(&k_input));

    // S = (r + k*a) mod L
    let mut x = [0i64; 64];
    for i in 0..32 {
        x[i] = r[i] as i64;
    }
    for i in 0..32 {
        for j in 0..32 {
            x[i + j] += (k[i] as i64) * (a[j] as i64);
        }
    }
    let s = mod_l(&mut x);

    let mut sig = [0u8; 64];
    sig[..32].copy_from_slice(&big_r);
    sig[32..].copy_from_slice(&s);
    sig
}

/// RFC 8032 §5.1.7 canonical-`S` predicate: `true` iff the signature scalar
/// `0 <= S < L` (little-endian). Used by [`verify`] to reject malleated
/// signatures. `L` here is the same group order as the `L` scalar table above.
fn s_is_canonical(s: &[u8; 32]) -> bool {
    // Group order L, little-endian bytes (2^252 + 277423177...883648493).
    const L_BYTES: [u8; 32] = [
        0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde,
        0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
    ];
    // Compare S < L from the most-significant byte down (little-endian storage).
    for i in (0..32).rev() {
        if s[i] < L_BYTES[i] {
            return true;
        }
        if s[i] > L_BYTES[i] {
            return false;
        }
    }
    false // S == L is non-canonical.
}

/// Verify a 64-byte Ed25519 signature over `msg` under `pubkey` (RFC 8032 §5.1.7).
/// Fail-closed: any malformed input or algebraic mismatch returns `false`.
pub fn verify(pubkey: &[u8; 32], msg: &[u8], sig: &[u8; 64]) -> bool {
    let mut big_r = [0u8; 32];
    big_r.copy_from_slice(&sig[..32]);
    let mut s = [0u8; 32];
    s.copy_from_slice(&sig[32..]);

    // Malleability guard (RFC 8032 §5.1.7): reject a non-canonical scalar
    // `S >= L`. Without this, adding L (or a multiple) to S yields a second,
    // distinct-but-verifying signature over the same message — signature
    // malleability. Honest signers always emit `S < L` (the signer reduces mod L),
    // so this never rejects a genuine signature.
    if !s_is_canonical(&s) {
        return false;
    }

    let neg_a = match unpack_neg(pubkey) {
        Some(p) => p,
        None => return false,
    };

    // k = SHA-512(R || A || msg) mod L
    let mut k_input = Vec::with_capacity(64 + msg.len());
    k_input.extend_from_slice(&big_r);
    k_input.extend_from_slice(pubkey);
    k_input.extend_from_slice(msg);
    let k = reduce(&sha512(&k_input));

    // Check: [S]B == R + [k]A  ⇔  [S]B + [k](-A) == R
    let mut p = scalarmult(&neg_a, &k);
    let q = scalarbase(&s);
    point_add(&mut p, &q);
    let check = pack_point(&p);
    check == big_r
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn sha512_known_answers() {
        // FIPS 180-4 / NIST examples.
        assert_eq!(
            sha512(b""),
            hex("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e")
                .as_slice()
        );
        assert_eq!(
            sha512(b"abc"),
            hex("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f")
                .as_slice()
        );
    }

    // Ground-truth vectors produced by the `cryptography` (OpenSSL) Ed25519
    // implementation for the seed below — an independent oracle for this port.
    const SEED: &str = "9d61b19deffcbeb2c4cf3d1e79fdae0e34bccbaacf9ec24bd0e75c4e5d6bde1e";
    const PUB: &str = "434331a594c95392ab1737b928378d7a84e8f5704a8f0afb3492fb812013492a";

    fn seed32() -> [u8; 32] {
        hex(SEED).as_slice().try_into().unwrap()
    }

    #[test]
    fn public_key_matches_oracle() {
        assert_eq!(public_key(&seed32()).as_slice(), hex(PUB).as_slice());
    }

    #[test]
    fn sign_matches_oracle_empty_message() {
        let sig = sign(&seed32(), b"");
        assert_eq!(
            sig.as_slice(),
            hex("9baa694754759baa425995ae027939b051aa6da0004c5b47ccdffbab454fcb6a9c79bb4fb8f0652f1fc6372f565ccd4483c2ed200c200933acd534116a67b006").as_slice()
        );
    }

    #[test]
    fn sign_matches_oracle_text_message() {
        let sig = sign(&seed32(), b"MIND evidence chain");
        assert_eq!(
            sig.as_slice(),
            hex("952526054bd2db944ecd33680f66b3356849006cda256db81d566a8528ae95b4a4bd19d0b19b0e6610e4f317a6ddac0765ce17f64588e5db1969053771bcb605").as_slice()
        );
    }

    #[test]
    fn sign_matches_oracle_binary_message() {
        let sig = sign(&seed32(), &hex("abcdef"));
        assert_eq!(
            sig.as_slice(),
            hex("581d3fc5211a6a684f74c8a8a2546cc14c54a75979635fc1b030207503dcb9301bb873534261b45dac52d428bcbd9030727c13875f296feec5c28f1d0a8d9504").as_slice()
        );
    }

    #[test]
    fn sign_verify_roundtrip() {
        let seed = seed32();
        let pk = public_key(&seed);
        for msg in [b"".as_slice(), b"a", b"MIND evidence chain", &[0xFFu8; 32]] {
            let sig = sign(&seed, msg);
            assert!(
                verify(&pk, msg, &sig),
                "verify must accept a valid signature"
            );
        }
    }

    #[test]
    fn verify_rejects_message_tamper() {
        let seed = seed32();
        let pk = public_key(&seed);
        let msg = b"MIND evidence chain";
        let sig = sign(&seed, msg);
        let mut bad = *msg;
        bad[0] ^= 0x01;
        assert!(!verify(&pk, &bad, &sig), "flipped message must fail");
    }

    #[test]
    fn verify_rejects_signature_tamper() {
        let seed = seed32();
        let pk = public_key(&seed);
        let msg = b"trace_hash";
        let sig = sign(&seed, msg);
        for idx in [0usize, 31, 32, 63] {
            let mut bad = sig;
            bad[idx] ^= 0x01;
            assert!(!verify(&pk, msg, &bad), "flipped sig byte {idx} must fail");
        }
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let seed = seed32();
        let msg = b"trace_hash";
        let sig = sign(&seed, msg);
        let mut other_seed = seed;
        other_seed[0] ^= 0x01;
        let other_pk = public_key(&other_seed);
        assert!(!verify(&other_pk, msg, &sig), "wrong key must fail");
    }

    #[test]
    fn determinism_same_seed_msg_same_sig() {
        let seed = seed32();
        let m = b"deterministic";
        assert_eq!(sign(&seed, m), sign(&seed, m));
    }

    // Malleability guard (§5.1.7): a signature whose scalar S is pushed to >= L
    // (here S + L, and S = L itself) must be rejected even though the underlying
    // point equation could otherwise be satisfied.
    #[test]
    fn verify_rejects_non_canonical_s() {
        let seed = seed32();
        let pk = public_key(&seed);
        let msg = b"malleability";
        let sig = sign(&seed, msg);
        assert!(verify(&pk, msg, &sig), "control: genuine sig verifies");

        // L (little-endian) — same value as the module's L / L_BYTES.
        const L_BYTES: [u8; 32] = [
            0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9,
            0xde, 0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
        ];
        // S' = S + L (mod 2^256, no carry out for a canonical S) — non-canonical.
        let mut malleated = sig;
        let mut carry = 0u16;
        for i in 0..32 {
            let sum = malleated[32 + i] as u16 + L_BYTES[i] as u16 + carry;
            malleated[32 + i] = (sum & 0xff) as u8;
            carry = sum >> 8;
        }
        assert!(
            !verify(&pk, msg, &malleated),
            "S + L (non-canonical scalar) must be rejected"
        );

        // S = L exactly is also non-canonical.
        assert!(!s_is_canonical(&L_BYTES), "S == L must be non-canonical");
    }
}
