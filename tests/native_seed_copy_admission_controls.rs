//! Historical native admission of user copies of private std seed helpers.
//!
//! The frozen compiler receives `std_blob ++ user` and keeps the LAST
//! definition of each bare name, so a user function named like a seed helper
//! replaces that helper for every caller, std callers included. The historical
//! bridge therefore refused every such collision. A collision is an ambiguity
//! only when the two definitions differ: a user copy whose text equals the
//! seed's single private definition (comments and trailing spaces aside)
//! denotes the same function whichever copy the compiler keeps.
//!
//! These controls pin both directions on the historical (unmarked compiler)
//! path: an exact private copy builds and runs, while a divergent, public or
//! open copy keeps the historical refusal. On Linux x86-64 the positive case
//! executes the real native image.
#![cfg(all(feature = "cross-module-imports", feature = "std-surface"))]

mod native_bridge_support;
use native_bridge_support::{Project, assert_native_result};

/// `sha_mask32`, `sha_ch` and `sha_maj` from `std/sha256.mind`, with different comments
/// outside and inside the bodies. Every other byte of each definition matches.
const MASK32_COPY: &str = "// sha_mask32: low word only.\n\
                           fn sha_mask32(x: i64) -> i64 {\n    x & 4294967295 // keep 32 bits\n}\n";
const CH_COPY: &str = "// sha_ch: choose y where x is set, z elsewhere.\n\
                       fn sha_ch(x: i64, y: i64, z: i64) -> i64 {\n    \
                       let notx: i64 = 4294967295 - (x & 4294967295);\n    \
                       sha_mask32((x & y) ^ (notx & z))\n}\n";
const MAJ_COPY: &str = "fn sha_maj(x: i64, y: i64, z: i64) -> i64 {\n    \
                        sha_mask32((x & y) ^ (x & z) ^ (y & z))\n}\n";
/// A call into the std seed: the image is open and keeps the seed.
const SEED_CALL: &str =
    "import std.sha256;\n\nfn keep_seed() -> i64 {\n    return sha256(0, 0, 0);\n}\n\n";
/// sha_maj(5, 3, 6) = 1 ^ 4 ^ 2 = 7; sha_ch(12, 10, 3) = 8 ^ 3 = 11.
const MAIN: &str = "fn main() -> i64 {\n    return sha_maj(5, 3, 6) + sha_ch(12, 10, 3);\n}\n";

fn build(source: &str) -> (i32, String, Option<Vec<u8>>, Project) {
    let p = Project::new("seedcopy");
    p.write("src/main.mind", source);
    let (code, err, artifact) = p.build_native("src/main.mind");
    (code, err, artifact, p)
}

/// The same build through the unmarked compiler on every host: the header's
/// historical path, which the lane-marked Linux stage1 no longer selects.
fn build_historical(source: &str) -> (i32, String, Option<Vec<u8>>, Project) {
    let p = Project::new("seedcopy");
    p.write("src/main.mind", source);
    let (code, err, artifact) = p.build_native_historical("src/main.mind");
    (code, err, artifact, p)
}

/// The measured downstream consumer shape: vendored private std helpers, comments
/// differing. Refused as `closure.std_symbol_collision` before this rule.
#[test]
fn exact_private_copies_of_seed_helpers_build_and_run() {
    let (code, err, artifact, p) = build(&format!("{MASK32_COPY}\n{CH_COPY}\n{MAJ_COPY}\n{MAIN}"));
    assert_eq!(code, 0, "an exact private copy must be admitted: {err}");
    assert!(
        !err.contains("closure."),
        "no closure refusal expected: {err}"
    );
    let bytes = artifact.expect("artifact written");
    assert_eq!(&bytes[0..4], b"\x7fELF", "must preserve ELF framing");
    assert_native_result(&p, &bytes, 18, "sha_maj(5,3,6) + sha_ch(12,10,3) == 18");
}

/// One divergent body keeps the historical refusal and names that helper:
/// the frozen compiler would bind std's own `sha_maj` callers to this body.
#[test]
fn divergent_copy_of_a_seed_helper_is_still_refused() {
    let divergent = MAJ_COPY.replace("^ (y & z)", "");
    let (code, err, artifact, _p) =
        build_historical(&format!("{MASK32_COPY}\n{CH_COPY}\n{divergent}\n{MAIN}"));
    assert_ne!(code, 0, "a divergent copy must refuse: {err}");
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("`sha_maj` collides with a std seed-blob symbol")
            && err.contains("closure.std_symbol_collision"),
        "refusal must be owned by the std-collision rule and name `sha_maj`: {err}"
    );
}

/// The same divergent copy in an image that keeps the seed (a call into
/// `std.sha256`): std's own `sha_maj` callers exist, so every compiler refuses.
#[test]
fn divergent_copy_in_an_open_image_is_still_refused() {
    let divergent = MAJ_COPY.replace("^ (y & z)", "");
    let (code, err, artifact, _p) = build(&format!(
        "{SEED_CALL}{MASK32_COPY}\n{CH_COPY}\n{divergent}\n{MAIN}"
    ));
    assert_ne!(code, 0, "a divergent copy must refuse: {err}");
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("`sha_maj` collides with a std seed-blob symbol")
            && err.contains("closure.std_symbol_collision"),
        "refusal must be owned by the std-collision rule and name `sha_maj`: {err}"
    );
}

/// The lane-marked compiler omits the seed of the closed divergent image
/// before the identical-copy guard, as 8b12cbee did (exit 16 there too): no
/// std caller of `sha_maj` is left and the user's bodies run:
/// (5 & 3) ^ (5 & 6) = 5, plus 11.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn divergent_copy_in_a_closed_image_omits_the_seed_and_runs() {
    let divergent = MAJ_COPY.replace("^ (y & z)", "");
    let (code, err, artifact, p) = build(&format!("{MASK32_COPY}\n{CH_COPY}\n{divergent}\n{MAIN}"));
    assert_eq!(code, 0, "a closed image omits the seed: {err}");
    let bytes = artifact.expect("artifact written");
    assert_native_result(&p, &bytes, 16, "sha_maj'(5,3,6) + sha_ch(12,10,3) == 16");
}

/// The seed helper is private; a public user copy is a different item and is
/// not covered by the identity rule.
#[test]
fn public_copy_of_a_private_seed_helper_is_still_refused() {
    let public = MASK32_COPY.replace("fn sha_mask32", "pub fn sha_mask32");
    let (code, err, artifact, _p) = build(&format!("{public}\n{CH_COPY}\n{MAJ_COPY}\n{MAIN}"));
    assert_ne!(code, 0, "a public copy must refuse: {err}");
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("`sha_mask32` collides with a std seed-blob symbol"),
        "refusal must name `sha_mask32`: {err}"
    );
}

/// An exact copy whose callee the user does not define stays OPEN: the user
/// closure must not silently bind `sha_mask32` to the seed's private body.
#[test]
fn exact_copy_with_an_undefined_callee_is_refused_as_unresolved() {
    let (code, err, artifact, _p) = build(&format!("{CH_COPY}\n{MAJ_COPY}\n{MAIN}"));
    assert_ne!(code, 0, "an open copy must refuse: {err}");
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("closure.unresolved_edge") && err.contains("`sha_mask32`"),
        "refusal must be the unresolved edge to `sha_mask32`: {err}"
    );
}
