//! Integration test for `mindc inspect` — the mic@3 artifact decoder/differ.
//!
//! `inspect` is the consumer/debug counterpart of `--emit-mic3`: it decodes a
//! mic@3 artifact into a structural summary + the canonical IR + (when present)
//! the evidence chain, and with `--diff` pinpoints the FIRST diverging byte
//! between two artifacts — the tool the self-host byte-identity gates need when
//! a reseed/loop stops being byte-identical. These tests shell out to the built
//! binary so they exercise the real CLI end to end.

mod common;

use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

fn emit_evidence(src_text: &str, tag: &str) -> std::path::PathBuf {
    let dir = common::scratch_dir("mindc_inspect");
    let src = dir.join(format!("mindc_inspect_{tag}.mind"));
    let art = dir.join(format!("mindc_inspect_{tag}.mic3"));
    std::fs::write(&src, src_text).unwrap();
    let out = mindc()
        .arg(&src)
        .arg("--emit-evidence")
        .arg(&art)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "emit-evidence failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    art
}

#[test]
fn inspect_reports_summary_and_evidence() {
    let art = emit_evidence("fn main() -> i64 { 42 }\n", "sum");

    let out = mindc().arg("inspect").arg(&art).output().unwrap();
    assert!(out.status.success(), "inspect exited non-zero");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("instrs:"), "missing structural summary:\n{s}");
    assert!(s.contains("trace_hash:"), "missing evidence chain:\n{s}");
    assert!(
        s.contains("determinism:      deterministic"),
        "missing determinism:\n{s}"
    );
    assert!(s.contains("--- canonical IR ---"), "missing IR body:\n{s}");

    // --json surfaces the same fields, charset-safe.
    let j = mindc()
        .arg("inspect")
        .arg(&art)
        .arg("--json")
        .output()
        .unwrap();
    assert!(j.status.success());
    let js = String::from_utf8_lossy(&j.stdout);
    assert!(
        js.contains("\"attested\":true") && js.contains("\"trace_hash\":\""),
        "bad json: {js}"
    );
}

#[test]
fn inspect_diff_detects_divergence_and_identity() {
    let a = emit_evidence("fn main() -> i64 { 42 }\n", "da");
    let b = emit_evidence("fn main() -> i64 { 43 }\n", "db");

    // Identical artifacts → exit 0, "identical: YES".
    let same = mindc()
        .arg("inspect")
        .arg(&a)
        .arg("--diff")
        .arg(&a)
        .output()
        .unwrap();
    assert_eq!(same.status.code(), Some(0), "self-diff must be identical");
    assert!(String::from_utf8_lossy(&same.stdout).contains("identical:  YES"));

    // Differing artifacts → exit 1, reports the first diverging byte.
    let diff = mindc()
        .arg("inspect")
        .arg(&a)
        .arg("--diff")
        .arg(&b)
        .output()
        .unwrap();
    assert_eq!(
        diff.status.code(),
        Some(1),
        "differing artifacts must exit 1"
    );
    let s = String::from_utf8_lossy(&diff.stdout);
    assert!(
        s.contains("identical:        NO"),
        "missing NO verdict:\n{s}"
    );
    assert!(s.contains("first_diff_byte:"), "missing byte offset:\n{s}");
}

#[test]
fn inspect_malformed_artifact_exits_1_not_panic() {
    let dir = common::scratch_dir("mindc_inspect");
    let bad = dir.join("mindc_inspect_bad.mic3");
    std::fs::write(&bad, b"not a mic3 artifact at all").unwrap();
    let out = mindc().arg("inspect").arg(&bad).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "malformed artifact must fail closed, not panic"
    );

    // --json parse-error path still emits a well-formed object to stdout (like verify).
    let j = mindc()
        .arg("inspect")
        .arg(&bad)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(j.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&j.stdout).contains("\"error\":\""),
        "json parse-error must emit an error object to stdout"
    );

    // --diff of a byte-identical GARBAGE artifact must NOT report "identical: YES"
    // (exit 0) — decodability is part of the exit-0 contract, so it fails closed.
    let self_diff = mindc()
        .arg("inspect")
        .arg(&bad)
        .arg("--diff")
        .arg(&bad)
        .output()
        .unwrap();
    assert_eq!(
        self_diff.status.code(),
        Some(1),
        "byte-identical garbage must not exit 0"
    );
}

/// A native ELF holds no IR, only the MIND PT_NOTE (the sha256 of its seeded
/// mic@3 module). `inspect` prints that note, found through the program
/// headers, and refuses a broken note header instead of printing a value.
#[test]
fn inspect_prints_the_mind_note_of_a_native_elf_and_refuses_a_broken_one() {
    // The frozen Rust-oracle ELF (native_elf_oracle) and a self-host ELF (the
    // backend_native_bridge output of the frozen seed) carry the same note shape.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for rel in [
        "examples/mindc_mind/testdata/native_elf_oracle/add.elf",
        "examples/mindc_mind/testdata/backend_native_bridge/add.elf",
    ] {
        check_note_of(&root.join(rel));
    }
}

fn check_note_of(elf_path: &std::path::Path) {
    let elf = std::fs::read(elf_path).expect("committed native ELF");
    let note: String = elf[elf.len() - 32..]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let out = mindc().arg("inspect").arg(elf_path).output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && s.contains(&format!("mind_note:        {note} (type 1, ")),
        "inspect must print the MIND note of the ELF:\n{s}"
    );
    // Flip one byte of the note name (MIND -> MINE): no MIND note, exit 1.
    let name_at = elf.len() - 40 + 3;
    assert_eq!(elf[name_at], b'D');
    let mut broken = elf.clone();
    broken[name_at] = b'E';
    let bad = common::scratch_dir("mindc_inspect").join("mindc_inspect_badnote.elf");
    std::fs::write(&bad, &broken).unwrap();
    let out = mindc().arg("inspect").arg(&bad).output().unwrap();
    assert_eq!(out.status.code(), Some(1), "a broken note must be refused");
    assert!(String::from_utf8_lossy(&out.stderr).contains("no MIND note"));
    // Flip the descriptor size (32 -> 33): a malformed MIND note, exit 1.
    let mut broken = elf.clone();
    broken[elf.len() - 48] = 33;
    std::fs::write(&bad, &broken).unwrap();
    let out = mindc().arg("inspect").arg(&bad).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "a wrong note size must be refused"
    );
    // A smaller descriptor (32 -> 16) stays inside the segment, so only the
    // descsz guard refuses it.
    let mut broken = elf.clone();
    broken[elf.len() - 48] = 16;
    std::fs::write(&bad, &broken).unwrap();
    let out = mindc().arg("inspect").arg(&bad).output().unwrap();
    assert_eq!(out.status.code(), Some(1), "descsz 16 must be refused");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("malformed MIND note"),
        "descsz 16 must be refused by the descsz guard"
    );
    let _ = std::fs::remove_file(&bad);
}

/// Golden: `inspect` of a mic@3 artifact prints each extern declaration with
/// its narrow parameter spellings (std.fs `open(pathname, flags: i32, mode: i32)`).
#[cfg(feature = "std-surface")]
#[test]
fn inspect_prints_extern_decls_with_narrow_params() {
    let dir = common::scratch_dir("mindc_inspect");
    let src = dir.join("mindc_inspect_extern.mind");
    let art = dir.join("mindc_inspect_extern.mic3");
    std::fs::write(
        &src,
        "extern \"C\" {\n    unsafe fn open(pathname: *const u8, flags: i32, mode: i32) -> i32;\n    \
         unsafe fn isatty(fd: i32) -> i32;\n}\n\nfn main() -> i64 {\n    return 0;\n}\n",
    )
    .unwrap();
    let out = mindc()
        .arg(&src)
        .arg("--emit-mic3")
        .arg(&art)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = mindc().arg("inspect").arg(&art).output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{s}");
    assert!(
        s.contains(
            "  extern fn open(!llvm.ptr, i32, i32) -> i32 varargs=false vararg_hints=[] callconv="
        ),
        "inspect must print the open decl with its i32 params:\n{s}"
    );
    assert!(s.contains("  extern fn isatty(i32) -> i32 "), "{s}");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&art);
}
