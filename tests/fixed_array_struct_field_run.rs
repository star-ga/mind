// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Executable MLIR gate for fixed arrays stored in struct fields.
//!
//! The test deliberately emits and loads a real shared artifact.  The
//! interpreter/check path alone cannot catch an aggregate field being lowered
//! as an opaque i64 record slot.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use libloading::{Library, Symbol};
use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;
use std::process::Command;

const SOURCE: &str = r#"
type Elem = i64
type Words = [Elem; 2]
type Four = [i64; 4]
type TwoWords = [i64; 2]
type ThreeWords = [i64; 3]

struct S { xs: [i64; 4], tail: i64 }
struct AliasS { xs: Four }
struct F { xs: [f64; 2] }
struct Inner { xs: [i64; 2] }
struct Outer { inner: Inner }
struct One { xs: [i64; 1], tail: i64 }
struct Empty { xs: [i64; 0], tail: i64 }
struct Wide { xs: [i64; 64] }
struct LeftOwner { xs: TwoWords }
struct RightOwner { xs: ThreeWords }
struct Narrow { xs: [u8; 4], sibling: i64 }
struct SignedNarrow { xs: [i8; 4] }
struct Narrow16 { xs: [u16; 4], sibling: i64 }
struct SignedNarrow16 { xs: [i16; 4], sibling: i64 }
struct AccessTrace { order: i64 }

fn traced_receiver(t: AccessTrace, s: Narrow) -> Narrow {
    t.order = t.order * 10 + 1
    return s
}

fn traced_index(t: AccessTrace) -> i64 {
    t.order = t.order * 10 + 2
    return 1
}

fn traced_value(t: AccessTrace) -> i64 {
    t.order = t.order * 10 + 3
    return 255
}

pub fn narrow_read_once() -> i64 {
    let t = AccessTrace { order: 0 }
    let s = Narrow { xs: [0, 128, 127, 255], sibling: 900 }
    let got = traced_receiver(t, s).xs[traced_index(t)]
    return t.order * 1000 + got
}

pub fn narrow_write_once() -> i64 {
    let t = AccessTrace { order: 0 }
    let s = Narrow { xs: [0, 128, 127, 255], sibling: 900 }
    traced_receiver(t, s).xs[traced_index(t)] = traced_value(t)
    return t.order * 10000 + s.xs[1] + s.sibling
}

fn read_struct(s: S) -> i64 {
    let a = s.xs
    return a[2] + s.tail
}

fn replace(s: S) -> S {
    s.xs = [1, 2, 9, 4]
    return s
}

fn index_mut(s: S) -> i64 {
    s.xs[1] = 77
    return s.xs[1] + s.tail
}

fn alias_field(s: AliasS) -> i64 { return s.xs[3] }

fn nested_index_mut(o: Outer) -> i64 {
    o.inner.xs[1] = 9
    return o.inner.xs[1]
}

fn branch_index_mut(s: S, flag: i64) -> i64 {
    if flag == 1 { s.xs[1] = 9 }
    return s.xs[1]
}

fn loop_index_mut(s: S) -> i64 {
    let mut i: i64 = 0
    while i < 2 { s.xs[i] = i + 5; i = i + 1 }
    return s.xs[1]
}

fn take(values: [i64; 2]) -> i64 { return values[1] }
fn make() -> [i64; 2] { return [8, 9] }
fn alias_take(values: Words) -> Elem { return values[1] }

fn float_read(s: F) -> f64 { return s.xs[1] }
fn one_read(s: One) -> i64 { return s.xs[0] + s.tail }
fn empty_read(s: Empty) -> i64 { return s.tail }
fn indexed_param4(s: S, i: i64) -> i64 { return s.xs[i] }
fn indexed_param64(s: Wide, i: i64) -> i64 { return s.xs[i] }

fn narrow_scan(s: Narrow) -> i64 {
    let mut i: i64 = 0
    let mut total: i64 = 0
    while i < 4 { total = total + s.xs[i]; i = i + 1 }
    return total
}

fn narrow_copy(s: Narrow) -> i64 {
    let values = s.xs
    return values[3] + values[0]
}

fn narrow_update(s: Narrow) -> i64 {
    s.xs[1] = 255
    return s.xs[1] + s.xs[0] + s.sibling
}

fn signed_narrow_scan(s: SignedNarrow) -> i64 {
    return s.xs[0] + s.xs[1] + s.xs[2] + s.xs[3]
}

fn narrow16_scan(s: Narrow16) -> i64 {
    let mut i: i64 = 0
    let mut total: i64 = 0
    while i < 4 { total = total + s.xs[i]; i = i + 1 }
    return total
}

fn narrow16_copy(s: Narrow16) -> i64 {
    let values = s.xs
    return values[3] + values[0]
}

fn narrow16_update(s: Narrow16) -> i64 {
    s.xs[1] = 65535
    return s.xs[1] + s.xs[0] + s.sibling
}

fn signed_narrow16_scan(s: SignedNarrow16) -> i64 {
    return s.xs[0] + s.xs[1] + s.xs[2] + s.xs[3]
}

fn signed_narrow16_copy(s: SignedNarrow16) -> i64 {
    let values = s.xs
    return values[0] + values[3]
}

fn signed_narrow16_update(s: SignedNarrow16) -> i64 {
    s.xs[2] = 65535
    return s.xs[0] + s.xs[2] + s.sibling
}

pub fn run() -> i64 {
    let s = S { xs: [10, 20, 30, 40], tail: 7 }
    let s = replace(s)
    let t = S { xs: [10, 20, 30, 40], tail: 7 }
    let a = AliasS { xs: [1, 2, 3, 6] }
    let o = Outer { inner: Inner { xs: [3, 4] } }
    return read_struct(s) + take(make()) + alias_take([11, 12]) + index_mut(t) + alias_field(a) + nested_index_mut(o) + branch_index_mut(t, 1) + loop_index_mut(t) + one_read(One { xs: [5], tail: 7 }) + empty_read(Empty { xs: [], tail: 13 })
}

pub fn float_run() -> f64 {
    let s = F { xs: [1.25, 2.5] }
    return float_read(s)
}

pub fn float_signed_zero() -> f64 {
    let s = F { xs: [-0.0, -3.25] }
    return s.xs[0]
}

pub fn indexed_read4() -> i64 {
    let s = S { xs: [10, 20, 30, 40], tail: 7 }
    return s.xs[2]
}

pub fn indexed_read64() -> i64 {
    let s = Wide { xs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] }
    return s.xs[37]
}

pub fn same_owner_left() -> i64 {
    let s = LeftOwner { xs: [11, 22] }
    return s.xs[1]
}

pub fn same_owner_right() -> i64 {
    let s = RightOwner { xs: [31, 32, 43] }
    return s.xs[2]
}

pub fn narrow_u8_run() -> i64 {
    let s = Narrow { xs: [0, 127, 128, 255], sibling: 900 }
    return narrow_scan(s) + narrow_copy(s) + narrow_update(s)
}

pub fn narrow_i8_run() -> i64 {
    let s = SignedNarrow { xs: [-128, -1, 0, 127] }
    return signed_narrow_scan(s)
}

pub fn narrow_u16_run() -> i64 {
    let s = Narrow16 { xs: [0, 32767, 32768, 65535], sibling: 900 }
    return narrow16_scan(s) + narrow16_copy(s) + narrow16_update(s)
}

pub fn narrow_i16_run() -> i64 {
    let s = SignedNarrow16 { xs: [-32768, -1, 0, 32767], sibling: 900 }
    return signed_narrow16_scan(s) + signed_narrow16_copy(s) + signed_narrow16_update(s)
}
"#;

#[test]
fn fixed_array_struct_fields_run_native_artifact() {
    let mindc = common::mindc_bin();
    if !mindc.exists() {
        common::gate::skipped(
            "fixed_array_struct_field_run",
            "fixed-array-struct-field-run: mindc not found; skipping",
        );
        return;
    }
    let dir = tempfile::tempdir().expect("fixed-array scratch");
    let source = dir.path().join("fixed_array_struct_field.mind");
    let shared = dir.path().join("fixed_array_struct_field.so");
    std::fs::write(&source, SOURCE).expect("write fixed-array source");
    let out = Command::new(&mindc)
        .arg(&source)
        .arg("--emit-shared")
        .arg(&shared)
        .output()
        .expect("run mindc");
    assert!(
        out.status.success() && shared.is_file(),
        "fixed-array struct-field compile failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    unsafe {
        let lib = Library::new(&shared).expect("load fixed-array shared library");
        let run: Symbol<unsafe extern "C" fn() -> i64> = lib.get(b"run").expect("load run");
        let float_run: Symbol<unsafe extern "C" fn() -> f64> =
            lib.get(b"float_run").expect("load float_run");
        let float_signed_zero: Symbol<unsafe extern "C" fn() -> f64> = lib
            .get(b"float_signed_zero")
            .expect("load float_signed_zero");
        let indexed_read4: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"indexed_read4").expect("load indexed_read4");
        let indexed_read64: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"indexed_read64").expect("load indexed_read64");
        let same_owner_left: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"same_owner_left").expect("load same_owner_left");
        let same_owner_right: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"same_owner_right").expect("load same_owner_right");
        let narrow_u8_run: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"narrow_u8_run").expect("load narrow_u8_run");
        let narrow_i8_run: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"narrow_i8_run").expect("load narrow_i8_run");
        let narrow_u16_run: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"narrow_u16_run").expect("load narrow_u16_run");
        let narrow_i16_run: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"narrow_i16_run").expect("load narrow_i16_run");
        let narrow_read_once: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"narrow_read_once").expect("load narrow_read_once");
        let narrow_write_once: Symbol<unsafe extern "C" fn() -> i64> = lib
            .get(b"narrow_write_once")
            .expect("load narrow_write_once");
        assert_eq!(run(), 16 + 9 + 12 + 84 + 6 + 9 + 9 + 6 + 12 + 13);
        assert_eq!(float_run().to_bits(), 2.5f64.to_bits());
        assert_eq!(float_signed_zero().to_bits(), (-0.0f64).to_bits());
        assert_eq!(indexed_read4(), 30);
        assert_eq!(indexed_read64(), 37);
        assert_eq!(same_owner_left(), 22);
        assert_eq!(same_owner_right(), 43);
        // scan=510, copy=255, update=1155; the sibling remains untouched.
        assert_eq!(narrow_u8_run(), 1920);
        assert_eq!(narrow_i8_run(), -2);
        // u16 high-bit values remain zero-extended; the update includes the
        // untouched sibling. i16 loads sign-extend, including 65535 -> -1.
        assert_eq!(narrow_u16_run(), 263_040);
        assert_eq!(narrow_i16_run(), -31_872);
        // Decimal event digits detect duplicated, omitted or reordered receiver,
        // index and RHS evaluation. The write also preserves the sibling field.
        assert_eq!(narrow_read_once(), 12_128);
        assert_eq!(narrow_write_once(), 1_231_155);
    }
}

#[test]
fn indexed_struct_field_read_does_not_expand_with_field_length() {
    let module = parser::parse(SOURCE).expect("fixed-array struct source must parse");
    let ir = lower_to_ir(&module).expect("lowering");
    let body = |name: &str| {
        ir.instrs
            .iter()
            .find_map(|instr| match instr {
                Instr::FnDef {
                    name: fn_name,
                    body,
                    ..
                } if fn_name == name => Some(body.as_slice()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing lowered function `{name}`"))
    };
    let small = body("indexed_param4");
    let wide = body("indexed_param64");
    for (name, instructions) in [("small", small), ("wide", wide)] {
        assert!(
            instructions.iter().all(|instr| {
                !matches!(
                    instr,
                    Instr::ArrayLoad { .. }
                        | Instr::ArrayStore { .. }
                        | Instr::ConstArray { .. }
                        | Instr::ConstDenseTensor { .. }
                )
            }),
            "{name} indexed field read unexpectedly materialized an aggregate: {instructions:?}"
        );
        assert!(
            instructions.iter().any(|instr| {
                matches!(instr, Instr::Call { name, .. } if name == "__mind_oob_check")
            }),
            "{name} indexed field read omitted its bounds check: {instructions:?}"
        );
    }
    assert_eq!(small.len(), wide.len(), "indexed read IR expanded with N");

    let narrow = body("narrow_scan");
    let narrow_debug = format!("{narrow:?}");
    assert!(
        narrow_debug.contains("name: \"__mind_load_i8\""),
        "u8 indexed field read did not use its declared-width load: {narrow:?}"
    );
    assert!(
        narrow_debug.contains("name: \"__mind_oob_check\""),
        "u8 indexed field read omitted its bounds check: {narrow:?}"
    );
    let update = body("narrow_update");
    assert!(
        format!("{update:?}").contains("name: \"__mind_store_i8\""),
        "u8 indexed field write did not use its declared-width store: {update:?}"
    );
    let narrow16 = body("narrow16_scan");
    assert!(
        format!("{narrow16:?}").contains("name: \"__mind_load_i16\""),
        "u16 indexed field read did not use its declared-width load: {narrow16:?}"
    );
    let update16 = body("signed_narrow16_update");
    assert!(
        format!("{update16:?}").contains("name: \"__mind_store_i16\""),
        "i16 indexed field write did not use its declared-width store: {update16:?}"
    );
    assert!(
        update16
            .iter()
            .any(|instr| matches!(instr, Instr::ConstI64(_, 8))),
        "i16 indexed field write lost the canonical eight-byte cell stride: {update16:?}"
    );
}

#[test]
fn unsupported_struct_array_cells_are_refused_before_lowering() {
    let mindc = common::mindc_bin();
    if !mindc.exists() {
        common::gate::skipped(
            "fixed_array_struct_field_unsupported",
            "fixed-array-struct-field-unsupported: mindc not found; skipping",
        );
        return;
    }
    let dir = tempfile::tempdir().expect("unsupported fixed-array scratch");
    let source = dir.path().join("unsupported_fixed_array_struct_field.mind");
    let shared = dir.path().join("unsupported_fixed_array_struct_field.so");
    std::fs::write(
        &source,
        "type U64Alias = u64\nstruct I {\n    xs: [i32; 2],\n}\nstruct F {\n    xs: [f32; 2],\n}\nstruct B {\n    xs: [bool; 2],\n}\nstruct U {\n    xs: [u64; 2],\n}\nstruct UA {\n    xs: [U64Alias; 2],\n}\nfn main() -> i64 {\n    let _i = I { xs: [1, 2] };\n    let _f = F { xs: [1.0, 2.0] };\n    let _b = B { xs: [1, 0] };\n    let _u = U { xs: [1, 2] };\n    let _ua = UA { xs: [1, 2] };\n    return 0;\n}\n",
    )
    .expect("write unsupported fixed-array source");
    let check = Command::new(&mindc)
        .arg("check")
        .arg(&source)
        .output()
        .expect("run mindc check");
    assert!(
        check.status.success(),
        "valid fixed-array types must pass check:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    let ir = Command::new(&mindc)
        .arg("--emit-ir")
        .arg(&source)
        .output()
        .expect("run mindc inspection");
    assert!(
        ir.status.success(),
        "valid fixed-array types must pass IR inspection:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&ir.stdout),
        String::from_utf8_lossy(&ir.stderr)
    );
    let build = Command::new(&mindc)
        .arg("--emit-shared")
        .arg(&shared)
        .arg(&source)
        .output()
        .expect("run mindc build");
    assert!(
        !build.status.success(),
        "unsupported fixed-array build unexpectedly passed"
    );
    let build_err = String::from_utf8_lossy(&build.stderr);
    assert!(
        build_err.contains("lower::fixed_struct_array_cell"),
        "build did not preserve structured runnable-capability refusal: {build_err}"
    );
    assert!(
        !build_err.contains("panicked at"),
        "unsupported fixed-array build reached a panic: {build_err}"
    );
}

#[test]
fn returned_record_array_resolver_controls_refuse_without_artifacts() {
    let mindc = common::require_mindc();
    let dir = tempfile::tempdir().expect("returned-record-array control scratch");
    let cases = [
        (
            "scalar-return",
            "struct Item { value: i64 }\nfn make_scalar() -> i64 { return 7; }\nfn main() -> i64 { return make_scalar()[0].value; }\n",
            "E6009",
        ),
        (
            "unknown-return",
            "struct Item { value: i64 }\nfn main() -> i64 { return missing_items()[0].value; }\n",
            "E2003",
        ),
        (
            "bag-record-array",
            "struct Item { value: i64 }\nstruct Bag { items: [Item; 1] }\nfn main(b: Bag) -> i64 { return b.items[0].value; }\n",
            "E6009",
        ),
    ];
    for (name, source_text, expected_code) in cases {
        let source = dir.path().join(format!("{name}.mind"));
        let shared = dir.path().join(format!("{name}.so"));
        std::fs::write(&source, source_text).expect("write resolver control");
        let out = Command::new(&mindc)
            .args([
                source.to_str().unwrap(),
                "--emit-shared",
                shared.to_str().unwrap(),
            ])
            .output()
            .expect("run returned-record-array resolver control");
        let rendered = format!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.status.success(),
            "{name} unexpectedly compiled: {rendered}"
        );
        assert!(
            rendered.contains(expected_code),
            "{name} lost its structured refusal ({expected_code}): {rendered}"
        );
        assert!(
            !rendered.contains("panicked at"),
            "{name} panicked: {rendered}"
        );
        assert!(!shared.exists(), "{name} emitted an artifact");
    }
}

#[test]
fn fixed_array_capability_gate_is_operation_scoped_and_project_aware() {
    let mindc = common::mindc_bin();
    if !mindc.exists() {
        common::gate::skipped(
            "fixed_array_struct_field_boundary",
            "fixed-array-struct-field-boundary: mindc not found; skipping",
        );
        return;
    }
    let root = common::scratch_dir("fixed_array_struct_field_boundary");
    let env_std = format!("{}/std", env!("CARGO_MANIFEST_DIR"));
    let run = |args: &[&str], cwd: Option<&std::path::Path>| {
        let mut command = Command::new(&mindc);
        command.args(args).env("MINDC_STD_DIR", &env_std);
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        command.output().expect("run mindc boundary control")
    };
    let text = |out: &std::process::Output| {
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    };

    // A declaration without a construction/access operation has no backend
    // work and retains base125 runnable behavior.
    let unused = root.join("unused.mind");
    let unused_so = root.join("unused.so");
    std::fs::write(
        &unused,
        "struct Unused { xs: [u8; 2] }\nfn main() -> i64 { return 0 }\n",
    )
    .expect("write unused declaration control");
    let out = run(
        &[
            "--emit-shared",
            unused_so.to_str().unwrap(),
            unused.to_str().unwrap(),
        ],
        None,
    );
    assert!(out.status.success(), "unused declaration: {}", text(&out));
    assert!(
        unused_so.is_file(),
        "unused declaration emitted no artifact"
    );

    // A narrow store must not reinterpret an opaque i64 handle as integer
    // payload and silently keep only its low byte.
    let opaque_rhs = root.join("opaque_rhs.mind");
    let opaque_rhs_so = root.join("opaque_rhs.so");
    std::fs::write(
        &opaque_rhs,
        "struct S { xs: [u8; 1] }\nfn main(s: S) -> i64 { s.xs[0] = \"x\"; return s.xs[0] }\n",
    )
    .expect("write opaque narrow assignment control");
    let out = run(
        &[
            "--emit-shared",
            opaque_rhs_so.to_str().unwrap(),
            opaque_rhs.to_str().unwrap(),
        ],
        None,
    );
    let opaque_rhs_text = text(&out);
    assert_eq!(out.status.code(), Some(1), "opaque rhs: {opaque_rhs_text}");
    assert!(
        opaque_rhs_text.contains("E2036") && !opaque_rhs_text.contains("panicked at"),
        "opaque narrow assignment did not fail structurally: {opaque_rhs_text}"
    );
    assert!(
        !opaque_rhs_so.exists(),
        "opaque narrow assignment emitted an artifact"
    );

    // A fixed-array field with an unsupported handle element still refuses
    // structurally; the narrow integer-cell path is covered separately.
    let mutation = root.join("mutation.mind");
    let mutation_so = root.join("mutation.so");
    std::fs::write(
        &mutation,
        "struct S { xs: [u64; 2] }\nfn main(s: S) -> i64 { s.xs[1] = 3; return 0 }\n",
    )
    .expect("write mutation control");
    let out = run(
        &[
            "--emit-shared",
            mutation_so.to_str().unwrap(),
            mutation.to_str().unwrap(),
        ],
        None,
    );
    let mutation_text = text(&out);
    assert_eq!(out.status.code(), Some(1), "mutation: {mutation_text}");
    assert!(
        mutation_text.contains("lower::fixed_struct_array_cell")
            && !mutation_text.contains("panicked at"),
        "mutation did not fail structurally: {mutation_text}"
    );
    assert!(
        !mutation_so.exists(),
        "mutation refusal emitted an artifact"
    );

    // The same unsupported owner can be used only through a loop body. This
    // exercises the parameter path without a local constructor and both
    // standalone runnable emitters.
    let loop_mutation = root.join("loop_mutation.mind");
    std::fs::write(
        &loop_mutation,
        "struct S { xs: [u64; 2] }\nfn main(s: S) -> i64 { for i in 0..1 { s.xs[i] = 3 } return 0 }\n",
    )
    .expect("write loop mutation control");
    for (flag, output) in [
        ("--emit-obj", root.join("loop-mutation.o")),
        ("--emit-shared", root.join("loop-mutation.so")),
    ] {
        let out = run(
            &[
                flag,
                output.to_str().unwrap(),
                loop_mutation.to_str().unwrap(),
            ],
            None,
        );
        let rendered = text(&out);
        assert_eq!(out.status.code(), Some(1), "loop mutation: {rendered}");
        assert!(
            rendered.contains("lower::fixed_struct_array_cell")
                && !rendered.contains("panicked at"),
            "loop mutation did not fail structurally: {rendered}"
        );
        assert!(!output.exists(), "loop mutation emitted an artifact");
    }

    // Container expressions must recurse into their elements as well; an
    // unsupported read nested in an array literal cannot become a generic
    // MLIR failure or a successful artifact.
    let array_read = root.join("array_read.mind");
    std::fs::write(
        &array_read,
        "struct S { xs: [u64; 2] }\nfn main(s: S) -> i64 { let values = [s.xs[1]]; return 0 }\n",
    )
    .expect("write array read control");
    for (flag, output) in [
        ("--emit-obj", root.join("array-read.o")),
        ("--emit-shared", root.join("array-read.so")),
    ] {
        let out = run(
            &[flag, output.to_str().unwrap(), array_read.to_str().unwrap()],
            None,
        );
        let rendered = text(&out);
        assert_eq!(out.status.code(), Some(1), "array read: {rendered}");
        assert!(
            rendered.contains("lower::fixed_struct_array_cell")
                && !rendered.contains("panicked at"),
            "array read did not fail structurally: {rendered}"
        );
        assert!(!output.exists(), "array read emitted an artifact");
    }

    #[cfg(feature = "cross-module-imports")]
    {
        // Imported schemas use the defining module's qualified owner. The
        // unsupported `a.Bad` operation must block both project emitters while
        // an unused imported declaration remains runnable.
        for (tag, main_src, expect_success, owner) in [
            (
                "used",
                "import a;\nimport b;\npub fn run(p: a.Bad) -> i64 { return p.xs[1] }\n",
                false,
                Some("a.Bad.xs"),
            ),
            (
                "unused",
                "import a;\nimport b;\npub fn run() -> i64 { return 0 }\n",
                true,
                None,
            ),
        ] {
            let project = root.join(format!("project_{tag}"));
            std::fs::create_dir_all(project.join("src")).expect("create boundary project");
            std::fs::write(
                project.join("Mind.toml"),
                "[package]\nname=\"fixed_array_boundary\"\nversion=\"0.1.0\"\n\n[build]\nemit=\"cdylib\"\n",
            )
            .expect("write boundary manifest");
            std::fs::write(
                project.join("src/a.mind"),
                "pub struct Bad { xs: [u64; 2] }\n",
            )
            .expect("write unsupported owner");
            std::fs::write(
                project.join("src/b.mind"),
                "pub struct Bad { xs: [i64; 2] }\n",
            )
            .expect("write supported colliding owner");
            std::fs::write(project.join("src/main.mind"), main_src).expect("write boundary entry");
            for kind in ["object", "cdylib"] {
                let artifact = project.join(format!("{tag}-{kind}.out"));
                // Object projects derive the final object suffix from the
                // requested stem; cdylib honors --out verbatim. Intermediates
                // under target/obj are not final-artifact evidence.
                let expected_artifact = if kind == "object" {
                    artifact.with_extension("o")
                } else {
                    let file_name = artifact.file_name().expect("boundary artifact name");
                    artifact
                        .parent()
                        .expect("boundary artifact parent")
                        .join(format!("lib{}.so", file_name.to_string_lossy()))
                };
                let out = run(
                    &[
                        "build",
                        "--backend=mlir",
                        "--emit",
                        kind,
                        "--no-cache",
                        "--out",
                        artifact.to_str().unwrap(),
                    ],
                    Some(&project),
                );
                let rendered = text(&out);
                assert_eq!(
                    out.status.success(),
                    expect_success,
                    "project {tag}/{kind}: {rendered}"
                );
                if let Some(owner) = owner {
                    assert!(rendered.contains(owner), "project owner lost: {rendered}");
                    assert!(
                        !rendered.contains("panicked at"),
                        "project panicked: {rendered}"
                    );
                    assert!(
                        !artifact.exists() && !expected_artifact.exists(),
                        "project refusal emitted an artifact"
                    );
                } else {
                    assert!(
                        expected_artifact.is_file(),
                        "project unused emitted no final artifact: {}",
                        expected_artifact.display()
                    );
                }
            }
        }
    }
}
