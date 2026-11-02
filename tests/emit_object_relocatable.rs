// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc build --emit object` writes a relocatable object, not an executable.
//!
//! The executable link used to run for this emit too, so `program.o` was a
//! linked PIE executable under an object-file name. The gate builds a small
//! project with `--emit object`, checks the ELF type is `ET_REL`, links the
//! object with the system C driver and runs the result.

#![cfg(all(target_os = "linux", feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::process::Command;

const TARGET: &str = "emit_object_relocatable";
const ET_REL: u16 = 1;

#[test]
fn emit_object_writes_a_relocatable_object_that_links() {
    let project_dir = common::scratch_dir(TARGET).join("proj");
    let _ = std::fs::remove_dir_all(&project_dir);
    std::fs::create_dir_all(&project_dir).expect("create project dir");
    std::fs::write(
        project_dir.join("Mind.toml"),
        "[package]\nname = \"emit_object\"\nversion = \"0.0.1\"\n\n[build]\nentry = \"main.mind\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(
        project_dir.join("main.mind"),
        "fn main() -> i64 {\n    let a = 40;\n    return a + 2 - 42\n}\n",
    )
    .expect("write main.mind");

    let build_out = Command::new(common::mindc_bin())
        .current_dir(&project_dir)
        .env("MINDC_RUNTIME_OBJ_CACHE", "0")
        .args([
            "build",
            "--backend",
            "mlir",
            "--target",
            "cpu",
            "--emit",
            "object",
            "--optimize",
            "release",
            "--no-cache",
            "--out",
            "program",
        ])
        .output()
        .expect("spawn mindc build");
    if !common::gate::compiled(TARGET, &build_out) {
        return;
    }

    let object = project_dir.join("program.o");
    let bytes = std::fs::read(&object).expect("read program.o");
    assert_eq!(&bytes[..4], b"\x7fELF", "program.o is an ELF file");
    let e_type = u16::from_le_bytes([bytes[16], bytes[17]]);
    assert_eq!(
        e_type, ET_REL,
        "program.o must be a relocatable object (ET_REL = 1), got e_type {e_type}"
    );

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let exe = project_dir.join("linked");
    let link = Command::new(&cc)
        .arg(&object)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("spawn cc");
    assert!(
        link.status.success(),
        "the object links into an executable: {}",
        String::from_utf8_lossy(&link.stderr)
    );
    let run = Command::new(&exe).output().expect("run the linked program");
    assert_eq!(
        run.status.code(),
        Some(0),
        "the linked program returns main's value 0"
    );
}
