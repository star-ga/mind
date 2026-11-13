// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Checked artifact emission and `mindc inspect` for the compiler CLI.

use std::{fs, process};

use super::CompileArgs;

pub(super) fn emit_mic3_if_requested(
    cli: &CompileArgs,
    products: &libmind::pipeline::CompileProducts,
) {
    let path = match &cli.emit_mic3 {
        Some(p) => p,
        None => return,
    };
    let bytes = libmind::ir::compact::emit_mic3_checked(&products.ir).unwrap_or_else(|err| {
        eprintln!("error[emit-mic3]: cannot encode {path}: {err}");
        process::exit(1);
    });
    if let Err(err) = fs::write(path, &bytes) {
        eprintln!("error[emit-mic3]: failed to write {path}: {err}");
        process::exit(1);
    }
    eprintln!("Wrote mic@3 artifact: {path} ({} bytes)", bytes.len());
}

/// `mindc inspect <artifact> [--diff OTHER]` — decode + pretty-print a mic@3
/// artifact and, with `--diff`, structurally compare two. The consumer/debug
/// counterpart of `--emit-mic3`: it surfaces MIND's deterministic canonical IR
/// and its tamper-evident evidence chain, and — via `--diff` — pinpoints WHERE
/// two artifacts diverge, exactly what the self-host byte-identity gates need
/// when a reseed or loop stops being byte-identical. Not a Rust `objdump` clone:
/// the value is the canonical-IR + evidence surface only MIND's wedge exposes.
///
/// Returns the process exit code: 0 = decoded (and, with `--diff`, identical);
/// 1 = artifacts differ (`--diff`) or a malformed artifact; 2 = I/O error.
pub(super) fn run_inspect(artifact: &str, json: bool, diff: Option<&str>) -> i32 {
    use libmind::ir::compact::{Determinism, mic3_evidence_report, parse_mic3_envelope};

    let bytes = match fs::read(artifact) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("error[inspect]: cannot read artifact {artifact}: {err}");
            return 2;
        }
    };

    // A native ELF carries no IR, only the MIND PT_NOTE: the sha256 of its
    // seeded mic@3 module. Decode and print that note (fail closed on a
    // malformed one); a mic@3 artifact is decoded below.
    if bytes.starts_with(b"\x7fELF") && diff.is_none() {
        return inspect_elf_note(artifact, &bytes, json);
    }

    // --diff MODE: mic@3 is canonical (RFC 0021), so byte-identity IS structural
    // identity. When the bytes differ, locate the first diverging byte and parse
    // each side to report parse status + instruction-count delta (a parse failure
    // is itself a reported difference, never a crash).
    if let Some(other) = diff {
        let other_bytes = match fs::read(other) {
            Ok(b) => b,
            Err(err) => {
                eprintln!("error[inspect]: cannot read artifact {other}: {err}");
                return 2;
            }
        };
        if bytes == other_bytes {
            // Canonical mic@3: byte-identity IS structural identity — but the
            // contract is "exit 0 = decoded", so two byte-identical GARBAGE files
            // (e.g. both zero-length from an aborted build, or a self-diff of a
            // corrupt artifact) must still fail closed rather than report a
            // confident "identical: YES".
            if let Err(err) = parse_mic3_envelope(&bytes) {
                eprintln!("error[inspect]: {artifact} did not parse as mic@3: {err:?}");
                if json {
                    println!(
                        "{{\"a\":\"{}\",\"b\":\"{}\",\"identical\":true,\"decoded\":false,\"bytes\":{}}}",
                        super::json_escape(artifact),
                        super::json_escape(other),
                        bytes.len()
                    );
                }
                return 1;
            }
            if json {
                println!(
                    "{{\"a\":\"{}\",\"b\":\"{}\",\"identical\":true,\"decoded\":true,\"bytes\":{}}}",
                    super::json_escape(artifact),
                    super::json_escape(other),
                    bytes.len()
                );
            } else {
                println!("identical:  YES ({} bytes)", bytes.len());
            }
            return 0;
        }
        let common = bytes.len().min(other_bytes.len());
        let first_diff = (0..common)
            .find(|&i| bytes[i] != other_bytes[i])
            .unwrap_or(common);
        let a_instrs = parse_mic3_envelope(&bytes)
            .map(|m| m.instrs.len() as i64)
            .unwrap_or(-1);
        let b_instrs = parse_mic3_envelope(&other_bytes)
            .map(|m| m.instrs.len() as i64)
            .unwrap_or(-1);
        if json {
            println!(
                "{{\"a\":\"{}\",\"b\":\"{}\",\"identical\":false,\"a_bytes\":{},\"b_bytes\":{},\"first_diff_offset\":{},\"a_instrs\":{},\"b_instrs\":{}}}",
                super::json_escape(artifact),
                super::json_escape(other),
                bytes.len(),
                other_bytes.len(),
                first_diff,
                a_instrs,
                b_instrs
            );
        } else {
            let fmt_side = |n: i64| {
                if n < 0 {
                    "PARSE FAIL".to_string()
                } else {
                    format!("{n} instrs")
                }
            };
            println!("identical:        NO");
            println!(
                "a:                {artifact} ({} bytes, {})",
                bytes.len(),
                fmt_side(a_instrs)
            );
            println!(
                "b:                {other} ({} bytes, {})",
                other_bytes.len(),
                fmt_side(b_instrs)
            );
            println!("first_diff_byte:  {first_diff}");
            let lo = first_diff.saturating_sub(4);
            let a_hi = (first_diff + 4).min(bytes.len());
            let b_hi = (first_diff + 4).min(other_bytes.len());
            println!("  a[{lo}..]: {}", super::hex_encode(&bytes[lo..a_hi]));
            println!("  b[{lo}..]: {}", super::hex_encode(&other_bytes[lo..b_hi]));
        }
        return 1;
    }

    // INSPECT MODE: decode + pretty-print one artifact.
    let module = match parse_mic3_envelope(&bytes) {
        Ok(m) => m,
        Err(err) => {
            let reason = format!("{err:?}");
            eprintln!("error[inspect]: {artifact} did not parse as mic@3: {reason}");
            // Mirror run_verify: a scripted --json consumer reads stdout, so emit a
            // well-formed error object there too (not just stderr).
            if json {
                println!(
                    "{{\"artifact\":\"{}\",\"error\":\"{}\"}}",
                    super::json_escape(artifact),
                    super::json_escape(&reason)
                );
            }
            return 1;
        }
    };
    let evidence = mic3_evidence_report(&bytes).ok();
    let det_str = |d: &Determinism| {
        if matches!(d, Determinism::Deterministic) {
            "deterministic"
        } else {
            "nondeterministic"
        }
    };

    if json {
        let (attested, trace_hash, determinism, fp_mode) = match &evidence {
            Some(r) => (
                true,
                super::hex_encode(&r.trace_hash),
                det_str(&r.determinism),
                r.fp_mode.as_str(),
            ),
            None => (false, String::new(), "", ""),
        };
        println!(
            "{{\"artifact\":\"{}\",\"bytes\":{},\"instrs\":{},\"next_id\":{},\"exports\":{},\"attested\":{},\"trace_hash\":\"{}\",\"determinism\":\"{}\",\"fp_mode\":\"{}\"}}",
            super::json_escape(artifact),
            bytes.len(),
            module.instrs.len(),
            module.next_id,
            module.exports.len(),
            attested,
            trace_hash,
            determinism,
            fp_mode
        );
    } else {
        println!("artifact:         {artifact}");
        println!("bytes:            {}", bytes.len());
        println!("instrs:           {}", module.instrs.len());
        println!("ssa_next_id:      {}", module.next_id);
        println!("exports:          {}", module.exports.len());
        match &evidence {
            Some(r) => {
                println!("attested:         YES");
                println!("trace_hash:       {}", super::hex_encode(&r.trace_hash));
                println!("determinism:      {}", det_str(&r.determinism));
                println!("fp_mode:          {}", r.fp_mode.as_str());
            }
            None => println!("attested:         no (no evidence_chain MAP)"),
        }
        println!("--- canonical IR ---");
        println!("{}", libmind::ir::format_ir_module_full(&module));
    }
    0
}

/// The MIND notes of a native ELF: PT_NOTE entries named `MIND` (namesz 5)
/// with a 32-byte descriptor, as `(type, descriptor)` in image order. Type 1
/// is the sha256 of the seeded mic@3 module; type 2 is the source note. `Err`
/// names the first rule the image breaks: a truncated header, a wrong size, a
/// repeated type, or no MIND note at all.
fn mind_notes_of_elf(elf: &[u8]) -> Result<Vec<(u32, [u8; 32])>, String> {
    let u16_at = |at: usize| {
        elf.get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let u32_at = |at: usize| {
        elf.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let u64_at = |at: usize| {
        elf.get(at..at + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    };
    if elf.get(4..6) != Some(&[2, 1]) {
        return Err("not a little-endian ELF64 image".to_string());
    }
    let bad = || "program header table runs past the image".to_string();
    let phoff = u64_at(0x20).ok_or_else(bad)? as usize;
    let (phentsize, phnum) = (u16_at(0x36).ok_or_else(bad)?, u16_at(0x38).ok_or_else(bad)?);
    let mut found: Vec<(u32, [u8; 32])> = Vec::new();
    for i in 0..phnum as usize {
        let ph = phoff + i * phentsize as usize;
        if u32_at(ph).ok_or_else(bad)? != 4 {
            continue;
        }
        let off = u64_at(ph + 8).ok_or_else(bad)? as usize;
        let size = u64_at(ph + 32).ok_or_else(bad)? as usize;
        let seg = elf
            .get(off..off.saturating_add(size))
            .ok_or("a PT_NOTE segment runs past the image")?;
        let mut at = 0;
        while at + 12 <= seg.len() {
            let word = |k: usize| u32::from_le_bytes(seg[at + k..at + k + 4].try_into().unwrap());
            let (namesz, descsz, kind) = (word(0) as usize, word(4) as usize, word(8));
            let name_end = at + 12 + namesz;
            let desc_at = at + 12 + namesz.next_multiple_of(4);
            let next = desc_at + descsz.next_multiple_of(4);
            if next > seg.len() {
                return Err("a note entry runs past its PT_NOTE segment".to_string());
            }
            if seg.get(at + 12..name_end) == Some(b"MIND\0".as_slice()) {
                if descsz != 32 || found.iter().any(|(t, _)| *t == kind) {
                    return Err(format!(
                        "malformed MIND note (type {kind}, descsz {descsz}, or a repeated type)"
                    ));
                }
                found.push((kind, seg[desc_at..desc_at + 32].try_into().unwrap()));
            }
            at = next;
        }
    }
    if found.is_empty() {
        return Err("no MIND note in any PT_NOTE segment".to_string());
    }
    Ok(found)
}

fn inspect_elf_note(artifact: &str, elf: &[u8], json: bool) -> i32 {
    let notes = match mind_notes_of_elf(elf) {
        Ok(notes) => notes,
        Err(reason) => {
            eprintln!("error[inspect]: {artifact}: {reason}");
            return 1;
        }
    };
    if json {
        let rows: Vec<String> = notes
            .iter()
            .map(|(kind, d)| {
                format!(
                    "{{\"type\":{kind},\"sha256\":\"{}\"}}",
                    super::hex_encode(d)
                )
            })
            .collect();
        println!(
            "{{\"artifact\":\"{}\",\"bytes\":{},\"kind\":\"elf\",\"mind_notes\":[{}]}}",
            super::json_escape(artifact),
            elf.len(),
            rows.join(",")
        );
        return 0;
    }
    println!("artifact:         {artifact}");
    println!("bytes:            {}", elf.len());
    println!("kind:             native ELF (the image holds no IR, only MIND notes)");
    for (kind, digest) in &notes {
        let what = match kind {
            1 => "sha256 of the seeded mic@3 module",
            2 => "source note",
            _ => "unknown MIND note type",
        };
        println!(
            "mind_note:        {} (type {kind}, {what})",
            super::hex_encode(digest)
        );
    }
    0
}
