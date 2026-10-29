// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The on-disk cache for the runtime-support `.o` behind `--emit-shared`: the
//! content key, the `MINDC_RUNTIME_OBJ_CACHE=0` bypass and the atomic write.
//! Output-neutral by construction: a hit returns the bytes a cold clang
//! compile of the same inputs would produce.

use std::io::Write;
use std::path::Path;

use tempfile::NamedTempFile;

use super::{BuildTools, MIND_RUNTIME_SUPPORT_C, clang_identity_string};

/// Environment variables clang reads that can alter emitted object bytes.
/// Folded into the runtime-obj cache key so an env change never yields a
/// false cache hit (the key is conservative: a different value — or set vs
/// unset — produces a different key, never a wrong reuse).
const CLANG_CACHE_ENV_VARS: &[&str] = &[
    "SOURCE_DATE_EPOCH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "CCC_OVERRIDE_OPTIONS",
    "COMPILER_PATH",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
];

/// Set to `0` to bypass the runtime-support `.o` disk cache entirely (no
/// read, no write). A measurement knob for COLD-path timing only: the cache
/// is output-neutral, so the emitted bytes are identical either way.
const RUNTIME_OBJ_CACHE_ENV: &str = "MINDC_RUNTIME_OBJ_CACHE";

/// Whether the runtime-obj cache is on for a given `MINDC_RUNTIME_OBJ_CACHE`
/// value. Only the exact value `0` disables it; unset or anything else keeps
/// the historical (cached) behaviour.
pub(super) fn runtime_obj_cache_enabled(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

/// Compute the on-disk cache path for the runtime-support `.o`, or `None`
/// if no cache directory is available.
///
/// Key = sha256(version-prefix ++ clang binary identity ++ clang-honoured
/// env ++ canonical arg vector ++ embedded C source bytes). The two random
/// per-build path args (the source-file input and the `-o` output) are
/// excluded — they vary per invocation but cannot change the emitted object
/// bytes under the fixed-basename + `-ffile-prefix-map` determinism contract.
/// The cache directory is machine-global, so the key pins the clang *binary*
/// (path + size + mtime + version), not merely its `--version` banner.
pub(super) fn runtime_obj_cache_path(
    tools: &BuildTools,
    args: &[String],
) -> Option<std::path::PathBuf> {
    // Checked before the clang identity probe, so a disabled cache costs no
    // `--version` spawn, no read and no write.
    if !runtime_obj_cache_enabled(std::env::var(RUNTIME_OBJ_CACHE_ENV).ok().as_deref()) {
        return None;
    }
    let cache_dir = dirs::cache_dir()?.join("mind").join("runtime-obj");

    let clang_identity = clang_identity_string(&tools.clang);

    let mut data: Vec<u8> = Vec::with_capacity(512 + MIND_RUNTIME_SUPPORT_C.len());
    data.extend_from_slice(b"mind-runtime-obj-v2\n");
    data.extend_from_slice(b"clang-identity=\n");
    data.extend_from_slice(clang_identity.as_bytes());
    data.push(b'\n');
    data.extend_from_slice(b"env=\n");
    for var in CLANG_CACHE_ENV_VARS {
        data.extend_from_slice(var.as_bytes());
        data.push(b'=');
        if let Ok(v) = std::env::var(var) {
            data.extend_from_slice(v.as_bytes());
        }
        data.push(b'\n');
    }
    data.extend_from_slice(b"args=\n");
    // Exclude the two random per-build path args (the source input that
    // follows `-x c`, and the `-o` output path). They never affect the
    // object bytes but would otherwise defeat the cache.
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "-o" {
            // Skip "-o" and its operand.
            i += 2;
            continue;
        }
        if a == "-x" {
            // Keep "-x c" (the mode), but the source path is the arg
            // immediately after "c"; the prefix-map arg also embeds the
            // random dir, so normalise it out below.
            data.extend_from_slice(a.as_bytes());
            data.push(b'\n');
            if i + 1 < args.len() {
                data.extend_from_slice(args[i + 1].as_bytes());
                data.push(b'\n');
            }
            // Skip the source-path operand that follows "-x c".
            i += 3;
            continue;
        }
        if let Some(stripped) = a.strip_prefix("-ffile-prefix-map=") {
            // The map RHS (`=.`) is what lands in the object; the LHS is the
            // random tempdir. Record only the stable RHS so the key is
            // path-independent yet flag-sensitive.
            data.extend_from_slice(b"-ffile-prefix-map=");
            if let Some(rhs) = stripped.split_once('=').map(|(_, r)| r) {
                data.extend_from_slice(rhs.as_bytes());
            }
            data.push(b'\n');
            i += 1;
            continue;
        }
        data.extend_from_slice(a.as_bytes());
        data.push(b'\n');
        i += 1;
    }
    data.extend_from_slice(b"source=\n");
    data.extend_from_slice(MIND_RUNTIME_SUPPORT_C.as_bytes());

    let key = crate::build::cache::sha256_hex(&data);
    Some(cache_dir.join(format!("{key}.o")))
}

/// Atomically write the cached runtime `.o` (temp file in the same dir +
/// rename, so a concurrent reader never observes a partial write).
pub(super) fn write_runtime_obj_cache(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = NamedTempFile::new_in(dest.parent().unwrap_or_else(|| Path::new(".")))?;
    tmp.write_all(bytes)?;
    tmp.flush()?;
    // `persist` performs the rename atomically on POSIX.
    tmp.persist(dest).map_err(|e| e.error)?;
    Ok(())
}
