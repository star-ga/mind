// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Deterministic linear-memory model for the tree-walking interpreter.
//!
//! `mindc test` runs `[test]` fns through the interpreter, not the native
//! artifact, so the `__mind_alloc` / `__mind_load_*` / `__mind_store_*`
//! intrinsics (implemented in C for compiled code, see
//! `runtime-support/mind_intrinsics.c`) need an interpreter-side model for
//! real memory-using MIND (`std.sha256`, `std.json` byte buffers) to run
//! without the native runtime.
//!
//! Design: a thread-local growable byte arena addressed by OFFSET, never by
//! host pointer. `__mind_alloc` is a bump allocator from a fixed non-zero
//! base, so the same program produces the same "addresses" — and therefore
//! the same results — on every run and every host (no address-dependent
//! behavior, per the determinism rules). Memory is zero-initialised (the
//! deterministic choice; native `malloc` contents are undefined, so no
//! correct program can observe a difference). Out-of-range access is a
//! fail-closed error, never a silent garbage read.
//!
//! Load/store semantics mirror the C intrinsics exactly so interpreted
//! results match compiled results: `load_i8` zero-extends to i64, stores
//! return 0, i64 values are 8 bytes little-endian (both supported
//! substrates are little-endian).

use std::cell::RefCell;

/// Fixed non-zero base "address" of the arena. Offsets below this are
/// invalid, so `__mind_alloc` never returns 0 for a successful allocation
/// (MIND code uses `p != 0` as the success check, matching the native ABI).
/// 8-byte aligned so every allocation is too (see `alloc`).
const MEM_BASE: i64 = 8;

/// Hard cap on total arena size — fail-closed against a runaway allocation
/// loop in an interpreted test, deterministic across hosts.
const MEM_CAP_BYTES: usize = 1 << 30; // 1 GiB

thread_local! {
    static MEM: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Clear the arena (bump pointer back to base). Called once per `mindc test`
/// case so each test sees identical fresh memory regardless of which worker
/// thread runs it or what ran before — run-to-run and order-independent.
pub fn reset() {
    MEM.with(|m| m.borrow_mut().clear());
}

/// Is `name` one of the memory intrinsics this model interprets?
///
/// Exactly the set the std byte-buffer code (`std.sha256` / `std.json` /
/// `std.io_canon` / `std.string`) uses; other `__mind_*` names keep their
/// existing fail-loud "unsupported" behavior.
pub(crate) fn is_memory_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "__mind_alloc"
            | "__mind_free"
            | "__mind_load_i8"
            | "__mind_store_i8"
            | "__mind_load_i64"
            | "__mind_store_i64"
    )
}

/// Evaluate one memory intrinsic over the thread-local arena.
///
/// All memory intrinsics are `(i64...) -> i64` by ABI; the caller has
/// already reduced the arguments to scalars.
pub(crate) fn eval_intrinsic(name: &str, args: &[i64]) -> Result<i64, String> {
    match (name, args) {
        ("__mind_alloc", [n]) => alloc(*n),
        ("__mind_free", [addr]) => free(*addr),
        ("__mind_load_i8", [addr]) => load_bytes::<1>(*addr).map(|b| b[0] as i64),
        ("__mind_store_i8", [addr, val]) => store_bytes(*addr, &[(*val & 0xFF) as u8]),
        ("__mind_load_i64", [addr]) => load_bytes::<8>(*addr).map(i64::from_le_bytes),
        ("__mind_store_i64", [addr, val]) => store_bytes(*addr, &val.to_le_bytes()),
        _ => Err(format!(
            "{name}: wrong argument count for memory intrinsic ({} given)",
            args.len()
        )),
    }
}

/// Bump-allocate `n` zeroed bytes; returns the (non-zero) offset address.
/// `n <= 0` returns 0, matching the native `__mind_alloc`.
fn alloc(n: i64) -> Result<i64, String> {
    if n <= 0 {
        return Ok(0);
    }
    // Round the size up to 8 so every allocation is 8-aligned, matching the
    // alignment guarantee compiled code gets from malloc.
    let n = (n as usize).div_ceil(8) * 8;
    MEM.with(|m| {
        let mut mem = m.borrow_mut();
        let top = mem.len();
        if top + n > MEM_CAP_BYTES {
            return Err(format!(
                "__mind_alloc: interpreter memory cap exceeded ({} + {} > {} bytes)",
                top, n, MEM_CAP_BYTES
            ));
        }
        let addr = MEM_BASE + top as i64;
        mem.resize(top + n, 0);
        Ok(addr)
    })
}

/// Materialize a Rust-side string into the arena as a NATIVE string record —
/// the `[addr: i64 | len: i64 | cap: i64]` triple over STRIDE-1 bytes that
/// compiled code passes for a `string`-typed value (std/string.mind layout;
/// consumed by e.g. std/json.mind's `jv_string_value_from_native`). Returns
/// the record pointer. The empty string gets `addr = 0, len = 0, cap = 0`
/// (`alloc(0)` is 0), matching `String::empty()`. Deterministic: bump
/// allocation, little-endian stores, no address randomness beyond the fixed
/// arena base.
pub(crate) fn materialize_str(s: &str) -> Result<i64, String> {
    let bytes = s.as_bytes();
    let n = bytes.len() as i64;
    let data = alloc(n)?;
    if n > 0 {
        store_bytes(data, bytes)?;
    }
    let rec = alloc(24)?;
    store_bytes(rec, &data.to_le_bytes())?;
    store_bytes(rec + 8, &n.to_le_bytes())?;
    store_bytes(rec + 16, &n.to_le_bytes())?;
    Ok(rec)
}

/// `__mind_free` — validated no-op (the arena is a bump allocator, exactly
/// like the self-host compiler's own runtime arena). `free(0)` is allowed,
/// matching C `free(NULL)`; any other address must lie inside the arena.
fn free(addr: i64) -> Result<i64, String> {
    if addr == 0 {
        return Ok(0);
    }
    MEM.with(|m| {
        let len = m.borrow().len();
        if addr < MEM_BASE || addr - MEM_BASE >= len as i64 {
            return Err(format!("__mind_free: invalid address {addr}"));
        }
        Ok(0)
    })
}

/// Bounds-checked read of `N` bytes at `addr`. Fail-closed on any
/// out-of-range access.
fn load_bytes<const N: usize>(addr: i64) -> Result<[u8; N], String> {
    MEM.with(|m| {
        let mem = m.borrow();
        let off = check_range(addr, N, mem.len())?;
        let mut out = [0u8; N];
        out.copy_from_slice(&mem[off..off + N]);
        Ok(out)
    })
}

/// Bounds-checked write of `bytes` at `addr`. Returns 0 (the native store
/// intrinsics' return value). Fail-closed on any out-of-range access.
fn store_bytes(addr: i64, bytes: &[u8]) -> Result<i64, String> {
    MEM.with(|m| {
        let mut mem = m.borrow_mut();
        let off = check_range(addr, bytes.len(), mem.len())?;
        mem[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(0)
    })
}

/// Map an intrinsic address to an arena offset, rejecting any access that is
/// not fully inside the allocated region.
fn check_range(addr: i64, size: usize, mem_len: usize) -> Result<usize, String> {
    if addr < MEM_BASE {
        return Err(format!(
            "memory access out of bounds: address {addr} below arena base {MEM_BASE}"
        ));
    }
    let off = (addr - MEM_BASE) as usize;
    if off.checked_add(size).is_none_or(|end| end > mem_len) {
        return Err(format!(
            "memory access out of bounds: [{addr}, {addr}+{size}) outside allocated \
             arena (top {})",
            MEM_BASE + mem_len as i64
        ));
    }
    Ok(off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_nonzero_zeroed_and_deterministic() {
        reset();
        let p = alloc(8).unwrap();
        assert_ne!(p, 0);
        assert_eq!(eval_intrinsic("__mind_load_i64", &[p]).unwrap(), 0);
        // Same program → same layout on every run: reset and repeat.
        reset();
        assert_eq!(alloc(8).unwrap(), p);
    }

    #[test]
    fn alloc_nonpositive_returns_zero() {
        reset();
        assert_eq!(alloc(0).unwrap(), 0);
        assert_eq!(alloc(-4).unwrap(), 0);
    }

    #[test]
    fn i8_roundtrip_zero_extends() {
        reset();
        let p = alloc(1).unwrap();
        // store_i8 truncates to the low byte; load_i8 zero-extends (matches
        // the C intrinsics).
        assert_eq!(eval_intrinsic("__mind_store_i8", &[p, 0x1FF]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_load_i8", &[p]).unwrap(), 0xFF);
    }

    #[test]
    fn i64_roundtrip_little_endian() {
        reset();
        let p = alloc(8).unwrap();
        eval_intrinsic("__mind_store_i64", &[p, -2]).unwrap();
        assert_eq!(eval_intrinsic("__mind_load_i64", &[p]).unwrap(), -2);
        // Little-endian byte order: low byte first.
        assert_eq!(eval_intrinsic("__mind_load_i8", &[p]).unwrap(), 0xFE);
    }

    #[test]
    fn out_of_bounds_is_fail_closed() {
        reset();
        let p = alloc(8).unwrap();
        // One-past-the-end i64 read straddles the arena top.
        assert!(eval_intrinsic("__mind_load_i64", &[p + 1]).is_err());
        assert!(eval_intrinsic("__mind_load_i8", &[p + 8]).is_err());
        assert!(eval_intrinsic("__mind_store_i8", &[0, 7]).is_err());
        assert!(eval_intrinsic("__mind_load_i8", &[MEM_BASE - 1]).is_err());
    }

    #[test]
    fn free_validates_and_is_noop() {
        reset();
        let p = alloc(8).unwrap();
        assert_eq!(eval_intrinsic("__mind_free", &[0]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_free", &[p]).unwrap(), 0);
        assert!(eval_intrinsic("__mind_free", &[p + 1024]).is_err());
        // The bytes survive free (bump arena semantics).
        eval_intrinsic("__mind_store_i8", &[p, 97]).unwrap();
        assert_eq!(eval_intrinsic("__mind_load_i8", &[p]).unwrap(), 97);
    }
}
