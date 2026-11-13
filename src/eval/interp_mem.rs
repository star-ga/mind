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
//! results match compiled results: `load_i8`, `load_i16`, and `load_i32`
//! zero-extend to i64, stores return 0, i64 values are 8 bytes
//! little-endian (both supported substrates are little-endian).

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

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
    /// Requested extents keyed by their deterministic arena address. The
    /// backing vector is rounded for alignment, but padding is not readable.
    static ALLOCS: RefCell<BTreeMap<i64, usize>> = const { RefCell::new(BTreeMap::new()) };
    /// Offsets returned by `materialize_str` — the pointers to native string
    /// RECORDS marshalled from a `Value::Str` at the `string`-typed-param
    /// boundary. Tracked so `==`/`!=` between two of them can fail loud instead
    /// of silently comparing arena offsets. Membership-only (never iterated to
    /// produce a value) and a `BTreeSet`, so it introduces no non-determinism.
    static STR_RECORDS: RefCell<BTreeSet<i64>> = const { RefCell::new(BTreeSet::new()) };
}

/// Clear the arena (bump pointer back to base) and the string-record tag set.
/// Called once per `mindc test` case so each test sees identical fresh memory
/// regardless of which worker thread runs it or what ran before — run-to-run
/// and order-independent.
pub fn reset() {
    MEM.with(|m| m.borrow_mut().clear());
    ALLOCS.with(|a| a.borrow_mut().clear());
    STR_RECORDS.with(|s| s.borrow_mut().clear());
    super::zeroed_array::reset();
    // The file-descriptor table lives beside the arena. A worker thread is
    // reused across tests, so a descriptor left open must not change the
    // next test's descriptor numbers.
    super::extern_c_intrinsics::reset_fake_fds();
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
            | "__mind_calloc"
            | "__mind_realloc"
            | "__mind_free"
            | "__mind_load_i8"
            | "__mind_store_i8"
            | "__mind_load_i16"
            | "__mind_store_i16"
            | "__mind_load_i32"
            | "__mind_store_i32"
            | "__mind_load_i64"
            | "__mind_store_i64"
            | "__mind_vec_zeroed"
    )
}

/// Evaluate one memory intrinsic over the thread-local arena.
///
/// All memory intrinsics are `(i64...) -> i64` by ABI; the caller has
/// already reduced the arguments to scalars.
pub(crate) fn eval_intrinsic(name: &str, args: &[i64]) -> Result<i64, String> {
    match (name, args) {
        ("__mind_alloc", [n]) => alloc(*n),
        ("__mind_calloc", [n]) => alloc(*n),
        ("__mind_realloc", [addr, new_size]) => realloc(*addr, *new_size),
        ("__mind_free", [addr]) => free(*addr),
        ("__mind_load_i8", [addr]) => load_bytes::<1>(*addr).map(|b| b[0] as i64),
        ("__mind_store_i8", [addr, val]) => store_bytes(*addr, &[(*val & 0xFF) as u8]),
        ("__mind_load_i16", [addr]) => load_bytes::<2>(*addr).map(|b| u16::from_le_bytes(b) as i64),
        ("__mind_store_i16", [addr, val]) => store_bytes(*addr, &(*val as i16).to_le_bytes()),
        ("__mind_load_i32", [addr]) => load_bytes::<4>(*addr).map(|b| u32::from_le_bytes(b) as i64),
        ("__mind_store_i32", [addr, val]) => store_bytes(*addr, &(*val as i32).to_le_bytes()),
        ("__mind_load_i64", [addr]) => load_bytes::<8>(*addr).map(i64::from_le_bytes),
        ("__mind_store_i64", [addr, val]) => store_bytes(*addr, &val.to_le_bytes()),
        ("__mind_vec_zeroed", [n]) => vec_zeroed(*n),
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
    let requested = usize::try_from(n)
        .map_err(|_| "__mind_alloc: size does not fit the host address model".to_string())?;
    let n = requested
        .checked_add(7)
        .and_then(|v| v.checked_div(8))
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| "__mind_alloc: size arithmetic overflow".to_string())?;
    MEM.with(|m| {
        let mut mem = m.borrow_mut();
        let top = mem.len();
        let end = top
            .checked_add(n)
            .ok_or_else(|| "__mind_alloc: arena size arithmetic overflow".to_string())?;
        if end > MEM_CAP_BYTES {
            return Err(format!(
                "__mind_alloc: interpreter memory cap exceeded ({} + {} > {} bytes)",
                top, n, MEM_CAP_BYTES
            ));
        }
        let addr = MEM_BASE
            .checked_add(
                i64::try_from(top).map_err(|_| "__mind_alloc: address overflow".to_string())?,
            )
            .ok_or_else(|| "__mind_alloc: address overflow".to_string())?;
        mem.resize(end, 0);
        ALLOCS.with(|a| a.borrow_mut().insert(addr, requested));
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
    // Tag this record pointer so every VALUE-semantics operation between two
    // marshalled strings (the full comparison class and field sugar — see
    // `is_str_record`) fails loud rather than silently using arena offsets.
    STR_RECORDS.with(|set| set.borrow_mut().insert(rec));
    Ok(rec)
}

/// Is `offset` the pointer to a native string RECORD materialized from a
/// `Value::Str` at the `string`-typed-param boundary (`materialize_str`)?
///
/// The predicate behind the marshalled-string-record CLASS guard
/// (`guard_str_record_value_op` in eval/mod.rs, consulted by both int-op
/// dispatchers, plus the field accessor `eval_field_access`): the whole
/// VALUE-semantics operation class on two such records — every comparison
/// (`== != < <= > >=`) and field sugar (`.len`) — fails loud, while POINTER
/// semantics (`rec + 8`, `__mind_load_*`, `rec == 0` null checks) stay legal.
/// Deterministic: `BTreeSet` membership only, never iterated for ordering.
pub(crate) fn is_str_record(offset: i64) -> bool {
    STR_RECORDS.with(|set| set.borrow().contains(&offset))
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

/// Minimum block for `realloc(NULL, 0)`. glibc `malloc(0)` returns a live
/// pointer; `alloc(0)` returns 0, so this edge allocates the same 24 bytes
/// the native-lane guard does (`src/build/native_bridge/alloc_guard.rs`).
const REALLOC_NULL_ZERO_BYTES: i64 = 24;

/// `__mind_realloc` — grow or shrink an allocation. `addr == 0` and
/// `new_size > 0` is `alloc(new_size)`. `addr == 0` and `new_size == 0` is
/// [`REALLOC_NULL_ZERO_BYTES`]. On success, copies min(old_size, new_size)
/// bytes to the new address and returns it. The old allocation remains valid
/// (bump-arena semantics: `free` is a no-op), matching the self-host
/// compiler's arena. A non-positive size of a live address returns 0.
fn realloc(addr: i64, new_size: i64) -> Result<i64, String> {
    if addr == 0 {
        if new_size == 0 {
            return alloc(REALLOC_NULL_ZERO_BYTES);
        }
        return alloc(new_size);
    }
    if new_size <= 0 {
        return Ok(0);
    }
    let old_size = ALLOCS.with(|a| {
        a.borrow()
            .get(&addr)
            .copied()
            .ok_or_else(|| format!("__mind_realloc: invalid address {addr}"))
    })?;
    let new_addr = alloc(new_size)?;
    let copy_len = old_size.min(new_size as usize);
    if copy_len > 0 {
        let data = load_bytes_dynamic(addr, copy_len)?;
        store_bytes(new_addr, &data)?;
    }
    Ok(new_addr)
}

/// Bounds-checked read of `size` bytes at `addr` (dynamic-size variant).
fn load_bytes_dynamic(addr: i64, size: usize) -> Result<Vec<u8>, String> {
    MEM.with(|m| {
        let mem = m.borrow();
        let off = check_range(addr, size, mem.len())?;
        Ok(mem[off..off + size].to_vec())
    })
}

/// `__mind_vec_zeroed` — allocate `n` zeroed i64 elements and return the
/// base address. Matches `runtime-support/mind_intrinsics.c`: a non-positive
/// `n` allocates nothing and returns 0.
fn vec_zeroed(n: i64) -> Result<i64, String> {
    if n <= 0 {
        super::zeroed_array::register(0, 0);
        return Ok(0);
    }
    let bytes = n
        .checked_mul(8)
        .ok_or_else(|| "__mind_vec_zeroed: size overflow".to_string())?;
    let base = alloc(bytes)?;
    // An `array<T>` slot that takes this buffer needs its length.
    super::zeroed_array::register(base, usize::try_from(n).unwrap_or(0));
    Ok(base)
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
pub(crate) fn store_bytes(addr: i64, bytes: &[u8]) -> Result<i64, String> {
    MEM.with(|m| {
        let mut mem = m.borrow_mut();
        let off = check_range(addr, bytes.len(), mem.len())?;
        mem[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(0)
    })
}

/// Bytes still inside the requested allocation that starts at or before
/// `addr`. One-past-the-end is `0`. An address outside every allocation is
/// an error, the same fail-closed rule as a load or a store.
pub(crate) fn writable_suffix(addr: i64) -> Result<usize, String> {
    if addr < MEM_BASE {
        return Err(format!(
            "memory access out of bounds: address {addr} below arena base {MEM_BASE}"
        ));
    }
    let allocation = ALLOCS.with(|a| {
        a.borrow()
            .range(..=addr)
            .next_back()
            .map(|(&start, &extent)| (start, extent))
    });
    let Some((start, extent)) = allocation else {
        return Err(format!(
            "memory access out of bounds: address {addr} outside every requested allocation"
        ));
    };
    let within = usize::try_from(addr - start)
        .map_err(|_| format!("memory access out of bounds: invalid address {addr}"))?;
    if within > extent {
        return Err(format!(
            "memory access out of bounds: address {addr} outside requested allocation at {start}"
        ));
    }
    Ok(extent - within)
}

/// Map an intrinsic address to an arena offset, rejecting any access that is
/// not fully inside the allocated region.
fn check_range(addr: i64, size: usize, mem_len: usize) -> Result<usize, String> {
    if addr < MEM_BASE {
        return Err(format!(
            "memory access out of bounds: address {addr} below arena base {MEM_BASE}"
        ));
    }
    let off = usize::try_from(addr - MEM_BASE)
        .map_err(|_| format!("memory access out of bounds: invalid address {addr}"))?;
    let end = off.checked_add(size);
    let allocation = ALLOCS.with(|a| {
        a.borrow()
            .range(..=addr)
            .next_back()
            .map(|(&start, &extent)| (start, extent))
    });
    let extent_ok = allocation.is_some_and(|(start, extent)| {
        usize::try_from(addr - start)
            .ok()
            .is_some_and(|within| within <= extent && size <= extent - within)
    });
    if end.is_none_or(|v| v > mem_len) || !extent_ok {
        let extent = allocation.map_or_else(
            || "every requested allocation".to_string(),
            |(start, extent)| {
                i64::try_from(extent)
                    .ok()
                    .and_then(|extent| start.checked_add(extent))
                    .map_or_else(
                        || format!("requested allocation at {start} with {extent} bytes"),
                        |end| format!("requested allocation extent [{start}, {end})"),
                    )
            },
        );
        return Err(format!(
            "memory access out of bounds: [{addr}, {addr}+{size}) outside {extent} \
             (arena top {})",
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
        assert!(alloc(i64::MAX).is_err());
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
    fn allocation_extent_rejects_padding_and_cross_allocation_reads() {
        reset();
        let p = alloc(1).unwrap();
        let q = alloc(8).unwrap();
        let padding_error = eval_intrinsic("__mind_load_i8", &[p + 1]).unwrap_err();
        assert!(
            padding_error.contains("requested allocation extent [8, 9)"),
            "missing requested extent in: {padding_error}"
        );
        assert!(eval_intrinsic("__mind_load_i64", &[p]).is_err());
        assert!(eval_intrinsic("__mind_load_i64", &[q - 1]).is_err());
        assert!(eval_intrinsic("__mind_store_i8", &[p + 1, 7]).is_err());
        assert_eq!(eval_intrinsic("__mind_load_i8", &[q]).unwrap(), 0);
    }

    #[test]
    fn string_records_and_reset_keep_valid_boundaries() {
        reset();
        let rec = materialize_str("abc").unwrap();
        assert!(is_str_record(rec));
        assert_eq!(eval_intrinsic("__mind_load_i64", &[rec + 8]).unwrap(), 3);
        reset();
        assert!(!is_str_record(rec));
        let p = alloc(8).unwrap();
        assert_eq!(p, MEM_BASE);
        assert_eq!(eval_intrinsic("__mind_store_i64", &[p, 42]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_load_i64", &[p]).unwrap(), 42);
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

    #[test]
    fn i32_load_zero_extends_and_does_not_clobber_neighbor() {
        reset();
        let buf = alloc(16).unwrap();
        // Same contract as the i32 surface probe: two adjacent u32 stores,
        // then a high-bit value that must come back as 4294967295, not -1.
        eval_intrinsic("__mind_store_i32", &[buf, 2_864_434_397]).unwrap();
        eval_intrinsic("__mind_store_i32", &[buf + 4, 287_454_020]).unwrap();
        assert_eq!(
            eval_intrinsic("__mind_load_i32", &[buf]).unwrap(),
            2_864_434_397
        );
        assert_eq!(
            eval_intrinsic("__mind_load_i32", &[buf + 4]).unwrap(),
            287_454_020
        );
        eval_intrinsic("__mind_store_i32", &[buf + 8, 4_294_967_295]).unwrap();
        assert_eq!(
            eval_intrinsic("__mind_load_i32", &[buf + 8]).unwrap(),
            4_294_967_295
        );
        assert_eq!(
            eval_intrinsic("__mind_load_i32", &[buf]).unwrap(),
            2_864_434_397
        );
        let mut mask = 0i64;
        if eval_intrinsic("__mind_load_i32", &[buf]).unwrap() == 2_864_434_397
            && eval_intrinsic("__mind_load_i32", &[buf + 4]).unwrap() == 287_454_020
        {
            mask += 1;
        }
        if eval_intrinsic("__mind_load_i32", &[buf + 8]).unwrap() == 4_294_967_295 {
            mask += 2;
        }
        assert_eq!(mask, 3);
    }

    #[test]
    fn i16_load_zero_extends_and_store_keeps_width() {
        reset();
        let p = alloc(4).unwrap();
        eval_intrinsic("__mind_store_i16", &[p, 65535]).unwrap();
        eval_intrinsic("__mind_store_i16", &[p + 2, 1]).unwrap();
        assert_eq!(eval_intrinsic("__mind_load_i16", &[p]).unwrap(), 65535);
        assert_eq!(eval_intrinsic("__mind_load_i16", &[p + 2]).unwrap(), 1);
    }

    #[test]
    fn realloc_copies_bytes_and_zero_addr_allocates() {
        reset();
        let p = alloc(4).unwrap();
        eval_intrinsic("__mind_store_i32", &[p, 2_864_434_397]).unwrap();
        let q = eval_intrinsic("__mind_realloc", &[p, 8]).unwrap();
        assert_ne!(q, 0);
        assert_eq!(
            eval_intrinsic("__mind_load_i32", &[q]).unwrap(),
            2_864_434_397
        );
        assert_eq!(eval_intrinsic("__mind_load_i32", &[q + 4]).unwrap(), 0);
        let fresh = eval_intrinsic("__mind_realloc", &[0, 4]).unwrap();
        assert_ne!(fresh, 0);
        assert_eq!(eval_intrinsic("__mind_load_i32", &[fresh]).unwrap(), 0);
        // realloc(NULL, 0) is a live minimum block; a negative size is 0.
        let null_zero = eval_intrinsic("__mind_realloc", &[0, 0]).unwrap();
        assert_ne!(null_zero, 0);
        assert_eq!(eval_intrinsic("__mind_load_i64", &[null_zero]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_realloc", &[0, -5]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_realloc", &[p, 0]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_realloc", &[p, -1]).unwrap(), 0);
        assert_eq!(
            eval_intrinsic("__mind_load_i32", &[p]).unwrap(),
            2_864_434_397
        );
    }

    #[test]
    fn calloc_returns_zeroed_bytes() {
        reset();
        let p = eval_intrinsic("__mind_calloc", &[8]).unwrap();
        assert_ne!(p, 0);
        assert_eq!(eval_intrinsic("__mind_load_i64", &[p]).unwrap(), 0);
    }

    #[test]
    fn vec_zeroed_is_zero_and_nonpositive_returns_zero() {
        reset();
        assert_eq!(eval_intrinsic("__mind_vec_zeroed", &[0]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_vec_zeroed", &[-1]).unwrap(), 0);
        let p = eval_intrinsic("__mind_vec_zeroed", &[2]).unwrap();
        assert_ne!(p, 0);
        assert_eq!(eval_intrinsic("__mind_load_i64", &[p]).unwrap(), 0);
        assert_eq!(eval_intrinsic("__mind_load_i64", &[p + 8]).unwrap(), 0);
    }
}
