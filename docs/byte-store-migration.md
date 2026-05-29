# Byte-Store Migration — closing `#306`

> **Status (2026-05-27).** Path-B intrinsics shipped (`__mind_load_i8` /
> `__mind_store_i8`, commit `0e7dd6c`). The pure-MIND std site migration to
> consume them — and the bootstrap keystone re-bless that follows — is the
> remaining work. This doc is the reproducible execution spec.

## Why this exists

`std.string` / `std.sha256` / `std.toml` / `std.tui` currently write single
bytes with `__mind_store_i64(addr + i, byte)`, where the intrinsic actually
writes 8 bytes. Each byte-store clobbers the next 7 bytes; subsequent
byte-stores at `i+1`, `i+2`, … overwrite that clobber. The LAST byte-store
of a buffer leaves 7 bytes of stale-zero write past `len`.

The runtime-support C shim currently masks the end-of-buffer OOB with a
7-byte allocation pad (`cc5a513`), so nothing crashes today. But:

- The garbage past `len` is a **cross-substrate bit-identity landmine.**
  NEON / RVV substrates may not have the same pad shape; the AVX2 reference
  oracle then disagrees with NEON byte-for-byte on the same Q16.16 input.
- A `__mind_realloc`-shrink that drops below the pad → real OOB on real
  hardware.
- The OOB-write of zeros is a write into memory you don't own, full stop —
  the rest of the safety story (RFC 0010 region allocator + GenRef) is
  undermined as long as this exists in the standard library.

Path-B (`0e7dd6c`) added `__mind_store_i8` / `__mind_load_i8` to the type-
checker intrinsic table + the runtime-support C shim. The intrinsics are
inert until the std code calls them — this migration switches the call
sites over.

## The exact call sites

Inventory captured 2026-05-27 against `mind@d27a0e8`. Re-verify with
`git grep '__mind_store_i64' std/` before executing.

### `std/string.mind` — 1 site

| Line | Current | Replace with |
|------|---------|---|
| 99   | `__mind_store_i64(new_addr + s.len, b);` | `__mind_store_i8(new_addr + s.len, b);` |

Also matches one byte-read at line 56:
`__mind_load_i64(s.addr + i) & 255` → can stay as-is (the `& 255` mask is
safe; replacing with `__mind_load_i8` is a follow-on cleanup, not required
for the OOB close).

### `std/sha256.mind` — ~25 byte-store sites

All `__mind_store_i64(...)` calls **without** a `* 8` multiplier on the
offset. Pattern: `__mind_store_i64(buf + <byte_offset>, <byte_value>)`.

Confirmed lines: 288, 293, 298, 308–315, 337–340. The header comment block
at lines 11–12 documents the old convention — that comment should be
updated as part of the migration to describe the new `__mind_store_i8` form.

The i64-aligned scratch stores at lines 175, 179, 245–252 (those have
`* 8` or are full-i64-value writes into the working `w[]` and `h[]`
arrays) are **legitimate i64 stores** — do not touch.

### `std/toml.mind` — MIXED (most sites are legitimate i64 stores; only 2 are byte-stores)

| Line | Form | Action |
|------|------|--------|
| 129  | `__mind_store_i64(dst + i, __mind_load_i64(src + i) & 255);` | **migrate** — byte copy. Replace store with `__mind_store_i8` (load side mask is fine). |
| 138  | `__mind_store_i64(dst + i * 8, __mind_load_i64(src + i * 8));` | **keep** — i64-aligned full-word copy. |
| 166, 237, 269, 270 | i64-aligned stores with `* 8` offsets | **keep** — legitimate i64 stores into the keys/vals backing arrays. |
| 498  | `__mind_store_i64(new_dat + old_len, b);` | **migrate** — byte append. Replace with `__mind_store_i8`. |

The TomlValue struct header writes at lines 71–96 (`h + 0`, `h + 8`, `h + 16`, …) are i64-aligned struct ABI — **keep as `__mind_store_i64`.**

### `std/tui.mind` — 0 byte-stores, 1 byte-read

No `__mind_store_i64` byte-store sites. Line 274 has a deliberate
`__mind_load_i64(buf)` followed by shifts to peel 4 bytes — that's a
documented packed-load pattern (see the comment block at line 260),
**not** a bug. Line 457 `__mind_load_i64(payload_addr + i) & 255` is the
same `& 255` mask pattern as `string.mind` — safe.

### `std/fs.mind` — 7 byte-stores (discovered post-inventory, 2026-05-29)

The original inventory (above) was scoped to the keystone-critical modules.
A follow-up `git grep` surfaced the same byte-store OOB pattern in the
`std.fs` POSIX surface (Task #268), which the keystone does not import — so
these never affected the bootstrap oracle, but they were genuine
out-of-bounds heap writes on their own.

| Line | Site | Form |
|------|------|------|
| 101  | `fs_copy_bytes_rec` | byte-by-byte copy `dst + i` |
| 128  | `fs_heap_copy` | NUL append at `dst + src_len` (into `src_len + 1` alloc — true 7-byte OOB) |
| 257  | `read_to_string` | NUL terminator at `buf + n` (into `file_sz + 1` alloc — true OOB when `n == file_sz`) |
| 382  | `mkdir_p_walk` | temp-NUL a `/` separator at `work + i` (mid-buffer clobber) |
| 384  | `mkdir_p_walk` | restore `/` (47) at `work + i` |
| 452  | `remove_recursive_entries` | write `/` (47) at `full + plen` |
| 454  | `remove_recursive_entries` | NUL terminator at `full + full_len` (into `full_len + 1` alloc — true OOB) |

All 7 migrated to `__mind_store_i8`. The i64-aligned struct/word stores in
`fs.mind` (`* 8` offsets and the 24-byte `DirEntry`/`FileStat`/`Vec` record
fields at `+0/+8/+16`) are legitimate and stay `__mind_store_i64`. Load
sites (`__mind_load_i64(...) & 255`) left as-is, same as the other modules.
Behavior verified via real-ELF ctypes round-trip (`store_i8` writes exactly
one byte; mid-buffer and end-of-buffer bytes untouched).

### `std/json.mind`, `std/regex.mind`, `std/process.mind`, `std/net.mind` — 34 byte-stores (discovered 2026-05-29)

A repo-wide `git grep '__mind_store_i64' std/` after the `fs.mind` follow-up
surfaced the same byte-store OOB pattern in four more POSIX/parser-surface
modules. None are imported by the keystone (`mindc_mind`) bootstrap, so they
never affected the oracle — but each is a genuine out-of-bounds heap write on
its own, identical in shape to the `string`/`toml`/`fs` cases.

**Byte-array sites** (1-byte payload into a byte buffer):

| File | Line | Site | Form |
|------|------|------|------|
| `json.mind`    | 168 | `jv_copy_bytes`        | byte-copy loop `dst + i` |
| `json.mind`    | 254 | `jvsb_push`            | byte append at `new_dat + old_len` (into `new_cap` alloc) |
| `regex.mind`   | 1032| `jvsb_push_rx` grow    | byte-copy loop `d + ci` |
| `regex.mind`   | 1037| `jvsb_push_rx` append  | byte append at `new_dat + old_len` |
| `process.mind` | 65  | `proc_copy_bytes_rec`  | byte-by-byte copy `dst + i` |
| `process.mind` | 73  | `proc_heap_copy`       | NUL terminator at `dst + src_len` (into `src_len + 1` alloc — true 7-byte OOB) |

**4-byte syscall-buffer sites** (little-endian `i32`/`socklen_t` written one
byte per index into a `__mind_alloc(4)` — the first `store_i64(buf + 0, …)`
writes 8 bytes into a 4-byte alloc = 4-byte OOB):

| File | Lines | Buffer | Consumed by |
|------|-------|--------|-------------|
| `process.mind` | 237–240 | `status_buf` (4 B) | `waitpid(.., *mut i32, ..)` |
| `net.mind`     | 94–97   | `optval` (4 B)     | `setsockopt(.., optval, 4)` |
| `net.mind`     | 105–108, 158–161, 264–267 | `lenptr` (4 B) ×3 | `getsockname` / `accept` / `recvfrom` |

**16-byte sockaddr** (`net.mind` 78–88, `net_build_sockaddr_in`): the `sa`
buffer is 16 B and the highest write (offset 7) ends in-bounds at byte 15, so
there was no OOB here — but the same writes were migrated to `__mind_store_i8`
for uniformity and to remove the load-bearing reliance on ascending-offset
clobber-then-overwrite ordering.

All 34 store statements migrated to `__mind_store_i8`. The i64-aligned struct/word
stores in these modules (`* 8` offsets, the `+0/+8/+16/+24/+32` record fields,
the JSON value-struct header writes at `h + 0 … h + 128`, the 8-byte handle
store at `net.mind:67`) are legitimate and stay `__mind_store_i64`. Load sites
(`__mind_load_i64(...) & 255`) are left as-is, same as the other modules —
masked-byte reads are a follow-on cleanup, not part of the OOB-write close.
Behavior in every valid region is byte-identical to the old i64 stores: byte
arrays had each trailing clobber immediately overwritten by the next ascending
write; the 4-byte and 16-byte buffers hold small values (`0`/`1`/`2`/`16`/port
and IP octets) whose high bytes were already explicitly written `0`. Confirmed
against a clean-HEAD parse+lower baseline.

## Migration mechanics

Per-file workflow:

1. **Edit** the .mind sources (the lines above) with the documented
   replacements.
2. **Run the std-surface test gate** —
   `cargo test --features "std-surface mlir-lowering"`. Functionality
   should be unchanged; the test suite that was green on `eaa24aa` should
   stay green.
3. **Bootstrap byte-identity** —
   `cargo test --features "std-surface mlir-lowering" --test phase_g_keystone_bootstrap`.
   This **will** initially fail: the libmindc_mind.so emits different bytes
   now (different intrinsic names in the lowered calls). That is the
   keystone re-bless step (next section).
4. **Re-bless the keystone** — the only step that requires care.

## Re-blessing the keystone

The bootstrap byte-identity test (`tests/phase_g_keystone_bootstrap.rs`)
compares `libmindc_mind.so`'s output against the committed oracle at
`examples/mindc_mind/libmindc_mind.so`. When the std sources change, the
oracle must be regenerated **from a clean build that has all migration
changes applied** — not from a partial state.

> **Critical caveat discovered 2026-05-28 mid-session.** The test is
> deliberately tolerant of environments without a fully-wired MLIR
> backend: when the local `mindc build` produces a stub (~1245 bytes)
> rather than a real ELF (~78 KB), the assertion logs a `WARNING:
> artifact type mismatch — built is stub but oracle is ELF` and
> **still reports PASS**. This is by design — the test is meant to
> run in environments without LLVM/MLIR provisioned and gives
> byte-identity verification only when both sides are ELF. The
> implication for the re-bless: **DO NOT capture the oracle from a
> stub-producing environment.** Doing so would silently bake in a
> 1,245-byte stub as the new oracle, breaking the byte-identity
> claim everywhere the full MLIR path runs (notably CI).
>
> Pre-rebless check: run the test with `-- --nocapture` and verify
> the log line reads `phase_g_03 KEYSTONE: byte-identical (~78 KB,
> SHA256 prefix …)` — NOT `stub path or oracle from different
> toolchain run`. If the stub-path message appears, **stop**: the
> re-bless session needs LLVM/MLIR installed and the `mlir-build`
> feature actually exercising the lowering chain (see
> `docs/install.md` for the prereqs).

The mature procedure:

1. Land the .mind migrations across all four files in a single commit
   (no staged half-migration). Verify the std-surface test suite is green.
2. Run the bootstrap test once to confirm it fails on the keystone (this
   confirms the .mind changes did flow through to libmindc_mind.so output).
3. Locate the keystone oracle (typically a `STD_SURFACE_KEYSTONE_HASH` /
   `KEYSTONE_BYTES` const in the test file or a sibling fixture). Capture
   the new hash from a clean build — **do not let the test "auto-update"
   its own oracle, that masks regressions silently.**
4. Update the oracle in a SEPARATE commit titled
   `chore(keystone): re-bless std-surface fixed-point after #306 migration`
   so the rebless is reviewable on its own.
5. Verify 7/7 phase_g_keystone_bootstrap passes against the new oracle.
6. Verify the full std-surface + mlir-lowering test suite is green.
7. Verify default-feature workspace stays green.
8. Now `v0.7.1` is one `git tag` away (see `CHANGELOG.md` `[Unreleased]`
   for the staged release notes).

## Why not do it in this session

The re-bless is risky to execute deep in a long context window — getting
the oracle wrong silently bakes in a stale or buggy byte-identity claim,
which then hides future cross-substrate regressions. The path-B
intrinsics work + this spec doc are written from a context where the
investigation is fresh; the actual migration + re-bless deserves a
fresh, focused session that does only this one thing and verifies it
end-to-end. The session start can simply:

> Open `docs/byte-store-migration.md`, execute steps 1–8.

That's the deliverable for the session that closes `#306` and unblocks
`v0.7.1`.

## Cross-references

- Intrinsic add: `0e7dd6c feat(intrinsics): add __mind_load_i8 / __mind_store_i8`
- Path-B commit message documents the rationale + non-rebless guarantee.
- `CHANGELOG.md` `[Unreleased]` ▸ "Known issues (release gate)" — the
  release-note slot is already staged and references this migration as
  the gate.
- The 7-byte runtime-support pad (commit `cc5a513`) is the temporary
  mitigation that lets the bug stay latent. The pad can be removed
  *after* the migration lands and bootstrap stays byte-identical for one
  full release cycle.
