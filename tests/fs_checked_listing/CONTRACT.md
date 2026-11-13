# Checked directory listing development contract

Author: STARGA Inc.

Status: MLIR CPU candidate tests passed; direct-native qualification is blocked.
See [EVIDENCE.md](EVIDENCE.md) for the exact compiler, results and limitations.
This additive module leaves `std.fs.read_dir` unchanged. It needs the Linux x86-64 raw syscall
surface present in compiler source `93238e8ef2ae014a79a51b9f23ef0d90e063b2a9`.
The previously accepted consumer compiler does not expose that surface. No
compiler pin, native qualification, seed or platform-support claim changes.

## API

`fs_read_dir_checked(path_addr, path_len, max_entries, max_name_bytes, out_slot)`
returns zero only after a complete enumeration reaches EOF and the owned
descriptor closes successfully. On success it writes one owned handle to the
aligned eight-byte output slot. On **every failure the output slot is unchanged**.
The caller must not overwrite an existing owned handle without freeing it.

The result retains the `std.fs` layout: a 24-byte vector `[entries, count, cap]`,
whose entries are 24-byte `[name_addr, name_len, flags]` records. Every name owns
an exact byte copy followed by NUL. Flags remain 1 for a directory, 2 for a
regular file, and 0 for other/unknown types. Unknown types are retained, not
treated as absent. Exact `.` and `..` are excluded. Names and enumeration order
are preserved; there is no UTF-8 normalization, decoding, sorting or filtering
by a table's expected names. Consumers sort when deterministic record bytes
require it. `fs_dir_list_free` releases a returned listing exactly once; null is
allowed, forged or already freed handles are outside its contract.

Input path storage must contain exactly `path_len` readable non-NUL bytes;
the function makes its own terminated copy. Lengths 1 through 4095 are supported.
The output slot must be valid writable aligned storage, disjoint from those
input bytes. Pointer arithmetic is checked but memory accessibility cannot be
proved by this raw-pointer API. Caller storage and the filesystem must not be
concurrently modified in ways that violate these preconditions.

Both nonnegative limits are caller-selected. `max_entries` bounds retained
entries. `max_name_bytes` bounds the sum of copied name lengths **including one
terminator per entry**. Zero limits allow a genuinely empty directory. Arithmetic
validation bounds entry storage, simultaneous vector growth, name storage and
fixed work storage within signed i64; it does not promise that those allocations
fit available memory. Allocation failure returns `-12` and releases all ownership.

The following failure classes are explicit:

| Status | Meaning |
|---|---|
| negative raw errno | open/read/close failed, including `-38` on unsupported hosts |
| `-22` | invalid scalar arguments, path NUL or overlapping output/path storage |
| `-36` | path exceeds the declared 4095-byte bound |
| `-12` | allocation failed |
| `-75` | a caller limit or checked size arithmetic would overflow |
| `-5` | malformed/truncated raw directory record |

Read and open errors are never EOF. After a successful open, close is attempted
exactly once on every path, including parse, capacity and allocation failure.
Close is not retried on EINTR; an earlier error takes precedence over a later
close error. Publication occurs only when both enumeration and close succeed.

## Raw ABI and parsing

The primitive calls are `__mind_sys_openat(-100, path, 589824, 0)`,
`__mind_sys_getdents64(fd, block, 4096)` and `__mind_sys_close(fd)`.
Flags are O_RDONLY, O_DIRECTORY and O_CLOEXEC. As with legacy `read_dir`, a
directory symlink is followed; this API does not establish path confinement.
The raw primitives return negative errno directly rather than consulting libc
thread-local errno. They have CPU runtime and MIND native emitter implementations
in the cited candidate source; both require actual execution qualification.

For each positive read, the byte count must be at most 4096. Before accessing
any fields there must be at least 20 remaining bytes. Each little-endian u16
`d_reclen` must be at least 20, a multiple of eight and at most the remaining
byte count. The type is at byte 18 and the name begins at byte 19. The first NUL
must occur inside this record, after at least one non-slash name byte. The code
never scans into the next record or stale tail of the block. `d_off` is opaque
and has no arithmetic role. Borrowed kernel-buffer names are copied before the
next read can overwrite them.

## Ownership and nonclaims

Path, work buffer, vector, vector backing, entry and name allocations have one
owner. Failed growth leaves the old backing owned; successful growth frees it.
Failed insertion frees the uninserted entry/name. Failure after any number of
complete records frees the whole temporary listing, which is never published.

Successful enumeration is not an atomic filesystem snapshot, a no-symlink
custody proof, a no-concurrent-writer guarantee or a durability barrier. The
caller must supply the appropriate stable-directory/exclusive-writer contract.
In particular, parser compaction must propagate this API's failure before any
MANIFEST replacement or journal deletion. It must not convert missing paths or
partial listings to successful empty tables. Those consumer changes belong to
the parser owner and are not implemented by this module.

## Required qualification

MIND executable tests must independently exercise real empty/nonempty listings,
missing/not-directory open failures, repeat listing under NOFILE 1024, exact raw
name bytes/order, capacity boundaries, partial enumeration then read failure,
close failure and all allocation sites. Synthetic syscall/allocator fault seams
must be explicit test-only source substitutions. Malformed blocks cover truncated
headers, short/zero/oversized/unaligned records, missing NUL, slash/empty names,
oversized reads and trailing fragments. No crash, timeout or build failure counts
as fault detection. Actual mutants must detect suppressed read/close/limit/record
guards and early publication as normal wrong outcomes. Consumer tests must show
failed enumeration cannot modify MANIFEST/journal. The module's MLIR tests have
run as recorded in EVIDENCE.md; they do not establish the consumer obligations
or direct-native acceptance.
