# Design note (0.10.4 candidate, not implemented): record the composed source image in the evidence

Status: design only. Nothing on this page is in the 0.10.3 compiler.

## Problem

`mindc build --backend native` does not send the entry file to the native
compiler as written. It composes a stdin image: a 16-byte header (v1) or a
48-byte header (v2 layout), the std blob (the seed modules in manifest order),
then the user region. The user region can differ from the files on disk: the
allocation guard respells each `__mind_alloc` / `__mind_realloc` with a size
that is not proven >= 1 and appends two guard functions; the open-host pass
rewrites `__mind_open` calls. Two routes that start from the same unit can
therefore compile two different programs. A reader must be able to tell which
program the compiler received.

## What 0.10.3 already records (measured)

- A type-2 note (`n_type` 2) is SHA-256 of the composed image without its
  header: std blob plus user region. So a type-2 ELF already records the
  digest of the composed program text. The header (user_lo, src_len, and the
  v2 layout values) is not in the digest.
- A type-1 note (`n_type` 1) is the mic@3 trace hash of the seeded and pruned
  IR. It records no image digest.
- `mindc` never writes the composed image, so no reader can recompute either
  digest without capturing the compiler's stdin.

## Requirements

1. Every native ELF build gives the SHA-256 of the full composed stdin image
   (header included: the header binds the layout and the user region bounds).
2. The IR trace hash stays where it is (type-1 note). No digest replaces
   another.
3. A reader recomputes the image digest from the unit with the release
   `mindc`, without running the native compiler seed and without a script.
4. An unsigned build stays byte-identical to today unless the change is a
   declared reseed event.
5. No claim of signing. The image digest is tamper evidence, not authorship.

## Options

A. Second note in the ELF (seed-bound). The native compiler writes a second
   `MIND` note, `n_type` 3, description = SHA-256 of the full stdin image.
   - In the artifact itself, which is the wedge form.
   - This is a change to the note writer in the pure-MIND compiler: seed
     source. Every native ELF digest moves, so it is a reseed event.
   - Every note reader must change in the same step: `hm_checked_note_record`
     (refuses a `PT_NOTE` whose size is not 52 and any `n_type` other than 1
     or 2), the image-case table row `note_type_three` (expects a refusal),
     and the gates that read the note at a fixed file-tail position
     (`elf[-40:-36]`, `got[-32:]`).

B. Native evidence record from the bridge (not seed-bound). `mindc build
   --backend native --emit-evidence <path>` writes a small evidence MAP next
   to the ELF, with the same MAP encoding as the mic@3 epilogue
   (lexicographically sorted keys, inline strings, `schema` = 1):
   `native.image_sha256` (full stdin image), `native.image_bytes`,
   `native.elf_sha256`, `native.note_kind` (1 or 2), `native.note_hash`,
   `native.seed_sha256` (the compiler ELF that ran), `native.toolchain`.
   Before it writes, the bridge reads the note back from the ELF it just got
   and refuses on a mismatch, as the mic@3 emit path already round-trips its
   artifact.
   - No change to the seed or to ELF bytes: the keystone and every pinned ELF
     digest stay the same.
   - The record is outside the ELF. The ELF note still binds the ELF; the
     record binds the ELF sha256 to the image sha256.
   - This is a Rust bridge change. It needs an explicit order for an edit in
     `src/build/native_bridge.rs`.

C. `mindc build --backend native --emit=image --out <file>`: write the
   composed image instead of compiling it. A reader then runs `sha256sum` on
   it. It is the smallest change and the most direct proof, and it can sit
   under B (same composition code). Same Rust edit question as B.

## Recommendation

0.10.4: B with C (bridge only, digest-neutral). Verify step for a release ELF:

```bash
mindc build U --backend native --release --no-cache --out R --emit-evidence R.ev
cmp E R && <compare R.ev with the published record>
```

Next reseed window: A, after a schema decision for `n_type` 3 and an update
of every note reader in the same change. Do not fold the image digest into
the type-1 trace hash: the mic@3 bytes do not contain the header or the
respelled source text, and a fold would change the meaning of an existing
field.

## Gates for the implementation

- Positive: for a unit whose user region the guard respells, the record's
  `native.image_sha256` equals `sha256sum` of the `--emit=image` output, and a
  type-2 note equals SHA-256 of that image without its header.
- Negative (mutation): change one byte of the user region after composition;
  the record check must fail. Remove the read-back; a test must go red.
- Byte identity: every pinned native ELF digest and the keystone unchanged
  (option B and C).
