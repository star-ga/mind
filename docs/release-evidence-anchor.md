# Evidence anchor of a native release ELF

This page gives the release step that anchors a published native ELF
(`mindc build <unit> --backend native`) to its unit source and to the release
compiler. Run it for every published ELF.

The anchor is tamper-evident. It is not signed: it proves that the bytes match
what the release compiler makes from the source. It says nothing about who
made the file.

## What a native ELF carries

Each native ELF has exactly one `PT_NOTE` segment with one note: name `MIND`,
32 bytes of description. The pure-MIND compiler writes it. `n_type` tells the
reader what the 32 bytes are:

| `n_type` | Description (32 bytes) | When |
|---|---|---|
| 1 | IR trace hash: SHA-256 of the canonical mic@3 bytes of the seeded and pruned IR, computed by the pure-MIND mic@3 twin | the twin can lower every function of the program |
| 2 | Source-image hash: SHA-256 of the composed source image that the compiler received (the std blob plus the user region, without the 16-byte stdin header) | the twin rejects one or more functions; those functions are lowered on the HOME path |

The record layout and its validator are in
[`compiler/NATIVE_SOURCE_NOTE.md`](compiler/NATIVE_SOURCE_NOTE.md). A reader must
check `n_type` before reading the hash. A type-2 note is not a
mic@3 trace hash. A full application unit (306 KB of source), built with a
0.10.4 pre-release compiler, got a type-2 note.

Measured on the 0.10.3 sources:

- `add` (the `add` fixture of `tests/_ref_mic3_dump.rs`, two functions):
  `n_type` 1.
- `testdata/native_cf/l1_return_in_loop.mind`: `n_type` 2. The note equals
  SHA-256 of bytes 16 to the end of the captured stdin image (420,060 of
  420,076 bytes: std blob 418,837 bytes, then the user region).

## Other commands, and what they do not check

- `mindc verify <file>` reads a mic@3 artifact. Given a native ELF it exits 1
  with `mic@3 body did not parse for SSA check`. It does not read ELF notes.
- `mindc <unit> --emit-evidence <path>` writes a mic@3 artifact of the IR of
  the standard (MLIR) pipeline. Its `trace_hash` is a different object from
  the ELF note (other IR, other id namespace). Measured on `add`: artifact
  `trace_hash` `58eea513...`, ELF note `bc193c21...`. Use it to attest a unit's
  IR. Do not cite it as the anchor of a native ELF.

## The release step

Inputs: the published ELF `E`, its unit source `U`, and the release compiler
(the same `mindc`, the same native compiler seed, the same std directory that
made `E`). Set `MINDC_NATIVE_ELF` and `MINDC_STD_DIR` exactly as for the
published build.

```bash
mindc build U --backend native --release --no-cache --out R && cmp E R
```

| Exit status | Meaning |
|---|---|
| 0 | anchored: the release compiler makes `E` byte for byte from `U`, the note included |
| 1 | not anchored: `E` differs from the rebuild (tampered code or note, other source, other compiler or seed) |
| other | the build failed (its own diagnostic) or `cmp` could not read a file (2) |

The rebuild recomputes the note from `U`, so equal files mean equal notes. To
show the note itself (kind and hash), compare `readelf -nW E` with
`readelf -nW R`.

Rules:

1. A row is published only with the sha256 of `U` and of `E`, and the exit
   status 0 of this step on that pair.
2. Run the step two times on two clean outputs. Both must give 0.
3. Never publish the note hash without its `n_type`.

## Tamper test (required once per release compiler)

Flip one byte of a copy of `E` and run the step on the copy. It must exit 1.
Do it for one code byte (the entry instruction, file offset = entry vaddr minus
the vaddr of the executable `PT_LOAD`, plus its file offset) and for one note
byte (the first description byte, 8 bytes after the `MIND\0\0\0\0` name).

Measured with the 0.10.3 compiler:

| Program (note kind) | Clean | Code byte flipped | Note byte flipped |
|---|---|---|---|
| `add` (`n_type` 1, code offset 288, note offset 460) | 0 | 1 | 1 |
| `l1_return_in_loop` (`n_type` 2, code offset 288, note offset 3100) | 0 | 1 | 1 |

## Limits

- The note of a type-2 ELF hashes the source image, not the IR. The type-1
  note is the only native note that is a mic@3 trace hash.
- The step trusts the release compiler to compute the note. A type-2 note
  was checked against `sha256sum` of the captured stdin image (equal, see
  above). For a type-1 note the Rust reference generator
  (`cargo test --features "std-surface cross-module-imports" --test _ref_mic3_dump -- --nocapture`,
  line `REF <name>: ... note=`) prints the expected hash for its fixtures. The
  committed `examples/mindc_mind/_ref_add.note` predates the current std
  content and does not equal the 0.10.3 `add` note; regenerate it before you
  use it as a reference.
- To bind an ELF sha256 into a (signed, opt-in) mic@3 artifact, use the
  attribute recipe in
  [`compiler/NATIVE_EVIDENCE_RECIPE.md`](compiler/NATIVE_EVIDENCE_RECIPE.md).
  Those attributes are the signer's statement; this step is the measurement.
- The composed image is not written by `mindc` today. The 0.10.4 design is in
  `docs/design/evidence-composed-image.md`.
