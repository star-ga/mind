# Native ELF source-note tagging

The static native emitter writes one 52-byte MIND provenance record in a
`PT_NOTE` segment. String-literal data can follow that record at a page-aligned
file offset. The record must therefore be located through the program header,
never by subtracting a constant from the file length.

The owner is `MIND` followed by a NUL and three padding bytes. The descriptor
contains 32 bytes. Type 1 identifies the canonical IR trace-hash route. Type 2
identifies the exact assembled source SHA-256 used when HOME lowering is needed.
These are different claims. Source-note acceptance does not prove that source
and executable semantics agree, authenticate a signer, or provide a signature.

`hm_source_note_offset` checks the bounded ELF64 little-endian x86-64 executable
header, program-header entry size and complete table bounds before walking it.
It requires exactly one `PT_NOTE`, an in-range 52-byte file/memory record with
alignment 4, the expected owner and descriptor sizes, and type 1 or 2. It
refuses absent, duplicate, truncated or unsupported records. Table lengths are
checked with division before multiplication; record offsets are checked before
reading. This is a deliberately narrow validator of this emitter's layout,
not a general ELF validator. Other valid ELF note arrangements are unsupported.

`hm_mark_source_note` performs the write only after those checks. It changes
the type to 2 while preserving the digest and every other byte; tagging a type-2
record again is idempotent. A refusal returns an empty `EmitState` with negative
status, without modifying the supplied buffer. The buffer remains an owned,
live compiler allocation; these checks cannot validate an arbitrary forged
pointer or establish the meaning of a caller-supplied digest. The source-hash
computation and the choice of route remain the caller's responsibility.

The prior implementation wrote to `file_length - 44`. With trailing literal
data, a valid build could change a literal character to byte 2 and leave the
actual source-hash note incorrectly marked as an IR hash. The regression gate
checks the actual emitted literal bytes and exact assembled-source digest,
alongside malformed-record refusals and unchanged input bytes on rejection.

Run the CI-wired native gate:

```sh
python3 examples/mindc_mind/self_host_source_note_smoke.py
```

It derives a current compiler from the frozen bounded seed without publishing
or changing that seed. An explicit diagnostic compiler can instead be supplied
with `--compiler /path/to/compiler.elf`; `--evidence-dir /new/path` retains the
input hashes, commands, outputs and exit statuses. Children have a 2 GiB address
space limit. Compiler qualification, a cgroup memory limit and independent
source review remain separate obligations.
