# Native source-input adapter

Author: STARGA Inc.

`native_source_driver.mind` provides the candidate `NSI_STDIN_V1` process
interface to `nsi_compile`. It is a MIND executable adapter. Official CLI
integration, general C extern admission and consumer qualification remain open.
The [diagnostic guide](native_source_driver_check.md) describes exact inputs,
compiler hashes, resource limits and reproduction.

## Wire format

Stdin contains exactly one little-endian frame, followed by EOF:

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 8 | Literal ASCII `MINDNSI1` |
| 8 | 8 | Header size, exactly32 |
| 16 | 8 | Flags, exactly0 |
| 24 | 8 | Nested payload length,17 through4194320 |
| 32 | 8 | Source user offset, nonnegative and below source length |
| 40 | 8 | Source length,1 through4194304 |
| 48 | source length | Exact source bytes |

Lengths and offsets are signed64-bit words. Nested length must equal source
length plus16. Unsupported headers, truncated frames and trailing bytes are
refused. The source-size bound is specific to this process profile. The adapter
adds16 private zero sentinel bytes; they are outside the source digest.

No addresses, imported symbols, route, image, digest or result flag come from
the frame. Route is0. The fixed layout policy is
`[1,268435456,268435456,67108864,4096,268435456]`: version1,256MiB arena,
256MiB overflow capacity,64MiB MIND stack,4KiB guard and256MiB external-runtime
reserve. Existing layout validation still applies; these budgets do not prove
that every input program fits.

The adapter hashes the complete source with `std.sha256`, then invokes
`nsi_compile` once. It requires a nonempty image, four zero diagnostic words
and valid ELF magic. The existing source-note parser checks the note structure;
the adapter checks all32 digest bytes before marking the note as type2
(source hash). Type1 denotes an IR trace hash and is not used for this claim.
Only a complete successful stdout write returns0. Stdout contains raw ELF
bytes, with no text prefix. A future production bridge must independently
validate the returned image, geometry and source binding before publishing it.

## Failure behavior and ownership

| Exit | Meaning | Exact stderr line |
| --- | --- | --- |
| 64 | Malformed, truncated or trailing frame | `malformed input` |
| 65 | Source, hash-input or image refusal | `source/image refused` |
| 70 | Normal allocation refusal | `allocation refused` |
| 74 | Input/output failure | `input/output failure` |

Each line ends with one newline. A normal SHA return2 is allocation refusal70;
other nonzero hash returns are source refusal65. The first control-buffer
allocation failure returns70 with empty stderr because no message storage
exists. Failure installing the dedicated process's SIGPIPE-ignore policy
returns74 with empty stderr. Error reporting is best effort if stderr itself
is unusable. An allocator trap, crash or timeout is not a normal refusal.

Reads and writes use bounded short-transfer loops. Negative or overcount
results fail; EINTR may refuse normally in this profile. The final one-byte
read must return0. SIGPIPE is ignored only in this dedicated executable so
closed stdout can be reported as an I/O failure. Owned source, control and
image storage is freed on normal return paths. Live-buffer and parser
preconditions of the called compiler components remain part of the contract.

This source note is a digest binding, not a signature, theorem certificate or
proof of semantic equivalence between source and executable. Successful
adapter tests do not advance a compiler seed or qualify a downstream pin.
