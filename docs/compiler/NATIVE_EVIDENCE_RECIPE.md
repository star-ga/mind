# Evidence and opt-in signing for native ymm programs

This page is the recipe for attaching a mic@3 evidence artifact, and an opt-in
post-quantum signature, to a program that calls the native-only AVX2 rows
(`__mind_vpaddd_i32x8` and its family). It states what the artifact attests and
what it does not. `tests/evidence_release_smoke.rs` runs every step below.

## What the artifact attests

- **The canonical IR body.** `trace_hash = SHA-256(canonical mic@3 bytes)` covers
  the lowered body, including each `Call` to an AVX2 row with its interned name and
  argument ids. Changing one body byte fails `mindc verify` (exit 1).
- **The semantic choice, by name.** The body records that the program calls
  `__mind_vpaddd_i32x8`. Which VEX bytes encode that call is NOT in mic@3.
- **The determinism class.** The label is re-derived at verify time from the hashed
  body. The Layer A rows are `Det::Pure`, so a program that uses only them is
  `deterministic` and passes `mindc verify --require-deterministic`.
  `__mind_cpu_features_v1` is `Det::World`: a program that reaches it is attested
  `nondeterministic` and `--require-deterministic` refuses it (exit 1). A program
  that dispatches on the CPU feature word is therefore attested nondeterministic,
  even when every path produces the same output.
- **Signer-asserted attributes.** `--evidence-attr KEY=VALUE` entries are inside
  the signature preimage, so on a signed artifact they are authenticated: change one
  byte and verification fails. `mindc verify` does NOT re-derive them. They are the
  signer's statement, not a measurement.

## What the artifact does not attest

- The native ELF bytes and the VEX encodings. They are a pure function of
  (source, compiler ELF). The recipe binds them only through the signer-asserted
  `starga.native.elf_sha256` and `starga.native.compiler_sha256` attributes.
- The ISA of the host that will run the ELF. `starga.native.isa` is a declaration.
  The run-time ISA guard in the ELF refuses a host without AVX2 (`isa.avx2_missing`,
  exit 132); the evidence does not.
- Which `libcuda.so.1` the dynamic loader resolves at run time, for programs that
  import the driver rows.
- Who produced the artifact, unless it is signed and the verifier pins the
  signer's public keys.

## Recipe

All seeds come from the environment and are never echoed. The values below are the
public TEST vectors the test suite uses; never use them for a release.

1. **Toolchain.** Build `mindc` with both signing features. They are OFF by
   default, and a seed given to a build without them is refused with no artifact:

       cargo build --release --bin mindc \
           --features "std-surface cross-module-imports evidence-mldsa evidence-slhdsa"

2. **Native executable.** Compile the program with the compiler ELF you will name
   in the evidence. This step writes no evidence and no signature:

       MINDC_STD_DIR=$REPO/std MINDC_NATIVE_ELF=<compiler.elf> \
           mindc build prog.mind --target cpu --emit binary --out prog.elf \
           --backend native --native-profile bounded
       sha256sum prog.elf <compiler.elf>

3. **Signed evidence over the same source.** Both hybrid seeds are required. Unset
   the other two key variables first:

       env -u MIND_EVIDENCE_MLDSA_KEY -u MIND_EVIDENCE_ED25519_KEY \
           MIND_EVIDENCE_MLDSA87_KEY=<64 hex> MIND_EVIDENCE_SLHDSA_KEY=<192 hex> \
           mindc prog.mind --target cpu --emit-evidence prog.mic3 \
           --evidence-attr starga.native.elf_sha256=<sha256 of prog.elf> \
           --evidence-attr starga.native.compiler_sha256=<sha256 of compiler.elf> \
           --evidence-attr starga.native.isa=x86-64-v3

   Expected stderr: `Wrote mic@3 evidence artifact: prog.mic3 (N bytes, self-check
   ok, pqc-hybrid-ml-dsa-87-slh-dsa-256s-signed)`.

4. **Verify.** Pin both public keys (or set `MIND_EVIDENCE_VERIFY_PUBKEYS`):

       mindc verify prog.mic3 --require-deterministic --require-signed \
           --signer-pubkey <ML-DSA-87 key hex> --signer-pubkey <SLH-DSA key hex>

   Exit 0 means: the body is untampered, it is SSA well-formed, it is deterministic,
   and both legs verify under the pinned keys.

## Rules the recipe depends on

- The flagship scheme is `pqc-hybrid-ml-dsa-87-slh-dsa-256s`: ML-DSA-87 (FIPS 204)
  AND SLH-DSA-SHAKE-256s (FIPS 205). Both legs must verify. Stripping one leg fails
  closed.
- One hybrid seed alone exits 1 and writes no artifact.
- A legacy Ed25519 seed is refused (`signing.retired_scheme`), also when the hybrid
  seeds are present.
- `MIND_EVIDENCE_MLDSA_KEY` alone selects the transition scheme `ml-dsa-65`, which
  is not the release profile. Unset it.
- Both legs sign without randomness, so a signed artifact is byte-identical across
  runs, and signing never changes `trace_hash`. Signing adds only `signature.*`
  entries: the unsigned artifact is byte-identical with or without the signing
  features compiled in.
- No `mic@N` bump: the AVX2 rows add callee names, not opcodes. An older verifier
  that does not know a name refuses the artifact. That is fail-closed.

## Status

Signing is **signed (opt-in)**: the capability ships and is tested, off by default.
Still owed before any claim that releases are signed: offline key generation, a
published public key, and a CI gate that reproduces and verifies a signed artifact.
