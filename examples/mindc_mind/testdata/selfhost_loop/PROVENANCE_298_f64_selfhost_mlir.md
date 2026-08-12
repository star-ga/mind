# #298 self-host f64 MLIR emit — self-host loop seed refreeze provenance

This artifact is the durable, independently-verifiable evidence for the RI-E1
self-host loop **seed refreeze** that accompanies #298 (scalar-`f64` MLIR emission
in the pure-MIND `selftest_emit_mlir`). Adding f64 emit grew `examples/mindc_mind/
main.mind`, so the emitted whole-compiler ELF grew, so the frozen bootstrap seed
`testdata/selfhost_loop/stage1.elf` had to be re-frozen (Rule 1b: expected
source-growth drift, NOT a determinism failure).

The refreeze was performed **only after** the PRE-refreeze fixed point was
established independently. Both measurements are recorded below verbatim.

## Provenance (this evidence run)
- **Source commit (main.mind + all #298 changes):** `7b696f098f2290fa0915db06d5a7058753d6a5f8` (the RI/RH boundary), plus the #298 working-tree changes committed together with this file.
- **mindc binary (per-run provenance, NOT an acceptance property):**
  - Phase-A authoritative gate: `sha256=280c0e597630741fab739be920a38640e29eb6e9c9f324370770a10c3a94c091` (size 7480152)
  - This evidence-capture run: `sha256=1f12140ebf1e5e90ab4e254d66e1bb139a632ca03ff98ab6676cd8da92e44d47`
  - The mindc bootstrap binary is **not** bit-reproducible across build environments; the two SHAs above are DIFFERENT binaries built from the SAME source `7b696f09`. Both produce the **identical** self-host output `11df237d…`. That is the acceptance property: **canonical-output determinism / a stable self-host fixed point**, not executable-byte identity. (Determinism is asserted for the tested conditions — x86-64 AVX2, this toolchain — not claimed universally across arbitrary machines/toolchains/substrates.)
- **Host / target:** hurryhost · AlmaLinux release 9.8 (Olive Jaguar) · x86_64-unknown-linux-gnu
- **Toolchain:** cargo 1.93.0 (083ac5135 2025-12-15) · LLVM/MLIR 20.1.2 (`/opt/llvm20`) · clang 20.1.2
- **Date (UTC):** 2026-08-12
- **RH note:** the RH downstream consumer stays pinned at `c6cb999` and is untouched by this change; it re-crosses its own consumer gate separately, if/when it consumes a newer compiler.

## Exact commands
```bash
export PATH=/opt/llvm20/bin:$HOME/.cargo/bin:$PATH
cd <repo>                       # a clean checkout with the #298 changes
./target/release/mindc build --release --emit=cdylib --out=/tmp/libmindc_ev.so   # #298 self-host .so
# PRE (old seed still in tree): expect PRIMARY fixed point but ORACLE-vs-frozen mismatch (Rule 1b)
MINDC_SO=/tmp/libmindc_ev.so python3.11 examples/mindc_mind/self_host_loop_smoke.py
# refreeze (separate provenance operation): promote the verified candidate to the seed
MINDC_SO=/tmp/libmindc_ev.so python3.11 examples/mindc_mind/self_host_loop_smoke.py --reseed
# POST (new seed): expect full fixed point + no drift
MINDC_SO=/tmp/libmindc_ev.so python3.11 examples/mindc_mind/self_host_loop_smoke.py
```
(`python3.11` — the smoke helper uses PEP-604 `X | None`; the AlmaLinux 9 default `python3` is 3.9.)

## PRE_REFREEZE  (old seed in tree; candidate fixed point established independently)
```
S_old (frozen seed) = 042bfea3a4ff02d78796e1d5993bcf1cdc16891d0edc3ce81ff9335ef3ff453f   size 2236900 B
B1  (stage1)        = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643   size 2244386 B
B2  (stage2)        = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643   size 2244386 B
B3  (stage3)        = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643   size 2244386 B
=> B1 = B2 = B3                        (candidate is a genuine self-host fixed point)
=> B1 != S_old                         (differs from the old seed)
classification      = EXPECTED_SOURCE_GROWTH / Rule 1b
```
Raw `self_host_loop_smoke.py` output (verbatim):
```
[self-host loop] combined=2781192B user_lo=380832 seed=2781208B  so=libmindc_ev.so  reseed=False
  seed = frozen pure-MIND stage0 ELF: 2236900B sha256=042bfea3a4ff02d78796e1d5993bcf1cdc16891d0edc3ce81ff9335ef3ff453f
  stage1 (frozen stage0 run natively): 2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
  stage2 (stage1 run natively):        2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
  stage3 (stage2 run natively):        2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
  FAIL  stage1 (11df237d...) != frozen seed (042bfea3...) — running the frozen pure-MIND stage0 did NOT reproduce it; the bootstrap is not fixed.
```
The `FAIL` line IS the PRE relationship (`B1 != S_old`), expected because the seed
predates the #298 main.mind growth. Independently reproduced across two distinct
mindc binaries (`280c0e59…` and `1f12140e…`), both from `7b696f09` — same output
`11df237d…`.

## Seed refreeze (separate provenance operation)
```
promoted artifact = examples/mindc_mind/testdata/selfhost_loop/stage1.elf  (+ MANIFEST.txt)
source            = the verified PRE-refreeze candidate B1
S_new             = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643   size 2244386 B
```
```
  RESEEDED  frozen bootstrap stage1.elf re-blessed: 2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
```

## POST_REFREEZE  (new seed; full fixed point re-established by actually running the generations)
```
S_new (frozen seed) = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643   size 2244386 B
POST_B1 (stage1)    = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
POST_B2 (stage2)    = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
POST_B3 (stage3)    = 11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
=> S_new = POST_B1 = POST_B2 = POST_B3     (fixed point holds on the new seed; no drift)
```
Raw `self_host_loop_smoke.py` output (verbatim):
```
[self-host loop] combined=2781192B user_lo=380832 seed=2781208B  so=libmindc_ev.so  reseed=False
  seed = frozen pure-MIND stage0 ELF: 2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
  stage1 (frozen stage0 run natively): 2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
  stage2 (stage1 run natively):        2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
  stage3 (stage2 run natively):        2244386B sha256=11df237d265288840c39f74bbc1f73af5b54760f46f9de35fc4fcf405f6ec643
  PASS  [PRIMARY] stage1 == stage2 == stage3 == frozen stage0 BYTE-IDENTICAL (2244386B, sha256=11df237d...) — MIND reproduces its compiler with ZERO Rust/LLVM in the chain (scalar subset, RI-E1).
  PASS  [ORACLE] fresh Rust .so output == frozen bootstrap (11df237d...) — no source drift.
```

## Independent reproduction recipe
1. Check out the #298 landing commit (the committed seed = `S_new = 11df237d…`).
2. `git show <parent>:examples/mindc_mind/testdata/selfhost_loop/stage1.elf | sha256sum`
   → `042bfea3…` (= `S_old`, git-verifiable from the parent commit).
3. Build the #298 self-host `.so` (command above) and run the loop with the parent's
   seed restored (`git checkout <parent> -- …/stage1.elf …/MANIFEST.txt`) → reproduces
   the PRE relationship `B1=B2=B3=11df237d ≠ S_old`.
4. Run the loop on the committed (new) seed → reproduces POST `S_new=B1=B2=B3=11df237d`.
