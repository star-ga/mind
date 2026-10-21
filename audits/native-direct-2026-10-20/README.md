# Native backend, direct route: findings (2026-10-20)

Measured on the pure-MIND compiler at star-ga/mind `main` (examples/mindc_mind/main.mind,
built as `target/release/libmind.so`), driven directly with the 21-module std seed
through `selftest_native_elf_u` (the route the Rust bridge `mindc build --backend native`
fences off). MLIR builds of the same programs are the oracle. Tooling and every probe
program live outside the repository; the four repro programs are in this directory
and are excluded from the parity count.

## What lowers natively today

| Scope | Result |
|-------|--------|
| `src/core/*` (8 modules, 73 functions) | lower on the real route, exit-code parity with MLIR, also as one combined program |
| everything else probed (18 modules) | refused on the real route |
| functions across 26 probed modules | 159 of 324 lower on the real route; 280 with a placeholder note (the lowerer alone); 286 after the two seed fixes below |

The native parity floor of this repository is therefore 8, and `scripts/native_parity.sh`
(through the Rust bridge) agrees: 8 of 8.

## Two gates, in order

1. **Note gate**: the mic@3 re-encoding that computes the PT_NOTE trace hash (`nb_trace_hash`)
   runs first and refuses: any string literal in a kept function, `&&` and `||`, `return`
   inside `while`, `break`/`continue` in a function with parameters, an if/else statement
   that writes a variable outside a loop, field and array stores, enum payload constructors.
2. **Lowering gate**: refuses escape sequences in literals, hex and byte literals, `~`,
   a const read while declared without `;`, duplicate const names, whole-struct
   reassignment of a `mut`, a branch that writes then exits, more than 16 struct types
   in the combined source, `loop {}`, closures, traits, methods, generics, tuples,
   `Option`/`Result`, libc externs, `__mind_now_ns`, `__mind_vec_zeroed`.

Porting the note-gate constructs alone moves `common/str`, `common/argv`, `common/log`,
`policy/policy`, `pipeline/syntax`, `common/rowtime`, `common/apiresp`, `policy/lane`
and `policy/pin` (the last four as reachable subsets) to native with proven parity.

## Silent wrong code (no refusal)

- **Loop-carried plain copy** (`repros/loop_copy.mind`): `b = a;` inside `while` emits
  wrong code; `b = a + 0;` is correct. This is the root cause of `std/sha256.mind`
  producing `9b33e8c3...` for `"abc"` instead of `ba7816bf...` on the native route;
  the frozen `stage1.elf` behind the bridge has the same bug. With `+ 0` on the
  carried copies in `compress_block` (scratch copy of the seed) the digest is correct.
  `evidence/hmac` and `evidence/chain` are wrong on the native route for this reason.
- **`extern "C" fn` before `main`** (`repros/extern_swallow.mind`): the declaration
  swallows the next function; `main` returning 1 exits 0. The bridge hides this by
  refusing the file as out of profile.

## Std seed bugs (a reseed event when fixed)

- `std/json.mind:66` and `std/toml.mind:50` both declare `const MAX_DEPTH` (duplicate
  across the seed) and both without a trailing `;` (`repros/const_no_semicolon.mind`,
  `repros/duplicate_const.mind`). Every JSON and TOML parse is refused natively until
  both are fixed; with a scratch fix, `evidence/canon`, `pipeline/config`,
  `pipeline/dag`, `pipeline/schema` and `common/tomlkv` reach full parity.
- Usable seed modules on the real route today: `arena`, `io`, `vec`; partial: `string`,
  `map`, `regex`, `cli`, `io_canon`, `ring`; refused: `json`, `toml`, most of `blas`;
  not lowerable: `time`; wrong: `sha256`.

## The bridge fence

In 21 of 94 corpus programs the Rust bridge refuses (`out-of-profile construct` or
`__mind_alloc has no definition in the closure`) while the direct route builds a
correct binary; no program went the other way. The fence protects against the two
silent miscompiles above but also hides working capability.

## Suggested order of fixes

1. The loop-carried copy and the `extern "C" fn` swallow (silent wrong code).
2. The two std-seed const defects (a reseed event).
3. The note-gate constructs, or computing the note over the lowered IR.
4. Escapes in literals, hex literals, whole-struct reassignment, branch-exit
   carries, the 16-struct table limit.

Reported from the MIND-CI native-parity work (star-ga/mind-ci, docs/NATIVE_DIRECT.md).
