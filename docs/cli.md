# MIND CLI Reference

## mindc - MIND Compiler

### Usage

```bash
mindc [OPTIONS] [FILE] [COMMAND]
```

### Commands

| Command | Description |
|---------|-------------|
| `build` | Build a MIND project (reads Mind.toml) |
| `run` | Build and run a MIND project |
| `conformance` | Run the Core v1 conformance suite |
| `ops` | Inspect compiler knowledge about Core profiles |
| `help` | Print help message |

### Options

| Flag | Description |
|------|-------------|
| `--version` | Print compiler version and component stability versions |
| `--stability` | Print short description of the public stability model |
| `--emit-ir` | Emit a **lossy dataflow summary** of the IR (the text view drops control flow, calls, returns, and function bodies) — **not** the canonical artifact; use `--emit-mic3` for the canonical, exhaustive `mic@3` binary |
| `--emit-grad-ir` | Emit gradient IR (requires `--autodiff`) |
| `--emit-mlir` | Emit MLIR text (requires `mlir-lowering` feature) |
| `--emit-obj <PATH>` | Emit object file to specified path (requires `aot` feature) |
| `--func <NAME>` | Focus on a specific function (for autodiff/MLIR) |
| `--autodiff` | Run autodiff for selected function |
| `--verify-only` | Only verify pipeline without emitting artifacts |
| `--target <TARGET>` | Execution target: `cpu` or `gpu` (default: `cpu`) |
| `--diagnostic-format <FORMAT>` | Output format: `human`, `short`, or `json` (default: `human`) |
| `--color <WHEN>` | ANSI color: `auto`, `always`, or `never` |
| `-h, --help` | Print help |

### Feature Flags

Build `mindc` with additional features:

```bash
cargo build --bin mindc                    # Minimal (fast build)
cargo build --bin mindc --features aot     # AOT compilation support
cargo build --bin mindc --features autodiff # Autodiff support
cargo build --bin mindc --features full    # aot + autodiff + cpu-exec + cpu-conv + pkg + ffi-c
```

`full` is **not** every feature — it enables 9 of the 25 declared features and
notably omits `std-surface` and `cross-module-imports`, so a `mindc` built from
it rejects programs that use the `std` surface or `use std.foo` imports. The
feature set CI builds and gates, and the one to use for real work, is:

```bash
cargo build --release --no-default-features \
  --features "mlir-build std-surface cross-module-imports" --bin mindc
```

### Native backend memory profile (`mindc build --backend native`)

`--native-profile <default|bounded>` selects the memory layout of the emitted
native image. It is a build flag like `--backend`, and it is only accepted with
`--backend native`.

| Profile | Arena | Guard gap | Stack | Load reservation | Seed used |
|---------|-------|-----------|-------|------------------|-----------|
| `default` | 1 GiB | 4 KiB | 4 GiB | 5,371,158,528 B page-rounded (whole compiler) | `testdata/selfhost_loop/stage1.elf` |
| `bounded` | 768 MiB | 1 MiB (unmapped) | 384 MiB | about 1.15 GiB | `testdata/selfhost_loop_bounded/stage1.elf` |

`default` images are byte-identical to the images built before profiles existed.
`bounded` images, and the bounded compiler seed itself, load and run under
`RLIMIT_AS` = 2 GiB. The arena and stack sizes come from one table in
`examples/mindc_mind/main.mind` (`nb_arena_bytes`, `nb_stack_bytes`,
`nb_guard_bytes`). The PT_LOAD headers, the stack address, the entry stack
pointer and every allocation guard all read that table, so the layout is never
patched into the headers after emission. The bridge checks that the artifact's
segments match the requested layout before it writes the file.
`MINDC_NATIVE_ELF` still overrides the seed.

How a bounded image fails:

| Condition | Outcome |
|-----------|---------|
| allocation past the arena cap | `error: native arena exhausted` on stderr and exit status 70, including when the arena is full to its last 8 bytes: the bounded allocator checks that header and payload fit before it writes anything. A default image keeps its historical sequence (header written before the check), so it runs `ud2` (SIGILL) at the same point, or takes SIGSEGV in the guard gap when the cursor is already at the arena end. Either way it stops; it never returns a block. |
| stack overflow | SIGSEGV on the unmapped guard gap. A frame larger than the 1 MiB guard could step over it; the largest frame in the compiler seed is 15,216 B. |
| malformed layout request (raw seed protocol) | `error[native-layout]: ...` on stderr, exit status 64 + reason, no image written |

`examples/mindc_mind/self_host_bounded_profile_smoke.py` gates the profile, and
its `--bless` mode re-pins the bounded seed.

### Examples

```bash
# Compile to IR
mindc program.mind --emit-ir

# Verify without output
mindc program.mind --verify-only

# Generate gradient IR
mindc program.mind --func main --autodiff --emit-grad-ir

# Compile to object file
mindc program.mind --emit-obj output.o

# Build a project
cd my-project && mindc build

# Run a project
cd my-project && mindc run

# JSON diagnostics for tooling
mindc program.mind --diagnostic-format json --verify-only
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Compilation error (parse, type check, verification) |
| 2 | I/O error (file not found, permission denied) |
