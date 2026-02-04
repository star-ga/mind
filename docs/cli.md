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
| `--emit-ir` | Emit canonical IR for the module |
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
cargo build --bin mindc --features full    # All features
```

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
