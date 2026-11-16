# Installing mindc

`mindc` is the MIND compiler driver. Pre-built binaries are published for
Linux x86_64, Linux ARM64, macOS Intel, macOS Apple Silicon, and Windows
x86_64. A pre-built binary needs no Rust toolchain.

## One-line install

### Linux and macOS

```sh
curl -sSL https://mindlang.dev/install.sh | sh
```

`https://mindlang.dev/install.sh` serves a copy of `scripts/install.sh` in
this repository.

The script detects the OS and the architecture, downloads the matching GitHub
release asset, and installs `mindc` to `/usr/local/bin` when that directory is
writable. Otherwise it installs to `~/.local/bin`. `MIND_INSTALL_DIR` overrides
the directory.

Pin a tag with `MIND_VERSION`. The script reads no positional arguments.

```sh
MIND_VERSION=v0.10.3 curl -sSL https://mindlang.dev/install.sh | sh
```

Preview the download and the install path. The script writes nothing.

```sh
MIND_DRY_RUN=1 curl -sSL https://mindlang.dev/install.sh | sh
```

When the release publishes `<asset>.sha256`, the script checks that file. If
that file is absent, the script installs the asset without the check. The
asset names are in the manual table below.

### Windows (PowerShell)

`scripts/install.ps1` reads `-Version`. A script fetched with `irm` and run
with `iex` does not receive arguments, so the pin form is a script block:

```powershell
& ([scriptblock]::Create((irm https://mindlang.dev/install.ps1))) -Version v0.10.3
```

From a clone:

```powershell
powershell -File scripts/install.ps1 -Version v0.10.3
```

The default install directory is `%USERPROFILE%\.local\bin`.

The published Windows asset is `mindc-windows-x64.zip`, with
`mindc-windows-x64.zip.sha256` beside it. `scripts/install.ps1` in this tree
still requests `mindc-<tag>-x86_64-pc-windows-msvc.exe` and a `SHA256SUMS`
file. Those names are not release assets. Use the manual steps for the
published file. Windows ARM64 has no pre-built binary.

## What a build needs

`mindc check`, `--emit-ir`, and `--emit-mlir` need nothing beyond the binary.

On x86-64 Linux, `mindc build` uses the native backend by default. That
backend needs no LLVM, no MLIR, and no clang. It emits an x86-64 Linux
executable for the native-lane-v1 subset. A program outside that subset is
refused.

`mindc run`, `mindc bench`, `--emit cdylib`, and `--emit object` need the MLIR
backend (`--backend mlir`, or `backend = "mlir"` under `[build]` in
Mind.toml). On any host other than x86-64 Linux, `mindc build` needs that same
selection. The MLIR route needs these tools on `PATH`:

- `mlir-opt` and `mlir-translate` (LLVM/MLIR)
- `clang`

If they are absent, codegen fails with a `tool not found` error. A build names
the missing tool and any installed copy under `/usr/lib/llvm-<N>/bin`. Tested
with LLVM/MLIR 23 (latest) and 20 (oldest supported). MLIR 18 builds most
programs but cannot bufferize a tensor carried across a loop or returned from
several exits. On Debian/Ubuntu, LLVM/MLIR 20: `apt install mlir-20-tools
clang-20`, and put `/usr/lib/llvm-20/bin` on `PATH`. LLVM/MLIR 23: add the
signed `llvm-toolchain-noble-23` apt source, `apt install mlir-23-tools
clang-23`, and put `/usr/lib/llvm-23/bin` on `PATH`. On macOS:
`brew install llvm`, and add its `bin` to `PATH`.

## Manual install

1. Open the [Releases page](https://github.com/star-ga/mind/releases).
2. Download one asset. Each asset has a sibling `<asset>.sha256` file.

   | Platform | Asset |
   |----------|-------|
   | Linux x86_64 | `mindc-linux-x64.tar.gz` |
   | Linux ARM64 | `mindc-linux-arm64.tar.gz` |
   | macOS Intel | `mindc-macos-x64.tar.gz` |
   | macOS Apple Silicon | `mindc-macos-arm64.tar.gz` |
   | Windows x86_64 | `mindc-windows-x64.zip` |

3. Check the sibling `.sha256` file (see below).
4. Extract the archive. The Unix archive contains `mindc`. The Windows archive
   contains `mindc.exe`. Put that file on `PATH`.

```sh
tar -xzf mindc-linux-x64.tar.gz
sudo mv mindc /usr/local/bin/mindc
mindc --version
```

```powershell
Expand-Archive .\mindc-windows-x64.zip -DestinationPath .
Move-Item .\mindc.exe $env:USERPROFILE\.local\bin\mindc.exe
mindc --version
```

## Verifying the download (SHA256)

Each asset has its own `<asset>.sha256` file. There is no combined
`SHA256SUMS` file. The Linux and macOS files are the output of `sha256sum` or
`shasum -a 256`. Run the check in the directory that holds the asset.

```sh
sha256sum -c mindc-linux-x64.tar.gz.sha256
shasum -a 256 -c mindc-macos-arm64.tar.gz.sha256
```

```powershell
$expected = (Get-Content .\mindc-windows-x64.zip.sha256).Split()[0].ToLower()
$actual = (Get-FileHash .\mindc-windows-x64.zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw "SHA-256 mismatch" }
```

## Build from source

Rust 1.85 or later (`rust-version` in `Cargo.toml`). This is the release
feature set:

```sh
cargo install --git https://github.com/star-ga/mind --tag v0.10.3 --locked --bin mindc --features "mlir-lowering mlir-build cross-module-imports autodiff evidence-mldsa evidence-slhdsa"
```

From a local clone of tag `v0.10.3`:

```sh
cargo build --release --locked --bin mindc --features "mlir-lowering mlir-build cross-module-imports autodiff evidence-mldsa evidence-slhdsa"
```

The binary is `target/release/mindc`.

## Verify your install

```sh
mindc --version
mindc --help
```

## Supported platforms

| Platform | Architecture | Binary |
|----------|-------------|--------|
| Linux | x86_64 | glibc 2.28 or newer. Not a static musl binary. Needs glibc and libgcc_s. |
| Linux | ARM64 | Pre-built. Same glibc 2.28 floor. |
| macOS | x86_64 | Separate archive. Not a universal binary. |
| macOS | arm64 | Separate archive. Not a universal binary. |
| Windows | x86_64 | `mindc-windows-x64.zip` |

The Linux binaries are built in a digest-pinned `manylinux_2_28` image. The
v0.10.2 Linux asset needed glibc 2.39. On x86-64 Linux the default native
`mindc build` needs no LLVM.
