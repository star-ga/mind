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

Pin a tag with `MIND_VERSION`. The script reads no positional arguments. Set
the variable on `sh`, the process that runs the script. A variable set before
`curl` goes to `curl`, and the script installs the latest release.

```sh
curl -sSL https://mindlang.dev/install.sh | MIND_VERSION=v0.10.4 sh
```

Preview the download and the install path. The script writes nothing.

```sh
curl -sSL https://mindlang.dev/install.sh | MIND_DRY_RUN=1 sh
```

When the release publishes `<asset>.sha256`, the script checks that file. If
that file is absent, the script installs the asset without the check. The
asset names are in the manual table below.

The script unpacks the whole archive in a temporary directory, copies only
`mindc` to the install directory, and removes the temporary directory when it
exits. The other archive members (`std/`, `stdlib_manifest.txt`, `SHA256SUMS`)
are not installed. See "What the archive holds" below.

### Windows (PowerShell)

```powershell
irm https://mindlang.dev/install.ps1 | iex
```

`scripts/install.ps1` reads `-Version`, `-InstallDir` and `-DryRun`. A script
fetched with `irm` and run with `iex` does not receive arguments, so the pin
form is a script block:

```powershell
& ([scriptblock]::Create((irm https://mindlang.dev/install.ps1))) -Version v0.10.4
```

From a clone:

```powershell
powershell -File scripts/install.ps1 -Version v0.10.4
```

The script runs in Windows PowerShell 5.1 and in PowerShell 7. It downloads
`mindc-windows-x64.zip` and `mindc-windows-x64.zip.sha256` from the release
page of the tag, and installs to `%USERPROFILE%\.local\bin` unless
`-InstallDir` names another directory. It does these steps in order:

1. It compares the SHA-256 of the zip with the first field of the first line
   of the `.sha256` file. On a mismatch it stops and installs nothing. If the
   release publishes no `.sha256` file (HTTP 404), it installs without the
   check and prints a warning. Any other error while it downloads that file
   stops the install.
2. It unpacks the whole zip in a new temporary directory and requires
   `mindc.exe` in it.
3. It copies only `mindc.exe` to the install directory and removes the
   temporary directory, with the other members of the zip (`std\`,
   `stdlib_manifest.txt`, `SHA256SUMS`). It removes that directory when a step
   fails too.
4. If no entry of your user `PATH` equals the install directory, it prepends
   the directory to your user `PATH`. The current shell keeps its old `PATH`.

`-DryRun` prints the asset URL, the checksum URL and the install path, and
writes nothing. Without `-Version` it still reads the latest tag from the
GitHub API.

Windows ARM64 and 32-bit Windows have no pre-built binary. The script stops
with a message on both.

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
4. Extract the archive into an empty directory. It holds the binary (`mindc`,
   or `mindc.exe` in the Windows archive), `std/`, `stdlib_manifest.txt` and
   `SHA256SUMS` (see "What the archive holds"). Put the binary on `PATH`.

```sh
mkdir mindc-dist && tar -xzf mindc-linux-x64.tar.gz -C mindc-dist
sudo mv mindc-dist/mindc /usr/local/bin/mindc
mindc --version
```

```powershell
Expand-Archive .\mindc-windows-x64.zip -DestinationPath .\mindc-dist
Move-Item .\mindc-dist\mindc.exe $env:USERPROFILE\.local\bin\mindc.exe
mindc --version
```

## Verifying the download (SHA256)

Each asset has its own `<asset>.sha256` file. The release has no combined
checksum file for the assets. (The `SHA256SUMS` inside each archive covers the
`std/` files and the manifest, not the archive or the binary; see below.) The
Linux and macOS files are the output of
`sha256sum` or `shasum -a 256`. Run the check in the directory that holds the
asset.

```sh
sha256sum -c mindc-linux-x64.tar.gz.sha256
shasum -a 256 -c mindc-macos-arm64.tar.gz.sha256
```

```powershell
$expected = (Get-Content .\mindc-windows-x64.zip.sha256).Split()[0].ToLower()
$actual = (Get-FileHash .\mindc-windows-x64.zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw "SHA-256 mismatch" }
```

## What the archive holds

All five archives hold the same members:

| Member | Content |
|--------|---------|
| `mindc` (`mindc.exe` in the zip) | The compiler binary. |
| `std/` | Every file under `std/` in the tag's tree, with the same relative paths. |
| `stdlib_manifest.txt` | A copy of `examples/mindc_mind/testdata/stdlib_manifest.txt` of the tag. |
| `SHA256SUMS` | The sha256 of every file under `std/` and of `stdlib_manifest.txt`. It does not list the binary: the sibling `<asset>.sha256` file covers the whole archive. |

`SHA256SUMS` is sorted with `LC_ALL=C`. It has one `<hash>  <path>` line (two
spaces) per file, in the same form on every platform, the Windows zip included.
The packer hashes each file in binary mode and writes the `*` that binary mode
prints before the path (Git Bash on Windows prints it by default) as a second
space. To recompute the file in the extracted directory and compare it with the
shipped one (use `shasum -a 256 -t` where `sha256sum` is missing; `-t` selects
text mode, which writes two spaces and no `*`):

```sh
find std stdlib_manifest.txt -type f | LC_ALL=C sort | xargs sha256sum -t | diff - SHA256SUMS
sha256sum -c SHA256SUMS
```

The `std/` directory is also compiled into the binary. `MINDC_STD_DIR` names a
directory of `<module>.mind` files that the native backend reads for its seed
modules (the `seed` rows of `stdlib_manifest.txt`) instead of the copy inside
the binary: `MINDC_STD_DIR=mindc-dist/std`.

Each release leg checks its archive before it uploads anything, and the
`release_profile` job of `ci.yml` runs the same checks on the two Linux legs
before a tag exists (`.github/actions/release-artifact-smoke`). What each check
proves, and where:

| Check | Runs on | Proves |
|-------|---------|--------|
| G-b | All five legs | `SHA256SUMS` lists exactly the files under `std/` and the manifest, and every listed hash matches. A line may be `<hash>  <path>` (what the packer writes) or `<hash> *<path>` (binary mode); both verify. |
| G-c | All five legs | `stdlib_manifest.txt` is byte-equal to the manifest of the tag's tree, and `std/` holds exactly the modules the manifest names. |
| G-a | All five legs | Every file under `std/`, and the manifest, is byte-equal to the file in the tag's tree, and its whole byte sequence occurs in the binary of the same archive. Occurrence shows that the bytes are inside the binary; `tests/std_bundled_module_table.rs` pins that every std file has a row in the binary's table. |
| G-d | The x86-64 Linux leg only | A native build (`--backend native`) of a small program that imports `std.sha256`, with `MINDC_STD_DIR` set to the archive's `std/`, gives a file byte-identical to the build that uses the embedded std. The program prints the SHA-256 of the empty message, and a `MINDC_STD_DIR` that names an empty directory is refused. Only the seed modules take part in this build: an appended comment in each of the 21 seed modules of the tree the gate was written against changed the file. The other modules are covered by G-a. |

G-d runs on no other leg: the native backend runs an x86-64 compiler seed, and
the macOS, Linux ARM64 and Windows legs have no such rebuild check. For those
legs the claim is G-a, G-b and G-c.

`installers.yml` checks both installers on this layout: it packs a released
`mindc` with the same action, serves that archive to `install.sh` and
`install.ps1` in place of the download, and requires that the install directory
holds `mindc` (`mindc.exe`) and nothing else and that the installed binary
prints the release version. Those runs use an already released binary, so they
test the installers on the layout and not the new binary. On its Windows leg it
also unpacks the zip that Git Bash's `sha256sum` packed, and requires that
`SHA256SUMS` has one `<hash>  <path>` line per file, lists exactly the files of
the zip and verifies with `sha256sum -c`. That is the `SHA256SUMS` part of G-b
on the Windows runner before a tag; G-a, G-c and G-d need a binary built from
the tag's tree and run there only at the tag.

## Build from source

Rust 1.85 or later (`rust-version` in `Cargo.toml`). This is the release
feature set:

```sh
cargo install --git https://github.com/star-ga/mind --tag v0.10.4 --locked --bin mindc --features "mlir-lowering mlir-build cross-module-imports autodiff evidence-mldsa evidence-slhdsa"
```

From a local clone of tag `v0.10.4`:

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
