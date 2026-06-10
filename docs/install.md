# Installing mindc

`mindc` is the MIND compiler driver. Pre-built binaries are available for Linux,
macOS, and Windows — no Rust toolchain required.

## One-line install

### Linux and macOS

```sh
curl -sSL https://mindlang.dev/install.sh | sh
```

The script:
- Detects your OS and architecture automatically
- Downloads the latest release from GitHub
- Verifies the SHA256 checksum
- Installs to `/usr/local/bin/mindc` (or `~/.local/bin/mindc` if `/usr/local/bin` is not writable)

To pin a specific version:

```sh
MINDC_VERSION=v0.7.1 curl -sSL https://mindlang.dev/install.sh | sh
```

To preview what the script would do without writing any files:

```sh
curl -sSL https://mindlang.dev/install.sh | sh -s -- --dry-run
```

Note: `https://mindlang.dev/install.sh` is a redirect to the canonical
`scripts/install.sh` in this repository. The redirect is configured separately
on the mindlang.dev Cloudflare Pages site.

### Windows (PowerShell)

```powershell
irm https://mindlang.dev/install.ps1 | iex
```

Or with a pinned version:

```powershell
$env:MINDC_VERSION = "v0.7.1"; irm https://mindlang.dev/install.ps1 | iex
```

The default install directory is `%USERPROFILE%\.local\bin`. The script adds
it to your user `PATH` automatically if it is not already present.

## Manual install

1. Go to the [Releases page](https://github.com/star-ga/mind/releases).
2. Download the asset for your platform:

   | Platform | Asset |
   |----------|-------|
   | Linux x86_64 | `mindc-<version>-x86_64-unknown-linux-musl` |
   | macOS (Intel + Apple Silicon) | `mindc-<version>-x86_64-apple-darwin-universal` |
   | Windows x86_64 | `mindc-<version>-x86_64-pc-windows-msvc.exe` |

3. Verify the checksum (see below).
4. Place the binary on your `PATH` and make it executable:

   ```sh
   # Linux / macOS
   chmod +x mindc-<version>-x86_64-unknown-linux-musl
   sudo mv mindc-<version>-x86_64-unknown-linux-musl /usr/local/bin/mindc
   mindc --version
   ```

   ```powershell
   # Windows
   Move-Item .\mindc-<version>-x86_64-pc-windows-msvc.exe C:\mindc\mindc.exe
   # Add C:\mindc to your PATH via System Properties > Environment Variables
   mindc --version
   ```

## Verifying the download (SHA256)

Every release includes a `SHA256SUMS` file. Use it to confirm the asset
you downloaded has not been tampered with.

```sh
# Linux
sha256sum --check --ignore-missing SHA256SUMS

# macOS
shasum -a 256 --check --ignore-missing SHA256SUMS

# Windows (PowerShell)
$hash = (Get-FileHash .\mindc-<version>-x86_64-pc-windows-msvc.exe -Algorithm SHA256).Hash.ToLower()
Select-String $hash .\SHA256SUMS
```

## Build from source

If you have Rust installed (1.85 or later):

```sh
cargo install --git https://github.com/star-ga/mind --bin mindc
```

Or from a local clone:

```sh
git clone https://github.com/star-ga/mind.git
cd mind
cargo build --release --bin mindc
# binary at: target/release/mindc
```

## Verify your install

```sh
mindc --version
mindc --help
```

## Supported platforms

| Platform | Architecture | Binary type |
|----------|-------------|-------------|
| Linux | x86_64 | Static (musl) — runs on any glibc-free or glibc distro |
| macOS | x86_64 + arm64 | Universal binary (lipo) |
| Windows | x86_64 | MSVC dynamic CRT |

Linux ARM64 and Windows ARM64 are not yet available as pre-built binaries.
Build from source on those platforms.
