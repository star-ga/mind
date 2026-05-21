# install.ps1 - mindc one-line installer for Windows (PowerShell)
#
# Usage:
#   irm https://mindlang.dev/install.ps1 | iex
#   .\install.ps1
#   .\install.ps1 -DryRun          # print what would happen, do nothing
#
# Environment / parameter overrides:
#   -Version <tag>       pin a specific release, e.g. -Version v0.6.9
#   -InstallDir <path>   override install destination
#
# Default install directory:
#   $env:USERPROFILE\.local\bin   (created if needed; no admin required)
#   Falls back to prompting for elevation if the user explicitly passes
#   -InstallDir "C:\Program Files\Mindc\bin"

[CmdletBinding()]
param(
    [string]$Version       = "",
    [string]$InstallDir    = "",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Repo   = "star-ga/mind"
$Binary = "mindc.exe"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step { param([string]$Msg) Write-Host $Msg }
function Fail       { param([string]$Msg) Write-Error "error: $Msg"; exit 1 }

# ── Detect architecture ───────────────────────────────────────────────────────

function Get-AssetSuffix {
    $arch = $env:PROCESSOR_ARCHITECTURE
    switch ($arch) {
        "AMD64"  { return "x86_64-pc-windows-msvc.exe" }
        "x86"    { Fail "32-bit Windows is not supported. Download manually from https://github.com/$Repo/releases" }
        "ARM64"  { Fail "Windows ARM64 is not yet supported. Download manually from https://github.com/$Repo/releases" }
        default  { Fail "Unknown processor architecture: $arch" }
    }
}

# ── Resolve version ───────────────────────────────────────────────────────────

function Resolve-Version {
    if ($Version -ne "") {
        Write-Step "Pinned version: $Version"
        return $Version
    }

    Write-Step "Fetching latest release tag from GitHub..."
    $apiUrl = "https://api.github.com/repos/$Repo/releases/latest"
    try {
        $response = Invoke-RestMethod -Uri $apiUrl -UseBasicParsing
        $tag = $response.tag_name
        if ([string]::IsNullOrEmpty($tag)) { Fail "Could not parse tag_name from GitHub API response" }
        Write-Step "Latest release: $tag"
        return $tag
    } catch {
        Fail "Failed to fetch latest release: $_"
    }
}

# ── Resolve install directory ─────────────────────────────────────────────────

function Resolve-InstallDir {
    if ($InstallDir -ne "") { return $InstallDir }
    return Join-Path $env:USERPROFILE ".local\bin"
}

# ── Verify SHA256 ─────────────────────────────────────────────────────────────

function Test-Sha256 {
    param([string]$FilePath, [string]$SumsContent, [string]$AssetName)

    $actualHash = (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash.ToLower()

    # Find the line: "<hash>  <asset-name>" or "<hash>  ./<asset-name>"
    $matchLine = $SumsContent -split "`n" | Where-Object { $_ -match [regex]::Escape($AssetName) } | Select-Object -First 1
    if ([string]::IsNullOrEmpty($matchLine)) {
        Fail "No checksum entry for $AssetName in SHA256SUMS"
    }

    $expectedHash = ($matchLine -split '\s+')[0].Trim().ToLower()
    if ($actualHash -ne $expectedHash) {
        Fail "SHA256 mismatch for ${AssetName}:`n  expected: $expectedHash`n  actual  : $actualHash"
    }
    Write-Step "Checksum OK."
}

# ── Main ──────────────────────────────────────────────────────────────────────

Write-Step "=== mindc installer (Windows) ==="

$assetSuffix = Get-AssetSuffix
$arch        = $env:PROCESSOR_ARCHITECTURE
Write-Step "Detected: OS=Windows  ARCH=$arch  asset suffix=$assetSuffix"

$tag         = Resolve-Version
$installDir  = Resolve-InstallDir
$version     = $tag.TrimStart('v')
$assetName   = "mindc-$tag-$assetSuffix"
$baseUrl     = "https://github.com/$Repo/releases/download/$tag"
$assetUrl    = "$baseUrl/$assetName"
$sumsUrl     = "$baseUrl/SHA256SUMS"
$destPath    = Join-Path $installDir $Binary

Write-Step "Asset URL : $assetUrl"
Write-Step "Install to: $destPath"

if ($DryRun) {
    Write-Step ""
    Write-Step "[dry-run] Would download: $assetUrl"
    Write-Step "[dry-run] Would verify  : $sumsUrl"
    Write-Step "[dry-run] Would install : $destPath"
    Write-Step "[dry-run] No files written."
    exit 0
}

# ── Download ──────────────────────────────────────────────────────────────────

$tmpDir   = [System.IO.Path]::GetTempPath()
$tmpAsset = Join-Path $tmpDir $assetName
$tmpSums  = Join-Path $tmpDir "SHA256SUMS"

Write-Step "Downloading $assetName..."
try {
    Invoke-WebRequest -Uri $assetUrl -OutFile $tmpAsset -UseBasicParsing
} catch {
    Fail "Download failed: $_"
}

Write-Step "Downloading SHA256SUMS..."
try {
    Invoke-WebRequest -Uri $sumsUrl -OutFile $tmpSums -UseBasicParsing
} catch {
    Fail "Download failed (SHA256SUMS): $_"
}

# ── Verify ────────────────────────────────────────────────────────────────────

Write-Step "Verifying checksum..."
$sumsContent = Get-Content -Path $tmpSums -Raw
Test-Sha256 -FilePath $tmpAsset -SumsContent $sumsContent -AssetName $assetName

# ── Install ───────────────────────────────────────────────────────────────────

if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

Copy-Item -Path $tmpAsset -Destination $destPath -Force
Write-Step "Installed: $destPath"

# Clean up temp files
Remove-Item $tmpAsset, $tmpSums -ErrorAction SilentlyContinue

# ── PATH advisory ──────────────────────────────────────────────────────────────

$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$installDir*") {
    Write-Step ""
    Write-Step "NOTE: $installDir is not in your PATH."
    Write-Step "Adding it to your user PATH..."
    $newPath = "$installDir;$currentPath"
    [System.Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Step "PATH updated. Restart your shell or run:"
    Write-Step "  `$env:PATH = `"$installDir;`$env:PATH`""
}

Write-Step ""
Write-Step "Run: mindc --version"
try {
    & $destPath --version
} catch {
    Write-Step "(Could not auto-run -- open a new shell and try: mindc --version)"
}
