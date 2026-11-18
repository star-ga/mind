# install.ps1 - mindc one-line installer for Windows (PowerShell)
#
# Downloads mindc-windows-x64.zip from a GitHub release of star-ga/mind,
# checks its SHA-256 against the mindc-windows-x64.zip.sha256 file published
# beside it, unpacks mindc.exe and copies it to the install directory.
# Runs in Windows PowerShell 5.1 and in PowerShell 7. Windows x64 only.
#
# Usage:
#   irm https://mindlang.dev/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://mindlang.dev/install.ps1))) -Version v0.10.3
#   .\install.ps1
#   .\install.ps1 -DryRun          # print the URLs and the install path, write nothing
#
# iex passes no arguments. To give -Version or -DryRun to a script fetched with
# irm, use the script block form.
#
# Parameters:
#   -Version <tag>       pin a release tag, e.g. -Version v0.10.3 (default: the latest release)
#   -InstallDir <path>   install destination (default: $env:USERPROFILE\.local\bin)
#   -DryRun              print the asset URL, the checksum URL and the install path;
#                        write nothing (without -Version it still reads the latest tag
#                        from the GitHub API)
#
# Checksum: a SHA-256 mismatch stops the install and nothing is installed. If the
# release has no .sha256 file (HTTP 404), the script installs without the check and
# prints a warning. Any other failure to fetch that file stops the install.
#
# The script creates the install directory if needed (no admin rights needed for the
# default directory) and never asks for elevation. If no entry of the user PATH equals
# the install directory, it prepends the directory to the user PATH; the current shell
# keeps its old PATH.
#
# Dot-sourcing this file (. .\install.ps1) installs nothing. It defines the functions
# and sets $Version, $InstallDir, $DryRun and $MindcRepo in the calling scope.

[CmdletBinding()]
param(
    [string]$Version       = "",
    [string]$InstallDir    = "",
    [switch]$DryRun
)

$MindcRepo = "star-ga/mind"

# The release asset for the architecture name $Arch (AMD64, x86 or ARM64).
function Get-MindcAssetName {
    param([string]$Arch)
    Set-StrictMode -Version Latest

    switch ($Arch) {
        "AMD64" { return "mindc-windows-x64.zip" }
        "x86"   { throw "error: 32-bit Windows is not supported. Download manually from https://github.com/$MindcRepo/releases" }
        "ARM64" { throw "error: Windows ARM64 is not yet supported. Download manually from https://github.com/$MindcRepo/releases" }
        default { throw "error: Unknown processor architecture: $Arch" }
    }
}

# The release tag: the pinned one, or the latest release from the GitHub API.
function Get-MindcTag {
    param([string]$Pinned)
    Set-StrictMode -Version Latest

    if ($Pinned -ne "") {
        Write-Host "Pinned version: $Pinned"
        $tag = $Pinned
    } else {
        Write-Host "Fetching latest release tag from GitHub..."
        try {
            $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$MindcRepo/releases/latest" -UseBasicParsing -ErrorAction Stop
        } catch {
            throw "error: Failed to fetch latest release: $($_.Exception.Message)"
        }
        $tag = if ($release.PSObject.Properties['tag_name']) { [string]$release.tag_name } else { "" }
        if ($tag -eq "") { throw "error: Could not parse tag_name from GitHub API response" }
        Write-Host "Latest release: $tag"
    }
    # The tag goes into a URL path: refuse anything that is not a plain tag name.
    if ($tag -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "error: Invalid release tag: $tag" }
    return $tag
}

# $true when one entry of the ';'-separated list names $Dir (case and a trailing
# backslash do not matter). A longer entry that merely contains $Dir does not match.
function Test-MindcPathEntry {
    param([string]$PathList, [string]$Dir)
    Set-StrictMode -Version Latest

    $want = $Dir.TrimEnd('\')
    foreach ($entry in ($PathList -split ';')) {
        if ($entry.TrimEnd('\') -ieq $want) { return $true }
    }
    return $false
}

# Saves the checksum file. $true: saved. $false: the release has none (HTTP 404).
# Any other failure throws.
function Save-MindcChecksum {
    param([string]$Url, [string]$OutFile)
    Set-StrictMode -Version Latest

    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -ErrorAction Stop
        return $true
    } catch {
        $response = if ($_.Exception.PSObject.Properties['Response']) { $_.Exception.Response } else { $null }
        if ($null -ne $response -and [int]$response.StatusCode -eq 404) { return $false }
        throw "error: Download failed ($Url): $($_.Exception.Message)"
    }
}

# Throws unless the SHA-256 of $FilePath equals the first field of the first line of $SumFile.
function Assert-MindcSha256 {
    param([string]$FilePath, [string]$SumFile, [string]$AssetName)
    Set-StrictMode -Version Latest

    $line = Get-Content -LiteralPath $SumFile -TotalCount 1
    $expected = if ($line) { ([string]$line).Trim() -split '\s+' | Select-Object -First 1 } else { "" }
    if ($expected -notmatch '^[0-9A-Fa-f]{64}$') {
        throw "error: The checksum file for $AssetName does not start with a SHA-256 hash"
    }
    $actual = (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash
    if ($actual -ine $expected) {
        throw "error: SHA-256 mismatch for ${AssetName}:`n  expected: $($expected.ToLowerInvariant())`n  actual  : $($actual.ToLowerInvariant())"
    }
    Write-Host "Checksum OK."
}

# Downloads, verifies and unpacks the release asset in a new temporary directory,
# then copies mindc.exe to $DestPath. The directory is removed on every exit path.
function Install-MindcAsset {
    param([string]$AssetUrl, [string]$SumUrl, [string]$AssetName, [string]$DestPath)
    Set-StrictMode -Version Latest

    $work = Join-Path ([IO.Path]::GetTempPath()) ("mindc-install-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $work | Out-Null
    try {
        $zipPath = Join-Path $work $AssetName
        $sumPath = Join-Path $work "$AssetName.sha256"
        $unpack  = Join-Path $work "unpacked"

        Write-Host "Downloading $AssetName..."
        try {
            Invoke-WebRequest -Uri $AssetUrl -OutFile $zipPath -UseBasicParsing -ErrorAction Stop
        } catch {
            throw "error: Download failed ($AssetUrl): $($_.Exception.Message)"
        }

        Write-Host "Verifying checksum..."
        if (Save-MindcChecksum -Url $SumUrl -OutFile $sumPath) {
            Assert-MindcSha256 -FilePath $zipPath -SumFile $sumPath -AssetName $AssetName
        } else {
            Write-Warning "No $AssetName.sha256 is published for this release; installing without a checksum check."
        }

        Expand-Archive -LiteralPath $zipPath -DestinationPath $unpack -Force
        $exe = Join-Path $unpack "mindc.exe"
        if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
            throw "error: Expected mindc.exe inside $AssetName but it was not found"
        }

        New-Item -ItemType Directory -Path (Split-Path -Parent $DestPath) -Force | Out-Null
        Copy-Item -LiteralPath $exe -Destination $DestPath -Force
        Write-Host "Installed: $DestPath"
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Prepends $Dir to the user PATH unless one entry already names it.
function Add-MindcToUserPath {
    param([string]$Dir)
    Set-StrictMode -Version Latest

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($null -eq $userPath) { $userPath = "" }
    if (Test-MindcPathEntry -PathList $userPath -Dir $Dir) { return }

    Write-Host ""
    Write-Host "NOTE: $Dir is not in your PATH. Adding it to your user PATH..."
    $newPath = if ($userPath -eq "") { $Dir } else { "$Dir;$userPath" }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "PATH updated. Restart your shell or run:"
    Write-Host "  `$env:PATH = `"$Dir;`$env:PATH`""
}

function Install-Mindc {
    param([string]$Version, [string]$InstallDir, [switch]$DryRun)
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"
    $ProgressPreference    = "SilentlyContinue"   # the progress bar slows Windows PowerShell 5.1 downloads
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        # Windows PowerShell 5.1 can default to a TLS version that github.com refuses.
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    }

    Write-Host "=== mindc installer (Windows) ==="

    # A 32-bit PowerShell on 64-bit Windows reports x86 in PROCESSOR_ARCHITECTURE and
    # the real architecture in PROCESSOR_ARCHITEW6432.
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    $assetName = Get-MindcAssetName -Arch $arch
    Write-Host "Detected: OS=Windows  ARCH=$arch  asset=$assetName"

    $tag = Get-MindcTag -Pinned $Version
    if ($InstallDir -eq "") {
        if (-not $env:USERPROFILE) { throw "error: USERPROFILE is not set; pass -InstallDir" }
        $InstallDir = Join-Path $env:USERPROFILE ".local\bin"
    }
    $InstallDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InstallDir)
    $destPath   = Join-Path $InstallDir "mindc.exe"
    $assetUrl   = "https://github.com/$MindcRepo/releases/download/$tag/$assetName"
    $sumUrl     = "$assetUrl.sha256"

    Write-Host "Asset URL    : $assetUrl"
    Write-Host "Checksum URL : $sumUrl"
    Write-Host "Install to   : $destPath"

    if ($DryRun) {
        Write-Host ""
        Write-Host "[dry-run] Would download: $assetUrl"
        Write-Host "[dry-run] Would verify  : $sumUrl"
        Write-Host "[dry-run] Would install : $destPath"
        Write-Host "[dry-run] No files written."
        return
    }

    Install-MindcAsset -AssetUrl $assetUrl -SumUrl $sumUrl -AssetName $assetName -DestPath $destPath
    Add-MindcToUserPath -Dir $InstallDir

    Write-Host ""
    Write-Host "Run: mindc --version"
    try {
        & $destPath --version | Out-Host
    } catch {
        Write-Host "(Could not auto-run -- open a new shell and try: mindc --version)"
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    Install-Mindc -Version $Version -InstallDir $InstallDir -DryRun:$DryRun
}
