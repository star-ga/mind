// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

mod manifest;

pub use manifest::MindManifest;

use std::collections::BTreeMap;
use std::fs;
use std::fs::File;

use std::io::Cursor;
use std::io::Read;

use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;

use anyhow::Context;

use anyhow::Result;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use sha2::Digest;
use sha2::Sha256;

use tar::Archive;
use tar::Builder;
use tar::Header;

pub fn build_package(out_path: &str, files: &[&str], manifest: &MindManifest) -> Result<()> {
    if manifest.files.len() != files.len() {
        return Err(anyhow!(
            "manifest lists {} files but {} were provided",
            manifest.files.len(),
            files.len()
        ));
    }

    let mut checksums = BTreeMap::new();

    let out_file =
        File::create(out_path).with_context(|| format!("unable to create {}", out_path))?;
    let encoder = GzEncoder::new(out_file, Compression::default());
    let mut builder = Builder::new(encoder);

    for (source, listed_name) in files.iter().zip(manifest.files.iter()) {
        if Path::new(listed_name).is_absolute() {
            return Err(anyhow!("manifest entry {listed_name} must be relative"));
        }
        let data = fs::read(source).with_context(|| format!("failed to read artifact {source}"))?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let checksum = format!("{:x}", hasher.finalize());
        let filename = Path::new(listed_name)
            .file_name()
            .ok_or_else(|| anyhow!("manifest entry {listed_name} has no file name"))?;

        let mut header = Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, filename, Cursor::new(data))?;

        checksums.insert(listed_name.clone(), checksum);
    }

    let mut manifest = manifest.clone();
    manifest.checksums = Some(checksums);

    let manifest_toml = manifest.to_toml()?;
    let manifest_bytes = manifest_toml.as_bytes();
    let mut header = Header::new_gnu();
    header.set_size(manifest_bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder.append_data(&mut header, "package.toml", Cursor::new(manifest_bytes))?;

    builder.finish()?;
    let mut encoder = builder.into_inner()?;
    encoder.try_finish()?;
    Ok(())
}

pub fn inspect_package(path: &str) -> Result<MindManifest> {
    let file = File::open(path).with_context(|| format!("unable to open package {path}"))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    let mut manifest_data = Vec::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        if entry.path()?.as_ref() == Path::new("package.toml") {
            entry.read_to_end(&mut manifest_data)?;
            break;
        }
    }

    if manifest_data.is_empty() {
        return Err(anyhow!("package.toml not found in package"));
    }

    let manifest_toml = String::from_utf8(manifest_data)?;
    let manifest: MindManifest = toml::from_str(&manifest_toml)?;
    Ok(manifest)
}

pub fn install_package(path: &str, target_dir: &str) -> Result<()> {
    let manifest = inspect_package(path)?;
    let target_root = if target_dir.is_empty() {
        default_install_dir(&manifest)?
    } else {
        PathBuf::from(target_dir)
    };
    fs::create_dir_all(&target_root)?;

    let file = File::open(path).with_context(|| format!("unable to open package {path}"))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let entry_path = entry.path()?.into_owned();
        if entry_path
            .components()
            .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(anyhow!("package contains invalid path traversal entries"));
        }
        if entry_path.as_path() == Path::new("package.toml") {
            // manifest handled separately
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            fs::write(target_root.join("package.toml"), buf)?;
            continue;
        }

        let dest = target_root.join(entry_path);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        entry.unpack(dest)?;
    }

    Ok(())
}

pub fn default_install_dir(manifest: &MindManifest) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("unable to determine home directory"))?;
    Ok(home
        .join(".mind")
        .join("packages")
        .join(format!("{}-{}", manifest.name, manifest.version)))
}
