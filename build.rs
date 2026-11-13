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

/// The committed std manifest: its `seed` rows, in file order, name the std
/// modules the native backend feeds to the frozen pure-MIND compiler.
const STD_MANIFEST: &str = "examples/mindc_mind/testdata/stdlib_manifest.txt";

/// Generate `native_std_seed.rs`: one `include_bytes!` per `seed` row of the
/// manifest, in manifest order, so `mindc build --backend native` carries the
/// std seed sources inside the binary instead of reading `std/` relative to the
/// working directory. The manifest stays the single list; nothing here copies
/// it. A tree without the manifest (a packaged crate) gets an empty table, and
/// the bridge then refuses at runtime because the pinned blob hash cannot match.
fn generate_native_std_seed() {
    println!("cargo:rerun-if-changed={STD_MANIFEST}");
    println!("cargo:rerun-if-changed=build.rs");
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let mut rows = String::new();
    if let Ok(text) = std::fs::read_to_string(root.join(STD_MANIFEST)) {
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut fields = line.split('\t');
            let (Some(module), Some("seed")) = (fields.next(), fields.next()) else {
                continue;
            };
            // The path is spelled through `CARGO_MANIFEST_DIR` when the crate
            // compiles, never baked in here: OUT_DIR is shared by every
            // checkout that shares a target directory, and an absolute path
            // would point one checkout's build at another checkout's std.
            let rel = format!("/std/{module}.mind");
            rows.push_str(&format!(
                "    ({module:?}, include_bytes!(concat!(env!(\"CARGO_MANIFEST_DIR\"), {rel:?}))),\n"
            ));
        }
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(out.join("native_std_seed.rs"), format!("&[\n{rows}]\n"))
        .expect("write native_std_seed.rs");
}

/// The two compiler seeds `src/build/native_bridge/seed.rs` embeds, with the
/// `MANIFEST.txt` that pins each one.
const NATIVE_SEEDS: [&str; 2] = [
    "examples/mindc_mind/testdata/selfhost_loop",
    "examples/mindc_mind/testdata/selfhost_loop_bounded",
];

/// Verify each embedded compiler seed against its pin when mindc is built; a
/// mismatch fails the build. The pin is the `stage1.elf<TAB>size<TAB>sha256` row
/// of the seed's `MANIFEST.txt`, parsed with the rules of `seed.rs`'s
/// `pinned_stage1`. Only a build with `std-surface` embeds the seeds, so only
/// that build checks them. Nothing is written to OUT_DIR: the crate keeps its
/// source-path `include_bytes!`.
fn verify_native_seeds() {
    if std::env::var_os("CARGO_FEATURE_STD_SURFACE").is_none() {
        return;
    }
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    for dir in NATIVE_SEEDS {
        println!("cargo:rerun-if-changed={dir}/stage1.elf");
        println!("cargo:rerun-if-changed={dir}/MANIFEST.txt");
    }
    for dir in NATIVE_SEEDS {
        let manifest_path = root.join(dir).join("MANIFEST.txt");
        let manifest = std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest_path.display()));
        let pin = manifest.lines().find_map(|line| {
            let mut f = line.split('\t');
            if f.next()? != "stage1.elf" {
                return None;
            }
            let size: usize = f.next()?.parse().ok()?;
            let sha = f.next()?.trim();
            (sha.len() == 64).then_some((size, sha))
        });
        let Some((size, sha)) = pin else {
            panic!("{dir}/MANIFEST.txt has no `stage1.elf` size/sha256 row");
        };
        let elf_path = root.join(dir).join("stage1.elf");
        let elf = std::fs::read(&elf_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", elf_path.display()));
        let actual: String = {
            use sha2::Digest as _;
            sha2::Sha256::digest(&elf)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        };
        if elf.len() != size || actual != sha {
            panic!(
                "the compiler seed {dir}/stage1.elf does not match its pinned manifest \
                 (file {} bytes, sha256 {actual}; pinned {size} bytes, sha256 {sha})",
                elf.len()
            );
        }
    }
}

fn main() {
    generate_native_std_seed();
    verify_native_seeds();
    if std::env::var("CARGO_FEATURE_FFI_EXAMPLES").is_ok() {
        println!("cargo:rerun-if-changed=examples/c/min.c");
        cc::Build::new()
            .file("examples/c/min.c")
            .warnings(false)
            .compile("mind_ffi_examples");
    }
}
