// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! Project management for MIND - reads Mind.toml and builds projects.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

/// Cross-module import resolution (Phase 10.6 item 9 / Phase 15
/// self-hosting prerequisite). Deliverable 1: the module table.
/// Gated; not yet wired into the type-checker (mirrors RFC 0002's
/// land-the-field-first sequencing). Default build never compiles it.
#[cfg(feature = "cross-module-imports")]
pub mod module_table;

/// RFC 0005 Phase C — std/*.mind sources baked into the binary at
/// compile time. The project loader prepends these to the module
/// table so `use std.vec` resolves in any project, no vendoring
/// required. Same gate as the rest of the cross-module work.
#[cfg(feature = "cross-module-imports")]
pub mod stdlib;

/// RFC 0008 §3 — `[test]` table in `Mind.toml`.
/// All fields default; an absent table is equivalent to default.
#[derive(Debug, Deserialize, Clone)]
pub struct TestConfig {
    /// Substring filter applied to test names at discovery time. Empty = all.
    #[serde(default)]
    pub filter: String,
    /// Run tests in parallel. Default true.
    #[serde(default = "default_test_parallel")]
    pub parallel: bool,
    /// Max parallel workers. 0 = available parallelism.
    #[serde(default)]
    pub threads: u32,
    /// Per-test timeout in seconds.
    #[serde(default = "default_test_timeout")]
    pub timeout: u32,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            filter: String::new(),
            parallel: default_test_parallel(),
            threads: 0,
            timeout: default_test_timeout(),
        }
    }
}

fn default_test_parallel() -> bool {
    true
}

fn default_test_timeout() -> u32 {
    30
}

/// RFC 0008 §3 — `[workspace]` table in `Mind.toml`.
/// Absent = single-package project (no change to existing behaviour).
#[derive(Debug, Deserialize, Clone, Default)]
pub struct WorkspaceConfig {
    /// Paths to workspace member roots (each must contain a `Mind.toml`).
    #[serde(default)]
    pub members: Vec<String>,
    /// Paths excluded from workspace glob expansions.
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// Project manifest from Mind.toml
#[derive(Debug, Deserialize, Clone)]
pub struct ProjectManifest {
    pub package: PackageInfo,
    #[serde(default)]
    pub build: BuildConfig,
    #[serde(default)]
    pub features: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub targets: HashMap<String, TargetConfig>,
    #[serde(default)]
    pub dependencies: HashMap<String, DependencySpec>,
    #[serde(default)]
    pub profile: HashMap<String, ProfileConfig>,
    /// RFC 0002 deliverable 3 — `[exports]` table in `Mind.toml`.
    #[serde(default)]
    pub exports: ExportsConfig,
    /// RFC 0007 (Mindcraft) Phase 1 — `[mindcraft]` configuration block.
    #[serde(default)]
    pub mindcraft: MindcraftConfig,
    /// RFC 0008 §3 — `[test]` table. Absent = defaults.
    #[serde(default)]
    pub test: TestConfig,
    /// RFC 0008 §3 — `[workspace]` table. Absent = single-package project.
    #[serde(default)]
    pub workspace: Option<WorkspaceConfig>,
}

/// RFC 0007 (Mindcraft) — `[mindcraft]` configuration block in
/// `Mind.toml`. All fields default; an absent table is equivalent
/// to the canonical "zero-config" state. See `docs/rfcs/0007-mindcraft.md`
/// §5 for the normative description.
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct MindcraftConfig {
    /// Published JSON-schema URL so editors offer completion + validation
    /// on `Mind.toml`. Stored verbatim — the compiler does not fetch it.
    #[serde(default, rename = "$schema")]
    pub schema: Option<String>,
    /// Per-rule severity overrides. Keys are rule ids (e.g.
    /// ``"lint::q16_overflow"``); values are ``off`` / ``info`` /
    /// ``warn`` / ``error``. Unknown ids are reported by ``mindc check``
    /// as a config-level diagnostic (not silently ignored), so typos
    /// don't get traded for surprise behaviour.
    #[serde(default)]
    pub rules: HashMap<String, RuleSeverity>,
    /// Formatter settings. Zero-config = the canonical layout; this
    /// block exists so projects with a deliberate house style can pin
    /// values — never as a tuning surface for "what looks nicer".
    #[serde(default)]
    pub format: MindcraftFormatConfig,
    /// Glob-scoped layered overrides. Later entries take precedence
    /// over earlier ones; each entry's `rules` map merges into the
    /// surrounding config (not replaces it).
    #[serde(default)]
    pub overrides: Vec<MindcraftOverride>,
    /// VCS integration — when ``use_ignore_file`` is true, the default
    /// file set for ``mindc check`` follows the repository's ignore
    /// rules (`.gitignore` etc.) so generated / vendored sources are
    /// excluded by default.
    #[serde(default)]
    pub vcs: MindcraftVcsConfig,
    /// Per-target sections. `[mindcraft.cpu]`, `[mindcraft.gpu]`,
    /// `[mindcraft.cerebras]` layer backend-specific rules first-class
    /// (e.g. a Q16.16-overflow rule that is `error` on a fixed-point
    /// target and `warn` elsewhere). Each section accepts the same
    /// keys as the surrounding block; merge order is base → per-target.
    #[serde(default)]
    pub cpu: Option<Box<MindcraftConfig>>,
    #[serde(default)]
    pub gpu: Option<Box<MindcraftConfig>>,
    #[serde(default)]
    pub cerebras: Option<Box<MindcraftConfig>>,
}

/// RFC 0007 §5 — severity model for individual rules. Mapped from
/// the lowercase TOML strings; deserialisation rejects any other
/// value with a clear error.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RuleSeverity {
    /// Rule disabled — no diagnostic emitted, no `--fix` mutation.
    Off,
    /// Informational note; does not affect exit code.
    Info,
    /// Warning; surfaces in CI output. Default for stylistic rules.
    Warn,
    /// Hard error; ``mindc check`` exits non-zero on any match.
    Error,
}

impl Default for RuleSeverity {
    fn default() -> Self {
        // Canonical default for any rule not explicitly mapped — see
        // RFC 0007 §6. Each rule ships with its own baseline; this
        // is only the fallback when the manifest mentions a rule by
        // id with no value (TOML's bare ``"lint::foo"`` form).
        RuleSeverity::Warn
    }
}

/// RFC 0007 §5 — formatter settings. Zero-config gives the canonical
/// layout; this block exists for deliberate house styles. Values are
/// validated by ``mindc fmt`` at configuration-load time.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MindcraftFormatConfig {
    /// Indent width in spaces. Canonical default 4; values <1 or >16
    /// are rejected at load time.
    #[serde(default = "default_format_indent")]
    pub indent_width: u8,
    /// Soft column limit for line wrapping. Canonical default 100.
    /// Values below 40 are rejected (impractically narrow).
    #[serde(default = "default_format_max_line_length")]
    pub max_line_length: u16,
    /// When true, the formatter inserts a trailing comma after the
    /// final element of multi-line collections. Canonical default
    /// ``true`` — matches the self-hosted front-end's own surface.
    #[serde(default = "default_format_trailing_comma")]
    pub trailing_comma: bool,
}

impl Default for MindcraftFormatConfig {
    fn default() -> Self {
        Self {
            indent_width: default_format_indent(),
            max_line_length: default_format_max_line_length(),
            trailing_comma: default_format_trailing_comma(),
        }
    }
}

fn default_format_indent() -> u8 {
    4
}

fn default_format_max_line_length() -> u16 {
    100
}

fn default_format_trailing_comma() -> bool {
    true
}

/// RFC 0007 §5 — glob-scoped override layer. Later entries in
/// `overrides = [...]` take precedence; each entry's `rules` map
/// merges into the surrounding config (not replaces).
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct MindcraftOverride {
    /// Glob patterns the override applies to (e.g.
    /// ``["tests/**", "bench/**"]``).
    #[serde(default)]
    pub includes: Vec<String>,
    /// Glob patterns explicitly excluded from this override layer
    /// even when matched by ``includes``.
    #[serde(default)]
    pub excludes: Vec<String>,
    /// Rule-severity overrides scoped to the matching files.
    #[serde(default)]
    pub rules: HashMap<String, RuleSeverity>,
    /// Format-config overrides scoped to the matching files. Absent =
    /// inherit from the surrounding block.
    #[serde(default)]
    pub format: Option<MindcraftFormatConfig>,
}

/// RFC 0007 §5 — VCS integration. When ``use_ignore_file`` is true,
/// the default file set for ``mindc check`` walks the repository's
/// ignore rules so generated / vendored sources are excluded.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MindcraftVcsConfig {
    #[serde(default = "default_vcs_use_ignore_file")]
    pub use_ignore_file: bool,
}

impl Default for MindcraftVcsConfig {
    fn default() -> Self {
        Self {
            use_ignore_file: default_vcs_use_ignore_file(),
        }
    }
}

fn default_vcs_use_ignore_file() -> bool {
    // Canonical default: respect the repo's .gitignore so generated /
    // vendored sources aren't surprise inputs to the toolchain.
    true
}

/// `[exports]` block in `Mind.toml`. Currently only `c_abi` is defined;
/// other ABI targets (python, wasm) are reserved for future RFCs.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ExportsConfig {
    #[serde(default)]
    pub c_abi: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
}

/// RFC 0008 §3 — backend target vocabulary (matches `parse_target` in `mindc.rs`).
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BuildTarget {
    #[default]
    Cpu,
    Gpu,
    Tpu,
    Npu,
    Lpu,
    Dpu,
    Fpga,
    Cerebras,
    Wasm,
}

impl BuildTarget {
    /// Canonical string name for display and path-keying.
    pub fn as_str(self) -> &'static str {
        match self {
            BuildTarget::Cpu => "cpu",
            BuildTarget::Gpu => "gpu",
            BuildTarget::Tpu => "tpu",
            BuildTarget::Npu => "npu",
            BuildTarget::Lpu => "lpu",
            BuildTarget::Dpu => "dpu",
            BuildTarget::Fpga => "fpga",
            BuildTarget::Cerebras => "cerebras",
            BuildTarget::Wasm => "wasm",
        }
    }

    /// Parse from a CLI string, accepting aliases (cuda, rocm, ...).
    /// Returns an error string on unknown input.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(BuildTarget::Cpu),
            "gpu" | "cuda" | "rocm" | "metal" | "webgpu" => Ok(BuildTarget::Gpu),
            "tpu" => Ok(BuildTarget::Tpu),
            "npu" | "ane" | "hexagon" => Ok(BuildTarget::Npu),
            "lpu" | "groq" => Ok(BuildTarget::Lpu),
            "dpu" | "smartnic" | "bluefield" => Ok(BuildTarget::Dpu),
            "fpga" | "hls" => Ok(BuildTarget::Fpga),
            "cerebras" | "wse" | "wse2" | "wse3" => Ok(BuildTarget::Cerebras),
            "wasm" => Ok(BuildTarget::Wasm),
            other => Err(format!(
                "unknown target '{}' (expected cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras|wasm)",
                other
            )),
        }
    }
}

/// RFC 0008 §3 — output artifact type.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EmitKind {
    #[default]
    Binary,
    Cdylib,
    Object,
}

impl EmitKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EmitKind::Binary => "binary",
            EmitKind::Cdylib => "cdylib",
            EmitKind::Object => "object",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "binary" => Ok(EmitKind::Binary),
            "cdylib" | "shared" => Ok(EmitKind::Cdylib),
            "object" | "obj" => Ok(EmitKind::Object),
            other => Err(format!(
                "unknown emit kind '{}' (expected binary|cdylib|object)",
                other
            )),
        }
    }
}

/// RFC 0008 §3 — optimization / profile level.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum OptimizeLevel {
    #[default]
    Debug,
    Release,
    Size,
}

impl OptimizeLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            OptimizeLevel::Debug => "debug",
            OptimizeLevel::Release => "release",
            OptimizeLevel::Size => "size",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "debug" => Ok(OptimizeLevel::Debug),
            "release" => Ok(OptimizeLevel::Release),
            "size" => Ok(OptimizeLevel::Size),
            other => Err(format!(
                "unknown optimize level '{}' (expected debug|release|size)",
                other
            )),
        }
    }

    /// Map to the `release: bool` flag used by the legacy pipeline.
    pub fn is_release(self) -> bool {
        matches!(self, OptimizeLevel::Release | OptimizeLevel::Size)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct BuildConfig {
    /// Entry-point source file (relative to manifest). Default `src/main.mind`.
    #[serde(default = "default_entry")]
    pub entry: String,
    /// Output artifact name without extension. Default: `package.name`.
    #[serde(default = "default_output")]
    pub output: String,
    /// Legacy optimization string (kept for backwards compat with existing
    /// `Mind.toml` files that set `optimization = "aggressive"`).
    #[serde(default = "default_optimization")]
    pub optimization: String,
    /// RFC 0008 §3 — backend target. Default `cpu`.
    #[serde(default)]
    pub target: BuildTarget,
    /// RFC 0008 §3 — artifact kind. Default `binary`.
    #[serde(default)]
    pub emit: EmitKind,
    /// RFC 0008 §3 — optimization level. Default `debug`.
    #[serde(default)]
    pub optimize: OptimizeLevel,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            entry: default_entry(),
            output: default_output(),
            optimization: default_optimization(),
            target: BuildTarget::default(),
            emit: EmitKind::default(),
            optimize: OptimizeLevel::default(),
        }
    }
}

fn default_entry() -> String {
    "src/main.mind".to_string()
}

fn default_output() -> String {
    "app".to_string()
}

fn default_optimization() -> String {
    "aggressive".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct TargetConfig {
    pub backend: String,
    #[serde(default)]
    pub simd: Option<String>,
    #[serde(default)]
    pub compute: Option<String>,
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum DependencySpec {
    /// Simple version string: `foo = "1.0"`.
    Simple(String),
    /// Inline table with a `path` field (Phase C / Phase D):
    /// `foo = { path = "../foo" }`.
    Path {
        path: String,
        #[serde(default)]
        features: Vec<String>,
    },
    /// Inline table with an explicit `version` field.
    Detailed {
        version: String,
        #[serde(default)]
        features: Vec<String>,
    },
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ProfileConfig {
    #[serde(rename = "opt-level", default)]
    pub opt_level: Option<u8>,
    #[serde(default)]
    pub debug: Option<bool>,
    #[serde(default)]
    pub lto: Option<bool>,
    #[serde(rename = "codegen-units", default)]
    pub codegen_units: Option<u8>,
}

/// Build options for project compilation
#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    pub release: bool,
    pub target: Option<String>,
    pub verbose: bool,
    /// RFC 0002 D3 — names from `Mind.toml [exports] c_abi`. Passed
    /// through to the compile pipeline so the AST → IR lowering pass
    /// extends `IRModule.exports` with these alongside any in-source
    /// `export { ... }` block.
    pub manifest_exports: Vec<String>,
}

/// Build result
#[derive(Debug)]
pub struct BuildResult {
    pub output_path: PathBuf,
    pub target: String,
    pub success: bool,
}

/// Find the project root by looking for Mind.toml
pub fn find_project_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;
    loop {
        let manifest_path = current.join("Mind.toml");
        if manifest_path.exists() {
            return Ok(current);
        }
        if !current.pop() {
            return Err(anyhow!(
                "Could not find Mind.toml in current directory or any parent"
            ));
        }
    }
}

/// Load project manifest from Mind.toml
pub fn load_manifest(project_root: &Path) -> Result<ProjectManifest> {
    let manifest_path = project_root.join("Mind.toml");
    let content = fs::read_to_string(&manifest_path)
        .with_context(|| format!("Failed to read {}", manifest_path.display()))?;

    let manifest: ProjectManifest = toml::from_str(&content)
        .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;

    Ok(manifest)
}

/// Collect all .mind source files from a project
pub fn collect_sources(project_root: &Path, entry: &str) -> Result<Vec<PathBuf>> {
    let entry_path = project_root.join(entry);
    if !entry_path.exists() {
        return Err(anyhow!("Entry file not found: {}", entry_path.display()));
    }

    let src_dir = entry_path.parent().unwrap_or(project_root);
    let mut sources = Vec::new();

    fn collect_recursive(dir: &Path, sources: &mut Vec<PathBuf>) -> Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    collect_recursive(&path, sources)?;
                } else if path.extension().map(|e| e == "mind").unwrap_or(false) {
                    sources.push(path);
                }
            }
        }
        Ok(())
    }

    collect_recursive(src_dir, &mut sources)?;
    Ok(sources)
}

/// Build a MIND project
pub fn build_project(opts: &BuildOptions) -> Result<BuildResult> {
    let project_root = find_project_root()?;
    let manifest = load_manifest(&project_root)?;

    // Determine target
    let target_name = opts.target.clone().unwrap_or_else(|| "cpu".to_string());
    let target_config = manifest.targets.get(&target_name);

    // Determine output name
    let output_name = if let Some(cfg) = target_config {
        cfg.output
            .clone()
            .unwrap_or_else(|| manifest.build.output.clone())
    } else {
        manifest.build.output.clone()
    };

    // Create target directory
    let profile_dir = if opts.release { "release" } else { "debug" };
    let target_dir = project_root.join("target").join(profile_dir);
    fs::create_dir_all(&target_dir)?;

    let output_path = target_dir.join(&output_name);

    // Collect sources
    let sources = collect_sources(&project_root, &manifest.build.entry)?;

    if opts.verbose {
        println!(
            "Building {} v{}",
            manifest.package.name, manifest.package.version
        );
        println!("  Target: {}", target_name);
        println!("  Profile: {}", profile_dir);
        println!("  Sources: {} files", sources.len());
    }

    // Resolve backend from target config
    let backend = if let Some(cfg) = target_config {
        cfg.backend.clone()
    } else {
        target_name.clone()
    };

    // Build each source file and link
    let compiled = compile_sources(&project_root, &sources, &backend, opts)?;

    // Link into final binary
    link_binary(&compiled, &output_path, &backend, opts)?;

    if opts.verbose {
        println!("  Output: {}", output_path.display());
    }

    Ok(BuildResult {
        output_path,
        target: target_name,
        success: true,
    })
}

/// Compile source files to object files
fn compile_sources(
    project_root: &Path,
    sources: &[PathBuf],
    backend: &str,
    opts: &BuildOptions,
) -> Result<Vec<PathBuf>> {
    let obj_dir = project_root.join("target").join("obj");
    fs::create_dir_all(&obj_dir)?;

    // Determine which file is the entry point
    let manifest = load_manifest(project_root)?;
    let entry_path = project_root.join(&manifest.build.entry);
    let entry_canonical = entry_path.canonicalize().unwrap_or(entry_path.clone());

    // RFC 0002 D3: thread Mind.toml [exports] c_abi through to each
    // per-file compile so the AST → IR lowering pass picks them up.
    // Clones once; pulled in to inner BuildOptions via a single field.
    let mut opts_with_exports = opts.clone();
    if opts_with_exports.manifest_exports.is_empty() {
        opts_with_exports.manifest_exports = manifest.exports.c_abi.clone();
    }
    let opts = &opts_with_exports;

    // Cross-module imports D3: build the whole-project module table
    // once, before the per-file compile loop, and set it at project
    // scope so each file's existing single-file pipeline resolves
    // symbols declared in sibling files. Gated; clears after the loop.
    // The per-file compile signature is unchanged (moat held).
    #[cfg(feature = "cross-module-imports")]
    {
        let src_root = project_root.join(&manifest.build.entry);
        let src_root = src_root.parent().unwrap_or(&project_root);
        // RFC 0005 Phase C — seed the parsed-modules list with the
        // bundled stdlib (`std.vec` / `std.string` / `std.map` /
        // `std.io`) before walking the project's own src tree. This
        // is what makes `use std.vec` work in a downstream `mind
        // build` without the user vendoring std/*.mind themselves.
        // The bundle is i64-ABI and uses only the seven `__mind_*`
        // intrinsics, so it parses identically to a user module.
        // User modules that happen to shadow a `std.*` name overwrite
        // the bundled entry via the last-write-wins contract on
        // `ModuleTable::insert` — same behaviour as Rust's
        // user-crate-wins-over-stdlib for `std::` shadowing.
        let mut parsed: Vec<(String, crate::ast::Module)> =
            crate::project::stdlib::parsed_stdlib_modules();
        for source in sources.iter() {
            if let Ok(text) = fs::read_to_string(source) {
                if let Ok(m) = crate::parser::parse(&text) {
                    parsed.push((
                        crate::project::module_table::module_path_of(source, src_root),
                        m,
                    ));
                }
            }
        }
        let refs: Vec<(String, &crate::ast::Module)> =
            parsed.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = crate::project::module_table::build_module_table(&refs);
        crate::type_checker::cm_set_project_table(Some(table));
    }

    let mut objects = Vec::new();

    for source in sources {
        let source_name = source
            .file_stem()
            .ok_or_else(|| anyhow!("Invalid source file: {}", source.display()))?
            .to_string_lossy();

        let obj_path = obj_dir.join(format!("{}.o", source_name));

        if opts.verbose {
            println!("  Compiling: {}", source.display());
        }

        // Check if this is the entry point
        let source_canonical = source.canonicalize().unwrap_or(source.clone());
        let is_entry = source_canonical == entry_canonical;

        // Compile with appropriate mode
        compile_single_source(source, &obj_path, backend, opts, is_entry)?;

        objects.push(obj_path);
    }

    #[cfg(feature = "cross-module-imports")]
    crate::type_checker::cm_set_project_table(None);

    Ok(objects)
}

/// Compile a single source file to native object code
#[allow(clippy::needless_return)]
fn compile_single_source(
    source: &Path,
    output: &Path,
    backend: &str,
    opts: &BuildOptions,
    is_entry: bool, // true if this is the main entry point
) -> Result<()> {
    use crate::eval;
    use crate::parser;
    use crate::pipeline::{compile_source_with_name, CompileOptions};
    use crate::runtime::types::BackendTarget;

    // Read source
    let source_code = fs::read_to_string(source)
        .with_context(|| format!("Failed to read source: {}", source.display()))?;

    // Parse and compile to IR
    let target = match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" | "rocm"
        | "rocm-mi300" | "metal" | "metal-m4" | "webgpu" | "directx" | "oneapi" => {
            BackendTarget::Gpu
        }
        // Wafer-scale: WSE-2 / WSE-3 generations both route through the
        // same logical BackendTarget; the generation is resolved in
        // the runtime library selected by find_runtime_lib.
        "cerebras" | "wse" | "wse2" | "wse3" | "cs2" | "cs3" => BackendTarget::Cerebras,
        _ => BackendTarget::Cpu,
    };

    // RFC 0002 D3: thread manifest-declared exports through to the
    // compile pipeline. Empty when [exports] is absent from Mind.toml.
    let compile_opts = CompileOptions {
        func: None,
        enable_autodiff: false,
        target,
        manifest_exports: opts.manifest_exports.clone(),
        ..Default::default()
    };

    // Try to compile - if parser doesn't support all syntax, fall back to embedding
    let _products = match compile_source_with_name(
        &source_code,
        Some(&source.to_string_lossy()),
        &compile_opts,
    ) {
        Ok(p) => p,
        Err(_) => {
            // Fall back to embedding source for runtime JIT
            return compile_embedded_source(source, &source_code, output, backend, opts, is_entry);
        }
    };

    // Parse again to get AST for IR lowering
    let module = match parser::parse_with_diagnostics(&source_code) {
        Ok(m) => m,
        Err(_) => {
            return compile_embedded_source(source, &source_code, output, backend, opts, is_entry);
        }
    };

    // Lower AST to IR, then to MLIR
    let ir_module = eval::lower_to_ir(&module);
    let preset_str = if opts.release { "arith-linalg" } else { "core" };
    let mlir_opts = eval::MlirEmitOptions {
        lower_preset: Some(preset_str.to_string()),
        mode: eval::MlirEmitMode::Executable,
    };
    let mlir = eval::emit_mlir_with_opts(&ir_module, &mlir_opts);

    // Use mlir-build if available
    #[cfg(feature = "mlir-build")]
    {
        use crate::eval::mlir_build;

        match mlir_build::resolve_tools() {
            Ok(tools) => {
                let build_opts = mlir_build::BuildOptions {
                    preset: if opts.release { "arith-linalg" } else { "core" },
                    emit_mlir_file: None,
                    emit_llvm_file: None,
                    emit_obj_file: Some(output),
                    emit_shared: None,
                    opt_pipeline: if opts.release {
                        Some("canonicalize,cse,loop-invariant-code-motion")
                    } else {
                        None
                    },
                    target_triple: Some(get_target_triple(backend)),
                };

                mlir_build::build_all(&mlir, &tools, &build_opts)
                    .map_err(|e| anyhow!("MLIR build failed: {}", e))?;

                return Ok(());
            }
            Err(_) => {
                // MLIR tools not available, fall back to embedded source
                return compile_embedded_source(
                    source,
                    &source_code,
                    output,
                    backend,
                    opts,
                    is_entry,
                );
            }
        }
    }

    #[cfg(not(feature = "mlir-build"))]
    {
        let _ = mlir; // Suppress unused variable warning
        compile_embedded_source(source, &source_code, output, backend, opts, is_entry)
    }
}

/// Compile source by embedding IR (preferred) or source (fallback) for runtime JIT.
///
/// When IR lowering succeeds, the generated wrapper includes:
///   1. `MIND_IR_<module>` - Pre-compiled MIC IR (tried first via `mind_main_ir`)
///   2. `MIND_SOURCE_<module>` - Raw source fallback (for older runtimes via `mind_main`)
fn compile_embedded_source(
    source: &Path,
    source_code: &str,
    output: &Path,
    backend: &str,
    _opts: &BuildOptions,
    is_entry: bool,
) -> Result<()> {
    let module_name = source
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .replace('-', "_");

    // Try to produce MIC IR (preferred: smaller, pre-parsed, no source exposure)
    let mic_ir = try_emit_mic(source_code);
    let escaped_ir = mic_ir.as_ref().map(|ir| {
        ir.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    });

    // Always produce escaped source as fallback for older runtimes
    let escaped_source = source_code
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");

    let c_code = if is_entry {
        let ir_const = if let Some(ref ir) = escaped_ir {
            format!(
                "static const char MIND_IR_{module}[] = \"{ir}\";\n",
                module = module_name,
                ir = ir,
            )
        } else {
            String::new()
        };
        let ir_dispatch = if escaped_ir.is_some() {
            format!(
                r#"
    /* Try IR-aware entry point first (v0.2.0+ runtime) */
    typedef int (*mind_main_ir_fn)(int argc, char** argv, const char* ir, const char* backend);
    mind_main_ir_fn mind_ir_ptr = (mind_main_ir_fn)dlsym(lib, "mind_main_ir");
    if (mind_ir_ptr) {{
        int result = mind_ir_ptr(argc, argv, MIND_IR_{module}, MIND_BACKEND);
        dlclose(lib);
        return result;
    }}
"#,
                module = module_name,
            )
        } else {
            String::new()
        };

        format!(
            r#"
/* Auto-generated MIND entry point wrapper */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

static const char MIND_SOURCE_{module}[] = "{source}";
{ir_const}static const char MIND_BACKEND[] = "{backend}";
static const char MIND_FILE[] = "{file}";

typedef int (*mind_main_fn)(int argc, char** argv, const char* source, const char* backend);

int main(int argc, char** argv) {{
    void* lib = dlopen("libmind_{backend}_linux-x64.so", RTLD_NOW);
    if (!lib) {{
        lib = dlopen("libmind_cpu_linux-x64.so", RTLD_NOW);
    }}
    if (!lib) {{
        fprintf(stderr, "Error: MIND runtime not found\\n");
        fprintf(stderr, "See https://mindlang.dev/enterprise for licensing.\\n");
        return 1;
    }}
{ir_dispatch}
    /* Fall back to source-based entry */
    mind_main_fn mind_main_ptr = (mind_main_fn)dlsym(lib, "mind_main");
    if (!mind_main_ptr) {{
        fprintf(stderr, "Error: mind_main not found in runtime\\n");
        dlclose(lib);
        return 1;
    }}

    int result = mind_main_ptr(argc, argv, MIND_SOURCE_{module}, MIND_BACKEND);
    dlclose(lib);
    return result;
}}
"#,
            module = module_name,
            source = escaped_source,
            ir_const = ir_const,
            ir_dispatch = ir_dispatch,
            backend = backend,
            file = source.file_name().unwrap_or_default().to_string_lossy(),
        )
    } else {
        // Non-entry module: export both IR and source
        let ir_export = if let Some(ref ir) = escaped_ir {
            format!(
                r#"
static const char MIND_MODULE_{module}_IR[] = "{ir}";

const char* mind_module_{module}_get_ir(void) {{
    return MIND_MODULE_{module}_IR;
}}
"#,
                module = module_name,
                ir = ir,
            )
        } else {
            String::new()
        };

        format!(
            r#"
/* Auto-generated MIND module: {file} */
static const char MIND_MODULE_{module}_SOURCE[] = "{source}";

const char* mind_module_{module}_get_source(void) {{
    return MIND_MODULE_{module}_SOURCE;
}}
{ir_export}"#,
            module = module_name,
            source = escaped_source,
            file = source.file_name().unwrap_or_default().to_string_lossy(),
            ir_export = ir_export,
        )
    };

    // Write C file and compile with cc
    let c_path = output.with_extension("c");
    fs::write(&c_path, &c_code)?;

    let status = Command::new("cc")
        .args(["-c", "-fPIC", "-O2"])
        .arg(&c_path)
        .arg("-o")
        .arg(output)
        .status()
        .with_context(|| "Failed to run C compiler")?;

    if !status.success() {
        return Err(anyhow!("C compilation failed for {}", source.display()));
    }

    // Clean up intermediate C file
    let _ = fs::remove_file(&c_path);

    Ok(())
}

/// Try to produce MIC IR from source. Returns None if parsing/lowering fails.
fn try_emit_mic(source_code: &str) -> Option<String> {
    use crate::eval;
    use crate::ir;
    use crate::parser;

    let module = parser::parse_with_diagnostics(source_code).ok()?;
    let mut ir_module = eval::lower_to_ir(&module);
    ir::prepare_ir_for_backend(&mut ir_module).ok()?;
    Some(ir::compact::emit_mic(&ir_module))
}

/// Get the LLVM target triple for a backend
#[allow(dead_code)]
fn get_target_triple(backend: &str) -> &'static str {
    match backend {
        // GPU backends use host triple for the launcher
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" | "rocm"
        | "rocm-mi300" | "webgpu" | "directx" | "oneapi" => {
            #[cfg(target_os = "linux")]
            {
                "x86_64-unknown-linux-gnu"
            }
            #[cfg(target_os = "macos")]
            {
                "aarch64-apple-darwin"
            }
            #[cfg(target_os = "windows")]
            {
                "x86_64-pc-windows-msvc"
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                "x86_64-unknown-linux-gnu"
            }
        }
        "metal" | "metal-m4" => "aarch64-apple-darwin",
        _ => {
            #[cfg(target_os = "linux")]
            {
                "x86_64-unknown-linux-gnu"
            }
            #[cfg(target_os = "macos")]
            {
                "aarch64-apple-darwin"
            }
            #[cfg(target_os = "windows")]
            {
                "x86_64-pc-windows-msvc"
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                "x86_64-unknown-linux-gnu"
            }
        }
    }
}

/// Link object files into a native executable binary
fn link_binary(
    objects: &[PathBuf],
    output: &Path,
    backend: &str,
    opts: &BuildOptions,
) -> Result<()> {
    // Check for runtime library
    let lib_dir = find_runtime_lib(backend)?;

    if opts.verbose {
        println!("  Linking with runtime: {}", lib_dir.display());
    }

    // Determine runtime library name
    #[allow(unused_variables)]
    let (runtime_lib, runtime_link) = get_runtime_lib_names(backend);

    // Try native linking first
    let link_result = native_link(objects, output, &lib_dir, runtime_link, opts);

    if link_result.is_ok() {
        return Ok(());
    }

    // Fallback to wrapper script if native linking fails
    if opts.verbose {
        println!("  Native linking failed, creating launcher script");
    }

    let project_root = find_project_root()?;
    let manifest = load_manifest(&project_root)?;
    let entry_path = project_root.join(&manifest.build.entry);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let script = format!(
            r#"#!/bin/bash
# {name} v{version} - Built with MIND
# Backend: {backend}
# Runtime: {runtime}

MIND_ROOT="{project_root}"
MIND_ENTRY="{entry}"
MIND_BACKEND="{backend}"
MIND_LIB_DIR="{lib_dir}"

export LD_LIBRARY_PATH="$MIND_LIB_DIR:$LD_LIBRARY_PATH"
export MIND_SOURCE_ROOT="$MIND_ROOT/src"
export MIND_BACKEND="$MIND_BACKEND"

# Execute the runtime interpreter
if [ -f "$MIND_LIB_DIR/mind-runtime" ]; then
    exec "$MIND_LIB_DIR/mind-runtime" --backend "$MIND_BACKEND" --entry "$MIND_ENTRY" "$@"
elif [ -f "$MIND_LIB_DIR/{runtime}" ]; then
    # Direct library execution via ld.so
    exec /lib64/ld-linux-x86-64.so.2 --library-path "$MIND_LIB_DIR" "$MIND_LIB_DIR/{runtime}" --entry "$MIND_ENTRY" "$@" 2>/dev/null || \
    exec /lib/ld-linux-x86-64.so.2 --library-path "$MIND_LIB_DIR" "$MIND_LIB_DIR/{runtime}" --entry "$MIND_ENTRY" "$@" 2>/dev/null || \
    {{
        echo "Error: MIND runtime not executable"
        echo "See https://mindlang.dev/enterprise for licensing."
        exit 1
    }}
else
    echo "Error: MIND runtime not found at $MIND_LIB_DIR/{runtime}"
    echo "See https://mindlang.dev/enterprise for licensing."
    exit 1
fi
"#,
            name = manifest.package.name,
            version = manifest.package.version,
            backend = backend,
            runtime = runtime_lib,
            project_root = project_root.display(),
            entry = entry_path.display(),
            lib_dir = lib_dir.display(),
        );

        fs::write(output, script)?;

        let mut perms = fs::metadata(output)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(output, perms)?;
    }

    #[cfg(not(unix))]
    {
        let script = format!(
            "@echo off\r\nrem {} v{} - Built with MIND\r\nset MIND_BACKEND={}\r\nset PATH=%PATH%;{}\r\n\"{}/mind-runtime.exe\" --backend {} --entry \"{}\" %*\r\n",
            manifest.package.name,
            manifest.package.version,
            backend,
            lib_dir.display(),
            lib_dir.display(),
            backend,
            entry_path.display(),
        );
        fs::write(output, script)?;
    }

    Ok(())
}

/// Get runtime library names for a backend
fn get_runtime_lib_names(backend: &str) -> (&'static str, &'static str) {
    match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_cuda_linux-x64.so", "mind_cuda_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_cuda_windows-x64.dll", "mind_cuda_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_cuda_linux-x64.so", "mind_cuda_linux-x64")
            }
        }
        "rocm" | "rocm-mi300" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_rocm_linux-x64.so", "mind_rocm_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_rocm_windows-x64.dll", "mind_rocm_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_rocm_linux-x64.so", "mind_rocm_linux-x64")
            }
        }
        "metal" | "metal-m4" => ("libmind_metal_macos-arm64.dylib", "mind_metal_macos-arm64"),
        "webgpu" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_webgpu_linux-x64.so", "mind_webgpu_linux-x64")
            }
            #[cfg(target_os = "macos")]
            {
                (
                    "libmind_webgpu_macos-arm64.dylib",
                    "mind_webgpu_macos-arm64",
                )
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_webgpu_windows-x64.dll", "mind_webgpu_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                ("libmind_webgpu_linux-x64.so", "mind_webgpu_linux-x64")
            }
        }
        "directx" => ("mind_directx_windows-x64.dll", "mind_directx_windows-x64"),
        "oneapi" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_oneapi_linux-x64.so", "mind_oneapi_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_oneapi_windows-x64.dll", "mind_oneapi_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_oneapi_linux-x64.so", "mind_oneapi_linux-x64")
            }
        }
        _ => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_cpu_linux-x64.so", "mind_cpu_linux-x64")
            }
            #[cfg(target_os = "macos")]
            {
                ("libmind_cpu_macos-arm64.dylib", "mind_cpu_macos-arm64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_cpu_windows-x64.dll", "mind_cpu_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                ("libmind_cpu_linux-x64.so", "mind_cpu_linux-x64")
            }
        }
    }
}

/// Attempt native linking with clang/gcc
fn native_link(
    objects: &[PathBuf],
    output: &Path,
    lib_dir: &Path,
    runtime_link: &str,
    opts: &BuildOptions,
) -> Result<()> {
    // Find a C compiler for linking
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());

    let mut cmd = Command::new(&cc);

    // Add object files
    for obj in objects {
        cmd.arg(obj);
    }

    // Output path
    cmd.arg("-o").arg(output);

    // Link against runtime library
    cmd.arg(format!("-L{}", lib_dir.display()));
    cmd.arg(format!("-l{}", runtime_link));

    // Link standard libraries
    cmd.arg("-ldl"); // For dlopen on Linux
    cmd.arg("-lpthread");
    cmd.arg("-lm");

    // Add rpath for runtime library lookup
    #[cfg(target_os = "linux")]
    {
        cmd.arg(format!("-Wl,-rpath,{}", lib_dir.display()));
        cmd.arg("-Wl,-rpath,$ORIGIN/../lib");
    }

    #[cfg(target_os = "macos")]
    {
        cmd.arg(format!("-Wl,-rpath,{}", lib_dir.display()));
        cmd.arg("-Wl,-rpath,@executable_path/../lib");
    }

    // Optimization flags
    if opts.release {
        cmd.arg("-O3");
        cmd.arg("-flto");
    }

    if opts.verbose {
        println!("  Link command: {:?}", cmd);
    }

    let status = cmd
        .status()
        .with_context(|| format!("Failed to run linker: {}", cc))?;

    if !status.success() {
        return Err(anyhow!(
            "Linking failed with exit code: {:?}",
            status.code()
        ));
    }

    Ok(())
}

/// Find the MIND runtime library for a backend.
///
/// Search order:
///   1. `MIND_LIB_DIR` environment override.
///   2. `~/.mind/lib` (canonical install path).
fn find_runtime_lib(backend: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Cannot determine home directory"))?;
    let mind_lib = home.join(".mind").join("lib");

    // Map backend to library name
    let lib_name = match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" => {
            "libmind_cuda_linux-x64.so"
        }
        "rocm" | "rocm-mi300" => "libmind_rocm_linux-x64.so",
        "metal" | "metal-m4" => "libmind_metal_macos-arm64.dylib",
        "webgpu" => "libmind_webgpu_linux-x64.so",
        // Wafer-scale: separate library per generation because the
        // CSL toolchain and host SDK pins differ for WSE-2 vs WSE-3.
        // Default "cerebras" selects WSE-3 (CS-3) since that is the
        // current shipping generation.
        "cerebras" | "wse3" | "cs3" => "libmind_cerebras_wse3_linux-x64.so",
        "wse2" | "cs2" => "libmind_cerebras_wse2_linux-x64.so",
        _ => "libmind_cpu_linux-x64.so",
    };

    // 1. MIND_LIB_DIR environment override (highest priority)
    if let Ok(env_dir) = std::env::var("MIND_LIB_DIR") {
        let env_path = PathBuf::from(&env_dir);
        if env_path.join(lib_name).exists() {
            return Ok(env_path);
        }
    }

    // 2. Canonical install path
    let lib_path = mind_lib.join(lib_name);
    if lib_path.exists() {
        return Ok(mind_lib);
    }

    // 3. Fallback: return the canonical dir if it exists at all so the
    //    caller can produce a precise "library file missing" error rather
    //    than a generic "directory not found".
    if mind_lib.exists() {
        return Ok(mind_lib);
    }

    Err(anyhow!(
        "MIND runtime not found for backend '{}'. See https://mindlang.dev/enterprise for licensing.",
        backend
    ))
}

/// Run a built project
pub fn run_project(args: &[String], opts: &BuildOptions) -> Result<i32> {
    // First build
    let result = build_project(opts)?;

    if !result.success {
        return Err(anyhow!("Build failed"));
    }

    // Then run
    let status = Command::new(&result.output_path)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("Failed to execute {}", result.output_path.display()))?;

    Ok(status.code().unwrap_or(1))
}

/// Test options
pub struct TestOptions {
    pub target: Option<String>,
    pub verbose: bool,
    pub filter: Option<String>,
}

/// Bench options
pub struct BenchOptions {
    pub target: Option<String>,
    pub verbose: bool,
    pub filter: Option<String>,
    pub iterations: Option<u32>,
    pub json: bool,
}

/// Run project tests (discover test files in tests/, build each, run)
pub fn test_project(opts: &TestOptions) -> Result<i32> {
    let project_root = find_project_root()?;
    let manifest = load_manifest(&project_root)?;
    let tests_dir = project_root.join("tests");

    if !tests_dir.exists() {
        println!("No tests directory found.");
        return Ok(0);
    }

    let test_files: Vec<PathBuf> = fs::read_dir(&tests_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "mind").unwrap_or(false))
        .filter(|p| {
            if let Some(ref filter) = opts.filter {
                p.file_stem()
                    .map(|s| s.to_string_lossy().contains(filter.as_str()))
                    .unwrap_or(false)
            } else {
                true
            }
        })
        .collect();

    if test_files.is_empty() {
        println!("No test files found.");
        return Ok(0);
    }

    let target = opts.target.clone().unwrap_or_else(|| "cpu".to_string());
    println!(
        "\nRunning {} test suite(s) for {} v{} (target: {})...\n",
        test_files.len(),
        manifest.package.name,
        manifest.package.version,
        target,
    );

    let mut total_pass = 0u32;
    let mut total_fail = 0u32;
    let start = std::time::Instant::now();

    for test_file in &test_files {
        let name = test_file.file_stem().unwrap_or_default().to_string_lossy();
        print!("  {}:", name);

        // Build the test file as a standalone binary
        let build_opts = BuildOptions {
            release: false,
            target: Some(target.clone()),
            verbose: false,
            ..Default::default()
        };

        // Use the test file as entry point
        let test_manifest_path = project_root.join("Mind.toml");
        let orig_manifest = fs::read_to_string(&test_manifest_path)?;

        // Temporarily patch entry to test file
        let test_entry = format!("tests/{}", test_file.file_name().unwrap().to_string_lossy());
        let patched = orig_manifest.replace(
            &format!("entry = \"{}\"", manifest.build.entry),
            &format!("entry = \"{}\"", test_entry),
        );
        fs::write(&test_manifest_path, &patched)?;

        let result = build_project(&build_opts);

        // Restore original manifest
        fs::write(&test_manifest_path, &orig_manifest)?;

        match result {
            Ok(build_result) if build_result.success => {
                let run_status = Command::new(&build_result.output_path)
                    .stdin(Stdio::null())
                    .stdout(if opts.verbose {
                        Stdio::inherit()
                    } else {
                        Stdio::piped()
                    })
                    .stderr(if opts.verbose {
                        Stdio::inherit()
                    } else {
                        Stdio::piped()
                    })
                    .status();

                match run_status {
                    Ok(status) if status.success() => {
                        println!(" PASS");
                        total_pass += 1;
                    }
                    Ok(_) => {
                        println!(" FAIL");
                        total_fail += 1;
                    }
                    Err(e) => {
                        println!(" ERROR ({})", e);
                        total_fail += 1;
                    }
                }
            }
            Ok(_) => {
                println!(" BUILD FAILED");
                total_fail += 1;
            }
            Err(e) => {
                println!(" BUILD ERROR ({})", e);
                total_fail += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\nResults: {} passed, {} failed ({:.1}ms total)",
        total_pass,
        total_fail,
        elapsed.as_secs_f64() * 1000.0,
    );

    Ok(if total_fail > 0 { 1 } else { 0 })
}

/// Run project benchmarks (discover bench files in bench/, build each with --release, run)
pub fn bench_project(opts: &BenchOptions) -> Result<i32> {
    let project_root = find_project_root()?;
    let manifest = load_manifest(&project_root)?;
    let bench_dir = project_root.join("bench");

    if !bench_dir.exists() {
        println!("No bench directory found.");
        return Ok(0);
    }

    let bench_files: Vec<PathBuf> = fs::read_dir(&bench_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "mind").unwrap_or(false))
        .filter(|p| {
            if let Some(ref filter) = opts.filter {
                p.file_stem()
                    .map(|s| s.to_string_lossy().contains(filter.as_str()))
                    .unwrap_or(false)
            } else {
                true
            }
        })
        .collect();

    if bench_files.is_empty() {
        println!("No benchmark files found.");
        return Ok(0);
    }

    let target = opts.target.clone().unwrap_or_else(|| "cpu".to_string());
    println!("================================================================================");
    println!(
        "{} v{} PERFORMANCE BENCHMARK",
        manifest.package.name, manifest.package.version,
    );
    println!("================================================================================");
    println!("Target: {}", target);
    println!("Bench files: {}", bench_files.len(),);
    println!("================================================================================\n");

    let mut any_fail = false;

    for bench_file in &bench_files {
        let name = bench_file.file_stem().unwrap_or_default().to_string_lossy();
        println!("Benchmark: {}", name);

        // Build with release optimizations
        let build_opts = BuildOptions {
            release: true,
            target: Some(target.clone()),
            verbose: false,
            ..Default::default()
        };

        let test_manifest_path = project_root.join("Mind.toml");
        let orig_manifest = fs::read_to_string(&test_manifest_path)?;

        let bench_entry = format!(
            "bench/{}",
            bench_file.file_name().unwrap().to_string_lossy()
        );
        let patched = orig_manifest.replace(
            &format!("entry = \"{}\"", manifest.build.entry),
            &format!("entry = \"{}\"", bench_entry),
        );
        fs::write(&test_manifest_path, &patched)?;

        let result = build_project(&build_opts);

        // Restore
        fs::write(&test_manifest_path, &orig_manifest)?;

        match result {
            Ok(build_result) if build_result.success => {
                let mut cmd = Command::new(&build_result.output_path);
                if let Some(iters) = opts.iterations {
                    cmd.arg(format!("--iterations={}", iters));
                }
                if opts.json {
                    cmd.arg("--json");
                }

                let status = cmd
                    .stdin(Stdio::null())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .status();

                match status {
                    Ok(s) if s.success() => {
                        println!();
                    }
                    Ok(_) => {
                        println!("  BENCHMARK FAILED\n");
                        any_fail = true;
                    }
                    Err(e) => {
                        println!("  ERROR: {}\n", e);
                        any_fail = true;
                    }
                }
            }
            Ok(_) => {
                println!("  BUILD FAILED\n");
                any_fail = true;
            }
            Err(e) => {
                println!("  BUILD ERROR: {}\n", e);
                any_fail = true;
            }
        }
    }

    println!("================================================================================");

    Ok(if any_fail { 1 } else { 0 })
}
