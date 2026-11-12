//! `[mind]` toolchain-pin enforcement for `Mind.toml`, and the `[build] backend`
//! route rule (see [`Backend`]).
//!
//! # Why this module exists
//!
//! The `[mind]` table is the ONLY *declared* cross-repo compatibility contract
//! in the ecosystem byte boundary: a downstream repo states the compiler window
//! it was written against (`mindc-min` / `mindc-max`) and the IR interchange
//! format it couples to (`ir-format`). For a long time `ProjectManifest` had no
//! field for the table, so every one of those declarations was parsed into
//! nothing and discarded. The consequence was fail-OPEN in the worst place: a
//! repo pinned to the `0.3.x` era built clean under `0.10.x` without a word, so
//! an INCOMPATIBLE compiler was indistinguishable from a compatible one at
//! build time.
//!
//! # The contract enforced here
//!
//! * The window is **half-open, `[min, max)`** — `mindc-min` inclusive,
//!   `mindc-max` exclusive. That is what the ecosystem manifests already mean:
//!   they pin `max` at the next minor (`0.10.2` / `0.11.0`), so an inclusive
//!   ceiling would admit the entire `0.11` line the repo never declared.
//! * Either bound may be omitted; an omitted bound is simply unbounded on that
//!   side. A bound that is PRESENT but unparsable is an error — never a
//!   permissive default, which would silently re-open the hole.
//! * `ir-format` must name a format this toolchain can actually emit. Declaring
//!   a format the compiler cannot produce is a compatibility claim the compiler
//!   cannot honour.
//! * The `[mind]` table itself is `deny_unknown_fields`: a typo'd key used to
//!   disable the contract silently, which is the same fail-open shape one level
//!   down. Strictness stops at this table because it is the one the COMPILER
//!   owns; unmodelled top-level tables elsewhere in `Mind.toml` belong to other
//!   tooling and are reported by `unrecognized_keys_note` instead of refused
//!   (a shipped ecosystem manifest carrying `[determinism]` or `[protection]`
//!   is not thereby incompatible with this compiler).
//!
//! Enforcement runs inside `load_manifest`, so every path that loads a manifest
//! to build inherits the refusal rather than each caller re-deriving it.
//!
//! deferred: `mindc check` (`src/check/mod.rs`) and `mindc fmt`
//! (`src/fmt/cli.rs`) load the manifest through `if let Ok(..) = load_manifest`,
//! so a pin violation makes them fall back to default config in silence instead
//! of reporting it. That is NOT a fail-open for the build — `build` propagates
//! the error and refuses — but it is a missing diagnostic on two read-only
//! surfaces. Upgrade path: give those two call sites an explicit `Err` arm that
//! surfaces the refusal as a config-level diagnostic, rather than widening this
//! module.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow};
use serde::Deserialize;

/// The compiler version compared against the declared window. Read from the
/// crate version so a release bump cannot leave the gate measuring a stale
/// number.
pub const RUNNING_TOOLCHAIN_VERSION: &str = env!("CARGO_PKG_VERSION");

/// IR interchange formats this toolchain can actually emit today.
///
/// `mic@1` is the textual form written by `--emit-mic`; `mic@3` is the
/// canonical binary artifact written by `--emit-mic3` and the object the
/// byte-identity gates compare. `mic@2` / MIC-B are deliberately ABSENT: the
/// v2 codec exists in-tree but no CLI invocation reaches it, so this compiler
/// cannot produce that format and must not accept a manifest claiming it can.
/// That is a statement about the EMITTER only — the v2 codec remains a supported
/// library parse surface (`compact::v2`), which is why the refusal says so
/// rather than implying the format is gone.
pub const EMITTABLE_IR_FORMATS: &[&str] = &["mic@1", "mic@3"];

/// The `[mind]` table of `Mind.toml` — the declared toolchain contract.
///
/// `deny_unknown_fields` is load-bearing: without it a misspelt `mindc-maxx`
/// deserialises to "no ceiling declared" and the repo believes it is pinned
/// when it is not.
#[derive(Debug, Deserialize, Clone, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MindSection {
    /// Oldest compiler this project declares itself compatible with
    /// (INCLUSIVE). Absent = unbounded below.
    #[serde(default, rename = "mindc-min")]
    pub mindc_min: Option<String>,
    /// First compiler this project does NOT declare compatibility with
    /// (EXCLUSIVE). Absent = unbounded above.
    #[serde(default, rename = "mindc-max")]
    pub mindc_max: Option<String>,
    /// Language-spec revision the sources target. Recorded, not yet enforced —
    /// there is no spec-revision registry in the compiler to check it against,
    /// and inventing one here would be a gate that asserts nothing.
    /// deferred: enforce once a spec-revision registry exists — upgrade path is
    /// to validate against that registry in `enforce`, alongside `ir-format`.
    #[serde(default, rename = "spec-version")]
    pub spec_version: Option<String>,
    /// IR interchange format the project couples to. Validated against
    /// [`EMITTABLE_IR_FORMATS`].
    #[serde(default, rename = "ir-format")]
    pub ir_format: Option<String>,
    /// Downstream consumer of this project's emitted IR, as a name + range
    /// (e.g. `"mind-runtime >= 0.2.4"`). Recorded, not enforced: the compiler
    /// has no view of which consumer is installed, so a check here could only
    /// assert something it cannot observe. Modelled rather than left unknown
    /// because a shipped ecosystem manifest declares it — under
    /// `deny_unknown_fields` an unmodelled REAL key is indistinguishable from a
    /// typo, and refusing it would be a build stop with no compatibility
    /// meaning.
    /// deferred: enforce once the toolchain can observe the installed consumer
    /// version — upgrade path is to resolve the consumer through the dependency
    /// table and compare here, alongside `ir-format`.
    #[serde(default, rename = "ir-consumer")]
    pub ir_consumer: Option<String>,
}

/// A `major.minor.patch` version, compared componentwise.
///
/// Deliberately total and dependency-free: the only versions compared are the
/// crate's own and the two declared bounds, and an input this cannot parse is
/// reported rather than coerced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Version(u64, u64, u64);

fn parse_version(raw: &str) -> Option<Version> {
    // Tolerate a build/pre-release suffix on the RUNNING version only in the
    // sense that it is split off before numeric parsing; the bounds themselves
    // are expected to be plain triples.
    let core = raw.split(['-', '+']).next().unwrap_or(raw);
    let mut parts = core.split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse().ok()?;
    let patch = parts.next().unwrap_or("0").trim().parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(Version(major, minor, patch))
}

/// Refuse when the running compiler falls outside the manifest's declared
/// window, or when the manifest declares an IR format this compiler cannot
/// emit.
///
/// `manifest_path` is threaded through purely so the diagnostic can name the
/// file an operator has to edit — a refusal that does not say WHERE the pin
/// lives just moves the search cost onto the operator.
pub fn enforce(section: &MindSection, running: &str, manifest_path: &Path) -> Result<()> {
    let where_ = manifest_path.display();

    let running_v = parse_version(running).ok_or_else(|| {
        anyhow!(
            "toolchain pin: cannot parse the running compiler version {running:?} \
             declared in {where_}'s window check"
        )
    })?;

    if let Some(min_raw) = &section.mindc_min {
        let min_v = parse_version(min_raw).ok_or_else(|| {
            anyhow!(
                "toolchain pin: `mindc-min = {min_raw:?}` in {where_} is not a \
                 major.minor.patch version"
            )
        })?;
        if running_v < min_v {
            return Err(anyhow!(
                "toolchain pin: this project declares `mindc-min = {min_raw:?}` in \
                 {where_}, but the running compiler is {running}. Build refused: the \
                 project was written for a NEWER toolchain than this one. Upgrade the \
                 compiler, or widen the declared window if {running} is genuinely \
                 supported."
            ));
        }
    }

    if let Some(max_raw) = &section.mindc_max {
        let max_v = parse_version(max_raw).ok_or_else(|| {
            anyhow!(
                "toolchain pin: `mindc-max = {max_raw:?}` in {where_} is not a \
                 major.minor.patch version"
            )
        })?;
        // EXCLUSIVE ceiling — see the module docs.
        if running_v >= max_v {
            return Err(anyhow!(
                "toolchain pin: this project declares `mindc-max = {max_raw:?}` \
                 (exclusive) in {where_}, but the running compiler is {running}. Build \
                 refused: {running} is outside the declared compatibility window, so \
                 its semantics were never validated against these sources. Re-validate \
                 and raise `mindc-max`, or build with a compiler below {max_raw}."
            ));
        }
    }

    if let Some(fmt) = &section.ir_format {
        if !EMITTABLE_IR_FORMATS.contains(&fmt.as_str()) {
            return Err(anyhow!(
                "toolchain pin: this project declares `ir-format = {fmt:?}` in {where_}, \
                 but this compiler can only emit {emittable:?}. Build refused: no \
                 `mindc` invocation produces {fmt:?}, so accepting the pin would attest \
                 an interchange compatibility the toolchain does not have. Migrate the \
                 consumer to `mic@3` (the canonical artifact format), or drop the key \
                 if the project does not consume compiler-emitted IR. (mic@2 / MIC-B \
                 still exist as a library PARSE surface — `compact::v2` — this key is \
                 about what the compiler can EMIT.)",
                emittable = EMITTABLE_IR_FORMATS
            ));
        }
    }

    Ok(())
}

/// Render the operator-facing note for top-level `Mind.toml` keys the compiler
/// does not model, or `None` when there are none.
///
/// Split out as a pure function so the wording is testable without capturing the
/// process's stderr, and so the decision "report, do not refuse" sits next to
/// the pin it is deliberately weaker than.
pub fn unrecognized_keys_note(
    unrecognized: &BTreeMap<String, toml::Value>,
    manifest_path: &Path,
) -> Option<String> {
    if unrecognized.is_empty() {
        return None;
    }
    // BTreeMap iteration is ordered, so the note is byte-stable across runs.
    let keys: Vec<&str> = unrecognized.keys().map(String::as_str).collect();
    Some(format!(
        "mindc: warning: unused manifest key(s) in {}: {} — the compiler does not \
         model these and will not act on them. If one is a typo of a key it does \
         model, the declaration it was meant to make is NOT in effect.",
        manifest_path.display(),
        keys.join(", ")
    ))
}

/// Print [`unrecognized_keys_note`] at most once per manifest path per process.
///
/// One `mindc build` loads the same manifest several times (resolve, build,
/// link). Repeating an identical warning once per load reads like three
/// different problems, which is how a real diagnostic gets trained into
/// background noise — so the message is deduplicated on the path it names.
pub fn report_unrecognized_keys(
    unrecognized: &BTreeMap<String, toml::Value>,
    manifest_path: &Path,
) {
    let Some(note) = unrecognized_keys_note(unrecognized, manifest_path) else {
        return;
    };
    static REPORTED: OnceLock<Mutex<BTreeSet<PathBuf>>> = OnceLock::new();
    let reported = REPORTED.get_or_init(|| Mutex::new(BTreeSet::new()));
    // A poisoned lock must not swallow the diagnostic: recover the set rather
    // than returning, or a panic elsewhere would silently disable reporting.
    let mut seen = match reported.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if seen.insert(manifest_path.to_path_buf()) {
        eprintln!("{note}");
    }
}

// ---------------------------------------------------------------------------
// `[build] backend` — the declared code-generation route
// ---------------------------------------------------------------------------

/// Code-generation backend: which toolchain turns the project into an artifact.
///
/// `Native` (the default, [`DEFAULT_BACKEND`]) is the frozen pure-MIND x86-64
/// native-ELF compiler behind the native bridge, with its own admission rules
/// and zero MLIR/LLVM/clang. `Mlir` is the MLIR-text → `mlir-opt` /
/// `mlir-translate` / `clang` pipeline, selected only by name.
///
/// One value selects one route, from either input: `mindc build --backend <b>`
/// and `[build] backend = "<b>"` meet in [`select_backend`]. A route that cannot
/// honor the declared value refuses it (`admit_mlir_route`) and never builds
/// with the other toolchain in its place: the artifact, its byte identity and its
/// evidence differ per route, so a substituted route is a different build.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// MLIR-text → mlir-opt/mlir-translate/clang (selected by name only).
    Mlir,
    /// Pure-MIND x86-64 native-ELF compiler, reached through the native bridge.
    Native,
}

/// The backend of `mindc build`, `run` and `bench` with no `--backend` flag and
/// no `[build] backend` key. The one place the default is named: no program
/// edit, host or environment variable changes it, and there is no `auto`.
pub const DEFAULT_BACKEND: Backend = Backend::Native;

/// Which input selected the backend; printed in the `backend:` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSource {
    /// `--backend` on the command line.
    Flag,
    /// `[build] backend` in the governing Mind.toml.
    Manifest,
    /// Neither: [`DEFAULT_BACKEND`].
    Default,
}

/// The first stderr line of each build: `backend: <native|mlir> (from
/// <flag|manifest|default>)`. No path and no time, so it is the same line on
/// each host and each run.
pub fn backend_line(backend: Backend, source: BackendSource) -> String {
    let from = match source {
        BackendSource::Flag => "flag",
        BackendSource::Manifest => "manifest",
        BackendSource::Default => "default",
    };
    format!("backend: {} (from {from})", backend.as_str())
}

/// The sentence after each native refusal: how to select the MLIR backend.
pub const MLIR_SELECTION_HINT: &str = "Select the MLIR backend with `--backend mlir` or \
     `[build] backend = \"mlir\"` (needs mlir-opt, mlir-translate, clang).";

/// The host check as the bridge applies it. `MINDC_NATIVE_ELF` is a test-only
/// substitution hook (it names another compiler, not the seed); with it unset
/// (`substituted == false`) the host check always runs.
pub fn native_host_gate(arch: &str, os: &str, substituted: bool) -> Option<String> {
    if substituted {
        return None;
    }
    native_host_refusal(arch, os)
}

/// Refuse a native build on a host the native backend does not run on
/// (`backend.native_host`). The frozen compiler seed is an x86-64 Linux ELF
/// and its output is an x86-64 Linux executable, so any other `(arch, os)`
/// (`std::env::consts` spellings) is refused before the seed starts.
pub fn native_host_refusal(arch: &str, os: &str) -> Option<String> {
    (arch != "x86_64" || os != "linux").then(|| {
        format!(
            "the native backend runs on x86-64 Linux only; this host is {arch} {os}. \
             Refusing before the compiler seed starts. (backend.native_host)"
        )
    })
}

/// Refuse, by name, a `mindc build` request the native backend cannot honor:
/// a target other than `cpu` (`backend.native_target`), `--package`
/// (`backend.native_package`), a workspace root (`backend.native_workspace`)
/// and emit `cdylib|object` (`backend.native_emit`). Target and emit are the
/// flag, else the manifest's `[build]` value. `--release`,
/// `--optimize` and `--no-cache` are accepted and do not change native bytes.
pub fn native_request_refusal(
    target: Option<&str>,
    package: Option<&str>,
    workspace_root: bool,
    emit: Option<&str>,
) -> Option<String> {
    if let Some(t) = target.filter(|t| *t != "cpu") {
        return Some(format!(
            "`--target={t}` (or `[build] target`): the native backend emits an x86-64 \
             Linux executable for the cpu target only. (backend.native_target)"
        ));
    }
    if let Some(p) = package {
        return Some(format!(
            "`--package {p}`: the native backend does not build workspace members. \
             (backend.native_package)"
        ));
    }
    if workspace_root {
        return Some(
            "this is a workspace root: the native backend does not build workspaces. \
             (backend.native_workspace)"
                .to_string(),
        );
    }
    match emit {
        None | Some("binary") => None,
        Some(other) => Some(format!(
            "`--emit={other}` (or `[build] emit`): the native backend emits a static \
             executable (ET_EXEC) only, never a library or an object file. (backend.native_emit)"
        )),
    }
}

/// Refuse a native selection for `mindc run` / `mindc bench`
/// (`backend.native_run` / `backend.native_bench`): in this release they have
/// no native implementation, and they never run the MLIR pipeline in silence.
pub fn native_command_refusal(command: &str) -> String {
    format!(
        "`mindc {command}` has no native backend in this release; refusing rather \
         than running the MLIR pipeline in silence. (backend.native_{command})"
    )
}

impl Backend {
    /// Canonical string name for display and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Mlir => "mlir",
            Backend::Native => "native",
        }
    }

    /// Parse from a CLI string. Returns an error string on unknown input.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "mlir" => Ok(Backend::Mlir),
            "native" => Ok(Backend::Native),
            other => Err(format!(
                "unknown backend '{}' (expected mlir|native)",
                other
            )),
        }
    }
}

/// Deserialize `[build] backend`. An unknown value fails the manifest load
/// (E2043), so every command that loads the manifest refuses it; no command
/// reads it as "unset" and builds with the default route.
pub(crate) fn deserialize_backend<'de, D>(de: D) -> Result<Option<Backend>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(de)?;
    Backend::parse(&raw).map(Some).map_err(|_| {
        serde::de::Error::custom(format!(
            "E2043: `[build] backend = \"{raw}\"` names no code-generation backend \
             (expected `mlir` or `native`); refusing rather than building with a default"
        ))
    })
}

/// The route rule for `mindc build`, `run` and `bench`: the flag, else
/// `[build] backend`, else [`DEFAULT_BACKEND`]; the source names which one.
///
/// A flag that DISAGREES with the manifest is refused (E2044) rather than given
/// precedence. `[build] backend` is the project's declared build contract; a
/// precedence rule in either direction builds, in silence, with a toolchain
/// that one of the two inputs did not ask for.
pub fn select_backend(
    flag: Option<&str>,
    declared: Option<Backend>,
) -> Result<(Backend, BackendSource), String> {
    let flag = flag.map(Backend::parse).transpose()?;
    match (flag, declared) {
        (Some(flag), Some(declared)) if flag != declared => Err(format!(
            "E2044: `--backend {}` disagrees with `[build] backend = \"{}\"` in Mind.toml; \
             refusing to choose one. Remove the flag or change the manifest.",
            flag.as_str(),
            declared.as_str()
        )),
        (Some(flag), _) => Ok((flag, BackendSource::Flag)),
        (None, Some(declared)) => Ok((declared, BackendSource::Manifest)),
        (None, None) => Ok((DEFAULT_BACKEND, BackendSource::Default)),
    }
}

/// `[build] backend` of the manifest that governs `mindc build <paths>`, found
/// by the same discovery the build uses: the bounded, git-aware scan from the
/// first explicit entry's directory, else the ancestor scan from the working
/// directory.
///
/// `None` when no manifest governs the build, or for a virtual workspace root
/// (no `[package]`). Any other manifest that does not load is an error here,
/// so an unknown value is refused (E2043) and nothing builds with the default.
pub fn governing_backend(paths: &[String]) -> Result<Option<Backend>> {
    Ok(governing_manifest(paths)?.and_then(|(_, manifest)| manifest.build.backend))
}

/// The root and manifest that govern `mindc build <paths>` (see
/// [`governing_backend`] for the discovery); `None` when none governs or loads.
///
/// The route is chosen before the build takes its lock, so this search follows
/// the build's rule itself (`build_lock::governing_root`): a temporary manifest
/// of a concurrent standalone build is never a project.
pub fn governing_manifest(paths: &[String]) -> Result<Option<(PathBuf, super::ProjectManifest)>> {
    let entry_dir = match paths.first() {
        None => None,
        Some(entry) => {
            let entry = std::path::absolute(entry)?;
            match super::canonical_dir(entry.parent().unwrap_or(entry.as_path()))? {
                Some(dir) => Some(dir),
                None => return Ok(None),
            }
        }
    };
    let root = super::build_lock::governing_root(|| match &entry_dir {
        None => Ok(super::find_project_root().ok()),
        Some(dir) => super::find_project_root_for_file_checked(dir),
    })?;
    let Some(root) = root else { return Ok(None) };
    match super::load_manifest(&root) {
        Ok(manifest) => Ok(Some((root, manifest))),
        // A virtual workspace root has no `[package]`: no route is declared.
        Err(_) if is_workspace_manifest(&root) => Ok(None),
        // Any other load error (an unknown `[build] backend` is E2043) is
        // refused here, never read as "undeclared" and built with the default.
        Err(e) => Err(e.context("manifest error")),
    }
}

/// Whether `root/Mind.toml` declares `[workspace]` (the same test `mindc` uses
/// to dispatch a workspace build).
fn is_workspace_manifest(root: &Path) -> bool {
    std::fs::read_to_string(root.join("Mind.toml")).is_ok_and(|t| t.contains("[workspace]"))
}

/// Refuse a manifest that the MLIR project pipeline cannot honor (E2043).
///
/// Every MLIR project build passes here: `mindc build` on the MLIR route,
/// `mindc run`, `mindc bench`, project test files and workspace members. Two
/// declarations reach this pipeline and cannot be honored by it:
/// `[build] backend = "native"`, which only the native bridge builds, and a
/// code-generation backend name written in `[targets.<name>] backend`, the key
/// that selects the TARGET CLASS. Before this check the second one fell through
/// the class match to `cpu` and built an MLIR artifact.
pub(crate) fn admit_mlir_route(
    declared: Option<Backend>,
    target_name: &str,
    target_backend: &str,
) -> Result<()> {
    if declared == Some(Backend::Native) {
        return Err(anyhow!(
            "E2043: `[build] backend = \"native\"` selects the native bridge, but this \
             command compiles through the MLIR project pipeline, which cannot honor it; \
             refusing rather than building an MLIR artifact. Build the package with \
             `mindc build` from its own root."
        ));
    }
    if let Ok(named) = Backend::parse(target_backend) {
        return Err(anyhow!(
            "E2043: `[targets.{target_name}] backend = \"{target_backend}\"` names a \
             code-generation backend, but that key selects the target class \
             (cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras|wasm); set `[build] backend = \"{}\"` \
             to select the code-generation route.",
            named.as_str()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_ordering_is_componentwise_not_lexical() {
        // The lexical trap: "0.10.2" < "0.3.0" as strings, which would invert
        // every window check the ecosystem actually relies on.
        assert!(parse_version("0.10.2").unwrap() > parse_version("0.3.0").unwrap());
        assert!(parse_version("0.10.2").unwrap() < parse_version("0.11.0").unwrap());
    }

    #[test]
    fn short_and_malformed_versions() {
        assert_eq!(parse_version("1"), Some(Version(1, 0, 0)));
        assert_eq!(parse_version("1.2"), Some(Version(1, 2, 0)));
        assert_eq!(parse_version("1.2.3.4"), None);
        assert_eq!(parse_version("not.a.version"), None);
        assert_eq!(parse_version(""), None);
    }

    #[test]
    fn empty_section_is_unconstrained() {
        let s = MindSection::default();
        assert!(enforce(&s, "0.10.2", Path::new("Mind.toml")).is_ok());
    }

    #[test]
    fn unrecognized_note_is_absent_when_nothing_is_unrecognized() {
        assert!(unrecognized_keys_note(&BTreeMap::new(), Path::new("Mind.toml")).is_none());
    }

    #[test]
    fn unrecognized_note_names_every_key_in_deterministic_order() {
        let mut m = BTreeMap::new();
        m.insert("zeta".to_string(), toml::Value::Boolean(true));
        m.insert("alpha".to_string(), toml::Value::Boolean(true));
        let note = unrecognized_keys_note(&m, Path::new("/p/Mind.toml")).expect("note");
        assert!(note.contains("alpha") && note.contains("zeta"), "{note}");
        assert!(note.contains("/p/Mind.toml"), "{note}");
        // Sorted, not insertion-ordered: the diagnostic must not vary run to run.
        assert!(
            note.find("alpha") < note.find("zeta"),
            "keys must be reported in sorted order: {note}"
        );
    }

    /// RULE 1.1/1.2: the flag, else the manifest, else native; `auto` is refused.
    #[test]
    fn backend_selection_order_and_default_is_native() {
        assert_eq!(DEFAULT_BACKEND, Backend::Native);
        let pick = |flag, declared| select_backend(flag, declared).unwrap();
        assert_eq!(pick(None, None), (Backend::Native, BackendSource::Default));
        let declared = Some(Backend::Mlir);
        assert_eq!(
            pick(None, declared),
            (Backend::Mlir, BackendSource::Manifest)
        );
        assert_eq!(
            pick(Some("mlir"), None),
            (Backend::Mlir, BackendSource::Flag)
        );
        assert!(select_backend(Some("auto"), None).is_err());
        assert!(
            select_backend(Some("native"), declared)
                .unwrap_err()
                .contains("E2044")
        );
        assert_eq!(
            backend_line(Backend::Native, BackendSource::Default),
            "backend: native (from default)"
        );
    }

    /// RULE 1.4 (m3): only x86-64 Linux passes; every other host is refused by key.
    /// Mutant: make `native_host_refusal` return `None` and this test fails.
    #[test]
    fn native_host_check_admits_x86_64_linux_only() {
        assert_eq!(native_host_refusal("x86_64", "linux"), None);
        for (arch, os) in [
            ("aarch64", "linux"),
            ("aarch64", "macos"),
            ("x86_64", "macos"),
        ]
        .into_iter()
        .chain([("x86_64", "windows"), ("riscv64", "linux")])
        {
            let msg = native_host_refusal(arch, os).expect("refused");
            assert!(msg.contains("(backend.native_host)"), "{arch} {os}: {msg}");
            // With the test-only hook unset, the bridge's gate always checks.
            assert_eq!(native_host_gate(arch, os, false), Some(msg));
            assert_eq!(native_host_gate(arch, os, true), None);
        }
    }

    /// RULE 1.3/1.5: the refusal sentence is verbatim, and each native request
    /// the backend cannot honor is refused by its own key.
    #[test]
    fn native_request_refusals_are_named() {
        assert_eq!(
            MLIR_SELECTION_HINT,
            "Select the MLIR backend with `--backend mlir` or `[build] backend = \"mlir\"` \
             (needs mlir-opt, mlir-translate, clang)."
        );
        let refuse = native_request_refusal;
        assert_eq!(refuse(Some("cpu"), None, false, Some("binary")), None);
        assert_eq!(refuse(None, None, false, None), None);
        let rows = [
            (
                refuse(Some("gpu"), None, false, None),
                "backend.native_target",
            ),
            (
                refuse(None, Some("m"), false, None),
                "backend.native_package",
            ),
            (refuse(None, None, true, None), "backend.native_workspace"),
            (
                refuse(None, None, false, Some("cdylib")),
                "backend.native_emit",
            ),
            (
                refuse(None, None, false, Some("object")),
                "backend.native_emit",
            ),
        ];
        for (msg, key) in rows {
            assert!(
                msg.is_some_and(|m| m.contains(&format!("({key})"))),
                "{key}"
            );
        }
        assert!(native_command_refusal("run").contains("(backend.native_run)"));
    }

    #[test]
    fn the_shipped_compiler_version_parses() {
        // Guards a release bump to a shape this parser rejects, which would
        // turn every pinned manifest into a hard error at once.
        assert!(parse_version(RUNNING_TOOLCHAIN_VERSION).is_some());
    }
}
