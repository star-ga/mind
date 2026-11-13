// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit tests for the single-file project scope.

use super::*;

fn table(module_path: &str, function: &str) -> super::super::module_table::ModuleTable {
    let source = format!("pub fn {function}() -> i64 {{ return 1; }}\n");
    let module = crate::parser::parse(&source).unwrap();
    super::super::module_table::build_module_table(&[(module_path.to_string(), &module)])
}

#[test]
fn discovery_links_a_sibling_imported_only_from_inside_a_function() {
    // The control root asked for: it must NOT be able to pass merely because
    // the frozen compiler refuses nested imports. Nothing here runs the
    // frozen compiler, or any backend at all. It asserts what DISCOVERY
    // resolved, so the only way it passes is if the nested import was seen.
    //
    // MUTATION: restore the Block-only walk and this goes red — the entry's
    // sole import lives inside a function body, so a Block-only walk sees an
    // import-free entry and returns SingleTranslationUnit instead of a
    // two-module project.
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"nested\"\nversion = \"0.1.0\"\n\n\
         [build]\ntarget = \"cpu\"\nentry = \"src/main.mind\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/helper.mind"),
        "export { helper }\n\npub fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n",
    )
    .unwrap();
    // The ONLY import is nested inside the function body.
    let main_src = "fn main() -> i64 {\n    import helper;\n    return helper(6);\n}\n";
    std::fs::write(root.join("src/main.mind"), main_src).unwrap();
    let _ = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(root)
        .status();

    let entry = root.join("src/main.mind");
    match discover_with_source(&entry, "cpu", main_src) {
        Ok(Discovery::Project(scope)) => {
            let linked: Vec<String> = scope
                .linked_sources()
                .map(|c| c.module_path().to_string())
                .collect();
            assert!(
                linked.iter().any(|m| m.ends_with("helper")),
                "discovery must link the sibling named by a NESTED import; linked {linked:?}"
            );
        }
        Ok(Discovery::SingleTranslationUnit) => panic!(
            "discovery missed the nested import and treated the entry as a single \
             translation unit — this is the Block-only walk's failure mode"
        ),
        Ok(Discovery::MissingProject(imports)) => {
            // Also acceptable evidence the import was SEEN, as long as it is named.
            assert!(
                imports.iter().any(|i| i.contains("helper")),
                "the nested import must be named; got {imports:?}"
            );
        }
        Err(e) => panic!("discovery failed: {e:#}"),
    }
}

#[test]
fn local_import_segments_are_sorted_and_exclude_std() {
    // Now driven by the parser's own record, and covering a NESTED import
    // the previous Block-only walk could not see.
    let (_module, raw) = crate::parser::parse_with_imports(
        "import zed;\nimport std.vec;\nimport crate.alpha;\nimport zed;\n\
         fn main() -> i64 {\n    import nested.dep;\n    return 1;\n}\n",
    )
    .unwrap();
    assert_eq!(
        local_import_segments(&raw).into_iter().collect::<Vec<_>>(),
        [
            vec![String::from("crate"), String::from("alpha")],
            vec![String::from("nested"), String::from("dep")],
            vec![String::from("zed")],
        ],
        "std excluded, duplicates absorbed, sorted, and the NESTED import present"
    );
}

#[test]
fn nested_scope_restores_the_previous_table_after_error() {
    crate::type_checker::cm_set_project_table(Some(table("crate.outer", "outer_fn")));
    let nested = || -> Result<()> {
        let _guard = ProjectTableGuard::install(table("crate.inner", "inner_fn"));
        assert_eq!(
            crate::type_checker::cm_imported_export_names(&["crate".into(), "inner".into()]),
            ["inner_fn"]
        );
        anyhow::bail!("test error")
    };
    assert!(nested().is_err());
    assert_eq!(
        crate::type_checker::cm_imported_export_names(&["crate".into(), "outer".into()]),
        ["outer_fn"]
    );
    crate::type_checker::cm_set_project_table(None);
}

#[test]
fn metadata_scope_is_broader_than_the_linked_body_closure() {
    let root = tempfile::tempdir().unwrap();
    let entry = root.path().join("main.mind");
    let types = root.path().join("types.mind");
    let malformed = root.path().join("stray.mind");
    let entry_source = "pub fn field(p: &Point) -> i64 { return p.y; }\n";
    fs::write(&entry, entry_source).unwrap();
    fs::write(&types, "struct Point { x: i64, y: i64 }\n").unwrap();
    fs::write(&malformed, "fn broken( -> i64 {\n").unwrap();

    let scope = capture_project_scope(
        &entry,
        entry_source,
        &[entry.clone(), types, malformed],
        root.path(),
    )
    .unwrap();

    assert_eq!(
        scope
            .sources()
            .iter()
            .map(CapturedSource::module_path)
            .collect::<Vec<_>>(),
        ["crate", "crate.types"]
    );
    assert_eq!(
        scope
            .linked_sources()
            .map(CapturedSource::module_path)
            .collect::<Vec<_>>(),
        ["crate"]
    );
}

/// A temporary manifest is never a project for import discovery either: a
/// killed build's leftover in the entry's directory is removed, and a live
/// standalone build's (its lock held, then the file removed and the lock
/// released) is waited out. Either way no project governs the entry, so its
/// local import is refused (E2042) as in a directory that never had one.
/// Read as a project, the entry's imports are scoped by another build's
/// manifest.
#[test]
fn a_temporary_manifest_never_scopes_an_entry() {
    use super::super::build_lock::{ProjectBuildLock, temporary_manifest};
    use std::sync::mpsc;
    use std::time::Duration;
    for live in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(
            root.join("helper.mind"),
            "export { seven }\nfn seven() -> i64 { return 7; }\n",
        )
        .unwrap();
        let source = "import helper;\n\nfn main() -> i64 {\n    return helper::seven();\n}\n";
        let entry = root.join("t.mind");
        fs::write(&entry, source).unwrap();
        let held = live.then(|| ProjectBuildLock::acquire(&root).unwrap());
        let manifest = root.join("Mind.toml");
        fs::write(&manifest, temporary_manifest("t", "t.mind")).unwrap();
        // Positive control: the planted text loads, so only the rule keeps it
        // from being a project.
        assert!(load_manifest(&root).is_ok());
        let (tx, rx) = mpsc::channel();
        let search = std::thread::spawn(move || {
            let found = match discover_with_source(&entry, DEFAULT_TARGET_BLOCK, source) {
                Ok(Discovery::MissingProject(imports)) => Ok(imports),
                Ok(Discovery::Project(_)) => Err("a project scopes the entry".to_string()),
                Ok(Discovery::SingleTranslationUnit) => Err("a single unit".to_string()),
                Err(error) => Err(format!("{error:#}")),
            };
            tx.send(found).unwrap();
        });
        let early = rx.recv_timeout(Duration::from_millis(300)).ok();
        if let Some(held) = held {
            assert_eq!(early, None, "discovery did not wait on the live build");
            fs::remove_file(&manifest).unwrap();
            drop(held);
        }
        let found = early.unwrap_or_else(|| rx.recv_timeout(Duration::from_secs(10)).unwrap());
        search.join().unwrap();
        assert_eq!(found, Ok(vec!["helper".to_string()]), "live {live}");
        assert!(
            !manifest.exists(),
            "live {live}: the manifest is still there"
        );
    }
}
