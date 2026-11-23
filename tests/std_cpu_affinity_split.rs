// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The split of `std/cpu_affinity.mind` into the mask logic and the OS wrappers
//! of `std/cpu_affinity_sys.mind`, pinned by the facts the repository can show
//! without the arch-mind scanner.
//!
//! The scanner (`scripts/arch_gate.sh`) honours a declared OS-effect module only
//! when it holds straight-line wrappers: no branch, no loop, no call to a
//! non-wrapper function. It scores `std.cpu_affinity` pure only when that module
//! makes no OS call of its own. Those two facts, the declaration in
//! `.arch-mind/rules.mind` and the `#[protection]` markers of the two governed
//! decisions are read from the sources here, so a change that would turn the
//! arch gate red turns this test red first. The scanner run itself stays in
//! `scripts/arch_gate.sh`.
//!
//! Every check has a positive control: the scan finds the real functions, and
//! the same scan flags a branch and a loop in a sample.

use std::path::Path;

fn read(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// `text` without its `//` and `/* */` comments, one blank for each.
fn code_of(text: &str) -> String {
    let (mut out, mut chars) = (String::new(), text.chars().peekable());
    while let Some(c) = chars.next() {
        match (c, chars.peek()) {
            ('/', Some('/')) => {
                for d in chars.by_ref() {
                    if d == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                chars.next();
                let mut prev = ' ';
                for d in chars.by_ref() {
                    if prev == '*' && d == '/' {
                        break;
                    }
                    prev = d;
                }
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// Identifier tokens, and the punctuation `&&`, `||` and `?` as tokens of their own.
fn tokens(code: &str) -> Vec<String> {
    let (mut out, chars) = (Vec::<String>::new(), code.chars().collect::<Vec<_>>());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_alphanumeric() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }
        let pair: String = chars[i..chars.len().min(i + 2)].iter().collect();
        if pair == "&&" || pair == "||" {
            out.push(pair);
            i += 2;
            continue;
        }
        if c == '?' {
            out.push("?".to_string());
        }
        i += 1;
    }
    out
}

/// The control-flow tokens an OS-effect module may not hold (the scanner's list:
/// a branch, an `assert`, a short-circuit operator, `?`, a loop).
const FORBIDDEN: &[&str] = &[
    "if", "else", "match", "assert", "&&", "||", "?", "while", "for", "loop",
];

fn control_flow_in(code: &str) -> Vec<String> {
    tokens(code)
        .into_iter()
        .filter(|t| FORBIDDEN.contains(&t.as_str()))
        .collect()
}

/// `(name, body)` of every `fn` in `code`, the body being the text between its
/// braces. Bodies of these files never nest a brace pair inside a string.
fn functions(code: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = code;
    while let Some(at) = rest.find("fn ") {
        let after = &rest[at + 3..];
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let Some(open) = after.find('{') else { break };
        // A declaration in an `extern` block ends in `;` before any `{`.
        if after.find(';').is_some_and(|semi| semi < open) {
            rest = &after[after.find(';').unwrap() + 1..];
            continue;
        }
        let (mut depth, mut end) = (0i32, open);
        for (k, c) in after[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            if depth == 0 {
                end = open + k;
                break;
            }
        }
        out.push((name, after[open + 1..end].to_string()));
        rest = &after[end + 1..];
    }
    out
}

#[test]
fn the_scan_flags_a_branch_and_a_loop_and_reads_functions() {
    let sample = "fn a(x: i64) -> i64 {\n    if x > 0 { return 1; }\n    x\n}\n\
                  fn b(x: i64) -> i64 {\n    while x < 3 { }\n    x && x\n}\n\
                  // if in a comment\nfn c() -> i64 { 0 }\n";
    let code = code_of(sample);
    assert_eq!(control_flow_in(&code), ["if", "while", "&&"]);
    let names: Vec<String> = functions(&code).into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["a", "b", "c"]);
    assert!(control_flow_in(&code_of("fn z() -> i64 { 0 } // if else while")).is_empty());
}

#[test]
fn the_mask_module_makes_no_os_call_of_its_own() {
    let code = code_of(&read("std/cpu_affinity.mind"));
    let toks = tokens(&code);
    for banned in [
        "extern",
        "sched_setaffinity",
        "sched_getaffinity",
        "__errno_location",
    ] {
        assert!(
            !toks.iter().any(|t| t == banned),
            "std/cpu_affinity.mind holds `{banned}`: the OS calls belong to std/cpu_affinity_sys.mind"
        );
    }
    assert!(
        code.contains("import std.cpu_affinity_sys;"),
        "std/cpu_affinity.mind must import std.cpu_affinity_sys"
    );
    // The scan is not vacuous: the four public names and the range guard are there.
    let names: Vec<String> = functions(&code).into_iter().map(|(n, _)| n).collect();
    for public in [
        "cpu_affinity_pin",
        "cpu_affinity_mask",
        "cpu_affinity_has",
        "cpu_affinity_pin_demo",
        "cpu_affinity_range",
    ] {
        assert!(names.iter().any(|n| n == public), "missing `{public}`");
    }
    // The wrappers it calls are the ones the OS module defines.
    for wrapper in [
        "cpu_affinity_sys_set",
        "cpu_affinity_sys_get",
        "cpu_affinity_sys_errno",
    ] {
        assert!(toks.iter().any(|t| t == wrapper), "never calls `{wrapper}`");
    }
}

#[test]
fn the_os_module_holds_three_straight_line_wrappers() {
    let code = code_of(&read("std/cpu_affinity_sys.mind"));
    assert_eq!(
        control_flow_in(&code),
        Vec::<String>::new(),
        "std/cpu_affinity_sys.mind holds a branch or a loop: the scanner would not honour the effect module"
    );
    let fns = functions(&code);
    let names: Vec<&str> = fns.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "cpu_affinity_sys_set",
            "cpu_affinity_sys_get",
            "cpu_affinity_sys_errno"
        ],
        "the module holds exactly its three wrappers"
    );
    assert_eq!(
        code.matches("pub fn ").count(),
        3,
        "every wrapper is public and there is no other function"
    );
    // The extern block: the two affinity calls take the mask as the C pointer
    // (`const cpu_set_t *` to set, `cpu_set_t *` to get), the same forms as
    // std.iouring declares (the std manifest lint pins that agreement), and the
    // errno cell reader.
    for decl in [
        "unsafe fn sched_setaffinity(pid: i32, cpusetsize: i64, mask: *const u8) -> i32;",
        "unsafe fn sched_getaffinity(pid: i32, cpusetsize: i64, mask: *mut u8) -> i32;",
        "unsafe fn __errno_location() -> *mut i32;",
    ] {
        assert!(code.contains(decl), "missing extern `{decl}`");
    }
    // Each wrapper reaches its OS call, with pid 0, and calls nothing else of the module.
    let wants: [(&str, &str); 3] = [
        (
            "cpu_affinity_sys_set",
            "sched_setaffinity(0, size, mask as *const u8)",
        ),
        (
            "cpu_affinity_sys_get",
            "sched_getaffinity(0, size, mask as *mut u8)",
        ),
        ("cpu_affinity_sys_errno", "__errno_location()"),
    ];
    for ((name, body), (want_name, call)) in fns.iter().zip(wants) {
        assert_eq!(name, want_name);
        assert!(body.contains(call), "`{name}` does not make `{call}`");
        assert!(
            !tokens(body).iter().any(|t| t.starts_with("cpu_affinity")),
            "`{name}` calls another function of the module"
        );
    }
}

#[test]
fn the_effect_module_is_declared_and_only_the_two_decisions_are_protected() {
    let rules = code_of(&read(".arch-mind/rules.mind"));
    let declared = rules
        .lines()
        .find(|l| l.contains("const OS_EFFECT_MODULES"))
        .expect("rules.mind declares OS_EFFECT_MODULES");
    let value = declared.split('"').nth(1).expect("a quoted path list");
    let entries: Vec<&str> = value.split(',').map(str::trim).collect();
    assert!(
        entries.contains(&"std/cpu_affinity_sys.mind"),
        "OS_EFFECT_MODULES does not list std/cpu_affinity_sys.mind: {entries:?}"
    );
    assert!(
        !entries.contains(&"std/cpu_affinity.mind"),
        "the mask module must not be declared an effect module: {entries:?}"
    );
    // Positive control: the two present entries are still read.
    assert!(entries.contains(&"std/posix_sys.mind") && entries.contains(&"std/procgroup_sys.mind"));
    let mask = read("std/cpu_affinity.mind");
    assert_eq!(mask.matches("#[protection]").count(), 1, "one marker");
    assert!(
        mask.contains("#[protection]\nfn cpu_affinity_range("),
        "the range guard carries the marker"
    );
    assert!(
        !read("std/cpu_affinity_sys.mind").contains("#[protection]"),
        "no marker on the OS wrappers"
    );
    assert!(
        read("std/os_evidence.mind").contains("#[protection]\npub fn cpu_affinity_pin_ev("),
        "cpu_affinity_pin_ev carries the marker"
    );
}

/// `(found, bare)` for `text`: the number of `pub fn *_ev` verdict wrappers, and
/// the names of those whose line before is not `#[protection]`.
fn ev_wrappers_without_marker(text: &str) -> (usize, Vec<String>) {
    let lines: Vec<&str> = text.lines().collect();
    let (mut found, mut bare) = (0, Vec::new());
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("pub fn ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.ends_with("_ev") {
            continue;
        }
        found += 1;
        if i == 0 || lines[i - 1] != "#[protection]" {
            bare.push(name);
        }
    }
    (found, bare)
}

/// Every `_ev` verdict wrapper of the std evidence modules carries
/// `#[protection]`, the marker the arch gate scores. `procgroup_record_verdicts`
/// writes records only and is not a wrapper. Positive control: the scan finds
/// the six wrappers, and it flags a sample wrapper without the marker.
#[test]
fn every_std_evidence_verdict_wrapper_is_protected() {
    let mut found = 0;
    for rel in ["std/os_evidence.mind", "std/x509_evidence.mind"] {
        let (n, bare) = ev_wrappers_without_marker(&read(rel));
        assert!(
            bare.is_empty(),
            "{rel}: verdict wrappers without #[protection]: {bare:?}"
        );
        found += n;
    }
    assert_eq!(
        found, 6,
        "fs_read_dir_checked_ev, procgroup_spawn_ev, procgroup_wait_all_ev, \
         cpu_affinity_pin_ev, x509_verify_server_chain_ev, x509_verify_server_chain_budget_ev"
    );
    let sample = "#[protection]\npub fn a_ev() {}\n\npub fn b_ev() {}\npub fn c() {}\n";
    assert_eq!(
        ev_wrappers_without_marker(sample),
        (2, vec!["b_ev".to_string()]),
        "the scan flags a wrapper without the marker"
    );
}
