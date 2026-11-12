// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! New support cannot compensate for a previously supported fixture refusing.

use super::Outcome;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

fn verdict(outcome: &Outcome) -> &'static str {
    match outcome {
        Outcome::Match => "MATCH",
        Outcome::RejectMatch => "REJECT_MATCH",
        _ => "NOT_PRESERVED",
    }
}

fn changed_cases<'a>(
    expected: &BTreeMap<&'a str, &'a str>,
    observed: &BTreeMap<&'a str, &'a str>,
) -> Vec<&'a str> {
    expected
        .iter()
        .filter_map(|(path, required)| (observed.get(path) != Some(required)).then_some(*path))
        .collect()
}

pub(super) fn assert_preserved(root: &Path, rows: &[(PathBuf, Outcome)]) {
    let mut expected = BTreeMap::new();
    let mut stale = Vec::new();
    for line in include_str!("../fixtures/g2_expected_outcomes.tsv").lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "malformed G2 capability row: {line}");
        let path = fields[0];
        assert!(
            matches!(fields[1], "MATCH" | "REJECT_MATCH"),
            "invalid G2 verdict"
        );
        let bytes = fs::read(root.join(path)).expect("pinned G2 fixture is missing");
        let digest = format!("{:x}", Sha256::digest(bytes));
        println!(
            "G2 row {path} hash={digest} pinned={} required={}",
            fields[2], fields[1]
        );
        if digest != fields[2] {
            stale.push(format!("{path} measured={digest} pinned={}", fields[2]));
        }
        assert!(
            expected.insert(path, fields[1]).is_none(),
            "duplicate G2 fixture {path}"
        );
    }
    assert!(
        !expected.is_empty(),
        "G2 capability baseline must not be empty"
    );
    let names: Vec<_> = rows
        .iter()
        .map(|(path, outcome)| {
            (
                path.strip_prefix(root)
                    .expect("G2 fixture escaped repo")
                    .to_string_lossy()
                    .into_owned(),
                verdict(outcome),
            )
        })
        .collect();
    let observed: BTreeMap<_, _> = names
        .iter()
        .map(|(name, status)| (name.as_str(), *status))
        .collect();
    for (path, required) in &expected {
        let seen = observed.get(path).copied().unwrap_or("UNMEASURED");
        println!("G2 outcome {path} measured={seen} required={required}");
    }
    let changed = changed_cases(&expected, &observed);
    assert!(
        stale.is_empty() && changed.is_empty(),
        "G2 capability regressions (new matches cannot offset them): stale={stale:?} outcomes={changed:?}"
    );
}

#[test]
fn a_new_match_cannot_hide_a_previous_match_refusing() {
    let required = BTreeMap::from([("old.mind", "MATCH")]);
    let balanced = BTreeMap::from([("old.mind", "NOT_PRESERVED"), ("new.mind", "MATCH")]);
    assert_eq!(changed_cases(&required, &balanced), vec!["old.mind"]);
    let preserved = BTreeMap::from([("old.mind", "MATCH"), ("new.mind", "MATCH")]);
    assert!(changed_cases(&required, &preserved).is_empty());
    assert_eq!(changed_cases(&required, &BTreeMap::new()), vec!["old.mind"]);
    let rejected = BTreeMap::from([("old.mind", "REJECT_MATCH")]);
    assert_eq!(changed_cases(&required, &rejected), vec!["old.mind"]);
}
