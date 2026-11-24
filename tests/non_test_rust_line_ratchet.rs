// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Non-test Rust under `src/` may not grow.
//!
//! The count is physical lines of tracked `src/**/*.rs`. A line is in the test
//! column when the file is `*_tests.rs`, when the file is reached only through
//! a `#[cfg(test)]` or `#[cfg(all(test, ...))]` module, or when the line sits
//! in such an item inside a product file. Nothing else is excluded: there is
//! no listed exception to apply. The non-test column is a maximum. The test
//! column is recorded so a line cannot move between columns unnoticed.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::process::Command;

/// Measured non-test lines. Growth fails. A shrink stays green.
const NON_TEST_RUST_LINE_MAX: usize = 150_927;
/// Measured test lines, the second column.
const TEST_RUST_LINES: usize = 34_066;

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn tracked_src_rs() -> Vec<String> {
    let out = Command::new("git")
        .current_dir(repo())
        .args(["ls-files", "-z", "--", "src"])
        .output()
        .expect("git ls-files");
    assert!(out.status.success(), "git ls-files failed");
    let mut files: Vec<String> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|p| p.ends_with(b".rs"))
        .map(|p| String::from_utf8(p.to_vec()).expect("utf-8 path"))
        .collect();
    files.sort();
    files
}

fn attr_is_test_only(attr: &str) -> bool {
    let inner = attr.trim();
    let inner = inner
        .strip_prefix("#![")
        .or_else(|| inner.strip_prefix("#["))
        .unwrap_or(inner);
    let inner = inner.strip_suffix(']').unwrap_or(inner).trim();
    if cfg_test_exact(inner) {
        return true;
    }
    let rest = inner
        .strip_prefix("cfg")
        .map(str::trim_start)
        .and_then(|s| s.strip_prefix('('))
        .and_then(|s| s.strip_suffix(')'))
        .map(str::trim);
    let Some(rest) = rest else {
        return false;
    };
    let rest = rest.trim();
    let Some(body) = rest.strip_prefix("all") else {
        return false;
    };
    let body = body.trim();
    let Some(body) = body.strip_prefix('(').and_then(|s| s.strip_suffix(')')) else {
        return false;
    };
    split_top(body).iter().any(|part| part.trim() == "test")
}

fn cfg_test_exact(inner: &str) -> bool {
    let Some(rest) = inner.strip_prefix("cfg") else {
        return false;
    };
    let rest = rest.trim_start();
    let Some(rest) = rest.strip_prefix('(').and_then(|s| s.strip_suffix(')')) else {
        return false;
    };
    rest.trim() == "test" && !inner.contains("all")
}

fn split_top(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut buf = String::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut buf));
                continue;
            }
            _ => {}
        }
        buf.push(c);
    }
    parts.push(buf);
    parts
}

struct Scan {
    chars: Vec<char>,
}

impl Scan {
    fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
        }
    }

    fn skip_ws(&self, mut i: usize) -> usize {
        let n = self.chars.len();
        while i < n {
            if self.starts(i, "//") {
                i = self.find_char(i, '\n').map(|k| k + 1).unwrap_or(n);
            } else if self.starts(i, "/*") {
                i = self.find_str(i + 2, "*/").map(|k| k + 2).unwrap_or(n);
            } else if self.chars[i].is_whitespace() {
                i += 1;
            } else {
                break;
            }
        }
        i
    }

    fn starts(&self, i: usize, s: &str) -> bool {
        let cs: Vec<char> = s.chars().collect();
        i + cs.len() <= self.chars.len() && self.chars[i..i + cs.len()] == cs[..]
    }

    fn find_char(&self, i: usize, c: char) -> Option<usize> {
        self.chars[i..].iter().position(|x| *x == c).map(|k| i + k)
    }

    fn find_str(&self, i: usize, s: &str) -> Option<usize> {
        let cs: Vec<char> = s.chars().collect();
        if cs.is_empty() {
            return Some(i);
        }
        self.chars[i..]
            .windows(cs.len())
            .position(|w| w == cs.as_slice())
            .map(|k| i + k)
    }

    fn slice(&self, a: usize, b: usize) -> String {
        self.chars[a..b].iter().collect()
    }

    fn parse_attr(&self, i: usize) -> Option<(String, usize)> {
        if !(self.starts(i, "#[") || self.starts(i, "#![")) {
            return None;
        }
        let mut j = if self.starts(i, "#![") { i + 3 } else { i + 2 };
        let mut depth = 1i32;
        let n = self.chars.len();
        while j < n && depth > 0 {
            if self.starts(j, "//") {
                j = self.find_char(j, '\n').map(|k| k + 1).unwrap_or(n);
                continue;
            }
            if self.starts(j, "/*") {
                j = self.find_str(j + 2, "*/").map(|k| k + 2).unwrap_or(n);
                continue;
            }
            let c = self.chars[j];
            if c == '"' {
                j = skip_string(&self.chars, j);
                continue;
            }
            if c == '\'' && (j + 1 >= n || self.chars[j + 1] != '\'') {
                j = skip_char_or_life(&self.chars, j);
                continue;
            }
            if c == '[' {
                depth += 1;
            } else if c == ']' {
                depth -= 1;
            }
            j += 1;
        }
        Some((self.slice(i, j), j))
    }

    fn read_word(&self, i: usize) -> (String, usize) {
        let i = self.skip_ws(i);
        let n = self.chars.len();
        if i >= n {
            return (String::new(), i);
        }
        let c = self.chars[i];
        if c.is_ascii_alphabetic() || c == '_' {
            let mut j = i + 1;
            while j < n && (self.chars[j].is_ascii_alphanumeric() || self.chars[j] == '_') {
                j += 1;
            }
            return (self.slice(i, j), j);
        }
        (String::new(), i)
    }

    fn item_kind(&self, i: usize) -> (String, Option<String>, usize) {
        let mut j = i;
        loop {
            let (w, k) = self.read_word(j);
            if w.is_empty() {
                return ("other".to_string(), None, j);
            }
            if w == "pub" {
                j = self.skip_ws(k);
                if j < self.chars.len() && self.chars[j] == '(' {
                    let mut depth = 0i32;
                    while j < self.chars.len() {
                        if self.chars[j] == '(' {
                            depth += 1;
                        } else if self.chars[j] == ')' {
                            depth -= 1;
                            if depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        j += 1;
                    }
                } else {
                    j = k;
                }
                continue;
            }
            if w == "crate" || w == "unsafe" || w == "async" || w == "extern" {
                j = k;
                if w == "extern" {
                    j = self.skip_ws(j);
                    if j < self.chars.len() && self.chars[j] == '"' {
                        j = skip_string(&self.chars, j);
                    }
                }
                continue;
            }
            if w == "const" {
                let (nxt, _) = self.read_word(k);
                if nxt == "fn" {
                    j = k;
                    continue;
                }
                return ("const".to_string(), None, k);
            }
            if w == "mod" {
                let (nm, _) = self.read_word(k);
                return ("mod".to_string(), Some(nm), k);
            }
            const KEPT: &[&str] = &[
                "fn",
                "struct",
                "enum",
                "impl",
                "trait",
                "type",
                "use",
                "const",
                "static",
                "union",
                "macro_rules",
                "thread_local",
                "if",
                "match",
                "while",
                "for",
                "loop",
                "let",
                "return",
                "break",
                "continue",
            ];
            if KEPT.contains(&w.as_str()) {
                return (w, None, k);
            }
            return ("field".to_string(), None, j);
        }
    }

    fn consume(&self, mut i: usize, kind: &str) -> usize {
        let n = self.chars.len();
        let mut paren = 0i32;
        let mut brack = 0i32;
        let mut brace = 0i32;
        let mut angle = 0i32;
        let mut started = false;
        while i < n {
            if self.starts(i, "//") {
                i = self.find_char(i, '\n').map(|k| k + 1).unwrap_or(n);
                continue;
            }
            if self.starts(i, "/*") {
                i = self.find_str(i + 2, "*/").map(|k| k + 2).unwrap_or(n);
                continue;
            }
            let c = self.chars[i];
            if c == '"' {
                i = skip_string(&self.chars, i);
                continue;
            }
            if c == '\'' && (i + 1 >= n || self.chars[i + 1] != '\'') {
                i = skip_char_or_life(&self.chars, i);
                continue;
            }
            match c {
                '(' => paren += 1,
                ')' if paren > 0 => paren -= 1,
                '[' => brack += 1,
                ']' if brack > 0 => brack -= 1,
                '<' if paren == 0 && brack == 0 && brace == 0 => angle += 1,
                '>' if angle > 0 && paren == 0 && brack == 0 && brace == 0 => angle -= 1,
                '{' if paren == 0 && brack == 0 && angle == 0 => {
                    brace += 1;
                    started = true;
                }
                '}' if brace > 0 && paren == 0 && brack == 0 => {
                    brace -= 1;
                    if brace == 0 && started {
                        i += 1;
                        let j = self.skip_ws(i);
                        if j < n && self.chars[j] == ';' {
                            return j + 1;
                        }
                        return i;
                    }
                }
                ';' if paren == 0 && brack == 0 && brace == 0 && angle == 0 => return i + 1,
                ',' if kind == "field" && paren == 0 && brack == 0 && brace == 0 && angle == 0 => {
                    return i + 1;
                }
                _ => {}
            }
            i += 1;
        }
        i
    }

    fn test_spans_of(text: &str) -> Vec<(usize, usize)> {
        let scan = Scan::new(text);
        let n = scan.chars.len();
        let mut spans = Vec::new();
        let mut i = 0;
        while i < n {
            i = scan.skip_ws(i);
            if i >= n {
                break;
            }
            if scan.starts(i, "#[") || scan.starts(i, "#![") {
                let start = i;
                let mut attrs = Vec::new();
                loop {
                    let Some((attr, next)) = scan.parse_attr(i) else {
                        break;
                    };
                    attrs.push(attr);
                    i = scan.skip_ws(next);
                    if !(scan.starts(i, "#[") || scan.starts(i, "#![")) {
                        break;
                    }
                }
                if attrs.is_empty() {
                    i += 1;
                    continue;
                }
                let test_only = attrs.iter().any(|a| attr_is_test_only(a));
                let (kind, _, _) = scan.item_kind(i);
                let end = scan.consume(i, &kind);
                if test_only {
                    spans.push((start, end));
                }
                i = end;
                continue;
            }
            i += 1;
        }
        spans
    }
}

fn skip_string(chars: &[char], i: usize) -> usize {
    let n = chars.len();
    let mut hashes = 0usize;
    let mut j = i;
    if chars[i] == 'r' {
        j += 1;
        while j < n && chars[j] == '#' {
            hashes += 1;
            j += 1;
        }
        if j < n && chars[j] == '"' {
            j += 1;
            let end: Vec<char> = std::iter::once('"')
                .chain(std::iter::repeat('#').take(hashes))
                .collect();
            while j + end.len() <= n && &chars[j..j + end.len()] != end.as_slice() {
                j += 1;
            }
            return (j + end.len()).min(n);
        }
        return i + 1;
    }
    if chars[i] != '"' {
        return i + 1;
    }
    let mut j = i + 1;
    while j < n {
        if chars[j] == '\\' {
            j += 2;
            continue;
        }
        if chars[j] == '"' {
            return j + 1;
        }
        j += 1;
    }
    n
}

fn skip_char_or_life(chars: &[char], i: usize) -> usize {
    let n = chars.len();
    if i + 1 < n && (chars[i + 1].is_ascii_alphabetic() || chars[i + 1] == '_') {
        let mut j = i + 2;
        while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
            j += 1;
        }
        return j;
    }
    let mut j = i + 1;
    if j < n && chars[j] == '\\' {
        j += 2;
    } else {
        j += 1;
    }
    if j < n && chars[j] == '\'' {
        j += 1;
    }
    j
}

fn find_mods(text: &str) -> Vec<(bool, Option<String>, String)> {
    let scan = Scan::new(text);
    let n = scan.chars.len();
    let mut found = Vec::new();
    let mut i = 0;
    let mut pending: Vec<String> = Vec::new();
    while i < n {
        i = scan.skip_ws(i);
        if i >= n {
            break;
        }
        if scan.starts(i, "#[") || scan.starts(i, "#![") {
            if let Some((attr, next)) = scan.parse_attr(i) {
                pending.push(attr);
                i = next;
                continue;
            }
            i += 1;
            continue;
        }
        let (kind, mod_name, after) = scan.item_kind(i);
        if kind == "mod" {
            if let Some(name) = mod_name {
                let test_only = pending.iter().any(|a| attr_is_test_only(a));
                let path = pending.iter().find_map(|a| path_attr(a));
                found.push((test_only, path, name));
                i = scan.consume(i, "mod");
                pending.clear();
                continue;
            }
        }
        pending.clear();
        i = (i + 1).max(after);
    }
    found
}

fn path_attr(attr: &str) -> Option<String> {
    let key = "path";
    let bytes = attr;
    let pos = bytes.find(key)?;
    let rest = bytes[pos + key.len()..].trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn module_file(
    src: &str,
    name: &str,
    path_attr: Option<&str>,
    have: &HashSet<String>,
) -> Option<String> {
    let parent = Path::new(src).parent().unwrap_or(Path::new(""));
    if let Some(rel) = path_attr {
        let target = parent.join(rel);
        let norm = normalize(&target);
        return have.contains(&norm).then_some(norm);
    }
    let base = if Path::new(src).file_name().and_then(|s| s.to_str()) == Some("mod.rs") {
        parent.to_path_buf()
    } else {
        parent.join(Path::new(src).file_stem().unwrap_or_default())
    };
    for cand in [
        base.join(format!("{name}.rs")),
        base.join(name).join("mod.rs"),
    ] {
        let norm = normalize(&cand);
        if have.contains(&norm) {
            return Some(norm);
        }
    }
    None
}

fn normalize(path: &Path) -> String {
    let mut out: Vec<&str> = Vec::new();
    for c in path.components() {
        match c {
            std::path::Component::Normal(s) => out.push(s.to_str().unwrap_or("")),
            std::path::Component::ParentDir => {
                out.pop();
            }
            _ => {}
        }
    }
    out.join("/")
}

fn marked_lines(text: &str, spans: &[(usize, usize)]) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let total = text.lines().count();
    let mut marked = BTreeSet::new();
    for &(a, b) in spans {
        if a >= b || a >= chars.len() {
            continue;
        }
        let last = (b - 1).min(chars.len() - 1);
        let upto_a = chars[..a].iter().filter(|c| **c == '\n').count();
        let upto_b = chars[..=last].iter().filter(|c| **c == '\n').count();
        for n in upto_a..=upto_b {
            if n < total {
                marked.insert(n);
            }
        }
    }
    marked.len()
}

/// `(non_test, test)` physical lines.
fn count_src_lines() -> (usize, usize) {
    let files = tracked_src_rs();
    let have: HashSet<String> = files.iter().cloned().collect();
    let texts: HashMap<String, String> = files
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(repo().join(p)).unwrap_or_default();
            (p.clone(), text)
        })
        .collect();
    let mut edges: Vec<(String, String, bool)> = Vec::new();
    for p in &files {
        for (test_only, path, name) in find_mods(&texts[p]) {
            if let Some(dst) = module_file(p, &name, path.as_deref(), &have) {
                edges.push((p.clone(), dst, test_only));
            }
        }
    }
    let forced: HashSet<String> = files
        .iter()
        .filter(|p| {
            Path::new(p)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .ends_with("_tests.rs")
        })
        .cloned()
        .collect();
    let mut wholly: HashSet<String> = edges
        .iter()
        .filter(|(_, _, test_only)| *test_only)
        .map(|(_, dst, _)| dst.clone())
        .collect();
    wholly.extend(forced.iter().cloned());
    let mut grew = true;
    while grew {
        grew = false;
        for (src, dst, _) in &edges {
            if wholly.contains(src) && wholly.insert(dst.clone()) {
                grew = true;
            }
        }
    }
    let product_edges: Vec<(String, String)> = edges
        .iter()
        .filter(|(src, _, test_only)| !test_only && !wholly.contains(src))
        .map(|(src, dst, _)| (src.clone(), dst.clone()))
        .collect();
    let mut rescued = HashSet::new();
    grew = true;
    while grew {
        grew = false;
        for (src, dst) in &product_edges {
            if wholly.contains(dst)
                && !wholly.contains(src)
                && !forced.contains(dst)
                && rescued.insert(dst.clone())
            {
                wholly.remove(dst);
                grew = true;
            }
        }
    }
    let mut non_test = 0usize;
    let mut test = 0usize;
    for p in &files {
        let text = &texts[p];
        let total = text.lines().count();
        if wholly.contains(p) {
            test += total;
            continue;
        }
        let spans = Scan::test_spans_of(text);
        let t = marked_lines(text, &spans);
        test += t;
        non_test += unmarked_lines(total, t);
    }
    (non_test, test)
}

fn non_test_exceeds(n: usize) -> bool {
    n > NON_TEST_RUST_LINE_MAX
}

/// Unmarked lines of one product file. Each physical line is in one column.
fn unmarked_lines(total: usize, marked: usize) -> usize {
    total - marked
}

fn physical_src_lines() -> usize {
    tracked_src_rs()
        .iter()
        .map(|p| {
            std::fs::read_to_string(repo().join(p))
                .unwrap_or_default()
                .lines()
                .count()
        })
        .sum()
}

#[test]
fn non_test_rust_lines_do_not_grow() {
    let (non_test, test_lines) = count_src_lines();
    assert!(
        !non_test_exceeds(non_test),
        "non-test Rust under src/ grew to {non_test} (maximum {NON_TEST_RUST_LINE_MAX}). \
         Delete or shrink host code; do not raise the maximum to hide the growth."
    );
    let physical = physical_src_lines();
    assert_eq!(
        non_test + test_lines,
        physical,
        "the two columns dropped a line (non-test {non_test}, test {test_lines}, \
         physical {physical}). Together they must equal every physical line of tracked \
         src. Shrinking the tree shrinks both sides; dropping a line only in the count does not."
    );
    assert_eq!(
        test_lines, TEST_RUST_LINES,
        "the test column moved (non-test is {non_test}). Re-pin TEST_RUST_LINES in the same commit \
         when test lines are added or removed; do not quiet a non-test increase by moving it here."
    );
}

#[test]
fn one_added_non_test_line_is_seen() {
    assert!(
        non_test_exceeds(NON_TEST_RUST_LINE_MAX + 1),
        "one added non-test line must fail the maximum"
    );
    assert!(!non_test_exceeds(NON_TEST_RUST_LINE_MAX));
    assert!(
        !non_test_exceeds(NON_TEST_RUST_LINE_MAX - 1),
        "shrinking non-test Rust must stay green"
    );
}

#[test]
fn a_dropped_line_per_file_is_not_a_shrink() {
    assert_eq!(unmarked_lines(10, 3), 7);
    assert_eq!(unmarked_lines(10, 3) + 3, 10);
    assert_eq!(unmarked_lines(4, 4), 0);
    assert_ne!(unmarked_lines(10, 3), 10usize.saturating_sub(3 + 1));
}
