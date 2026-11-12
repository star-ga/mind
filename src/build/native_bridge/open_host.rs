// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Hosted `__mind_open` error sentinel.
//!
//! `docs/runtime-open-errors.md` specifies libc's `-1` for the hosted runtime.
//! The MLIR backend returns that sentinel. The pinned native emitter
//! (`nb_emit_open`) is the raw `open` syscall and returns `-errno`. The
//! compiler binary's own arena probe reads that raw value, so the emitter
//! stays as it is. User images are normalized here, before admission and
//! before the emitter runs, so both see one program and a failed open is `-1`
//! on both backends. A descriptor (`fd >= 0`) is unchanged. Sources that do
//! not call `__mind_open` are left byte-identical.

const NAME: &[u8] = b"__mind_open";
const HOSTED: &[u8] = b"lane_open_hosted";
const HELPER_DEF: &[u8] = b"fn lane_open_hosted(";
const HELPER: &[u8] = b"
fn lane_open_hosted(path: i64) -> i64 {
    let fd: i64 = __mind_open(path);
    if fd < 0 {
        return 0 - 1;
    }
    return fd;
}
";

pub(super) fn host_open_errors(sources: &mut [Vec<u8>]) {
    if sources
        .iter()
        .any(|src| src.windows(HELPER_DEF.len()).any(|w| w == HELPER_DEF))
    {
        return;
    }
    if !sources.iter().any(|src| contains_call(src)) {
        return;
    }
    let mut defined = false;
    for src in sources.iter_mut() {
        if !contains_call(src) {
            continue;
        }
        *src = rewrite_calls(src);
        if !defined {
            src.extend_from_slice(HELPER);
            defined = true;
        }
    }
}

fn contains_call(src: &[u8]) -> bool {
    scan(src).is_some()
}

fn rewrite_calls(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() + 16);
    let mut i = 0;
    while i < src.len() {
        if let Some(next) = scan(&src[i..]) {
            let at = i + next;
            out.extend_from_slice(&src[i..at]);
            out.extend_from_slice(HOSTED);
            i = at + NAME.len();
        } else {
            out.extend_from_slice(&src[i..]);
            break;
        }
    }
    out
}

/// Offset of the next `__mind_open` call in `src`. A preceding identifier
/// character means this is a longer name, not the intrinsic.
fn scan(src: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + NAME.len() <= src.len() {
        if src[i..].starts_with(NAME) && !prev_ident(src, i) && call_paren(src, i + NAME.len()) {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn prev_ident(src: &[u8], i: usize) -> bool {
    i > 0 && (src[i - 1].is_ascii_alphanumeric() || src[i - 1] == b'_')
}

fn call_paren(src: &[u8], mut j: usize) -> bool {
    while j < src.len() && matches!(src[j], b' ' | b'\t' | b'\n' | b'\r') {
        j += 1;
    }
    j < src.len() && src[j] == b'('
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_without_open_is_byte_identical() {
        let src = b"fn main() -> i64 {\n    return 7;\n}".to_vec();
        let mut sources = vec![src.clone()];
        host_open_errors(&mut sources);
        assert_eq!(sources, vec![src]);
    }

    #[test]
    fn a_longer_name_is_not_a_call() {
        let src = b"fn main() -> i64 {\n    return not__mind_open(1);\n}".to_vec();
        let mut sources = vec![src.clone()];
        host_open_errors(&mut sources);
        assert_eq!(sources, vec![src]);
    }

    #[test]
    fn a_failed_open_folds_to_the_hosted_sentinel() {
        let mut sources =
            vec![b"fn main() -> i64 {\n    return __mind_open(p) + 100;\n}\n".to_vec()];
        host_open_errors(&mut sources);
        let text = String::from_utf8(sources[0].clone()).unwrap();
        assert!(text.contains("return lane_open_hosted(p) + 100;"));
        assert!(text.contains("let fd: i64 = __mind_open(path);"));
        assert!(text.contains("if fd < 0 {"));
        assert!(
            text.contains("return 0 - 1;"),
            "a mutant that drops the hosted sentinel must fail here"
        );
        assert_eq!(text.matches("fn lane_open_hosted(").count(), 1);
    }

    #[test]
    fn a_second_pass_does_not_rewrite_the_helper() {
        let mut sources = vec![b"fn main() -> i64 { return __mind_open(0); }\n".to_vec()];
        host_open_errors(&mut sources);
        let once = sources[0].clone();
        host_open_errors(&mut sources);
        assert_eq!(sources[0], once);
        let text = String::from_utf8(once).unwrap();
        assert_eq!(text.matches("__mind_open(").count(), 1);
        assert_eq!(text.matches("lane_open_hosted(").count(), 2);
    }

    #[test]
    fn only_the_file_that_calls_open_changes() {
        let untouched = b"fn other() -> i64 { return 1; }\n".to_vec();
        let mut sources = vec![
            untouched.clone(),
            b"fn main() -> i64 { return __mind_open(p); }\n".to_vec(),
        ];
        host_open_errors(&mut sources);
        assert_eq!(sources[0], untouched);
        let text = String::from_utf8(sources[1].clone()).unwrap();
        assert!(text.contains("lane_open_hosted(p)"));
        assert!(text.contains("return 0 - 1;"));
    }

    #[test]
    fn whitespace_before_the_paren_is_still_a_call() {
        let mut sources = vec![b"fn main() -> i64 { return __mind_open\n(p); }\n".to_vec()];
        host_open_errors(&mut sources);
        let text = String::from_utf8(sources[0].clone()).unwrap();
        assert!(text.contains("lane_open_hosted\n(p)"));
        assert!(text.contains("return 0 - 1;"));
    }
}
