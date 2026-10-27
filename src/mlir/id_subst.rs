// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Boundary-safe `%<id>` rewriting for the std-surface While / value-`if`
//! lowering: a loop-carried or let-init value id is replaced by its block
//! argument in already-emitted MLIR text. Pure text functions with no
//! lowering state, kept out of `lowering.rs` (module-size ratchet).

/// Replace every occurrence of `%{init_id}` with `%{arg_name}` in `text`
/// for each `(init_id, arg_name, _post_id)` triple.
///
/// Boundary rule: a match is valid only when the character immediately
/// after the digit run is NOT another ASCII digit.  This prevents `%1`
/// from being substituted inside `%10`, `%11`, etc.
///
/// The substitution is applied from largest init_id to smallest so that
/// multi-digit ids (e.g. `%12`) are processed before single-digit ones
/// (e.g. `%1`) and thus can never be partially clobbered.
pub(super) fn substitute_ids(text: &str, args: &[(usize, String, usize)]) -> String {
    // Sort descending by init_id so longer numeric ids are replaced first.
    let mut sorted: Vec<&(usize, String, usize)> = args.iter().collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.0));

    let mut result = text.to_owned();
    for (init_id, arg_name, _) in &sorted {
        result = substitute_one(&result, *init_id, arg_name);
    }
    result
}

/// Return `%{arg_name}` if `id` matches any `init_id` in `args`, otherwise
/// return `%{id}`.
pub(super) fn substitute_single_id(id: usize, args: &[(usize, String, usize)]) -> String {
    for (init_id, arg_name, _) in args {
        if *init_id == id {
            return format!("%{arg_name}");
        }
    }
    format!("%{id}")
}

/// Replace all boundary-safe occurrences of `%{id}` in `text` with
/// `%{replacement}`.  A match is boundary-safe when the character
/// immediately following the digit run is not an ASCII digit.
/// Gated on `std-surface`: its only caller is the std-surface let-init
/// inlining used by the value-`if` branch rewriter.
fn substitute_one(text: &str, id: usize, replacement: &str) -> String {
    let needle = format!("%{id}");
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let nlen = needle.len();
    let mut pos = 0usize;
    while pos < bytes.len() {
        // Fast check: does the slice starting here begin with needle?
        if bytes.len() - pos >= nlen && &bytes[pos..pos + nlen] == needle.as_bytes() {
            // Boundary check: char after the needle must not be an ASCII digit.
            let after = pos + nlen;
            let next_is_digit = after < bytes.len() && bytes[after].is_ascii_digit();
            if !next_is_digit {
                out.push('%');
                out.push_str(replacement);
                pos += nlen;
                continue;
            }
        }
        out.push(bytes[pos] as char);
        pos += 1;
    }
    out
}
