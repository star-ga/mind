// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A dynamic `__mind_alloc` or `__mind_realloc` size in the native-lane user
//! image runs through a guard with the C-runtime result for a non-positive size.
//!
//! The C runtime (`runtime-support/mind_intrinsics.c`) and the evaluator
//! (`src/eval/interp_mem.rs`) both return 0 from `__mind_alloc(n)` when
//! `n <= 0`. The marked native compiler bumps its arena by any size, so the
//! lane admits only a size it can prove positive
//! (`src/ir/native_lane_profile/alloc_size.rs`). Every other `__mind_alloc` in
//! a user function body is respelled as a call to [`GUARD`], which returns 0
//! for `n <= 0` and calls `__mind_alloc(n)` otherwise. Every other
//! `__mind_realloc` is respelled as a call to [`REALLOC_GUARD`]: a positive
//! size calls `__mind_realloc`, and a non-positive size returns 0, except
//! `realloc(NULL, 0)`, which returns a 24-byte block (glibc `malloc(0)` is a
//! live pointer; the evaluator allocates the same block). Each guard's own
//! intrinsic call is a positive literal or is proven by the dominating
//! `n <= 0` test, so the lane admits the image and native and MLIR agree.
//!
//! The respelling keeps every byte offset: each guard name has the length of
//! the callee it replaces, and the definition is appended after the last
//! source, so spans, lines and the file origins of the composed region do not
//! move. An image with no dynamic size is returned unchanged, byte for byte.
//!
//! Fail-closed, per callee: an image that does not parse, is not UTF-8,
//! already names that callee's guard, or has a call whose span does not start
//! with the callee is not respelled for that callee, and admission refuses it
//! as before. Naming one guard does not block the other.

use crate::ast::Node;

/// The guard's name: exactly as long as `__mind_alloc`.
pub(super) const GUARD: &str = "__lane_alloc";
const CALLEE: &str = "__mind_alloc";
const _: () = assert!(GUARD.len() == CALLEE.len());
/// The guard, appended after the last user source. The leading newline ends a
/// final line comment that has no newline of its own.
const DEFINITION: &str = "\nfn __lane_alloc(n: i64) -> i64 {\n    if n <= 0 {\n        return 0;\n    }\n    return __mind_alloc(n);\n}\n";

/// The realloc guard's name: exactly as long as `__mind_realloc`.
pub(super) const REALLOC_GUARD: &str = "__lane_realloc";
const REALLOC_CALLEE: &str = "__mind_realloc";
const _: () = assert!(REALLOC_GUARD.len() == REALLOC_CALLEE.len());
/// Appended with [`DEFINITION`] when a realloc size is routed. `24` is the
/// bounded allocator's minimum block, and the size `__mind_realloc(0, 0)`
/// allocates in the evaluator.
const REALLOC_DEFINITION: &str = "\nfn __lane_realloc(p: i64, n: i64) -> i64 {\n    if n <= 0 {\n        if p == 0 {\n            if n == 0 {\n                return __mind_alloc(24);\n            }\n        }\n        return 0;\n    }\n    return __mind_realloc(p, n);\n}\n";

#[derive(Clone, Copy)]
struct Route {
    callee: &'static str,
    guard: &'static str,
    /// Index of the size operand (`0` for alloc, `1` for realloc).
    size_at: usize,
    definition: &'static str,
}

const ALLOC_ROUTE: Route = Route {
    callee: CALLEE,
    guard: GUARD,
    size_at: 0,
    definition: DEFINITION,
};
const REALLOC_ROUTE: Route = Route {
    callee: REALLOC_CALLEE,
    guard: REALLOC_GUARD,
    size_at: 1,
    definition: REALLOC_DEFINITION,
};

/// `sources`, with every dynamic alloc and realloc size routed through its
/// guard when `lane_v1`; otherwise, or when there is nothing to route, unchanged.
pub(super) fn route_dynamic_sizes(sources: Vec<Vec<u8>>, lane_v1: bool) -> Vec<Vec<u8>> {
    let region = crate::build::native_image::compose_user_region(&sources);
    let text = match std::str::from_utf8(&region) {
        Ok(text) if lane_v1 => text,
        _ => return sources,
    };
    let mut patches: Vec<(usize, &'static str)> = Vec::new();
    let mut definitions = String::new();
    for route in [ALLOC_ROUTE, REALLOC_ROUTE] {
        if !text.contains(route.callee) || text.contains(route.guard) {
            continue;
        }
        let sites = dynamic_sites(text, &route);
        if sites.is_empty() {
            continue;
        }
        patches.extend(sites.into_iter().map(|at| (at, route.guard)));
        definitions.push_str(route.definition);
    }
    if patches.is_empty() {
        return sources;
    }
    let mut out = sources;
    let mut start = 0;
    for source in out.iter_mut() {
        let end = start + source.len();
        for &(at, guard) in patches.iter().filter(|&&(at, _)| at >= start && at < end) {
            source[at - start..at - start + guard.len()].copy_from_slice(guard.as_bytes());
        }
        start = end + 1;
    }
    if let Some(last) = out.last_mut() {
        last.extend_from_slice(definitions.as_bytes());
    }
    out
}

/// Region offsets of `route.callee` calls in user function bodies whose size
/// operand is not a positive literal.
fn dynamic_sites(text: &str, route: &Route) -> Vec<usize> {
    let Ok(module) = crate::parser::parse(text) else {
        return Vec::new();
    };
    let mut sites = Vec::new();
    for item in &module.items {
        if let Node::FnDef(fd, _) = item {
            all(&fd.body, route, &mut sites);
        }
    }
    let bytes = text.as_bytes();
    let spelled =
        |at: usize| bytes.get(at..at + route.callee.len()) == Some(route.callee.as_bytes());
    if !sites.iter().all(|&at| spelled(at)) {
        return Vec::new();
    }
    sites
}

/// Calls inside every construct the lane source walker admits; a call inside
/// any other construct is left as it is, and that construct refuses anyway.
fn all(nodes: &[Node], route: &Route, sites: &mut Vec<usize>) {
    nodes.iter().for_each(|n| collect(n, route, sites));
}

fn collect(node: &Node, route: &Route, sites: &mut Vec<usize>) {
    match node {
        Node::Call { callee, args, span } => {
            if callee == route.callee
                && args.len() == route.size_at + 1
                && !crate::ir::native_lane_profile::alloc_size_is_positive_literal(
                    &args[route.size_at],
                )
            {
                sites.push(span.start());
            }
            all(args, route, sites);
        }
        Node::Paren(a, _)
        | Node::Neg { operand: a, .. }
        | Node::Not { operand: a, .. }
        | Node::BitNot { operand: a, .. }
        | Node::Let { value: a, .. }
        | Node::Assign { value: a, .. }
        | Node::As { expr: a, .. }
        | Node::Return { value: Some(a), .. } => collect(a, route, sites),
        Node::Binary { left, right, .. }
        | Node::Bitwise { left, right, .. }
        | Node::Logical { left, right, .. }
        | Node::IndexAccess {
            receiver: left,
            index: right,
            ..
        } => {
            collect(left, route, sites);
            collect(right, route, sites);
        }
        Node::Block { stmts, .. } => all(stmts, route, sites),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect(cond, route, sites);
            all(then_branch, route, sites);
            if let Some(b) = else_branch {
                all(b, route, sites);
            }
        }
        Node::While { cond, body, .. } => {
            collect(cond, route, sites);
            all(body, route, sites);
        }
        Node::For {
            start, end, body, ..
        } => {
            collect(start, route, sites);
            collect(end, route, sites);
            all(body, route, sites);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(sources: &[&str]) -> Vec<String> {
        let bytes = sources.iter().map(|s| s.as_bytes().to_vec()).collect();
        route_dynamic_sizes(bytes, true)
            .into_iter()
            .map(|s| String::from_utf8(s).unwrap())
            .collect()
    }

    #[test]
    fn a_dynamic_size_is_respelled_in_place_and_the_guard_appended() {
        let src = "fn f(n: i64) -> i64 { return __mind_alloc(n * 8) + __mind_alloc(16); }";
        let out = route(&[src]);
        let want = "fn f(n: i64) -> i64 { return __lane_alloc(n * 8) + __mind_alloc(16); }";
        assert_eq!(out, vec![format!("{want}{DEFINITION}")]);
    }

    #[test]
    fn nested_and_every_admitted_position_is_found() {
        let src = "fn f(n: i64) -> i64 { let a = __mind_alloc(n); \
                   __mind_store_i64(a, __mind_alloc(__mind_alloc(n) - a)); \
                   while n > 0 { if 1 { x = __mind_alloc(0); } } return (__mind_alloc(-1)); }";
        let out = route(&[src]).remove(0);
        assert_eq!(out.matches(GUARD).count(), 6, "{out}");
        assert_eq!(
            out.matches("__mind_alloc(").count(),
            1,
            "only the guard's own call"
        );
    }

    #[test]
    fn literal_sizes_and_other_lanes_leave_the_bytes_unchanged() {
        let src = "fn main() -> i64 { return __mind_alloc(8) + __mind_alloc((4)); }";
        assert_eq!(route(&[src]), vec![src.to_string()]);
        let dynamic = "fn f(n: i64) -> i64 { return __mind_alloc(n); }";
        let unmarked = route_dynamic_sizes(vec![dynamic.as_bytes().to_vec()], false);
        assert_eq!(unmarked, vec![dynamic.as_bytes().to_vec()]);
        let realloc =
            "fn f(p: i64) -> i64 { return __mind_realloc(p, 64) + __mind_realloc(p, (32)); }";
        assert_eq!(route(&[realloc]), vec![realloc.to_string()]);
        let unmarked_realloc = route_dynamic_sizes(vec![realloc.as_bytes().to_vec()], false);
        let dynamic_realloc = "fn f(p: i64, n: i64) -> i64 { return __mind_realloc(p, n); }";
        assert_eq!(
            route_dynamic_sizes(vec![dynamic_realloc.as_bytes().to_vec()], false),
            vec![dynamic_realloc.as_bytes().to_vec()]
        );
        assert_eq!(unmarked_realloc, vec![realloc.as_bytes().to_vec()]);
    }

    #[test]
    fn a_dynamic_realloc_size_is_respelled_and_the_pointer_is_not() {
        // The size is operand 1. A dynamic pointer with a positive literal size
        // stays; a literal pointer with a dynamic size is routed.
        let sized = "fn f(p: i64, n: i64) -> i64 { return __mind_realloc(p, n * 2); }";
        let out = route(&[sized]).remove(0);
        let want = "fn f(p: i64, n: i64) -> i64 { return __lane_realloc(p, n * 2); }";
        assert_eq!(out, format!("{want}{REALLOC_DEFINITION}"));
        let pointer = "fn f(n: i64) -> i64 { return __mind_realloc(n, 16); }";
        assert_eq!(route(&[pointer]), vec![pointer.to_string()]);
        let zero =
            "fn f(p: i64) -> i64 { return __mind_realloc(p, 0) + __mind_realloc(p, 0 - 1); }";
        let z = route(&[zero]).remove(0);
        assert_eq!(z.matches(REALLOC_GUARD).count(), 3, "{z}");
        assert!(
            z.matches("__mind_realloc(").count() == 1,
            "only the guard's own realloc: {z}"
        );
    }

    #[test]
    fn the_census_arithmetic_realloc_is_routed_beside_its_alloc() {
        // The census read_fd_all shape: cap is assigned, then passed as the
        // realloc size.
        let src = "fn read_fd_all(fd: i64, rec: i64) -> i64 { \
                   let mut cap: i64 = 8; let mut buf: i64 = __mind_alloc(cap); \
                   cap = cap * 2; buf = __mind_realloc(buf, cap); return buf; }";
        let out = route(&[src]).remove(0);
        assert!(out.contains("__lane_alloc(cap)"), "{out}");
        assert!(out.contains("__lane_realloc(buf, cap)"), "{out}");
        assert!(out.contains(DEFINITION), "{out}");
        assert!(out.contains(REALLOC_DEFINITION), "{out}");
        assert_eq!(out.matches("__mind_realloc(").count(), 1, "{out}");
    }

    #[test]
    fn an_image_that_names_the_realloc_guard_is_unchanged() {
        let named = "fn __lane_realloc(p: i64, n: i64) -> i64 { return 0; } \
                     fn f(p: i64, n: i64) -> i64 { return __mind_realloc(p, n); }";
        assert_eq!(route(&[named]), vec![named.to_string()]);
        // Naming the alloc guard does not block realloc routing.
        let alloc_named = "fn __lane_alloc(n: i64) -> i64 { return n; } \
                           fn f(p: i64, n: i64) -> i64 { return __mind_realloc(p, n); }";
        let out = route(&[alloc_named]).remove(0);
        assert!(out.contains("__lane_realloc(p, n)"), "{out}");
        assert!(out.contains(REALLOC_DEFINITION), "{out}");
        let broken = "fn f(p: i64, n: i64) -> i64 { return __mind_realloc(p, n); ";
        assert_eq!(route(&[broken]), vec![broken.to_string()]);
        let konst = "const K: i64 = __mind_realloc(0, 0 - 1); fn main() -> i64 { return K; }";
        assert_eq!(route(&[konst]), vec![konst.to_string()]);
    }

    #[test]
    fn realloc_sites_map_to_their_own_source() {
        let a = "fn a(p: i64, n: i64) -> i64 { return __mind_realloc(p, n); }";
        let b = "fn main() -> i64 { return __mind_realloc(0, 8) + a(__mind_realloc(0, 0)); }";
        let out = route(&[a, b]);
        assert_eq!(out[0], a.replace(REALLOC_CALLEE, REALLOC_GUARD));
        let b_want = "fn main() -> i64 { return __mind_realloc(0, 8) + a(__lane_realloc(0, 0)); }";
        assert_eq!(out[1], format!("{b_want}{REALLOC_DEFINITION}"));
    }

    #[test]
    fn an_image_that_cannot_be_respelled_safely_is_unchanged() {
        // Already names the guard.
        let named = "fn __lane_alloc(n: i64) -> i64 { return n; } \
                     fn f(n: i64) -> i64 { return __mind_alloc(n); }";
        assert_eq!(route(&[named]), vec![named.to_string()]);
        // Does not parse.
        let broken = "fn f(n: i64) -> i64 { return __mind_alloc(n); ";
        assert_eq!(route(&[broken]), vec![broken.to_string()]);
        // A const initializer is not a function body.
        let konst = "const K: i64 = __mind_alloc(0 - 1); fn main() -> i64 { return K; }";
        assert_eq!(route(&[konst]), vec![konst.to_string()]);
    }

    #[test]
    fn sites_map_to_their_own_source_and_the_guard_follows_the_last() {
        let a = "fn a(n: i64) -> i64 { return __mind_alloc(n); }";
        let b = "fn main() -> i64 { return __mind_alloc(8) + a(__mind_alloc(0)); }";
        let out = route(&[a, b]);
        assert_eq!(out[0], a.replace(CALLEE, GUARD));
        let b_want = "fn main() -> i64 { return __mind_alloc(8) + a(__lane_alloc(0)); }";
        assert_eq!(out[1], format!("{b_want}{DEFINITION}"));
    }
}
