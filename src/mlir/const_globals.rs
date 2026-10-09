// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Deterministic placement of dense tensor constants for the native build.
//!
//! WHY. A module-level `const T: [i64; N]` table lowers to an
//! `arith.constant dense<...> : tensor<Nxi64>` inside every function that
//! reads it. On the `arith-linalg` preset, `one-shot-bufferize` with
//! `bufferize-function-boundaries=true` turns each such constant into a
//! private `memref.global` (via `bufferization::getGlobalFor`), creating it the
//! first time a function that uses it is bufferized and moving it to the front
//! of the module. Module bufferization visits functions in the order
//! `getFuncOpsOrderedByCalls` returns, and that order is read out of a
//! `DenseMap<func::FuncOp, unsigned>` keyed by operation POINTER (an
//! `llvm::find_if` over the map picks the next function with no pending
//! callees). Heap addresses move with ASLR, so when the constants live in two
//! or more functions the global ORDER — and, for same-shape constants, their
//! `_0`/`_1` NAMES — changed from run to run. Measured on LLVM 18 and 20: twelve
//! one-accessor tables built to four different `.so` hashes of one size;
//! `--mlir-disable-threading` did not help, `setarch -R` (ASLR off) did.
//!
//! FIX. Before `mlir-opt` runs, every distinct dense tensor constant is declared
//! as a `memref.global` in first-use order. `getGlobalFor` first scans the
//! module for an existing global with the same initial value and alignment and
//! reuses it, so bufferization creates nothing and the global order is the one
//! written here. The declarations reproduce exactly what bufferization would
//! have created had it visited the functions in source order — the same
//! `__constant_<shape>` names with the same `_<k>` disambiguation, the same
//! `{alignment = 64 : i64}` (the One-Shot default buffer alignment), the same
//! reverse-insertion order at the module head.
//!
//! SCOPE. The text is changed only when dense tensor constants occur in TWO OR
//! MORE functions — the only case with a pointer-ordered choice. A program whose
//! constants all sit in one function was already deterministic and is returned
//! byte-identical, so its artifact does not move. Like the `nobuiltin` marking,
//! this is a property of the native build, not of the program: `--emit-mlir`
//! never shows it.

use std::collections::HashSet;

/// The One-Shot Bufferize default `bufferAlignment`; `getGlobalFor` reuses a
/// pre-declared global only when its alignment matches this exactly.
const BUFFER_ALIGNMENT: u64 = 64;

/// One distinct dense tensor constant, in first-use order.
struct DenseConst<'a> {
    /// `dense<...>` exactly as emitted (the global's initializer).
    attr: &'a str,
    /// The tensor body between `tensor<` and `>`, e.g. `40xi64`.
    shape: &'a str,
}

/// Declare every dense tensor constant of `mlir` as an ordered `memref.global`
/// when two or more distinct constants occur across two or more functions;
/// otherwise return `mlir` unchanged. Idempotent: running it on its own output
/// returns that output.
pub fn predeclare_dense_constant_globals(mlir: &str) -> String {
    let (consts, funcs) = collect(mlir);
    if consts.len() < 2 || funcs < 2 {
        return mlir.to_string();
    }
    if consts.iter().all(|c| declared(mlir, c)) {
        return mlir.to_string();
    }
    let mut taken = existing_constant_symbols(mlir);
    let mut decls: Vec<String> = Vec::with_capacity(consts.len());
    for c in &consts {
        let name = unique_name(&symbol_base(c.shape), &taken);
        decls.push(format!(
            "  memref.global \"private\" constant @{name} : memref<{}> = {} {{alignment = {BUFFER_ALIGNMENT} : i64}}",
            c.shape, c.attr
        ));
        taken.insert(name);
    }
    // Bufferization moves each new global to the module FRONT, so the global it
    // creates first ends up last: emit in reverse first-use order.
    decls.reverse();
    insert_after_module_header(mlir, &decls.join("\n"))
}

/// Distinct static-shape dense tensor constants of `mlir` in first-use order,
/// and the number of distinct function definitions that contain one.
fn collect(mlir: &str) -> (Vec<DenseConst<'_>>, usize) {
    let mut consts: Vec<DenseConst<'_>> = Vec::new();
    let mut func = 0usize;
    let mut funcs: Vec<usize> = Vec::new();
    for line in mlir.lines() {
        let t = line.trim();
        if t.starts_with("func.func @") && t.ends_with('{') {
            func += 1;
            continue;
        }
        let Some((attr, shape)) = dense_tensor_constant(t) else {
            continue;
        };
        if !funcs.contains(&func) {
            funcs.push(func);
        }
        if !consts.iter().any(|c| c.attr == attr && c.shape == shape) {
            consts.push(DenseConst { attr, shape });
        }
    }
    (consts, funcs.len())
}

/// `(attr, shape)` of a `%x = arith.constant dense<...> : tensor<shape>` line
/// with a static shape and no encoding; `None` for anything else (vectors,
/// scalars, dynamic or encoded tensors are left to the pipeline untouched).
fn dense_tensor_constant(line: &str) -> Option<(&str, &str)> {
    let at = line.find("arith.constant dense<")?;
    let rest = &line[at + "arith.constant ".len()..];
    let split = rest.rfind(" : tensor<")?;
    let attr = &rest[..split];
    let shape = rest[split + " : tensor<".len()..].strip_suffix('>')?;
    let static_shape = !shape.is_empty()
        && shape
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
    static_shape.then_some((attr, shape))
}

/// The name `getGlobalFor` gives a constant of this shape: `__constant_` +
/// the dims joined by `x` + `x` + the element type (`tensor<i64>` -> `xi64`).
fn symbol_base(shape: &str) -> String {
    let mut dims: Vec<&str> = Vec::new();
    let mut rest = shape;
    while let Some((dim, tail)) = rest.split_once('x') {
        if dim.is_empty() || !dim.bytes().all(|b| b.is_ascii_digit()) {
            break;
        }
        dims.push(dim);
        rest = tail;
    }
    format!("__constant_{}x{rest}", dims.join("x"))
}

/// True iff `mlir` already declares `c` as a global `getGlobalFor` would reuse.
fn declared(mlir: &str, c: &DenseConst<'_>) -> bool {
    let tail = format!(
        ": memref<{}> = {} {{alignment = {BUFFER_ALIGNMENT} : i64}}",
        c.shape, c.attr
    );
    mlir.lines()
        .any(|l| l.trim_start().starts_with("memref.global") && l.trim_end().ends_with(&tail))
}

/// Every `@__constant_*` symbol already named in `mlir`, so a declaration never
/// shadows or collides with one.
fn existing_constant_symbols(mlir: &str) -> HashSet<String> {
    const PREFIX: &str = "@__constant_";
    mlir.match_indices(PREFIX)
        .map(|(i, _)| {
            let rest = &mlir[i + 1..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '$' || c == '.'))
                .unwrap_or(rest.len());
            rest[..end].to_string()
        })
        .collect()
}

/// `base`, or `base_<k>` for the smallest free `k` — the `SymbolTable::insert`
/// renaming bufferization applies to a clashing name.
fn unique_name(base: &str, taken: &HashSet<String>) -> String {
    if !taken.contains(base) {
        return base.to_string();
    }
    (0usize..)
        .map(|k| format!("{base}_{k}"))
        .find(|n| !taken.contains(n))
        .unwrap_or_else(|| base.to_string())
}

/// Insert `block` on the line after the first `module ... {` header, or at the
/// top when the text has no explicit module (mlir-opt wraps it implicitly).
fn insert_after_module_header(mlir: &str, block: &str) -> String {
    let mut out = String::with_capacity(mlir.len() + block.len() + 2);
    let mut done = false;
    for line in mlir.split_inclusive('\n') {
        out.push_str(line);
        let t = line.trim();
        if !done && t.starts_with("module") && t.ends_with('{') {
            if !line.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(block);
            out.push('\n');
            done = true;
        }
    }
    if done {
        out
    } else {
        format!("{block}\n{mlir}")
    }
}

#[cfg(test)]
mod tests {
    use super::predeclare_dense_constant_globals as pre;

    const TWO_FUNCS: &str = "module {\n  func.func @a(%0: i64) -> i64 {\n  %1 = arith.constant dense<[1, 2]> : tensor<2xi64>\n    return %0 : i64\n  }\n  func.func @b(%0: i64) -> i64 {\n  %1 = arith.constant dense<[3, 4, 5]> : tensor<3xi64>\n  %2 = arith.constant dense<[6, 7]> : tensor<2xi64>\n  %3 = arith.constant dense<[1, 2]> : tensor<2xi64>\n    return %0 : i64\n  }\n}\n";

    #[test]
    fn declares_globals_in_reverse_first_use_order_with_bufferization_names() {
        let got = pre(TWO_FUNCS);
        let head: Vec<&str> = got.lines().skip(1).take(3).collect();
        assert_eq!(
            head,
            [
                "  memref.global \"private\" constant @__constant_2xi64_0 : memref<2xi64> = dense<[6, 7]> {alignment = 64 : i64}",
                "  memref.global \"private\" constant @__constant_3xi64 : memref<3xi64> = dense<[3, 4, 5]> {alignment = 64 : i64}",
                "  memref.global \"private\" constant @__constant_2xi64 : memref<2xi64> = dense<[1, 2]> {alignment = 64 : i64}",
            ]
        );
        // Everything after the declarations is the input, byte for byte.
        let rest: String = got.split_inclusive('\n').skip(4).collect();
        let input_rest: String = TWO_FUNCS.split_inclusive('\n').skip(1).collect();
        assert_eq!(rest, input_rest);
    }

    #[test]
    fn is_idempotent() {
        let once = pre(TWO_FUNCS);
        assert_ne!(once, TWO_FUNCS);
        assert_eq!(pre(&once), once);
    }

    #[test]
    fn single_function_constants_are_left_byte_identical() {
        let one = "module {\n  func.func @a() -> i64 {\n  %1 = arith.constant dense<[1, 2]> : tensor<2xi64>\n  %2 = arith.constant dense<[3]> : tensor<1xi64>\n  }\n  func.func @b() -> i64 {\n  %c = arith.constant 0 : i64\n  }\n}\n";
        assert_eq!(pre(one), one);
    }

    #[test]
    fn one_shared_constant_has_no_order_to_fix() {
        let shared = "module {\n  func.func @a() {\n  %1 = arith.constant dense<[1, 2]> : tensor<2xi64>\n  }\n  func.func @b() {\n  %1 = arith.constant dense<[1, 2]> : tensor<2xi64>\n  }\n}\n";
        assert_eq!(pre(shared), shared);
    }

    #[test]
    fn ignores_vectors_scalars_dynamic_and_encoded_tensors() {
        let other = "module {\n  func.func @a() {\n  %1 = arith.constant dense<0> : vector<4xi64>\n  %2 = arith.constant 7 : i64\n  }\n  func.func @b() {\n  %1 = arith.constant dense<1> : tensor<4xi64, #enc>\n  %2 = arith.constant dense<0> : vector<8xi32>\n  }\n}\n";
        assert_eq!(pre(other), other);
    }

    #[test]
    fn avoids_existing_symbols_and_names_rank0_like_bufferization() {
        let src = "module {\n  func.func private @__constant_2xi64()\n  func.func @a() {\n  %1 = arith.constant dense<[1, 2]> : tensor<2xi64>\n  }\n  func.func @b() {\n  %1 = arith.constant dense<5> : tensor<i64>\n  }\n}\n";
        let got = pre(src);
        assert!(
            got.contains("@__constant_2xi64_0 : memref<2xi64> = dense<[1, 2]>"),
            "{got}"
        );
        assert!(
            got.contains("@__constant_xi64 : memref<i64> = dense<5>"),
            "{got}"
        );
    }

    #[test]
    fn prepends_when_there_is_no_module_header() {
        let bare = "func.func @a() {\n  %1 = arith.constant dense<[1]> : tensor<1xi64>\n}\nfunc.func @b() {\n  %1 = arith.constant dense<[2]> : tensor<1xi64>\n}\n";
        let got = pre(bare);
        assert!(
            got.starts_with("  memref.global \"private\" constant @__constant_1xi64_0"),
            "{got}"
        );
        assert!(got.ends_with(bare));
    }
}
