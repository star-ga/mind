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

//! MIND functions are never C library functions: `nobuiltin` marking and the
//! object-level symbol namespace.
//!
//! WHY. The native build hands LLVM IR to `clang -O3 -x ir`. LLVM's library-call
//! recognition keys on the symbol NAME: a function called `pow`, `sin`, `exp`,
//! `log`, `cos`, ... is treated as the C library routine of that name, so its
//! MIND body is ignored at every call site the optimizer can reason about — a
//! literal-argument call is constant-folded to the value the HOST toolchain's
//! folder computes (`pow(-2, 0.5)` became a `sqrt(-2)` libcall, `sin(1)` became
//! host libm's `sin(1)`). The MIND kernel never ran, and different host folders
//! pick different bits, which breaks cross-substrate byte-identity. The callee
//! does not have to be defined in the module: a caller that imports
//! `std.detmath` holds only a DECLARATION of `pow`, and that is enough for the
//! fold.
//!
//! Command-line flags cannot fix this: `-fno-builtin` / `-ffreestanding` are
//! front-end options that do nothing for `-x ir` input. The LLVM-level contract
//! is the `nobuiltin` function attribute: a call whose callee carries it is
//! never recognised as a library builtin (constant folding and simplify-libcalls
//! both check it). The lowering spells the two worlds apart: a MIND function or
//! a runtime-support function is a `func.func` symbol called with `func.call`;
//! an `extern "C"` function is an `llvm.func` declaration called with
//! `llvm.call`. So the marking is applied UNIFORMLY to every `func.func`
//! definition AND declaration (closed world) — never through a list of libm
//! names, which would be a second list that has to agree with LLVM's and
//! silently drifts. An `llvm.func` is a C function by declaration and is left
//! as written.
//!
//! THE NAMESPACE. The marking closes the call direction only. A MIND definition
//! that keeps a C name as its object symbol is still the definition the linker
//! binds every C reference to: the runtime support object's `write` and
//! `memcpy` calls ran a program's `fn write` / `fn memcpy`, and a library call
//! the code generator synthesises binds the same way. In an executable there is
//! no reason for a MIND function to have a C-visible name, so there every
//! `func.func` symbol is renamed to `mind.<name>` ([`namespace_mind_symbols`]).
//! `.` is not a C identifier character: no C library can define or import such
//! a symbol. The names that stay plain are the ones C code binds by name:
//! `main`, a runtime-support export, the `__mind_` runtime namespace, and a
//! name a native object of the build calls. A shared library or an object file
//! keeps plain names: there every function is the surface a host binds by name.
//!
//! WHERE. Both are applied by the build driver to the text it hands to
//! `mlir-opt`, not by the lowering emitter: `--emit-mlir` is an inspection
//! surface that the self-hosted front-end reproduces byte-for-byte, and the
//! attribute and the object symbol are properties of the native build, not of
//! the program's meaning. The IR, mic@3 and the evidence chain keep the source
//! names.
//! Two spellings are emitted together. MLIR 20 copies bare
//! `passthrough = ["nobuiltin"]` onto `llvm.func` and `mlir-translate` emits
//! `attributes #N = { nobuiltin }` on the `define`. MLIR 23 discards a bare
//! attribute whose name is an `llvm.func` property and reads
//! `llvm.passthrough` instead, so the bare spelling alone never reaches the
//! IR: `clang -O3` then rewrites `pow(x, 0.5)` into an unlinked `sqrt` and
//! the shared object fails to load. MLIR 20 ignores the `llvm.` spelling, so
//! the translated LLVM IR is unchanged there. The attribute does not affect
//! code generation of the marked function's own body.

use std::cell::RefCell;
use std::collections::BTreeSet;

/// The MLIR passthrough entry that becomes LLVM's `nobuiltin` fn attribute.
const NOBUILTIN_ENTRY: &str = "\"nobuiltin\"";

/// Both spellings, for a definition that has no attribute dictionary yet.
/// See the module docs for why MLIR 20 and MLIR 23 each read one of them.
const NOBUILTIN_ATTRS: &str = "passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"]";

/// Object-level prefix of a MIND function in an executable.
pub const MIND_SYMBOL_PREFIX: &str = "mind.";

/// The runtime-support namespace. Its names are never C library names.
const RUNTIME_PREFIX: &str = "__mind_";

thread_local! {
    /// `Some(keep)` while this thread compiles the objects of ONE executable:
    /// the `func.func` names that stay plain. `None` keeps every name.
    static NAMESPACE_KEEP: RefCell<Option<BTreeSet<String>>> = const { RefCell::new(None) };
}

/// Scope guard for [`install_namespace`]: the namespace ends with the build.
pub struct NamespaceGuard;

impl Drop for NamespaceGuard {
    fn drop(&mut self) {
        NAMESPACE_KEEP.with(|k| *k.borrow_mut() = None);
    }
}

/// Give every object this thread builds, until the guard drops, the MIND
/// symbol namespace: all `func.func` names but `keep` become `mind.<name>`.
pub fn install_namespace(keep: BTreeSet<String>) -> NamespaceGuard {
    NAMESPACE_KEEP.with(|k| *k.borrow_mut() = Some(keep));
    NamespaceGuard
}

/// Lower `mlir` with `preset` for the native build, mark every MIND function
/// `nobuiltin`, then apply the installed symbol namespace, if any (see the
/// module docs for why the build driver, not the emitter, does this).
pub fn lower_for_native_build(mlir: &str, preset: &str) -> Result<String, String> {
    let marked = crate::eval::mlir_export::apply_lowering(mlir, preset)
        .map(|l| mark_functions_nobuiltin(&l))?;
    Ok(NAMESPACE_KEEP.with(|k| match k.borrow().as_ref() {
        Some(keep) => namespace_mind_symbols(&marked, keep),
        None => marked,
    }))
}

/// The symbol of a `func.func` header line (definition or declaration).
fn func_symbol(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("func.func ")?;
    let rest = rest.strip_prefix("private ").unwrap_or(rest);
    let rest = rest.strip_prefix('@')?;
    rest.find('(').map(|paren| &rest[..paren])
}

fn is_symbol_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$')
}

/// Rename every `func.func` symbol of `mlir` that is not in `keep` to
/// `mind.<name>`, at its header and at every `@name` reference.
///
/// A name in the `__mind_` runtime namespace, a name that already has the
/// prefix, and a quoted symbol stay as written. Text in a string literal or a
/// `//` comment is not a reference. `llvm.func` symbols are C names and are
/// never in the renamed set.
pub fn namespace_mind_symbols(mlir: &str, keep: &BTreeSet<String>) -> String {
    let renamed: BTreeSet<&str> = mlir
        .lines()
        .filter_map(func_symbol)
        .filter(|name| {
            !name.is_empty()
                && name.bytes().all(is_symbol_byte)
                && !keep.contains(*name)
                && !name.starts_with(RUNTIME_PREFIX)
                && !name.starts_with(MIND_SYMBOL_PREFIX)
        })
        .collect();
    if renamed.is_empty() {
        return mlir.to_string();
    }
    let bytes = mlir.as_bytes();
    let mut out = String::with_capacity(mlir.len() + mlir.len() / 16);
    let (mut i, mut copied, mut in_string) = (0, 0, false);
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_string => i += 1,
            b'"' => in_string = !in_string,
            b'/' if !in_string && bytes.get(i + 1) == Some(&b'/') => {
                i += bytes[i..].iter().take_while(|b| **b != b'\n').count();
                continue;
            }
            b'@' if !in_string => {
                let start = i + 1;
                let len = bytes[start..]
                    .iter()
                    .take_while(|b| is_symbol_byte(**b))
                    .count();
                if renamed.contains(&mlir[start..start + len]) {
                    out.push_str(&mlir[copied..start]);
                    out.push_str(MIND_SYMBOL_PREFIX);
                    copied = start;
                }
                i = start + len;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    out.push_str(&mlir[copied..]);
    out
}

/// Mark every `func.func` in `mlir` — definition or declaration — `nobuiltin`.
///
/// A definition is a `func.func @name(...)` header line ending in the `{` that
/// opens its body; a declaration is a `func.func private @x(...) -> T` line.
/// Every other line is left untouched. An existing `attributes {...}`
/// dictionary is extended in place (and an existing `passthrough` list gains
/// the entry), so the function is total over any header the emitter produces.
/// Idempotent.
pub fn mark_functions_nobuiltin(mlir: &str) -> String {
    let mut out = String::with_capacity(mlir.len() + mlir.len() / 16);
    for line in mlir.split_inclusive('\n') {
        let (body, eol) = match line.strip_suffix('\n') {
            Some(b) => (b, "\n"),
            None => (line, ""),
        };
        match mark_header(body) {
            Some(marked) => {
                out.push_str(&marked);
                out.push_str(eol);
            }
            None => out.push_str(line),
        }
    }
    out
}

/// Return the marked header for a `func.func` line, or `None` to keep the line.
fn mark_header(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let declaration = trimmed.starts_with("func.func private @");
    if !declaration && !trimmed.starts_with("func.func @") {
        return None;
    }
    let (head, open) = match line.trim_end().strip_suffix('{') {
        Some(head) => (head.trim_end(), " {"),
        None if declaration => (line.trim_end(), ""),
        None => return None,
    };
    if head.contains(NOBUILTIN_ENTRY) {
        return None;
    }
    if let Some(at) = head.find(" attributes {") {
        let dict_open = at + " attributes {".len();
        let (before, dict) = head.split_at(dict_open);
        let merged = with_nobuiltin_spellings(dict);
        return Some(format!("{before}{merged}{open}"));
    }
    Some(format!("{head} attributes {{{NOBUILTIN_ATTRS}}}{open}"))
}

/// Add both `passthrough` and `llvm.passthrough` nobuiltin entries to an
/// attribute-dictionary body (the text after `attributes {`, including the
/// closing `}`).
fn with_nobuiltin_spellings(dict: &str) -> String {
    let with_bare = insert_list_entry(dict, "passthrough");
    insert_list_entry(&with_bare, "llvm.passthrough")
}

/// Insert [`NOBUILTIN_ENTRY`] at the front of the array attribute `name`,
/// creating the attribute when it is absent. `passthrough` does not match
/// `llvm.passthrough`.
fn insert_list_entry(dict: &str, name: &str) -> String {
    let key = format!("{name} = [");
    let at = find_list(dict, name);
    if let Some(p) = at {
        let list_open = p + key.len();
        let rest = &dict[list_open..];
        let close = rest.find(']').unwrap_or(rest.len());
        if rest[..close].contains(NOBUILTIN_ENTRY) {
            return dict.to_string();
        }
        let sep = if rest.starts_with(']') { "" } else { ", " };
        let (pre, post) = dict.split_at(list_open);
        return format!("{pre}{NOBUILTIN_ENTRY}{sep}{post}");
    }
    if name == "llvm.passthrough" {
        if let Some(p) = find_list(dict, "passthrough") {
            let list_open = p + "passthrough = [".len();
            if let Some(rel) = dict[list_open..].find(']') {
                let insert_at = list_open + rel + 1;
                let (pre, post) = dict.split_at(insert_at);
                return format!("{pre}, {key}{NOBUILTIN_ENTRY}]{post}");
            }
        }
    }
    let sep = if dict.starts_with('}') { "" } else { ", " };
    format!("{key}{NOBUILTIN_ENTRY}]{sep}{dict}")
}

/// Byte offset of `name = [` , ignoring a bare `passthrough` that is the
/// suffix of `llvm.passthrough`.
fn find_list(dict: &str, name: &str) -> Option<usize> {
    let key = format!("{name} = [");
    let mut start = 0;
    while let Some(rel) = dict[start..].find(&key) {
        let p = start + rel;
        if name == "passthrough" && dict[..p].ends_with("llvm.") {
            start = p + key.len();
            continue;
        }
        return Some(p);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{mark_functions_nobuiltin, namespace_mind_symbols};
    use std::collections::BTreeSet;

    #[test]
    fn marks_definitions_and_declarations_and_keeps_everything_else() {
        let src = "module {\n  func.func private @ext(i64) -> i64\n  func.func @sin(%0: f64) -> f64 {\n    return %0 : f64\n  }\n  func.func @main() -> (i64, i64) {\n    llvm.func @x() {\n  }\n}\n";
        let got = mark_functions_nobuiltin(src);
        let want = "module {\n  func.func private @ext(i64) -> i64 attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"]}\n  func.func @sin(%0: f64) -> f64 attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"]} {\n    return %0 : f64\n  }\n  func.func @main() -> (i64, i64) attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"]} {\n    llvm.func @x() {\n  }\n}\n";
        assert_eq!(got, want);
    }

    #[test]
    fn is_idempotent() {
        let once = mark_functions_nobuiltin("  func.func @pow(%0: f64, %1: f64) -> f64 {\n");
        assert_eq!(mark_functions_nobuiltin(&once), once);
    }

    #[test]
    fn merges_into_existing_attribute_dictionaries() {
        assert_eq!(
            mark_functions_nobuiltin(
                "  func.func @f() -> i64 attributes {llvm.emit_c_interface} {"
            ),
            "  func.func @f() -> i64 attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"], llvm.emit_c_interface} {"
        );
        assert_eq!(
            mark_functions_nobuiltin(
                "  func.func @f() attributes {passthrough = [\"noinline\"]} {"
            ),
            "  func.func @f() attributes {passthrough = [\"nobuiltin\", \"noinline\"], llvm.passthrough = [\"nobuiltin\"]} {"
        );
        assert_eq!(
            mark_functions_nobuiltin("  func.func @f() attributes {} {"),
            "  func.func @f() attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"]} {"
        );
        assert_eq!(
            mark_functions_nobuiltin("  func.func @f() attributes {passthrough = []} {"),
            "  func.func @f() attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"]} {"
        );
        assert_eq!(
            mark_functions_nobuiltin(
                "  func.func @f() attributes {llvm.passthrough = [\"noinline\"]} {"
            ),
            "  func.func @f() attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\", \"noinline\"]} {"
        );
    }

    #[test]
    fn marks_a_declaration_once_and_merges_its_dictionary() {
        let decl = "  func.func private @pow(f64, f64) -> f64";
        let once = mark_functions_nobuiltin(decl);
        assert_eq!(
            once,
            "  func.func private @pow(f64, f64) -> f64 attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"]}"
        );
        assert_eq!(mark_functions_nobuiltin(&once), once);
        assert_eq!(
            mark_functions_nobuiltin("  func.func private @f(i64) -> i64 attributes {a = 1}"),
            "  func.func private @f(i64) -> i64 attributes {passthrough = [\"nobuiltin\"], llvm.passthrough = [\"nobuiltin\"], a = 1}"
        );
    }

    fn keep(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn the_namespace_renames_mind_symbols_and_only_those() {
        let src = "module {\n  func.func private @pow(f64, f64) -> f64\n  func.func private @vec_new() -> i64\n  func.func private @__mind_alloc(i64) -> i64\n  func.func @write(%0: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {\n    %1 = func.call @pow(%0, %0) : (f64, f64) -> f64\n    %2 = func.call @vec_new() : () -> i64\n    %3 = func.call @__mind_alloc(%0) : (i64) -> i64\n    return %0 : i64\n  }\n  func.func @main() -> i64 {\n    %0 = func.call @write(%0) : (i64) -> i64\n    return %0 : i64\n  }\n}\n";
        let got = namespace_mind_symbols(src, &keep(&["main", "vec_new"]));
        assert_eq!(got.matches("@mind.pow").count(), 2, "{got}");
        assert_eq!(got.matches("@mind.write").count(), 2, "{got}");
        assert!(got.contains("func.func @main() -> i64 {"), "{got}");
        assert_eq!(got.matches("@vec_new").count(), 2, "{got}");
        assert_eq!(got.matches("@__mind_alloc").count(), 2, "{got}");
        assert!(!got.contains("@pow") && !got.contains("@write"), "{got}");
        assert!(!got.contains("mind.mind."), "{got}");
        // A second pass finds nothing left to rename.
        assert_eq!(
            namespace_mind_symbols(&got, &keep(&["main", "vec_new"])),
            got
        );
    }

    #[test]
    fn the_namespace_leaves_strings_comments_and_c_symbols_alone() {
        let src = "  llvm.func @puts(!llvm.ptr) -> i32\n  // func.func @puts(\n  func.func @f() -> i64 {\n    %0 = llvm.mlir.constant(\"@f \\\" @f\") : i64\n    %1 = llvm.call @puts(%0) : (i64) -> i32 // @f\n    %2 = func.call @f() : () -> i64\n";
        let got = namespace_mind_symbols(src, &keep(&["main"]));
        assert_eq!(got.matches("@mind.f").count(), 2, "{got}");
        assert_eq!(got.matches("@f").count(), 3, "{got}");
        assert_eq!(got.matches("@puts").count(), 3, "{got}");
        // Nothing to rename: the text is returned as written.
        let plain = "  func.func @main() -> i64 {\n";
        assert_eq!(namespace_mind_symbols(plain, &keep(&["main"])), plain);
    }

    #[test]
    fn leaves_text_without_definitions_byte_identical() {
        let src = "module {\n  // func.func @commented() {\n  llvm.func @w(%a: i64) {\n  }\n}";
        assert_eq!(mark_functions_nobuiltin(src), src);
    }
}
