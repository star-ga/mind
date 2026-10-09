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

//! Closed-world `nobuiltin` marking of MIND function definitions.
//!
//! WHY. The native build hands LLVM IR to `clang -O3 -x ir`. LLVM's library-call
//! recognition keys on the symbol NAME: a definition called `pow`, `sin`, `exp`,
//! `log`, `cos`, ... is treated as the C library routine of that name, so its
//! MIND body is ignored at every call site the optimizer can reason about — a
//! literal-argument call is constant-folded to the value the HOST toolchain's
//! folder computes (`pow(-2, 0.5)` became a `sqrt(-2)` libcall, `sin(1)` became
//! host libm's `sin(1)`). The MIND kernel never ran, and different host folders
//! pick different bits, which breaks cross-substrate byte-identity.
//!
//! Command-line flags cannot fix this: `-fno-builtin` / `-ffreestanding` are
//! front-end options that do nothing for `-x ir` input. The LLVM-level contract
//! is the `nobuiltin` function attribute: a call whose callee carries it is
//! never recognised as a library builtin (constant folding and simplify-libcalls
//! both check it). Every MIND function is a user definition that must mean
//! exactly its body, so the marking is applied UNIFORMLY to every `func.func`
//! definition (closed world) — never through a list of libm names, which would
//! be a second list that has to agree with LLVM's and silently drifts.
//!
//! WHERE. The marking is applied by the build driver to the text it hands to
//! `mlir-opt`, not by the lowering emitter: `--emit-mlir` is an inspection
//! surface that the self-hosted front-end reproduces byte-for-byte, and the
//! attribute is a property of the native build, not of the program's meaning.
//! `passthrough = ["nobuiltin"]` survives `convert-func-to-llvm` (onto the
//! `llvm.func`) and `mlir-translate` (as an `attributes #N = { nobuiltin }`
//! group on the `define`), and does not affect code generation of the marked
//! function's own body.

/// The MLIR passthrough entry that becomes LLVM's `nobuiltin` fn attribute.
const NOBUILTIN_ENTRY: &str = "\"nobuiltin\"";

/// Lower `mlir` with `preset` for the native build, then mark every MIND
/// definition `nobuiltin` (see the module docs for why the build driver, not
/// the emitter, applies it).
pub fn lower_for_native_build(mlir: &str, preset: &str) -> Result<String, String> {
    crate::eval::mlir_export::apply_lowering(mlir, preset).map(|l| mark_definitions_nobuiltin(&l))
}

/// Mark every `func.func` DEFINITION in `mlir` as `nobuiltin`.
///
/// A definition is a `func.func @name(...)` header line ending in the `{` that
/// opens its body; declarations (`func.func private @x(...) -> T`) are left
/// untouched, as is every other line. An existing `attributes {...}` dictionary
/// is extended in place (and an existing `passthrough` list gains the entry),
/// so the function is total over any header the emitter produces. Idempotent.
pub fn mark_definitions_nobuiltin(mlir: &str) -> String {
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

/// Return the marked header for a definition line, or `None` to keep the line.
fn mark_header(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("func.func @") {
        return None;
    }
    let head = line.trim_end().strip_suffix('{')?.trim_end();
    if head.contains(NOBUILTIN_ENTRY) {
        return None;
    }
    if let Some(at) = head.find(" attributes {") {
        let dict_open = at + " attributes {".len();
        let (before, dict) = head.split_at(dict_open);
        let merged = if let Some(p) = dict.find("passthrough = [") {
            let list_open = p + "passthrough = [".len();
            let (pre, rest) = dict.split_at(list_open);
            let sep = if rest.starts_with(']') { "" } else { ", " };
            format!("{pre}{NOBUILTIN_ENTRY}{sep}{rest}")
        } else {
            let sep = if dict.starts_with('}') { "" } else { ", " };
            format!("passthrough = [{NOBUILTIN_ENTRY}]{sep}{dict}")
        };
        return Some(format!("{before}{merged} {{"));
    }
    Some(format!(
        "{head} attributes {{passthrough = [{NOBUILTIN_ENTRY}]}} {{"
    ))
}

#[cfg(test)]
mod tests {
    use super::mark_definitions_nobuiltin;

    #[test]
    fn marks_definitions_and_keeps_everything_else() {
        let src = "module {\n  func.func private @ext(i64) -> i64\n  func.func @sin(%0: f64) -> f64 {\n    return %0 : f64\n  }\n  func.func @main() -> (i64, i64) {\n    llvm.func @x() {\n  }\n}\n";
        let got = mark_definitions_nobuiltin(src);
        let want = "module {\n  func.func private @ext(i64) -> i64\n  func.func @sin(%0: f64) -> f64 attributes {passthrough = [\"nobuiltin\"]} {\n    return %0 : f64\n  }\n  func.func @main() -> (i64, i64) attributes {passthrough = [\"nobuiltin\"]} {\n    llvm.func @x() {\n  }\n}\n";
        assert_eq!(got, want);
    }

    #[test]
    fn is_idempotent() {
        let once = mark_definitions_nobuiltin("  func.func @pow(%0: f64, %1: f64) -> f64 {\n");
        assert_eq!(mark_definitions_nobuiltin(&once), once);
    }

    #[test]
    fn merges_into_existing_attribute_dictionaries() {
        assert_eq!(
            mark_definitions_nobuiltin(
                "  func.func @f() -> i64 attributes {llvm.emit_c_interface} {"
            ),
            "  func.func @f() -> i64 attributes {passthrough = [\"nobuiltin\"], llvm.emit_c_interface} {"
        );
        assert_eq!(
            mark_definitions_nobuiltin(
                "  func.func @f() attributes {passthrough = [\"noinline\"]} {"
            ),
            "  func.func @f() attributes {passthrough = [\"nobuiltin\", \"noinline\"]} {"
        );
        assert_eq!(
            mark_definitions_nobuiltin("  func.func @f() attributes {} {"),
            "  func.func @f() attributes {passthrough = [\"nobuiltin\"]} {"
        );
        assert_eq!(
            mark_definitions_nobuiltin("  func.func @f() attributes {passthrough = []} {"),
            "  func.func @f() attributes {passthrough = [\"nobuiltin\"]} {"
        );
    }

    #[test]
    fn leaves_text_without_definitions_byte_identical() {
        let src = "module {\n  // func.func @commented() {\n  llvm.func @w(%a: i64) {\n  }\n}";
        assert_eq!(mark_definitions_nobuiltin(src), src);
    }
}
