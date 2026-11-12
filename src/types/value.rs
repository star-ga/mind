// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use super::TensorType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    ScalarI32,
    ScalarI64,
    ScalarF32,
    ScalarF64,
    ScalarBool,
    Tensor(TensorType),
    GradMap(Vec<(String, TensorType)>),
}

impl ValueType {
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            ValueType::ScalarI32
                | ValueType::ScalarI64
                | ValueType::ScalarF32
                | ValueType::ScalarF64
                | ValueType::ScalarBool
        )
    }
}

/// Primitive scalar names in spec order. A primitive's type code is its
/// position in this text (`docs/specs/v1.0.md` §2.1).
pub const PRIM_SCALAR_NAMES: &str = "i8 i16 i32 i64 u8 u16 u32 u64 isize usize bool f32 f64";

/// Register class of a primitive scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarClass {
    Int,
    Float,
}

/// One row of the primitive scalar table: the facts every consumer derives
/// its own shape from (MLIR token, extension, struct-field size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarDesc {
    pub code: u8,
    pub bits: u16,
    pub signed: bool,
    pub is_bool: bool,
    pub class: ScalarClass,
}

const fn row(code: u8, bits: u16, signed: bool, is_bool: bool, class: ScalarClass) -> ScalarDesc {
    ScalarDesc {
        code,
        bits,
        signed,
        is_bool,
        class,
    }
}

use ScalarClass::{Float, Int};

/// Row `k` describes the `k`-th name of [`PRIM_SCALAR_NAMES`]. `isize` and
/// `usize` are 64 bits on every target the compiler accepts.
pub const PRIM_SCALAR_ROWS: &[ScalarDesc] = &[
    row(0, 8, true, false, Int),
    row(1, 16, true, false, Int),
    row(2, 32, true, false, Int),
    row(3, 64, true, false, Int),
    row(4, 8, false, false, Int),
    row(5, 16, false, false, Int),
    row(6, 32, false, false, Int),
    row(7, 64, false, false, Int),
    row(8, 64, true, false, Int),
    row(9, 64, false, false, Int),
    row(10, 8, false, true, Int),
    row(11, 32, true, false, Float),
    row(12, 64, true, false, Float),
];

impl ScalarDesc {
    /// The source name of this primitive, read from [`PRIM_SCALAR_NAMES`].
    pub fn name(&self) -> &'static str {
        PRIM_SCALAR_NAMES
            .split(' ')
            .nth(usize::from(self.code))
            .unwrap_or("")
    }

    /// Size in bytes (`bool` is one byte, C `_Bool`).
    pub fn bytes(&self) -> usize {
        usize::from(self.bits / 8)
    }

    /// An integer (or `bool`) narrower than 64 bits: the caller extends it.
    pub fn is_narrow_int(&self) -> bool {
        self.class == Int && self.bits < 64
    }

    /// The signless MLIR type of this width (`i8`..`i64`, `f32`, `f64`).
    pub fn signless(&self) -> &'static str {
        match (self.class, self.bits) {
            (Float, 32) => "f32",
            (Float, _) => "f64",
            (Int, 8) => "i8",
            (Int, 16) => "i16",
            (Int, 32) => "i32",
            (Int, _) => "i64",
        }
    }
}

/// The primitive scalar spelled `name`, or `None` for any other name.
pub fn prim_scalar(name: &str) -> Option<ScalarDesc> {
    let code = PRIM_SCALAR_NAMES.split(' ').position(|n| n == name)?;
    PRIM_SCALAR_ROWS.get(code).copied()
}

/// The only reader of a scalar type spelling: resolves an annotation to its
/// primitive row, or `None` for a pointer, aggregate or user type.
pub fn scalar_desc_of(ty: &crate::ast::TypeAnn) -> Option<ScalarDesc> {
    use crate::ast::TypeAnn;
    let name = match ty {
        TypeAnn::ScalarI32 => "i32",
        TypeAnn::ScalarI64 => "i64",
        TypeAnn::ScalarU32 => "u32",
        TypeAnn::ScalarF32 => "f32",
        TypeAnn::ScalarF64 => "f64",
        TypeAnn::ScalarBool => "bool",
        TypeAnn::Named(name) => name.as_str(),
        _ => return None,
    };
    prim_scalar(name)
}

#[cfg(test)]
mod table_tests {
    use super::{PRIM_SCALAR_NAMES, PRIM_SCALAR_ROWS, ScalarClass};

    /// (name, bits, signed, is_bool, class) of one table row.
    type Row = (String, u16, bool, bool, ScalarClass);

    fn read(rel: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    fn rust_rows() -> Vec<Row> {
        PRIM_SCALAR_ROWS
            .iter()
            .enumerate()
            .map(|(k, d)| {
                assert_eq!(usize::from(d.code), k, "row {k} carries code {}", d.code);
                (d.name().to_string(), d.bits, d.signed, d.is_bool, d.class)
            })
            .collect()
    }

    /// The normative table in `docs/specs/v1.0.md` §2.1, in code order.
    fn spec_rows() -> Vec<Row> {
        let text = read("docs/specs/v1.0.md");
        let start = text
            .find("### 2.1 Primitive scalar table")
            .expect("spec §2.1 primitive scalar table");
        let mut rows = Vec::new();
        for line in text[start..].lines().skip(1) {
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            if cells.len() < 8 {
                if rows.is_empty() {
                    continue;
                }
                break;
            }
            let Ok(code) = cells[1].parse::<usize>() else {
                continue;
            };
            assert_eq!(code, rows.len(), "spec row codes are positions");
            let flag = |c: &str| c == "yes";
            let class = match cells[6] {
                "INT" => ScalarClass::Int,
                "FLOAT" => ScalarClass::Float,
                other => panic!("spec class {other}"),
            };
            rows.push((
                cells[2].trim_matches('`').to_string(),
                cells[3].parse().expect("spec bits"),
                flag(cells[4]),
                flag(cells[5]),
                class,
            ));
        }
        rows
    }

    /// The text of the self-host function `name` in nb_prim_desc.mind.
    fn self_host_fn(text: &str, name: &str) -> String {
        let start = text
            .find(&format!("pub fn {name}("))
            .unwrap_or_else(|| panic!("self-host primitive table: `{name}` is missing"));
        let end = start + text[start..].find("\n}\n").expect("function end");
        text[start..end].to_string()
    }

    /// Each `if <var> == K {` ... `return <expr>;` pair of a self-host if-chain.
    fn if_chain(body: &str, var: &str) -> Vec<(usize, String)> {
        let head = format!("if {var} == ");
        let mut out = Vec::new();
        for arm in body.split(head.as_str()).skip(1) {
            let k: usize = arm[..arm.find(' ').expect("arm key")]
                .trim()
                .parse()
                .expect("arm key literal");
            let ret = arm.find("return ").expect("arm return") + "return ".len();
            let expr = arm[ret..ret + arm[ret..].find(';').expect("arm ;")].trim();
            out.push((k, expr.to_string()));
        }
        out
    }

    /// The self-host primitive table (nb_prim_desc.mind): the names text from
    /// `prim_names_byte` / `prim_names_byte_hi` and one `prim_row(f, bits,
    /// signed, is_bool, class, ctag)` per code in `prim_code_field`. Returns the
    /// names text, the rows and the C ABI registry tag column.
    fn self_host_rows() -> (String, Vec<Row>, Vec<u8>) {
        let text = read("examples/mindc_mind/nb_prim_desc.mind");
        let len: usize = self_host_fn(&text, "prim_names_len")
            .lines()
            .nth(1)
            .expect("prim_names_len body")
            .trim()
            .parse()
            .expect("prim_names_len literal");
        let mut bytes = vec![0u8; len];
        let mut seen = vec![false; len];
        for name in ["prim_names_byte", "prim_names_byte_hi"] {
            for (i, b) in if_chain(&self_host_fn(&text, name), "i") {
                assert!(i < len && !seen[i], "names byte {i} twice or out of range");
                seen[i] = true;
                bytes[i] = b.parse().expect("names byte literal");
            }
        }
        assert!(seen.iter().all(|&s| s), "every names byte is defined");
        let names = String::from_utf8(bytes).expect("names text is ASCII");
        let mut rows = Vec::new();
        let mut tags = Vec::new();
        for (code, expr) in if_chain(&self_host_fn(&text, "prim_code_field"), "c") {
            assert_eq!(code, rows.len(), "self-host rows are in code order");
            let args: Vec<i64> = expr
                .strip_prefix("prim_row(f,")
                .and_then(|a| a.strip_suffix(')'))
                .expect("row form prim_row(f, bits, signed, is_bool, class, ctag)")
                .split(',')
                .map(|a| a.trim().parse().expect("prim_row literal"))
                .collect();
            assert_eq!(args.len(), 5, "prim_row arity");
            tags.push(u8::try_from(args[4]).expect("registry tag"));
            let class = match args[3] {
                0 => ScalarClass::Int,
                1 => ScalarClass::Float,
                other => panic!("self-host class {other}"),
            };
            let name = names.split(' ').nth(code).expect("one name per code");
            rows.push((
                name.to_string(),
                u16::try_from(args[0]).expect("bits"),
                args[1] == 1,
                args[2] == 1,
                class,
            ));
        }
        (names, rows, tags)
    }

    /// The MCABI001 tag the Rust registry writes for the scalar `name`: the
    /// last descriptor byte of `extern "C" { fn probe() -> name; }`.
    fn rust_registry_tag(name: &str) -> u8 {
        use crate::ir::native_c_abi::{NativeCAbi, SymbolContext};
        let source = format!("extern \"C\" {{ fn probe() -> {name}; }}");
        let ast = crate::parser::parse(&source).expect("registry probe parses");
        let reserved = std::collections::BTreeSet::new();
        let abi = NativeCAbi::from_ast(
            &ast,
            &SymbolContext {
                imported: None,
                reserved: &reserved,
            },
        )
        .unwrap_or_else(|e| panic!("registry probe for `{name}`: {}", e.kind));
        *abi.descriptor_bytes().last().expect("descriptor bytes")
    }

    #[test]
    fn rust_rows_equal_the_spec_table() {
        let names: Vec<&str> = PRIM_SCALAR_NAMES.split(' ').collect();
        assert_eq!(names.len(), PRIM_SCALAR_ROWS.len(), "one row per name");
        let spec = spec_rows();
        assert_eq!(spec.len(), 13, "spec table rows");
        assert_eq!(rust_rows(), spec);
    }

    #[test]
    fn self_host_rows_equal_the_spec_table() {
        let (names, host, tags) = self_host_rows();
        assert_eq!(names, PRIM_SCALAR_NAMES, "self-host names text");
        assert_eq!(host, spec_rows());
        for (code, name) in names.split(' ').enumerate() {
            assert_eq!(
                tags[code],
                rust_registry_tag(name),
                "registry tag of `{name}`"
            );
        }
    }
}
