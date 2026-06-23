// Regression: `mindc fmt` must not drop item declarations nested inside a
// `module NAME { … }` block.
//
// A `module NAME { … }` header parses to a transparent `Node::Block` whose
// statements ARE the module's items (`const`, `struct`, `enum`, `type`,
// `import`, `export`). The printer used to route every non-fn Block statement
// through `emit_expr`, which prints ONLY an item's name — so
// `const VERSION: i32 = 1` collapsed to the bare identifier `VERSION`, silently
// erasing the declaration. The formatted source then referenced an undefined
// symbol: a fmt silent miscompile (a valid program became invalid / changed
// value). See src/fmt/printer.rs `emit_body_stmts`.
//
// This test fails on the pre-fix printer (the declaration keyword is gone from
// the output and the re-parsed module loses the item) and passes after it.

use libmind::ast::Node;
use libmind::fmt::format_source;
use libmind::parser::parse;
use libmind::project::MindcraftFormatConfig;

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default()).expect("source must format")
}

/// Count top-level item nodes of a given kind after flattening the transparent
/// `module`-block `Block` wrappers (mirrors how the compiler splices them).
fn count_kind(src: &str, want: fn(&Node) -> bool) -> usize {
    let module = parse(src).expect("source must parse");
    let mut n = 0;
    let mut stack: Vec<&Node> = module.items.iter().collect();
    while let Some(node) = stack.pop() {
        if want(node) {
            n += 1;
        }
        if let Node::Block { stmts, .. } = node {
            for s in stmts {
                stack.push(s);
            }
        }
    }
    n
}

#[test]
fn module_block_const_survives_fmt() {
    let src = "module governance {\n    const VERSION: i32 = 1\n}\n";
    // Sanity: the original source defines exactly one const.
    assert_eq!(
        count_kind(src, |n| matches!(n, Node::Const { .. })),
        1,
        "fixture must declare one const"
    );

    let out = fmt(src);
    // The `const` keyword and its value must survive — not collapse to `VERSION`.
    assert!(
        out.contains("const VERSION"),
        "fmt dropped the const declaration; got:\n{out}"
    );
    assert!(
        out.contains("= 1"),
        "fmt dropped the const value; got:\n{out}"
    );
    // Structural round-trip: the formatted source must still declare the const.
    assert_eq!(
        count_kind(&out, |n| matches!(n, Node::Const { .. })),
        1,
        "fmt'd source lost the const declaration; got:\n{out}"
    );
}

#[test]
fn module_block_struct_survives_fmt() {
    let src = "module types {\n    struct Point {\n        x: i64,\n        y: i64,\n    }\n}\n";
    assert_eq!(count_kind(src, |n| matches!(n, Node::StructDef { .. })), 1);

    let out = fmt(src);
    assert!(
        out.contains("struct Point"),
        "fmt dropped the struct declaration; got:\n{out}"
    );
    assert!(out.contains("x: i64") && out.contains("y: i64"));
    assert_eq!(
        count_kind(&out, |n| matches!(n, Node::StructDef { .. })),
        1,
        "fmt'd source lost the struct declaration; got:\n{out}"
    );
}

#[test]
fn module_block_enum_survives_fmt() {
    let src = "module palette {\n    enum Color {\n        Red,\n        Green,\n    }\n}\n";
    assert_eq!(count_kind(src, |n| matches!(n, Node::EnumDef { .. })), 1);

    let out = fmt(src);
    assert!(
        out.contains("enum Color"),
        "fmt dropped the enum declaration; got:\n{out}"
    );
    assert_eq!(
        count_kind(&out, |n| matches!(n, Node::EnumDef { .. })),
        1,
        "fmt'd source lost the enum declaration; got:\n{out}"
    );
}
