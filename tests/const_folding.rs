use mind::ast::Node;
use mind::opt::fold;
use mind::parser;

#[test]
fn folds_simple_arith() {
    let m = parser::parse("1 + 2 * 3").unwrap();
    let node = &m.items[0];
    let f = fold::fold(node);
    if let Node::Lit(_, _) = f {
        // folded to literal
    } else {
        panic!("not folded");
    }
}
