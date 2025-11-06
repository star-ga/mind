//! # Example
//! ```
//! use mind::{parser, eval};
//! let module = parser::parse("1 + 2 * 3").unwrap();
//! assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);
//! ```

use chumsky::prelude::*;

use crate::ast::{BinOp, Literal, Module, Node};

pub fn parser() -> impl Parser<char, Module, Error = Simple<char>> {
    let int = text::int(10).map(|s: String| Node::Lit(Literal::Int(s.parse().unwrap())));
    let ident = text::ident().map(|s: String| Node::Lit(Literal::Ident(s)));

    let expr = recursive(|expr| {
        let atom = choice((
            int.clone(),
            ident.clone(),
            just('(')
                .ignore_then(expr.clone())
                .then_ignore(just(')'))
                .map(|node| Node::Paren(Box::new(node))),
        ))
        .padded();

        let product = atom
            .clone()
            .then(
                (choice((just('*').to(BinOp::Mul), just('/').to(BinOp::Div))).then(atom.clone()))
                    .repeated(),
            )
            .foldl(|left, (op, right)| Node::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });

        product
            .clone()
            .then(
                (choice((just('+').to(BinOp::Add), just('-').to(BinOp::Sub))).then(product))
                    .repeated(),
            )
            .foldl(|left, (op, right)| Node::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
    });

    expr.repeated().at_least(1).map(|items| Module { items })
}

pub fn parse(input: &str) -> Result<Module, Vec<Simple<char>>> {
    parser().parse(input)
}
