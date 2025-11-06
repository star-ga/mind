//! # Example
//! ```
//! use mind::{parser, eval};
//! let module = parser::parse("1 + 2 * 3").unwrap();
//! assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);
//! ```

use chumsky::prelude::*;

use crate::ast::{BinOp, Literal, Module, Node};

fn kw(s: &'static str) -> impl Parser<char, &'static str, Error = Simple<char>> {
    text::keyword(s).to(s)
}

pub fn parser() -> impl Parser<char, Module, Error = Simple<char>> {
    let int = text::int(10).map(|s: String| Node::Lit(Literal::Int(s.parse().unwrap())));
    let ident = text::ident().map(|s: String| Node::Lit(Literal::Ident(s.clone())));

    let expr = recursive(|expr| {
        let atom = choice((
            int.clone(),
            ident.clone(),
            just('(')
                .ignore_then(expr.clone())
                .then_ignore(just(')'))
                .map(|e| Node::Paren(Box::new(e))),
        ))
        .padded();

        let product = atom
            .clone()
            .then(
                (choice((just('*').to(BinOp::Mul), just('/').to(BinOp::Div)))
                    .padded()
                    .then(atom.clone()))
                .repeated(),
            )
            .foldl(|l, (op, r)| Node::Binary {
                op,
                left: Box::new(l),
                right: Box::new(r),
            });

        product
            .clone()
            .then(
                (choice((just('+').to(BinOp::Add), just('-').to(BinOp::Sub)))
                    .padded()
                    .then(product))
                .repeated(),
            )
            .foldl(|l, (op, r)| Node::Binary {
                op,
                left: Box::new(l),
                right: Box::new(r),
            })
    });

    let ident_str = ident.clone().map(|n| {
        if let Node::Lit(Literal::Ident(s)) = n {
            s
        } else {
            unreachable!()
        }
    });

    let let_stmt = kw("let")
        .padded()
        .ignore_then(ident_str.clone())
        .then_ignore(just('=').padded())
        .then(expr.clone())
        .map(|(name, value)| Node::Let {
            name,
            value: Box::new(value),
        });

    let assign_stmt = ident_str
        .then_ignore(just('=').padded())
        .then(expr.clone())
        .map(|(name, value)| Node::Assign {
            name,
            value: Box::new(value),
        });

    let stmt = choice((let_stmt, assign_stmt, expr.clone())).padded();

    let stmts = stmt
        .separated_by(one_of(";\n").repeated().at_least(1))
        .allow_trailing()
        .at_least(1);

    stmts.map(|items| Module { items })
}

pub fn parse(input: &str) -> Result<Module, Vec<Simple<char>>> {
    parser().parse(input)
}
