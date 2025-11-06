use chumsky::prelude::*;
use crate::ast::{Literal, Node, Module};

pub fn parser() -> impl Parser<char, Module, Error = Simple<char>> {
    let ident = text::ident().map(|s: String| Node::Lit(Literal::Ident(s)));
    let int   = text::int(10).map(|s: String| Node::Lit(Literal::Int(s.parse().unwrap())));
    let node  = choice((ident, int));

    node
        .repeated()
        .at_least(1)
        .map(|items| Module { items })
}

pub fn parse(input: &str) -> Result<Module, Vec<Simple<char>>> {
    parser().parse(input)
}
