//! # Example
//! ```
//! use mind::{parser, eval};
//! let module = parser::parse("1 + 2 * 3").unwrap();
//! assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);
//! ```

use chumsky::prelude::*;

use crate::ast::{BinOp, Literal, Module, Node, Span, TypeAnn};
use crate::diagnostics::Diagnostic as PrettyDiagnostic;

fn kw(s: &'static str) -> impl Parser<char, &'static str, Error = Simple<char>> {
    text::keyword(s).to(s)
}

pub fn parser() -> impl Parser<char, Module, Error = Simple<char>> {
    let int = text::int(10).map_with_span(|s: String, sp: std::ops::Range<usize>| {
        let span = Span::new(sp.start, sp.end);
        Node::Lit(Literal::Int(s.parse().unwrap()), span)
    });
    let ident_expr = text::ident().map_with_span(|s: String, sp: std::ops::Range<usize>| {
        let span = Span::new(sp.start, sp.end);
        Node::Lit(Literal::Ident(s), span)
    });
    let dotted_ident = text::ident::<char, Simple<char>>()
        .then(just('.').ignore_then(text::ident()).repeated())
        .map(|(first, rest)| {
            let mut name = first;
            for part in rest {
                name.push('.');
                name.push_str(&part);
            }
            name
        });

    let expr = recursive(|expr| {
        let tuple_or_paren = just('(')
            .ignore_then(expr.clone().separated_by(just(',').padded()).allow_trailing())
            .then_ignore(just(')').padded())
            .map_with_span(|items, sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                if items.len() == 1 {
                    Node::Paren(Box::new(items.into_iter().next().unwrap()), span)
                } else {
                    Node::Tuple { elements: items, span }
                }
            });

        let wrt_list = just(',')
            .padded()
            .ignore_then(kw("wrt"))
            .ignore_then(just('=').padded())
            .ignore_then(just('[').padded())
            .ignore_then(text::ident().padded().separated_by(just(',').padded()).allow_trailing())
            .then_ignore(just(']').padded());

        let grad_call = kw("grad")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone().then(wrt_list.or_not()))
            .then_ignore(just(')').padded())
            .map_with_span(|(loss, maybe_wrt), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallGrad { loss: Box::new(loss), wrt: maybe_wrt.unwrap_or_default(), span }
            })
            .boxed();

        let call = dotted_ident
            .clone()
            .map_with_span(|name, sp: std::ops::Range<usize>| (name, Span::new(sp.start, sp.end)))
            .then(
                just('(')
                    .padded()
                    .ignore_then(expr.clone().separated_by(just(',').padded()).allow_trailing())
                    .then_ignore(just(')').padded()),
            )
            .map_with_span(|((callee, _callee_span), args), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::Call { callee, args, span }
            });

        let atom =
            choice((grad_call, call, int.clone(), ident_expr.clone(), tuple_or_paren)).padded();

        let product = atom
            .clone()
            .then(
                (choice((just('*').to(BinOp::Mul), just('/').to(BinOp::Div)))
                    .padded()
                    .then(atom.clone()))
                .repeated(),
            )
            .foldl(|l, (op, r)| {
                let span = Span::new(l.span_start(), r.span_end());
                Node::Binary { op, left: Box::new(l), right: Box::new(r), span }
            });

        product
            .clone()
            .then(
                (choice((just('+').to(BinOp::Add), just('-').to(BinOp::Sub)))
                    .padded()
                    .then(product))
                .repeated(),
            )
            .foldl(|l, (op, r)| {
                let span = Span::new(l.span_start(), r.span_end());
                Node::Binary { op, left: Box::new(l), right: Box::new(r), span }
            })
    });

    let dtype =
        choice((just("i32").to("i32".to_string()), just("f32").to("f32".to_string()))).padded();

    let dim = choice((text::int(10).map(|s: String| s), text::ident().map(|s: String| s))).padded();

    let dims = just('(')
        .ignore_then(dim.clone().separated_by(just(',').padded()).allow_trailing())
        .then_ignore(just(')'))
        .padded();

    let type_ann = choice((
        dtype.clone().map(|_| TypeAnn::ScalarI32),
        kw("Tensor")
            .ignore_then(just('['))
            .ignore_then(dtype.clone())
            .then_ignore(just(',').padded())
            .then(dims.clone())
            .then_ignore(just(']'))
            .map(|(dt, shape)| TypeAnn::Tensor { dtype: dt, dims: shape }),
    ))
    .padded()
    .boxed();

    let let_stmt = kw("let")
        .padded()
        .ignore_then(text::ident().map_with_span(|s: String, sp: std::ops::Range<usize>| {
            let span = Span::new(sp.start, sp.end);
            (s, span)
        }))
        .then(just(':').ignore_then(type_ann.clone()).or_not().padded())
        .then_ignore(just('=').padded())
        .then(expr.clone())
        .map_with_span(|(((name, _name_span), ann), value), sp: std::ops::Range<usize>| {
            let span = Span::new(sp.start, sp.end);
            Node::Let { name, ann, value: Box::new(value), span }
        });

    let assign_stmt = text::ident()
        .map_with_span(|s: String, _| s)
        .then_ignore(just('=').padded())
        .then(expr.clone())
        .map_with_span(|(name, value), sp: std::ops::Range<usize>| {
            let span = Span::new(sp.start, sp.end);
            Node::Assign { name, value: Box::new(value), span }
        });

    let stmt = choice((let_stmt, assign_stmt, expr.clone())).padded();

    let stmts =
        stmt.separated_by(one_of(";\n").repeated().at_least(1)).allow_trailing().at_least(1);

    stmts.map(|items| Module { items })
}

pub fn parse(input: &str) -> Result<Module, Vec<Simple<char>>> {
    parser().parse(input)
}

/// Parse with pretty diagnostics instead of raw chumsky errors.
pub fn parse_with_diagnostics(input: &str) -> Result<Module, Vec<PrettyDiagnostic>> {
    let (maybe_module, errs) = parser().parse_recovery(input);
    if errs.is_empty() {
        Ok(maybe_module.expect("parser returned no module without errors"))
    } else {
        let ds = errs
            .into_iter()
            .map(|e| crate::diagnostics::Diagnostic::from_chumsky(input, e))
            .collect();
        Err(ds)
    }
}
