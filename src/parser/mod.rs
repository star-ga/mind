//! # Example
//! ```
//! use mind::{parser, eval};
//! let module = parser::parse("1 + 2 * 3").unwrap();
//! assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);
//! ```

use chumsky::prelude::*;

use crate::ast::BinOp;

use crate::ast::Literal;

use crate::ast::Module;

use crate::ast::Node;

use crate::ast::Span;

use crate::ast::TypeAnn;

use crate::diagnostics::Diagnostic as PrettyDiagnostic;
use crate::types::ConvPadding;

fn kw(s: &'static str) -> impl Parser<char, &'static str, Error = Simple<char>> {
    text::keyword(s).to(s)
}

pub fn parser() -> impl Parser<char, Module, Error = Simple<char>> {
    enum ReduceArg {
        Axes(Vec<i32>),
        Keepdims(bool),
    }

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
        let bool_lit = choice((kw("true").to(true), kw("false").to(false)))
            .padded()
            .boxed();
        let signed_int = just('-')
            .or_not()
            .then(text::int(10))
            .map(|(sign, digits): (Option<char>, String)| {
                let mut value = digits.parse::<i32>().unwrap();
                if sign.is_some() {
                    value = -value;
                }
                value
            })
            .padded();
        let axes_list = just('[')
            .padded()
            .ignore_then(signed_int.separated_by(just(',').padded()).allow_trailing())
            .then_ignore(just(']').padded());

        let tuple_or_paren = just('(')
            .ignore_then(
                expr.clone()
                    .separated_by(just(',').padded())
                    .allow_trailing(),
            )
            .then_ignore(just(')').padded())
            .map_with_span(|items, sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                if items.len() == 1 {
                    Node::Paren(Box::new(items.into_iter().next().unwrap()), span)
                } else {
                    Node::Tuple {
                        elements: items,
                        span,
                    }
                }
            });

        let wrt_list = just(',')
            .padded()
            .ignore_then(kw("wrt"))
            .ignore_then(just('=').padded())
            .ignore_then(just('[').padded())
            .ignore_then(
                text::ident()
                    .padded()
                    .separated_by(just(',').padded())
                    .allow_trailing(),
            )
            .then_ignore(just(']').padded());

        let grad_call = kw("grad")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone().then(wrt_list.or_not()))
            .then_ignore(just(')').padded())
            .map_with_span(|(loss, maybe_wrt), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallGrad {
                    loss: Box::new(loss),
                    wrt: maybe_wrt.unwrap_or_default(),
                    span,
                }
            })
            .boxed();

        let reduce_arg = choice((
            kw("axes")
                .ignore_then(just('=').padded())
                .ignore_then(axes_list)
                .map(ReduceArg::Axes),
            kw("keepdims")
                .ignore_then(just('=').padded())
                .ignore_then(bool_lit.clone())
                .map(ReduceArg::Keepdims),
        ))
        .boxed();

        let tensor_sum_call = just("tensor.sum")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then(
                just(',')
                    .padded()
                    .ignore_then(reduce_arg.clone())
                    .repeated(),
            )
            .then_ignore(just(')').padded())
            .map_with_span(|(x, extras), sp: std::ops::Range<usize>| {
                let mut axes = Vec::new();
                let mut keepdims = false;
                for arg in extras {
                    match arg {
                        ReduceArg::Axes(v) => axes = v,
                        ReduceArg::Keepdims(v) => keepdims = v,
                    }
                }
                let span = Span::new(sp.start, sp.end);
                Node::CallTensorSum {
                    x: Box::new(x),
                    axes,
                    keepdims,
                    span,
                }
            });

        let tensor_mean_call = just("tensor.mean")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then(
                just(',')
                    .padded()
                    .ignore_then(reduce_arg.clone())
                    .repeated(),
            )
            .then_ignore(just(')').padded())
            .map_with_span(|(x, extras), sp: std::ops::Range<usize>| {
                let mut axes = Vec::new();
                let mut keepdims = false;
                for arg in extras {
                    match arg {
                        ReduceArg::Axes(v) => axes = v,
                        ReduceArg::Keepdims(v) => keepdims = v,
                    }
                }
                let span = Span::new(sp.start, sp.end);
                Node::CallTensorMean {
                    x: Box::new(x),
                    axes,
                    keepdims,
                    span,
                }
            });

        let reshape_dims = just('(')
            .padded()
            .ignore_then(
                choice((text::int(10), text::ident()))
                    .map(|s: String| s)
                    .padded()
                    .separated_by(just(',').padded())
                    .allow_trailing(),
            )
            .then_ignore(just(')').padded());

        let tensor_reshape_call = just("tensor.reshape")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(reshape_dims)
            .then_ignore(just(')').padded())
            .map_with_span(|(x, dims), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallReshape {
                    x: Box::new(x),
                    dims,
                    span,
                }
            });

        let expand_axis = kw("axis")
            .ignore_then(just('=').padded())
            .ignore_then(signed_int);

        let tensor_expand_dims_call = just("tensor.expand_dims")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(expand_axis)
            .then_ignore(just(')').padded())
            .map_with_span(|(x, axis), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallExpandDims {
                    x: Box::new(x),
                    axis,
                    span,
                }
            });

        let squeeze_axes = kw("axes")
            .ignore_then(just('=').padded())
            .ignore_then(axes_list);

        let tensor_squeeze_call = just("tensor.squeeze")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then(just(',').padded().ignore_then(squeeze_axes).or_not())
            .then_ignore(just(')').padded())
            .map_with_span(|(x, maybe_axes), sp: std::ops::Range<usize>| {
                let axes = maybe_axes.unwrap_or_default();
                let span = Span::new(sp.start, sp.end);
                Node::CallSqueeze {
                    x: Box::new(x),
                    axes,
                    span,
                }
            });

        let transpose_axes = kw("axes")
            .ignore_then(just('=').padded())
            .ignore_then(axes_list);

        let tensor_transpose_call = just("tensor.transpose")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then(just(',').padded().ignore_then(transpose_axes).or_not())
            .then_ignore(just(')').padded())
            .map_with_span(|(x, maybe_axes), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallTranspose {
                    x: Box::new(x),
                    axes: maybe_axes,
                    span,
                }
            });

        let tensor_index_call = just("tensor.index")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(
                kw("axis")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(',').padded())
            .then(
                kw("i")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(')').padded())
            .map_with_span(|((x, axis), i), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallIndex {
                    x: Box::new(x),
                    axis,
                    i,
                    span,
                }
            });

        let tensor_slice_call = just("tensor.slice")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(
                kw("axis")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(',').padded())
            .then(
                kw("start")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(',').padded())
            .then(
                kw("end")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(')').padded())
            .map_with_span(|(((x, axis), start), end), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallSlice {
                    x: Box::new(x),
                    axis,
                    start,
                    end,
                    span,
                }
            });

        let tensor_slice_stride_call = just("tensor.slice_stride")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(
                kw("axis")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(',').padded())
            .then(
                kw("start")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(',').padded())
            .then(
                kw("end")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(',').padded())
            .then(
                kw("step")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(')').padded())
            .map_with_span(
                |((((x, axis), start), end), step), sp: std::ops::Range<usize>| {
                    let span = Span::new(sp.start, sp.end);
                    Node::CallSliceStride {
                        x: Box::new(x),
                        axis,
                        start,
                        end,
                        step,
                        span,
                    }
                },
            );

        let tensor_gather_call = just("tensor.gather")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(
                kw("axis")
                    .ignore_then(just('=').padded())
                    .ignore_then(signed_int),
            )
            .then_ignore(just(',').padded())
            .then(
                kw("idx")
                    .ignore_then(just('=').padded())
                    .to(())
                    .or_not()
                    .then(expr.clone())
                    .map(|(_, idx_expr)| idx_expr),
            )
            .then_ignore(just(')').padded())
            .map_with_span(|((x, axis), idx), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallGather {
                    x: Box::new(x),
                    axis,
                    idx: Box::new(idx),
                    span,
                }
            });

        let tensor_dot_call = just("tensor.dot")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(expr.clone())
            .then_ignore(just(')').padded())
            .map_with_span(|(a, b), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallDot {
                    a: Box::new(a),
                    b: Box::new(b),
                    span,
                }
            });

        let tensor_matmul_call = just("tensor.matmul")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(expr.clone())
            .then_ignore(just(')').padded())
            .map_with_span(|(a, b), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallMatMul {
                    a: Box::new(a),
                    b: Box::new(b),
                    span,
                }
            });

        enum Conv2dArg {
            StrideH(usize),
            StrideW(usize),
            Padding(ConvPadding),
        }

        let stride_value = signed_int
            .map_with_span(|v, sp: std::ops::Range<usize>| (v, sp))
            .try_map(|(value, sp), _| {
                if value <= 0 {
                    Err(Simple::custom(sp, "stride must be positive"))
                } else {
                    Ok(value as usize)
                }
            });

        let padding_value = just('"')
            .ignore_then(filter(|c| *c != '"').repeated().collect::<String>())
            .then_ignore(just('"'))
            .map_with_span(|s, sp: std::ops::Range<usize>| (s, sp))
            .try_map(|(value, sp), _| {
                value
                    .parse()
                    .map(Conv2dArg::Padding)
                    .map_err(|_| Simple::custom(sp, "padding must be \"valid\" or \"same\""))
            });

        let conv2d_arg = choice((
            kw("stride_h")
                .ignore_then(just('=').padded())
                .ignore_then(stride_value.map(Conv2dArg::StrideH)),
            kw("stride_w")
                .ignore_then(just('=').padded())
                .ignore_then(stride_value.map(Conv2dArg::StrideW)),
            kw("padding")
                .ignore_then(just('=').padded())
                .ignore_then(padding_value),
        ));

        let tensor_relu_call = just("tensor.relu")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(')').padded())
            .map_with_span(|x, sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::CallTensorRelu {
                    x: Box::new(x),
                    span,
                }
            });

        let tensor_conv2d_call = just("tensor.conv2d")
            .ignore_then(just('(').padded())
            .ignore_then(expr.clone())
            .then_ignore(just(',').padded())
            .then(expr.clone())
            .then(just(',').padded().ignore_then(conv2d_arg).repeated())
            .then_ignore(just(')').padded())
            .map_with_span(|((x, w), extras), sp: std::ops::Range<usize>| {
                let mut stride_h = 1usize;
                let mut stride_w = 1usize;
                let mut padding = ConvPadding::Valid;
                for arg in extras {
                    match arg {
                        Conv2dArg::StrideH(v) => stride_h = v,
                        Conv2dArg::StrideW(v) => stride_w = v,
                        Conv2dArg::Padding(p) => padding = p,
                    }
                }
                let span = Span::new(sp.start, sp.end);
                Node::CallTensorConv2d {
                    x: Box::new(x),
                    w: Box::new(w),
                    stride_h,
                    stride_w,
                    padding,
                    span,
                }
            });

        let call = dotted_ident
            .map_with_span(|name, sp: std::ops::Range<usize>| (name, Span::new(sp.start, sp.end)))
            .then(
                just('(')
                    .padded()
                    .ignore_then(
                        expr.clone()
                            .separated_by(just(',').padded())
                            .allow_trailing(),
                    )
                    .then_ignore(just(')').padded()),
            )
            .map_with_span(
                |((callee, _callee_span), args), sp: std::ops::Range<usize>| {
                    let span = Span::new(sp.start, sp.end);
                    Node::Call { callee, args, span }
                },
            );

        let atom = choice((
            grad_call,
            tensor_sum_call,
            tensor_mean_call,
            tensor_reshape_call,
            tensor_expand_dims_call,
            tensor_squeeze_call,
            tensor_transpose_call,
            tensor_index_call,
            tensor_slice_call,
            tensor_slice_stride_call,
            tensor_gather_call,
            tensor_dot_call,
            tensor_matmul_call,
            tensor_relu_call,
            tensor_conv2d_call,
            call,
            int,
            ident_expr,
            tuple_or_paren,
        ))
        .padded()
        .boxed();

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
                Node::Binary {
                    op,
                    left: Box::new(l),
                    right: Box::new(r),
                    span,
                }
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
                Node::Binary {
                    op,
                    left: Box::new(l),
                    right: Box::new(r),
                    span,
                }
            })
    });

    let dtype = choice((
        just("i32").to("i32".to_string()),
        just("f32").to("f32".to_string()),
    ))
    .padded();

    let dim = choice((
        text::int(10).map(|s: String| s),
        text::ident().map(|s: String| s),
    ))
    .padded();

    let dims = just('(')
        .ignore_then(dim.separated_by(just(',').padded()).allow_trailing())
        .then_ignore(just(')'))
        .padded();

    let type_ann = choice((
        dtype.clone().map(|_| TypeAnn::ScalarI32),
        kw("Tensor")
            .ignore_then(just('['))
            .ignore_then(dtype.clone())
            .then_ignore(just(',').padded())
            .then(dims)
            .then_ignore(just(']'))
            .map(|(dt, shape)| TypeAnn::Tensor {
                dtype: dt,
                dims: shape,
            }),
    ))
    .padded()
    .boxed();

    let let_stmt = kw("let")
        .padded()
        .ignore_then(
            text::ident().map_with_span(|s: String, sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                (s, span)
            }),
        )
        .then(just(':').ignore_then(type_ann.clone()).or_not().padded())
        .then_ignore(just('=').padded())
        .then(expr.clone())
        .map_with_span(
            |(((name, _name_span), ann), value), sp: std::ops::Range<usize>| {
                let span = Span::new(sp.start, sp.end);
                Node::Let {
                    name,
                    ann,
                    value: Box::new(value),
                    span,
                }
            },
        );

    let assign_stmt = text::ident()
        .map_with_span(|s: String, _| s)
        .then_ignore(just('=').padded())
        .then(expr.clone())
        .map_with_span(|(name, value), sp: std::ops::Range<usize>| {
            let span = Span::new(sp.start, sp.end);
            Node::Assign {
                name,
                value: Box::new(value),
                span,
            }
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
