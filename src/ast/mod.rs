use crate::types::ConvPadding;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    start: usize,
    end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Int(i64),
    Ident(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeAnn {
    ScalarI32,
    Tensor { dtype: String, dims: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Lit(Literal, Span),
    Binary {
        op: BinOp,
        left: Box<Node>,
        right: Box<Node>,
        span: Span,
    },
    Paren(Box<Node>, Span),
    Tuple {
        elements: Vec<Node>,
        span: Span,
    },
    Call {
        callee: String,
        args: Vec<Node>,
        span: Span,
    },
    CallGrad {
        loss: Box<Node>,
        wrt: Vec<String>,
        span: Span,
    },
    CallTensorSum {
        x: Box<Node>,
        axes: Vec<i32>,
        keepdims: bool,
        span: Span,
    },
    CallTensorMean {
        x: Box<Node>,
        axes: Vec<i32>,
        keepdims: bool,
        span: Span,
    },
    CallReshape {
        x: Box<Node>,
        dims: Vec<String>,
        span: Span,
    },
    CallExpandDims {
        x: Box<Node>,
        axis: i32,
        span: Span,
    },
    CallSqueeze {
        x: Box<Node>,
        axes: Vec<i32>,
        span: Span,
    },
    CallTranspose {
        x: Box<Node>,
        axes: Option<Vec<i32>>,
        span: Span,
    },
    CallIndex {
        x: Box<Node>,
        axis: i32,
        i: i32,
        span: Span,
    },
    CallSlice {
        x: Box<Node>,
        axis: i32,
        start: i32,
        end: i32,
        span: Span,
    },
    CallSliceStride {
        x: Box<Node>,
        axis: i32,
        start: i32,
        end: i32,
        step: i32,
        span: Span,
    },
    CallGather {
        x: Box<Node>,
        axis: i32,
        idx: Box<Node>,
        span: Span,
    },
    CallDot {
        a: Box<Node>,
        b: Box<Node>,
        span: Span,
    },
    CallMatMul {
        a: Box<Node>,
        b: Box<Node>,
        span: Span,
    },
    CallTensorRelu {
        x: Box<Node>,
        span: Span,
    },
    CallTensorConv2d {
        x: Box<Node>,
        w: Box<Node>,
        stride_h: usize,
        stride_w: usize,
        padding: ConvPadding,
        span: Span,
    },
    Let {
        name: String,
        ann: Option<TypeAnn>,
        value: Box<Node>,
        span: Span,
    },
    Assign {
        name: String,
        value: Box<Node>,
        span: Span,
    },
}

impl Node {
    pub fn span(&self) -> Span {
        match self {
            Node::Lit(_, span)
            | Node::Binary { span, .. }
            | Node::Paren(_, span)
            | Node::Tuple { span, .. }
            | Node::Call { span, .. }
            | Node::CallGrad { span, .. }
            | Node::CallTensorSum { span, .. }
            | Node::CallTensorMean { span, .. }
            | Node::CallReshape { span, .. }
            | Node::CallExpandDims { span, .. }
            | Node::CallSqueeze { span, .. }
            | Node::CallTranspose { span, .. }
            | Node::CallIndex { span, .. }
            | Node::CallSlice { span, .. }
            | Node::CallSliceStride { span, .. }
            | Node::CallGather { span, .. }
            | Node::CallDot { span, .. }
            | Node::CallMatMul { span, .. }
            | Node::CallTensorRelu { span, .. }
            | Node::CallTensorConv2d { span, .. }
            | Node::Let { span, .. }
            | Node::Assign { span, .. } => *span,
        }
    }

    pub fn span_start(&self) -> usize {
        self.span().start()
    }

    pub fn span_end(&self) -> usize {
        self.span().end()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Module {
    pub items: Vec<Node>,
}
