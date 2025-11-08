#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Int(i64),
    Ident(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    Lit(Literal),
    Binary { op: BinOp, left: Box<Node>, right: Box<Node> },
    Paren(Box<Node>),
    Let { name: String, ann: Option<TypeAnn>, value: Box<Node> },
    Assign { name: String, value: Box<Node> },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Module {
    pub items: Vec<Node>,
}
