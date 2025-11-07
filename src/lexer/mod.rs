use logos::Logos;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
pub enum Token {
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().unwrap())]
    Int(i64),

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
}

pub fn lex(input: &str) -> Vec<Token> {
    Token::lexer(input).filter_map(Result::ok).collect()
}
