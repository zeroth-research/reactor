/*!
# The abstract syntax of the accepted SystemVerilog subset

The surface syntax after parsing, before typing. Operators are kept as
written (`>`, `!`, `&&`, `~^`, …); lowering rewrites them into [`SvOp`]s.
The shapes are plain trees without sharing, so that they map one-to-one to
Lean inductive types.

[`SvOp`]: crate::SvOp
*/

use crate::value::BitVec;
use std::fmt;

/// A source position, 1-based.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// A located error, from any phase of the translation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Error {
    pub span: Span,
    pub msg: String,
}

impl Error {
    pub fn new(span: Span, msg: impl Into<String>) -> Self {
        Self {
            span,
            msg: msg.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.span, self.msg)
    }
}

impl std::error::Error for Error {}

pub type Name = String;

/// The type of a signal or expression: a 2-state vector `[width-1:0]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Ty {
    pub width: usize,
    pub signed: bool,
}

impl Ty {
    pub fn unsigned(width: usize) -> Self {
        Self {
            width,
            signed: false,
        }
    }

    /// The type of conditions and of comparison results.
    pub const BIT: Ty = Ty {
        width: 1,
        signed: false,
    };
}

/// SystemVerilog syntax: `logic signed [7:0]`.
impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "logic")?;
        if self.signed {
            write!(f, " signed")?;
        }
        if self.width > 1 {
            write!(f, " [{}:0]", self.width - 1)?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Module {
    pub name: Name,
    /// The ports of the ANSI header, in order.
    pub ports: Vec<Port>,
    /// The internal signals.
    pub decls: Vec<Decl>,
    pub items: Vec<Item>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Input,
    Output,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Port {
    pub dir: Dir,
    pub decl: Decl,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decl {
    pub name: Name,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Item {
    /// `assign lhs = rhs;`, with `span` the position of `lhs`.
    Assign { lhs: Name, rhs: Expr, span: Span },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExprKind {
    /// A sized literal (`8'd3`, `4'sb1010`); `ty.width` is the size.
    Lit {
        ty: Ty,
        value: BitVec,
    },
    Var(Name),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// `c ? a : b`
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnOp {
    /// `~`
    BitNot,
    /// `!`
    LogNot,
    /// `-`
    Neg,
    /// `+`
    Plus,
    /// `&`
    RedAnd,
    /// `|`
    RedOr,
    /// `^`
    RedXor,
    /// `~&`
    RedNand,
    /// `~|`
    RedNor,
    /// `~^`, `^~`
    RedXnor,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `~^`, `^~`
    BitXnor,
    Add,
    Sub,
    Mul,
    Eq,
    Ne,
    /// `===`: equal to `==` in 2-state (there are no x/z values).
    CaseEq,
    /// `!==`
    CaseNe,
    Lt,
    Le,
    Gt,
    Ge,
    /// `&&`
    LogAnd,
    /// `||`
    LogOr,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `<<<`
    AShl,
    /// `>>>`
    AShr,
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnOp::*;
        f.write_str(match self {
            BitNot => "~",
            LogNot => "!",
            Neg => "-",
            Plus => "+",
            RedAnd => "&",
            RedOr => "|",
            RedXor => "^",
            RedNand => "~&",
            RedNor => "~|",
            RedXnor => "~^",
        })
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BinOp::*;
        f.write_str(match self {
            BitAnd => "&",
            BitOr => "|",
            BitXor => "^",
            BitXnor => "~^",
            Add => "+",
            Sub => "-",
            Mul => "*",
            Eq => "==",
            Ne => "!=",
            CaseEq => "===",
            CaseNe => "!==",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            LogAnd => "&&",
            LogOr => "||",
            Shl => "<<",
            Shr => ">>",
            AShl => "<<<",
            AShr => ">>>",
        })
    }
}

/// Fully parenthesized SystemVerilog syntax, for tests and messages.
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::Lit { ty, value } => {
                let s = if ty.signed { "s" } else { "" };
                write!(f, "{}'{s}d{}", ty.width, value.unsigned())
            }
            ExprKind::Var(x) => write!(f, "{x}"),
            ExprKind::Unary(op, e) => write!(f, "({op}{e})"),
            ExprKind::Binary(op, a, b) => write!(f, "({a} {op} {b})"),
            ExprKind::Cond(c, a, b) => write!(f, "({c} ? {a} : {b})"),
        }
    }
}
