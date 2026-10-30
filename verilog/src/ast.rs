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

impl Module {
    /// The clock of the first `always_ff` block, if any. After
    /// [`typing::check`](crate::typing::check), all blocks have this clock.
    pub fn clock(&self) -> Option<&Name> {
        self.items.iter().find_map(|item| match item {
            Item::AlwaysFf { clock, .. } => Some(clock),
            Item::Assign { .. } => None,
        })
    }

    /// The signals assigned in `always_ff` blocks, in source order.
    pub fn registers(&self) -> Vec<&Name> {
        self.items
            .iter()
            .filter(|item| matches!(item, Item::AlwaysFf { .. }))
            .flat_map(Item::targets)
            .collect()
    }
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
    /// `always_ff @(posedge clock) body`, with `span` the position of
    /// `always_ff` and `clock_span` that of `clock`.
    AlwaysFf {
        clock: Name,
        clock_span: Span,
        body: Stmt,
        span: Span,
    },
}

impl Item {
    /// The signals the item drives, in the order of their first
    /// assignment.
    pub fn targets(&self) -> Vec<&Name> {
        match self {
            Item::Assign { lhs, .. } => vec![lhs],
            Item::AlwaysFf { body, .. } => {
                let mut out = Vec::new();
                body.targets(&mut out);
                out
            }
        }
    }

    /// The signals the item reads, with the position of every read.
    pub fn reads(&self) -> Vec<(&Name, Span)> {
        let mut out = Vec::new();
        match self {
            Item::Assign { rhs, .. } => rhs.reads(&mut out),
            Item::AlwaysFf { body, .. } => body.reads(&mut out),
        }
        out
    }
}

/// A procedural statement.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StmtKind {
    /// `begin … end`; also the null statement `;` (empty).
    Block(Vec<Stmt>),
    /// `if (c) s1 else s2`; a missing `else` is an empty block, and
    /// `else if` is an `if` in the `else` branch.
    If(Expr, Box<Stmt>, Box<Stmt>),
    /// `lhs <= rhs;`, with `span` the position of `lhs`.
    NonBlocking { lhs: Name, rhs: Expr },
}

impl Stmt {
    pub fn empty(span: Span) -> Self {
        Stmt {
            kind: StmtKind::Block(Vec::new()),
            span,
        }
    }

    /// Adds the signals assigned in the statement to `out`, each once, in
    /// the order of their first assignment.
    fn targets<'s>(&'s self, out: &mut Vec<&'s Name>) {
        match &self.kind {
            StmtKind::Block(stmts) => stmts.iter().for_each(|s| s.targets(out)),
            StmtKind::If(_, a, b) => {
                a.targets(out);
                b.targets(out);
            }
            StmtKind::NonBlocking { lhs, .. } => {
                if !out.contains(&lhs) {
                    out.push(lhs);
                }
            }
        }
    }

    fn reads<'s>(&'s self, out: &mut Vec<(&'s Name, Span)>) {
        match &self.kind {
            StmtKind::Block(stmts) => stmts.iter().for_each(|s| s.reads(out)),
            StmtKind::If(c, a, b) => {
                c.reads(out);
                a.reads(out);
                b.reads(out);
            }
            StmtKind::NonBlocking { rhs, .. } => rhs.reads(out),
        }
    }
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

impl Expr {
    /// Adds every occurrence of a signal in the expression to `out`.
    pub fn reads<'e>(&'e self, out: &mut Vec<(&'e Name, Span)>) {
        match &self.kind {
            ExprKind::Lit { .. } => {}
            ExprKind::Var(x) => out.push((x, self.span)),
            ExprKind::Unary(_, a) => a.reads(out),
            ExprKind::Binary(_, a, b) => {
                a.reads(out);
                b.reads(out);
            }
            ExprKind::Cond(c, a, b) => {
                c.reads(out);
                a.reads(out);
                b.reads(out);
            }
        }
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

/// SystemVerilog syntax on one line, with every `else` written out, for
/// tests and messages.
impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            StmtKind::Block(stmts) => {
                write!(f, "begin")?;
                for s in stmts {
                    write!(f, " {s}")?;
                }
                write!(f, " end")
            }
            StmtKind::If(c, a, b) => write!(f, "if ({c}) {a} else {b}"),
            StmtKind::NonBlocking { lhs, rhs } => write!(f, "{lhs} <= {rhs};"),
        }
    }
}
