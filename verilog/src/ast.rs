/*!
# The abstract syntax of the accepted SystemVerilog subset

The surface syntax after parsing, before typing. Operators are kept as
written (`>`, `!`, `&&`, `~^`, …); lowering rewrites them into [`SvOp`]s.
The shapes are plain trees without sharing, so that they map one-to-one to
Lean inductive types.

[`SvOp`]: crate::SvOp
*/

use crate::value::BitVec;
use std::collections::BTreeSet;
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

    /// `int`, the type of unsized decimal literals and integer parameters:
    /// with 2-state values, the same as `logic signed [31:0]`.
    pub const INT: Ty = Ty {
        width: 32,
        signed: true,
    };
}

/// The value of a constant expression, with its type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Const {
    pub ty: Ty,
    pub value: BitVec,
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

/// A module with the values of its parameters substituted: the frontend
/// elaborates a module once for each set of parameter values.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Module {
    pub name: Name,
    /// The parameters and local parameters, in order, with their values.
    pub params: Vec<Param>,
    /// The ports of the ANSI header, in order.
    pub ports: Vec<Port>,
    /// The internal signals.
    pub decls: Vec<Decl>,
    pub items: Vec<Item>,
}

impl Module {
    /// The clocks: the signals whose edges clock an `always_ff` block, and
    /// those connected to the clock ports of instances.
    pub fn clocks(&self) -> BTreeSet<&Name> {
        self.items
            .iter()
            .flat_map(|item| match item {
                Item::AlwaysFf { clock, .. } => vec![clock],
                Item::Instance(i) => i.clocks().collect(),
                Item::Assign { .. } | Item::AlwaysComb { .. } => Vec::new(),
            })
            .collect()
    }

    /// The instances, in source order.
    pub fn instances(&self) -> impl Iterator<Item = &Instance> {
        self.items.iter().filter_map(|item| match item {
            Item::Instance(i) => Some(i),
            _ => None,
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

/// `parameter` or `localparam`, with its value.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Param {
    pub name: Name,
    pub ty: Ty,
    pub value: BitVec,
    /// Whether instances can override it.
    pub overridable: bool,
    pub span: Span,
}

/// `module #(.N(v), …) name (.port(actual), …);`
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Instance {
    pub module: Name,
    pub name: Name,
    /// The parameter values given, evaluated in the instantiating module.
    pub params: Vec<Override>,
    pub conns: Vec<Conn>,
    /// The position of the instance name.
    pub span: Span,
}

/// `.N(v)` in `#(…)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Override {
    pub name: Name,
    pub value: Const,
    pub span: Span,
}

/// `.port(actual)`, `.port` (short for `.port(port)`) or `.port()`
/// (unconnected).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Conn {
    pub port: Name,
    pub actual: Option<Expr>,
    pub span: Span,
    /// The port of the instantiated module; filled in by
    /// [`elab`](crate::elab).
    pub formal: Option<Formal>,
}

/// A port of an instantiated module, as seen from the connection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Formal {
    pub dir: Dir,
    pub ty: Ty,
    /// Whether it is a clock of the module (see [`Module::clocks`]).
    pub clock: bool,
    /// Whether the actual is the name of a signal of the port's width, so
    /// that the port is that signal. An actual of another width (or any
    /// other expression) is assigned to the port, which is then a signal
    /// of its own.
    pub shared: bool,
}

impl Conn {
    /// The signal of the instantiating module that the port is (see
    /// [`Formal::shared`]).
    pub fn shared(&self) -> Option<&Name> {
        match (&self.formal, self.actual.as_ref().map(|e| &e.kind)) {
            (Some(f), Some(ExprKind::Var(x))) if f.shared => Some(x),
            _ => None,
        }
    }
}

impl Instance {
    /// The signals connected to clock ports.
    pub fn clocks(&self) -> impl Iterator<Item = &Name> {
        self.conns
            .iter()
            .filter_map(|c| match (&c.formal, &c.actual) {
                (Some(f), Some(_)) if f.clock => c.shared(),
                _ => None,
            })
    }

    /// The connections of ports with direction `dir` (clock ports
    /// included), with the actual if there is one.
    pub fn conns(&self, dir: Dir) -> impl Iterator<Item = (&Conn, &Formal)> {
        self.conns.iter().filter_map(move |c| match &c.formal {
            Some(f) if f.dir == dir => Some((c, f)),
            _ => None,
        })
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Item {
    /// `assign lhs = rhs;`, with `span` the position of `lhs`.
    Assign { lhs: Lvalue, rhs: Expr, span: Span },
    /// `always_ff @(posedge clock) body` or `always_ff @(negedge clock)
    /// body`, with `span` the position of `always_ff` and `clock_span` that
    /// of `clock`. With an asynchronous reset, `@(posedge clock or posedge
    /// rst) if (rst) … else …` (or `negedge rst_n` and `if (!rst_n)`): the
    /// body is the `if` as written, see [`Stmt::reset_values`].
    AlwaysFf {
        clock: Name,
        edge: Edge,
        clock_span: Span,
        reset: Option<Reset>,
        body: Stmt,
        span: Span,
    },
    /// `always_comb body`, with `span` the position of `always_comb`.
    AlwaysComb { body: Stmt, span: Span },
    /// A module instance.
    Instance(Instance),
}

impl Item {
    /// The signals the item drives, in the order of their first
    /// assignment.
    pub fn targets(&self) -> Vec<&Name> {
        match self {
            Item::Assign { lhs, .. } => {
                let mut out = Vec::new();
                lhs.targets(&mut out);
                out
            }
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => {
                let mut out = Vec::new();
                body.targets(&mut out);
                out
            }
            // the signals connected to outputs
            Item::Instance(i) => i
                .conns(Dir::Output)
                .filter_map(|(c, _)| match c.actual.as_ref().map(|e| &e.kind) {
                    Some(ExprKind::Var(y)) => Some(y),
                    _ => None,
                })
                .collect(),
        }
    }

    /// The body of an `always_ff` or `always_comb` block.
    pub fn body(&self) -> Option<&Stmt> {
        match self {
            Item::Assign { .. } | Item::Instance(_) => None,
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => Some(body),
        }
    }

    /// The position of the item: of `lhs` for an `assign`, of the name of
    /// an instance, else of the keyword.
    pub fn span(&self) -> Span {
        match self {
            Item::Assign { span, .. }
            | Item::AlwaysFf { span, .. }
            | Item::AlwaysComb { span, .. } => *span,
            Item::Instance(i) => i.span,
        }
    }

    /// The signals the item reads, with the position of every read.
    pub fn reads(&self) -> Vec<(&Name, Span)> {
        let mut out = Vec::new();
        match self {
            Item::Assign { lhs, rhs, .. } => {
                rhs.reads(&mut out);
                lhs.reads(&mut out);
            }
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => body.reads(&mut out),
            // the actuals of inputs, clocks included
            Item::Instance(i) => {
                for (c, _) in i.conns(Dir::Input) {
                    if let Some(e) = &c.actual {
                        e.reads(&mut out);
                    }
                }
            }
        }
        out
    }
}

/// The edge of a clock that an `always_ff` block runs at.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Edge {
    /// The rising edge: the clock goes from 0 to 1.
    Pos,
    /// The falling edge: from 1 to 0.
    Neg,
}

/// The asynchronous reset of an `always_ff` block: the event `posedge rst`
/// (active high, `edge` is [`Edge::Pos`]) or `negedge rst_n` (active low),
/// with `span` the position of the name in the event list.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Reset {
    pub name: Name,
    pub edge: Edge,
    pub span: Span,
}

/// SystemVerilog syntax: `posedge`, `negedge`.
impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Edge::Pos => write!(f, "posedge"),
            Edge::Neg => write!(f, "negedge"),
        }
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
    /// `case (sel) labels: stmt … default: stmt endcase`: the first item
    /// with a label equal to `sel` runs, else `default` (if any). The
    /// `default` can be written anywhere among the items.
    Case {
        sel: Expr,
        items: Vec<CaseItem>,
        default: Option<Box<Stmt>>,
    },
    /// `lhs <= rhs;`, with `span` the position of `lhs`.
    NonBlocking { lhs: Lvalue, rhs: Expr },
    /// `lhs = rhs;`, with `span` the position of `lhs`.
    Blocking { lhs: Lvalue, rhs: Expr },
}

/// The target of an assignment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Lvalue {
    pub kind: LvalueKind,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LvalueKind {
    Var(Name),
    /// `y[…]`; a variable index only in a block (not in `assign`).
    Select(Name, Select),
    /// `{l1, l2, …}`: `l1` takes the high bits of the value.
    Concat(Vec<Lvalue>),
}

impl Lvalue {
    /// Adds the signals assigned to `out`, each once.
    fn targets<'l>(&'l self, out: &mut Vec<&'l Name>) {
        match &self.kind {
            LvalueKind::Var(y) | LvalueKind::Select(y, _) => {
                if !out.contains(&y) {
                    out.push(y);
                }
            }
            LvalueKind::Concat(ls) => ls.iter().for_each(|l| l.targets(out)),
        }
    }

    /// Adds the signals read by the indices of selects to `out`.
    pub fn reads<'l>(&'l self, out: &mut Vec<(&'l Name, Span)>) {
        match &self.kind {
            LvalueKind::Var(_) => {}
            LvalueKind::Select(_, sel) => sel.reads(out),
            LvalueKind::Concat(ls) => ls.iter().for_each(|l| l.reads(out)),
        }
    }
}

/// `l1, l2, …: body` in a `case`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CaseItem {
    pub labels: Vec<Expr>,
    pub body: Stmt,
}

impl Stmt {
    pub fn empty(span: Span) -> Self {
        Stmt {
            kind: StmtKind::Block(Vec::new()),
            span,
        }
    }

    /// The statements directly inside `self`, in source order (`default`
    /// last).
    pub fn children(&self) -> Vec<&Stmt> {
        match &self.kind {
            StmtKind::Block(stmts) => stmts.iter().collect(),
            StmtKind::If(_, a, b) => vec![a, b],
            StmtKind::Case { items, default, .. } => items
                .iter()
                .map(|i| &i.body)
                .chain(default.as_deref())
                .collect(),
            StmtKind::NonBlocking { .. } | StmtKind::Blocking { .. } => Vec::new(),
        }
    }

    /// The assignments of the reset branch of the body of an `always_ff`
    /// block with an asynchronous reset: the `then` branch of the `if` the
    /// body consists of (possibly inside `begin … end`), in source order.
    /// Typing makes sure it holds only assignments of constants to whole
    /// registers, so other statements are skipped.
    pub fn reset_values(&self) -> Vec<(&Name, &Expr)> {
        fn walk<'s>(s: &'s Stmt, out: &mut Vec<(&'s Name, &'s Expr)>) {
            match &s.kind {
                StmtKind::Block(stmts) => stmts.iter().for_each(|s| walk(s, out)),
                StmtKind::NonBlocking { lhs, rhs } => {
                    if let LvalueKind::Var(q) = &lhs.kind {
                        out.push((q, rhs));
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        if let Some(branch) = self.reset_branch() {
            walk(branch, &mut out);
        }
        out
    }

    /// The `then` branch of the `if` that the body of a block with an
    /// asynchronous reset consists of.
    pub fn reset_branch(&self) -> Option<&Stmt> {
        match &self.kind {
            StmtKind::If(_, then, _) => Some(then),
            StmtKind::Block(stmts) if stmts.len() == 1 => stmts[0].reset_branch(),
            _ => None,
        }
    }

    /// Adds the signals assigned in the statement to `out`, each once, in
    /// the order of their first assignment.
    fn targets<'s>(&'s self, out: &mut Vec<&'s Name>) {
        match &self.kind {
            StmtKind::NonBlocking { lhs, .. } | StmtKind::Blocking { lhs, .. } => lhs.targets(out),
            _ => self.children().into_iter().for_each(|s| s.targets(out)),
        }
    }

    /// Adds every occurrence of a signal in an expression of the statement
    /// to `out`, in source order.
    fn reads<'s>(&'s self, out: &mut Vec<(&'s Name, Span)>) {
        match &self.kind {
            StmtKind::Block(stmts) => stmts.iter().for_each(|s| s.reads(out)),
            StmtKind::If(c, a, b) => {
                c.reads(out);
                a.reads(out);
                b.reads(out);
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                sel.reads(out);
                for item in items {
                    item.labels.iter().for_each(|l| l.reads(out));
                    item.body.reads(out);
                }
                if let Some(d) = default {
                    d.reads(out);
                }
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                rhs.reads(out);
                lhs.reads(out);
            }
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
    /// Sizing also makes one for `'0` and `'1` at the width of their
    /// context.
    Lit {
        ty: Ty,
        value: BitVec,
    },
    /// A parameter or an unsized literal (`5` is an `int`, `'hff` 32 bits
    /// unsigned), with its value; `text` is the source text. Sizing folds
    /// an operator expression on constants (`N - 1`) into one constant at
    /// the type of its context, with the expression as its text.
    Const {
        text: String,
        ty: Ty,
        value: BitVec,
    },
    Var(Name),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// `c ? a : b`
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `{a, b, …}`: `a` gives the high bits.
    Concat(Vec<Expr>),
    /// `{n{a, b, …}}`: `n` copies of `{a, b, …}`.
    Repl(usize, Vec<Expr>),
    /// `x[…]`: selects apply to signals only.
    Select(Name, Select),
    /// `N'(e)`, `signed'(e)`, `unsigned'(e)`, `$signed(e)`, `$unsigned(e)`.
    Cast(Cast, Box<Expr>),
    /// `'0` (false) or `'1` (true): every bit 0 or 1, at the width of the
    /// context (1 bit where the width is self-determined). Sizing replaces
    /// it by a literal.
    Fill(bool),
    /// An implicit resize, made by [`sizing`](crate::sizing): the value of
    /// the expression, truncated to the width of the type, or extended to
    /// it with copies of its top bit if the type is signed and with zeros
    /// if not, taking the signedness of the type.
    Resize(Ty, Box<Expr>),
}

/// The bits picked by `x[…]`, as written.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Select {
    /// `x[i]`
    Bit(Index),
    /// `x[hi:lo]`, with constant bounds.
    Part { hi: usize, lo: usize },
    /// `x[base +: width]`: bits `base+width-1` down to `base`.
    Up(Index, usize),
    /// `x[base -: width]`: bits `base` down to `base-width+1`.
    Down(Index, usize),
}

/// A bit position in a select: a decimal constant or any expression.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Index {
    Const(usize),
    Expr(Box<Expr>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cast {
    /// `N'(e)`: resize to `N` bits.
    Size(usize),
    /// `signed'(e)`, `$signed(e)`
    Signed,
    /// `unsigned'(e)`, `$unsigned(e)`
    Unsigned,
}

impl Select {
    /// The bits `(hi, lo)` the select picks if they do not depend on a
    /// signal (`None` if `lo` would be negative).
    pub fn constant(&self) -> Option<Option<(usize, usize)>> {
        match self {
            Select::Bit(Index::Const(i)) => Some(Some((*i, *i))),
            Select::Part { hi, lo } => Some(Some((*hi, *lo))),
            Select::Up(Index::Const(b), w) => Some(Some((b + w - 1, *b))),
            Select::Down(Index::Const(b), w) => Some((b + 1).checked_sub(*w).map(|lo| (*b, lo))),
            Select::Bit(Index::Expr(_)) | Select::Up(..) | Select::Down(..) => None,
        }
    }

    /// The index expression, if it is not a constant.
    pub fn index(&self) -> Option<&Expr> {
        match self {
            Select::Bit(Index::Expr(e))
            | Select::Up(Index::Expr(e), _)
            | Select::Down(Index::Expr(e), _) => Some(e),
            _ => None,
        }
    }

    /// The number of bits picked.
    pub fn width(&self) -> usize {
        match self {
            Select::Bit(_) => 1,
            Select::Part { hi, lo } => (hi + 1).saturating_sub(*lo),
            Select::Up(_, w) | Select::Down(_, w) => *w,
        }
    }

    fn reads<'s>(&'s self, out: &mut Vec<(&'s Name, Span)>) {
        if let Some(e) = self.index() {
            e.reads(out);
        }
    }
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
    /// `/`: sizing requires a nonzero constant divisor.
    Div,
    /// `%`: sizing requires a nonzero constant divisor.
    Mod,
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
            Div => "/",
            Mod => "%",
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
            ExprKind::Lit { .. } | ExprKind::Const { .. } => {}
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
            ExprKind::Concat(es) | ExprKind::Repl(_, es) => es.iter().for_each(|e| e.reads(out)),
            ExprKind::Select(x, sel) => {
                out.push((x, self.span));
                sel.reads(out);
            }
            ExprKind::Cast(_, e) | ExprKind::Resize(_, e) => e.reads(out),
            ExprKind::Fill(_) => {}
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
            ExprKind::Const { text, .. } => write!(f, "{text}"),
            ExprKind::Var(x) => write!(f, "{x}"),
            ExprKind::Unary(op, e) => write!(f, "({op}{e})"),
            ExprKind::Binary(op, a, b) => write!(f, "({a} {op} {b})"),
            ExprKind::Cond(c, a, b) => write!(f, "({c} ? {a} : {b})"),
            ExprKind::Concat(es) => write!(f, "{{{}}}", list(es)),
            ExprKind::Repl(n, es) => write!(f, "{{{n}{{{}}}}}", list(es)),
            ExprKind::Select(x, sel) => write!(f, "{x}{sel}"),
            ExprKind::Cast(Cast::Size(n), e) => write!(f, "{n}'({e})"),
            ExprKind::Cast(Cast::Signed, e) => write!(f, "signed'({e})"),
            ExprKind::Cast(Cast::Unsigned, e) => write!(f, "unsigned'({e})"),
            ExprKind::Fill(b) => write!(f, "'{}", u8::from(*b)),
            // the expression as written: messages show source text
            ExprKind::Resize(_, e) => write!(f, "{e}"),
        }
    }
}

fn list(items: &[impl fmt::Display]) -> String {
    items
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Index::Const(i) => write!(f, "{i}"),
            Index::Expr(e) => write!(f, "{e}"),
        }
    }
}

impl fmt::Display for Select {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Select::Bit(i) => write!(f, "[{i}]"),
            Select::Part { hi, lo } => write!(f, "[{hi}:{lo}]"),
            Select::Up(b, w) => write!(f, "[{b} +: {w}]"),
            Select::Down(b, w) => write!(f, "[{b} -: {w}]"),
        }
    }
}

impl fmt::Display for Lvalue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LvalueKind::Var(y) => write!(f, "{y}"),
            LvalueKind::Select(y, sel) => write!(f, "{y}{sel}"),
            LvalueKind::Concat(ls) => write!(f, "{{{}}}", list(ls)),
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
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                write!(f, "case ({sel})")?;
                for item in items {
                    let labels: Vec<String> = item.labels.iter().map(|l| l.to_string()).collect();
                    write!(f, " {}: {}", labels.join(", "), item.body)?;
                }
                if let Some(d) = default {
                    write!(f, " default: {d}")?;
                }
                write!(f, " endcase")
            }
            StmtKind::NonBlocking { lhs, rhs } => write!(f, "{lhs} <= {rhs};"),
            StmtKind::Blocking { lhs, rhs } => write!(f, "{lhs} = {rhs};"),
        }
    }
}
