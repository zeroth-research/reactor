//! Expressions: operator chains re-associated by precedence, operands, selects, literals.

use super::{Cx, Result, system_call};
use crate::ast::{BinOp, Cast, Error, Expr, ExprKind, Index, Name, Select, Span, Ty, UnOp};
use crate::constant::{self, Failure, Literal, X_HINT};
use crate::value::BitVec;
use sv_parser as sv;
use sv_parser::RefNodes;

/// An expression of the grammar, or a constant expression, which the
/// grammar keeps apart; both become an [`Expr`].
#[derive(Clone, Copy)]
pub(super) enum Node<'a> {
    Expr(&'a sv::Expression),
    Const(&'a sv::ConstantExpression),
}

/// Applies the operator `op` to the last two expressions of `out`.
pub(super) fn reduce(out: &mut Vec<Expr>, op: Op) {
    let b = Box::new(out.pop().expect("operand"));
    let a = Box::new(out.pop().expect("operand"));
    let (kind, span) = match op {
        Op::Bin(op, span) => (ExprKind::Binary(op, a, b), span),
        Op::Cond(then, span) => (ExprKind::Cond(a, Box::new(then), b), span),
    };
    out.push(Expr { kind, span });
}

/// An operator of a flattened chain: a binary operator, or `? then :` of a
/// conditional, which acts as a right-associative binary operator between
/// condition and else branch.
pub(super) enum Op {
    Bin(BinOp, Span),
    Cond(Expr, Span),
}

/// The binding strength of an operator (higher binds tighter), after
/// IEEE 1800-2017 Table 11-2 (`**` binds tighter still).
pub(super) fn prec(op: Option<BinOp>) -> u8 {
    use BinOp::*;
    match op {
        Some(Pow) => 11,
        Some(Mul | Div | Mod) => 10,
        Some(Add | Sub) => 9,
        Some(Shl | Shr | AShl | AShr) => 8,
        Some(Lt | Le | Gt | Ge) => 7,
        Some(Eq | Ne | CaseEq | CaseNe) => 6,
        Some(BitAnd) => 5,
        Some(BitXor | BitXnor) => 4,
        Some(BitOr) => 3,
        Some(LogAnd) => 2,
        Some(LogOr) => 1,
        None => 0, // ?:
    }
}

impl Op {
    fn prec(&self) -> u8 {
        match self {
            Op::Bin(op, _) => prec(Some(*op)),
            Op::Cond(..) => prec(None),
        }
    }
}

/// An unsized literal (or `'0`, `'1`) that gives `e` its width, if there is
/// one: a leaf reached through the operands that take the width of their
/// context (arithmetic and bitwise operands, the left operand of a shift
/// and the base of `**`, the arms of `?:`, sign casts). IEEE 1800 forbids such an expression where
/// its width counts, as in a concatenation (§11.4.12): an unsized literal
/// has no definite width. Parameters have a type and do not count, nor do
/// literals in self-determined places (`{a == 1}`, `{a << 1}`).
pub(super) fn unsized_width(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Const {
            is_unsized: true, ..
        }
        | ExprKind::Fill(_) => Some(e),
        ExprKind::Unary(UnOp::BitNot | UnOp::Neg | UnOp::Plus, a) => unsized_width(a),
        ExprKind::Binary(op, a, b) => match op {
            BinOp::Shl | BinOp::Shr | BinOp::AShl | BinOp::AShr | BinOp::Pow => unsized_width(a),
            BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Mod
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::BitXnor => unsized_width(a).or_else(|| unsized_width(b)),
            _ => None,
        },
        ExprKind::Cond(_, a, b) => unsized_width(a).or_else(|| unsized_width(b)),
        ExprKind::Cast(Cast::Signed | Cast::Unsigned, a) => unsized_width(a),
        _ => None,
    }
}

impl Cx<'_> {
    /// The select `[…]` after a signal name in an expression or a
    /// procedural assignment target, if any.
    pub(super) fn select(&self, select: &sv::Select) -> Result<Option<Select>> {
        let (member, bits, part) = &select.nodes;
        self.absent(member, "member select")?;
        let bits = &bits.nodes.0;
        Ok(match (bits.as_slice(), part) {
            ([], None) => None,
            ([b], None) => Some(Select::Bit(self.index(&b.nodes.1)?)),
            ([], Some(p)) => Some(self.part(&p.nodes.1)?),
            _ => return self.err(select, "multi-dimensional selects not supported"),
        })
    }

    /// `x[hi:lo]`, `x[b +: w]` or `x[b -: w]`.
    pub(super) fn part(&self, p: &sv::PartSelectRange) -> Result<Select> {
        Ok(match p {
            sv::PartSelectRange::ConstantRange(r) => {
                let (hi, _, lo) = &r.nodes;
                Select::Part {
                    hi: self.const_int(hi)?,
                    lo: self.const_int(lo)?,
                }
            }
            sv::PartSelectRange::IndexedRange(r) => {
                let (base, op, width) = &r.nodes;
                self.indexed(self.index(base)?, op, self.const_int(width)?)
            }
        })
    }

    /// `[base +: width]` or `[base -: width]`.
    pub(super) fn indexed(&self, base: Index, op: &sv::Symbol, width: usize) -> Select {
        if self.text(op) == "+:" {
            Select::Up(base, width)
        } else {
            Select::Down(base, width)
        }
    }

    /// A bit position: an integer constant (see [`constant::exact`]), or any
    /// expression.
    pub(super) fn index(&self, e: &sv::Expression) -> Result<Index> {
        let e = self.expr(e)?;
        match constant::exact(&e) {
            Some(Ok(value)) if value < 0 => {
                Err(Error::new(e.span, format!("negative index {value}")))
            }
            Some(Ok(value)) => Ok(Index::Const(value as usize)),
            Some(Err(msg)) => Err(Error::new(e.span, msg)),
            None => Ok(Index::Expr(Box::new(e))),
        }
    }

    /// A hierarchical identifier that is a plain name.
    pub(super) fn simple_hierarchical<'h>(
        &self,
        h: &'h sv::HierarchicalIdentifier,
    ) -> Result<&'h sv::Identifier> {
        let (root, path, id) = &h.nodes;
        if root.is_some() || !path.is_empty() {
            return self.err(h, "hierarchical names not supported");
        }
        Ok(id)
    }

    pub(super) fn expr(&self, e: &sv::Expression) -> Result<Expr> {
        let mut operands = Vec::new();
        let mut ops = Vec::new();
        self.flatten(Node::Expr(e), &mut operands, &mut ops)?;
        Ok(self.reassociate(operands, ops))
    }

    /// A constant expression of the grammar (a range bound, a parameter
    /// value, …), built like any other expression: parameters are
    /// constants, other names stay signals, which [`constant::value`]
    /// rejects.
    pub(super) fn const_expression(&self, e: &sv::ConstantExpression) -> Result<Expr> {
        let mut operands = Vec::new();
        let mut ops = Vec::new();
        self.flatten(Node::Const(e), &mut operands, &mut ops)?;
        Ok(self.reassociate(operands, ops))
    }

    /// Flattens a chain of binary operators and conditionals into its
    /// operands and operators, in source order.
    pub(super) fn flatten(
        &self,
        e: Node,
        operands: &mut Vec<Expr>,
        ops: &mut Vec<Op>,
    ) -> Result<()> {
        match e {
            Node::Expr(sv::Expression::Binary(b)) => {
                let (l, op, attrs, r) = &b.nodes;
                self.absent(attrs, "attribute")?;
                self.flatten(Node::Expr(l), operands, ops)?;
                ops.push(self.binop(op)?);
                self.flatten(Node::Expr(r), operands, ops)
            }
            Node::Const(sv::ConstantExpression::Binary(b)) => {
                let (l, op, attrs, r) = &b.nodes;
                self.absent(attrs, "attribute")?;
                self.flatten(Node::Const(l), operands, ops)?;
                ops.push(self.binop(op)?);
                self.flatten(Node::Const(r), operands, ops)
            }
            Node::Expr(sv::Expression::ConditionalExpression(c)) => {
                let (pred, q, attrs, then, _, els) = &c.nodes;
                self.absent(attrs, "attribute")?;
                let pred = match pred.nodes.0.contents().as_slice() {
                    [sv::ExpressionOrCondPattern::Expression(p)] => p.as_ref(),
                    _ => return self.err(pred, "`&&&` and patterns not supported"),
                };
                self.flatten(Node::Expr(pred), operands, ops)?;
                ops.push(Op::Cond(self.expr(then)?, self.span(q)));
                self.flatten(Node::Expr(els), operands, ops)
            }
            Node::Const(sv::ConstantExpression::Ternary(c)) => {
                let (pred, q, attrs, then, _, els) = &c.nodes;
                self.absent(attrs, "attribute")?;
                self.flatten(Node::Const(pred), operands, ops)?;
                ops.push(Op::Cond(self.const_expression(then)?, self.span(q)));
                self.flatten(Node::Const(els), operands, ops)
            }
            e => {
                operands.push(self.operand(e)?);
                Ok(())
            }
        }
    }

    /// An expression that is not a binary operation or a conditional.
    pub(super) fn operand(&self, e: Node) -> Result<Expr> {
        match e {
            Node::Expr(sv::Expression::Primary(p)) => self.primary(p),
            Node::Expr(sv::Expression::Unary(u)) => {
                let (op, attrs, p) = &u.nodes;
                self.unary(op, attrs, p, || self.primary(p))
            }
            Node::Expr(sv::Expression::IncOrDecExpression(e)) => {
                self.err(e.as_ref(), "`++`/`--` not supported")
            }
            Node::Expr(sv::Expression::OperatorAssignment(e)) => {
                self.err(e.as_ref(), "assignment in expression not supported")
            }
            Node::Expr(e) => self.unsupported(e),
            Node::Const(sv::ConstantExpression::ConstantPrimary(p)) => self.const_primary(p),
            Node::Const(sv::ConstantExpression::Unary(u)) => {
                let (op, attrs, p) = &u.nodes;
                self.unary(op, attrs, p, || self.const_primary(p))
            }
            Node::Const(e) => self.unsupported(e),
        }
    }

    /// `op p`, with `operand` building `p`. `-2147483648` is the least
    /// `int`: the literal 2147483648 is no `int` by itself, but negated at
    /// once it is one constant.
    pub(super) fn unary<'a>(
        &self,
        op: &sv::UnaryOperator,
        attrs: &'a Vec<sv::AttributeInstance>,
        p: impl Into<RefNodes<'a>>,
        operand: impl FnOnce() -> Result<Expr>,
    ) -> Result<Expr> {
        self.absent(attrs, "attribute")?;
        let span = self.span(op);
        let op = self.unop(op)?;
        let digits: String = self
            .text(p)
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '_')
            .collect();
        if op == UnOp::Neg && digits == "2147483648" {
            let kind = ExprKind::Const {
                text: format!("-{digits}"),
                ty: Ty::INT,
                value: BitVec::from_signed(32, i32::MIN),
                is_unsized: true,
            };
            return Ok(Expr { kind, span });
        }
        let kind = ExprKind::Unary(op, Box::new(operand()?));
        Ok(Expr { kind, span })
    }

    /// A parameter `id` as a constant, or else the signal `id`.
    pub(super) fn name(&self, id: Name, span: Span) -> Expr {
        let kind = match self.params.borrow().get(&id) {
            Some(c) => ExprKind::Const {
                text: id,
                ty: c.ty,
                value: c.value.clone(),
                is_unsized: false,
            },
            None => ExprKind::Var(id),
        };
        Expr { kind, span }
    }

    pub(super) fn primary(&self, p: &sv::Primary) -> Result<Expr> {
        let span = self.span(p);
        let kind = match p {
            sv::Primary::PrimaryLiteral(l) => return self.primary_literal(l, p),
            sv::Primary::Hierarchical(h) => {
                let (scope, id, select) = &h.nodes;
                self.absent(scope, "package or class scope")?;
                let id = self.identifier(self.simple_hierarchical(id)?)?;
                if self.is_array(&id) {
                    let elem = self.element(id, select, span)?;
                    return Ok(Expr {
                        kind: ExprKind::Elem(Box::new(elem)),
                        span,
                    });
                }
                let param = self.params.borrow().contains_key(&id);
                match (self.select(select)?, param) {
                    (None, _) => return Ok(self.name(id, span)),
                    (Some(_), true) => {
                        return self.err(p, format!("select of parameter `{id}` not supported"));
                    }
                    (Some(sel), false) => ExprKind::Select(id, sel),
                }
            }
            sv::Primary::MintypmaxExpression(m) => match &m.nodes.0.nodes.1 {
                sv::MintypmaxExpression::Expression(e) => return self.expr(e),
                m => return self.err(m, "min:typ:max expression not supported"),
            },
            sv::Primary::Concatenation(c) => {
                let (c, select) = &c.nodes;
                self.absent(select, "select of a concatenation")?;
                let items = c.nodes.0.nodes.1.contents();
                let items = items.into_iter().map(|e| self.expr(e));
                ExprKind::Concat(self.concatenation(items.collect::<Result<_>>()?)?)
            }
            sv::Primary::MultipleConcatenation(c) => {
                let (c, select) = &c.nodes;
                self.absent(select, "select of a concatenation")?;
                let (n, c) = &c.nodes.0.nodes.1;
                let n = self.int_of(n, &self.expr(n)?)?;
                if n == 0 {
                    return self.err(p, "replication count 0 not supported");
                }
                let items = c.nodes.0.nodes.1.contents();
                let items = items.into_iter().map(|e| self.expr(e));
                ExprKind::Repl(n, self.concatenation(items.collect::<Result<_>>()?)?)
            }
            sv::Primary::Cast(c) => {
                let (ty, _, e) = &c.nodes;
                ExprKind::Cast(self.cast(ty, p)?, Box::new(self.expr(&e.nodes.1)?))
            }
            sv::Primary::FunctionSubroutineCall(c) => return self.call(&c.nodes.0, span),
            p => return self.unsupported(p),
        };
        Ok(Expr { kind, span })
    }

    /// A primary of a constant expression; see [`Cx::const_expression`].
    pub(super) fn const_primary(&self, p: &sv::ConstantPrimary) -> Result<Expr> {
        let span = self.span(p);
        let kind = match p {
            sv::ConstantPrimary::PrimaryLiteral(l) => return self.primary_literal(l, p),
            sv::ConstantPrimary::PsParameter(x) => {
                let (id, select) = &x.nodes;
                let id = match id {
                    sv::PsParameterIdentifier::Scope(s) => {
                        let (scope, id) = &s.nodes;
                        self.absent(scope, "package or class scope")?;
                        &id.nodes.0
                    }
                    id => return self.unsupported(id),
                };
                return self.const_name(self.identifier(id)?, select, p);
            }
            sv::ConstantPrimary::GenvarIdentifier(id) => {
                return Ok(self.name(self.identifier(&id.nodes.0)?, span));
            }
            sv::ConstantPrimary::FormalPort(x) => {
                let (id, select) = &x.nodes;
                return self.const_name(self.identifier(&id.nodes.0)?, select, p);
            }
            sv::ConstantPrimary::MintypmaxExpression(m) => match &m.nodes.0.nodes.1 {
                sv::ConstantMintypmaxExpression::Unary(e) => return self.const_expression(e),
                m => return self.err(m, "min:typ:max expression not supported"),
            },
            sv::ConstantPrimary::Concatenation(c) => {
                let (c, select) = &c.nodes;
                self.absent(select, "select of a concatenation")?;
                let items = c.nodes.0.nodes.1.contents();
                let items = items.into_iter().map(|e| self.const_expression(e));
                ExprKind::Concat(self.concatenation(items.collect::<Result<_>>()?)?)
            }
            sv::ConstantPrimary::MultipleConcatenation(c) => {
                let (c, select) = &c.nodes;
                self.absent(select, "select of a concatenation")?;
                let (n, c) = &c.nodes.0.nodes.1;
                let n = self.const_int(n)?;
                if n == 0 {
                    return self.err(p, "replication count 0 not supported");
                }
                let items = c.nodes.0.nodes.1.contents();
                let items = items.into_iter().map(|e| self.const_expression(e));
                ExprKind::Repl(n, self.concatenation(items.collect::<Result<_>>()?)?)
            }
            sv::ConstantPrimary::ConstantCast(c) => {
                let (ty, _, e) = &c.nodes;
                let e = self.const_expression(&e.nodes.1)?;
                ExprKind::Cast(self.cast(ty, p)?, Box::new(e))
            }
            sv::ConstantPrimary::Dollar(_) => {
                return self.err(p, "`$` not supported in constant expressions");
            }
            sv::ConstantPrimary::ConstantFunctionCall(c) => {
                // a plain name may parse as a call without arguments
                if let sv::SubroutineCall::TfCall(t) = &c.nodes.0.nodes.0
                    && let (id, attrs, None) = &t.nodes
                    && attrs.is_empty()
                {
                    let name = self.text(id);
                    let mut chars = name.chars();
                    if chars
                        .next()
                        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
                    {
                        return Ok(self.name(name, span));
                    }
                }
                return self.call(&c.nodes.0.nodes.0, span);
            }
            p => return self.unsupported(p),
        };
        Ok(Expr { kind, span })
    }

    /// The name `id` with the select `select` in a constant expression: a
    /// parameter, or a signal, which [`constant::value`] rejects.
    pub(super) fn const_name(
        &self,
        id: Name,
        select: &sv::ConstantSelect,
        p: &sv::ConstantPrimary,
    ) -> Result<Expr> {
        let selected = !self.text(select).is_empty();
        if selected && self.params.borrow().contains_key(&id) {
            return self.err(p, format!("select of parameter `{id}` not supported"));
        }
        Ok(self.name(id, self.span(p)))
    }

    /// The kind of a cast `t'(…)` at `p`: a size cast, or a sign cast.
    pub(super) fn cast<'a>(
        &self,
        t: &sv::CastingType,
        p: impl Into<RefNodes<'a>> + Copy,
    ) -> Result<Cast> {
        Ok(match t {
            sv::CastingType::ConstantPrimary(n) => {
                let n = self.int_of(n.as_ref(), &self.const_primary(n)?)?;
                if n == 0 {
                    return self.err(p, "cast to 0 bits");
                }
                Cast::Size(n)
            }
            sv::CastingType::Signing(s) if self.text(s.as_ref()) == "signed" => Cast::Signed,
            sv::CastingType::Signing(_) => Cast::Unsigned,
            // `W'(e)` parses as a cast to a type named `W`
            sv::CastingType::SimpleType(t)
                if self.params.borrow().contains_key(&self.text(t.as_ref())) =>
            {
                let size = self.name(self.text(t.as_ref()), self.span(t.as_ref()));
                let n = self.int_of(t.as_ref(), &size)?;
                if n == 0 {
                    return self.err(p, "cast to 0 bits");
                }
                Cast::Size(n)
            }
            t => {
                return self.err(t, format!("cast to `{}` not supported", self.text(t)));
            }
        })
    }

    /// The items of `{a, b, …}`. An unsized literal is no item: IEEE 1800
    /// forbids it (simulators give it 32 bits, or 1 for `'1`).
    pub(super) fn concatenation(&self, items: Vec<Expr>) -> Result<Vec<Expr>> {
        for e in &items {
            if let Some(lit) = unsized_width(e) {
                let what = if lit.span == e.span {
                    format!("unsized literal `{e}` in a concatenation")
                } else {
                    format!(
                        "`{e}` in a concatenation has the width of the unsized literal `{lit}` \
                         (32 bits or more)"
                    )
                };
                return Err(Error::new(
                    lit.span,
                    format!("{what}: write its width (e.g. `8'd1`)"),
                ));
            }
        }
        Ok(items)
    }

    /// A call; only `$signed(e)` and `$unsigned(e)`, which are casts, and
    /// `$clog2(n)` of a constant, which is folded into an integer constant.
    pub(super) fn call(&self, c: &sv::SubroutineCall, span: Span) -> Result<Expr> {
        let unsupported = || self.err(c, format!("call `{}` not supported", self.text(c)));
        let sv::SubroutineCall::SystemTfCall(call) = c else {
            return unsupported();
        };
        let Some((name, args)) = system_call(call) else {
            return unsupported();
        };
        let name = self.text(name);
        let [Some(e)] = args.as_slice() else {
            return self.err(c, format!("`{name}` takes one argument"));
        };
        let cast = match name.as_str() {
            "$signed" => Cast::Signed,
            "$unsigned" => Cast::Unsigned,
            "$clog2" => {
                let arg = self.expr(e)?;
                let n = constant::value(&arg).or_else(|f| match f {
                    Failure::NotConstant(why) => {
                        self.err(c, format!("`$clog2` of a constant only: {why}"))
                    }
                    Failure::Error(e) => Err(e),
                })?;
                let value = constant::clog2(&n).map_err(|m| Error::new(span, m))?;
                let kind = ExprKind::Const {
                    text: format!("$clog2({arg})"),
                    ty: Ty::INT,
                    value: BitVec::from_signed(32, value),
                    is_unsized: false,
                };
                return Ok(Expr { kind, span });
            }
            _ => return unsupported(),
        };
        let kind = ExprKind::Cast(cast, Box::new(self.expr(e)?));
        Ok(Expr { kind, span })
    }

    /// A literal of a primary `p`: a number, `'0` or `'1`.
    pub(super) fn primary_literal<'a>(
        &self,
        l: &sv::PrimaryLiteral,
        p: impl Into<RefNodes<'a>> + Copy,
    ) -> Result<Expr> {
        let span = self.span(p);
        match l {
            sv::PrimaryLiteral::Number(n) => match n.as_ref() {
                sv::Number::IntegralNumber(n) => self.literal(n),
                n => self.err(n, "real numbers not supported"),
            },
            sv::PrimaryLiteral::UnbasedUnsizedLiteral(l) => {
                let kind = match self.text(l.as_ref()).trim() {
                    "'0" => ExprKind::Fill(false),
                    "'1" => ExprKind::Fill(true),
                    t => {
                        return self.err(
                            p,
                            format!("literal `{t}` with x/z digits not supported ({X_HINT})"),
                        );
                    }
                };
                Ok(Expr { kind, span })
            }
            l => self.unsupported(l),
        }
    }

    /// A literal: sized `N'[s]<base><digits>`, without x/z digits (values
    /// are 2-state) and with a value that fits in `N` bits, or unsized: a
    /// decimal, which is an integer, or based (`'hff`, `'sd5`).
    pub(super) fn literal(&self, n: &sv::IntegralNumber) -> Result<Expr> {
        let text: String = self
            .text(n)
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '_')
            .collect();
        let span = self.span(n);
        let kind = match constant::literal(&text).map_err(|m| Error::new(span, m))? {
            Literal::Sized(ty, value) => ExprKind::Lit { ty, value },
            Literal::Int(value) => ExprKind::Const {
                text,
                ty: Ty::INT,
                value: BitVec::from_signed(32, value),
                is_unsized: true,
            },
            // `'sd5` is an integer, `'d5` a 32-bit unsigned vector
            Literal::Unsized(ty, value) => ExprKind::Const {
                text,
                ty,
                value,
                is_unsized: true,
            },
        };
        Ok(Expr { kind, span })
    }

    /// Builds the tree of a flattened chain by precedence (shunting-yard).
    /// Binary operators associate to the left, `?:` to the right.
    pub(super) fn reassociate(&self, operands: Vec<Expr>, ops: Vec<Op>) -> Expr {
        let mut operands = operands.into_iter();
        let mut out = vec![operands.next().expect("operand")];
        let mut stack: Vec<Op> = Vec::new();
        for (op, operand) in ops.into_iter().zip(operands) {
            let right_assoc = matches!(op, Op::Cond(..));
            while let Some(top) = stack.last() {
                if top.prec() > op.prec() || (top.prec() == op.prec() && !right_assoc) {
                    let top = stack.pop().unwrap();
                    reduce(&mut out, top);
                } else {
                    break;
                }
            }
            stack.push(op);
            out.push(operand);
        }
        while let Some(op) = stack.pop() {
            reduce(&mut out, op);
        }
        assert_eq!(out.len(), 1);
        out.pop().unwrap()
    }

    pub(super) fn unop(&self, op: &sv::UnaryOperator) -> Result<UnOp> {
        use UnOp::*;
        Ok(match self.text(op).as_str() {
            "~" => BitNot,
            "!" => LogNot,
            "-" => Neg,
            "+" => Plus,
            "&" => RedAnd,
            "|" => RedOr,
            "^" => RedXor,
            "~&" => RedNand,
            "~|" => RedNor,
            "~^" | "^~" => RedXnor,
            s => return self.err(op, format!("operator `{s}` not supported")),
        })
    }

    pub(super) fn binop(&self, op: &sv::BinaryOperator) -> Result<Op> {
        use BinOp::*;
        let span = self.span(op);
        Ok(Op::Bin(
            match self.text(op).as_str() {
                "&" => BitAnd,
                "|" => BitOr,
                "^" => BitXor,
                "~^" | "^~" => BitXnor,
                "+" => Add,
                "-" => Sub,
                "*" => Mul,
                "==" => Eq,
                "!=" => Ne,
                "===" => CaseEq,
                "!==" => CaseNe,
                "<" => Lt,
                "<=" => Le,
                ">" => Gt,
                ">=" => Ge,
                "&&" => LogAnd,
                "||" => LogOr,
                "<<" => Shl,
                ">>" => Shr,
                "<<<" => AShl,
                ">>>" => AShr,
                "/" => Div,
                "%" => Mod,
                "**" => Pow,
                s => return self.err(op, format!("operator `{s}` not supported")),
            },
            span,
        ))
    }
}
