//! Procedural blocks, event controls and statements.

use super::decl::unknown_value;
use super::{Cx, Result};
use crate::ast::{
    CaseItem, Edge, Error, Expr, ExprKind, Item, Lvalue, LvalueKind, Name, Reset, Span, Stmt,
    StmtKind, UnOp, Written,
};
use sv_parser as sv;

/// The clock and the asynchronous reset of a block with the two events `a`
/// and `b` and the body `body`: the reset is the event the body's single
/// `if` tests, with the polarity of its edge.
fn clock_and_reset(
    a: &(Name, Edge, Span),
    b: &(Name, Edge, Span),
    body: &Stmt,
) -> Result<((Name, Edge, Span), Reset)> {
    if a.0 == b.0 {
        return Err(Error::new(
            b.2,
            format!("`{}` appears twice in the event list", b.0),
        ));
    }
    let (name, low, cond_span) = reset_test(body).ok_or_else(|| {
        Error::new(
            body.span,
            "a block with two events needs an asynchronous reset: its body must \
             be a single `if` testing one of the events, `if (rst)` for \
             `posedge rst` or `if (!rst_n)` for `negedge rst_n`",
        )
    })?;
    let (r, c) = if *name == a.0 {
        (a, b)
    } else if *name == b.0 {
        (b, a)
    } else {
        return Err(Error::new(
            cond_span,
            format!(
                "a block with two events needs an asynchronous reset: its `if` \
                 must test `{}` or `{}`",
                a.0, b.0
            ),
        ));
    };
    if low != (r.1 == Edge::Neg) {
        let test = if r.1 == Edge::Neg {
            format!("!{}", r.0)
        } else {
            r.0.to_string()
        };
        return Err(Error::new(
            cond_span,
            format!(
                "the block resets on `{} {}`, so its `if` must test `{test}`",
                r.1, r.0
            ),
        ));
    }
    let reset = Reset {
        name: r.0.clone(),
        edge: r.1,
        span: r.2,
    };
    Ok((c.clone(), reset))
}

/// What the `if` of a block with an asynchronous reset tests: the body is a
/// single `if` (possibly inside `begin … end`) whose condition is `x`, `!x`
/// or `~x` for a name `x`. Gives `x`, whether it is negated, and the
/// position of the condition.
pub(super) fn reset_test(body: &Stmt) -> Option<(&Name, bool, Span)> {
    match &body.kind {
        StmtKind::Block(stmts) if stmts.len() == 1 => reset_test(&stmts[0]),
        StmtKind::If(c, _, _) => match &c.kind {
            ExprKind::Var(x) => Some((x, false, c.span)),
            ExprKind::Unary(UnOp::LogNot | UnOp::BitNot, e) => match &e.kind {
                ExprKind::Var(x) => Some((x, true, c.span)),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

impl Cx<'_> {
    /// `always_comb stmt`, or `always_ff @(posedge clk) stmt` (or
    /// `negedge`): one edge of one clock signal, optionally with an
    /// asynchronous reset `@(posedge clk or posedge rst) if (rst) … else …`
    /// (or `negedge rst_n` and `if (!rst_n)`; events in either order,
    /// separated by `or` or `,`). The reset is the event the `if` tests.
    ///
    /// A plain `always` is classified by its event control alone: with
    /// edge events it is a clocked block (as `always_ff`), with `@*` or
    /// `@(*)` a combinational one (as `always_comb`), and with a list of
    /// signals without edges (`@(a or b)`, `@(a, b)`) also a combinational
    /// one, which typing accepts only if the list names every signal the
    /// block reads. Synthesis ignores such a list, a simulator runs the
    /// block only when a listed signal changes: with a missing signal the
    /// two disagree. A plain `always` without event control is rejected.
    pub(super) fn always(&self, a: &sv::AlwaysConstruct) -> Result<Item> {
        let (kw, stmt) = &a.nodes;
        let plain = match kw {
            sv::AlwaysKeyword::AlwaysFf(_) => false,
            sv::AlwaysKeyword::AlwaysComb(_) => {
                return Ok(Item::AlwaysComb {
                    body: self.statement(stmt)?,
                    span: self.span(kw),
                    written: Written::Keyword,
                });
            }
            sv::AlwaysKeyword::AlwaysLatch(_) => {
                return self.err(kw, "`always_latch` not supported");
            }
            sv::AlwaysKeyword::Always(_) => true,
        };
        let needs = if plain {
            "a plain `always` needs an event control: `@(posedge <clock>)`, \
             `@(negedge <clock>)` or `@*`"
        } else {
            "`always_ff` needs `@(posedge <clock>)` or `@(negedge <clock>)`"
        };
        let (label, attrs, item) = &stmt.nodes;
        self.absent(label, "statement label")?;
        self.absent(attrs, "attribute")?;
        let sv::StatementItem::ProceduralTimingControlStatement(t) = item else {
            return self.err(stmt, needs);
        };
        let (ctrl, body) = &t.nodes;
        let event = match ctrl {
            sv::ProceduralTimingControl::EventControl(e) => match e.as_ref() {
                sv::EventControl::EventExpression(e) => &e.nodes.1.nodes.1,
                sv::EventControl::Asterisk(_) | sv::EventControl::ParenAsterisk(_) if plain => {
                    return Ok(Item::AlwaysComb {
                        body: self.stmt(body)?,
                        span: self.span(kw),
                        written: Written::Star,
                    });
                }
                e => return self.err(e, needs),
            },
            c => return self.err(c, needs),
        };
        let mut events = Vec::new();
        self.events(event, &mut events)?;
        if plain && events.iter().all(|e| e.nodes.0.is_none()) {
            let names = events
                .into_iter()
                .map(|e| self.level_event(e))
                .collect::<Result<Vec<_>>>()?;
            return Ok(Item::AlwaysComb {
                body: self.stmt(body)?,
                span: self.span(kw),
                written: Written::List(names),
            });
        }
        if let Some(e) = events.get(2) {
            return self.err(
                *e,
                "at most two events supported: a clock and an asynchronous reset",
            );
        }
        let events = events
            .into_iter()
            .map(|e| self.event(e))
            .collect::<Result<Vec<_>>>()?;
        let body = self.stmt(body)?;
        let (clock, edge, clock_span, reset) = match &events[..] {
            [(name, edge, span)] => (name.clone(), *edge, *span, None),
            [a, b] => {
                let ((clock, edge, span), reset) = clock_and_reset(a, b, &body)?;
                (clock, edge, span, Some(reset))
            }
            _ => unreachable!("one or two events"),
        };
        Ok(Item::AlwaysFf {
            clock,
            edge,
            clock_span,
            reset,
            body,
            span: self.span(kw),
            plain,
        })
    }

    /// Adds the events of an event list (`a or b`, `a, b`, parentheses) to
    /// `out`, in source order.
    pub(super) fn events<'e>(
        &self,
        e: &'e sv::EventExpression,
        out: &mut Vec<&'e sv::EventExpressionExpression>,
    ) -> Result<()> {
        match e {
            sv::EventExpression::Expression(e) => out.push(e),
            sv::EventExpression::Or(e) => {
                self.events(&e.nodes.0, out)?;
                self.events(&e.nodes.2, out)?;
            }
            sv::EventExpression::Comma(e) => {
                self.events(&e.nodes.0, out)?;
                self.events(&e.nodes.2, out)?;
            }
            sv::EventExpression::Paren(e) => self.events(&e.nodes.0.nodes.1, out)?,
            sv::EventExpression::Sequence(e) => {
                return self.err(e, "sequence events not supported");
            }
        }
        Ok(())
    }

    /// One event of a level list `@(a or b)` of a plain `always`: a signal
    /// name (an array name stands for all its elements) and its position.
    pub(super) fn level_event(
        &self,
        event: &sv::EventExpressionExpression,
    ) -> Result<(Name, Span)> {
        let (_, x, iff) = &event.nodes;
        self.absent(iff, "`iff`")?;
        let span = self.span(x);
        // the text, not `expr`: an array name alone is no expression
        let text = self.text(x);
        let name = text.trim();
        let mut chars = name.chars();
        let ident = chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$');
        if !ident {
            return Err(Error::new(span, "an event must be on a signal name"));
        }
        Ok((name.to_string(), span))
    }

    /// One event `posedge x` or `negedge x` of an `always_ff` block: the
    /// name, the edge and the position of the name.
    pub(super) fn event(
        &self,
        event: &sv::EventExpressionExpression,
    ) -> Result<(Name, Edge, Span)> {
        let (edge, clock, iff) = &event.nodes;
        let edge = match edge {
            Some(sv::EdgeIdentifier::Posedge(_)) => Edge::Pos,
            Some(sv::EdgeIdentifier::Negedge(_)) => Edge::Neg,
            Some(e) => {
                return self.err(
                    e,
                    format!(
                        "`{}` not supported: use separate `posedge` and `negedge` blocks",
                        self.text(e)
                    ),
                );
            }
            None => {
                return self.err(
                    clock,
                    "level-sensitive event not supported, use `posedge` or `negedge`",
                );
            }
        };
        self.absent(iff, "`iff`")?;
        let span = self.span(clock);
        let ExprKind::Var(name) = self.expr(clock)?.kind else {
            return Err(Error::new(span, "an event must be on a signal name"));
        };
        Ok((name, edge, span))
    }

    pub(super) fn stmt(&self, s: &sv::StatementOrNull) -> Result<Stmt> {
        match s {
            sv::StatementOrNull::Statement(s) => self.statement(s),
            sv::StatementOrNull::Attribute(a) => {
                let (attrs, semi) = &a.nodes;
                self.absent(attrs, "attribute")?;
                Ok(Stmt::empty(self.span(semi)))
            }
        }
    }

    /// A statement of an `always_ff` or `always_comb` block; that each
    /// kind of block has only its kind of assignment (`<=` or `=`) is
    /// checked in [`typing`](crate::typing).
    pub(super) fn statement(&self, s: &sv::Statement) -> Result<Stmt> {
        let (label, attrs, item) = &s.nodes;
        self.absent(label, "statement label")?;
        let parallel = self.parallel_case(attrs, item)?;
        let span = self.span(item);
        let kind = match item {
            sv::StatementItem::NonblockingAssignment(a) => {
                let (lhs, _, delay, rhs) = &a.0.nodes;
                self.absent(delay, "intra-assignment delay")?;
                let lhs = self.variable_lvalue(lhs)?;
                let span = lhs.span;
                let kind = match unknown_value(&self.text(rhs)) {
                    Some(size) => StmtKind::Unknown {
                        lhs,
                        blocking: false,
                        size,
                    },
                    None => StmtKind::NonBlocking {
                        lhs,
                        rhs: self.expr(rhs)?,
                    },
                };
                return Ok(Stmt { kind, span });
            }
            sv::StatementItem::BlockingAssignment(a) => match &a.0 {
                sv::BlockingAssignment::OperatorAssignment(a) => {
                    let (lhs, op, rhs) = &a.nodes;
                    if self.text(op) != "=" {
                        return self.err(op, format!("`{}` not supported", self.text(op)));
                    }
                    let lhs = self.variable_lvalue(lhs)?;
                    let span = lhs.span;
                    let kind = match unknown_value(&self.text(rhs)) {
                        Some(size) => StmtKind::Unknown {
                            lhs,
                            blocking: true,
                            size,
                        },
                        None => StmtKind::Blocking {
                            lhs,
                            rhs: self.expr(rhs)?,
                        },
                    };
                    return Ok(Stmt { kind, span });
                }
                sv::BlockingAssignment::Variable(a) => {
                    return self.err(&a.nodes.2, "intra-assignment delay not supported");
                }
                a => return self.unsupported(a),
            },
            sv::StatementItem::ConditionalStatement(c) => {
                let (unique, _, cond, then, elifs, els) = &c.nodes;
                if let Some(u) = unique {
                    return self.err(u, format!("`{}` not supported", self.text(u)));
                }
                // `else if` chains nest to the right
                let mut branches = vec![(cond, then)];
                branches.extend(elifs.iter().map(|(_, _, c, s)| (c, s)));
                let mut acc = match els {
                    Some((_, s)) => self.stmt(s)?,
                    None => Stmt::empty(span),
                };
                for (i, (cond, then)) in branches.into_iter().enumerate().rev() {
                    let span = if i == 0 {
                        span
                    } else {
                        self.span(&elifs[i - 1].1)
                    };
                    acc = Stmt {
                        kind: StmtKind::If(
                            self.cond_predicate(&cond.nodes.1)?,
                            Box::new(self.stmt(then)?),
                            Box::new(acc),
                        ),
                        span,
                    };
                }
                return Ok(acc);
            }
            sv::StatementItem::SeqBlock(b) => {
                let stmts = self.seq_block(b, "declarations in blocks not supported")?;
                StmtKind::Block(stmts.iter().map(|s| self.stmt(s)).collect::<Result<_>>()?)
            }
            sv::StatementItem::CaseStatement(c) => match c.as_ref() {
                sv::CaseStatement::Normal(c) => return self.case(c, parallel, span),
                sv::CaseStatement::Matches(c) => {
                    return self.err(&c.nodes.3, "`case … matches` not supported");
                }
                sv::CaseStatement::Inside(c) => {
                    return self.err(&c.nodes.3, "`case … inside` not supported");
                }
            },
            i => return self.unsupported(i),
        };
        Ok(Stmt { kind, span })
    }

    /// `case (sel) … endcase`: no `casez`/`casex`, no
    /// `unique`/`priority`, at most one `default`.
    pub(super) fn case(
        &self,
        c: &sv::CaseStatementNormal,
        parallel: Option<Span>,
        span: Span,
    ) -> Result<Stmt> {
        let (unique, kw, sel, first, rest, _) = &c.nodes;
        if let Some(u) = unique {
            return self.err(u, format!("`{}` not supported", self.text(u)));
        }
        if !matches!(kw, sv::CaseKeyword::Case(_)) {
            return self.err(kw, format!("`{}` not supported", self.text(kw)));
        }
        let sel = self.expr(&sel.nodes.1.nodes.0)?;
        let mut items = Vec::new();
        let mut default: Option<Box<Stmt>> = None;
        for item in std::iter::once(first).chain(rest) {
            match item {
                sv::CaseItem::NonDefault(i) => {
                    let (labels, _, body) = &i.nodes;
                    let labels = labels
                        .contents()
                        .into_iter()
                        .map(|l| self.expr(&l.nodes.0))
                        .collect::<Result<_>>()?;
                    let body = self.stmt(body)?;
                    items.push(CaseItem { labels, body });
                }
                sv::CaseItem::Default(d) => {
                    let (kw, _, body) = &d.nodes;
                    if let Some(prev) = &default {
                        let at = prev.span;
                        return self
                            .err(kw, format!("second `default` (first one's body at {at})"));
                    }
                    default = Some(Box::new(self.stmt(body)?));
                }
            }
        }
        Ok(Stmt {
            kind: StmtKind::Case {
                sel,
                items,
                default,
                parallel,
            },
            span,
        })
    }

    /// The attributes of a statement: only `(* parallel_case *)` on a
    /// `case` is accepted (its position is returned); any other attribute
    /// is rejected, since some change what synthesis builds (`full_case`
    /// makes missing cases don't-cares).
    pub(super) fn parallel_case(
        &self,
        attrs: &[sv::AttributeInstance],
        item: &sv::StatementItem,
    ) -> Result<Option<Span>> {
        let mut parallel = None;
        self.attributes(attrs, "parallel_case", |spec| {
            if let Some((_, v)) = &spec.nodes.1 {
                return self.err(v, "`parallel_case` takes no value");
            }
            let is_case = matches!(
                item,
                sv::StatementItem::CaseStatement(c)
                    if matches!(c.as_ref(), sv::CaseStatement::Normal(_))
            );
            if !is_case {
                return self.err(spec, "`parallel_case` is only supported on a `case`");
            }
            parallel = Some(self.span(spec));
            Ok(())
        })?;
        Ok(parallel)
    }

    /// The condition of an `if`: one expression, no patterns.
    pub(super) fn cond_predicate(&self, p: &sv::CondPredicate) -> Result<Expr> {
        match p.nodes.0.contents().as_slice() {
            [sv::ExpressionOrCondPattern::Expression(e)] => self.expr(e),
            _ => self.err(p, "`&&&` and patterns not supported"),
        }
    }

    /// The target of `=` or `<=`: a signal, a select of a signal, or a
    /// concatenation of targets.
    pub(super) fn variable_lvalue(&self, lv: &sv::VariableLvalue) -> Result<Lvalue> {
        let span = self.span(lv);
        let kind = match lv {
            sv::VariableLvalue::Identifier(lv) => {
                let (scope, id, select) = &lv.nodes;
                self.absent(scope, "package or class scope")?;
                let id = self.identifier(self.simple_hierarchical(&id.nodes.0)?)?;
                if self.is_array(&id) {
                    LvalueKind::Elem(Box::new(self.element(id, select, span)?))
                } else {
                    match self.select(select)? {
                        None => LvalueKind::Var(id),
                        Some(sel) => LvalueKind::Select(id, sel),
                    }
                }
            }
            sv::VariableLvalue::Lvalue(c) => LvalueKind::Concat(
                c.nodes
                    .0
                    .nodes
                    .1
                    .contents()
                    .into_iter()
                    .map(|l| self.variable_lvalue(l))
                    .collect::<Result<_>>()?,
            ),
            lv => return self.err(lv, "assignment patterns and streams not supported"),
        };
        Ok(Lvalue { kind, span })
    }
}
