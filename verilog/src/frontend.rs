/*!
# From source text to the AST

Parses with `sv-parser` and lowers its syntax tree into [`ast::Module`],
rejecting everything outside the accepted subset with a located error.

`sv-parser` ignores operator precedence: a chain of binary operators comes
out nested to the right in source order (`a * b + c` as `a * (b + c)`), and
`?:` is nested the same way (`a + b ? c : d` as `a + (b ? c : d)`). The
frontend therefore flattens every chain back into its source order and
re-associates it with the precedence table of IEEE 1800-2017, Table 11-2
(see [`prec`]). Parenthesized subexpressions are separate nodes and are
not affected.

[`ast::Module`]: crate::ast::Module
*/

use crate::ast::{
    BinOp, CaseItem, Cast, Decl, Dir, Error, Expr, ExprKind, Index, Item, Lvalue, LvalueKind,
    Module, Port, Select, Span, Stmt, StmtKind, Ty, UnOp,
};
use crate::value::BitVec;
use num_bigint::BigUint;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use sv_parser as sv;
use sv_parser::{Iter, RefNode, RefNodes, SyntaxTree};

type Result<T> = std::result::Result<T, Error>;

const VARIABLE_TARGET: &str =
    "variable index in an assignment target not supported (assign under a `case` instead)";

/// Parses a file holding exactly one module.
pub fn parse_file(path: impl AsRef<Path>) -> Result<Module> {
    let path = path.as_ref();
    let source = std::fs::read_to_string(path)
        .map_err(|e| Error::new(Span::default(), format!("{}: {e}", path.display())))?;
    parse(&source, path)
}

/// Parses `source` holding exactly one module; `path` is used for messages
/// of the preprocessor only.
pub fn parse(source: &str, path: impl AsRef<Path>) -> Result<Module> {
    let defines: HashMap<String, Option<sv::Define>> = HashMap::new();
    let includes: [PathBuf; 0] = [];
    let mut lines = vec![0];
    lines.extend(source.match_indices('\n').map(|(i, _)| i + 1));
    let at = |offset: usize| {
        let line = lines.partition_point(|&start| start <= offset);
        Span {
            line,
            col: offset - lines[line - 1] + 1,
        }
    };
    let (tree, _) =
        sv::parse_sv_str(source, path, &defines, &includes, false, false).map_err(|e| match e {
            sv::Error::Parse(Some((_, offset))) => Error::new(at(offset), "syntax error"),
            e => Error::new(Span::default(), e.to_string()),
        })?;
    let cx = Cx {
        tree: &tree,
        at: &at,
    };
    let source_text = (&tree)
        .into_iter()
        .find_map(|n| match n {
            RefNode::SourceText(x) => Some(x),
            _ => None,
        })
        .ok_or_else(|| Error::new(Span::default(), "empty source"))?;
    cx.source_text(source_text)
}

struct Cx<'t> {
    tree: &'t SyntaxTree,
    at: &'t dyn Fn(usize) -> Span,
}

/// Whether the node contains any token; optional syntax that is absent or
/// empty contains none.
fn has_tokens<'a>(node: impl Into<RefNodes<'a>>) -> bool {
    Iter::new(node.into()).any(|n| matches!(n, RefNode::Locate(_)))
}

impl Cx<'_> {
    /// The position of the first token of the node.
    fn span<'a>(&self, node: impl Into<RefNodes<'a>>) -> Span {
        Iter::new(node.into())
            .find_map(|n| match n {
                RefNode::Locate(l) => self.tree.get_origin(l).map(|(_, off)| (self.at)(off)),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// The source text of the node, without trailing whitespace.
    fn text<'a>(&self, node: impl Into<RefNodes<'a>>) -> String {
        self.tree.get_str_trim(node).unwrap_or("").to_string()
    }

    fn err<'a, T>(&self, node: impl Into<RefNodes<'a>>, msg: impl Into<String>) -> Result<T> {
        Err(Error::new(self.span(node), msg))
    }

    /// Rejects the node as outside the subset, naming its syntactic kind.
    fn unsupported<'a, T>(&self, node: impl Into<RefNode<'a>>) -> Result<T> {
        let node: RefNode = node.into();
        let span = self.span(RefNodes(vec![node.clone()]));
        Err(Error::new(span, format!("unsupported: {node}")))
    }

    /// Rejects the node if it contains tokens, i.e. if the optional syntax
    /// `what` is present.
    fn absent<'a>(&self, node: impl Into<RefNodes<'a>> + Copy, what: &str) -> Result<()> {
        if has_tokens(node) {
            self.err(node, format!("{what} not supported"))
        } else {
            Ok(())
        }
    }

    fn identifier(&self, id: &sv::Identifier) -> Result<String> {
        match id {
            sv::Identifier::SimpleIdentifier(_) => Ok(self.text(id)),
            sv::Identifier::EscapedIdentifier(_) => {
                self.err(id, "escaped identifiers not supported")
            }
        }
    }

    // ------------------------------------------------------------------
    // Modules and declarations
    // ------------------------------------------------------------------

    fn source_text(&self, st: &sv::SourceText) -> Result<Module> {
        let (_, timeunits, descriptions) = &st.nodes;
        self.absent(timeunits, "timeunits declaration")?;
        let mut modules = Vec::new();
        for d in descriptions {
            match d {
                sv::Description::ModuleDeclaration(m) => match m.as_ref() {
                    sv::ModuleDeclaration::Ansi(m) => modules.push(m.as_ref()),
                    sv::ModuleDeclaration::Nonansi(_) => {
                        return self.err(m.as_ref(), "non-ANSI port list not supported");
                    }
                    m => return self.unsupported(m),
                },
                d => return self.unsupported(d),
            }
        }
        match modules.as_slice() {
            [m] => self.module(m),
            [] => Err(Error::new(Span::default(), "no module")),
            [_, m, ..] => self.err(*m, "more than one module not supported"),
        }
    }

    fn module(&self, m: &sv::ModuleDeclarationAnsi) -> Result<Module> {
        let (header, timeunits, items, _, _) = &m.nodes;
        let (attrs, _, lifetime, name, imports, params, ports, _) = &header.nodes;
        self.absent(attrs, "attribute")?;
        self.absent(lifetime, "lifetime")?;
        self.absent(imports, "package import")?;
        self.absent(params, "parameter port list")?;
        self.absent(timeunits, "timeunits declaration")?;
        let mut module = Module {
            name: self.identifier(&name.nodes.0)?,
            ports: Vec::new(),
            decls: Vec::new(),
            items: Vec::new(),
        };
        if let Some(ports) = ports
            && let Some(list) = &ports.nodes.0.nodes.1
        {
            let mut prev = None;
            for (attrs, port) in list.contents() {
                self.absent(attrs, "attribute")?;
                let p = self.port(port, prev)?;
                prev = Some((p.dir, p.decl.ty));
                module.ports.push(p);
            }
        }
        for item in items {
            self.module_item(item, &mut module)?;
        }
        Ok(module)
    }

    /// A port of the ANSI header. Direction and type are inherited from the
    /// previous port where omitted (`input logic [7:0] a, b`).
    fn port(&self, p: &sv::AnsiPortDeclaration, prev: Option<(Dir, Ty)>) -> Result<Port> {
        let (dir, ty, id) = match p {
            sv::AnsiPortDeclaration::Net(n) => {
                let (header, id, dims, default) = &n.nodes;
                self.absent(dims, "unpacked dimension")?;
                self.absent(default, "port default")?;
                match header {
                    None => (None, None, id),
                    Some(sv::NetPortHeaderOrInterfacePortHeader::NetPortHeader(h)) => {
                        let (dir, ty) = &h.nodes;
                        let ty = match ty {
                            sv::NetPortType::DataType(t) => {
                                let (net, ty) = &t.nodes;
                                if let Some(net) = net {
                                    self.net_type(net)?;
                                }
                                self.data_type_or_implicit(ty)?
                            }
                            t => return self.unsupported(t),
                        };
                        (dir.as_ref(), Some(ty), id)
                    }
                    Some(h) => return self.unsupported(h),
                }
            }
            sv::AnsiPortDeclaration::Variable(v) => {
                let (header, id, dims, default) = &v.nodes;
                self.absent(dims, "unpacked dimension")?;
                self.absent(default, "port default")?;
                match header {
                    None => (None, None, id),
                    Some(h) => {
                        let (dir, ty) = &h.nodes;
                        let ty = match &ty.nodes.0 {
                            sv::VarDataType::DataType(t) => self.data_type(t)?,
                            sv::VarDataType::Var(v) => self.data_type_or_implicit(&v.nodes.1)?,
                        };
                        (dir.as_ref(), Some(ty), id)
                    }
                }
            }
            p => return self.unsupported(p),
        };
        let dir = match dir {
            Some(sv::PortDirection::Input(_)) => Dir::Input,
            Some(sv::PortDirection::Output(_)) => Dir::Output,
            Some(d) => return self.unsupported(d),
            None => match prev {
                Some((dir, _)) => dir,
                None => return self.err(id, "port without direction"),
            },
        };
        let ty = match (ty, prev) {
            (Some(ty), _) => ty,
            (None, Some((_, ty))) => ty,
            (None, None) => unreachable!("the first port has a header"),
        };
        Ok(Port {
            dir,
            decl: Decl {
                name: self.identifier(&id.nodes.0)?,
                ty,
                span: self.span(id),
            },
        })
    }

    fn net_type(&self, net: &sv::NetType) -> Result<()> {
        match net {
            sv::NetType::Wire(_) => Ok(()),
            n => self.err(n, format!("net type `{}` not supported", self.text(n))),
        }
    }

    fn data_type_or_implicit(&self, t: &sv::DataTypeOrImplicit) -> Result<Ty> {
        match t {
            sv::DataTypeOrImplicit::DataType(t) => self.data_type(t),
            sv::DataTypeOrImplicit::ImplicitDataType(t) => {
                let (signing, dims) = &t.nodes;
                self.vector(signing, dims)
            }
        }
    }

    fn data_type(&self, t: &sv::DataType) -> Result<Ty> {
        match t {
            sv::DataType::Vector(v) => {
                let (kind, signing, dims) = &v.nodes;
                match kind {
                    sv::IntegerVectorType::Logic(_) | sv::IntegerVectorType::Bit(_) => {}
                    sv::IntegerVectorType::Reg(_) => {
                        return self.err(kind, "`reg` not supported, use `logic`");
                    }
                }
                self.vector(signing, dims)
            }
            t => self.err(t, format!("type `{}` not supported", self.text(t))),
        }
    }

    /// `[signed|unsigned] [hi:0]`, at most one packed dimension.
    fn vector(&self, signing: &Option<sv::Signing>, dims: &[sv::PackedDimension]) -> Result<Ty> {
        let signed = matches!(signing, Some(sv::Signing::Signed(_)));
        let width = match dims {
            [] => 1,
            [sv::PackedDimension::Range(r)] => {
                let (hi, _, lo) = &r.nodes.0.nodes.1.nodes;
                let (hi, lo) = (self.const_int(hi)?, self.const_int(lo)?);
                if lo != 0 || hi < lo {
                    return self.err(
                        r.as_ref(),
                        format!("range [{hi}:{lo}] not supported, only [N-1:0]"),
                    );
                }
                hi + 1
            }
            [d] => return self.unsupported(d),
            [_, d, ..] => return self.err(d, "more than one packed dimension not supported"),
        };
        Ok(Ty { width, signed })
    }

    /// A decimal integer literal (parameters are not supported yet).
    fn const_int<'a>(&self, e: impl Into<RefNodes<'a>> + Copy) -> Result<usize> {
        match self.decimal(e) {
            Some(n) => n,
            None => {
                let text = self.text(e);
                self.err(e, format!("expected a decimal integer, got `{text}`"))
            }
        }
    }

    /// The value of the node if it is a decimal integer literal.
    fn decimal<'a>(&self, e: impl Into<RefNodes<'a>> + Copy) -> Option<Result<usize>> {
        let text = self.text(e).replace('_', "");
        (!text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())).then(|| {
            text.parse()
                .or_else(|_| self.err(e, format!("`{text}` too large")))
        })
    }

    fn module_item(&self, item: &sv::NonPortModuleItem, m: &mut Module) -> Result<()> {
        let sv::NonPortModuleItem::ModuleOrGenerateItem(item) = item else {
            return self.unsupported(item);
        };
        let item = match item.as_ref() {
            sv::ModuleOrGenerateItem::ModuleItem(i) => i,
            sv::ModuleOrGenerateItem::Module(i) => {
                return self.err(i.as_ref(), "module instances not supported yet");
            }
            i => return self.unsupported(i),
        };
        let (attrs, item) = &item.nodes;
        self.absent(attrs, "attribute")?;
        match item {
            sv::ModuleCommonItem::ModuleOrGenerateItemDeclaration(d) => match d.as_ref() {
                sv::ModuleOrGenerateItemDeclaration::PackageOrGenerateItemDeclaration(d) => {
                    self.declaration(d, m)
                }
                d => self.unsupported(d),
            },
            sv::ModuleCommonItem::ContinuousAssign(a) => match a.as_ref() {
                sv::ContinuousAssign::Net(a) => {
                    let (_, strength, delay, list, _) = &a.nodes;
                    self.absent(strength, "drive strength")?;
                    self.absent(delay, "delay")?;
                    for a in list.nodes.0.contents() {
                        let (lhs, _, rhs) = &a.nodes;
                        let lhs = self.net_lvalue(lhs)?;
                        let rhs = self.expr(rhs)?;
                        let span = lhs.span;
                        m.items.push(Item::Assign { lhs, rhs, span });
                    }
                    Ok(())
                }
                a => self.err(a, "delay not supported"),
            },
            sv::ModuleCommonItem::AlwaysConstruct(a) => {
                m.items.push(self.always(a)?);
                Ok(())
            }
            i => self.unsupported(i),
        }
    }

    /// `always_comb stmt`, or `always_ff @(posedge clk) stmt`: one edge
    /// of one clock signal, no other events (asynchronous resets come later).
    fn always(&self, a: &sv::AlwaysConstruct) -> Result<Item> {
        let (kw, stmt) = &a.nodes;
        match kw {
            sv::AlwaysKeyword::AlwaysFf(_) => {}
            sv::AlwaysKeyword::AlwaysComb(_) => {
                return Ok(Item::AlwaysComb {
                    body: self.statement(stmt)?,
                    span: self.span(kw),
                });
            }
            sv::AlwaysKeyword::AlwaysLatch(_) => {
                return self.err(kw, "`always_latch` not supported");
            }
            sv::AlwaysKeyword::Always(_) => {
                return self.err(
                    kw,
                    "plain `always` not supported, use `always_ff` or `always_comb`",
                );
            }
        }
        let (label, attrs, item) = &stmt.nodes;
        self.absent(label, "statement label")?;
        self.absent(attrs, "attribute")?;
        let sv::StatementItem::ProceduralTimingControlStatement(t) = item else {
            return self.err(stmt, "`always_ff` needs `@(posedge <clock>)`");
        };
        let (ctrl, body) = &t.nodes;
        let event = match ctrl {
            sv::ProceduralTimingControl::EventControl(e) => match e.as_ref() {
                sv::EventControl::EventExpression(e) => &e.nodes.1.nodes.1,
                e => return self.err(e, "`always_ff` needs `@(posedge <clock>)`"),
            },
            c => return self.err(c, "`always_ff` needs `@(posedge <clock>)`"),
        };
        let sv::EventExpression::Expression(event) = event else {
            return self.err(
                event,
                "only one event `posedge <clock>` supported (no asynchronous reset)",
            );
        };
        let (edge, clock, iff) = &event.nodes;
        match edge {
            Some(sv::EdgeIdentifier::Posedge(_)) => {}
            Some(e) => {
                return self.err(
                    e,
                    format!("`{}` not supported, only `posedge`", self.text(e)),
                );
            }
            None => return self.err(clock, "level-sensitive event not supported, use `posedge`"),
        }
        self.absent(iff, "`iff`")?;
        let clock_span = self.span(clock);
        let ExprKind::Var(clock) = self.expr(clock)?.kind else {
            return Err(Error::new(clock_span, "the clock must be a signal name"));
        };
        Ok(Item::AlwaysFf {
            clock,
            clock_span,
            body: self.stmt(body)?,
            span: self.span(kw),
        })
    }

    // ------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------

    fn stmt(&self, s: &sv::StatementOrNull) -> Result<Stmt> {
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
    fn statement(&self, s: &sv::Statement) -> Result<Stmt> {
        let (label, attrs, item) = &s.nodes;
        self.absent(label, "statement label")?;
        self.absent(attrs, "attribute")?;
        let span = self.span(item);
        let kind = match item {
            sv::StatementItem::NonblockingAssignment(a) => {
                let (lhs, _, delay, rhs) = &a.0.nodes;
                self.absent(delay, "intra-assignment delay")?;
                let lhs = self.variable_lvalue(lhs)?;
                let rhs = self.expr(rhs)?;
                let span = lhs.span;
                return Ok(Stmt {
                    kind: StmtKind::NonBlocking { lhs, rhs },
                    span,
                });
            }
            sv::StatementItem::BlockingAssignment(a) => match &a.0 {
                sv::BlockingAssignment::OperatorAssignment(a) => {
                    let (lhs, op, rhs) = &a.nodes;
                    if self.text(op) != "=" {
                        return self.err(op, format!("`{}` not supported", self.text(op)));
                    }
                    let lhs = self.variable_lvalue(lhs)?;
                    let rhs = self.expr(rhs)?;
                    let span = lhs.span;
                    return Ok(Stmt {
                        kind: StmtKind::Blocking { lhs, rhs },
                        span,
                    });
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
                let (_, label, decls, stmts, _, end_label) = &b.nodes;
                self.absent(label, "block name")?;
                self.absent(end_label, "block name")?;
                if let Some(d) = decls.first() {
                    return self.err(d, "declarations in blocks not supported");
                }
                StmtKind::Block(stmts.iter().map(|s| self.stmt(s)).collect::<Result<_>>()?)
            }
            sv::StatementItem::CaseStatement(c) => match c.as_ref() {
                sv::CaseStatement::Normal(c) => return self.case(c, span),
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
    fn case(&self, c: &sv::CaseStatementNormal, span: Span) -> Result<Stmt> {
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
            },
            span,
        })
    }

    /// The condition of an `if`: one expression, no patterns.
    fn cond_predicate(&self, p: &sv::CondPredicate) -> Result<Expr> {
        match p.nodes.0.contents().as_slice() {
            [sv::ExpressionOrCondPattern::Expression(e)] => self.expr(e),
            _ => self.err(p, "`&&&` and patterns not supported"),
        }
    }

    /// The target of `=` or `<=`: a signal, a select of a signal, or a
    /// concatenation of targets.
    fn variable_lvalue(&self, lv: &sv::VariableLvalue) -> Result<Lvalue> {
        let span = self.span(lv);
        let kind = match lv {
            sv::VariableLvalue::Identifier(lv) => {
                let (scope, id, select) = &lv.nodes;
                self.absent(scope, "package or class scope")?;
                let id = self.identifier(self.simple_hierarchical(&id.nodes.0)?)?;
                match self.select(select)? {
                    None => LvalueKind::Var(id),
                    Some(sel) => LvalueKind::Select(id, sel),
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

    fn declaration(&self, d: &sv::PackageOrGenerateItemDeclaration, m: &mut Module) -> Result<()> {
        // (name, unpacked dimensions, initializer)
        let mut add = |id: &sv::Identifier, ty: Ty, dims: bool, init: bool| -> Result<()> {
            if dims {
                return self.err(id, "unpacked dimension not supported");
            }
            if init {
                return self.err(
                    id,
                    "declaration with initializer not supported, use `assign`",
                );
            }
            m.decls.push(Decl {
                name: self.identifier(id)?,
                ty,
                span: self.span(id),
            });
            Ok(())
        };
        match d {
            sv::PackageOrGenerateItemDeclaration::NetDeclaration(n) => match n.as_ref() {
                sv::NetDeclaration::NetType(n) => {
                    let (net, strength, vs, ty, delay, list, _) = &n.nodes;
                    self.net_type(net)?;
                    self.absent(strength, "strength")?;
                    self.absent(vs, "`vectored`/`scalared`")?;
                    self.absent(delay, "delay")?;
                    let ty = self.data_type_or_implicit(ty)?;
                    for a in list.nodes.0.contents() {
                        let (id, dims, init) = &a.nodes;
                        add(&id.nodes.0, ty, has_tokens(dims), init.is_some())?;
                    }
                    Ok(())
                }
                n => self.unsupported(n),
            },
            sv::PackageOrGenerateItemDeclaration::DataDeclaration(d) => match d.as_ref() {
                sv::DataDeclaration::Variable(v) => {
                    let (konst, _, lifetime, ty, list, _) = &v.nodes;
                    self.absent(konst, "`const`")?;
                    self.absent(lifetime, "lifetime")?;
                    let ty = self.data_type_or_implicit(ty)?;
                    for a in list.nodes.0.contents() {
                        match a {
                            sv::VariableDeclAssignment::Variable(a) => {
                                let (id, dims, init) = &a.nodes;
                                add(&id.nodes.0, ty, has_tokens(dims), init.is_some())?;
                            }
                            a => return self.unsupported(a),
                        }
                    }
                    Ok(())
                }
                d => self.unsupported(d),
            },
            sv::PackageOrGenerateItemDeclaration::Empty(_) => Ok(()),
            d => self.unsupported(d),
        }
    }

    /// The target of `assign`: like [`variable_lvalue`](Self::variable_lvalue),
    /// with constant selects only.
    fn net_lvalue(&self, lv: &sv::NetLvalue) -> Result<Lvalue> {
        let span = self.span(lv);
        let lv = match lv {
            sv::NetLvalue::Identifier(lv) => lv,
            sv::NetLvalue::Lvalue(c) => {
                let ls = c.nodes.0.nodes.1.contents();
                let ls = ls.into_iter().map(|l| self.net_lvalue(l));
                let kind = LvalueKind::Concat(ls.collect::<Result<_>>()?);
                return Ok(Lvalue { kind, span });
            }
            lv => return self.err(lv, "assignment patterns not supported"),
        };
        let (id, select) = &lv.nodes;
        let id = match id {
            sv::PsOrHierarchicalNetIdentifier::PackageScope(p) => {
                let (scope, id) = &p.nodes;
                self.absent(scope, "package scope")?;
                &id.nodes.0
            }
            sv::PsOrHierarchicalNetIdentifier::HierarchicalNetIdentifier(h) => {
                self.simple_hierarchical(&h.nodes.0)?
            }
        };
        let id = self.identifier(id)?;
        let (member, bits, part) = &select.nodes;
        self.absent(member, "member select")?;
        let index = |e: &sv::ConstantExpression| match self.decimal(e) {
            Some(i) => i.map(Index::Const),
            None => self.err(e, VARIABLE_TARGET),
        };
        let bits = &bits.nodes.0;
        let sel = match (bits.as_slice(), part) {
            ([], None) => None,
            ([b], None) => Some(Select::Bit(index(&b.nodes.1)?)),
            ([], Some(p)) => Some(match &p.nodes.1 {
                sv::ConstantPartSelectRange::ConstantRange(r) => {
                    let (hi, _, lo) = &r.nodes;
                    Select::Part {
                        hi: self.const_int(hi)?,
                        lo: self.const_int(lo)?,
                    }
                }
                sv::ConstantPartSelectRange::ConstantIndexedRange(r) => {
                    let (base, op, width) = &r.nodes;
                    self.indexed(index(base)?, op, self.const_int(width)?)
                }
            }),
            _ => return self.err(select, "multi-dimensional selects not supported"),
        };
        let kind = match sel {
            None => LvalueKind::Var(id),
            Some(sel) => LvalueKind::Select(id, sel),
        };
        Ok(Lvalue { kind, span })
    }

    /// The select `[…]` after a signal name in an expression or a
    /// procedural assignment target, if any.
    fn select(&self, select: &sv::Select) -> Result<Option<Select>> {
        let (member, bits, part) = &select.nodes;
        self.absent(member, "member select")?;
        let bits = &bits.nodes.0;
        Ok(match (bits.as_slice(), part) {
            ([], None) => None,
            ([b], None) => Some(Select::Bit(self.index(&b.nodes.1)?)),
            ([], Some(p)) => Some(match &p.nodes.1 {
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
            }),
            _ => return self.err(select, "multi-dimensional selects not supported"),
        })
    }

    /// `[base +: width]` or `[base -: width]`.
    fn indexed(&self, base: Index, op: &sv::Symbol, width: usize) -> Select {
        if self.text(op) == "+:" {
            Select::Up(base, width)
        } else {
            Select::Down(base, width)
        }
    }

    /// A bit position: a decimal constant, or any expression.
    fn index(&self, e: &sv::Expression) -> Result<Index> {
        match self.decimal(e) {
            Some(i) => i.map(Index::Const),
            None => Ok(Index::Expr(Box::new(self.expr(e)?))),
        }
    }

    /// A hierarchical identifier that is a plain name.
    fn simple_hierarchical<'h>(
        &self,
        h: &'h sv::HierarchicalIdentifier,
    ) -> Result<&'h sv::Identifier> {
        let (root, path, id) = &h.nodes;
        if root.is_some() || !path.is_empty() {
            return self.err(h, "hierarchical names not supported");
        }
        Ok(id)
    }

    // ------------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------------

    fn expr(&self, e: &sv::Expression) -> Result<Expr> {
        let mut operands = Vec::new();
        let mut ops = Vec::new();
        self.flatten(e, &mut operands, &mut ops)?;
        Ok(reassociate(operands, ops))
    }

    /// Flattens a chain of binary operators and conditionals into its
    /// operands and operators, in source order.
    fn flatten(
        &self,
        e: &sv::Expression,
        operands: &mut Vec<Expr>,
        ops: &mut Vec<Op>,
    ) -> Result<()> {
        match e {
            sv::Expression::Binary(b) => {
                let (l, op, attrs, r) = &b.nodes;
                self.absent(attrs, "attribute")?;
                self.flatten(l, operands, ops)?;
                ops.push(Op::Bin(self.binop(op)?, self.span(op)));
                self.flatten(r, operands, ops)
            }
            sv::Expression::ConditionalExpression(c) => {
                let (pred, q, attrs, then, _, els) = &c.nodes;
                self.absent(attrs, "attribute")?;
                let pred = match pred.nodes.0.contents().as_slice() {
                    [sv::ExpressionOrCondPattern::Expression(p)] => p.as_ref(),
                    _ => return self.err(pred, "`&&&` and patterns not supported"),
                };
                self.flatten(pred, operands, ops)?;
                ops.push(Op::Cond(self.expr(then)?, self.span(q)));
                self.flatten(els, operands, ops)
            }
            e => {
                operands.push(self.operand(e)?);
                Ok(())
            }
        }
    }

    /// An expression that is not a binary operation or a conditional.
    fn operand(&self, e: &sv::Expression) -> Result<Expr> {
        match e {
            sv::Expression::Primary(p) => self.primary(p),
            sv::Expression::Unary(u) => {
                let (op, attrs, p) = &u.nodes;
                self.absent(attrs, "attribute")?;
                let span = self.span(op);
                let kind = ExprKind::Unary(self.unop(op)?, Box::new(self.primary(p)?));
                Ok(Expr { kind, span })
            }
            sv::Expression::IncOrDecExpression(e) => {
                self.err(e.as_ref(), "`++`/`--` not supported")
            }
            sv::Expression::OperatorAssignment(e) => {
                self.err(e.as_ref(), "assignment in expression not supported")
            }
            e => self.unsupported(e),
        }
    }

    fn primary(&self, p: &sv::Primary) -> Result<Expr> {
        let span = self.span(p);
        let kind = match p {
            sv::Primary::PrimaryLiteral(l) => match l.as_ref() {
                sv::PrimaryLiteral::Number(n) => match n.as_ref() {
                    sv::Number::IntegralNumber(n) => self.literal(n)?,
                    n => return self.err(n, "real numbers not supported"),
                },
                sv::PrimaryLiteral::UnbasedUnsizedLiteral(_) => {
                    return self.err(
                        p,
                        "unsized literal not supported, write the width (e.g. `8'hff`)",
                    );
                }
                l => return self.unsupported(l),
            },
            sv::Primary::Hierarchical(h) => {
                let (scope, id, select) = &h.nodes;
                self.absent(scope, "package or class scope")?;
                let id = self.identifier(self.simple_hierarchical(id)?)?;
                match self.select(select)? {
                    None => ExprKind::Var(id),
                    Some(sel) => ExprKind::Select(id, sel),
                }
            }
            sv::Primary::MintypmaxExpression(m) => match &m.nodes.0.nodes.1 {
                sv::MintypmaxExpression::Expression(e) => return self.expr(e),
                m => return self.err(m, "min:typ:max expression not supported"),
            },
            sv::Primary::Concatenation(c) => {
                let (c, select) = &c.nodes;
                self.absent(select, "select of a concatenation")?;
                ExprKind::Concat(self.concatenation(c)?)
            }
            sv::Primary::MultipleConcatenation(c) => {
                let (c, select) = &c.nodes;
                self.absent(select, "select of a concatenation")?;
                let (n, c) = &c.nodes.0.nodes.1;
                let n = self.const_int(n)?;
                if n == 0 {
                    return self.err(p, "replication count 0 not supported");
                }
                ExprKind::Repl(n, self.concatenation(c)?)
            }
            sv::Primary::Cast(c) => {
                let (ty, _, e) = &c.nodes;
                let cast = match ty {
                    sv::CastingType::ConstantPrimary(n) => {
                        let n = self.const_int(n.as_ref())?;
                        if n == 0 {
                            return self.err(p, "cast to 0 bits");
                        }
                        Cast::Size(n)
                    }
                    sv::CastingType::Signing(s) if self.text(s.as_ref()) == "signed" => {
                        Cast::Signed
                    }
                    sv::CastingType::Signing(_) => Cast::Unsigned,
                    t => {
                        return self.err(t, format!("cast to `{}` not supported", self.text(t)));
                    }
                };
                ExprKind::Cast(cast, Box::new(self.expr(&e.nodes.1)?))
            }
            sv::Primary::FunctionSubroutineCall(c) => return self.call(&c.nodes.0, span),
            p => return self.unsupported(p),
        };
        Ok(Expr { kind, span })
    }

    /// The items of `{a, b, …}`.
    fn concatenation(&self, c: &sv::Concatenation) -> Result<Vec<Expr>> {
        c.nodes
            .0
            .nodes
            .1
            .contents()
            .into_iter()
            .map(|e| self.expr(e))
            .collect()
    }

    /// A call; only `$signed(e)` and `$unsigned(e)`, which are casts.
    fn call(&self, c: &sv::SubroutineCall, span: Span) -> Result<Expr> {
        let unsupported = || self.err(c, format!("call `{}` not supported", self.text(c)));
        let sv::SubroutineCall::SystemTfCall(call) = c else {
            return unsupported();
        };
        let (name, args) = match call.as_ref() {
            sv::SystemTfCall::ArgOptionl(a) => {
                let (name, args) = &a.nodes;
                let args = match args.as_ref().map(|a| &a.nodes.1) {
                    Some(sv::ListOfArguments::Ordered(a)) if a.nodes.1.is_empty() => {
                        a.nodes.0.contents()
                    }
                    _ => return unsupported(),
                };
                (name, args)
            }
            sv::SystemTfCall::ArgExpression(a) => {
                let (name, args) = &a.nodes;
                let (args, clocking) = &args.nodes.1;
                if clocking.is_some() {
                    return unsupported();
                }
                (name, args.contents())
            }
            sv::SystemTfCall::ArgDataType(_) => return unsupported(),
        };
        let cast = match self.text(name).as_str() {
            "$signed" => Cast::Signed,
            "$unsigned" => Cast::Unsigned,
            _ => return unsupported(),
        };
        match args.as_slice() {
            [Some(e)] => {
                let kind = ExprKind::Cast(cast, Box::new(self.expr(e)?));
                Ok(Expr { kind, span })
            }
            _ => self.err(c, format!("`{}` takes one argument", self.text(name))),
        }
    }

    /// A sized literal `N'[s]<base><digits>`, without x/z digits (values
    /// are 2-state) and with a value that fits in `N` bits.
    fn literal(&self, n: &sv::IntegralNumber) -> Result<ExprKind> {
        let text: String = self
            .text(n)
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '_')
            .collect();
        let Some((size, rest)) = text.split_once('\'').filter(|(size, _)| !size.is_empty()) else {
            return self.err(
                n,
                format!(
                    "unsized literal `{text}` not supported, write the width (e.g. `8'd{}`)",
                    text.trim_start_matches('\'')
                ),
            );
        };
        let width: usize = size
            .parse()
            .or_else(|_| self.err(n, format!("literal size `{size}` too large")))?;
        let (signed, rest) = match rest.strip_prefix(['s', 'S']) {
            Some(rest) => (true, rest),
            None => (false, rest),
        };
        let mut chars = rest.chars();
        let radix = match chars.next().map(|c| c.to_ascii_lowercase()) {
            Some('b') => 2,
            Some('o') => 8,
            Some('d') => 10,
            Some('h') => 16,
            _ => return self.err(n, format!("malformed literal `{text}`")),
        };
        let digits = chars.as_str();
        if digits.contains(['x', 'X', 'z', 'Z', '?']) {
            return self.err(
                n,
                format!("literal `{text}` with x/z digits not supported (2-state)"),
            );
        }
        let value = BigUint::parse_bytes(digits.as_bytes(), radix)
            .ok_or_else(|| Error::new(self.span(n), format!("malformed literal `{text}`")))?;
        let value = BitVec::new(width, value).ok_or_else(|| {
            Error::new(
                self.span(n),
                format!("literal `{text}` does not fit in {width} bits"),
            )
        })?;
        Ok(ExprKind::Lit {
            ty: Ty { width, signed },
            value,
        })
    }

    fn unop(&self, op: &sv::UnaryOperator) -> Result<UnOp> {
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

    fn binop(&self, op: &sv::BinaryOperator) -> Result<BinOp> {
        use BinOp::*;
        Ok(match self.text(op).as_str() {
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
            "**" => return self.err(op, "operator `**` not supported"),
            s => return self.err(op, format!("operator `{s}` not supported")),
        })
    }
}

/// An operator of a flattened chain: a binary operator, or `? then :` of a
/// conditional, which acts as a right-associative binary operator between
/// condition and else branch.
enum Op {
    Bin(BinOp, Span),
    Cond(Expr, Span),
}

/// The binding strength of an operator (higher binds tighter), after
/// IEEE 1800-2017 Table 11-2.
pub fn prec(op: Option<BinOp>) -> u8 {
    use BinOp::*;
    match op {
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

/// Builds the tree of a flattened chain by precedence (shunting-yard).
/// Binary operators associate to the left, `?:` to the right.
fn reassociate(operands: Vec<Expr>, ops: Vec<Op>) -> Expr {
    fn reduce(out: &mut Vec<Expr>, op: Op) {
        let b = out.pop().expect("operand");
        let a = out.pop().expect("operand");
        let (kind, span) = match op {
            Op::Bin(op, span) => (ExprKind::Binary(op, Box::new(a), Box::new(b)), span),
            Op::Cond(then, span) => (
                ExprKind::Cond(Box::new(a), Box::new(then), Box::new(b)),
                span,
            ),
        };
        out.push(Expr { kind, span });
    }
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
