/*!
# From source text to the AST

Parses with `sv-parser` into a [`Design`], and lowers the syntax tree of
one of its modules into an [`ast::Module`], rejecting everything outside
the accepted subset with a located error.

A module is lowered once for each set of parameter values: the values are
substituted while lowering, so the result has no parameters left. Ranges,
replication counts, select bounds and cast sizes are [`constant`]
expressions. In other expressions, a parameter becomes a constant of its
type and an unsized literal a constant of type `int` (`5`) or of 32
unsigned bits (`'hff`); `'0` and `'1` take the width of their context. An
integer constant (built from unsized literals and `int` parameters with
`+ - * / % **` and `$clog2`) keeps its operators, so that
[`sizing`](crate::sizing) evaluates it at the type of its context; only
`**` and `$clog2` are folded here. An unsized literal is no item of a
concatenation (IEEE 1800 forbids it). Instances keep their parameter
values and connections; [`elab`](crate::elab) elaborates the modules they
name.

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
    BinOp, CaseItem, Cast, Conn, Const, Decl, Dir, Edge, Error, Expr, ExprKind, Index, Instance,
    Item, Lvalue, LvalueKind, Module, Name, Override, Param, Port, Reset, Select, Span, Stmt,
    StmtKind, Ty, UnOp,
};
use crate::constant::{self, Failure, Literal};
use crate::value::BitVec;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use sv_parser as sv;
use sv_parser::{Iter, RefNode, RefNodes, SyntaxTree};

type Result<T> = std::result::Result<T, Error>;

const VARIABLE_TARGET: &str = "a continuous assignment needs a constant select: an `assign` \
     drives fixed bits (assign the bit in an `always_comb` instead)";

/// Parses a file and elaborates its top module with the default parameter
/// values.
pub fn parse_file(path: impl AsRef<Path>) -> Result<Module> {
    let d = parse_design_file(path)?;
    d.module(&d.top()?, &[])
}

/// Parses `source` and elaborates its top module with the default
/// parameter values; `path` is used for messages of the preprocessor only.
pub fn parse(source: &str, path: impl AsRef<Path>) -> Result<Module> {
    let d = parse_design(source, path)?;
    d.module(&d.top()?, &[])
}

/// Parses a file.
pub fn parse_design_file(path: impl AsRef<Path>) -> Result<Design> {
    let path = path.as_ref();
    let source = std::fs::read_to_string(path)
        .map_err(|e| Error::new(Span::default(), format!("{}: {e}", path.display())))?;
    parse_design(&source, path)
}

/// Parses `source`; `path` is used for messages of the preprocessor only.
pub fn parse_design(source: &str, path: impl AsRef<Path>) -> Result<Design> {
    let defines: HashMap<String, Option<sv::Define>> = HashMap::new();
    let includes: [PathBuf; 0] = [];
    let mut lines = vec![0];
    lines.extend(source.match_indices('\n').map(|(i, _)| i + 1));
    let (tree, _) =
        sv::parse_sv_str(source, path, &defines, &includes, false, false).map_err(|e| match e {
            sv::Error::Parse(Some((_, offset))) => Error::new(at(&lines, offset), "syntax error"),
            e => Error::new(Span::default(), e.to_string()),
        })?;
    let d = Design { tree, lines };
    let mut seen: BTreeMap<Name, Span> = BTreeMap::new();
    let cx = d.cx(&[]);
    for m in d.decls()? {
        let id = &m.nodes.0.nodes.3;
        let name = cx.identifier(&id.nodes.0)?;
        if let Some(prev) = seen.insert(name.clone(), cx.span(id)) {
            return cx.err(id, format!("module `{name}` already declared at {prev}"));
        }
    }
    Ok(d)
}

/// The position of a byte offset, with `lines` the offsets at which lines
/// start.
fn at(lines: &[usize], offset: usize) -> Span {
    let line = lines.partition_point(|&start| start <= offset);
    Span {
        line,
        col: offset - lines[line - 1] + 1,
    }
}

/// A parsed source file. Its modules are elaborated on demand: once for
/// each set of parameter values.
pub struct Design {
    tree: SyntaxTree,
    lines: Vec<usize>,
}

impl Design {
    fn cx<'t>(&'t self, overrides: &'t [Override]) -> Cx<'t> {
        Cx {
            tree: &self.tree,
            lines: &self.lines,
            params: RefCell::new(BTreeMap::new()),
            overrides,
            used: RefCell::new(BTreeSet::new()),
        }
    }

    /// The module declarations, in source order.
    fn decls(&self) -> Result<Vec<&sv::ModuleDeclarationAnsi>> {
        let cx = self.cx(&[]);
        let st = (&self.tree)
            .into_iter()
            .find_map(|n| match n {
                RefNode::SourceText(x) => Some(x),
                _ => None,
            })
            .ok_or_else(|| Error::new(Span::default(), "empty source"))?;
        let (_, timeunits, descriptions) = &st.nodes;
        cx.absent(timeunits, "timeunits declaration")?;
        let mut modules = Vec::new();
        for d in descriptions {
            match d {
                sv::Description::ModuleDeclaration(m) => match m.as_ref() {
                    sv::ModuleDeclaration::Ansi(m) => modules.push(m.as_ref()),
                    sv::ModuleDeclaration::Nonansi(_) => {
                        return cx.err(m.as_ref(), "non-ANSI port list not supported");
                    }
                    m => return cx.unsupported(m),
                },
                d => return cx.unsupported(d),
            }
        }
        if modules.is_empty() {
            return Err(Error::new(Span::default(), "no module"));
        }
        Ok(modules)
    }

    fn find(&self, name: &str) -> Option<&sv::ModuleDeclarationAnsi> {
        let cx = self.cx(&[]);
        let decls = self.decls().ok()?;
        decls
            .into_iter()
            .find(|m| cx.text(&m.nodes.0.nodes.3) == name)
    }

    /// The names of the modules, in source order.
    pub fn names(&self) -> Vec<Name> {
        let cx = self.cx(&[]);
        let decls = self.decls().unwrap_or_default();
        decls
            .into_iter()
            .map(|m| cx.text(&m.nodes.0.nodes.3))
            .collect()
    }

    /// Whether the file declares the module `name`.
    pub fn has(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// The top module: the one no module instantiates.
    pub fn top(&self) -> Result<Name> {
        let cx = self.cx(&[]);
        let mut instantiated = BTreeSet::new();
        for m in self.decls()? {
            for n in Iter::new(m.into()) {
                if let RefNode::ModuleInstantiation(i) = n {
                    instantiated.insert(cx.text(&i.nodes.0));
                }
            }
        }
        let tops: Vec<Name> = self
            .names()
            .into_iter()
            .filter(|n| !instantiated.contains(n))
            .collect();
        match tops.as_slice() {
            [top] => Ok(top.clone()),
            [] => Err(Error::new(
                Span::default(),
                "no top module: every module is instantiated (recursive instantiation \
                 is not supported)",
            )),
            _ => Err(Error::new(
                Span::default(),
                format!(
                    "several top modules ({}): choose one",
                    tops.iter()
                        .map(|t| format!("`{t}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )),
        }
    }

    /// Elaborates the module `name` with the parameter values `overrides`
    /// (the others take their defaults). The instances in it are not
    /// elaborated (see [`elab`](crate::elab)).
    pub fn module(&self, name: &str, overrides: &[Override]) -> Result<Module> {
        let m = self
            .find(name)
            .ok_or_else(|| Error::new(Span::default(), format!("no module `{name}`")))?;
        self.cx(overrides).module(m)
    }
}

struct Cx<'t> {
    tree: &'t SyntaxTree,
    lines: &'t [usize],
    /// The parameters declared so far, with their values.
    params: RefCell<BTreeMap<Name, Const>>,
    /// The parameter values given by the instance.
    overrides: &'t [Override],
    /// The overrides used so far.
    used: RefCell<BTreeSet<Name>>,
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
                RefNode::Locate(l) => self.tree.get_origin(l).map(|(_, off)| at(self.lines, off)),
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

    fn module(&self, m: &sv::ModuleDeclarationAnsi) -> Result<Module> {
        let (header, timeunits, items, _, _) = &m.nodes;
        let (attrs, _, lifetime, name, imports, params, ports, _) = &header.nodes;
        self.absent(attrs, "attribute")?;
        self.absent(lifetime, "lifetime")?;
        self.absent(imports, "package import")?;
        self.absent(timeunits, "timeunits declaration")?;
        let mut module = Module {
            name: self.identifier(&name.nodes.0)?,
            params: Vec::new(),
            ports: Vec::new(),
            decls: Vec::new(),
            items: Vec::new(),
        };
        if let Some(params) = params {
            self.parameter_port_list(params, &mut module)?;
        }
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
        // with a parameter port list, a `parameter` in the body is local
        let body_overridable = params.is_none();
        for item in items {
            self.module_item(item, &mut module, body_overridable)?;
        }
        for o in self.overrides {
            if !self.used.borrow().contains(&o.name) {
                let msg = if module.params.iter().any(|p| p.name == o.name) {
                    format!("parameter `{}` of `{}` is local", o.name, module.name)
                } else {
                    format!("`{}` has no parameter `{}`", module.name, o.name)
                };
                return Err(Error::new(o.span, msg));
            }
        }
        Ok(module)
    }

    /// `#(parameter N = 8, localparam M = N + 1, …)` or `#(N = 8, …)`.
    fn parameter_port_list(&self, l: &sv::ParameterPortList, m: &mut Module) -> Result<()> {
        match l {
            sv::ParameterPortList::Assignment(a) => {
                let (first, rest) = &a.nodes.1.nodes.1;
                self.param_assignments(None, first, true, m)?;
                for (_, d) in rest {
                    self.parameter_port_declaration(d, m)?;
                }
                Ok(())
            }
            sv::ParameterPortList::Declaration(d) => {
                for d in d.nodes.1.nodes.1.contents() {
                    self.parameter_port_declaration(d, m)?;
                }
                Ok(())
            }
            sv::ParameterPortList::Empty(_) => Ok(()),
        }
    }

    fn parameter_port_declaration(
        &self,
        d: &sv::ParameterPortDeclaration,
        m: &mut Module,
    ) -> Result<()> {
        match d {
            sv::ParameterPortDeclaration::ParameterDeclaration(p) => {
                self.parameter_declaration(p, true, m)
            }
            sv::ParameterPortDeclaration::LocalParameterDeclaration(p) => match p.as_ref() {
                sv::LocalParameterDeclaration::Param(p) => {
                    let (_, ty, list) = &p.nodes;
                    let ty = self.param_type(ty)?;
                    self.param_assignments(ty, list, false, m)
                }
                p => self.err(p, "type parameters not supported"),
            },
            sv::ParameterPortDeclaration::ParamList(p) => {
                let (ty, list) = &p.nodes;
                let ty = self.param_data_type(ty)?;
                self.param_assignments(Some(ty), list, true, m)
            }
            d => self.err(d, "type parameters not supported"),
        }
    }

    fn parameter_declaration(
        &self,
        p: &sv::ParameterDeclaration,
        overridable: bool,
        m: &mut Module,
    ) -> Result<()> {
        match p {
            sv::ParameterDeclaration::Param(p) => {
                let (_, ty, list) = &p.nodes;
                let ty = self.param_type(ty)?;
                self.param_assignments(ty, list, overridable, m)
            }
            p => self.err(p, "type parameters not supported"),
        }
    }

    /// The declared type of a parameter, `None` if it has none (it then
    /// takes the type of its value).
    fn param_type(&self, t: &sv::DataTypeOrImplicit) -> Result<Option<Ty>> {
        match t {
            sv::DataTypeOrImplicit::DataType(t) => self.param_data_type(t).map(Some),
            sv::DataTypeOrImplicit::ImplicitDataType(t) => {
                let (signing, dims) = &t.nodes;
                if signing.is_none() && dims.is_empty() {
                    Ok(None)
                } else {
                    self.vector(signing, dims).map(Some)
                }
            }
        }
    }

    /// `int`, or a vector type.
    fn param_data_type(&self, t: &sv::DataType) -> Result<Ty> {
        match t {
            sv::DataType::Atom(a) => match &a.nodes {
                (sv::IntegerAtomType::Int(_), None) => Ok(Ty::INT),
                _ => self.err(
                    t,
                    format!(
                        "parameter type `{}` not supported, use `int` or `logic`",
                        self.text(t)
                    ),
                ),
            },
            t => self.data_type(t),
        }
    }

    /// `N = v, M = w, …` of a parameter declaration with type `ty` (`None`:
    /// the type of the value). An overridable parameter takes the value of
    /// the instance if it gives one.
    fn param_assignments(
        &self,
        ty: Option<Ty>,
        list: &sv::ListOfParamAssignments,
        overridable: bool,
        m: &mut Module,
    ) -> Result<()> {
        for a in list.nodes.0.contents() {
            let (id, dims, default) = &a.nodes;
            self.absent(dims, "unpacked dimension")?;
            let name = self.identifier(&id.nodes.0)?;
            let span = self.span(id);
            if let Some(prev) = m.params.iter().find(|p| p.name == name) {
                return self.err(id, format!("`{name}` already declared at {}", prev.span));
            }
            let given = self.overrides.iter().find(|o| o.name == name);
            let value = match (given, default) {
                (Some(o), _) if overridable => {
                    self.used.borrow_mut().insert(name.clone());
                    o.value.clone()
                }
                (_, Some((_, e))) => self.const_expr(e)?,
                (_, None) => {
                    return self.err(
                        id,
                        format!(
                            "parameter `{name}` has no value: give it a default or set it \
                             in the instance"
                        ),
                    );
                }
            };
            // a typed parameter takes its value as in an assignment: extended
            // as the value's signedness says, or truncated
            let ty = ty.unwrap_or(value.ty);
            let value = Const {
                ty,
                value: constant::resize(ty.width, &value).value,
            };
            self.params.borrow_mut().insert(name.clone(), value.clone());
            m.params.push(Param {
                name,
                ty,
                value: value.value,
                overridable,
                span,
            });
        }
        Ok(())
    }

    /// The value of a constant expression (see [`constant`]).
    fn const_expr<'a>(&self, e: impl Into<RefNodes<'a>> + Copy) -> Result<Const> {
        let text = self.text(e);
        constant::eval(&text, &self.params.borrow()).or_else(|f| match f {
            Failure::NotConstant(why) => self.err(
                e,
                format!("expected a constant expression, got `{text}`: {why}"),
            ),
            Failure::Error(msg) => self.err(e, msg),
        })
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

    /// A non-negative integer constant (unsized literals and `int`
    /// parameters, see [`constant`]).
    fn const_int<'a>(&self, e: impl Into<RefNodes<'a>> + Copy) -> Result<usize> {
        let text = self.text(e);
        let c = constant::eval(&text, &self.params.borrow()).or_else(|f| match f {
            Failure::NotConstant(why) => self.err(
                e,
                format!("expected a constant integer, got `{text}`: {why}"),
            ),
            Failure::Error(msg) => self.err(e, msg),
        })?;
        match constant::as_int(&c) {
            Some(n) if n >= 0 => Ok(n as usize),
            Some(n) => self.err(e, format!("`{text}` is negative ({n})")),
            None => self.err(
                e,
                format!(
                    "`{text}` has type `{}`, expected an integer (unsized literals and \
                     `int` parameters)",
                    c.ty
                ),
            ),
        }
    }

    /// An item of the module body; `overridable` says whether a
    /// `parameter` there can be set by instances.
    fn module_item(
        &self,
        item: &sv::NonPortModuleItem,
        m: &mut Module,
        overridable: bool,
    ) -> Result<()> {
        let sv::NonPortModuleItem::ModuleOrGenerateItem(item) = item else {
            if let sv::NonPortModuleItem::GenerateRegion(g) = item {
                return self.err(g.as_ref(), "generate regions not supported");
            }
            return self.unsupported(item);
        };
        let item = match item.as_ref() {
            sv::ModuleOrGenerateItem::ModuleItem(i) => i,
            sv::ModuleOrGenerateItem::Module(i) => {
                let (attrs, i) = &i.nodes;
                self.absent(attrs, "attribute")?;
                return self.instantiation(i, m);
            }
            sv::ModuleOrGenerateItem::Parameter(p) => {
                return self.err(p.as_ref(), "`defparam` not supported, use `#(.N(…))`");
            }
            i => return self.unsupported(i),
        };
        let (attrs, item) = &item.nodes;
        self.absent(attrs, "attribute")?;
        match item {
            sv::ModuleCommonItem::ModuleOrGenerateItemDeclaration(d) => match d.as_ref() {
                sv::ModuleOrGenerateItemDeclaration::PackageOrGenerateItemDeclaration(d) => {
                    self.declaration(d, m, overridable)
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

    /// `module #(.N(v), …) u1 (.a(x), …), u2 (…);`: parameter values and
    /// ports by name only.
    fn instantiation(&self, i: &sv::ModuleInstantiation, m: &mut Module) -> Result<()> {
        let (module, values, list, _) = &i.nodes;
        let module = self.identifier(&module.nodes.0)?;
        let mut params = Vec::new();
        if let Some(values) = values {
            match &values.nodes.1.nodes.1 {
                None => {}
                Some(sv::ListOfParameterAssignments::Ordered(o)) => {
                    return self.err(
                        o.as_ref(),
                        "positional parameter values not supported, write `#(.N(…))`",
                    );
                }
                Some(sv::ListOfParameterAssignments::Named(n)) => {
                    for a in n.nodes.0.contents() {
                        let (_, id, value) = &a.nodes;
                        let name = self.identifier(&id.nodes.0)?;
                        let Some(e) = &value.nodes.1 else {
                            return self.err(a, format!("no value for parameter `{name}`"));
                        };
                        if params.iter().any(|o: &Override| o.name == name) {
                            return self.err(id, format!("parameter `{name}` given twice"));
                        }
                        params.push(Override {
                            name,
                            value: self.const_expr(e)?,
                            span: self.span(id),
                        });
                    }
                }
            }
        }
        for h in list.contents() {
            let (name, conns) = &h.nodes;
            let (id, dims) = &name.nodes;
            self.absent(dims, "instance array")?;
            let mut out = Vec::new();
            match &conns.nodes.1 {
                None => {}
                Some(sv::ListOfPortConnections::Ordered(o)) => {
                    let empty = o.nodes.0.contents().iter().all(|c| !has_tokens(*c));
                    if !empty {
                        return self.err(
                            o.as_ref(),
                            "positional port connections not supported, write `.port(signal)`",
                        );
                    }
                }
                Some(sv::ListOfPortConnections::Named(n)) => {
                    for c in n.nodes.0.contents() {
                        let c = match c {
                            sv::NamedPortConnection::Identifier(c) => c,
                            sv::NamedPortConnection::Asterisk(c) => {
                                return self.err(
                                    c.as_ref(),
                                    "`.*` not supported, connect each port by name",
                                );
                            }
                        };
                        let (attrs, _, port, actual) = &c.nodes;
                        self.absent(attrs, "attribute")?;
                        let name = self.identifier(&port.nodes.0)?;
                        let span = self.span(port);
                        let actual = match actual {
                            // `.a` is `.a(a)`
                            None => Some(Expr {
                                kind: ExprKind::Var(name.clone()),
                                span,
                            }),
                            Some(p) => match &p.nodes.1 {
                                None => None,
                                Some(e) => Some(self.expr(e)?),
                            },
                        };
                        out.push(Conn {
                            port: name,
                            actual,
                            span,
                            formal: None,
                        });
                    }
                }
            }
            m.items.push(Item::Instance(Instance {
                module: module.clone(),
                name: self.identifier(&id.nodes.0)?,
                params: params.clone(),
                conns: out,
                span: self.span(id),
            }));
        }
        Ok(())
    }

    /// `always_comb stmt`, or `always_ff @(posedge clk) stmt` (or
    /// `negedge`): one edge of one clock signal, optionally with an
    /// asynchronous reset `@(posedge clk or posedge rst) if (rst) … else …`
    /// (or `negedge rst_n` and `if (!rst_n)`; events in either order,
    /// separated by `or` or `,`). The reset is the event the `if` tests.
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
            return self.err(
                stmt,
                "`always_ff` needs `@(posedge <clock>)` or `@(negedge <clock>)`",
            );
        };
        let (ctrl, body) = &t.nodes;
        let event = match ctrl {
            sv::ProceduralTimingControl::EventControl(e) => match e.as_ref() {
                sv::EventControl::EventExpression(e) => &e.nodes.1.nodes.1,
                e => {
                    return self.err(
                        e,
                        "`always_ff` needs `@(posedge <clock>)` or `@(negedge <clock>)`",
                    );
                }
            },
            c => {
                return self.err(
                    c,
                    "`always_ff` needs `@(posedge <clock>)` or `@(negedge <clock>)`",
                );
            }
        };
        let mut events = Vec::new();
        self.events(event, &mut events)?;
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
                if a.0 == b.0 {
                    return Err(Error::new(
                        b.2,
                        format!("`{}` appears twice in the event list", b.0),
                    ));
                }
                let (name, low, cond_span) = reset_test(&body).ok_or_else(|| {
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
                (c.0.clone(), c.1, c.2, Some(reset))
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
        })
    }

    /// Adds the events of an event list (`a or b`, `a, b`, parentheses) to
    /// `out`, in source order.
    fn events<'e>(
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

    /// One event `posedge x` or `negedge x` of an `always_ff` block: the
    /// name, the edge and the position of the name.
    fn event(&self, event: &sv::EventExpressionExpression) -> Result<(Name, Edge, Span)> {
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

    fn declaration(
        &self,
        d: &sv::PackageOrGenerateItemDeclaration,
        m: &mut Module,
        overridable: bool,
    ) -> Result<()> {
        match d {
            sv::PackageOrGenerateItemDeclaration::LocalParameterDeclaration(p) => {
                return match &p.0 {
                    sv::LocalParameterDeclaration::Param(p) => {
                        let (_, ty, list) = &p.nodes;
                        let ty = self.param_type(ty)?;
                        self.param_assignments(ty, list, false, m)
                    }
                    p => self.err(p, "type parameters not supported"),
                };
            }
            sv::PackageOrGenerateItemDeclaration::ParameterDeclaration(p) => {
                return self.parameter_declaration(&p.0, overridable, m);
            }
            _ => {}
        }
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
        let index = |e: &sv::ConstantExpression| {
            let text = self.text(e);
            match constant::eval(&text, &self.params.borrow()) {
                Ok(_) => self.const_int(e).map(Index::Const),
                Err(Failure::NotConstant(_)) => self.err(e, VARIABLE_TARGET),
                Err(Failure::Error(msg)) => self.err(e, msg),
            }
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

    /// A bit position: an integer constant, or any expression.
    fn index(&self, e: &sv::Expression) -> Result<Index> {
        match self.leaf_expr(e)? {
            Leaf::Int { value, span, .. } if value < 0 => {
                Err(Error::new(span, format!("negative index {value}")))
            }
            Leaf::Int { value, .. } => Ok(Index::Const(value as usize)),
            Leaf::Expr(e) => Ok(Index::Expr(Box::new(e))),
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
        let l = self.leaf_expr(e)?;
        self.leaf_to_expr(l)
    }

    /// An expression, as a [`Leaf`]: an integer constant stays a number.
    fn leaf_expr(&self, e: &sv::Expression) -> Result<Leaf> {
        let mut operands = Vec::new();
        let mut ops = Vec::new();
        self.flatten(e, &mut operands, &mut ops)?;
        self.reassociate(operands, ops)
    }

    /// The expression of a leaf.
    fn leaf_to_expr(&self, l: Leaf) -> Result<Expr> {
        match l {
            Leaf::Expr(e) | Leaf::Int { expr: e, .. } => Ok(e),
        }
    }

    /// Flattens a chain of binary operators and conditionals into its
    /// operands and operators, in source order.
    fn flatten(
        &self,
        e: &sv::Expression,
        operands: &mut Vec<Leaf>,
        ops: &mut Vec<Op>,
    ) -> Result<()> {
        match e {
            sv::Expression::Binary(b) => {
                let (l, op, attrs, r) = &b.nodes;
                self.absent(attrs, "attribute")?;
                self.flatten(l, operands, ops)?;
                ops.push(self.binop(op)?);
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
    fn operand(&self, e: &sv::Expression) -> Result<Leaf> {
        match e {
            sv::Expression::Primary(p) => self.primary(p),
            sv::Expression::Unary(u) => {
                let (op, attrs, p) = &u.nodes;
                self.absent(attrs, "attribute")?;
                let span = self.span(op);
                let op = self.unop(op)?;
                match (op, self.primary(p)?) {
                    (UnOp::Plus | UnOp::Neg, Leaf::Int { value, expr, .. }) => {
                        let value = match op {
                            UnOp::Neg => {
                                constant::int_neg(value).map_err(|m| Error::new(span, m))?
                            }
                            _ => value,
                        };
                        let kind = ExprKind::Unary(op, Box::new(expr));
                        Ok(Leaf::Int {
                            value,
                            expr: Expr { kind, span },
                            span,
                        })
                    }
                    (op, l) => {
                        let kind = ExprKind::Unary(op, Box::new(self.leaf_to_expr(l)?));
                        Ok(Leaf::Expr(Expr { kind, span }))
                    }
                }
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

    fn primary(&self, p: &sv::Primary) -> Result<Leaf> {
        let span = self.span(p);
        let kind = match p {
            sv::Primary::PrimaryLiteral(l) => match l.as_ref() {
                sv::PrimaryLiteral::Number(n) => match n.as_ref() {
                    sv::Number::IntegralNumber(n) => return self.literal(n),
                    n => return self.err(n, "real numbers not supported"),
                },
                sv::PrimaryLiteral::UnbasedUnsizedLiteral(l) => {
                    let kind = match self.text(l.as_ref()).trim() {
                        "'0" => ExprKind::Fill(false),
                        "'1" => ExprKind::Fill(true),
                        t => {
                            return self.err(
                                p,
                                format!("literal `{t}` with x/z digits not supported (2-state)"),
                            );
                        }
                    };
                    return Ok(Leaf::Expr(Expr { kind, span }));
                }
                l => return self.unsupported(l),
            },
            sv::Primary::Hierarchical(h) => {
                let (scope, id, select) = &h.nodes;
                self.absent(scope, "package or class scope")?;
                let id = self.identifier(self.simple_hierarchical(id)?)?;
                let param = self.params.borrow().get(&id).cloned();
                match (self.select(select)?, param) {
                    (None, Some(c)) => {
                        return Ok(match constant::as_int(&c) {
                            Some(value) => Leaf::int(value, id, Ty::INT, span),
                            None => Leaf::Expr(Expr {
                                kind: ExprKind::Const {
                                    text: id,
                                    ty: c.ty,
                                    value: c.value,
                                },
                                span,
                            }),
                        });
                    }
                    (Some(_), Some(_)) => {
                        return self.err(p, format!("select of parameter `{id}` not supported"));
                    }
                    (None, None) => ExprKind::Var(id),
                    (Some(sel), None) => ExprKind::Select(id, sel),
                }
            }
            sv::Primary::MintypmaxExpression(m) => match &m.nodes.0.nodes.1 {
                sv::MintypmaxExpression::Expression(e) => return self.leaf_expr(e),
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
                    // `W'(e)` parses as a cast to a type named `W`
                    sv::CastingType::SimpleType(t)
                        if self.params.borrow().contains_key(&self.text(t.as_ref())) =>
                    {
                        let n = self.const_int(t.as_ref())?;
                        if n == 0 {
                            return self.err(p, "cast to 0 bits");
                        }
                        Cast::Size(n)
                    }
                    t => {
                        return self.err(t, format!("cast to `{}` not supported", self.text(t)));
                    }
                };
                ExprKind::Cast(cast, Box::new(self.expr(&e.nodes.1)?))
            }
            sv::Primary::FunctionSubroutineCall(c) => return self.call(&c.nodes.0, span),
            p => return self.unsupported(p),
        };
        Ok(Leaf::Expr(Expr { kind, span }))
    }

    /// The items of `{a, b, …}`. An unsized literal is no item: IEEE 1800
    /// forbids it (simulators give it 32 bits, or 1 for `'1`).
    fn concatenation(&self, c: &sv::Concatenation) -> Result<Vec<Expr>> {
        let item = |e: &sv::Expression| {
            let e = self.expr(e)?;
            let literal = match &e.kind {
                ExprKind::Const { text, .. } => {
                    text.starts_with(|c: char| c.is_ascii_digit() || c == '\'')
                }
                ExprKind::Fill(_) => true,
                _ => false,
            };
            if literal {
                return Err(Error::new(
                    e.span,
                    format!(
                        "unsized literal `{e}` in a concatenation: write its width (e.g. `8'd1`)"
                    ),
                ));
            }
            Ok(e)
        };
        c.nodes.0.nodes.1.contents().into_iter().map(item).collect()
    }

    /// A call; only `$signed(e)` and `$unsigned(e)`, which are casts, and
    /// `$clog2(n)` of an integer constant.
    fn call(&self, c: &sv::SubroutineCall, span: Span) -> Result<Leaf> {
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
        let name = self.text(name);
        let [Some(e)] = args.as_slice() else {
            return self.err(c, format!("`{name}` takes one argument"));
        };
        let cast = match name.as_str() {
            "$signed" => Cast::Signed,
            "$unsigned" => Cast::Unsigned,
            "$clog2" => {
                return match self.leaf_expr(e)? {
                    Leaf::Int { value, expr, .. } => {
                        let value = constant::clog2(value).map_err(|m| Error::new(span, m))?;
                        Ok(Leaf::int(value, format!("$clog2({expr})"), Ty::INT, span))
                    }
                    Leaf::Expr(_) => self.err(
                        c,
                        "`$clog2` of an integer constant only (unsized literals and `int` \
                         parameters)",
                    ),
                };
            }
            _ => return unsupported(),
        };
        let kind = ExprKind::Cast(cast, Box::new(self.expr(e)?));
        Ok(Leaf::Expr(Expr { kind, span }))
    }

    /// A literal: sized `N'[s]<base><digits>`, without x/z digits (values
    /// are 2-state) and with a value that fits in `N` bits, or an unsized
    /// decimal, which is an integer.
    fn literal(&self, n: &sv::IntegralNumber) -> Result<Leaf> {
        let text: String = self
            .text(n)
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '_')
            .collect();
        let span = self.span(n);
        match constant::literal(&text).map_err(|m| Error::new(span, m))? {
            Literal::Sized(ty, value) => Ok(Leaf::Expr(Expr {
                kind: ExprKind::Lit { ty, value },
                span,
            })),
            Literal::Int(value) => Ok(Leaf::int(value, text, Ty::INT, span)),
            // `'sd5` is an integer, `'d5` a 32-bit unsigned vector
            Literal::Unsized(ty, value) if ty.signed => {
                let v = num_traits::ToPrimitive::to_i64(&value.signed()).expect("32 bits");
                Ok(Leaf::int(v, text, ty, span))
            }
            Literal::Unsized(ty, value) => Ok(Leaf::Expr(Expr {
                kind: ExprKind::Const { text, ty, value },
                span,
            })),
        }
    }

    /// Builds the tree of a flattened chain by precedence (shunting-yard).
    /// Binary operators associate to the left, `?:` to the right. `+ - * /
    /// % **` on integer constants fold into one integer constant.
    fn reassociate(&self, operands: Vec<Leaf>, ops: Vec<Op>) -> Result<Leaf> {
        let mut operands = operands.into_iter();
        let mut out = vec![operands.next().expect("operand")];
        let mut stack: Vec<Op> = Vec::new();
        for (op, operand) in ops.into_iter().zip(operands) {
            let right_assoc = matches!(op, Op::Cond(..));
            while let Some(top) = stack.last() {
                if top.prec() > op.prec() || (top.prec() == op.prec() && !right_assoc) {
                    let top = stack.pop().unwrap();
                    self.reduce(&mut out, top)?;
                } else {
                    break;
                }
            }
            stack.push(op);
            out.push(operand);
        }
        while let Some(op) = stack.pop() {
            self.reduce(&mut out, op)?;
        }
        assert_eq!(out.len(), 1);
        Ok(out.pop().unwrap())
    }

    fn reduce(&self, out: &mut Vec<Leaf>, op: Op) -> Result<()> {
        let b = out.pop().expect("operand");
        let a = out.pop().expect("operand");
        let symbol = match &op {
            Op::Bin(BinOp::Add, _) => Some("+"),
            Op::Bin(BinOp::Sub, _) => Some("-"),
            Op::Bin(BinOp::Mul, _) => Some("*"),
            Op::Bin(BinOp::Div, _) => Some("/"),
            Op::Bin(BinOp::Mod, _) => Some("%"),
            Op::Pow(_) => Some("**"),
            _ => None,
        };
        if let (Some(symbol), Leaf::Int { value: x, .. }, Leaf::Int { value: y, .. }) =
            (symbol, &a, &b)
        {
            let at = op.span();
            let value = constant::int_op(symbol, *x, *y).map_err(|m| Error::new(at, m))?;
            let (Leaf::Int { expr: ea, span, .. }, Leaf::Int { expr: eb, .. }) = (a, b) else {
                unreachable!("integer constants");
            };
            out.push(match op {
                // `**` is folded: there is no operator for it
                Op::Pow(_) => Leaf::int(value, format!("({ea} ** {eb})"), Ty::INT, span),
                Op::Bin(bin, at) => Leaf::Int {
                    value,
                    expr: Expr {
                        kind: ExprKind::Binary(bin, Box::new(ea), Box::new(eb)),
                        span: at,
                    },
                    span,
                },
                Op::Cond(..) => unreachable!("not an arithmetic operator"),
            });
            return Ok(());
        }
        let (kind, span) = match op {
            Op::Bin(op, span) => (
                ExprKind::Binary(
                    op,
                    Box::new(self.leaf_to_expr(a)?),
                    Box::new(self.leaf_to_expr(b)?),
                ),
                span,
            ),
            Op::Pow(span) => {
                return Err(Error::new(
                    span,
                    "`**` only between integer constants (unsized literals and `int` parameters)",
                ));
            }
            Op::Cond(then, span) => (
                ExprKind::Cond(
                    Box::new(self.leaf_to_expr(a)?),
                    Box::new(then),
                    Box::new(self.leaf_to_expr(b)?),
                ),
                span,
            ),
        };
        out.push(Leaf::Expr(Expr { kind, span }));
        Ok(())
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

    fn binop(&self, op: &sv::BinaryOperator) -> Result<Op> {
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
                "**" => return Ok(Op::Pow(span)),
                s => return self.err(op, format!("operator `{s}` not supported")),
            },
            span,
        ))
    }
}

/// An operand while an expression is built: an integer constant (unsized
/// literals and `int` parameters, see [`constant`]) keeps its exact value
/// beside its expression until it meets anything else, so that `N - 1`
/// can be an integer (a range bound, a replication count) where one is
/// needed.
enum Leaf {
    Expr(Expr),
    Int {
        value: i64,
        /// The expression as written, with parameters and unsized
        /// literals as [`ExprKind::Const`] leaves (`**` and `$clog2`
        /// folded into one): in a value expression it is sized like any
        /// other, so that its width and signedness come from the context.
        expr: Expr,
        span: Span,
    },
}

impl Leaf {
    /// The expression of an integer constant: a constant leaf.
    fn int(value: i64, text: String, ty: Ty, span: Span) -> Leaf {
        let kind = ExprKind::Const {
            text,
            ty,
            value: BitVec::from_signed(ty.width, value),
        };
        Leaf::Int {
            value,
            expr: Expr { kind, span },
            span,
        }
    }
}

/// What the `if` of a block with an asynchronous reset tests: the body is a
/// single `if` (possibly inside `begin … end`) whose condition is `x`, `!x`
/// or `~x` for a name `x`. Gives `x`, whether it is negated, and the
/// position of the condition.
fn reset_test(body: &Stmt) -> Option<(&Name, bool, Span)> {
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

/// An operator of a flattened chain: a binary operator, `**`, or `? then :`
/// of a conditional, which acts as a right-associative binary operator
/// between condition and else branch.
enum Op {
    Bin(BinOp, Span),
    Pow(Span),
    Cond(Expr, Span),
}

/// The binding strength of an operator (higher binds tighter), after
/// IEEE 1800-2017 Table 11-2 (`**` binds tighter still).
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
            Op::Pow(_) => 11,
            Op::Cond(..) => prec(None),
        }
    }

    fn span(&self) -> Span {
        match self {
            Op::Bin(_, s) | Op::Pow(s) | Op::Cond(_, s) => *s,
        }
    }
}
