//! Modules, parameters, ports, declarations, instances and initial values.

use super::arrays::{Load, readmem, resolve};
use super::{Cx, LARGEST, Result, VARIABLE_TARGET, has_tokens};
use crate::ast::{
    self, Array, Conn, Const, Decl, Dir, Elem, Elems, Error, Expr, ExprKind, Index, Init, Instance,
    Item, Lvalue, LvalueKind, Module, Name, Override, Param, Port, Select, Span, Ty,
};
use crate::constant::{self, Failure};
use std::collections::BTreeMap;
use sv_parser as sv;
use sv_parser::RefNodes;

/// If `text` is a literal that is x in every bit (`'x`, `4'bxxxx`, `1'hx`,
/// in parentheses or not), its size: `None` if unsized.
pub(super) fn unknown_value(text: &str) -> Option<Option<usize>> {
    let mut t: String = text
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '_')
        .collect();
    while t.starts_with('(') && t.ends_with(')') {
        t = t[1..t.len() - 1].to_string();
    }
    let (size, rest) = t.split_once('\'')?;
    let size = match size {
        "" => None,
        n => Some(n.parse().ok()?),
    };
    let rest = rest.to_ascii_lowercase();
    let rest = rest.strip_prefix('s').unwrap_or(&rest);
    let digits = match rest.strip_prefix(['b', 'o', 'd', 'h']) {
        Some(d) if !d.is_empty() => d,
        _ => rest,
    };
    (!digits.is_empty() && digits.chars().all(|c| c == 'x')).then_some(size)
}

/// The declared type of a value parameter.
#[derive(Clone, Copy)]
pub(super) enum ParamTy {
    /// None (`parameter N = 8`): the type of the final value.
    OfValue,
    /// A data type or a range (`parameter int N`, `parameter [3:0] K`).
    Typed(Ty),
    /// A signing but no range (`parameter signed P`): signed or unsigned as
    /// declared, with the width of the final value, overrides included
    /// (IEEE 1800-2017 §6.20.2); the bits are kept.
    Signing(bool),
}

impl Cx<'_> {
    pub(super) fn module(&self, m: &sv::ModuleDeclarationAnsi) -> Result<Module> {
        let (header, timeunits, items, _, _) = &m.nodes;
        let (attrs, _, lifetime, name, imports, params, ports, _) = &header.nodes;
        let blackbox = self.module_attributes(attrs)?;
        self.absent(lifetime, "lifetime")?;
        self.absent(imports, "package import")?;
        self.absent(timeunits, "timeunits declaration")?;
        let mut module = Module {
            name: self.identifier(&name.nodes.0)?,
            params: Vec::new(),
            ports: Vec::new(),
            decls: Vec::new(),
            arrays: Vec::new(),
            items: Vec::new(),
            inits: Vec::new(),
            pins: Vec::new(),
            blackbox,
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
                let (dir, decl) = if blackbox {
                    self.blackbox_port(port, prev, &module.name)?
                } else {
                    self.port(port, prev)?
                };
                prev = Some((dir, decl.ty));
                match dir {
                    Some(dir) => module.ports.push(Port { dir, decl }),
                    None => module.pins.push(decl),
                }
            }
        }
        // with a parameter port list, a `parameter` in the body is local
        let body_overridable = params.is_none();
        for item in items {
            let before = (module.items.len(), module.decls.len());
            let loads = self.loads.borrow().len() + self.inits.borrow().len();
            self.module_item(item, &mut module, body_overridable)?;
            let after = (module.items.len(), module.decls.len());
            let loaded = self.loads.borrow().len() + self.inits.borrow().len();
            if blackbox && (before != after || loads != loaded || !self.arrays.borrow().is_empty())
            {
                return self.err(
                    item,
                    format!(
                        "the blackbox `{}` has a body: a blackbox declares only its ports and \
                         parameters",
                        module.name
                    ),
                );
            }
        }
        module.arrays = self.arrays.take();
        module.inits = self.inits.take();
        self.unknown_values(&module)?;
        resolve(&mut module, self.loads.take())?;
        module.constants();
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

    /// The attributes of a module: only `(* blackbox *)`, without a value.
    /// Whether it is there.
    pub(super) fn module_attributes(&self, attrs: &[sv::AttributeInstance]) -> Result<bool> {
        let mut blackbox = false;
        for a in attrs {
            for spec in a.nodes.1.contents() {
                let (name, value) = &spec.nodes;
                if self.text(name) != "blackbox" {
                    return self.err(
                        spec,
                        format!("attribute `{}` not supported", self.text(spec)),
                    );
                }
                if value.is_some() {
                    return self.err(spec, "`blackbox` takes no value");
                }
                blackbox = true;
            }
        }
        Ok(blackbox)
    }

    /// A port of the blackbox `module`: its width must not depend on a
    /// parameter, since the values given to the parameters of a blackbox
    /// are not used.
    pub(super) fn blackbox_port(
        &self,
        p: &sv::AnsiPortDeclaration,
        prev: Option<(Option<Dir>, Ty)>,
        module: &str,
    ) -> Result<(Option<Dir>, Decl)> {
        let params = self.params.take();
        let without = self.port(p, prev);
        self.params.replace(params);
        match without {
            Ok(port) => Ok(port),
            Err(e) => match self.port(p, prev) {
                Ok((_, decl)) => Err(Error::new(
                    e.span,
                    format!(
                        "the width of port `{}` of the blackbox `{module}` depends on a \
                         parameter: the widths of blackbox ports must be constants",
                        decl.name
                    ),
                )),
                Err(e) => Err(e),
            },
        }
    }

    /// `#(parameter N = 8, localparam M = N + 1, …)` or `#(N = 8, …)`.
    pub(super) fn parameter_port_list(
        &self,
        l: &sv::ParameterPortList,
        m: &mut Module,
    ) -> Result<()> {
        match l {
            sv::ParameterPortList::Assignment(a) => {
                let (first, rest) = &a.nodes.1.nodes.1;
                self.param_assignments(ParamTy::OfValue, first, true, m)?;
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

    pub(super) fn parameter_port_declaration(
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
                self.param_assignments(ParamTy::Typed(ty), list, true, m)
            }
            d => self.err(d, "type parameters not supported"),
        }
    }

    pub(super) fn parameter_declaration(
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

    /// The declared type of a parameter (see [`ParamTy`]).
    pub(super) fn param_type(&self, t: &sv::DataTypeOrImplicit) -> Result<ParamTy> {
        match t {
            sv::DataTypeOrImplicit::DataType(t) => self.param_data_type(t).map(ParamTy::Typed),
            sv::DataTypeOrImplicit::ImplicitDataType(t) => match &t.nodes {
                (None, dims) if dims.is_empty() => Ok(ParamTy::OfValue),
                (Some(signing), dims) if dims.is_empty() => {
                    Ok(ParamTy::Signing(matches!(signing, sv::Signing::Signed(_))))
                }
                (signing, dims) => self.vector(signing, dims).map(ParamTy::Typed),
            },
        }
    }

    /// `int`, or a vector type.
    pub(super) fn param_data_type(&self, t: &sv::DataType) -> Result<Ty> {
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

    /// `N = v, M = w, …` of a parameter declaration with type `ty`. An
    /// overridable parameter takes the value of the instance if it gives
    /// one.
    pub(super) fn param_assignments(
        &self,
        ty: ParamTy,
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
            let (value, at) = match (given, default) {
                (Some(o), _) if overridable => {
                    self.used.borrow_mut().insert(name.clone());
                    (o.value.clone(), o.span)
                }
                (_, Some((_, e))) => {
                    let e = match e {
                        sv::ConstantParamExpression::ConstantMintypmaxExpression(m) => {
                            match m.as_ref() {
                                sv::ConstantMintypmaxExpression::Unary(e) => e,
                                m => return self.err(m, "min:typ:max expression not supported"),
                            }
                        }
                        sv::ConstantParamExpression::Dollar(d) => {
                            return self.err(d.as_ref(), "`$` not supported as a parameter value");
                        }
                        e => return self.err(e, "type parameters not supported"),
                    };
                    (self.const_expression(e)?, self.span(e))
                }
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
            // a typed parameter takes its value as in an assignment; an
            // untyped one has the type of its value
            let failed = |f| match f {
                Failure::NotConstant(why) => Err(Error::new(
                    at,
                    format!("expected a constant expression, got `{value}`: {why}"),
                )),
                Failure::Error(e) => Err(e),
            };
            let value = match ty {
                ParamTy::Typed(ty) => constant::assigned(&value, ty).or_else(failed)?,
                ParamTy::OfValue => constant::value(&value).or_else(failed)?,
                ParamTy::Signing(signed) => {
                    let c = constant::value(&value).or_else(failed)?;
                    Const {
                        ty: Ty { signed, ..c.ty },
                        value: c.value,
                    }
                }
            };
            let ty = value.ty;
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

    /// A port of the ANSI header, with its direction (`None` for `inout`).
    /// Direction and type are inherited from the previous port (`prev`)
    /// where omitted (`input logic [7:0] a, b`).
    pub(super) fn port(
        &self,
        p: &sv::AnsiPortDeclaration,
        prev: Option<(Option<Dir>, Ty)>,
    ) -> Result<(Option<Dir>, Decl)> {
        let (dir, ty, id, default) = match p {
            sv::AnsiPortDeclaration::Net(n) => {
                let (header, id, dims, default) = &n.nodes;
                self.absent(dims, "unpacked dimension")?;
                match header {
                    None => (None, None, id, default),
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
                        (dir.as_ref(), Some(ty), id, default)
                    }
                    Some(h) => return self.unsupported(h),
                }
            }
            sv::AnsiPortDeclaration::Variable(v) => {
                let (header, id, dims, default) = &v.nodes;
                self.absent(dims, "unpacked dimension")?;
                match header {
                    None => (None, None, id, default),
                    Some(h) => {
                        let (dir, ty) = &h.nodes;
                        let ty = match &ty.nodes.0 {
                            sv::VarDataType::DataType(t) => self.data_type(t)?,
                            sv::VarDataType::Var(v) => self.data_type_or_implicit(&v.nodes.1)?,
                        };
                        (dir.as_ref(), Some(ty), id, default)
                    }
                }
            }
            p => return self.unsupported(p),
        };
        let dir = match dir {
            Some(sv::PortDirection::Input(_)) => Some(Dir::Input),
            Some(sv::PortDirection::Output(_)) => Some(Dir::Output),
            Some(sv::PortDirection::Inout(_)) => None,
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
        let name = self.identifier(&id.nodes.0)?;
        if let Some((_, value)) = default {
            match dir {
                Some(Dir::Input) => {
                    return self.err(value, "default values of inputs not supported");
                }
                None => return self.err(value, "default values of `inout` ports not supported"),
                Some(Dir::Output) => {}
            }
            // a constant expression in the grammar, so evaluated here
            let span = self.span(id);
            let text = self.text(value);
            match unknown_value(&text) {
                Some(size) => self.unknown.borrow_mut().push((name.clone(), size, span)),
                None => {
                    // an initial value, like one in a declaration
                    let e = self.const_expression(value)?;
                    if let Some(x) = constant::signal(&e) {
                        return self.err(
                            value,
                            format!("expected a constant expression, got `{text}`: `{x}` is not a parameter"),
                        );
                    }
                    let value = e;
                    self.inits.borrow_mut().push(Init {
                        target: name.clone(),
                        value,
                        span,
                    });
                }
            }
        }
        let decl = Decl {
            name,
            ty,
            span: self.span(id),
        };
        Ok((dir, decl))
    }

    pub(super) fn net_type(&self, net: &sv::NetType) -> Result<()> {
        match net {
            sv::NetType::Wire(_) => Ok(()),
            n => self.err(n, format!("net type `{}` not supported", self.text(n))),
        }
    }

    pub(super) fn data_type_or_implicit(&self, t: &sv::DataTypeOrImplicit) -> Result<Ty> {
        match t {
            sv::DataTypeOrImplicit::DataType(t) => self.data_type(t),
            sv::DataTypeOrImplicit::ImplicitDataType(t) => {
                let (signing, dims) = &t.nodes;
                self.vector(signing, dims)
            }
        }
    }

    pub(super) fn data_type(&self, t: &sv::DataType) -> Result<Ty> {
        match t {
            sv::DataType::Vector(v) => {
                let (kind, signing, dims) = &v.nodes;
                // `reg`, `logic` and `bit` are the same 2-state vector here
                // (`reg` and `logic` are one 4-state type in the standard)
                let _ = kind;
                self.vector(signing, dims)
            }
            t => self.err(t, format!("type `{}` not supported", self.text(t))),
        }
    }

    /// `[signed|unsigned] [hi:0]`, at most one packed dimension.
    pub(super) fn vector(
        &self,
        signing: &Option<sv::Signing>,
        dims: &[sv::PackedDimension],
    ) -> Result<Ty> {
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

    /// A non-negative integer given by a constant expression of the grammar.
    pub(super) fn const_int(&self, e: &sv::ConstantExpression) -> Result<usize> {
        self.int_of(e, &self.const_expression(e)?)
    }

    /// The constant expression `e`, written at `node`, as a non-negative
    /// integer (see [`constant::int`]).
    pub(super) fn int_of<'a>(
        &self,
        node: impl Into<RefNodes<'a>> + Copy,
        e: &Expr,
    ) -> Result<usize> {
        let text = self.text(node);
        match constant::int(e) {
            Ok(n) if n > LARGEST as i64 => self.err(
                node,
                format!(
                    "`{text}` is {n}: widths, sizes, counts and bounds up to {LARGEST} are \
                     supported (IEEE 1800 lets tools limit vectors to that many bits)"
                ),
            ),
            Ok(n) => usize::try_from(n)
                .or_else(|_| self.err(node, format!("`{text}` is negative ({n})"))),
            Err(Failure::NotConstant(why)) => self.err(
                node,
                format!("expected a constant integer, got `{text}`: {why}"),
            ),
            Err(Failure::Error(e)) => Err(e),
        }
    }

    /// An item of the module body; `overridable` says whether a
    /// `parameter` there can be set by instances.
    pub(super) fn module_item(
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
            sv::ModuleCommonItem::InitialConstruct(i) => self.initial(i),
            i => self.unsupported(i),
        }
    }

    /// `module #(.N(v), …) u1 (.a(x), …), u2 (…);`: parameter values and
    /// ports by name only.
    pub(super) fn instantiation(&self, i: &sv::ModuleInstantiation, m: &mut Module) -> Result<()> {
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
                        let value = match e {
                            sv::ParamExpression::MintypmaxExpression(m) => match m.as_ref() {
                                sv::MintypmaxExpression::Expression(x) => self.expr(x)?,
                                m => return self.err(m, "min:typ:max expression not supported"),
                            },
                            sv::ParamExpression::Dollar(d) => {
                                return self
                                    .err(d.as_ref(), "`$` not supported as a parameter value");
                            }
                            e => return self.err(e, "type parameters not supported"),
                        };
                        // evaluated where the parameter is declared, at its type
                        if let Some(x) = constant::signal(&value) {
                            return self.err(
                                e,
                                format!(
                                    "expected a constant expression, got `{}`: `{x}` is not a \
                                     parameter",
                                    self.text(e)
                                ),
                            );
                        }
                        params.push(Override {
                            name,
                            value,
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
                pins: Vec::new(),
                span: self.span(id),
            }));
        }
        Ok(())
    }

    pub(super) fn declaration(
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
        // (name, type, unpacked dimensions, value, whether a net)
        let mut add = |id: &sv::Identifier,
                       ty: Ty,
                       dims: Vec<&sv::UnpackedDimension>,
                       value: Option<&sv::Expression>,
                       net: bool|
         -> Result<()> {
            let name = self.identifier(id)?;
            let span = self.span(id);
            let Some(size) = self.array_size(id, &dims)? else {
                m.decls.push(Decl {
                    name: name.clone(),
                    ty,
                    span,
                });
                match value {
                    // `wire y = e;` is `assign y = e;`
                    Some(e) if net => {
                        let lhs = Lvalue {
                            kind: LvalueKind::Var(name),
                            span,
                        };
                        let rhs = self.expr(e)?;
                        m.items.push(Item::Assign { lhs, rhs, span });
                    }
                    Some(e) => self.initializer(name, span, e)?,
                    None => {}
                }
                return Ok(());
            };
            if let Some(e) = value {
                return self.err(
                    e,
                    "initial values of whole arrays not supported (load the array from a file \
                     with `$readmemb` or `$readmemh`)",
                );
            }
            for k in 0..size {
                m.decls.push(Decl {
                    name: ast::element(&name, k),
                    ty,
                    span,
                });
            }
            self.arrays.borrow_mut().push(Array {
                name,
                ty,
                size,
                span,
                rom: None,
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
                        let value = init.as_ref().map(|(_, e)| e);
                        add(&id.nodes.0, ty, dims.iter().collect(), value, true)?;
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
                                let mut unpacked = Vec::new();
                                for d in dims {
                                    match d {
                                        sv::VariableDimension::UnpackedDimension(d) => {
                                            unpacked.push(d.as_ref())
                                        }
                                        d => {
                                            return self.err(
                                                d,
                                                "dynamic, associative and queue arrays not \
                                                 supported",
                                            );
                                        }
                                    }
                                }
                                let value = init.as_ref().map(|(_, e)| e);
                                add(&id.nodes.0, ty, unpacked, value, false)?;
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
    pub(super) fn net_lvalue(&self, lv: &sv::NetLvalue) -> Result<Lvalue> {
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
            let x = self.const_expression(e)?;
            match constant::signal(&x) {
                Some(_) => self.err(e, VARIABLE_TARGET),
                None => self.int_of(e, &x).map(Index::Const),
            }
        };
        let bits = &bits.nodes.0;
        let range = |p: &sv::ConstantPartSelectRange| -> Result<Select> {
            Ok(match p {
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
            })
        };
        if self.is_array(&id) {
            let (i, sel) = match (bits.as_slice(), part) {
                ([b], None) => (index(&b.nodes.1)?, None),
                ([b, c], None) => (index(&b.nodes.1)?, Some(Select::Bit(index(&c.nodes.1)?))),
                ([b], Some(p)) => (index(&b.nodes.1)?, Some(range(&p.nodes.1)?)),
                _ => return self.err(select, self.whole_array(&id)),
            };
            let elem = Elem {
                array: id,
                index: i,
                sel,
                elems: Elems::Signals(Vec::new()),
            };
            let kind = LvalueKind::Elem(Box::new(elem));
            return Ok(Lvalue { kind, span });
        }
        let sel = match (bits.as_slice(), part) {
            ([], None) => None,
            ([b], None) => Some(Select::Bit(index(&b.nodes.1)?)),
            ([], Some(p)) => Some(range(&p.nodes.1)?),
            _ => return self.err(select, "multi-dimensional selects not supported"),
        };
        let kind = match sel {
            None => LvalueKind::Var(id),
            Some(sel) => LvalueKind::Select(id, sel),
        };
        Ok(Lvalue { kind, span })
    }

    /// `initial` with blocking assignments of initial values, `q = c;` or
    /// `m[3] = c;`, and `$readmemb("file", m);` or `$readmemh(…)` calls
    /// (one statement, or several in `begin … end`).
    pub(super) fn initial(&self, i: &sv::InitialConstruct) -> Result<()> {
        let only = "`initial` may only give signals initial values (`q = 8'd0;`) and load \
                    arrays from files (`$readmemb(\"file\", m);`)";
        let (_, s) = &i.nodes;
        let stmts: Vec<&sv::Statement> = match s {
            sv::StatementOrNull::Statement(s) => match &s.nodes.2 {
                sv::StatementItem::SeqBlock(b) => {
                    let (_, label, decls, stmts, _, end_label) = &b.nodes;
                    self.absent(label, "block name")?;
                    self.absent(end_label, "block name")?;
                    if let Some(d) = decls.first() {
                        return self.err(d, only);
                    }
                    let mut out = Vec::new();
                    for s in stmts {
                        match s {
                            sv::StatementOrNull::Statement(s) => out.push(s.as_ref()),
                            sv::StatementOrNull::Attribute(a) => self.absent(a, "attribute")?,
                        }
                    }
                    out
                }
                _ => vec![s],
            },
            sv::StatementOrNull::Attribute(a) => {
                self.absent(a, "attribute")?;
                Vec::new()
            }
        };
        for s in stmts {
            let (label, attrs, item) = &s.nodes;
            self.absent(label, "statement label")?;
            self.absent(attrs, "attribute")?;
            let c = match item {
                sv::StatementItem::BlockingAssignment(a) => {
                    self.initial_assignment(&a.0, only)?;
                    continue;
                }
                sv::StatementItem::NonblockingAssignment(_) => {
                    return self.err(
                        s,
                        "`initial` gives initial values with blocking assignments (`q = 8'd0;`), \
                         not `<=`",
                    );
                }
                sv::StatementItem::SubroutineCallStatement(c) => c,
                _ => return self.err(s, only),
            };
            let sv::SubroutineCallStatement::SubroutineCall(c) = c.as_ref() else {
                return self.err(c.as_ref(), only);
            };
            let sv::SubroutineCall::SystemTfCall(call) = &c.0 else {
                return self.err(&c.0, only);
            };
            let (name, args) = match call.as_ref() {
                sv::SystemTfCall::ArgOptionl(a) => {
                    let (name, args) = &a.nodes;
                    match args.as_ref().map(|a| &a.nodes.1) {
                        Some(sv::ListOfArguments::Ordered(a)) if a.nodes.1.is_empty() => {
                            (name, a.nodes.0.contents())
                        }
                        _ => return self.err(&c.0, only),
                    }
                }
                sv::SystemTfCall::ArgExpression(a) => {
                    let (name, args) = &a.nodes;
                    let (args, clocking) = &args.nodes.1;
                    if clocking.is_some() {
                        return self.err(&c.0, only);
                    }
                    (name, args.contents())
                }
                sv::SystemTfCall::ArgDataType(_) => return self.err(&c.0, only),
            };
            let task = self.text(name);
            let binary = match task.as_str() {
                "$readmemb" => true,
                "$readmemh" => false,
                _ => return self.err(&c.0, only),
            };
            let span = self.span(&c.0);
            let (file, m) = match args.as_slice() {
                [Some(f), Some(m)] => (f, m),
                [_, _, _, ..] => {
                    return self.err(
                        &c.0,
                        format!("`{task}` with start and end addresses not supported"),
                    );
                }
                _ => return self.err(&c.0, format!("`{task}` takes a file name and an array")),
            };
            let file_text = self.text(file);
            let file_name = match file {
                sv::Expression::Primary(p)
                    if matches!(
                        p.as_ref(),
                        sv::Primary::PrimaryLiteral(l)
                            if matches!(l.as_ref(), sv::PrimaryLiteral::StringLiteral(_))
                    ) && !file_text.contains('\\') =>
                {
                    file_text.trim_matches('"').to_string()
                }
                f => return self.err(f, format!("the file of `{task}` must be a string literal")),
            };
            let m_text = self.text(m);
            let arrays = self.arrays.borrow();
            let Some(array) = arrays.iter().find(|a| a.name == m_text) else {
                return self.err(m, format!("`{m_text}` is not an array declared before"));
            };
            if let Some(l) = self.loads.borrow().iter().find(|l| l.array == m_text) {
                return self.err(&c.0, format!("`{m_text}` is already loaded at {}", l.span));
            }
            let path = self.dir.join(&file_name);
            let text = std::fs::read_to_string(&path).map_err(|e| {
                Error::new(
                    self.span(file),
                    format!("cannot read `{}`: {e}", path.display()),
                )
            })?;
            let values = readmem(&text, binary, array)
                .map_err(|msg| Error::new(span, format!("{}: {msg}", path.display())))?;
            self.loads.borrow_mut().push(Load {
                array: m_text,
                binary,
                file: file_name,
                path,
                values,
                span,
            });
        }
        Ok(())
    }

    /// `q = c;` or `m[3] = c;` in `initial`.
    pub(super) fn initial_assignment(&self, a: &sv::BlockingAssignment, only: &str) -> Result<()> {
        let sv::BlockingAssignment::OperatorAssignment(a) = a else {
            return self.err(a, only);
        };
        let (lhs, op, rhs) = &a.nodes;
        if self.text(op) != "=" {
            return self.err(op, format!("`{}` not supported", self.text(op)));
        }
        let l = self.variable_lvalue(lhs)?;
        let target = match &l.kind {
            LvalueKind::Var(x) => x.clone(),
            LvalueKind::Elem(e) if e.sel.is_none() => match e.index {
                Index::Const(k) => ast::element(&e.array, k),
                Index::Expr(_) => {
                    return self.err(
                        lhs,
                        "the initial value of an element needs a constant address (`m[3] = \
                         8'd0;`)",
                    );
                }
            },
            _ => {
                return self.err(
                    lhs,
                    "initial values are given to whole signals and elements (`q = 8'd0;`, \
                     `m[3] = 8'd0;`), not to selects or concatenations",
                );
            }
        };
        self.initializer(target, l.span, rhs)
    }

    /// The initial value `e` of `target` at `span`.
    pub(super) fn initializer(&self, target: Name, span: Span, e: &sv::Expression) -> Result<()> {
        match unknown_value(&self.text(e)) {
            Some(size) => self.unknown.borrow_mut().push((target, size, span)),
            None => {
                let value = self.expr(e)?;
                self.inits.borrow_mut().push(Init {
                    target,
                    value,
                    span,
                });
            }
        }
        Ok(())
    }

    /// Checks the initial values that are x in every bit: an x literal
    /// narrower than its signal is extended with zeros, so it would say
    /// that some bits are 0; and no signal has two initial values.
    pub(super) fn unknown_values(&self, m: &Module) -> Result<()> {
        let signals = m.ports.iter().map(|p| &p.decl).chain(&m.decls);
        let widths: BTreeMap<&Name, usize> = signals.map(|d| (&d.name, d.ty.width)).collect();
        for (y, size, span) in self.unknown.borrow().iter() {
            if let (Some(n), Some(w)) = (size, widths.get(y))
                && n < w
            {
                return Err(Error::new(
                    *span,
                    format!(
                        "the initial value of `{y}` has {n} x bits, but `{y}` has {w} bits, so \
                         the others would be 0: values are 2-state, x is accepted only as the \
                         whole initial value (`'x`), meaning none"
                    ),
                ));
            }
        }
        let unknown = self.unknown.borrow();
        let mut all: Vec<(&Name, Span)> = (m.inits.iter().map(|i| (&i.target, i.span)))
            .chain(unknown.iter().map(|(y, _, span)| (y, *span)))
            .collect();
        all.sort_by_key(|(_, span)| *span);
        let mut seen: BTreeMap<&Name, Span> = BTreeMap::new();
        for (y, span) in all {
            if let Some(prev) = seen.insert(y, span) {
                return Err(Error::new(
                    span,
                    format!("`{y}` already has an initial value at {prev}"),
                ));
            }
        }
        Ok(())
    }
}
