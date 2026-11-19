/*!
# Elaborating a module hierarchy

[`elaborate`] turns a [`Design`] into a tree of instances: the top module,
elaborated with its parameter values, and for each of its instances the
instantiated module, elaborated with the values the instance gives, and so
on. Every instance is a separate copy, so two instances of a module never
share signals. Recursive instantiation is rejected.

Elaboration also fills in the [`Formal`] of every connection, so that the
instantiating module can be checked on its own: a connection must name a
port, at most once, and every input must be
connected. Outputs may be left unconnected (`.y()` or not mentioned).

A blackbox (a module marked `(* blackbox *)`, see
[`Module::blackbox`](crate::ast::Module::blackbox)) is elaborated like any
module, but the values given to its parameters have no effect (its port
widths are constants). Its `inout` ports go to
[`Instance::pins`](crate::ast::Instance::pins), each connected to a pin of
the instantiating module or to nothing; only the top module and
blackboxes have `inout` ports.

The signals of a hierarchy have *paths*: `x` in the top module, `u.x` in
its instance `u`, `u.v.x` deeper down. Several paths may denote the same
signal: a port connected to the name of a signal of its width is that
signal. [`Inst::naming`] gives every path its *name* in the translated
module: the path of the
outermost signal it is connected to. So the ports of the top module keep
their names, and the other names are paths, which simulators accept as
hierarchical names (`dut.u.x`). An input connected to any other
expression, or to a name of another width, keeps its own path, `u.a`; the
instantiating module drives it. The ports of a blackbox are no signals and
have no paths: the signals connected to its outputs are free values.
*/

use crate::ast::{
    Dir, Error, ErrorKind, ExprKind, Formal, Instance, Item, Message, Module, Name, Override, Span,
};
use crate::frontend::{self, Design};
use std::collections::{BTreeMap, BTreeSet};

type Result<T> = std::result::Result<T, Error>;

/// An elaborated module and its instances.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Inst {
    pub module: Module,
    /// The elaborated instances, by instance name, in source order.
    pub children: Vec<(Name, Inst)>,
}

/// The path of an instance in the hierarchy: empty for the top module, `u`
/// for its instance `u`, `u.v` for the instance `v` of `u`. The signal `x`
/// of the instance has the path `u.v.x` ([`Path::signal`]).
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Path {
    /// `""` or the names of the instances, each followed by `.`.
    prefix: String,
}

impl Path {
    /// The path of the top module.
    pub fn top() -> Self {
        Self::default()
    }

    pub fn is_top(&self) -> bool {
        self.prefix.is_empty()
    }

    /// The path of the instance `u` of this instance.
    pub fn child(&self, u: &str) -> Self {
        Self {
            prefix: format!("{}{u}.", self.prefix),
        }
    }

    /// The path of the signal (or port) `x` of this instance: `u.v.x`.
    pub fn signal(&self, x: &str) -> Name {
        format!("{}{x}", self.prefix)
    }
}

/// `u.v`, empty for the top module.
impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.prefix.trim_end_matches('.'))
    }
}

/// Parses a file and elaborates its top module with the default parameter
/// values.
pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Inst> {
    let d = frontend::parse_design_file(path)?;
    elaborate(&d, &d.top()?, &[])
}

/// Parses `source` and elaborates its top module with the default
/// parameter values.
pub fn load(source: &str, path: impl AsRef<std::path::Path>) -> Result<Inst> {
    let d = frontend::parse_design(source, path)?;
    elaborate(&d, &d.top()?, &[])
}

/// Elaborates the module `top` of `d` with the parameter values
/// `overrides`, and all instances below it.
pub fn elaborate(d: &Design, top: &str, overrides: &[Override]) -> Result<Inst> {
    if d.is_blackbox(top) {
        return Err(Error::unlocated(
            ErrorKind::User,
            format!("`{top}` is a blackbox: it cannot be the top module"),
        ));
    }
    let top = inst(d, top, overrides, &Path::top(), &mut Vec::new())?;
    debug_assert!(connected(&top), "a connection without its port");
    Ok(top)
}

/// Whether every connection of every instance in the hierarchy has its
/// [`Formal`] (what elaboration fills in).
fn connected(inst: &Inst) -> bool {
    let module = inst.module.instances();
    module.flat_map(|i| &i.conns).all(|c| c.formal.is_some())
        && inst.children.iter().all(|(_, c)| connected(c))
}

/// Adds the path of the instance to an error from inside it.
pub(crate) fn context<M: Message>(e: M, path: &Path) -> M {
    if path.is_top() {
        e
    } else {
        Message::map_msg(e, |msg| format!("{msg} (in instance `{path}`)"))
    }
}

/// Moves the connections of the `inout` ports of `child` (pins of a
/// blackbox) from `i`'s connections to [`Instance::pins`], checking that
/// each is connected to one of the `pins` of the module `module_name`;
/// adds the ports to `seen`.
fn connect_pins(
    i: &mut Instance,
    child: &Inst,
    pins: &BTreeMap<Name, usize>,
    module_name: &str,
    seen: &mut BTreeSet<Name>,
) -> Result<()> {
    let err = |span: Span, msg: String| Err(Error::new(span, msg));
    // `inout` ports are accepted only as pins of blackboxes
    let unsupported_err = |span: Span, msg: String| Err(Error::unsupported(span, msg));
    let child_pins = &child.module.pins;
    if let (false, Some(p)) = (child.module.blackbox, child_pins.first()) {
        return unsupported_err(
            i.span,
            format!(
                "`{}` has an `inout` port `{}`: only blackboxes and the top module may \
                 have `inout` ports",
                i.module, p.name
            ),
        );
    }
    let (pin_conns, conns): (Vec<_>, Vec<_>) = std::mem::take(&mut i.conns)
        .into_iter()
        .partition(|c| child_pins.iter().any(|p| p.name == c.port));
    i.conns = conns;
    for c in &pin_conns {
        if !seen.insert(c.port.clone()) {
            return err(c.span, format!("port `{}` connected twice", c.port));
        }
        let Some(e) = &c.actual else { continue };
        let formal = child_pins.iter().find(|p| p.name == c.port).expect("a pin");
        match &e.kind {
            ExprKind::Var(x) if pins.contains_key(x) => {
                if pins[x] != formal.ty.width {
                    return err(
                        e.span,
                        format!(
                            "`inout` port `{}` of `{}` has {}, `{x}` {}",
                            c.port,
                            i.name,
                            bits(formal.ty.width),
                            pins[x]
                        ),
                    );
                }
            }
            _ => {
                return unsupported_err(
                    e.span,
                    format!(
                        "`inout` port `{}` of `{}` must be connected to an `inout` port of \
                         `{module_name}` (or left unconnected)",
                        c.port, i.name
                    ),
                );
            }
        }
    }
    i.pins = pin_conns;
    Ok(())
}

/// Connects the instance `i` of `child`, in a module with the signals
/// `widths` and the pins `pins` (`inout` ports) named `module_name`: fills
/// in the formal of each connection and moves the connections of pins to
/// [`Instance::pins`].
fn connect(
    i: &mut Instance,
    child: &Inst,
    widths: &BTreeMap<Name, usize>,
    pins: &BTreeMap<Name, usize>,
    module_name: &str,
) -> Result<()> {
    let err = |span: Span, msg: String| Err(Error::new(span, msg));
    let ports = &child.module.ports;
    let clocks = child.module.clocks();
    let mut seen = BTreeSet::new();
    connect_pins(i, child, pins, module_name, &mut seen)?;
    for c in &mut i.conns {
        let width = |x: &Name| widths.get(x).copied();
        let Some(p) = ports.iter().find(|p| p.decl.name == c.port) else {
            return err(c.span, format!("`{}` has no port `{}`", i.module, c.port));
        };
        if !seen.insert(c.port.clone()) {
            return err(c.span, format!("port `{}` connected twice", c.port));
        }
        let shared = match c.actual.as_ref().map(|e| &e.kind) {
            Some(ExprKind::Var(x)) => width(x) == Some(p.decl.ty.width),
            _ => false,
        };
        c.formal = Some(Formal {
            dir: p.dir,
            ty: p.decl.ty,
            clock: clocks.contains(&p.decl.name),
            shared,
            blackbox: child.module.blackbox,
        });
    }
    for p in ports.iter().filter(|p| p.dir == Dir::Input) {
        let connected = i
            .conns
            .iter()
            .any(|c| c.port == p.decl.name && c.actual.is_some());
        if !connected {
            return err(
                i.span,
                format!("input `{}` of `{}` is not connected", p.decl.name, i.name),
            );
        }
    }
    Ok(())
}

/// Elaborates the module `name` at `prefix`; `stack` holds the modules
/// instantiating it.
fn inst(
    d: &Design,
    name: &str,
    overrides: &[Override],
    prefix: &Path,
    stack: &mut Vec<Name>,
) -> Result<Inst> {
    let mut module = d.module(name, overrides).map_err(|e| context(e, prefix))?;
    stack.push(name.to_string());
    let widths: BTreeMap<Name, usize> = (module.ports.iter().map(|p| &p.decl))
        .chain(&module.decls)
        .map(|d| (d.name.clone(), d.ty.width))
        .collect();
    let pins: BTreeMap<Name, usize> = (module.pins.iter())
        .map(|d| (d.name.clone(), d.ty.width))
        .collect();
    let module_name = module.name.clone();
    let mut children = Vec::new();
    for item in &mut module.items {
        let Item::Instance(i) = item else {
            continue;
        };
        let err = |span: Span, msg: String| Err(context(Error::new(span, msg), prefix));
        if !d.has(&i.module) {
            return err(
                i.span,
                format!(
                    "no module `{}` (for a vendor cell or another module defined outside the \
                     design, declare a stub with its ports, marked `(* blackbox *)`: its \
                     outputs are then free in every time step)",
                    i.module
                ),
            );
        }
        if stack.contains(&i.module) {
            let msg = format!("recursive instantiation of `{}`", i.module);
            return Err(context(Error::unsupported(i.span, msg), prefix));
        }
        let child = inst(d, &i.module, &i.params, &prefix.child(&i.name), stack)?;
        connect(i, &child, &widths, &pins, &module_name).map_err(|e| context(e, prefix))?;
        children.push((i.name.clone(), child));
    }
    // with the ports known, the signals connected to instances too
    frontend::constants(&mut module);
    stack.pop();
    Ok(Inst { module, children })
}

/// `n` bits, as in messages: `1 bit`, `2 bits`.
fn bits(n: usize) -> String {
    if n == 1 {
        "1 bit".to_string()
    } else {
        format!("{n} bits")
    }
}

impl Inst {
    /// A module without instances.
    pub fn leaf(module: Module) -> Self {
        Self {
            module,
            children: Vec::new(),
        }
    }

    /// Every instance of the hierarchy with its path, the top module first,
    /// each module before its instances, the instances in source order.
    pub fn iter(&self) -> impl Iterator<Item = (Path, &Inst)> {
        let mut stack = vec![(Path::top(), self)];
        std::iter::from_fn(move || {
            let (path, inst) = stack.pop()?;
            for (u, child) in inst.children.iter().rev() {
                stack.push((path.child(u), child));
            }
            Some((path, inst))
        })
    }

    /// The name in the translated module of every signal path, in order:
    /// the signals of a module (ports, internal signals) before those of
    /// its instances.
    pub fn naming(&self) -> Vec<(Name, Name)> {
        let mut out = Vec::new();
        let mut map = BTreeMap::new();
        // the names given to the ports of each instance by its parent
        let mut given: BTreeMap<Path, BTreeMap<Name, Name>> = BTreeMap::new();
        // the ports of a blackbox are no signals of the design: its
        // outputs are the signals they are connected to (and a blackbox
        // has no instances)
        for (path, inst) in self.iter().filter(|(_, i)| !i.module.blackbox) {
            let ports = given.remove(&path).unwrap_or_default();
            let m = &inst.module;
            for d in m.ports.iter().map(|p| &p.decl).chain(&m.decls) {
                let signal = path.signal(&d.name);
                let name = ports
                    .get(&d.name)
                    .cloned()
                    .unwrap_or_else(|| signal.clone());
                map.insert(signal.clone(), name.clone());
                out.push((signal, name));
            }
            for i in m.instances() {
                let mut connected = BTreeMap::new();
                for c in &i.conns {
                    if let Some(y) = c.shared() {
                        connected.insert(c.port.clone(), map[&path.signal(y)].clone());
                    }
                }
                given.insert(path.child(&i.name), connected);
            }
        }
        out
    }

    /// The signals connected to outputs of blackboxes, by name (see
    /// [`Inst::naming`]), in order (a module's before its instances').
    pub fn blackbox_outputs(&self) -> Vec<Name> {
        let naming: BTreeMap<Name, Name> = self.naming().into_iter().collect();
        let mut out = Vec::new();
        for (path, i) in self.iter() {
            for u in i.module.instances() {
                for (c, _) in u.conns(Dir::Output).filter(|(_, f)| f.blackbox) {
                    if let Some(y) = c.shared() {
                        out.push(naming[&path.signal(y)].clone());
                    }
                }
            }
        }
        out
    }

    /// The paths of the registers, in order (a module's before its
    /// instances').
    pub fn registers(&self) -> Vec<Name> {
        let registers = |(path, i): (Path, &Inst)| {
            let qs = i.module.registers().into_iter();
            qs.map(|q| path.signal(q)).collect::<Vec<_>>()
        };
        self.iter().flat_map(registers).collect()
    }

    /// The paths of the registers with an initial value, in order (a
    /// module's before its instances').
    pub fn initialized(&self) -> Vec<Name> {
        let initialized = |(path, i): (Path, &Inst)| {
            let m = &i.module;
            let inits: BTreeSet<&Name> = m.inits.iter().map(|i| &i.target).collect();
            let qs = m.registers().into_iter().filter(|q| inits.contains(q));
            qs.map(|q| path.signal(q)).collect::<Vec<_>>()
        };
        self.iter().flat_map(initialized).collect()
    }

    /// Drops the initial values of all registers, so that they start with
    /// any value (constants and ROMs keep their values): for hardware that
    /// does not keep initial values, such as ASICs.
    pub fn ignore_initializers(&mut self) {
        let m = &mut self.module;
        let registers: BTreeSet<Name> = m.registers().into_iter().cloned().collect();
        m.inits.retain(|i| !registers.contains(&i.target));
        for (_, child) in &mut self.children {
            child.ignore_initializers();
        }
    }

    /// The module at the path `prefix` (`""` or ending in `.`).
    pub fn at(&self, prefix: &str) -> &Inst {
        match prefix.split_once('.') {
            None => self,
            Some((u, rest)) => {
                let child = &self
                    .children
                    .iter()
                    .find(|(n, _)| n == u)
                    .expect("a path")
                    .1;
                child.at(rest)
            }
        }
    }
}
