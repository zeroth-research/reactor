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

The signals of a hierarchy have *paths*: `x` in the top module, `u.x` in
its instance `u`, `u.v.x` deeper down. Several paths may denote the same
signal: a port connected to a signal name is that signal. [`Inst::naming`]
gives every path its *name* in the translated module: the path of the
outermost signal it is connected to. So the ports of the top module keep
their names, and the other names are paths, which simulators accept as
hierarchical names (`dut.u.x`). An input connected to an expression
other than a name keeps its own path, `u.a`; the instantiating module
drives it.
*/

use crate::ast::{Dir, Error, ExprKind, Formal, Item, Module, Name, Override, Span};
use crate::frontend::{self, Design};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

type Result<T> = std::result::Result<T, Error>;

/// An elaborated module and its instances.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Inst {
    pub module: Module,
    /// The elaborated instances, by instance name, in source order.
    pub children: Vec<(Name, Inst)>,
}

/// Parses a file and elaborates its top module with the default parameter
/// values.
pub fn load_file(path: impl AsRef<Path>) -> Result<Inst> {
    let d = frontend::parse_design_file(path)?;
    elaborate(&d, &d.top()?, &[])
}

/// Parses `source` and elaborates its top module with the default
/// parameter values.
pub fn load(source: &str, path: impl AsRef<Path>) -> Result<Inst> {
    let d = frontend::parse_design(source, path)?;
    elaborate(&d, &d.top()?, &[])
}

/// Elaborates the module `top` of `d` with the parameter values
/// `overrides`, and all instances below it.
pub fn elaborate(d: &Design, top: &str, overrides: &[Override]) -> Result<Inst> {
    inst(d, top, overrides, "", &mut Vec::new())
}

/// Adds the path of the instance to an error from inside it.
pub fn context(e: Error, path: &str) -> Error {
    if path.is_empty() {
        e
    } else {
        let path = path.trim_end_matches('.');
        Error::new(e.span, format!("{} (in instance `{path}`)", e.msg))
    }
}

/// Elaborates the module `name` at `prefix` (`""` or a path ending in
/// `.`); `stack` holds the modules instantiating it.
fn inst(
    d: &Design,
    name: &str,
    overrides: &[Override],
    prefix: &str,
    stack: &mut Vec<Name>,
) -> Result<Inst> {
    let mut module = d.module(name, overrides).map_err(|e| context(e, prefix))?;
    stack.push(name.to_string());
    let mut children = Vec::new();
    for item in &mut module.items {
        let Item::Instance(i) = item else {
            continue;
        };
        let err = |span: Span, msg: String| Err(context(Error::new(span, msg), prefix));
        if !d.has(&i.module) {
            return err(i.span, format!("no module `{}`", i.module));
        }
        if stack.contains(&i.module) {
            return err(i.span, format!("recursive instantiation of `{}`", i.module));
        }
        let path = format!("{prefix}{}.", i.name);
        let child = inst(d, &i.module, &i.params, &path, stack)?;
        let ports = &child.module.ports;
        let clock = child.module.clock();
        let mut seen = BTreeSet::new();
        for c in &mut i.conns {
            let Some(p) = ports.iter().find(|p| p.decl.name == c.port) else {
                return err(c.span, format!("`{}` has no port `{}`", i.module, c.port));
            };
            if !seen.insert(c.port.clone()) {
                return err(c.span, format!("port `{}` connected twice", c.port));
            }
            c.formal = Some(Formal {
                dir: p.dir,
                ty: p.decl.ty,
                clock: clock == Some(&p.decl.name),
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
        children.push((i.name.clone(), child));
    }
    stack.pop();
    Ok(Inst { module, children })
}

impl Inst {
    /// A module without instances.
    pub fn leaf(module: Module) -> Self {
        Self {
            module,
            children: Vec::new(),
        }
    }

    /// The name in the translated module of every signal path, in order:
    /// the signals of a module (ports, internal signals) before those of
    /// its instances.
    pub fn naming(&self) -> Vec<(Name, Name)> {
        let mut out = Vec::new();
        let mut map = BTreeMap::new();
        self.walk("", &BTreeMap::new(), &mut out, &mut map);
        out
    }

    fn walk(
        &self,
        prefix: &str,
        given: &BTreeMap<Name, Name>,
        out: &mut Vec<(Name, Name)>,
        map: &mut BTreeMap<Name, Name>,
    ) {
        let m = &self.module;
        let decls = m.ports.iter().map(|p| &p.decl).chain(&m.decls);
        for d in decls {
            let path = format!("{prefix}{}", d.name);
            let name = given.get(&d.name).cloned().unwrap_or_else(|| path.clone());
            map.insert(path.clone(), name.clone());
            out.push((path, name));
        }
        for (u, child) in &self.children {
            let i = m.instances().find(|i| i.name == *u).expect("an instance");
            let mut connected = BTreeMap::new();
            for c in &i.conns {
                if let Some(ExprKind::Var(y)) = c.actual.as_ref().map(|e| &e.kind) {
                    let name = map[&format!("{prefix}{y}")].clone();
                    connected.insert(c.port.clone(), name);
                }
            }
            child.walk(&format!("{prefix}{u}."), &connected, out, map);
        }
    }

    /// The paths of the registers, in order (a module's before its
    /// instances').
    pub fn registers(&self) -> Vec<Name> {
        let mut out = Vec::new();
        self.walk_registers("", &mut out);
        out
    }

    fn walk_registers(&self, prefix: &str, out: &mut Vec<Name>) {
        out.extend(
            self.module
                .registers()
                .into_iter()
                .map(|q| format!("{prefix}{q}")),
        );
        for (u, child) in &self.children {
            child.walk_registers(&format!("{prefix}{u}."), out);
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
