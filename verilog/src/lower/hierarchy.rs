//! The translation of one instance and the composition of its children's modules.

use super::builder::Builder;
use super::order::{Pending, await_order, dependencies, reset_of};
use super::{Part, Result, init_name, internal, literal, title};
use crate::ast::{Dir, Edge, Error, Expr, Item, Lvalue, LvalueKind, Name, Span};
use crate::elab::{self, Inst};
use crate::{Atom, Module, Sort, SvOp, Var, Wire};
use crate::{sizing, typing};
use base::var::X;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// What the translation of a hierarchy collects.
pub(super) struct Out<'n> {
    /// The name of every signal path (see [`Inst::naming`]).
    pub(super) naming: &'n BTreeMap<Name, Name>,
    /// The variables by name.
    pub(super) vars: BTreeMap<Name, Var>,
    /// The paths of the registers.
    pub(super) registers: Vec<Name>,
    /// The paths of the registers with an initial value.
    pub(super) initialized: Vec<Name>,
    /// The widths of the registers, by path.
    pub(super) register_widths: BTreeMap<Name, usize>,
    /// The paths of the registers with an asynchronous reset.
    pub(super) reset_registers: Vec<Name>,
    pub(super) wire_names: HashMap<Wire, Name>,
    pub(super) instances: Vec<Part>,
    /// The reads that may select bits outside their signal, as warnings.
    pub(super) out_of_range: Vec<Error>,
    /// The instances of blackboxes, as warnings.
    pub(super) blackboxes: Vec<Error>,
}

impl Out<'_> {
    /// The variable named `name`, made on first use.
    fn var(&mut self, name: &str, width: usize) -> Var {
        *self
            .vars
            .entry(name.to_string())
            .or_insert_with(|| Var::new(Sort(width)))
    }
}

/// Translates the instance at path `prefix` (`""` or ending in `.`): its
/// own items become atoms, its instances modules, which are composed with
/// the signals of the module hidden.
pub(super) fn lower_rec(inst: &Inst, prefix: &str, out: &mut Out) -> Result<Module> {
    let ctx = |e: Error| elab::context(e, prefix);
    let mut env = typing::check(&inst.module).map_err(ctx)?;
    // every implicit resize made explicit, so that the terms below need
    // operands of equal types only
    let sized = sizing::module(&inst.module, &env).map_err(ctx)?;
    typing::explicit(&env, &sized).map_err(|e| {
        ctx(Error::new(
            e.span,
            format!("internal error: sizing left `{}`", e.msg),
        ))
    })?;
    let m = &sized;
    let registers: Vec<Name> = m.registers().into_iter().cloned().collect();
    // the initial values of registers (the others have `q@init`)
    let inits: BTreeMap<&Name, &Expr> = m.inits.iter().map(|i| (&i.target, &i.value)).collect();

    // an input connected to an expression other than a name is a signal
    // `u.a` of the module, driven by an `assign`
    let mut connections = Vec::new();
    for i in m.instances() {
        // nothing reads the inputs of a blackbox
        for (c, f) in i.conns(Dir::Input).filter(|(_, f)| !f.blackbox) {
            let Some(e) = c.actual.as_ref().filter(|_| c.shared().is_none()) else {
                continue;
            };
            let name = format!("{}.{}", i.name, c.port);
            let sig = typing::Sig {
                ty: f.ty,
                dir: None,
                span: c.span,
            };
            env.insert(name.clone(), sig);
            let lhs = Lvalue {
                kind: LvalueKind::Var(name),
                span: c.span,
            };
            connections.push(Item::Assign {
                lhs,
                rhs: e.clone(),
                span: c.span,
            });
        }
    }

    // in declaration order, so that variable ids (and printing) follow it
    let mut vars: BTreeMap<Name, Var> = BTreeMap::new();
    let decls = m.ports.iter().map(|p| &p.decl).chain(&m.decls);
    let locals = decls
        .map(|d| d.name.clone())
        .chain(connections.iter().flat_map(|c| c.targets()).cloned());
    for x in locals.collect::<Vec<_>>() {
        let name = out.naming[&format!("{prefix}{x}")].clone();
        vars.insert(x.clone(), out.var(&name, env[&x].ty.width));
    }
    for q in &registers {
        let path = format!("{prefix}{q}");
        if inits.contains_key(q) {
            out.initialized.push(path.clone());
        } else {
            vars.insert(init_name(q), out.var(&init_name(&path), env[q].ty.width));
        }
        out.register_widths.insert(path.clone(), env[q].ty.width);
        out.registers.push(path);
    }
    for (q, _) in m.items.iter().filter_map(reset_of).flat_map(|(_, v)| v) {
        out.reset_registers.push(format!("{prefix}{q}"));
    }

    // the await order is computed on the AST, so that a loop is reported
    // as such before any term is built
    let items = m
        .items
        .iter()
        .filter(|item| !matches!(item, Item::Instance(_)))
        .chain(&connections);
    // by name rather than path: ports connected to the same signal are one
    // variable (`.a(y), .y(y)` makes `assign y = a` a loop)
    let naming = out.naming;
    let canon = |x: &Name| {
        naming
            .get(&format!("{prefix}{x}"))
            .cloned()
            .unwrap_or_else(|| x.clone())
    };
    let pending: Vec<Pending> = items
        .map(|item| {
            let ctrl: Vec<Name> = item.targets().into_iter().cloned().collect();
            let awaits: BTreeSet<Name> = match item {
                // the init action awaits the initial values, the update
                // the next value of the clock (for the edge test); both
                // await the next value of an asynchronous reset
                Item::AlwaysFf { clock, .. } => ctrl
                    .iter()
                    .filter(|q| !inits.contains_key(q))
                    .map(|q| init_name(q))
                    .chain([clock.clone()])
                    .chain(reset_of(item).map(|(r, _)| r.name.clone()))
                    .collect(),
                // `assign y = y` awaits itself, a block reads its own
                // signals from its local values
                Item::Assign { .. } => item.reads().into_iter().map(|(x, _)| x.clone()).collect(),
                Item::AlwaysComb { .. } => item
                    .reads()
                    .into_iter()
                    .map(|(x, _)| x)
                    .filter(|x| !ctrl.contains(x))
                    .cloned()
                    .collect(),
                Item::Instance(_) => unreachable!("not an atom"),
            };
            let deps = match item {
                Item::AlwaysComb { body, .. } if ctrl.len() > 1 => {
                    let mut deps: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
                    for (y, d) in dependencies(body) {
                        deps.entry(canon(&y))
                            .or_default()
                            .extend(d.iter().map(&canon));
                    }
                    Some(deps)
                }
                _ => None,
            };
            Pending {
                item,
                reset: reset_of(item).map(|(r, _)| canon(&r.name)),
                ctrl: ctrl.iter().map(&canon).collect(),
                awaits: awaits.iter().map(&canon).collect(),
                deps,
                span: item.span(),
            }
        })
        .collect();

    let mut atoms = Vec::new();
    // the number of edge tests named so far, per name
    let mut edges: BTreeMap<Name, usize> = BTreeMap::new();
    for item in await_order(pending).map_err(ctx)? {
        let span = item.span();
        let atom = match item {
            Item::Assign { lhs, rhs, .. } => {
                let mut b = Builder::new(&env, &vars, false);
                b.start(&item.targets(), |_| None);
                let (w, _) = b.expr(rhs)?;
                b.assign(lhs, w, span)?;
                b.finish_all(span)?;
                out.out_of_range.extend(b.out_of_range.drain(..).map(ctx));
                out.add_names(prefix, b.names);
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::combinatorial(scope, b.terms).map_err(|e| internal(span, e))?
            }
            Item::AlwaysFf {
                clock, edge, body, ..
            } => {
                let regs = item.targets();
                let reset = reset_of(item);
                let mut b = Builder::new(&env, &vars, true);
                let e = b.edge(clock, *edge, span)?;
                let base = format!("{clock}@{}", if *edge == Edge::Pos { "pos" } else { "neg" });
                let k = edges.entry(base.clone()).or_insert(0);
                *k += 1;
                let name = if *k == 1 { base } else { format!("{base}#{k}") };
                b.names.insert(e, name);
                b.start(&regs, |q| Some(*vars[q]));
                b.stmt(body)?;
                let active = reset
                    .as_ref()
                    .map(|(r, values)| (b.reset_level(r), values.as_slice()));
                b.finish_guarded(e, active, span)?;
                out.add_names(prefix, std::mem::take(&mut b.names));
                // the init action: the initial value (given, or any), or
                // the reset value if the reset is active in the first round
                let mut init = Builder::new(&env, &vars, true);
                let active = reset.as_ref().map(|(r, _)| init.reset_level(r));
                for q in &regs {
                    let var = vars[*q];
                    let value = reset.iter().flat_map(|(_, v)| v).find(|(r, _)| r == q);
                    let start = match (inits.get(q), active.zip(value)) {
                        // a constant initial value directly: `X(q) := 8'd5`
                        (Some(v), None) if let Some(c) = literal(v) => {
                            init.emit_to(SvOp::Lit(c), &[], X(var), span)?;
                            continue;
                        }
                        (Some(v), _) => init.expr(v)?.0,
                        (None, _) => {
                            init.used.insert(init_name(q));
                            X(vars[&init_name(q)])
                        }
                    };
                    match active.zip(value) {
                        Some((a, (_, c))) => {
                            let (c, _) = init.expr(c)?;
                            init.emit_to(SvOp::Ite, &[a, c, start], X(var), span)?;
                        }
                        None => {
                            init.emit_to(SvOp::Id, &[start], X(var), span)?;
                        }
                    }
                }
                b.used.extend(init.used);
                out.out_of_range.extend(b.out_of_range.drain(..).map(ctx));
                out.out_of_range
                    .extend(init.out_of_range.drain(..).map(ctx));
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::sequential(scope, init.terms, b.terms).map_err(|e| internal(span, e))?
            }
            Item::AlwaysComb { body, .. } => {
                let mut b = Builder::new(&env, &vars, false);
                b.start(&item.targets(), |_| None);
                b.stmt(body)?;
                b.finish_all(span)?;
                out.out_of_range.extend(b.out_of_range.drain(..).map(ctx));
                out.add_names(prefix, b.names);
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::combinatorial(scope, b.terms).map_err(|e| internal(span, e))?
            }
            Item::Instance(_) => unreachable!("not an atom"),
        };
        atoms.push(atom);
    }

    if inst.children.is_empty() {
        let private: BTreeSet<Var> = m.decls.iter().map(|d| vars[&d.name]).collect();
        return Module::partially_observable(atoms, |v| private.contains(v))
            .map_err(|e| internal(Span::default(), e));
    }

    // the module's own atoms observe everything, since its instances may
    // read and write its internal signals; hiding happens in the
    // composition
    let own =
        Module::partially_observable(atoms, |_| false).map_err(|e| internal(Span::default(), e))?;
    let mut modules = vec![(own, Span::default())];
    // the outputs of blackboxes are external, like inputs, so not hidden
    let free: BTreeSet<&Name> = m
        .instances()
        .flat_map(|i| i.conns(Dir::Output))
        .filter(|(_, f)| f.blackbox)
        .filter_map(|(c, _)| c.shared())
        .collect();
    let mut hidden: BTreeSet<Var> = (m.decls.iter())
        .filter(|d| !free.contains(&d.name))
        .map(|d| vars[&d.name])
        .collect();
    hidden.extend(
        connections
            .iter()
            .flat_map(|c| c.targets())
            .map(|x| vars[x]),
    );
    for (u, child) in &inst.children {
        let i = m.instances().find(|i| i.name == *u).expect("an instance");
        if child.module.blackbox {
            let free: Vec<String> = (i.conns(Dir::Output))
                .filter_map(|(c, _)| c.shared())
                .map(|y| format!("`{}`", out.naming[&format!("{prefix}{y}")]))
                .collect();
            let mut what = match free.as_slice() {
                [] => Vec::new(),
                [y] => vec![format!("{y} takes any value in every time step")],
                ys => vec![format!(
                    "{} take any value in every time step",
                    ys.join(", ")
                )],
            };
            if i.conns(Dir::Input).next().is_some() {
                what.push("nothing reads its inputs".to_string());
            }
            let what = match what.as_slice() {
                [] => String::new(),
                _ => format!(": {}", what.join(", and ")),
            };
            let msg = format!(
                "`{u}` is an instance of the blackbox `{}`, whose behaviour is not modelled{what}",
                child.module.name
            );
            out.blackboxes.push(ctx(Error::new(i.span, msg)));
            continue;
        }
        let path = format!("{prefix}{u}.");
        let module = lower_rec(child, &path, out)?;
        out.instances.push(Part {
            path: format!("{prefix}{u}"),
            title: title(&child.module),
            module: module.clone(),
        });
        modules.push((module, i.span));
        // unconnected outputs are signals of the instance only
        for p in child.module.ports.iter().filter(|p| p.dir == Dir::Output) {
            let connected = i
                .conns
                .iter()
                .any(|c| c.port == p.decl.name && c.actual.is_some());
            if !connected {
                hidden.insert(out.vars[&format!("{path}{}", p.decl.name)]);
            }
        }
    }
    if let Some((span, msg)) = instance_loop(&modules, &out.vars) {
        return Err(ctx(Error::new(span, msg)));
    }
    Module::compose_hiding(modules.into_iter().map(|(m, _)| m), |v| hidden.contains(v))
        .map_err(|e| internal(Span::default(), e))
}

impl Out<'_> {
    /// Adds the names of intermediate values of an atom of the instance at
    /// `prefix`.
    fn add_names(&mut self, prefix: &str, names: HashMap<Wire, Name>) {
        self.wire_names
            .extend(names.into_iter().map(|(w, n)| (w, format!("{prefix}{n}"))));
    }
}

/// A combinational loop through the atoms of several modules (each of them
/// free of loops), with the position of the instance of the first module
/// on it that is not the first of `modules`.
pub(super) fn instance_loop(
    modules: &[(Module, Span)],
    vars: &BTreeMap<Name, Var>,
) -> Option<(Span, String)> {
    let atoms: Vec<(&Atom, Span)> = modules
        .iter()
        .flat_map(|(m, span)| m.atoms().iter().map(move |a| (a, *span)))
        .collect();
    let name = |v: &Var| {
        vars.iter()
            .find(|(_, w)| *w == v)
            .map_or("?", |(n, _)| n.as_str())
    };
    // the variable by which atom `a` awaits atom `b`
    let via = |a: usize, b: usize| {
        atoms[a]
            .0
            .wait()
            .iter()
            .find(|v| atoms[b].0.ctrl().contains(v))
            .copied()
    };
    // depth-first search for a cycle
    let n = atoms.len();
    let mut state = vec![0u8; n]; // 0 new, 1 on the stack, 2 done
    let mut stack: Vec<usize> = Vec::new();
    fn dfs(
        a: usize,
        n: usize,
        via: &dyn Fn(usize, usize) -> Option<Var>,
        state: &mut [u8],
        stack: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        state[a] = 1;
        stack.push(a);
        for b in 0..n {
            if via(a, b).is_none() {
                continue;
            }
            if state[b] == 1 {
                let start = stack.iter().position(|&x| x == b).unwrap();
                return Some(stack[start..].to_vec());
            }
            if state[b] == 0
                && let Some(c) = dfs(b, n, via, state, stack)
            {
                return Some(c);
            }
        }
        stack.pop();
        state[a] = 2;
        None
    }
    for a in 0..n {
        if state[a] == 0
            && let Some(cycle) = dfs(a, n, &via, &mut state, &mut stack)
        {
            // each atom awaits the next: the data flows backwards
            let mut signals: Vec<&str> = (0..cycle.len())
                .map(|k| name(&via(cycle[k], cycle[(k + 1) % cycle.len()]).unwrap()))
                .collect();
            signals.reverse();
            signals.push(signals[0]);
            let span = cycle
                .iter()
                .map(|&a| atoms[a].1)
                .find(|s| *s != Span::default())
                .unwrap_or_default();
            return Some((
                span,
                format!(
                    "combinational loop through instances: {}",
                    signals.join(" -> ")
                ),
            ));
        }
    }
    None
}
