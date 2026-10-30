/*!
# From the AST to a reactive module

Every signal becomes a variable of the width of its type: inputs are
external, outputs are interface variables, internal signals are private.

Every `assign` becomes a combinatorial atom that awaits the next values
`X(·)` of the signals it reads and writes `X(lhs)`. The expression becomes
one term per operator, with the source operators rewritten into [`SvOp`],
which has no `>`, `>=` and no logical operators:

| source | ops |
|---|---|
| `a > b`, `a >= b` | `ULt`/`SLt`, `ULe`/`SLe` with swapped operands |
| `<`, `<=` | `ULt`/`ULe`, or `SLt`/`SLe` for signed operands |
| `!a`, `a && b`, `a \|\| b` | `Not`/`And`/`Or` on `RedOr` of the operands (none for 1 bit) |
| `c ? a : b` | `Ite` on `RedOr c` (none for 1 bit) |
| `~^`, `~&`, `~\|` | `Not` of `Xor`, `RedAnd`, `RedOr` |
| `===`, `!==` | `Eq`, `Ne` |
| unary `+` | nothing |

Every `always_ff` block becomes a sequential atom controlling the
registers it assigns. One round of the module is one clock cycle, and
the clock itself is no variable: its rising edge is the step from one
round to the next. So the update reads the *latched* values `*x` (the
values of the previous cycle, just before the edge) and writes the next
values `X(q)` (the values after the edge). It awaits nothing, so a
register never closes a combinational loop. The statements are executed
symbolically with a map from each register to the wire holding its value
after the edge so far, starting with `*q` (a register keeps its value on
paths that do not assign it). `q <= e` sets the entry of `q` to the wire of
`e`, so the last assignment wins; at `if`, both branches start from the
same map and the entries that differ are merged with `Ite`. The init
action sets every register to an external variable `q@init`, so all
initial states are possible.

The atoms are sorted so that each comes after the atoms controlling what it
awaits; a cycle is a combinational loop and is rejected.
*/

use crate::ast::{
    self, BinOp, Dir, Error, Expr, ExprKind, Item, Name, Span, Stmt, StmtKind, Ty, UnOp,
};
use crate::typing;
use crate::{Atom, Module, Sort, SvOp, Term, Var, Wire};
use base::var::X;
use std::collections::{BTreeMap, BTreeSet};

type Result<T> = std::result::Result<T, Error>;

/// A module with the names of its variables.
pub struct Translation {
    pub name: Name,
    pub module: Module,
    /// Every source signal, and `q@init` for every register `q`, by name.
    pub vars: BTreeMap<Name, Var>,
    /// The clock of the `always_ff` blocks; it is not a variable of the
    /// module, since a round is a whole clock cycle.
    pub clock: Option<Name>,
    /// The signals assigned in `always_ff` blocks, in source order.
    pub registers: Vec<Name>,
    /// Inputs other than the clock that nothing reads; they are not
    /// variables of the module.
    pub unread_inputs: Vec<Name>,
}

/// The name of the external variable holding the initial value of the
/// register `q`. Generated names contain `@`, which SystemVerilog
/// identifiers cannot, so they never collide with source names.
pub fn init_name(q: &str) -> Name {
    format!("{q}@init")
}

impl Translation {
    /// The name of a variable of the module.
    pub fn name(&self, var: &Var) -> &str {
        self.vars
            .iter()
            .find(|(_, v)| *v == var)
            .map_or("?", |(n, _)| n.as_str())
    }

    /// The variable of a signal; panics if there is none.
    pub fn var(&self, name: &str) -> Var {
        self.vars[name]
    }

    /// The module printed with the signal names.
    pub fn print(&self) -> String {
        crate::print::module(&self.module, &self.name, |v| self.name(v).to_string())
    }
}

/// Checks and translates a module.
pub fn lower(m: &ast::Module) -> Result<Translation> {
    let env = typing::check(m)?;
    let clock = m.clock().cloned();
    let registers: Vec<Name> = m.registers().into_iter().cloned().collect();
    // in declaration order, so that variable ids (and printing) follow it
    let decls = m.ports.iter().map(|p| &p.decl).chain(&m.decls);
    let mut vars: BTreeMap<Name, Var> = decls
        .map(|d| (d.name.clone(), Var::new(Sort(d.ty.width))))
        .collect();
    for q in &registers {
        vars.insert(init_name(q), Var::new(Sort(env[q].ty.width)));
    }

    // the await order is computed on the AST, so that a loop is reported
    // as such before any term is built
    let pending: Vec<Pending> = m
        .items
        .iter()
        .map(|item| {
            let ctrl: Vec<Name> = item.targets().into_iter().cloned().collect();
            let (awaits, span) = match item {
                Item::Assign { rhs, span, .. } => {
                    let mut reads = Vec::new();
                    rhs.reads(&mut reads);
                    (reads.into_iter().map(|(x, _)| x.clone()).collect(), *span)
                }
                // only the init action awaits, and only external variables
                Item::AlwaysFf { span, .. } => (ctrl.iter().map(|q| init_name(q)).collect(), *span),
            };
            Pending {
                item,
                ctrl,
                awaits,
                span,
            }
        })
        .collect();
    let read: BTreeSet<&Name> = m
        .items
        .iter()
        .flat_map(|item| item.reads().into_iter().map(|(x, _)| x))
        .collect();
    let unread_inputs = m
        .ports
        .iter()
        .map(|p| &p.decl.name)
        .filter(|x| env[*x].dir == Some(Dir::Input) && !read.contains(x))
        .filter(|x| Some(*x) != clock.as_ref())
        .cloned()
        .collect();

    let mut atoms = Vec::new();
    for item in await_order(pending)? {
        let atom = match item {
            Item::Assign { lhs, rhs, span } => {
                let mut b = Builder::new(&env, &vars, false);
                b.used.insert(lhs.clone());
                b.expr(rhs, Some(X(vars[lhs])))?;
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::combinatorial(scope, b.terms).map_err(|e| internal(*span, e))?
            }
            Item::AlwaysFf { body, span, .. } => {
                let regs = item.targets();
                let mut b = Builder::new(&env, &vars, true);
                b.used.extend(regs.iter().map(|q| (*q).clone()));
                let mut next: Vec<(&Name, Wire)> = regs.iter().map(|q| (*q, *vars[*q])).collect();
                b.stmt(body, &mut next)?;
                for (q, w) in next {
                    b.finish(w, X(vars[q]), *span)?;
                }
                let mut init = Vec::new();
                for q in &regs {
                    let q0 = vars[&init_name(q)];
                    let term = Term::function(SvOp::Id, [X(vars[*q])], [X(q0)]);
                    init.push(term.map_err(|e| internal(*span, e))?);
                    b.used.insert(init_name(q));
                }
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::sequential(scope, init, b.terms).map_err(|e| internal(*span, e))?
            }
        };
        atoms.push(atom);
    }

    let private: BTreeSet<Var> = m.decls.iter().map(|d| vars[&d.name]).collect();
    let module = Module::partially_observable(atoms, |v| private.contains(v))
        .map_err(|e| internal(Span::default(), e))?;
    Ok(Translation {
        name: m.name.clone(),
        module,
        vars,
        clock,
        registers,
        unread_inputs,
    })
}

fn internal(span: Span, e: String) -> Error {
    Error::new(span, format!("internal error: {e}"))
}

/// An item before sorting, with the signals it controls and those it
/// awaits.
struct Pending<'m> {
    item: &'m Item,
    ctrl: Vec<Name>,
    awaits: BTreeSet<Name>,
    span: Span,
}

impl Pending<'_> {
    /// A signal that `self` awaits and `other` controls.
    fn awaits_from<'p>(&self, other: &'p Pending) -> Option<&'p Name> {
        other.ctrl.iter().find(|x| self.awaits.contains(*x))
    }
}

/// Sorts the items so that each comes after the items controlling what it
/// awaits, keeping the source order where possible; a cycle is an error.
fn await_order<'m>(mut pending: Vec<Pending<'m>>) -> Result<Vec<&'m Item>> {
    let mut sorted = Vec::new();
    while !pending.is_empty() {
        let ready = |a: &Pending| !pending.iter().any(|b| a.awaits_from(b).is_some());
        match pending.iter().position(ready) {
            Some(i) => sorted.push(pending.remove(i).item),
            None => return Err(combinational_loop(&pending)),
        }
    }
    Ok(sorted)
}

/// An error naming a cycle among items none of which is ready.
fn combinational_loop(pending: &[Pending]) -> Error {
    // every item awaits some other pending item: walk until one repeats,
    // recording the signal by which each item is entered
    let mut path: Vec<(usize, Option<&Name>)> = vec![(0, None)];
    loop {
        let at = path.last().unwrap().0;
        let (next, x) = pending
            .iter()
            .enumerate()
            .find_map(|(i, b)| pending[at].awaits_from(b).map(|x| (i, x)))
            .expect("not ready");
        if let Some(start) = path.iter().position(|&(i, _)| i == next) {
            let inner = path[start + 1..].iter().map(|(_, x)| x.unwrap().as_str());
            let cycle: Vec<&str> = [x.as_str()]
                .into_iter()
                .chain(inner)
                .chain([x.as_str()])
                .collect();
            return Error::new(
                pending[next].span,
                format!("combinational loop: {}", cycle.join(" -> ")),
            );
        }
        path.push((next, Some(x)));
    }
}

/// Builds the terms of one atom.
struct Builder<'a> {
    env: &'a typing::Env,
    vars: &'a BTreeMap<Name, Var>,
    terms: Vec<Term>,
    /// The signals the atom reads or writes.
    used: BTreeSet<Name>,
    /// Whether signals are read latched (`*x`, in `always_ff`) or awaited
    /// (`X(x)`).
    latched: bool,
}

impl<'a> Builder<'a> {
    fn new(env: &'a typing::Env, vars: &'a BTreeMap<Name, Var>, latched: bool) -> Self {
        Self {
            env,
            vars,
            terms: Vec::new(),
            used: BTreeSet::new(),
            latched,
        }
    }

    /// Lowers a statement of an `always_ff` block, with `next` holding the
    /// wire of the value after the edge of each register so far.
    fn stmt(&mut self, s: &Stmt, next: &mut [(&Name, Wire)]) -> Result<()> {
        match &s.kind {
            StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| self.stmt(s, next)),
            StmtKind::If(c, a, b) => {
                let (wc, tc) = self.expr(c, None)?;
                let wc = self.truth(wc, tc, c.span)?;
                let mut then = next.to_vec();
                self.stmt(a, &mut then)?;
                let mut els = next.to_vec();
                self.stmt(b, &mut els)?;
                for ((entry, (_, wa)), (_, wb)) in next.iter_mut().zip(then).zip(els) {
                    entry.1 = if wa == wb {
                        wa
                    } else {
                        let sort = Sort(self.env[entry.0].ty.width);
                        self.emit(SvOp::Ite, &[wc, wa, wb], sort, None, s.span)?
                    };
                }
                Ok(())
            }
            StmtKind::NonBlocking { lhs, rhs } => {
                let (w, _) = self.expr(rhs, None)?;
                let entry = next.iter_mut().find(|(q, _)| *q == lhs).expect("a target");
                entry.1 = w;
                Ok(())
            }
        }
    }

    /// Makes `dest` hold the value of `w`: renames `w` to `dest` if a term
    /// of this atom writes `w` and none reads it, else adds `dest := w`.
    fn finish(&mut self, w: Wire, dest: Wire, span: Span) -> Result<()> {
        let writer = self.terms.iter().position(|t| t.write() == [w]);
        let read = self.terms.iter().any(|t| t.read().contains(&w));
        match writer {
            Some(i) if !read => {
                let t = &self.terms[i];
                let op = t.itype().clone();
                let term = if t.read().is_empty() {
                    Term::constant(op, [dest])
                } else {
                    Term::function(op, [dest], t.read().iter().copied())
                };
                self.terms[i] = term.map_err(|e| internal(span, e))?;
            }
            _ => {
                self.emit(SvOp::Id, &[w], *w.dtype(), Some(dest), span)?;
            }
        }
        Ok(())
    }

    /// Adds the term `dest := op(reads)`, with a fresh wire of `sort` for
    /// `dest` if none is given.
    fn emit(
        &mut self,
        op: SvOp,
        reads: &[Wire],
        sort: Sort,
        dest: Option<Wire>,
        span: Span,
    ) -> Result<Wire> {
        let w = dest.unwrap_or_else(|| Wire::scalar(sort));
        let term = if reads.is_empty() {
            Term::constant(op, [w])
        } else {
            Term::function(op, [w], reads.iter().copied())
        };
        self.terms.push(term.map_err(|e| internal(span, e))?);
        Ok(w)
    }

    /// The wire holding whether `w` (of type `t`) is non-zero.
    fn truth(&mut self, w: Wire, t: Ty, span: Span) -> Result<Wire> {
        if t.width == 1 {
            Ok(w)
        } else {
            self.emit(SvOp::RedOr, &[w], Sort(1), None, span)
        }
    }

    /// Lowers `e`, writing its value to `dest` if given, and returns the
    /// wire holding the value and its type.
    fn expr(&mut self, e: &Expr, dest: Option<Wire>) -> Result<(Wire, Ty)> {
        let span = e.span;
        match &e.kind {
            ExprKind::Lit { ty, value } => {
                let w = self.emit(SvOp::Lit(value.clone()), &[], Sort(ty.width), dest, span)?;
                Ok((w, *ty))
            }
            ExprKind::Var(x) => {
                let t = typing::lookup(self.env, x, span)?.ty;
                self.used.insert(x.clone());
                let var = self.vars[x];
                let w = if self.latched { *var } else { X(var) };
                match dest {
                    None => Ok((w, t)),
                    Some(_) => Ok((self.emit(SvOp::Id, &[w], Sort(t.width), dest, span)?, t)),
                }
            }
            ExprKind::Unary(op, a) => {
                let (wa, ta) = self.expr(a, None)?;
                let t = typing::unary(*op, ta);
                let s = Sort(t.width);
                let w = match op {
                    UnOp::Plus => match dest {
                        None => wa,
                        Some(_) => self.emit(SvOp::Id, &[wa], s, dest, span)?,
                    },
                    UnOp::BitNot => self.emit(SvOp::Not, &[wa], s, dest, span)?,
                    UnOp::Neg => self.emit(SvOp::Neg, &[wa], s, dest, span)?,
                    UnOp::LogNot => {
                        let b = self.truth(wa, ta, span)?;
                        self.emit(SvOp::Not, &[b], s, dest, span)?
                    }
                    UnOp::RedAnd => self.emit(SvOp::RedAnd, &[wa], s, dest, span)?,
                    UnOp::RedOr => self.emit(SvOp::RedOr, &[wa], s, dest, span)?,
                    UnOp::RedXor => self.emit(SvOp::RedXor, &[wa], s, dest, span)?,
                    UnOp::RedNand | UnOp::RedNor | UnOp::RedXnor => {
                        let red = match op {
                            UnOp::RedNand => SvOp::RedAnd,
                            UnOp::RedNor => SvOp::RedOr,
                            _ => SvOp::RedXor,
                        };
                        let r = self.emit(red, &[wa], s, None, span)?;
                        self.emit(SvOp::Not, &[r], s, dest, span)?
                    }
                };
                Ok((w, t))
            }
            ExprKind::Binary(op, a, b) => {
                let (wa, ta) = self.expr(a, None)?;
                let (wb, tb) = self.expr(b, None)?;
                let t = typing::binary(*op, ta, tb, span)?;
                let s = Sort(t.width);
                let (lt, le) = if ta.signed {
                    (SvOp::SLt, SvOp::SLe)
                } else {
                    (SvOp::ULt, SvOp::ULe)
                };
                let w = match op {
                    BinOp::BitAnd => self.emit(SvOp::And, &[wa, wb], s, dest, span)?,
                    BinOp::BitOr => self.emit(SvOp::Or, &[wa, wb], s, dest, span)?,
                    BinOp::BitXor => self.emit(SvOp::Xor, &[wa, wb], s, dest, span)?,
                    BinOp::BitXnor => {
                        let x = self.emit(SvOp::Xor, &[wa, wb], s, None, span)?;
                        self.emit(SvOp::Not, &[x], s, dest, span)?
                    }
                    BinOp::Add => self.emit(SvOp::Add, &[wa, wb], s, dest, span)?,
                    BinOp::Sub => self.emit(SvOp::Sub, &[wa, wb], s, dest, span)?,
                    BinOp::Mul => self.emit(SvOp::Mul, &[wa, wb], s, dest, span)?,
                    BinOp::Eq | BinOp::CaseEq => self.emit(SvOp::Eq, &[wa, wb], s, dest, span)?,
                    BinOp::Ne | BinOp::CaseNe => self.emit(SvOp::Ne, &[wa, wb], s, dest, span)?,
                    BinOp::Lt => self.emit(lt, &[wa, wb], s, dest, span)?,
                    BinOp::Le => self.emit(le, &[wa, wb], s, dest, span)?,
                    BinOp::Gt => self.emit(lt, &[wb, wa], s, dest, span)?,
                    BinOp::Ge => self.emit(le, &[wb, wa], s, dest, span)?,
                    BinOp::LogAnd | BinOp::LogOr => {
                        let x = self.truth(wa, ta, span)?;
                        let y = self.truth(wb, tb, span)?;
                        let op = if *op == BinOp::LogAnd {
                            SvOp::And
                        } else {
                            SvOp::Or
                        };
                        self.emit(op, &[x, y], s, dest, span)?
                    }
                    BinOp::Shl | BinOp::AShl => self.emit(SvOp::Shl, &[wa, wb], s, dest, span)?,
                    BinOp::Shr => self.emit(SvOp::LShr, &[wa, wb], s, dest, span)?,
                    BinOp::AShr => {
                        let op = if ta.signed { SvOp::AShr } else { SvOp::LShr };
                        self.emit(op, &[wa, wb], s, dest, span)?
                    }
                };
                Ok((w, t))
            }
            ExprKind::Cond(c, a, b) => {
                let (wc, tc) = self.expr(c, None)?;
                let wc = self.truth(wc, tc, span)?;
                let (wa, ta) = self.expr(a, None)?;
                let (wb, tb) = self.expr(b, None)?;
                let t = typing::cond(ta, tb, span)?;
                let w = self.emit(SvOp::Ite, &[wc, wa, wb], Sort(t.width), dest, span)?;
                Ok((w, t))
            }
        }
    }
}
