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

The atoms are sorted so that each comes after the atoms controlling what it
awaits; a cycle is a combinational loop and is rejected.
*/

use crate::ast::{self, BinOp, Dir, Error, Expr, ExprKind, Item, Name, Span, Ty, UnOp};
use crate::typing;
use crate::{Atom, Module, Sort, SvOp, Term, Var, Wire};
use base::var::X;
use std::collections::{BTreeMap, BTreeSet};

type Result<T> = std::result::Result<T, Error>;

/// A module with the names of its variables.
pub struct Translation {
    pub name: Name,
    pub module: Module,
    /// Every source signal, by name.
    pub vars: BTreeMap<Name, Var>,
    /// Inputs that nothing reads; they are not variables of the module.
    pub unread_inputs: Vec<Name>,
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
    // in declaration order, so that variable ids (and printing) follow it
    let decls = m.ports.iter().map(|p| &p.decl).chain(&m.decls);
    let vars: BTreeMap<Name, Var> = decls
        .map(|d| (d.name.clone(), Var::new(Sort(d.ty.width))))
        .collect();

    // the await order is computed on the AST, so that a loop is reported
    // as such before any term is built
    let pending: Vec<Pending> = m
        .items
        .iter()
        .map(|item| match item {
            Item::Assign { lhs, rhs, span } => {
                let mut reads = BTreeSet::new();
                free_vars(rhs, &mut reads);
                Pending {
                    item,
                    ctrl: lhs.clone(),
                    reads,
                    span: *span,
                }
            }
        })
        .collect();
    let read: BTreeSet<&Name> = pending.iter().flat_map(|a| &a.reads).collect();
    let unread_inputs = m
        .ports
        .iter()
        .filter(|p| p.dir == Dir::Input && !read.contains(&p.decl.name))
        .map(|p| p.decl.name.clone())
        .collect();

    let mut atoms = Vec::new();
    for item in await_order(pending)? {
        match item {
            Item::Assign { lhs, rhs, span } => {
                let mut b = Builder {
                    env: &env,
                    vars: &vars,
                    terms: Vec::new(),
                    used: BTreeSet::from([lhs.clone()]),
                };
                b.expr(rhs, Some(X(vars[lhs])))?;
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                atoms.push(Atom::combinatorial(scope, b.terms).map_err(|e| internal(*span, e))?);
            }
        }
    }

    let private: BTreeSet<Var> = m.decls.iter().map(|d| vars[&d.name]).collect();
    let module = Module::partially_observable(atoms, |v| private.contains(v))
        .map_err(|e| internal(Span::default(), e))?;
    Ok(Translation {
        name: m.name.clone(),
        module,
        vars,
        unread_inputs,
    })
}

fn internal(span: Span, e: String) -> Error {
    Error::new(span, format!("internal error: {e}"))
}

/// The signals an expression reads.
fn free_vars(e: &Expr, out: &mut BTreeSet<Name>) {
    match &e.kind {
        ExprKind::Lit { .. } => {}
        ExprKind::Var(x) => {
            out.insert(x.clone());
        }
        ExprKind::Unary(_, a) => free_vars(a, out),
        ExprKind::Binary(_, a, b) => {
            free_vars(a, out);
            free_vars(b, out);
        }
        ExprKind::Cond(c, a, b) => {
            free_vars(c, out);
            free_vars(a, out);
            free_vars(b, out);
        }
    }
}

/// An item before sorting, with the signal it controls and those it awaits.
struct Pending<'m> {
    item: &'m Item,
    ctrl: Name,
    reads: BTreeSet<Name>,
    span: Span,
}

/// Sorts the items so that each comes after the items controlling what it
/// awaits, keeping the source order where possible; a cycle is an error.
fn await_order<'m>(mut pending: Vec<Pending<'m>>) -> Result<Vec<&'m Item>> {
    let mut sorted = Vec::new();
    while !pending.is_empty() {
        let ready = |a: &Pending| !a.reads.iter().any(|x| pending.iter().any(|b| &b.ctrl == x));
        match pending.iter().position(ready) {
            Some(i) => sorted.push(pending.remove(i).item),
            None => return Err(combinational_loop(&pending)),
        }
    }
    Ok(sorted)
}

/// An error naming a cycle among items none of which is ready.
fn combinational_loop(pending: &[Pending]) -> Error {
    // every item awaits some other pending item: walk until one repeats
    let mut path: Vec<usize> = vec![0];
    loop {
        let a = &pending[*path.last().unwrap()];
        let next = pending
            .iter()
            .position(|b| a.reads.contains(&b.ctrl))
            .expect("not ready");
        if let Some(start) = path.iter().position(|&i| i == next) {
            let cycle: Vec<&str> = path[start..]
                .iter()
                .chain([&next])
                .map(|&i| pending[i].ctrl.as_str())
                .collect();
            return Error::new(
                pending[next].span,
                format!("combinational loop: {}", cycle.join(" -> ")),
            );
        }
        path.push(next);
    }
}

/// Builds the terms of one atom.
struct Builder<'a> {
    env: &'a typing::Env,
    vars: &'a BTreeMap<Name, Var>,
    terms: Vec<Term>,
    /// The signals the atom reads or writes.
    used: BTreeSet<Name>,
}

impl Builder<'_> {
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
                let w = X(self.vars[x]);
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
