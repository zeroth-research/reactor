/*!
# Into `theory::any`

[`export`] renames a module over [`Op`] term by term into a module of the
`theory` crate's union `theory::any`, the form the Python side takes: each
variable becomes one of sort `BitVec(w, [1, 1])` (a 1×1 matrix of `w`-bit
words), each wire one of the same sort, each op the op of `theory::bv`
of the same name (`HAVOC` that of `theory::any`), with the same wires in
the same places. Atoms keep their order and kind, and the module is built
by the same constructors, which check it again.

A constant of `theory::bv` holds a tensor, whose type depends on features
of `theory` that this crate leaves off; the caller passes the function that
makes the constant op of a value and a width.
*/

use super::{Desugared, Op};
use crate::ast::{Error, ErrorKind};
use ::theory::any::{Combinatorial, Differential, Sequential, Sort};
use ::theory::bv::BV;
use dataflow::diagram::{Term, Wire};
use std::collections::{BTreeSet, HashMap, HashSet};

/// A module of `theory::any`.
pub type Module = reactive::Module<Combinatorial, Sequential, Differential, Sort>;
/// A variable of a module of `theory::any`.
pub type Var = reactive::Var<Sort>;

/// The module of a [`Desugared`] module in `theory::any`.
pub struct Exported {
    pub module: Module,
    /// The new variable of each variable of the desugared module.
    pub vars: HashMap<super::Var, Var>,
}

/// The sort of a word of `w` bits.
pub fn sort(w: usize) -> Sort {
    Sort::BitVec(w, [1, 1])
}

/// The module of `d` in `theory::any`, with the constants made by
/// `constant(value, width)`.
pub fn export(
    d: &Desugared,
    mut constant: impl FnMut(u64, usize) -> BV,
) -> Result<Exported, Error> {
    let internal = |e: String| Error::unlocated(ErrorKind::Internal, e);
    let m = &d.module;
    let mut vars = HashMap::new();
    let mut wires: HashMap<super::Wire, Wire<Sort>> = HashMap::new();
    for old in m
        .extl()
        .iter()
        .chain(m.intf().iter())
        .chain(m.prvt().iter())
    {
        let new = Var::new(sort(old.dtype().width()));
        vars.insert(*old, new);
        wires.insert(**old, *new);
        wires.insert(reactive::var::X(old), reactive::var::X(new));
    }
    let mut atoms = vec![];
    for a in m.atoms() {
        let scope: BTreeSet<Var> = (a.ctrl().iter().chain(a.wait().iter()))
            .chain(a.read().iter())
            .map(|v| vars[v])
            .collect();
        let mut locals = HashMap::new();
        let mut block = |terms: &mut dyn Iterator<Item = &super::Term>| {
            let mut out = vec![];
            for t in terms {
                let mut wire = |w: &super::Wire| {
                    *wires.get(w).unwrap_or_else(|| {
                        locals
                            .entry(*w)
                            .or_insert_with(|| Wire::new(sort(w.dtype().width())))
                    })
                };
                let write: Vec<Wire<Sort>> = t.write().iter().map(&mut wire).collect();
                let read: Vec<Wire<Sort>> = t.read().iter().map(&mut wire).collect();
                let op = op(t.itype(), &mut constant)?;
                out.push(Term::function(op, write, read)?);
            }
            Ok::<_, String>(out)
        };
        let atom = if a.init().len() == a.update().len()
            && a.init().iter().zip(a.update().iter()).all(|(x, y)| {
                x.itype() == y.itype() && x.write() == y.write() && x.read() == y.read()
            }) {
            let assign = block(&mut a.init().iter()).map_err(internal)?;
            reactive::Atom::combinatorial::<Combinatorial, _, _>(&scope, assign)
        } else {
            let init = block(&mut a.init().iter()).map_err(internal)?;
            let update = block(&mut a.update().iter()).map_err(internal)?;
            let update = update.into_iter().map(|t| {
                let op = Sequential::from(t.itype().clone());
                Term::function(op, t.write().to_vec(), t.read().to_vec())
            });
            let update: Vec<_> = update.collect::<Result<_, _>>().map_err(internal)?;
            reactive::Atom::sequential(&scope, init, update)
        };
        atoms.push(atom.map_err(internal)?);
    }
    let hidden: HashSet<Var> = m.prvt().iter().map(|v| vars[v]).collect();
    let module = Module::partially_observable(atoms, |v| hidden.contains(v)).map_err(internal)?;
    Ok(Exported { module, vars })
}

/// The op of `theory::any` of `op`.
fn op(op: &Op, constant: &mut impl FnMut(u64, usize) -> BV) -> Result<Combinatorial, String> {
    Ok(Combinatorial::BV(match *op {
        Op::Const { value, width } => constant(value, width),
        Op::Havoc(s) => return Ok(Combinatorial::HAVOC(sort(s.width()))),
        Op::Zero(_) => return Err("ZERO in an init or update block".to_string()),
        Op::Id => BV::Id(),
        Op::Not => BV::Not(),
        Op::And => BV::And(),
        Op::Or => BV::Or(),
        Op::Xor => BV::Xor(),
        Op::Neg => BV::Neg(),
        Op::Add => BV::Add(),
        Op::Sub => BV::Sub(),
        Op::Mul => BV::Mul(),
        Op::UDiv => BV::UDiv(),
        Op::UMod => BV::UMod(),
        Op::Eq => BV::Eq(),
        Op::Ne => BV::Ne(),
        Op::ULt => BV::ULt(),
        Op::ULe => BV::ULe(),
        Op::Ite => BV::Ite(),
        Op::BitSelect { high, low } => BV::BitSelect { high, low },
        Op::Extend { extra } => BV::Extend { extra },
    }))
}
