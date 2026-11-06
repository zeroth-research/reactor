//! Helpers shared by the `atom` and `module` integration tests: the
//! uninterpreted `Ops` theory and the example modules built over it.
#![allow(dead_code)]

use base::tangent::Tangent;
use base::wire::Wire;
use dataflow::diagram::{Block, Term};
use reactive::AtomInterface;
use reactive::var::{Var, Wire3, X, d, d_ref};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use theory::{Combinatorial, Differential, Sequential, Theory};

/// An uninterpreted sort: its name and how many times it is a tangent (`0` for
/// values), e.g., `Srt("real", 1)` is the sort of the derivatives of `real`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Srt(pub &'static str, pub u8);

/// The name of a variable in the tests: the id of its latched wire, so the
/// same wires get the same name in every atom
pub fn name<S: Clone>(var: &Wire3<S>) -> Var<S> {
    Var::new(u32::try_from(var.id()).unwrap(), var.dtype().clone())
}

/// The interface of the variables, under their names
pub fn interface<'a, S: Clone + PartialEq + Debug + 'a>(
    vars: impl IntoIterator<Item = &'a Wire3<S>>,
) -> AtomInterface<S> {
    AtomInterface::try_from_iter(vars.into_iter().map(name)).unwrap()
}

/// Builds the block with the boundary `read -> write`, discarding the inputs
/// no term reads
pub fn block<T, R, W, V>(read: R, write: W, terms: V) -> Result<Block<T>, String>
where
    T: Theory,
    T::Sort: Clone + Debug,
    R: IntoIterator<Item = Wire<T::Sort>>,
    W: IntoIterator<Item = Wire<T::Sort>>,
    V: IntoIterator<Item = Term<T>>,
{
    let read: Vec<Wire<T::Sort>> = read.into_iter().collect();
    let mut terms: Vec<Term<T>> = terms.into_iter().collect();

    let used: HashSet<&Wire<T::Sort>> = terms.iter().flat_map(Term::read).collect();
    let nothing: [Wire<T::Sort>; 0] = [];
    let discards = read
        .iter()
        .filter(|w| !used.contains(w))
        .map(|w| Term::function(T::discard(w.dtype()), nothing.clone(), [w.clone()]))
        .collect::<Result<Vec<_>, _>>()?;
    terms.extend(discards);

    Block::new(read, write, terms)
}

/// Builds the atom over `Ops` controlling `ctrl`, awaiting `wait` and
/// reading `read` (see [`generic_atom`])
pub fn atom(
    ctrl: &[Wire3<Srt>],
    wait: &[Wire3<Srt>],
    read: &[Wire3<Srt>],
    init: impl IntoIterator<Item = Term<Ops>>,
    update: impl IntoIterator<Item = Term<Ops>>,
    delay: impl IntoIterator<Item = Term<Ops>>,
) -> Result<Atom, String> {
    generic_atom(ctrl, wait, read, init, update, delay)
}

/// Builds the atom controlling `ctrl`, awaiting `wait` and reading `read`,
/// with blocks of the given terms over the wires of the variables. The blocks
/// take the wires in the order of the names of the variables (the order they
/// were created in), and discard the inputs they do not read.
pub fn generic_atom<I, J, F, S>(
    ctrl: &[Wire3<S>],
    wait: &[Wire3<S>],
    read: &[Wire3<S>],
    init: impl IntoIterator<Item = Term<I>>,
    update: impl IntoIterator<Item = Term<J>>,
    delay: impl IntoIterator<Item = Term<F>>,
) -> Result<reactive::Atom<I, J, F, S>, String>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + PartialEq + Debug + Tangent,
{
    // the variables are named after their latched wires, which order them
    // like their names
    let sorted = |vars: &[Wire3<S>]| vars.iter().cloned().collect::<BTreeSet<_>>();
    let (ctrl, wait, read) = (sorted(ctrl), sorted(wait), sorted(read));

    let ltc = |vars: &BTreeSet<Wire3<S>>| vars.iter().map(|v| (**v).clone()).collect::<Vec<_>>();
    let nxt = |vars: &BTreeSet<Wire3<S>>| vars.iter().map(X).collect::<Vec<_>>();
    let der = |vars: &BTreeSet<Wire3<S>>| vars.iter().map(d).collect::<Vec<_>>();

    let init = block(nxt(&wait), nxt(&ctrl), init)?;
    let update = block([ltc(&read), nxt(&wait)].concat(), nxt(&ctrl), update)?;
    let delay = block([ltc(&read), der(&wait)].concat(), der(&ctrl), delay)?;

    reactive::Atom::new(
        interface(&ctrl),
        interface(&wait),
        interface(&read),
        init,
        update,
        delay,
    )
}

/// The init leaving the next values of the variables unconstrained
pub fn havoc<I: Combinatorial>(vars: &[Wire3<I::Sort>]) -> Vec<Term<I>>
where
    I::Sort: Clone,
{
    let havoc = |v: &Wire3<I::Sort>| Term::constant(I::havoc(v.dtype()), [X(v)]).unwrap();
    vars.iter().map(havoc).collect()
}

/// The update keeping the values of the variables
pub fn skip<J: Sequential>(vars: &[Wire3<J::Sort>]) -> Vec<Term<J>>
where
    J::Sort: Clone,
{
    let skip =
        |v: &Wire3<J::Sort>| Term::function(J::skip(v.dtype()), [X(v)], [(**v).clone()]).unwrap();
    vars.iter().map(skip).collect()
}

/// The delay keeping the variables constant
pub fn zero<F: Differential>(vars: &[Wire3<F::Sort>]) -> Vec<Term<F>>
where
    F::Sort: Clone,
{
    let zero = |v: &Wire3<F::Sort>| Term::constant(F::zero(d_ref(v).dtype()), [d(v)]).unwrap();
    vars.iter().map(zero).collect()
}

/// The uninterpreted sort of values named `name`
pub fn srt(name: &'static str) -> Srt {
    Srt(name, 0)
}

impl Tangent for Srt {
    fn tangent(&self) -> Self {
        Srt(self.0, self.1 + 1)
    }
}

// Prints the tangents with primes, e.g., `real'`
impl Display for Srt {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}{}", self.0, "'".repeat(self.1 as usize))
    }
}

#[derive(Clone, Debug)]
#[allow(unused)]
pub struct Ops(pub &'static str);

impl Theory for Ops {
    type Sort = Srt;
    const NAME: &'static str = "Ops";

    fn check<R, W, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Self::Sort, E>>,
        W: IntoIterator<Item = Result<Self::Sort, E>>,
    {
        Ok(())
    }

    fn discard(_domain: &Self::Sort) -> Self {
        Ops("DISCARD")
    }
}

impl Combinatorial for Ops {
    fn havoc(_range: &Self::Sort) -> Self {
        Ops("HAVOC")
    }
}

impl Sequential for Ops {
    fn skip(_range: &Self::Sort) -> Self {
        Ops("SKIP")
    }
}

impl Differential for Ops {
    fn zero(_range: &Self::Sort) -> Self {
        Ops("ZERO")
    }
}

impl Display for Ops {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub type Module = reactive::Module<Ops, Ops, Ops>;
pub type Atom = reactive::Atom<Ops, Ops, Ops>;

pub fn mk_op(name: &'static str) -> Ops {
    Ops(name)
}

#[macro_export]
macro_rules! term {
    ($itype:expr, $write:expr) => {
        Term::constant($itype, $write)
    };

    ($itype:expr, $write:expr, $read:expr) => {
        Term::function($itype, $write, $read)
    };
}

#[allow(clippy::vec_init_then_push)]
pub fn example_counter() -> Result<(Module, HashMap<Var<Srt>, String>), String> {
    let x = Wire3::new(srt("real"));
    let y = Wire3::new(srt("real"));
    let z = Wire3::new(srt("real"));
    let y0 = Wire3::new(srt("real"));
    let z0 = Wire3::new(srt("real"));

    let mut init: Vec<Term<Ops>> = Vec::new();

    let tmp1 = Wire::new(srt("real"));
    init.push(term!(mk_op("ZERO"), [tmp1])?);

    let tmp2 = Wire::new(srt("bool"));
    let tmp3 = Wire::new(srt("bool"));
    init.push(term!(mk_op("ID"), [X(x)], [tmp1])?);
    init.push(term!(mk_op("ABS"), [tmp2], [X(y0)])?);
    init.push(term!(mk_op("ID"), [X(y)], [tmp2])?);
    init.push(term!(mk_op("ABS"), [tmp3], [X(z0)])?);
    init.push(term!(mk_op("ID"), [X(z)], [tmp3])?);

    let mut update: Vec<Term<Ops>> = Vec::new();

    let tmp4 = Wire::new(srt("bool"));
    let tmp5 = Wire::new(srt("bool"));
    let tmp6 = Wire::new(srt("bool"));
    update.push(term!(mk_op("ZERO"), [tmp1])?);
    update.push(term!(mk_op("LEQ"), [tmp4], [x, y])?);
    update.push(term!(mk_op("LEQ"), [tmp5], [x, z])?);
    update.push(term!(mk_op("OR"), [tmp6], [tmp4, tmp5])?);

    let tmp7 = Wire::new(srt("real"));
    let tmp8 = Wire::new(srt("real"));
    update.push(term!(mk_op("ONE"), [tmp7])?);
    update.push(term!(mk_op("ADD"), [tmp8], [*x, tmp7])?);

    update.push(term!(mk_op("ITE"), [X(x)], [tmp6, tmp8, tmp1])?);
    update.push(term!(mk_op("ID"), [X(y)], [y0])?);
    update.push(term!(mk_op("ID"), [X(z)], [z0])?);

    let ctrl = [x, y, z];
    let atom = atom(
        &ctrl,
        &[y0, z0],
        &[x, y, z, y0, z0],
        init,
        update,
        zero(&ctrl),
    )?;
    let module = Module::observable([atom])?;
    // every module variable needs a name: the display indexes them all
    let varnames = [(x, "x"), (y, "y"), (z, "z"), (y0, "y0"), (z0, "z0")]
        .into_iter()
        .map(|(var, varname)| (name(&var), varname.to_string()))
        .collect();

    Ok((module, varnames))
}

#[allow(clippy::vec_init_then_push)]
pub fn example_peterson1() -> Result<Module, String> {
    let stype = "{outCS, reqCS, inCS}";
    let pc1 = Wire3::new(srt(stype));
    let x1 = Wire3::new(srt("bool"));
    let pc2 = Wire3::new(srt(stype));
    let x2 = Wire3::new(srt("bool"));

    let mut init: Vec<Term<Ops>> = Vec::new();
    init.push(term!(mk_op("CONST(outCS)"), [X(pc1)]).unwrap());
    init.push(term!(mk_op("CONST(true)"), [X(x1)]).unwrap());

    let mut update: Vec<Term<Ops>> = Vec::new();
    let out_cs = Wire::new(srt(stype));
    let cond1 = Wire::new(srt("bool"));
    update.push(term!(mk_op("CONST(outCS)"), [out_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond1], [out_cs, *pc1]).unwrap());

    let req_cs = Wire::new(srt(stype));
    let cond2 = Wire::new(srt("bool"));
    update.push(term!(mk_op("CONST(reqCS)"), [req_cs]).unwrap());
    let tmp11 = Wire::new(srt("bool"));
    update.push(term!(mk_op("EQ"), [tmp11], [req_cs, *pc1]).unwrap());

    let tmp12 = Wire::new(srt("bool"));
    let tmp13 = Wire::new(srt("bool"));
    let tmp14 = Wire::new(srt("bool"));
    update.push(term!(mk_op("EQ"), [tmp12], [out_cs, *pc2]).unwrap());
    update.push(term!(mk_op("NEQ"), [tmp13], [x1, x2]).unwrap());
    update.push(term!(mk_op("OR"), [tmp14], [tmp12, tmp13]).unwrap());
    update.push(term!(mk_op("AND"), [cond2], [tmp14, tmp11]).unwrap());

    let in_cs = Wire::new(srt(stype));
    let cond3 = Wire::new(srt("bool"));
    update.push(term!(mk_op("CONST(inCS)"), [in_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond3], [in_cs, *pc1]).unwrap());

    let const_true = Wire::new(srt("bool"));
    update.push(term!(mk_op("CONST(true)"), [const_true]).unwrap());

    update.push(
        term!(
            mk_op("CASE"),
            [X(pc1), X(x1)],
            [
                cond1, req_cs, *x2, cond2, in_cs, *x1, cond3, out_cs, *x1, const_true, *pc1, *x1,
            ]
        )
        .unwrap(),
    );

    let ctrl = [pc1, x1];
    let atom = atom(&ctrl, &[], &[pc1, x1, pc2, x2], init, update, zero(&ctrl))?;
    Module::observable([atom])
}

pub fn example_tiny1(
    external: Wire3<Srt>,
    interface: Wire3<Srt>,
    wait: bool,
) -> Result<Module, String> {
    let private = Wire3::new(srt("Tny"));
    let temp = Wire::new(srt("Tny"));

    let cons = Term::constant(mk_op("CONST"), [temp]).unwrap();

    let update = if wait {
        Term::function(
            mk_op("AWAIT"),
            [X(interface), X(private)],
            [X(external), *private, temp],
        )
        .unwrap()
    } else {
        Term::function(
            mk_op("SEQ"),
            [X(interface), X(private)],
            [*external, *private, temp],
        )
        .unwrap()
    };

    let init = Term::constant(mk_op("INIT"), [X(interface), X(private)]).unwrap();

    let ctrl = [interface, private];
    let (awaited, read) = if wait {
        (vec![external], vec![private])
    } else {
        (vec![], vec![external, private])
    };
    let prvt = [name(&private)];

    let atom = atom(&ctrl, &awaited, &read, [init], [cons, update], zero(&ctrl))?;
    Module::partially_observable([atom], |v| prvt.contains(v))
}
