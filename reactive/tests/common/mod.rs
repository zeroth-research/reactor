//! Helpers shared by the `atom` and `module` integration tests: the
//! uninterpreted `Ops` theory, the blocks and the example modules built over
//! it.
#![allow(dead_code)]

use base::tangent::Tangent;
use dataflow::diagram::{Block, Term, Wire};
use reactive::AtomInterface;
use reactive::var::Var;
use std::collections::HashMap;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use theory::{Combinatorial, Differential, Sequential, Theory};

/// An uninterpreted sort: its name and how many times it is a tangent (`0` for
/// values), e.g., `Srt("real", 1)` is the sort of the derivatives of `real`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Srt(pub &'static str, pub u8);

/// The variable of the name, holding values of the sort named `sort`
pub fn var(name: u32, sort: &'static str) -> Var<Srt> {
    Var::new(name, srt(sort))
}

/// The interface of the variables
pub fn iface<S: Clone + PartialEq + Debug>(
    vars: impl IntoIterator<Item = Var<S>>,
) -> AtomInterface<S> {
    AtomInterface::try_from_iter(vars).unwrap()
}

/// A fresh wire of the sort of values named `sort`
pub fn wire(sort: &'static str) -> Wire<Srt> {
    Wire::new(srt(sort))
}

/// A fresh wire of the tangent of the sort of values named `sort`: a wire
/// carrying derivatives
pub fn dwire(sort: &'static str) -> Wire<Srt> {
    Wire::new(srt(sort).tangent())
}

/// A fresh wire carrying values of the variable
pub fn wire_of<S: Clone>(var: &Var<S>) -> Wire<S> {
    Wire::new(var.dtype().clone())
}

/// A fresh wire carrying derivatives of the variable
pub fn dwire_of<S: Tangent>(var: &Var<S>) -> Wire<S> {
    Wire::new(var.dtype().tangent())
}

/// Fresh wires carrying values of the variables, in order
pub fn wires<S: Clone>(vars: &AtomInterface<S>) -> Vec<Wire<S>> {
    vars.iter().map(wire_of).collect()
}

/// Fresh wires carrying derivatives of the variables, in order
pub fn dwires<S: Tangent>(vars: &AtomInterface<S>) -> Vec<Wire<S>> {
    vars.iter().map(dwire_of).collect()
}

/// The wires, each carrying a variable, in the order of the names of the
/// variables: the order of the interfaces
pub fn by_name<S>(wires: impl IntoIterator<Item = (Var<S>, Wire<S>)>) -> Vec<Wire<S>> {
    let mut wires: Vec<(Var<S>, Wire<S>)> = wires.into_iter().collect();
    wires.sort_by_key(|(var, _)| var.name());
    wires.into_iter().map(|(_, wire)| wire).collect()
}

/// The term discarding the wire
pub fn discard<T: Theory>(wire: &Wire<T::Sort>) -> Term<T>
where
    T::Sort: Clone,
{
    let nothing: [Wire<T::Sort>; 0] = [];
    Term::function(T::discard(wire.dtype()), nothing, [wire.clone()]).unwrap()
}

/// The init `X(wait) -> X(ctrl)` leaving the controlled variables
/// unconstrained
pub fn havoc_init<I, S>(ctrl: &AtomInterface<S>, wait: &AtomInterface<S>) -> Block<I>
where
    I: Combinatorial<Sort = S>,
    S: Clone + Debug,
{
    let (inputs, outputs) = (wires(wait), wires(ctrl));
    let havoc = |w: &Wire<S>| Term::constant(I::havoc(w.dtype()), [w.clone()]).unwrap();
    let terms = outputs.iter().map(havoc).chain(inputs.iter().map(discard));
    Block::new(inputs.clone(), outputs.clone(), terms.collect::<Vec<_>>()).unwrap()
}

/// The update `read ++ X(wait) -> X(ctrl)` keeping the controlled
/// variables, all of which it reads
pub fn skip_update<J, S>(
    ctrl: &AtomInterface<S>,
    wait: &AtomInterface<S>,
    read: &AtomInterface<S>,
) -> Block<J>
where
    J: Sequential<Sort = S>,
    S: Clone + Debug,
{
    let inputs = [wires(read), wires(wait)].concat();
    let outputs = wires(ctrl);

    let mut used = vec![false; inputs.len()];
    let mut terms: Vec<Term<J>> = Vec::new();
    for (var, out) in ctrl.iter().zip(outputs.iter()) {
        let at = read
            .iter()
            .position(|v| v == var)
            .expect("skip reads the variables it keeps");
        used[at] = true;
        let skip = J::skip(out.dtype());
        terms.push(Term::function(skip, [out.clone()], [inputs[at].clone()]).unwrap());
    }
    let unused = inputs.iter().zip(used).filter(|(_, used)| !used);
    terms.extend(unused.map(|(w, _)| discard(w)));

    Block::new(inputs, outputs, terms).unwrap()
}

/// The delay `read ++ d(wait) -> d(ctrl)` keeping the controlled variables
/// constant
pub fn zero_delay<F, S>(
    ctrl: &AtomInterface<S>,
    wait: &AtomInterface<S>,
    read: &AtomInterface<S>,
) -> Block<F>
where
    F: Differential<Sort = S>,
    S: Clone + Debug + Tangent,
{
    let inputs = [wires(read), dwires(wait)].concat();
    let outputs = dwires(ctrl);
    let zero = |w: &Wire<S>| Term::constant(F::zero(w.dtype()), [w.clone()]).unwrap();
    let terms = outputs.iter().map(zero).chain(inputs.iter().map(discard));
    Block::new(inputs.clone(), outputs.clone(), terms.collect::<Vec<_>>()).unwrap()
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

/// A counter over the variables `x`, `y`, `z` and the inputs `y0`, `z0`
/// (named `0` to `4`), with the names of the variables
pub fn example_counter() -> Result<(Module, HashMap<Var<Srt>, String>), String> {
    let [x, y, z, y0, z0] = [0, 1, 2, 3, 4].map(|n| var(n, "real"));
    let ctrl = iface([x, y, z]);
    let wait = iface([y0, z0]);
    let read = iface([x, y, z, y0, z0]);

    // X(y0), X(z0) -> X(x), X(y), X(z)
    let init = {
        let [ny0, nz0, nx, ny, nz] = ["real"; 5].map(wire);
        let tmp1 = wire("real");
        let tmp2 = wire("bool");
        let tmp3 = wire("bool");
        let terms = [
            term!(mk_op("ZERO"), [tmp1])?,
            term!(mk_op("ID"), [nx], [tmp1])?,
            term!(mk_op("ABS"), [tmp2], [ny0])?,
            term!(mk_op("ID"), [ny], [tmp2])?,
            term!(mk_op("ABS"), [tmp3], [nz0])?,
            term!(mk_op("ID"), [nz], [tmp3])?,
        ];
        Block::new([ny0, nz0], [nx, ny, nz], terms)?
    };

    // x, y, z, y0, z0, X(y0), X(z0) -> X(x), X(y), X(z)
    let update = {
        let [x, y, z, y0, z0, ny0, nz0] = ["real"; 7].map(wire);
        let [nx, ny, nz] = ["real"; 3].map(wire);
        let tmp1 = wire("real");
        let [tmp4, tmp5, tmp6] = ["bool"; 3].map(wire);
        let [tmp7, tmp8] = ["real"; 2].map(wire);
        let terms = [
            term!(mk_op("ZERO"), [tmp1])?,
            term!(mk_op("LEQ"), [tmp4], [x, y])?,
            term!(mk_op("LEQ"), [tmp5], [x, z])?,
            term!(mk_op("OR"), [tmp6], [tmp4, tmp5])?,
            term!(mk_op("ONE"), [tmp7])?,
            term!(mk_op("ADD"), [tmp8], [x, tmp7])?,
            term!(mk_op("ITE"), [nx], [tmp6, tmp8, tmp1])?,
            term!(mk_op("ID"), [ny], [y0])?,
            term!(mk_op("ID"), [nz], [z0])?,
            discard(&ny0),
            discard(&nz0),
        ];
        Block::new([x, y, z, y0, z0, ny0, nz0], [nx, ny, nz], terms)?
    };

    let delay = zero_delay(&ctrl, &wait, &read);
    let atom = Atom::new(ctrl, wait, read, init, update, delay)?;
    let module = Module::observable([atom])?;

    // every module variable needs a name: the display indexes them all
    let varnames = [(x, "x"), (y, "y"), (z, "z"), (y0, "y0"), (z0, "z0")]
        .into_iter()
        .map(|(var, varname)| (var, varname.to_string()))
        .collect();

    Ok((module, varnames))
}

/// One of the two processes of Peterson's mutual exclusion protocol
pub fn example_peterson1() -> Result<Module, String> {
    let stype = "{outCS, reqCS, inCS}";
    let [pc1, x1, pc2, x2] = [var(0, stype), var(1, "bool"), var(2, stype), var(3, "bool")];
    let ctrl = iface([pc1, x1]);
    let wait = iface([]);
    let read = iface([pc1, x1, pc2, x2]);

    // () -> X(pc1), X(x1)
    let init = {
        let (npc1, nx1) = (wire(stype), wire("bool"));
        let terms = [
            term!(mk_op("CONST(outCS)"), [npc1])?,
            term!(mk_op("CONST(true)"), [nx1])?,
        ];
        Block::new([], [npc1, nx1], terms)?
    };

    // pc1, x1, pc2, x2 -> X(pc1), X(x1)
    let update = {
        let (pc1, x1, pc2, x2) = (wire(stype), wire("bool"), wire(stype), wire("bool"));
        let (npc1, nx1) = (wire(stype), wire("bool"));
        let [out_cs, req_cs, in_cs] = [stype; 3].map(wire);
        let [cond1, cond2, cond3, const_true] = ["bool"; 4].map(wire);
        let [tmp11, tmp12, tmp13, tmp14] = ["bool"; 4].map(wire);
        let terms = [
            term!(mk_op("CONST(outCS)"), [out_cs])?,
            term!(mk_op("EQ"), [cond1], [out_cs, pc1])?,
            term!(mk_op("CONST(reqCS)"), [req_cs])?,
            term!(mk_op("EQ"), [tmp11], [req_cs, pc1])?,
            term!(mk_op("EQ"), [tmp12], [out_cs, pc2])?,
            term!(mk_op("NEQ"), [tmp13], [x1, x2])?,
            term!(mk_op("OR"), [tmp14], [tmp12, tmp13])?,
            term!(mk_op("AND"), [cond2], [tmp14, tmp11])?,
            term!(mk_op("CONST(inCS)"), [in_cs])?,
            term!(mk_op("EQ"), [cond3], [in_cs, pc1])?,
            term!(mk_op("CONST(true)"), [const_true])?,
            term!(
                mk_op("CASE"),
                [npc1, nx1],
                [
                    cond1, req_cs, x2, cond2, in_cs, x1, cond3, out_cs, x1, const_true, pc1, x1
                ]
            )?,
        ];
        Block::new([pc1, x1, pc2, x2], [npc1, nx1], terms)?
    };

    let delay = zero_delay(&ctrl, &wait, &read);
    let atom = Atom::new(ctrl, wait, read, init, update, delay)?;
    Module::observable([atom])
}

/// The module controlling `interface` and the private `private` from
/// `external`, which it awaits if `wait` and reads otherwise
pub fn example_tiny1(
    external: Var<Srt>,
    interface: Var<Srt>,
    private: Var<Srt>,
    wait: bool,
) -> Result<Module, String> {
    let ctrl = iface([interface, private]);
    let (awaited, read) = if wait {
        (iface([external]), iface([private]))
    } else {
        (iface([]), iface([external, private]))
    };

    // X(external) if awaited -> X(interface), X(private)
    let init = {
        let (ni, np) = (wire_of(&interface), wire_of(&private));
        let ne = wait.then(|| wire_of(&external));
        let mut terms = vec![Term::constant(mk_op("INIT"), [ni, np])?];
        terms.extend(ne.iter().map(discard));
        Block::new(ne, by_name([(interface, ni), (private, np)]), terms)?
    };

    // private, X(external) if awaited, else external, private ->
    // X(interface), X(private)
    let update = {
        let (ni, np) = (wire_of(&interface), wire_of(&private));
        let (e, p) = (wire_of(&external), wire_of(&private));
        let temp = wire("Tny");
        let (op, inputs) = if wait {
            ("AWAIT", vec![p, e])
        } else {
            ("SEQ", by_name([(external, e), (private, p)]))
        };
        let terms = [
            Term::constant(mk_op("CONST"), [temp])?,
            Term::function(mk_op(op), [ni, np], [e, p, temp])?,
        ];
        Block::new(inputs, by_name([(interface, ni), (private, np)]), terms)?
    };

    let delay = zero_delay(&ctrl, &awaited, &read);
    let atom = Atom::new(ctrl, awaited, read, init, update, delay)?;
    Module::partially_observable([atom], |v| *v == private)
}
