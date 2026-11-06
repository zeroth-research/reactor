//! Helpers shared by the `atom` and `module` integration tests: the
//! uninterpreted `Ops` theory, the blocks and the example modules built over
//! it.
#![allow(dead_code)]

use base::tangent::Tangent;
use dataflow::diagram::Block;
use reactive::AtomInterface;
use reactive::var::Var;
use std::collections::HashMap;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use theory::Theory;

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

/// The sorts of the variables, in order
pub fn sorts<S: Clone>(vars: &AtomInterface<S>) -> Vec<S> {
    vars.iter().map(|var| var.dtype().clone()).collect()
}

/// The sorts of the derivatives of the variables, in order
pub fn tangents<S: Tangent>(vars: &AtomInterface<S>) -> Vec<S> {
    vars.iter().map(|var| var.dtype().tangent()).collect()
}

/// The boundary `X(wait) -> X(ctrl)` of an init: the sorts of its inputs
/// and of its outputs
pub fn init_boundary<S: Clone>(
    ctrl: &AtomInterface<S>,
    wait: &AtomInterface<S>,
) -> (Vec<S>, Vec<S>) {
    (sorts(wait), sorts(ctrl))
}

/// The boundary `read ++ X(wait) -> X(ctrl)` of an update: the sorts of its
/// inputs and of its outputs
pub fn update_boundary<S: Clone>(
    ctrl: &AtomInterface<S>,
    wait: &AtomInterface<S>,
    read: &AtomInterface<S>,
) -> (Vec<S>, Vec<S>) {
    ([sorts(read), sorts(wait)].concat(), sorts(ctrl))
}

/// The boundary `read ++ d(wait) -> d(ctrl)` of a delay: the sorts of its
/// inputs and of its outputs
pub fn delay_boundary<S: Clone + Tangent>(
    ctrl: &AtomInterface<S>,
    wait: &AtomInterface<S>,
    read: &AtomInterface<S>,
) -> (Vec<S>, Vec<S>) {
    ([sorts(read), tangents(wait)].concat(), tangents(ctrl))
}

/// The init `X(wait) -> X(ctrl)` leaving the controlled variables
/// unconstrained, each by its own `HAVOC`
pub fn havoc_init(ctrl: &AtomInterface<Srt>, wait: &AtomInterface<Srt>) -> Block<Ops> {
    let (dom, cod) = init_boundary(ctrl, wait);
    let ops = cod.into_iter().map(|s| (Ops("HAVOC"), [], s));
    Block::pointwise(dom, ops).unwrap()
}

/// The update `read ++ X(wait) -> X(ctrl)` keeping the controlled
/// variables, all of which it reads, each through its own `SKIP`
pub fn skip_update(
    ctrl: &AtomInterface<Srt>,
    wait: &AtomInterface<Srt>,
    read: &AtomInterface<Srt>,
) -> Block<Ops> {
    let (dom, _) = update_boundary(ctrl, wait, read);
    let ops = ctrl.iter().map(|var| {
        let at = read
            .iter()
            .position(|v| v == var)
            .expect("skip reads the variables it keeps");
        (Ops("SKIP"), [at], *var.dtype())
    });
    Block::pointwise(dom, ops).unwrap()
}

/// The delay `read ++ d(wait) -> d(ctrl)` keeping the controlled variables
/// constant, each by its own `ZERO`
pub fn zero_delay(
    ctrl: &AtomInterface<Srt>,
    wait: &AtomInterface<Srt>,
    read: &AtomInterface<Srt>,
) -> Block<Ops> {
    let (dom, cod) = delay_boundary(ctrl, wait, read);
    let ops = cod.into_iter().map(|s| (Ops("ZERO"), [], s));
    Block::pointwise(dom, ops).unwrap()
}

/// The uninterpreted sort of values named `name`
pub fn srt(name: &'static str) -> Srt {
    Srt(name, 0)
}

/// The sort of the derivatives of the values of the sort named `name`
pub fn dsrt(name: &'static str) -> Srt {
    srt(name).tangent()
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

impl Display for Ops {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub type Module = reactive::Module<Ops>;
pub type Atom = reactive::Atom<Ops>;

pub fn mk_op(name: &'static str) -> Ops {
    Ops(name)
}

/// A counter over the variables `x`, `y`, `z` and the inputs `y0`, `z0`
/// (named `0` to `4`), with the names of the variables
pub fn example_counter() -> Result<(Module, HashMap<Var<Srt>, String>), String> {
    let [x, y, z, y0, z0] = [0, 1, 2, 3, 4].map(|n| var(n, "real"));
    let ctrl = iface([x, y, z]);
    let wait = iface([y0, z0]);
    let read = iface([x, y, z, y0, z0]);

    // X(y0), X(z0) -> X(x), X(y), X(z): x = 0, y = y0, z = z0
    let (dom, _) = init_boundary(&ctrl, &wait);
    let ops = [
        (mk_op("ZERO"), vec![], srt("real")),
        (mk_op("ID"), vec![0], srt("real")),
        (mk_op("ID"), vec![1], srt("real")),
    ];
    let init = Block::pointwise(dom, ops)?;

    // x, y, z, y0, z0, X(y0), X(z0) -> X(x), X(y), X(z): x counts up while
    // below y or z, y = y0, z = z0
    let (dom, _) = update_boundary(&ctrl, &wait, &read);
    let ops = [
        (mk_op("COUNT"), vec![0, 1, 2], srt("real")),
        (mk_op("ID"), vec![3], srt("real")),
        (mk_op("ID"), vec![4], srt("real")),
    ];
    let update = Block::pointwise(dom, ops)?;

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
    let ops = [
        (mk_op("CONST(outCS)"), vec![], srt(stype)),
        (mk_op("CONST(true)"), vec![], srt("bool")),
    ];
    let init = Block::pointwise([], ops)?;

    // pc1, x1, pc2, x2 -> X(pc1), X(x1): the protocol step
    let (dom, cod) = update_boundary(&ctrl, &wait, &read);
    let update = Block::op(mk_op("PETERSON"), dom, cod)?;

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
    let (dom, cod) = init_boundary(&ctrl, &awaited);
    let init = Block::op(mk_op("INIT"), dom, cod)?;

    // private, X(external) if awaited, else external, private ->
    // X(interface), X(private)
    let op = if wait { "AWAIT" } else { "SEQ" };
    let (dom, cod) = update_boundary(&ctrl, &awaited, &read);
    let update = Block::op(mk_op(op), dom, cod)?;

    let delay = zero_delay(&ctrl, &awaited, &read);
    let atom = Atom::new(ctrl, awaited, read, init, update, delay)?;
    Module::partially_observable([atom], |v| *v == private)
}
