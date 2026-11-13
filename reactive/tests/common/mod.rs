//! Helpers shared by the `atom` and `module` integration tests: the
//! uninterpreted `Ops` theory and the example modules built over it.
#![allow(dead_code)]

use base::boundary::{Boundary, Mono, Poly};
use base::tangent::Tangent;
use dataflow::diagram::Term;
use dataflow::diagram::Wire;
use reactive::var::{Var, X};
use std::collections::HashMap;
use std::fmt;
use std::fmt::{Display, Formatter};
use theory::{Combinatorial, Differential, Sequential};

/// An uninterpreted sort: its name and how many times it is a tangent (`0` for
/// values), e.g., `Srt("real", 1)` is the sort of the derivatives of `real`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Srt(pub &'static str, pub u8);

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

/// An uninterpreted operation: its name and its boundary, the sorts of the wires it
/// reads (`dom`) and writes (`cod`)
#[derive(Clone, Debug)]
pub struct Ops {
    pub name: &'static str,
    pub dom: Vec<Srt>,
    pub cod: Vec<Srt>,
}

impl Ops {
    /// The operation `name` with the domain `dom` and the codomain `cod`
    pub fn new<D, C>(name: &'static str, dom: D, cod: C) -> Self
    where
        D: IntoIterator<Item = Srt>,
        C: IntoIterator<Item = Srt>,
    {
        let (dom, cod) = (dom.into_iter().collect(), cod.into_iter().collect());
        Ops { name, dom, cod }
    }

    /// The operation `name` that writes the wires `write` and reads the wires `read`:
    /// the uninterpreted operations apply to wires of any sorts
    pub fn at(name: &'static str, write: &[Wire<Srt>], read: &[Wire<Srt>]) -> Self {
        let sorts = |wires: &[Wire<Srt>]| wires.iter().map(|w| *w.dtype()).collect::<Vec<_>>();
        Ops::new(name, sorts(read), sorts(write))
    }
}

impl Boundary<Srt> for Ops {
    fn dom(&self) -> Poly<Srt> {
        self.dom.iter().copied().collect::<Mono<Srt>>().into()
    }

    fn cod(&self) -> Poly<Srt> {
        self.cod.iter().copied().collect::<Mono<Srt>>().into()
    }
}

impl Combinatorial<Srt> for Ops {
    fn havoc(range: &Srt) -> Self {
        Ops::new("HAVOC", [], [*range])
    }
}

impl Sequential<Srt> for Ops {
    fn skip(range: &Srt) -> Self {
        Ops::new("SKIP", [*range], [*range])
    }
}

impl Differential<Srt> for Ops {
    fn zero(range: &Srt) -> Self {
        Ops::new("ZERO", [], [*range])
    }
}

impl Display for Ops {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub type Module = reactive::Module<Ops, Ops, Ops, Srt>;
pub type Atom = reactive::Atom<Ops, Ops, Ops, Srt>;

/// The term of the uninterpreted operation `name` that writes `write` and reads `read`
pub fn function<D, U, W, R>(name: &'static str, write: W, read: R) -> Result<Term<Ops, Srt>, String>
where
    D: Into<Wire<Srt>>,
    U: Into<Wire<Srt>>,
    W: IntoIterator<Item = D>,
    R: IntoIterator<Item = U>,
{
    let write: Vec<Wire<Srt>> = write.into_iter().map(Into::into).collect();
    let read: Vec<Wire<Srt>> = read.into_iter().map(Into::into).collect();
    Term::function(Ops::at(name, &write, &read), write, read)
}

/// The term of the uninterpreted operation `name` that writes `write` and reads nothing
pub fn constant<D, W>(name: &'static str, write: W) -> Result<Term<Ops, Srt>, String>
where
    D: Into<Wire<Srt>>,
    W: IntoIterator<Item = D>,
{
    let write: Vec<Wire<Srt>> = write.into_iter().map(Into::into).collect();
    Term::constant(Ops::at(name, &write, &[]), write)
}

/// The term of the uninterpreted operation named `$name`, at the sorts of its wires
#[macro_export]
macro_rules! term {
    ($name:expr, $write:expr $(,)?) => {
        $crate::common::constant($name, $write)
    };

    ($name:expr, $write:expr, $read:expr $(,)?) => {
        $crate::common::function($name, $write, $read)
    };
}

#[allow(clippy::vec_init_then_push)]
pub fn example_counter() -> Result<(Module, HashMap<Var<Srt>, String>), String> {
    let x = Var::new(srt("real"));
    let y = Var::new(srt("real"));
    let z = Var::new(srt("real"));
    let y0 = Var::new(srt("real"));
    let z0 = Var::new(srt("real"));

    let mut init: Vec<Term<Ops, Srt>> = Vec::new();

    let tmp1 = Wire::new(srt("real"));
    init.push(term!("ZERO", [tmp1])?);

    let tmp2 = Wire::new(srt("bool"));
    let tmp3 = Wire::new(srt("bool"));
    init.push(term!("ID", [X(x)], [tmp1])?);
    init.push(term!("ABS", [tmp2], [X(y0)])?);
    init.push(term!("ID", [X(y)], [tmp2])?);
    init.push(term!("ABS", [tmp3], [X(z0)])?);
    init.push(term!("ID", [X(z)], [tmp3])?);

    let mut update: Vec<Term<Ops, Srt>> = Vec::new();

    let tmp4 = Wire::new(srt("bool"));
    let tmp5 = Wire::new(srt("bool"));
    let tmp6 = Wire::new(srt("bool"));
    update.push(term!("ZERO", [tmp1])?);
    update.push(term!("LEQ", [tmp4], [x, y])?);
    update.push(term!("LEQ", [tmp5], [x, z])?);
    update.push(term!("OR", [tmp6], [tmp4, tmp5])?);

    let tmp7 = Wire::new(srt("real"));
    let tmp8 = Wire::new(srt("real"));
    update.push(term!("ONE", [tmp7])?);
    update.push(term!("ADD", [tmp8], [*x, tmp7])?);

    update.push(term!("ITE", [X(x)], [tmp6, tmp8, tmp1])?);
    update.push(term!("ID", [X(y)], [y0])?);
    update.push(term!("ID", [X(z)], [z0])?);

    let obs = [x, y, z, y0, z0];

    let module = Module::sequential(&obs, init, update)?;
    // every module variable needs a name: the display indexes them all
    let varnames = [(x, "x"), (y, "y"), (z, "z"), (y0, "y0"), (z0, "z0")]
        .into_iter()
        .map(|(var, name)| (var, name.to_string()))
        .collect();

    Ok((module, varnames))
}

#[allow(clippy::vec_init_then_push)]
pub fn example_peterson1() -> Result<Module, String> {
    let stype = "{outCS, reqCS, inCS}";
    let pc1 = Var::new(srt(stype));
    let x1 = Var::new(srt("bool"));
    let pc2 = Var::new(srt(stype));
    let x2 = Var::new(srt("bool"));

    let mut init: Vec<Term<Ops, Srt>> = Vec::new();
    init.push(term!("CONST(outCS)", [X(pc1)]).unwrap());
    init.push(term!("CONST(true)", [X(x1)]).unwrap());

    let mut update: Vec<Term<Ops, Srt>> = Vec::new();
    let out_cs = Wire::new(srt(stype));
    let cond1 = Wire::new(srt("bool"));
    update.push(term!("CONST(outCS)", [out_cs]).unwrap());
    update.push(term!("EQ", [cond1], [out_cs, *pc1]).unwrap());

    let req_cs = Wire::new(srt(stype));
    let cond2 = Wire::new(srt("bool"));
    update.push(term!("CONST(reqCS)", [req_cs]).unwrap());
    let tmp11 = Wire::new(srt("bool"));
    update.push(term!("EQ", [tmp11], [req_cs, *pc1]).unwrap());

    let tmp12 = Wire::new(srt("bool"));
    let tmp13 = Wire::new(srt("bool"));
    let tmp14 = Wire::new(srt("bool"));
    update.push(term!("EQ", [tmp12], [out_cs, *pc2]).unwrap());
    update.push(term!("NEQ", [tmp13], [x1, x2]).unwrap());
    update.push(term!("OR", [tmp14], [tmp12, tmp13]).unwrap());
    update.push(term!("AND", [cond2], [tmp14, tmp11]).unwrap());

    let in_cs = Wire::new(srt(stype));
    let cond3 = Wire::new(srt("bool"));
    update.push(term!("CONST(inCS)", [in_cs]).unwrap());
    update.push(term!("EQ", [cond3], [in_cs, *pc1]).unwrap());

    let const_true = Wire::new(srt("bool"));
    update.push(term!("CONST(true)", [const_true]).unwrap());

    update.push(
        term!(
            "CASE",
            [X(pc1), X(x1)],
            [
                cond1, req_cs, *x2, cond2, in_cs, *x1, cond3, out_cs, *x1, const_true, *pc1, *x1,
            ]
        )
        .unwrap(),
    );

    let obs = [pc1, x1, pc2, x2];
    Module::sequential(&obs, init, update)
}

pub fn example_tiny1(
    external: Var<Srt>,
    interface: Var<Srt>,
    wait: bool,
) -> Result<Module, String> {
    let private = Var::new(srt("Tny"));
    let temp = Wire::new(srt("Tny"));

    let cons = term!("CONST", [temp]).unwrap();

    let update = if wait {
        term!(
            "AWAIT",
            [X(interface), X(private)],
            [X(external), *private, temp],
        )
        .unwrap()
    } else {
        term!(
            "SEQ",
            [X(interface), X(private)],
            [*external, *private, temp],
        )
        .unwrap()
    };

    let init = term!("INIT", [X(interface), X(private)]).unwrap();

    let vars = [external, interface, private];
    let _obs = [external, interface];
    let prvt = [private];

    let atom = Atom::sequential(&vars, [init], [cons, update])?;
    Module::partially_observable([atom], |v| prvt.contains(v))
}
