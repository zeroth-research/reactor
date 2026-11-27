//! Helpers shared by the `atom` and `module` integration tests: the
//! uninterpreted `Ops` theory and the example modules built over it.
#![allow(dead_code)]

use base::boundary::{Boundary, Mono, Poly};
use base::differentiable::Differentiable;
use dataflow::diagram::Term;
use dataflow::diagram::Wire;
use reactive::var::{Var, X};
use std::collections::HashMap;
use std::fmt;
use std::fmt::{Display, Formatter};
use theory::{Combinatorial, Differential, Sequential};

/// The sorts of the tests, named by strings: each its own tangent sort.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sort(pub &'static str);

impl Differentiable for Sort {
    type Tangent = Self;

    fn tangent(self) -> Self {
        self
    }
}

impl Display for Sort {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The uninterpreted instructions of the tests: a name, and the sorts it reads
/// and writes, which [`term!`] takes from the wires of its term.
#[derive(Clone, Debug)]
#[allow(unused)]
pub struct Ops {
    pub name: &'static str,
    pub dom: Vec<Sort>,
    pub cod: Vec<Sort>,
}

impl Combinatorial<Sort> for Ops {
    fn havoc(range: &Sort) -> Self {
        Ops::new("HAVOC", vec![], vec![*range])
    }
}

impl Sequential<Sort> for Ops {
    fn skip(range: &Sort) -> Self {
        Ops::new("SKIP", vec![*range], vec![*range])
    }
}

impl Differential<Sort> for Ops {
    fn zero(range: &Sort) -> Self {
        Ops::new("ZERO", vec![], vec![*range])
    }
}

impl Display for Ops {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Ops {
    pub fn new(name: &'static str, dom: Vec<Sort>, cod: Vec<Sort>) -> Self {
        Ops { name, dom, cod }
    }
}

impl Boundary<Sort> for Ops {
    fn dom(&self) -> Poly<Sort> {
        self.dom.iter().copied().collect::<Mono<_>>().into()
    }

    fn cod(&self) -> Poly<Sort> {
        self.cod.iter().copied().collect::<Mono<_>>().into()
    }
}

/// An instruction that takes its boundary from the sorts of the wires it
/// reads, `dom`, and writes, `cod`.
pub trait Typed {
    fn typed(self, dom: Vec<Sort>, cod: Vec<Sort>) -> Self;
}

impl Typed for Ops {
    fn typed(self, dom: Vec<Sort>, cod: Vec<Sort>) -> Self {
        Ops { dom, cod, ..self }
    }
}

fn wires<D: Into<Wire<Sort>>>(wires: impl IntoIterator<Item = D>) -> Vec<Wire<Sort>> {
    wires.into_iter().map(Into::into).collect()
}

fn sorts(wires: &[Wire<Sort>]) -> Vec<Sort> {
    wires.iter().map(|w| *w.dtype()).collect()
}

/// `Term::function` of `itype` typed by the sorts of its wires.
pub fn function<T, D, U>(
    itype: T,
    write: impl IntoIterator<Item = D>,
    read: impl IntoIterator<Item = U>,
) -> Result<Term<T, Sort>, String>
where
    T: Typed + Boundary<Sort>,
    D: Into<Wire<Sort>>,
    U: Into<Wire<Sort>>,
{
    let (write, read) = (wires(write), wires(read));
    Term::function(itype.typed(sorts(&read), sorts(&write)), write, read)
}

/// `Term::constant` of `itype` typed by the sorts of its wires.
pub fn constant<T, D>(itype: T, write: impl IntoIterator<Item = D>) -> Result<Term<T, Sort>, String>
where
    T: Typed + Boundary<Sort>,
    D: Into<Wire<Sort>>,
{
    let write = wires(write);
    Term::constant(itype.typed(vec![], sorts(&write)), write)
}

pub type Module = reactive::Module<Ops, Ops, Ops, Sort>;
pub type Atom = reactive::Atom<Ops, Ops, Ops, Sort>;

/// An instruction of the given name, of the sorts of the wires of the term
/// that [`term!`] builds of it.
pub fn mk_op(name: &'static str) -> Ops {
    Ops::new(name, vec![], vec![])
}

/// The term of an instruction writing `$write`, and reading `$read` when
/// given, typed by the sorts of these wires.
#[macro_export]
macro_rules! term {
    ($itype:expr, $write:expr $(,)?) => {
        $crate::common::constant($itype, $write)
    };

    ($itype:expr, $write:expr, $read:expr $(,)?) => {
        $crate::common::function($itype, $write, $read)
    };
}

#[allow(clippy::vec_init_then_push)]
pub fn example_counter() -> Result<(Module, HashMap<Var<Sort>, String>), String> {
    let x = Var::new(Sort("real"));
    let y = Var::new(Sort("real"));
    let z = Var::new(Sort("real"));
    let y0 = Var::new(Sort("real"));
    let z0 = Var::new(Sort("real"));

    let mut init: Vec<Term<Ops, Sort>> = Vec::new();

    let tmp1 = Wire::new(Sort("real"));
    init.push(term!(mk_op("ZERO"), [tmp1])?);

    let tmp2 = Wire::new(Sort("bool"));
    let tmp3 = Wire::new(Sort("bool"));
    init.push(term!(mk_op("ID"), [X(x)], [tmp1])?);
    init.push(term!(mk_op("ABS"), [tmp2], [X(y0)])?);
    init.push(term!(mk_op("ID"), [X(y)], [tmp2])?);
    init.push(term!(mk_op("ABS"), [tmp3], [X(z0)])?);
    init.push(term!(mk_op("ID"), [X(z)], [tmp3])?);

    let mut update: Vec<Term<Ops, Sort>> = Vec::new();

    let tmp4 = Wire::new(Sort("bool"));
    let tmp5 = Wire::new(Sort("bool"));
    let tmp6 = Wire::new(Sort("bool"));
    update.push(term!(mk_op("ZERO"), [tmp1])?);
    update.push(term!(mk_op("LEQ"), [tmp4], [x, y])?);
    update.push(term!(mk_op("LEQ"), [tmp5], [x, z])?);
    update.push(term!(mk_op("OR"), [tmp6], [tmp4, tmp5])?);

    let tmp7 = Wire::new(Sort("real"));
    let tmp8 = Wire::new(Sort("real"));
    update.push(term!(mk_op("ONE"), [tmp7])?);
    update.push(term!(mk_op("ADD"), [tmp8], [*x, tmp7])?);

    update.push(term!(mk_op("ITE"), [X(x)], [tmp6, tmp8, tmp1])?);
    update.push(term!(mk_op("ID"), [X(y)], [y0])?);
    update.push(term!(mk_op("ID"), [X(z)], [z0])?);

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
    let stype = Sort("{outCS, reqCS, inCS}");
    let pc1 = Var::new(stype);
    let x1 = Var::new(Sort("bool"));
    let pc2 = Var::new(stype);
    let x2 = Var::new(Sort("bool"));

    let mut init: Vec<Term<Ops, Sort>> = Vec::new();
    init.push(term!(mk_op("CONST(outCS)"), [X(pc1)]).unwrap());
    init.push(term!(mk_op("CONST(true)"), [X(x1)]).unwrap());

    let mut update: Vec<Term<Ops, Sort>> = Vec::new();
    let out_cs = Wire::new(stype);
    let cond1 = Wire::new(Sort("bool"));
    update.push(term!(mk_op("CONST(outCS)"), [out_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond1], [out_cs, *pc1]).unwrap());

    let req_cs = Wire::new(stype);
    let cond2 = Wire::new(Sort("bool"));
    update.push(term!(mk_op("CONST(reqCS)"), [req_cs]).unwrap());
    let tmp11 = Wire::new(Sort("bool"));
    update.push(term!(mk_op("EQ"), [tmp11], [req_cs, *pc1]).unwrap());

    let tmp12 = Wire::new(Sort("bool"));
    let tmp13 = Wire::new(Sort("bool"));
    let tmp14 = Wire::new(Sort("bool"));
    update.push(term!(mk_op("EQ"), [tmp12], [out_cs, *pc2]).unwrap());
    update.push(term!(mk_op("NEQ"), [tmp13], [x1, x2]).unwrap());
    update.push(term!(mk_op("OR"), [tmp14], [tmp12, tmp13]).unwrap());
    update.push(term!(mk_op("AND"), [cond2], [tmp14, tmp11]).unwrap());

    let in_cs = Wire::new(stype);
    let cond3 = Wire::new(Sort("bool"));
    update.push(term!(mk_op("CONST(inCS)"), [in_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond3], [in_cs, *pc1]).unwrap());

    let const_true = Wire::new(Sort("bool"));
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

    let obs = [pc1, x1, pc2, x2];
    Module::sequential(&obs, init, update)
}

pub fn example_tiny1(
    external: Var<Sort>,
    interface: Var<Sort>,
    wait: bool,
) -> Result<Module, String> {
    let private = Var::new(Sort("Tny"));
    let temp = Wire::new(Sort("Tny"));

    let cons = term!(mk_op("CONST"), [temp]).unwrap();

    let update = if wait {
        term!(
            mk_op("AWAIT"),
            [X(interface), X(private)],
            [X(external), *private, temp],
        )
        .unwrap()
    } else {
        term!(
            mk_op("SEQ"),
            [X(interface), X(private)],
            [*external, *private, temp],
        )
        .unwrap()
    };

    let init = term!(mk_op("INIT"), [X(interface), X(private)]).unwrap();

    let vars = [external, interface, private];
    let _obs = [external, interface];
    let prvt = [private];

    let atom = Atom::sequential(&vars, [init], [cons, update])?;
    Module::partially_observable([atom], |v| prvt.contains(v))
}
