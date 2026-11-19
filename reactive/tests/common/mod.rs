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

/// An uninterpreted sort, named by a string: each its own tangent sort.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sort(pub &'static str);

/// The sorts most tests use.
pub const REAL: Sort = Sort("real");
pub const BOOL: Sort = Sort("bool");

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

/// An uninterpreted operation: its name, and the sorts of the wires it
/// writes and reads, in order, which make its boundary.
#[derive(Clone, Debug)]
pub struct Ops {
    name: &'static str,
    write: Vec<Sort>,
    read: Vec<Sort>,
}

/// The uninterpreted operation `name` that writes wires of the sorts
/// `write` and reads wires of the sorts `read`, in order, as a term of it
/// lists them: `term!(mk_op(name, write, read), [..], [..])`.
pub fn mk_op<const W: usize, const R: usize>(
    name: &'static str,
    write: [Sort; W],
    read: [Sort; R],
) -> Ops {
    Ops {
        name,
        write: write.into(),
        read: read.into(),
    }
}

impl Boundary<Sort> for Ops {
    fn dom(&self) -> Poly<Sort> {
        self.read.iter().copied().collect::<Mono<Sort>>().into()
    }

    fn cod(&self) -> Poly<Sort> {
        self.write.iter().copied().collect::<Mono<Sort>>().into()
    }
}

impl Combinatorial<Sort> for Ops {
    fn havoc(range: &Sort) -> Self {
        mk_op("HAVOC", [*range], [])
    }
}

impl Sequential<Sort> for Ops {
    fn skip(range: &Sort) -> Self {
        mk_op("SKIP", [*range], [*range])
    }
}

impl Differential<Sort> for Ops {
    fn zero(range: &Sort) -> Self {
        mk_op("ZERO", [*range], [])
    }
}

impl Display for Ops {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub type Module = reactive::Module<Ops, Ops, Ops, Sort>;
pub type Atom = reactive::Atom<Ops, Ops, Ops, Sort>;

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
pub fn example_counter() -> Result<(Module, HashMap<Var<Sort>, String>), String> {
    let x = Var::new(REAL);
    let y = Var::new(REAL);
    let z = Var::new(REAL);
    let y0 = Var::new(REAL);
    let z0 = Var::new(REAL);

    let mut init: Vec<Term<Ops, Sort>> = Vec::new();

    let tmp1 = Wire::new(REAL);
    init.push(term!(mk_op("ZERO", [REAL], []), [tmp1])?);

    let tmp2 = Wire::new(REAL);
    let tmp3 = Wire::new(REAL);
    init.push(term!(mk_op("ID", [REAL], [REAL]), [X(x)], [tmp1])?);
    init.push(term!(mk_op("ABS", [REAL], [REAL]), [tmp2], [X(y0)])?);
    init.push(term!(mk_op("ID", [REAL], [REAL]), [X(y)], [tmp2])?);
    init.push(term!(mk_op("ABS", [REAL], [REAL]), [tmp3], [X(z0)])?);
    init.push(term!(mk_op("ID", [REAL], [REAL]), [X(z)], [tmp3])?);

    let mut update: Vec<Term<Ops, Sort>> = Vec::new();

    let tmp4 = Wire::new(BOOL);
    let tmp5 = Wire::new(BOOL);
    let tmp6 = Wire::new(BOOL);
    update.push(term!(mk_op("ZERO", [REAL], []), [tmp1])?);
    update.push(term!(mk_op("LEQ", [BOOL], [REAL, REAL]), [tmp4], [x, y])?);
    update.push(term!(mk_op("LEQ", [BOOL], [REAL, REAL]), [tmp5], [x, z])?);
    update.push(term!(
        mk_op("OR", [BOOL], [BOOL, BOOL]),
        [tmp6],
        [tmp4, tmp5]
    )?);

    let tmp7 = Wire::new(REAL);
    let tmp8 = Wire::new(REAL);
    update.push(term!(mk_op("ONE", [REAL], []), [tmp7])?);
    update.push(term!(
        mk_op("ADD", [REAL], [REAL, REAL]),
        [tmp8],
        [*x, tmp7]
    )?);

    let ite = mk_op("ITE", [REAL], [BOOL, REAL, REAL]);
    update.push(term!(ite, [X(x)], [tmp6, tmp8, tmp1])?);
    update.push(term!(mk_op("ID", [REAL], [REAL]), [X(y)], [y0])?);
    update.push(term!(mk_op("ID", [REAL], [REAL]), [X(z)], [z0])?);

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
    let x1 = Var::new(BOOL);
    let pc2 = Var::new(stype);
    let x2 = Var::new(BOOL);

    // the program counter's values, and its comparison
    let pc = |name| mk_op(name, [stype], []);
    let eq = || mk_op("EQ", [BOOL], [stype, stype]);
    let tt = || mk_op("CONST(true)", [BOOL], []);

    let mut init: Vec<Term<Ops, Sort>> = Vec::new();
    init.push(term!(pc("CONST(outCS)"), [X(pc1)]).unwrap());
    init.push(term!(tt(), [X(x1)]).unwrap());

    let mut update: Vec<Term<Ops, Sort>> = Vec::new();
    let out_cs = Wire::new(stype);
    let cond1 = Wire::new(BOOL);
    update.push(term!(pc("CONST(outCS)"), [out_cs]).unwrap());
    update.push(term!(eq(), [cond1], [out_cs, *pc1]).unwrap());

    let req_cs = Wire::new(stype);
    let cond2 = Wire::new(BOOL);
    update.push(term!(pc("CONST(reqCS)"), [req_cs]).unwrap());
    let tmp11 = Wire::new(BOOL);
    update.push(term!(eq(), [tmp11], [req_cs, *pc1]).unwrap());

    let tmp12 = Wire::new(BOOL);
    let tmp13 = Wire::new(BOOL);
    let tmp14 = Wire::new(BOOL);
    update.push(term!(eq(), [tmp12], [out_cs, *pc2]).unwrap());
    update.push(term!(mk_op("NEQ", [BOOL], [BOOL, BOOL]), [tmp13], [x1, x2]).unwrap());
    update.push(term!(mk_op("OR", [BOOL], [BOOL, BOOL]), [tmp14], [tmp12, tmp13]).unwrap());
    update.push(term!(mk_op("AND", [BOOL], [BOOL, BOOL]), [cond2], [tmp14, tmp11]).unwrap());

    let in_cs = Wire::new(stype);
    let cond3 = Wire::new(BOOL);
    update.push(term!(pc("CONST(inCS)"), [in_cs]).unwrap());
    update.push(term!(eq(), [cond3], [in_cs, *pc1]).unwrap());

    let const_true = Wire::new(BOOL);
    update.push(term!(tt(), [const_true]).unwrap());

    // four cases, each a guard and the values of the program counter and of
    // the flag it selects
    let case = mk_op(
        "CASE",
        [stype, BOOL],
        [
            BOOL, stype, BOOL, BOOL, stype, BOOL, BOOL, stype, BOOL, BOOL, stype, BOOL,
        ],
    );
    update.push(
        term!(
            case,
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
    let tny = Sort("Tny");
    let (extl, intf) = (*external.dtype(), *interface.dtype());
    let private = Var::new(tny);
    let temp = Wire::new(tny);

    let cons = Term::constant(mk_op("CONST", [tny], []), [temp]).unwrap();

    let update = if wait {
        Term::function(
            mk_op("AWAIT", [intf, tny], [extl, tny, tny]),
            [X(interface), X(private)],
            [X(external), *private, temp],
        )
        .unwrap()
    } else {
        Term::function(
            mk_op("SEQ", [intf, tny], [extl, tny, tny]),
            [X(interface), X(private)],
            [*external, *private, temp],
        )
        .unwrap()
    };

    let init = Term::constant(mk_op("INIT", [intf, tny], []), [X(interface), X(private)]).unwrap();

    let vars = [external, interface, private];
    let _obs = [external, interface];
    let prvt = [private];

    let atom = Atom::sequential(&vars, [init], [cons, update])?;
    Module::partially_observable([atom], |v| prvt.contains(v))
}
