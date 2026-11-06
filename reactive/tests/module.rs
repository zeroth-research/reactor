//! Tests for module-level structure: partial observability and parallel
//! composition.

mod common;

use base::tangent::Tangent;
use base::wire::Wire;
use common::{
    Atom, Module, Ops, Srt, atom, block, example_counter, example_peterson1, example_tiny1,
    generic_atom, mk_op, name, skip, srt, zero,
};
use dataflow::diagram::Term;
use reactive::AtomInterface;
use reactive::var::{Var, Wire3, X, d};
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

/// The combinatorial module setting `x` to the next value of `y`
fn copy(x: &Wire3<Srt>, y: &Wire3<Srt>) -> Result<Module, String> {
    let assign = || [term!(mk_op("ID"), [X(x)], [X(y)]).unwrap()];
    let a = atom(&[*x], &[*y], &[], assign(), assign(), zero(&[*x]))?;
    Module::observable([a])
}

/// The module stepping `x` and `p` from the latched values of `p` and `e`
///
/// ```text
///     M: x' = p, p' = e   (controls x and p, reads external e)
/// ```
fn step(x: &Wire3<Srt>, p: &Wire3<Srt>, e: &Wire3<Srt>) -> Result<Module, String> {
    let init = Term::constant(mk_op("INIT"), [X(x), X(p)]).unwrap();
    let update = Term::function(mk_op("STEP"), [X(x), X(p)], [*p, *e]).unwrap();
    let ctrl = [*x, *p];
    let a = atom(&ctrl, &[], &[*p, *e], [init], [update], zero(&ctrl))?;
    Module::observable([a])
}

#[test]
fn can_instantiate_partially_observable_module() {
    let (m, _) = example_counter().unwrap();
    let vars = m.obs().clone();
    let mut obs: Vec<Var<Srt>> = Vec::new();
    let mut prvt: Vec<Var<Srt>> = Vec::new();
    for var in vars {
        if var.name() == 0 {
            prvt.push(var);
        } else {
            obs.push(var);
        }
    }

    let m = Module::partially_observable(m.atoms().iter().cloned(), |v| prvt.contains(v));
    //print!("{}", m);
    assert!(m.is_ok());
}

#[test]
fn cannot_instantiate_external_unobservable_wire() {
    let (m, _) = example_counter().unwrap();
    let vars = m.obs().clone();
    // the fourth observable is an external variable: making it private must
    // fail, privates have to be controlled
    let target = vars.iter().nth(3).unwrap().name();
    let mut obs: Vec<Var<Srt>> = Vec::new();
    let mut prvt: Vec<Var<Srt>> = Vec::new();
    for var in vars {
        if var.name() == target {
            prvt.push(var);
        } else {
            obs.push(var);
        }
    }

    let m = Module::partially_observable(m.atoms().iter().cloned(), |v| prvt.contains(v));
    print!("{:?}", m);

    assert!(m.is_err());
}

#[test]
fn can_compose_example_peterson1_with_empty_module() {
    let m1 = example_peterson1().unwrap();
    let m2 = Module::empty();

    let _m3 = Module::compose([m1, m2]).unwrap();
}

#[test]
fn can_instantiate_example_tiny1_0123() {
    let x = Wire3::new(srt("Tny"));
    let y = Wire3::new(srt("Tny"));
    let m = example_tiny1(x, y, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_instantiate_example_tiny1_2301() {
    let x = Wire3::new(srt("Tny"));
    let y = Wire3::new(srt("Tny"));
    let m = example_tiny1(y, x, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_compose_example_tiny1() {
    let x = Wire3::new(srt("x"));
    let y = Wire3::new(srt("y"));
    let m1 = example_tiny1(x, y, false).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::compose([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn cannot_compose_example_tiny1_with_cyclic_await() {
    let x = Wire3::new(srt("Tny"));
    let y = Wire3::new(srt("Tny"));
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, true).unwrap();

    let m3 = Module::compose([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn can_compose_example_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = Wire3::new(srt("Tny"));
    let y = Wire3::new(srt("Tny"));
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::compose([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn can_compose_three_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = Wire3::new(srt("Tny"));
    let y = Wire3::new(srt("Tny"));
    let z = Wire3::new(srt("Tny"));
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();
    let m3 = example_tiny1(y, z, false).unwrap();

    let m4 = Module::compose([m1, m2, m3]);
    assert!(m4.is_ok());
}

#[test]
fn compose_seq() {
    // define two modules:
    //  M1: read external "y" and write it to "x"
    //  M2: read external "x" and write it to "z"
    //
    //  M1 and M2 are compatible (disjoint interface variables and acyclic waiting dependencies),
    //  and we test that they are composable
    let x = Wire3::new(srt("real"));
    let y = Wire3::new(srt("real"));
    let z = Wire3::new(srt("real"));

    let m1 = copy(&x, &y).unwrap();
    let m2 = copy(&z, &x).unwrap();

    Module::compose([m1, m2]).unwrap();
}

/// Composes two coupled modules while hiding their shared variable.
///
/// ```text
///     M1: x' = X(y)   (controls x, reads external y)
///     M2: z' = X(x)   (controls z, reads x)
/// ```
///
/// Hiding `x` at composition keeps the coupling between the components but
/// makes `x` private in the composite: only `z` remains on the interface,
/// with `y` still external. The pure composition `comp` is the special case
/// hiding nothing, and hiding an *external* variable is rejected — privates
/// must be controlled.
#[test]
fn compose_hiding() {
    let x = &Wire3::new(srt("real"));
    let y = &Wire3::new(srt("real"));
    let z = &Wire3::new(srt("real"));

    let m1 = copy(x, y).unwrap();
    let m2 = copy(z, x).unwrap();

    // hiding the shared variable privatises it in the composite
    let m = Module::compose_hiding([m1.clone(), m2.clone()], |v| *v == name(x)).unwrap();
    let ids =
        |i: &reactive::ModuleInterface<_>| i.iter().map(|v| v.name() as usize).collect::<Vec<_>>();
    assert_eq!(ids(m.prvt()), vec![x.id()]);
    assert_eq!(ids(m.intf()), vec![z.id()]);
    assert_eq!(ids(m.extl()), vec![y.id()]);
    // the hidden variable leaves the observables but stays controlled
    assert_eq!(ids(m.obs()), vec![y.id(), z.id()]);
    assert_eq!(ids(m.ctrl()), vec![x.id(), z.id()]);

    // the pure composition is the special case that hides nothing
    let m = Module::compose([m1.clone(), m2.clone()]).unwrap();
    assert_eq!(m.prvt().len(), 0);
    assert_eq!(ids(m.intf()), vec![x.id(), z.id()]);
    assert_eq!(ids(m.extl()), vec![y.id()]);

    // an external variable cannot be hidden
    assert!(Module::compose_hiding([m1, m2], |v| *v == name(y)).is_err());
}

/// Chains an atomic constructor with the unary hiding operator: the
/// constructors are fully observable, and `hiding` is the intentional step
/// that makes variables private.
///
/// ```text
///     M: x' = p, p' = e   (controls x and p, reads external e)
/// ```
#[test]
fn constructor_then_hiding() {
    let x = &Wire3::new(srt("real"));
    let p = &Wire3::new(srt("real"));
    let e = &Wire3::new(srt("real"));

    let m = step(x, p, e).unwrap();

    let ids =
        |i: &reactive::ModuleInterface<_>| i.iter().map(|v| v.name() as usize).collect::<Vec<_>>();
    assert_eq!(ids(m.intf()), vec![x.id(), p.id()]);
    assert_eq!(m.prvt().len(), 0);

    // hiding privatises the controlled variable and keeps the rest
    let m = m.clone().hide(|v| *v == name(p)).unwrap();
    assert_eq!(ids(m.prvt()), vec![p.id()]);
    assert_eq!(ids(m.intf()), vec![x.id()]);
    assert_eq!(ids(m.extl()), vec![e.id()]);

    // hiding nothing is the identity on the interface
    let m = m.hide(|_| false).unwrap();
    assert_eq!(ids(m.prvt()), vec![p.id()]);
    assert_eq!(ids(m.intf()), vec![x.id()]);

    // an external variable cannot be hidden
    let m = step(x, p, e).and_then(|m| m.hide(|v| *v == name(e)));
    assert!(m.is_err());
}

#[test]
fn compose_seq_2() {
    let x = Wire3::new(srt("real"));
    let y = Wire3::new(srt("real"));
    let z = Wire3::new(srt("real"));
    let y0 = Wire3::new(srt("real"));
    let z0 = Wire3::new(srt("real"));
    let inv = Wire3::new(srt("real"));

    // class Module(smt.Module):
    //     def init(self, extl) -> None:
    //         y0, z0 = extl
    //         return Int(0), X(y0), X(z0)  # = x, y, z
    //
    //     def update(self, ctrl, extl) -> None:
    //         x, y, z = ctrl
    //
    //         cond = Or(x < y, x < z)
    //         xn = Ite(cond, x + Int(1), Int(0))
    //
    //         return xn, y, z
    //
    let init: Vec<Term<Ops>> = [
        term!(mk_op("Const(0)"), [X(x)]).unwrap(),
        term!(mk_op("Id"), [X(y)], [X(y0)]).unwrap(),
        term!(mk_op("Id"), [X(z)], [X(z0)]).unwrap(),
    ]
    .to_vec();

    let tmps = [
        Wire::new(srt("real")),
        Wire::new(srt("real")),
        Wire::new(srt("real")),
        Wire::new(srt("real")),
        Wire::new(srt("real")),
        Wire::new(srt("real")),
    ];
    let update: Vec<Term<Ops>> = [
        term!(mk_op("Lt"), [tmps[0]], [x, y]).unwrap(),
        term!(mk_op("Lt"), [tmps[1]], [x, z]).unwrap(),
        term!(mk_op("Or"), [tmps[2]], [tmps[0], tmps[1]]).unwrap(),
        term!(mk_op("Const(0)"), [tmps[3]]).unwrap(),
        term!(mk_op("Const(1)"), [tmps[4]]).unwrap(),
        term!(mk_op("Add"), [tmps[5]], [*x, tmps[4]]).unwrap(),
        term!(mk_op("Ite"), [X(x)], [tmps[2], tmps[5], tmps[3]]).unwrap(),
        term!(mk_op("Id"), [X(y)], [y]).unwrap(),
        term!(mk_op("Id"), [X(z)], [z]).unwrap(),
    ]
    .to_vec();
    let ctrl = &[x, y, z];
    let a1 = atom(ctrl, &[y0, z0], ctrl, init, update, zero(ctrl)).unwrap();
    let m1 = Module::observable([a1]).unwrap();

    //
    // class Inv(smt.Module):
    //     def init(self, extl) -> None:
    //         x, y, z = extl
    //         return Or(X(x) <= X(y), X(x) <= X(z))
    //
    //     def update(self, inv, extl) -> None:
    //         x, y, z = extl
    //         return Or(X(x) <= X(y), X(x) <= X(z))
    let tmps = [
        Wire::new(srt("real")),
        Wire::new(srt("real")),
        Wire::new(srt("real")),
    ];
    let assign: Vec<Term<Ops>> = [
        term!(mk_op("Le"), [tmps[0]], [X(x), X(y)]).unwrap(),
        term!(mk_op("Le"), [tmps[1]], [X(x), X(z)]).unwrap(),
        term!(mk_op("Or"), [tmps[2]], [tmps[0], tmps[1]]).unwrap(),
        term!(mk_op("Id"), [X(inv)], [tmps[2]]).unwrap(),
    ]
    .to_vec();

    // a combinatorial atom: the same assignment initialises and updates
    let a2 = atom(
        &[inv],
        &[x, y, z],
        &[],
        assign.clone(),
        assign,
        zero(&[inv]),
    )
    .unwrap();
    let m2 = Module::observable([a2]).unwrap();

    let _m = Module::compose([m1, m2]).unwrap();
    println!("{:?}", _m);
}

#[test]
fn more_controlled_than_external() {
    let x = &Wire3::new(srt("A"));
    let y = &Wire3::new(srt("B"));
    let z = &Wire3::new(srt("C"));

    let init = Term::constant(mk_op("A"), [X(y), X(z)]).unwrap();
    let update = Term::function(mk_op("A"), [X(y), X(z)], [*y, *z]).unwrap();

    let ctrl = [*y, *z];
    let a = atom(&ctrl, &[], &ctrl, [init], [update], zero(&ctrl));
    assert!(a.is_ok());
    let m = Module::partially_observable([a.unwrap()], |v| *v == name(y) || *v == name(z));
    assert!(m.is_ok());
    // the atom does not know `x`
    assert!(!m.unwrap().obs().contains(&name(x)));
}

#[allow(unused)]
#[derive(Clone, Copy)]
struct SeqOps(&'static str);
#[allow(unused)]
#[derive(Clone, Copy)]
struct DifOps(&'static str);

impl Theory for SeqOps {
    type Sort = Srt;
    const NAME: &'static str = "SeqOps";
    fn check<R, W, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Self::Sort, E>>,
        W: IntoIterator<Item = Result<Self::Sort, E>>,
    {
        Ok(())
    }

    fn discard(_domain: &Self::Sort) -> Self {
        Self("DISCARD")
    }
}

impl Theory for DifOps {
    type Sort = Srt;
    const NAME: &'static str = "DifOps";
    fn check<R, W, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Self::Sort, E>>,
        W: IntoIterator<Item = Result<Self::Sort, E>>,
    {
        Ok(())
    }

    fn discard(_domain: &Self::Sort) -> Self {
        Self("DISCARD")
    }
}

impl Differential for DifOps {
    fn zero(_range: &Self::Sort) -> Self {
        Self("ZERO")
    }
}
impl Sequential for SeqOps {
    fn skip(_range: &Self::Sort) -> Self {
        Self("SKIP")
    }
}

impl Combinatorial for SeqOps {
    fn havoc(_range: &Self::Sort) -> Self {
        Self("HAVOC")
    }
}

impl fmt::Display for DifOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Display for SeqOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The visibility sets of a constructed module, compared against the exact
/// expected variables:
///
/// ```text
///     M: x' = p, p' = e   (controls x and p, reads external e)
/// ```
///
/// The read-only `e` must land in `extl` and not in `intf`; the controlled
/// `x` and `p` in `intf`; and the derived sets must be exactly
/// `obs = extl U intf` and `ctrl = intf U prvt` (interfaces are ordered by
/// variable creation). Exact comparison guards the inference against future
/// regressions — a leaked, dropped, or misclassified variable fails here.
#[test]
fn constructor_interfaces_are_exact() {
    let x = &Wire3::new(srt("real"));
    let p = &Wire3::new(srt("real"));
    let e = &Wire3::new(srt("real"));

    let m = step(x, p, e).unwrap();

    let ids =
        |i: &reactive::ModuleInterface<_>| i.iter().map(|v| v.name() as usize).collect::<Vec<_>>();
    assert_eq!(ids(m.extl()), vec![e.id()]);
    assert_eq!(ids(m.intf()), vec![x.id(), p.id()]);
    assert_eq!(ids(m.prvt()), Vec::<usize>::new());
    assert_eq!(ids(m.obs()), vec![x.id(), p.id(), e.id()]);
    assert_eq!(ids(m.ctrl()), vec![x.id(), p.id()]);
}

/// Hiding must move the hidden variable exactly: `p` leaves `intf` and `obs`
/// and enters `prvt`, while staying in `ctrl`; every other set is untouched.
#[test]
fn hiding_interfaces_are_exact() {
    let x = &Wire3::new(srt("real"));
    let p = &Wire3::new(srt("real"));
    let e = &Wire3::new(srt("real"));

    let m = step(x, p, e).unwrap().hide(|v| *v == name(p)).unwrap();

    let ids =
        |i: &reactive::ModuleInterface<_>| i.iter().map(|v| v.name() as usize).collect::<Vec<_>>();
    assert_eq!(ids(m.extl()), vec![e.id()]);
    assert_eq!(ids(m.intf()), vec![x.id()]);
    assert_eq!(ids(m.prvt()), vec![p.id()]);
    assert_eq!(ids(m.obs()), vec![x.id(), e.id()]);
    assert_eq!(ids(m.ctrl()), vec![x.id(), p.id()]);
}

#[test]
#[allow(non_snake_case)]
fn heterogeneous_composition() {
    let x = Wire3::new(srt("A"));
    let y = Wire3::new(srt("B"));
    let z = Wire3::new(srt("C"));

    let init = Term::constant(SeqOps::havoc(&srt("A")), [X(x)]).unwrap();
    let jump = Term::function(SeqOps::skip(&srt("A")), [X(x)], [x]).unwrap();
    let P = generic_atom(&[x], &[], &[x], [init], [jump], zero::<DifOps>(&[x]));
    let P = reactive::Module::observable([P.unwrap()]).unwrap();

    let init = Term::constant(SeqOps::havoc(&srt("B")), [X(y)]).unwrap();
    let flow = Term::constant(DifOps::zero(&srt("B").tangent()), [d(y)]).unwrap();
    let Q = generic_atom(&[y], &[], &[y], [init], skip::<SeqOps>(&[y]), [flow]);
    let Q = reactive::Module::observable([Q.unwrap()]).unwrap();

    let comb = Term::function(SeqOps("+"), [X(z)], [X(x), X(y)]).unwrap();
    let R = generic_atom(
        &[z],
        &[x, y],
        &[],
        [comb.clone()],
        [comb],
        zero::<DifOps>(&[z]),
    );
    let R = reactive::Module::observable([R.unwrap()]).unwrap();

    let S = reactive::Module::compose([P, Q, R]);
    assert!(S.is_ok());
}

/// The atom controlling the variable `v` with a constant, under the wires
/// of its blocks only
fn constant(v: Var<Srt>) -> Atom {
    let ctrl = AtomInterface::try_from_iter([v]).unwrap();
    let (init, update, der) = (
        Wire::new(*v.dtype()),
        Wire::new(*v.dtype()),
        Wire::new(v.dtype().tangent()),
    );
    let init = block([], [init], [term!(mk_op("CONST"), [init]).unwrap()]).unwrap();
    let update = block([], [update], [term!(mk_op("CONST"), [update]).unwrap()]).unwrap();
    let delay = block([], [der], [term!(mk_op("ZERO"), [der]).unwrap()]).unwrap();
    Atom::new(
        ctrl,
        AtomInterface::empty(),
        AtomInterface::empty(),
        init,
        update,
        delay,
    )
    .unwrap()
}

/// The atom controlling the variable `w` with the latched value of the
/// variable `v`, under the wires of its blocks only
fn follow(w: Var<Srt>, v: Var<Srt>) -> Atom {
    let (ctrl, read) = (
        AtomInterface::try_from_iter([w]).unwrap(),
        AtomInterface::try_from_iter([v]).unwrap(),
    );
    let init = Wire::new(*w.dtype());
    let init = block([], [init], [term!(mk_op("CONST"), [init]).unwrap()]).unwrap();
    let (rd, wt) = (Wire::new(*v.dtype()), Wire::new(*w.dtype()));
    let update = block([rd], [wt], [term!(mk_op("ID"), [wt], [rd]).unwrap()]).unwrap();
    let (rd, der) = (Wire::new(*v.dtype()), Wire::new(w.dtype().tangent()));
    let delay = block([rd], [der], [term!(mk_op("ZERO"), [der]).unwrap()]).unwrap();
    Atom::new(ctrl, AtomInterface::empty(), read, init, update, delay).unwrap()
}

/// Atoms connect by the names of their variables: an atom reading the name
/// another atom controls reads that variable, whatever the wires of their
/// blocks.
#[test]
fn atoms_connect_by_name() {
    let v = Var::new(7, srt("real"));
    let w = Var::new(8, srt("real"));

    let m = Module::observable([constant(v), follow(w, v)]).unwrap();
    assert!(m.is_closed());
    assert!(m.ctrl().contains(&v));
    assert_eq!(m.obs().len(), 2);
}

/// Variables of the same name are one variable, and must agree on the sort,
/// both within a module and across composed modules.
#[test]
fn coupled_variables_agree_on_sorts() {
    let (v_real, v_bool) = (Var::new(7, srt("real")), Var::new(7, srt("bool")));
    let w = Var::new(8, srt("real"));

    let (a, b) = (constant(v_real), follow(w, v_bool));
    assert!(Module::observable([a.clone(), b.clone()]).is_err());

    let ma = Module::observable([a]).unwrap();
    let mb = Module::observable([b]).unwrap();
    assert!(Module::compose([ma, mb]).is_err());
}
