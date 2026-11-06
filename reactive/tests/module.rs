//! Tests for module-level structure: partial observability and parallel
//! composition.

mod common;

use common::{
    Atom, Module, Srt, example_counter, example_peterson1, example_tiny1, havoc_init, iface, mk_op,
    skip_update, sorts, srt, var, zero_delay,
};
use dataflow::diagram::Block;
use reactive::ModuleInterface;
use reactive::var::Var;
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

/// The names of the variables, in order
fn names(vars: &ModuleInterface<Srt>) -> Vec<u32> {
    vars.iter().map(Var::name).collect()
}

/// The combinatorial module setting `x` to the next value of `y`
fn copy(x: Var<Srt>, y: Var<Srt>) -> Result<Module, String> {
    let (ctrl, wait, read) = (iface([x]), iface([y]), iface([]));

    // X(y) -> X(x), both as init and as update
    let (dom, cod) = (sorts(&wait), sorts(&ctrl));
    let assign = || Block::op(mk_op("ID"), dom.clone(), cod.clone());
    let delay = zero_delay(&ctrl, &wait, &read);

    let a = Atom::new(ctrl, wait, read, assign()?, assign()?, delay)?;
    Module::observable([a])
}

/// The module stepping `x` and `p` from the latched values of `p` and `e`
///
/// ```text
///     M: x' = p, p' = e   (controls x and p, reads external e)
/// ```
fn step(x: Var<Srt>, p: Var<Srt>, e: Var<Srt>) -> Result<Module, String> {
    let (ctrl, wait, read) = (iface([x, p]), iface([]), iface([p, e]));

    // () -> X(x), X(p)
    let init = Block::op(mk_op("INIT"), [], sorts(&ctrl))?;

    // p, e -> X(x), X(p)
    let update = Block::op(mk_op("STEP"), sorts(&read), sorts(&ctrl))?;

    let delay = zero_delay(&ctrl, &wait, &read);
    let a = Atom::new(ctrl, wait, read, init, update, delay)?;
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
fn cannot_instantiate_external_unobservable_variable() {
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
    let [x, y, p] = [0, 1, 2].map(|n| var(n, "Tny"));
    let m = example_tiny1(x, y, p, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_instantiate_example_tiny1_2301() {
    let [x, y, p] = [0, 1, 2].map(|n| var(n, "Tny"));
    let m = example_tiny1(y, x, p, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_compose_example_tiny1() {
    let (x, y) = (var(0, "x"), var(1, "y"));
    let [p1, p2] = [2, 3].map(|n| var(n, "Tny"));
    let m1 = example_tiny1(x, y, p1, false).unwrap();
    let m2 = example_tiny1(y, x, p2, false).unwrap();

    let m3 = Module::compose([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn cannot_compose_example_tiny1_with_cyclic_await() {
    let [x, y, p1, p2] = [0, 1, 2, 3].map(|n| var(n, "Tny"));
    let m1 = example_tiny1(x, y, p1, true).unwrap();
    let m2 = example_tiny1(y, x, p2, true).unwrap();

    let m3 = Module::compose([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn can_compose_example_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let [x, y, p1, p2] = [0, 1, 2, 3].map(|n| var(n, "Tny"));
    let m1 = example_tiny1(x, y, p1, true).unwrap();
    let m2 = example_tiny1(y, x, p2, false).unwrap();

    let m3 = Module::compose([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn can_compose_three_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let [x, y, z, p1, p2, p3] = [0, 1, 2, 3, 4, 5].map(|n| var(n, "Tny"));
    let m1 = example_tiny1(x, y, p1, true).unwrap();
    let m2 = example_tiny1(y, x, p2, false).unwrap();
    let m3 = example_tiny1(y, z, p3, false).unwrap();

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
    let [x, y, z] = [0, 1, 2].map(|n| var(n, "real"));

    let m1 = copy(x, y).unwrap();
    let m2 = copy(z, x).unwrap();

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
    let [x, y, z] = [0, 1, 2].map(|n| var(n, "real"));

    let m1 = copy(x, y).unwrap();
    let m2 = copy(z, x).unwrap();

    // hiding the shared variable privatises it in the composite
    let m = Module::compose_hiding([m1.clone(), m2.clone()], |v| *v == x).unwrap();
    assert_eq!(names(m.prvt()), vec![x.name()]);
    assert_eq!(names(m.intf()), vec![z.name()]);
    assert_eq!(names(m.extl()), vec![y.name()]);
    // the hidden variable leaves the observables but stays controlled
    assert_eq!(names(m.obs()), vec![y.name(), z.name()]);
    assert_eq!(names(m.ctrl()), vec![x.name(), z.name()]);

    // the pure composition is the special case that hides nothing
    let m = Module::compose([m1.clone(), m2.clone()]).unwrap();
    assert_eq!(m.prvt().len(), 0);
    assert_eq!(names(m.intf()), vec![x.name(), z.name()]);
    assert_eq!(names(m.extl()), vec![y.name()]);

    // an external variable cannot be hidden
    assert!(Module::compose_hiding([m1, m2], |v| *v == y).is_err());
}

/// Chains a fully observable module with the unary hiding operator: `hiding`
/// is the intentional step that makes variables private.
///
/// ```text
///     M: x' = p, p' = e   (controls x and p, reads external e)
/// ```
#[test]
fn constructor_then_hiding() {
    let [x, p, e] = [0, 1, 2].map(|n| var(n, "real"));

    let m = step(x, p, e).unwrap();
    assert_eq!(names(m.intf()), vec![x.name(), p.name()]);
    assert_eq!(m.prvt().len(), 0);

    // hiding privatises the controlled variable and keeps the rest
    let m = m.clone().hide(|v| *v == p).unwrap();
    assert_eq!(names(m.prvt()), vec![p.name()]);
    assert_eq!(names(m.intf()), vec![x.name()]);
    assert_eq!(names(m.extl()), vec![e.name()]);

    // hiding nothing is the identity on the interface
    let m = m.hide(|_| false).unwrap();
    assert_eq!(names(m.prvt()), vec![p.name()]);
    assert_eq!(names(m.intf()), vec![x.name()]);

    // an external variable cannot be hidden
    let m = step(x, p, e).and_then(|m| m.hide(|v| *v == e));
    assert!(m.is_err());
}

#[test]
fn compose_seq_2() {
    let [x, y, z, y0, z0, inv] = [0, 1, 2, 3, 4, 5].map(|n| var(n, "real"));

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
    let (ctrl, wait, read) = (iface([x, y, z]), iface([y0, z0]), iface([x, y, z]));

    let real = srt("real");

    // X(y0), X(z0) -> X(x), X(y), X(z)
    let ops = [
        (mk_op("Const(0)"), vec![], real),
        (mk_op("Id"), vec![0], real),
        (mk_op("Id"), vec![1], real),
    ];
    let init = Block::pointwise([real; 2], ops).unwrap();

    // x, y, z, X(y0), X(z0) -> X(x), X(y), X(z): x counts up while below y
    // or z
    let ops = [
        (mk_op("Count"), vec![0, 1, 2], real),
        (mk_op("Id"), vec![1], real),
        (mk_op("Id"), vec![2], real),
    ];
    let update = Block::pointwise([real; 5], ops).unwrap();

    let delay = zero_delay(&ctrl, &wait, &read);
    let a1 = Atom::new(ctrl, wait, read, init, update, delay).unwrap();
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
    let (ctrl, wait, read) = (iface([inv]), iface([x, y, z]), iface([]));

    // X(x), X(y), X(z) -> X(inv), both as init and as update: a
    // combinatorial atom
    let assign = || Block::op(mk_op("Inv"), [real; 3], [real]).unwrap();

    let delay = zero_delay(&ctrl, &wait, &read);
    let a2 = Atom::new(ctrl, wait, read, assign(), assign(), delay).unwrap();
    let m2 = Module::observable([a2]).unwrap();

    let _m = Module::compose([m1, m2]).unwrap();
    println!("{:?}", _m);
}

/// A module may hide all the variables it controls.
#[test]
fn more_controlled_than_external() {
    let (y, z) = (var(1, "B"), var(2, "C"));
    let (ctrl, wait, read) = (iface([y, z]), iface([]), iface([y, z]));

    // () -> X(y), X(z)
    let init = Block::op(mk_op("A"), [], sorts(&ctrl)).unwrap();
    // y, z -> X(y), X(z)
    let update = Block::op(mk_op("A"), sorts(&read), sorts(&ctrl)).unwrap();
    let delay = zero_delay(&ctrl, &wait, &read);

    let a = Atom::new(ctrl, wait, read, init, update, delay);
    assert!(a.is_ok());
    let m = Module::partially_observable([a.unwrap()], |v| *v == y || *v == z);
    assert!(m.unwrap().obs().is_empty());
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
/// name). Exact comparison guards the inference of the module interfaces
/// against future regressions — a leaked, dropped, or misclassified
/// variable fails here.
#[test]
fn constructor_interfaces_are_exact() {
    let [x, p, e] = [0, 1, 2].map(|n| var(n, "real"));
    let m = step(x, p, e).unwrap();

    assert_eq!(names(m.extl()), vec![e.name()]);
    assert_eq!(names(m.intf()), vec![x.name(), p.name()]);
    assert_eq!(names(m.prvt()), Vec::<u32>::new());
    assert_eq!(names(m.obs()), vec![x.name(), p.name(), e.name()]);
    assert_eq!(names(m.ctrl()), vec![x.name(), p.name()]);
}

/// Hiding must move the hidden variable exactly: `p` leaves `intf` and `obs`
/// and enters `prvt`, while staying in `ctrl`; every other set is untouched.
#[test]
fn hiding_interfaces_are_exact() {
    let [x, p, e] = [0, 1, 2].map(|n| var(n, "real"));
    let m = step(x, p, e).unwrap().hide(|v| *v == p).unwrap();

    assert_eq!(names(m.extl()), vec![e.name()]);
    assert_eq!(names(m.intf()), vec![x.name()]);
    assert_eq!(names(m.prvt()), vec![p.name()]);
    assert_eq!(names(m.obs()), vec![x.name(), e.name()]);
    assert_eq!(names(m.ctrl()), vec![x.name(), p.name()]);
}

#[test]
#[allow(non_snake_case)]
fn heterogeneous_composition() {
    let (x, y, z) = (var(0, "A"), var(1, "B"), var(2, "C"));

    // x' = x, from an arbitrary initial value
    let (ctrl, wait, read) = (iface([x]), iface([]), iface([x]));
    let init = havoc_init::<SeqOps, _>(&ctrl, &wait);
    let jump = skip_update::<SeqOps, _>(&ctrl, &wait, &read);
    let flow = zero_delay::<DifOps, _>(&ctrl, &wait, &read);
    let P = reactive::Atom::new(ctrl, wait, read, init, jump, flow).unwrap();
    let P = reactive::Module::observable([P]).unwrap();

    // dy = 0, from an arbitrary initial value
    let (ctrl, wait, read) = (iface([y]), iface([]), iface([y]));
    let init = havoc_init::<SeqOps, _>(&ctrl, &wait);
    let jump = skip_update::<SeqOps, _>(&ctrl, &wait, &read);
    let flow = zero_delay::<DifOps, _>(&ctrl, &wait, &read);
    let Q = reactive::Atom::new(ctrl, wait, read, init, jump, flow).unwrap();
    let Q = reactive::Module::observable([Q]).unwrap();

    // z = x + y, at every step
    let (ctrl, wait, read) = (iface([z]), iface([x, y]), iface([]));
    let (dom, cod) = (sorts(&wait), sorts(&ctrl));
    let comb = || Block::op(SeqOps("+"), dom.clone(), cod.clone()).unwrap();
    let flow = zero_delay::<DifOps, _>(&ctrl, &wait, &read);
    let R = reactive::Atom::new(ctrl, wait, read, comb(), comb(), flow).unwrap();
    let R = reactive::Module::observable([R]).unwrap();

    let S = reactive::Module::compose([P, Q, R]);
    assert!(S.is_ok());
}

/// The atom controlling the variable `v` with a constant
fn constant(v: Var<Srt>) -> Atom {
    let (ctrl, wait, read) = (iface([v]), iface([]), iface([]));

    // () -> X(v)
    let cod = sorts(&ctrl);
    let assign = || Block::op(mk_op("CONST"), [], cod.clone()).unwrap();
    let delay = zero_delay(&ctrl, &wait, &read);
    Atom::new(ctrl, wait, read, assign(), assign(), delay).unwrap()
}

/// The atom controlling the variable `w` with the latched value of the
/// variable `v`
fn follow(w: Var<Srt>, v: Var<Srt>) -> Atom {
    let (ctrl, wait, read) = (iface([w]), iface([]), iface([v]));

    let init = havoc_init(&ctrl, &wait);
    // v -> X(w)
    let update = Block::op(mk_op("ID"), sorts(&read), sorts(&ctrl)).unwrap();
    let delay = zero_delay(&ctrl, &wait, &read);
    Atom::new(ctrl, wait, read, init, update, delay).unwrap()
}

/// Atoms connect by the names of their variables: an atom reading the name
/// another atom controls reads that variable.
#[test]
fn atoms_connect_by_name() {
    let v = var(7, "real");
    let w = var(8, "real");

    let m = Module::observable([constant(v), follow(w, v)]).unwrap();
    assert!(m.is_closed());
    assert!(m.ctrl().contains(&v));
    assert_eq!(m.obs().len(), 2);
}

/// Variables of the same name are one variable, and must agree on the sort,
/// both within a module and across composed modules.
#[test]
fn coupled_variables_agree_on_sorts() {
    let (v_real, v_bool) = (var(7, "real"), var(7, "bool"));
    let w = var(8, "real");

    let (a, b) = (constant(v_real), follow(w, v_bool));
    assert!(Module::observable([a.clone(), b.clone()]).is_err());

    let ma = Module::observable([a]).unwrap();
    let mb = Module::observable([b]).unwrap();
    assert!(Module::compose([ma, mb]).is_err());
}
