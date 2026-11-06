//! Tests for atoms: their construction from interfaces and blocks bound by
//! position, and atomic modules, made of a single atom.

mod common;

use common::{
    Atom, Module, Ops, Srt, delay_boundary, dsrt, example_counter, example_peterson1, havoc_init,
    iface, init_boundary, mk_op, skip_update, sorts, srt, var, zero_delay,
};
use dataflow::diagram::Block;
use reactive::var::Var;
use std::collections::HashSet;

#[test]
fn can_instantiate_and_print_sequential_module() {
    let (module, names) = example_counter().unwrap();
    // the naming function makes totality explicit: unnamed variables fall
    // back to their id instead of panicking
    print!("{}", module.with_varnames(|v| names[v].as_str()));

    let atom = &module.atoms()[0];
    let shown = atom.with_varnames(|v| names[v].as_str()).to_string();
    print!("{shown}");

    // the inputs and outputs of the blocks print as the views of the
    // variables they are bound to by position
    assert!(shown.contains("X(y) := ID X(y0)\n"));
    assert!(shown.contains("X(x) := COUNT (x, y, z)\n"));
    assert!(shown.contains("X(y) := ID y0\n"));
    assert!(shown.contains("d(x) := ZERO \n"));
    assert!(shown.contains(":= DISCARD X(z0)\n"));
}

#[test]
fn can_instantiate_example_peterson1() {
    let m = example_peterson1().unwrap();

    assert!(m.is_open());
    assert_eq!(m.extl().len(), 2);
    assert_eq!(m.intf().len(), 2);
}

/// The init and the update give the next values of all the controlled
/// variables: a block leaving one out does not fit the atom.
#[test]
fn blocks_write_all_ctrl() {
    let [x, y, x0] = [0, 1, 2].map(|n| var(n, "real"));
    let (ctrl, wait, read) = (iface([x, y]), iface([x0]), iface([x, y]));
    let real = srt("real");

    // X(x0) -> X(x), X(y)
    let init = || {
        let ops = [(mk_op("ID"), vec![0], real), (mk_op("ID"), vec![0], real)];
        Block::pointwise([real], ops).unwrap()
    };
    let delay = || zero_delay(&ctrl, &wait, &read);

    // x, y, X(x0) -> X(x), X(y)
    let ops = [(mk_op("ID"), vec![0], real), (mk_op("ID"), vec![1], real)];
    let update = Block::pointwise([real; 3], ops).unwrap();
    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        update,
        delay(),
    );
    assert!(a.is_ok());

    // x, y, X(x0) -> X(x), leaving `y` out
    let update = Block::pointwise([real; 3], [(mk_op("ID"), vec![0], real)]).unwrap();
    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        update,
        delay(),
    );
    assert!(a.is_err());
}

/// Builds an open differential module encoding the drifting clocks
///
/// ```text
///     dx = 2 dt
///     dy = 1 dt
/// ```
///
/// where `t` is an *input*: the module does not control it, so it is
/// inferred external and the module is open. The clocks `x` and `y` drift at
/// rates 2 and 1 relative to `t`, expressed by delay terms that *await* the
/// input's derivative `d(t)`: `d(x) = d(t) + d(t)` and `d(y) = Id(d(t))`.
/// Since the rates are relative to the input clock rather than constant in
/// absolute time, these are drifting clocks and not a linear ODE.
#[test]
fn differential_drifting_clocks() {
    use base::tangent::Tangent;
    use theory::lra::{LRA, Sort};

    // Real scalars: 1x1 matrices.
    let scalar = Sort::real([1, 1]);
    // The input clock, and the two clocks drifting relative to it.
    let [t, x, y] = [0, 1, 2].map(|n| Var::new(n, scalar));
    // The atom controls the clocks, reads them to keep them in the update,
    // and awaits the input clock.
    let (ctrl, wait, read) = (iface([x, y]), iface([t]), iface([x, y]));

    // X(t) -> X(x), X(y): the initial values are left unconstrained.
    let (dom, cod) = init_boundary(&ctrl, &wait);
    let init = Block::havoc(dom, cod).unwrap();

    // x, y, d(t) -> d(x), d(y): the derivatives, both driven by the input's
    // derivative `d(t)`, the third input.
    let (dom, _) = delay_boundary(&ctrl, &wait, &read);
    let ops = [
        // dx = 2 dt, synthesised tensor-free as dt + dt
        (LRA::TAdd(1, 1, 1), vec![2, 2], scalar.tangent()),
        // dy = 1 dt
        (LRA::TId(1, 1, 1), vec![2], scalar.tangent()),
    ];
    let delay = Block::pointwise(dom, ops).unwrap();

    // x, y, X(t) -> X(x), X(y): the clocks keep their values.
    let update = skip_update(&ctrl, &wait, &read);

    let atom: reactive::Atom<LRA, LRA, LRA> =
        reactive::Atom::new(ctrl, wait, read, init, update, delay)
            .expect("differential atom should be well-formed");
    let module =
        reactive::Module::observable([atom]).expect("differential module should be well-formed");

    let names = |vars: &reactive::ModuleInterface<Sort>| -> HashSet<u32> {
        vars.iter().map(Var::name).collect()
    };

    // An open module: the input clock is external.
    assert!(module.is_open());
    assert_eq!(names(module.extl()), HashSet::from([t.name()]));

    // The clocks are controlled by the module, hence interface variables.
    assert_eq!(names(module.intf()), HashSet::from([x.name(), y.name()]));

    assert_eq!(module.atoms().len(), 1);
    let atom = &module.atoms()[0];

    // The atom controls the clocks and awaits the input.
    assert_eq!(names(atom.ctrl()), HashSet::from([x.name(), y.name()]));
    assert_eq!(names(atom.wait()), HashSet::from([t.name()]));

    // Block sizes: 2 derivative terms, 2 init terms, and 2 skip terms in the
    // update. The blocks discard the inputs they do not read: init and
    // update the next value of `t`, and the delay the latched values of the
    // clocks, which only the update reads.
    assert_eq!(atom.delay().len(), 2 + 2);
    assert_eq!(atom.init().len(), 2 + 1);
    assert_eq!(atom.update().len(), 2 + 1);

    // All three variables are observable.
    assert_eq!(module.obs().len(), 3);
}

/// Proposes a delay encoding
///
/// ```text
///     dx = x
/// ```
///
/// and expects it to be rejected: the delay term drives the derivative
/// `d(x)` (of the tangent sort) directly from the value of `x`, and the
/// operations on derivatives do not read values.
#[test]
fn differential_module_rejects_dx_equals_x() {
    use base::tangent::Tangent;
    use theory::lra::{LRA, Sort};

    let x = Var::new(0, Sort::real([1, 1]));

    // x -> d(x), with dx = x: already ill-formed as a block.
    let ops = [(LRA::TId(1, 1, 1), vec![0], x.dtype().tangent())];
    assert!(Block::pointwise([*x.dtype()], ops).is_err());
}

/// An atom whose initial state is left unconstrained and whose delay is
/// zero: only the update says anything.
#[test]
fn jump_atom_ok() {
    let [x, y] = [0, 1].map(|n| var(n, "real"));
    let (ctrl, wait, read) = (iface([x]), iface([]), iface([y]));

    // y -> X(x): x' = y
    let update = Block::op(mk_op("ID"), sorts(&read), sorts(&ctrl)).unwrap();
    let init = havoc_init(&ctrl, &wait);
    let delay = zero_delay(&ctrl, &wait, &read);

    let a = Atom::new(ctrl, wait, read, init, update, delay).unwrap();
    let m = Module::observable([a]).unwrap();

    assert!(m.is_open());
    assert_eq!(m.atoms().len(), 1);
    let atom = &m.atoms()[0];

    // One HAVOC init and one ZERO delay for the single controlled variable,
    // next to the update. The delay discards the latched value of `y`,
    // which only the update reads.
    assert_eq!(atom.init().len(), 1);
    assert_eq!(atom.update().len(), 1);
    assert_eq!(atom.delay().len(), 1 + 1);
}

/// An atom setting its variable once at initialisation and keeping it ever
/// after.
#[test]
fn constant_atom_ok() {
    let x = var(0, "real");
    let (ctrl, wait, read) = (iface([x]), iface([]), iface([x]));

    // () -> X(x)
    let init = Block::op(mk_op("CONST(0)"), [], sorts(&ctrl)).unwrap();
    let update = skip_update(&ctrl, &wait, &read);
    let delay = zero_delay(&ctrl, &wait, &read);

    let a = Atom::new(ctrl, wait, read, init, update, delay).unwrap();
    let m = Module::observable([a]).unwrap();

    assert!(m.is_closed());
    assert_eq!(m.atoms().len(), 1);
    let atom = &m.atoms()[0];

    // The init, next to one SKIP update and one ZERO delay for the single
    // controlled variable. The delay discards the latched value of `x`,
    // which only the update reads.
    assert_eq!(atom.init().len(), 1);
    assert_eq!(atom.update().len(), 1);
    assert_eq!(atom.delay().len(), 1 + 1);
}

/// The variables of the timed automaton, the clock `x` and the location
/// `loc`, with the init and the update of the switch: both blocks control
/// both variables, the update reads both.
///
/// ```text
///     init:  loc = off, x = 0
///     jump:  when x >= 1, toggle loc and reset x; otherwise hold
/// ```
fn switch() -> (Var<Srt>, Var<Srt>, Block<Ops>, Block<Ops>) {
    let (x, loc) = (var(0, "real"), var(1, "{off, on}"));

    // () -> X(x), X(loc)
    let ops = [
        (mk_op("CONST(0)"), vec![], *x.dtype()),
        (mk_op("CONST(off)"), vec![], *loc.dtype()),
    ];
    let init = Block::pointwise([], ops).unwrap();

    // x, loc -> X(x), X(loc): guard x >= 1; reset the clock and toggle the
    // location when it fires, hold both otherwise
    let ops = [
        (mk_op("RESET_IF(x >= 1)"), vec![0], *x.dtype()),
        (mk_op("TOGGLE_IF(x >= 1)"), vec![0, 1], *loc.dtype()),
    ];
    let update = Block::pointwise([*x.dtype(), *loc.dtype()], ops).unwrap();

    (x, loc, init, update)
}

/// A hybrid module encoding a simple timed automaton: a switch with one
/// clock.
///
/// ```text
///     loc ∈ {off, on},  clock x
///
///     init:  loc = off, x = 0
///     flow:  dx = 1, dloc = 0
///     jump:  when x >= 1, toggle loc and reset x; otherwise hold
/// ```
///
/// The discrete jump drives the next values, the continuous flow drives the
/// derivatives — the clock at rate 1, the location with zero drift, since
/// it only changes at jumps.
#[test]
fn simple_timed_automaton() {
    let (x, loc, init, update) = switch();
    let vars = iface([x, loc]);

    // x, loc -> d(x), d(loc): dx = 1, dloc = 0
    let (dom, _) = delay_boundary(&vars, &iface([]), &vars);
    let ops = [
        (mk_op("ONE"), vec![], dsrt("real")),
        (mk_op("ZERO"), vec![], dsrt("{off, on}")),
    ];
    let delay = Block::pointwise(dom, ops).unwrap();

    let a = Atom::new(vars.clone(), iface([]), vars, init, update, delay).unwrap();
    let m = Module::observable([a]).unwrap();

    // A closed module controlling both the clock and the location.
    assert!(m.is_closed());
    assert_eq!(m.intf().len(), 2);
    assert_eq!(m.atoms().len(), 1);
    let atom = &m.atoms()[0];
    assert_eq!(atom.ctrl().len(), 2);

    // The three blocks, with an operation per controlled variable, and the
    // delay discarding the latched values of both variables, which only the
    // update reads.
    assert_eq!(atom.init().len(), 2);
    assert_eq!(atom.update().len(), 2);
    assert_eq!(atom.delay().len(), 2 + 2);
}

/// Forgetting the zero drift of the discrete location leaves its derivative
/// out of the flow, and the flow does not fit the timed automaton.
#[test]
fn hybrid_rejects_missing_flow() {
    let (x, loc, init, update) = switch();
    let vars = iface([x, loc]);

    // x, loc -> d(x): dx = 1, but no dloc = 0
    let (dom, _) = delay_boundary(&vars, &iface([]), &vars);
    let delay = Block::pointwise(dom, [(mk_op("ONE"), vec![], dsrt("real"))]).unwrap();
    assert!(Atom::new(vars.clone(), iface([]), vars, init, update, delay).is_err());
}

/// The atom controlling `ctrl` with the operation `op` of the latched value
/// of `read`
fn follow(ctrl: Var<Srt>, read: Var<Srt>, op: &'static str) -> Atom {
    let (ctrl, wait, read) = (iface([ctrl]), iface([]), iface([read]));

    // read -> X(ctrl)
    let update = Block::op(mk_op(op), sorts(&read), sorts(&ctrl)).unwrap();
    let init = havoc_init(&ctrl, &wait);
    let delay = zero_delay(&ctrl, &wait, &read);
    Atom::new(ctrl, wait, read, init, update, delay).unwrap()
}

#[test]
fn coupled_atoms_can_construct_module() {
    let [x, y, z] = [0, 1, 2].map(|n| var(n, "real"));

    let a1 = follow(y, x, "u1");
    let a2 = follow(z, y, "u2");

    let m = Module::observable([a1, a2]);
    assert!(m.is_ok());
}

/// The blocks of an atom have the boundaries `X(wait) -> X(ctrl)` (init),
/// `read ++ X(wait) -> X(ctrl)` (update), and `read ++ d(wait) -> d(ctrl)`
/// (delay), each in the order of the interfaces; the atom binds its
/// variables to them by position, and checks their sorts.
#[test]
fn blocks_have_the_boundaries_of_the_interfaces() {
    // x : Int controlled, read (x : Int, e : Bool), wait (e : Bool)
    let (x, e) = (var(0, "Int"), var(1, "Bool"));
    let (ctrl, wait, read) = (iface([x]), iface([e]), iface([x, e]));

    let (int, bool) = (srt("Int"), srt("Bool"));

    // X(e) -> X(x)
    let init = || Block::op(mk_op("F"), [bool], [int]).unwrap();
    // x, e, X(e) -> X(x)
    let update = || Block::op(mk_op("G"), [int, bool, bool], [int]).unwrap();
    // x, e, d(e) -> d(x)
    let delay = || Block::zero([int, bool, dsrt("Bool")], [dsrt("Int")]).unwrap();

    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        update(),
        delay(),
    );
    let a = a.unwrap();
    assert_eq!(a.ctrl(), &ctrl);
    assert_eq!(a.read(), &read);
    assert_eq!(a.wait(), &wait);

    // the inputs of the update out of the order of the interfaces
    let swapped = Block::op(mk_op("G"), [bool, int, bool], [int]).unwrap();
    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        swapped,
        delay(),
    );
    assert!(a.is_err());

    // a delay giving the next value of `x` instead of its derivative
    let next = Block::op(mk_op("C"), [int, bool, dsrt("Bool")], [int]).unwrap();
    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        update(),
        next,
    );
    assert!(a.is_err());

    // an init leaving out the awaited `e`
    let short = Block::op(mk_op("C"), [], [int]).unwrap();
    let a = Atom::new(ctrl, wait, read, short, update(), delay());
    assert!(a.is_err());
}

/// A variable is not both controlled and awaited by an atom.
#[test]
fn controlled_variables_are_not_awaited() {
    let x = iface([var(0, "Int")]);

    // the identity on X(x), or on d(x)
    let (int, dint) = (srt("Int"), dsrt("Int"));
    let init = Block::op(mk_op("ID"), [int], [int]).unwrap();
    let update = Block::op(mk_op("ID"), [int], [int]).unwrap();
    let delay = Block::op(mk_op("ID"), [dint], [dint]).unwrap();

    let a = Atom::new(x.clone(), x, iface([]), init, update, delay);
    assert!(a.is_err());
}

/// The same name is the same variable in all the interfaces of an atom, of
/// one sort.
#[test]
fn interfaces_agree_on_sorts() {
    // `x` is controlled as an Int
    let ctrl = iface([var(0, "Int")]);

    // () -> X(x)
    let init = || Block::op(mk_op("C"), [], [srt("Int")]).unwrap();
    // x -> X(x)
    let update = |sort| Block::op(mk_op("C"), [srt(sort)], [srt("Int")]).unwrap();
    // x -> d(x)
    let delay = |sort| Block::zero([srt(sort)], [dsrt("Int")]).unwrap();

    // read as a Bool
    let read = iface([var(0, "Bool")]);
    let a = Atom::new(
        ctrl.clone(),
        iface([]),
        read,
        init(),
        update("Bool"),
        delay("Bool"),
    );
    assert!(a.is_err());

    // read as an Int, it is fine
    let read = iface([var(0, "Int")]);
    let a = Atom::new(ctrl, iface([]), read, init(), update("Int"), delay("Int"));
    assert!(a.is_ok());
}

/// The interfaces, and so the boundaries of the blocks, follow the order of
/// the names, not the order the variables are mentioned in.
#[test]
fn boundaries_follow_the_order_of_names() {
    // `e` is named before `x`
    let (x, e) = (var(1, "Int"), var(0, "Bool"));
    let (ctrl, wait, read) = (iface([x]), iface([]), iface([x, e]));
    assert_eq!(read.iter().cloned().collect::<Vec<_>>(), [e, x]);

    let (int, bool) = (srt("Int"), srt("Bool"));
    let inputs = |mentioned: bool| if mentioned { [int, bool] } else { [bool, int] };

    // () -> X(x)
    let init = || Block::op(mk_op("C"), [], [int]).unwrap();
    // e, x -> X(x), or x, e -> X(x) in the order of mention
    let update = |mentioned| Block::op(mk_op("G"), inputs(mentioned), [int]).unwrap();
    // e, x -> d(x), or x, e -> d(x) in the order of mention
    let delay = |mentioned| Block::zero(inputs(mentioned), [dsrt("Int")]).unwrap();

    let (c, w, r) = (ctrl.clone(), wait.clone(), read.clone());
    assert!(Atom::new(c, w, r, init(), update(false), delay(false)).is_ok());

    // in the order of mention
    let a = Atom::new(ctrl, wait, read, init(), update(true), delay(true));
    assert!(a.is_err());
}
