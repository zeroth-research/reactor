//! Tests for atoms: their construction from interfaces and blocks bound by
//! position, and atomic modules, made of a single atom.

mod common;

use common::{
    Atom, Module, Ops, Srt, discard, dwire, dwire_of, example_counter, example_peterson1,
    havoc_init, iface, mk_op, skip_update, var, wire, wire_of, wires, zero_delay,
};
use dataflow::diagram::{Block, Term};
use reactive::var::Var;
use std::collections::HashSet;

#[test]
fn can_instantiate_and_print_sequential_module() {
    let (module, names) = example_counter().unwrap();
    // the naming function makes totality explicit: unnamed variables fall
    // back to their id instead of panicking
    print!("{}", module.with_varnames(|v| names[v].as_str()));

    let atom = &module.atoms()[0];
    print!("{}", atom.with_varnames(|v| names[v].as_str()));
}

#[test]
fn can_instantiate_example_peterson1() {
    let m = example_peterson1().unwrap();

    assert!(m.is_open());
    assert_eq!(m.extl().len(), 2);
    assert_eq!(m.intf().len(), 2);
}

/// The init and the update write the next values of all the controlled
/// variables: they are the outputs of the blocks, each written exactly once.
#[test]
fn blocks_write_all_ctrl() {
    let [x, y, x0] = [0, 1, 2].map(|n| var(n, "real"));
    let (ctrl, wait, read) = (iface([x, y]), iface([x0]), iface([x, y]));

    // X(x0) -> X(x), X(y)
    let init = || {
        let [nx0, nx, ny] = ["real"; 3].map(wire);
        let terms = [
            term!(mk_op("ID"), [nx], [nx0]).unwrap(),
            term!(mk_op("ID"), [ny], [nx0]).unwrap(),
        ];
        Block::new([nx0], [nx, ny], terms).unwrap()
    };
    let delay = || zero_delay(&ctrl, &wait, &read);

    // x, y, X(x0) -> X(x), X(y)
    let [x, y, nx0, nx, ny] = ["real"; 5].map(wire);
    let terms = [
        term!(mk_op("ID"), [nx], [x]).unwrap(),
        term!(mk_op("ID"), [ny], [y]).unwrap(),
        discard(&nx0),
    ];
    let update = Block::new([x, y, nx0], [nx, ny], terms).unwrap();
    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        update,
        delay(),
    );
    assert!(a.is_ok());

    // an update leaving `y` unwritten
    let partial = [
        term!(mk_op("ID"), [nx], [x]).unwrap(),
        discard(&y),
        discard(&nx0),
    ];
    assert!(Block::new([x, y, nx0], [nx, ny], partial.clone()).is_err());

    // nor is it an update once `y` is left out of its outputs
    let update = Block::new([x, y, nx0], [nx], partial).unwrap();
    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        update,
        delay(),
    );
    assert!(a.is_err());

    // an update writing `y` twice
    let twice = [
        term!(mk_op("ID"), [nx], [x]).unwrap(),
        term!(mk_op("ID"), [ny], [y]).unwrap(),
        term!(mk_op("ID"), [ny], [x]).unwrap(),
        discard(&nx0),
    ];
    assert!(Block::new([x, y, nx0], [nx, ny], twice).is_err());
}

/// A block reads only its inputs, and the wires its terms write before.
#[test]
fn block_with_invalid_read() {
    let p = wire("P");

    // X(x) -> X(y), X(z), reading `p` besides
    let (nx, ny, nz) = (wire("A"), wire("B"), wire("C"));
    let init = Term::function(mk_op("i"), [ny, nz], [p, nx]).unwrap();
    let block = Block::new([nx], [ny, nz], [init]);
    println!("{:?}", block);
    assert!(block.is_err());
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
    use theory::lra::{LRA, Sort};

    // Real scalars: 1x1 matrices.
    let scalar = Sort::real([1, 1]);
    // The input clock, and the two clocks drifting relative to it.
    let [t, x, y] = [0, 1, 2].map(|n| Var::new(n, scalar));
    // The atom controls the clocks, reads them to keep them in the update,
    // and awaits the input clock.
    let (ctrl, wait, read) = (iface([x, y]), iface([t]), iface([x, y]));

    // X(t) -> X(x), X(y): the initial values are left unconstrained.
    let init = {
        let [nt, nx, ny] = [t, x, y].map(|v| wire_of(&v));
        let terms = [
            Term::constant(LRA::AnyReal(1, 1), [nx]).unwrap(),
            Term::constant(LRA::AnyReal(1, 1), [ny]).unwrap(),
            discard(&nt),
        ];
        Block::new([nt], [nx, ny], terms).unwrap()
    };

    // x, y, d(t) -> d(x), d(y): the derivatives, both driven by the input's
    // derivative `d(t)`.
    let delay = {
        let [dt, dx, dy] = [t, x, y].map(|v| dwire_of(&v));
        let [x, y] = [x, y].map(|v| wire_of(&v));
        let terms = [
            // dx = 2 dt, synthesised tensor-free as dt + dt
            Term::function(LRA::TAdd(1, 1, 1), [dx], [dt, dt]).unwrap(),
            // dy = 1 dt
            Term::function(LRA::TId(1, 1, 1), [dy], [dt]).unwrap(),
            discard(&x),
            discard(&y),
        ];
        Block::new([x, y, dt], [dx, dy], terms).unwrap()
    };

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
    use theory::lra::{LRA, Sort};

    let x = Var::new(0, Sort::real([1, 1]));

    // dx = x: already ill-formed at the term level.
    let delay = Term::function(LRA::TId(1, 1, 1), [dwire_of(&x)], [wire_of(&x)]);
    assert!(delay.is_err());
}

/// An atom whose initial state is left unconstrained and whose delay is
/// zero: only the update says anything.
#[test]
fn jump_atom_ok() {
    let [x, y] = [0, 1].map(|n| var(n, "real"));
    let (ctrl, wait, read) = (iface([x]), iface([]), iface([y]));

    // y -> X(x): x' = y
    let update = {
        let (y, nx) = (wire("real"), wire("real"));
        Block::new([y], [nx], [term!(mk_op("ID"), [nx], [y]).unwrap()]).unwrap()
    };
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

/// A block writes its outputs, never its inputs: writing the latched value
/// of a variable instead of its next value is rejected.
#[test]
fn blocks_do_not_write_their_inputs() {
    // x, y -> X(x), writing the latched `x` instead of the next `X(x)`
    let [x, y, nx] = ["real"; 3].map(wire);
    let update = [term!(mk_op("ID"), [x], [y]).unwrap()];
    assert!(Block::new([x, y], [nx], update).is_err());
}

/// An atom setting its variable once at initialisation and keeping it ever
/// after.
#[test]
fn constant_atom_ok() {
    let x = var(0, "real");
    let (ctrl, wait, read) = (iface([x]), iface([]), iface([x]));

    // () -> X(x)
    let init = {
        let nx = wire("real");
        Block::new([], [nx], [term!(mk_op("CONST(0)"), [nx]).unwrap()]).unwrap()
    };
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
    let init = {
        let (nx, nloc) = (wire_of(&x), wire_of(&loc));
        let terms = [
            Term::constant(mk_op("CONST(off)"), [nloc]).unwrap(),
            Term::constant(mk_op("CONST(0)"), [nx]).unwrap(),
        ];
        Block::new([], [nx, nloc], terms).unwrap()
    };

    // x, loc -> X(x), X(loc): guard x >= 1; toggle the location and reset
    // the clock when it fires, hold both otherwise
    let update = {
        let [x, loc, nx, nloc] = [x, loc, x, loc].map(|v| wire_of(&v));
        let guard = wire("bool");
        let terms = [
            Term::function(mk_op("GEQ(1)"), [guard], [x]).unwrap(),
            Term::function(mk_op("TOGGLE_IF"), [nloc], [guard, loc]).unwrap(),
            Term::function(mk_op("RESET_IF"), [nx], [guard, x]).unwrap(),
        ];
        Block::new([x, loc], [nx, nloc], terms).unwrap()
    };

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
    let delay = {
        let (dx, dloc) = (dwire_of(&x), dwire_of(&loc));
        let [x, loc] = [x, loc].map(|v| wire_of(&v));
        let terms = [
            Term::constant(mk_op("ONE"), [dx]).unwrap(),
            Term::constant(mk_op("ZERO"), [dloc]).unwrap(),
            discard(&x),
            discard(&loc),
        ];
        Block::new([x, loc], [dx, dloc], terms).unwrap()
    };

    let a = Atom::new(vars.clone(), iface([]), vars, init, update, delay).unwrap();
    let m = Module::observable([a]).unwrap();

    // A closed module controlling both the clock and the location.
    assert!(m.is_closed());
    assert_eq!(m.intf().len(), 2);
    assert_eq!(m.atoms().len(), 1);
    let atom = &m.atoms()[0];
    assert_eq!(atom.ctrl().len(), 2);

    // The three blocks, with the guard as a temporary wire of the update,
    // and the delay discarding the latched values of both variables, which
    // only the update reads.
    assert_eq!(atom.init().len(), 2);
    assert_eq!(atom.update().len(), 3);
    assert_eq!(atom.delay().len(), 2 + 2);
}

/// Forgetting the zero drift of the discrete location leaves its derivative
/// unwritten, and the flow of the timed automaton is rejected.
#[test]
fn hybrid_rejects_missing_flow() {
    let (x, loc, _, _) = switch();

    // x, loc -> d(x), d(loc): dx = 1, but no dloc = 0
    let (dx, dloc) = (dwire_of(&x), dwire_of(&loc));
    let [x, loc] = [x, loc].map(|v| wire_of(&v));
    let terms = [
        Term::constant(mk_op("ONE"), [dx]).unwrap(),
        discard(&x),
        discard(&loc),
    ];
    assert!(Block::new([x, loc], [dx, dloc], terms).is_err());
}

/// The atom controlling `ctrl` with the operation `op` of the latched value
/// of `read`
fn follow(ctrl: Var<Srt>, read: Var<Srt>, op: &'static str) -> Atom {
    let (ctrl, wait, read) = (iface([ctrl]), iface([]), iface([read]));

    // read -> X(ctrl)
    let update = {
        let (r, n) = (wires(&read), wires(&ctrl));
        let terms = [term!(mk_op(op), [n[0]], [r[0]]).unwrap()];
        Block::new(r, n, terms).unwrap()
    };
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

    // X(e) -> X(x)
    let init = || {
        let (ne, nx) = (wire("Bool"), wire("Int"));
        Block::new([ne], [nx], [term!(mk_op("F"), [nx], [ne]).unwrap()]).unwrap()
    };
    // x, e, X(e) -> X(x)
    let update = || {
        let (x, e, ne, nx) = (wire("Int"), wire("Bool"), wire("Bool"), wire("Int"));
        let terms = [term!(mk_op("G"), [nx], [x, e, ne]).unwrap()];
        Block::new([x, e, ne], [nx], terms).unwrap()
    };
    // x, e, d(e) -> d(x)
    let delay = || {
        let (x, e, de, dx) = (wire("Int"), wire("Bool"), dwire("Bool"), dwire("Int"));
        let zero = term!(mk_op("ZERO"), [dx]).unwrap();
        let terms = [zero, discard(&x), discard(&e), discard(&de)];
        Block::new([x, e, de], [dx], terms).unwrap()
    };

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
    let swapped = {
        let (x, e, ne, nx) = (wire("Int"), wire("Bool"), wire("Bool"), wire("Int"));
        let terms = [term!(mk_op("G"), [nx], [x, e, ne]).unwrap()];
        Block::new([e, x, ne], [nx], terms).unwrap()
    };
    let a = Atom::new(
        ctrl.clone(),
        wait.clone(),
        read.clone(),
        init(),
        swapped,
        delay(),
    );
    assert!(a.is_err());

    // a delay writing the next value of `x` instead of its derivative
    let next = {
        let (x, e, de, nx) = (wire("Int"), wire("Bool"), dwire("Bool"), wire("Int"));
        let constant = term!(mk_op("C"), [nx]).unwrap();
        let terms = [constant, discard(&x), discard(&e), discard(&de)];
        Block::new([x, e, de], [nx], terms).unwrap()
    };
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
    let short = {
        let nx = wire("Int");
        Block::new([], [nx], [term!(mk_op("C"), [nx]).unwrap()]).unwrap()
    };
    let a = Atom::new(ctrl, wait, read, short, update(), delay());
    assert!(a.is_err());
}

/// A block takes every input of its boundary, and has to read it: the inputs
/// it has no use for, it discards.
#[test]
fn unread_inputs_are_discarded() {
    let (x, e) = (var(0, "Int"), var(1, "Int"));
    let (ctrl, wait, read) = (iface([x]), iface([e]), iface([]));

    // X(e) -> X(x), ignoring `e`
    let (ne, nx) = (wire("Int"), wire("Int"));
    let constant = term!(mk_op("CONST"), [nx]).unwrap();
    assert!(Block::new([ne], [nx], [constant.clone()]).is_err());
    let init = Block::new([ne], [nx], [constant, discard(&ne)]).unwrap();

    // X(e) -> X(x), awaiting `e`
    let update = {
        let (ne, nx) = (wire("Int"), wire("Int"));
        Block::new([ne], [nx], [term!(mk_op("ID"), [nx], [ne]).unwrap()]).unwrap()
    };
    let delay = zero_delay(&ctrl, &wait, &read);
    let a = Atom::new(ctrl, wait, read, init, update, delay).unwrap();

    let ops: Vec<_> = a.init().iter().map(|t| t.itype().0).collect();
    assert_eq!(ops, ["CONST", "DISCARD"]);

    // the update reads all its inputs, nothing is discarded
    assert_eq!(a.update().len(), 1);
}

/// A variable is not both controlled and awaited by an atom.
#[test]
fn controlled_variables_are_not_awaited() {
    let x = iface([var(0, "Int")]);

    // the identity on X(x), or on d(x)
    let (nxi, nxo) = (wire("Int"), wire("Int"));
    let init = Block::new([nxi], [nxo], [term!(mk_op("ID"), [nxo], [nxi]).unwrap()]).unwrap();
    let (nxi, nxo) = (wire("Int"), wire("Int"));
    let update = Block::new([nxi], [nxo], [term!(mk_op("ID"), [nxo], [nxi]).unwrap()]).unwrap();
    let (dxi, dxo) = (dwire("Int"), dwire("Int"));
    let delay = Block::new([dxi], [dxo], [term!(mk_op("ID"), [dxo], [dxi]).unwrap()]).unwrap();

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
    let init = || {
        let nx = wire("Int");
        Block::new([], [nx], [term!(mk_op("C"), [nx]).unwrap()]).unwrap()
    };
    // x -> X(x)
    let update = |sort| {
        let (x, nx) = (wire(sort), wire("Int"));
        Block::new([x], [nx], [term!(mk_op("C"), [nx]).unwrap(), discard(&x)]).unwrap()
    };
    // x -> d(x)
    let delay = |sort| {
        let (x, dx) = (wire(sort), dwire("Int"));
        Block::new(
            [x],
            [dx],
            [term!(mk_op("ZERO"), [dx]).unwrap(), discard(&x)],
        )
        .unwrap()
    };

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

    // () -> X(x)
    let init = || {
        let nx = wire("Int");
        Block::new([], [nx], [term!(mk_op("C"), [nx]).unwrap()]).unwrap()
    };
    // e, x -> X(x), or x, e -> X(x) in the order of mention
    let update = |mentioned: bool| {
        let (x, e, nx) = (wire("Int"), wire("Bool"), wire("Int"));
        let inputs = if mentioned { [x, e] } else { [e, x] };
        Block::new(inputs, [nx], [term!(mk_op("G"), [nx], [x, e]).unwrap()]).unwrap()
    };
    // e, x -> d(x), or x, e -> d(x) in the order of mention
    let delay = |mentioned: bool| {
        let (x, e, dx) = (wire("Int"), wire("Bool"), dwire("Int"));
        let inputs = if mentioned { [x, e] } else { [e, x] };
        let terms = [
            term!(mk_op("ZERO"), [dx]).unwrap(),
            discard(&x),
            discard(&e),
        ];
        Block::new(inputs, [dx], terms).unwrap()
    };

    let (c, w, r) = (ctrl.clone(), wait.clone(), read.clone());
    assert!(Atom::new(c, w, r, init(), update(false), delay(false)).is_ok());

    // in the order of mention
    let a = Atom::new(ctrl, wait, read, init(), update(true), delay(true));
    assert!(a.is_err());
}
