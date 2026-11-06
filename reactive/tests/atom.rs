//! Tests for atoms: their construction from interfaces and blocks bound by
//! position, and atomic modules, made of a single atom.

mod common;

use base::wire::Wire;
use common::{
    Atom, Module, Ops, atom, block, example_counter, example_peterson1, generic_atom, havoc, mk_op,
    skip, srt, zero,
};
use dataflow::diagram::{Block, Term};
use reactive::AtomInterface;
use reactive::var::{Var, Wire3, X, d};
use theory::Theory;

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
/// variables: they are the outputs of their blocks.
#[test]
fn blocks_write_all_ctrl() {
    let x = Wire3::new(srt("real"));
    let y = Wire3::new(srt("real"));
    let x0 = Wire3::new(srt("real"));

    let init = || {
        [
            term!(mk_op("ID"), [X(x)], [X(x0)]).unwrap(),
            term!(mk_op("ID"), [X(y)], [X(x0)]).unwrap(),
        ]
    };
    let update = || {
        [
            term!(mk_op("ID"), [X(x)], [x]).unwrap(),
            term!(mk_op("ID"), [X(y)], [y]).unwrap(),
        ]
    };
    let (ctrl, wait, read) = (&[x, y], &[x0], &[x, y]);

    let a = atom(ctrl, wait, read, init(), update(), zero(ctrl));
    assert!(a.is_ok());

    // the update leaves `y` unwritten
    let partial = [term!(mk_op("ID"), [X(x)], [x]).unwrap()];
    assert!(atom(ctrl, wait, read, init(), partial, zero(ctrl)).is_err());

    // the init leaves `y` unwritten
    let partial = [term!(mk_op("ID"), [X(x)], [X(x0)]).unwrap()];
    assert!(atom(ctrl, wait, read, partial, update(), zero(ctrl)).is_err());

    // the delay leaves `y` unwritten
    assert!(atom(ctrl, wait, read, init(), update(), zero(&[x])).is_err());
}

/// A block reads only its inputs, and the wires its terms write.
#[test]
fn atom_with_invalid_read() {
    let x = Wire3::new(srt("A"));
    let y = Wire3::new(srt("B"));
    let z = Wire3::new(srt("C"));
    let p = Wire::new(srt("P"));

    let init = Term::function(mk_op("i"), [X(y), X(z)], [p, X(x)]).unwrap();
    let update = Term::function(mk_op("u"), [X(y), X(z)], [p, X(x), *y, *z]).unwrap();

    let ctrl = &[y, z];
    let a = atom(ctrl, &[x], ctrl, [init], [update], zero(ctrl));
    println!("{:?}", a);
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
    use std::collections::HashSet;
    use theory::lra::{LRA, Sort};

    // Real scalars: 1x1 matrices.
    let scalar = Sort::real([1, 1]);
    // The input clock, and the two clocks drifting relative to it.
    let t = Wire3::new(scalar);
    let x = Wire3::new(scalar);
    let y = Wire3::new(scalar);

    // delay: the derivatives, both driven by the input's derivative `d(t)`.
    let delay = [
        // dx = 2 dt, synthesised tensor-free as dt + dt
        Term::function(LRA::TAdd(1, 1, 1), [d(x)], [d(t), d(t)]).unwrap(),
        // dy = 1 dt
        Term::function(LRA::TId(1, 1, 1), [d(y)], [d(t)]).unwrap(),
    ];

    // init: the initial values are left unconstrained.
    let init = [
        Term::constant(LRA::AnyReal(1, 1), [X(x)]).unwrap(),
        Term::constant(LRA::AnyReal(1, 1), [X(y)]).unwrap(),
    ];

    // The atom controls the clocks, keeps them in the update, and awaits the
    // derivative of the input clock.
    let ctrl = &[x, y];
    let atom: reactive::Atom<LRA, LRA, LRA> =
        generic_atom(ctrl, &[t], ctrl, init, skip(ctrl), delay)
            .expect("differential atom should be well-formed");
    let module =
        reactive::Module::observable([atom]).expect("differential module should be well-formed");

    // An open module: the input clock is external.
    assert!(module.is_open());
    let extl: HashSet<usize> = module.extl().iter().map(|v| v.name() as usize).collect();
    assert_eq!(extl, HashSet::from([t.id()]));

    // The clocks are controlled by the module, hence interface variables.
    let intf: HashSet<usize> = module.intf().iter().map(|v| v.name() as usize).collect();
    assert_eq!(intf, HashSet::from([x.id(), y.id()]));

    assert_eq!(module.atoms().len(), 1);
    let atom = &module.atoms()[0];

    // The atom controls the clocks and awaits the input.
    let ctrl: HashSet<usize> = atom.ctrl().iter().map(|v| v.name() as usize).collect();
    assert_eq!(ctrl, HashSet::from([x.id(), y.id()]));

    let wait: HashSet<usize> = atom.wait().iter().map(|v| v.name() as usize).collect();
    assert_eq!(wait, HashSet::from([t.id()]));

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

/// Proposes a differential module encoding
///
/// ```text
///     dx = x
/// ```
///
/// and expects it to be rejected: the delay term drives the derivative
/// `d(x)` (of the tangent sort) directly from the value wire `x`, and
/// the operations on derivatives do not read values.
#[test]
fn differential_module_rejects_dx_equals_x() {
    use theory::lra::{LRA, Sort};

    let scalar = Sort::real([1, 1]);
    let x = Wire3::new(scalar);

    // dx = x: already ill-formed at the term level.
    let delay = Term::function(LRA::TId(1, 1, 1), [d(x)], [x]);

    let atom = delay.and_then(|delay| {
        let init = [Term::constant(LRA::AnyReal(1, 1), [X(x)]).unwrap()];
        generic_atom::<LRA, LRA, LRA, _>(&[x], &[], &[x], init, skip(&[x]), [delay])
    });
    assert!(atom.is_err());
}

/// An atom whose initial state is left unconstrained and whose delay is
/// zero: only the update says anything.
#[test]
fn jump_atom_ok() {
    let x = Wire3::new(srt("real"));
    let y = Wire3::new(srt("real"));

    // x' = y: x is controlled, y is read
    let update = [Term::function(mk_op("ID"), [X(x)], [y]).unwrap()];
    let a = atom(&[x], &[], &[y], havoc(&[x]), update, zero(&[x])).unwrap();
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

/// The update drives next wires; writing a latched wire leaves the next
/// wire of the controlled variable unwritten.
#[test]
fn jump_atom_rejects_latched_write() {
    let x = Wire3::new(srt("real"));
    let y = Wire3::new(srt("real"));

    // writes the latched wire `x` instead of the next wire `X(x)`
    let update = [Term::function(mk_op("ID"), [*x], [y]).unwrap()];
    assert!(atom(&[x], &[], &[y], havoc(&[x]), update, zero(&[x])).is_err());
}

/// An atom setting its variable once at initialisation and keeping it ever
/// after.
#[test]
fn constant_atom_ok() {
    let x = Wire3::new(srt("real"));

    let init = [Term::constant(mk_op("CONST(0)"), [X(x)]).unwrap()];
    let a = atom(&[x], &[], &[x], init, skip(&[x]), zero(&[x])).unwrap();
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

/// The init drives next wires; writing a latched wire leaves the next wire
/// of the controlled variable unwritten.
#[test]
fn constant_atom_rejects_latched_write() {
    let x = Wire3::new(srt("real"));

    // writes the latched wire `x` instead of the next wire `X(x)`
    let init = [Term::constant(mk_op("CONST(0)"), [*x]).unwrap()];
    assert!(atom(&[x], &[], &[x], init, skip(&[x]), zero(&[x])).is_err());
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
/// The discrete jump drives the next wires, the continuous flow drives the
/// derivatives — the clock at rate 1, the location with zero drift, since
/// it only changes at jumps.
#[test]
fn simple_timed_automaton() {
    let x = Wire3::new(srt("real"));
    let loc = Wire3::new(srt("{off, on}"));

    // init: loc = off, x = 0
    let init = [
        Term::constant(mk_op("CONST(off)"), [X(loc)]).unwrap(),
        Term::constant(mk_op("CONST(0)"), [X(x)]).unwrap(),
    ];

    // jump: guard x >= 1; toggle the location and reset the clock when it
    // fires, hold both otherwise
    let guard = Wire::new(srt("bool"));
    let update = [
        Term::function(mk_op("GEQ(1)"), [guard], [x]).unwrap(),
        Term::function(mk_op("TOGGLE_IF"), [X(loc)], [guard, *loc]).unwrap(),
        Term::function(mk_op("RESET_IF"), [X(x)], [guard, *x]).unwrap(),
    ];

    // flow: dx = 1, dloc = 0
    let delay = [
        Term::constant(mk_op("ONE"), [d(x)]).unwrap(),
        Term::constant(mk_op("ZERO"), [d(loc)]).unwrap(),
    ];

    let vars = &[x, loc];
    let a = atom(vars, &[], vars, init, update, delay).unwrap();
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
/// unwritten, and the timed automaton is rejected.
#[test]
fn hybrid_rejects_missing_flow() {
    let x = Wire3::new(srt("real"));
    let loc = Wire3::new(srt("{off, on}"));

    let init = [
        Term::constant(mk_op("CONST(off)"), [X(loc)]).unwrap(),
        Term::constant(mk_op("CONST(0)"), [X(x)]).unwrap(),
    ];

    let guard = Wire::new(srt("bool"));
    let update = [
        Term::function(mk_op("GEQ(1)"), [guard], [x]).unwrap(),
        Term::function(mk_op("TOGGLE_IF"), [X(loc)], [guard, *loc]).unwrap(),
        Term::function(mk_op("RESET_IF"), [X(x)], [guard, *x]).unwrap(),
    ];

    // flow: dx = 1, but no dloc = 0
    let delay = [Term::constant(mk_op("ONE"), [d(x)]).unwrap()];

    let vars = &[x, loc];
    assert!(atom(vars, &[], vars, init, update, delay).is_err());
}

#[test]
fn coupled_atoms_can_construct_module() {
    let x = Wire3::new(srt("real"));
    let y = Wire3::new(srt("real"));
    let z = Wire3::new(srt("real"));

    let u1 = Term::function(mk_op("u1"), [X(y)], [x]).unwrap();
    let a1 = atom(&[y], &[], &[x], havoc(&[y]), [u1], zero(&[y])).unwrap();

    let u2 = Term::function(mk_op("u2"), [X(z)], [y]).unwrap();
    let a2 = atom(&[z], &[], &[y], havoc(&[z]), [u2], zero(&[z])).unwrap();

    let m = Module::observable([a1, a2]);
    assert!(m.is_ok());
}

/// The interface of the variables, of the given names and sorts
fn vars(vars: &[(u32, &'static str)]) -> AtomInterface<common::Srt> {
    AtomInterface::try_from_iter(vars.iter().map(|&(n, s)| Var::new(n, srt(s)))).unwrap()
}

/// A fresh wire of the sort
fn wire(sort: &'static str) -> Wire<common::Srt> {
    Wire::new(srt(sort))
}

/// A fresh wire of the tangent of the sort
fn tangent(sort: &'static str) -> Wire<common::Srt> {
    Wire::new(common::Srt(sort, 1))
}

/// The term discarding the wire
fn discard(w: Wire<common::Srt>) -> Term<Ops> {
    let nothing: [Wire<common::Srt>; 0] = [];
    Term::function(Ops::discard(w.dtype()), nothing, [w]).unwrap()
}

/// The blocks of an atom have the boundaries `X(wait) -> X(ctrl)` (init),
/// `read ++ X(wait) -> X(ctrl)` (update), and `read ++ d(wait) -> d(ctrl)`
/// (delay), each in the order of the interfaces; the atom binds its
/// variables to them by position, and checks their sorts.
#[test]
fn blocks_have_the_boundaries_of_the_interfaces() {
    // x : Int controlled, read (x : Int, e : Bool), wait (e : Bool)
    let (ctrl, read, wait) = (
        vars(&[(0, "Int")]),
        vars(&[(0, "Int"), (1, "Bool")]),
        vars(&[(1, "Bool")]),
    );

    let init = || {
        let (we, wx) = (wire("Bool"), wire("Int"));
        Block::new([we], [wx], [term!(mk_op("F"), [wx], [we]).unwrap()]).unwrap()
    };
    let update = || {
        let (x, e, we, wx) = (wire("Int"), wire("Bool"), wire("Bool"), wire("Int"));
        let terms = [term!(mk_op("G"), [wx], [x, e, we]).unwrap()];
        Block::new([x, e, we], [wx], terms).unwrap()
    };
    let delay = || {
        let (x, e, de, dx) = (wire("Int"), wire("Bool"), tangent("Bool"), tangent("Int"));
        let zero = term!(mk_op("ZERO"), [dx]).unwrap();
        let terms = [zero, discard(x), discard(e), discard(de)];
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
        let (x, e, we, wx) = (wire("Int"), wire("Bool"), wire("Bool"), wire("Int"));
        let terms = [term!(mk_op("G"), [wx], [x, e, we]).unwrap()];
        Block::new([e, x, we], [wx], terms).unwrap()
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
        let (x, e, de, wx) = (wire("Int"), wire("Bool"), tangent("Bool"), wire("Int"));
        let terms = [
            term!(mk_op("C"), [wx]).unwrap(),
            discard(x),
            discard(e),
            discard(de),
        ];
        Block::new([x, e, de], [wx], terms).unwrap()
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
        let wx = wire("Int");
        Block::new([], [wx], [term!(mk_op("C"), [wx]).unwrap()]).unwrap()
    };
    let a = Atom::new(ctrl, wait, read, short, update(), delay());
    assert!(a.is_err());
}

/// A block takes every input of its boundary, and has to read it: the inputs
/// it has no use for, it discards.
#[test]
fn unread_inputs_are_discarded() {
    let x = Wire3::new(srt("Int"));
    let e = Wire3::new(srt("Int"));

    // the init ignores `e`, the update awaits it
    let init = [term!(mk_op("CONST"), [X(x)]).unwrap()];
    assert!(Block::new([X(e)], [X(x)], init.clone()).is_err());

    let a = atom(
        &[x],
        &[e],
        &[],
        init,
        [term!(mk_op("ID"), [X(x)], [X(e)]).unwrap()],
        zero(&[x]),
    );
    let a = a.unwrap();

    assert_eq!(a.init().read(), [X(e)]);
    let ops: Vec<_> = a.init().iter().map(|t| t.itype().0).collect();
    assert_eq!(ops, ["CONST", "DISCARD"]);

    // the update reads all its inputs, nothing is discarded
    assert_eq!(a.update().read(), [X(e)]);
    assert_eq!(a.update().len(), 1);
}

/// A variable is not both controlled and awaited by an atom.
#[test]
fn controlled_variables_are_not_awaited() {
    let ctrl = vars(&[(0, "Int")]);
    let (wi, wo) = (wire("Int"), wire("Int"));
    let init = Block::new([wi], [wo], [term!(mk_op("ID"), [wo], [wi]).unwrap()]).unwrap();
    let (wi, wo) = (wire("Int"), wire("Int"));
    let update = Block::new([wi], [wo], [term!(mk_op("ID"), [wo], [wi]).unwrap()]).unwrap();
    let (di, dout) = (tangent("Int"), tangent("Int"));
    let delay = Block::new([di], [dout], [term!(mk_op("ID"), [dout], [di]).unwrap()]).unwrap();

    let a = Atom::new(ctrl.clone(), ctrl, vars(&[]), init, update, delay);
    assert!(a.is_err());
}

/// The same name is the same variable in all the interfaces of an atom, of
/// one sort.
#[test]
fn interfaces_agree_on_sorts() {
    // `x` is controlled as an Int and read as a Bool
    let (ctrl, read) = (vars(&[(0, "Int")]), vars(&[(0, "Bool")]));

    let init = || {
        let wx = wire("Int");
        Block::new([], [wx], [term!(mk_op("C"), [wx]).unwrap()]).unwrap()
    };
    let update = |sort| {
        let (x, wx) = (wire(sort), wire("Int"));
        Block::new([x], [wx], [term!(mk_op("C"), [wx]).unwrap(), discard(x)]).unwrap()
    };
    let delay = |sort| {
        let (x, dx) = (wire(sort), tangent("Int"));
        Block::new([x], [dx], [term!(mk_op("ZERO"), [dx]).unwrap(), discard(x)]).unwrap()
    };

    let a = Atom::new(
        ctrl.clone(),
        vars(&[]),
        read,
        init(),
        update("Bool"),
        delay("Bool"),
    );
    assert!(a.is_err());

    // read as an Int, it is fine
    let read = vars(&[(0, "Int")]);
    let a = Atom::new(ctrl, vars(&[]), read, init(), update("Int"), delay("Int"));
    assert!(a.is_ok());
}

/// The interfaces, and so the boundaries of the blocks, follow the order of
/// the names, not the order the variables were created in.
#[test]
fn boundaries_follow_the_order_of_names() {
    let x = Wire3::new(srt("Int"));
    let e = Wire3::new(srt("Bool"));

    // `e` is named before `x`
    let (vx, ve) = (Var::new(1, srt("Int")), Var::new(0, srt("Bool")));
    let ctrl = AtomInterface::try_from_iter([vx]).unwrap();
    let read = AtomInterface::try_from_iter([vx, ve]).unwrap();
    assert_eq!(read.iter().cloned().collect::<Vec<_>>(), [ve, vx]);

    let init = || block([], [X(x)], [term!(mk_op("C"), [X(x)]).unwrap()]).unwrap();
    let update = |read: [Wire<_>; 2]| {
        block(read, [X(x)], [term!(mk_op("G"), [X(x)], [*x, *e]).unwrap()]).unwrap()
    };
    let delay = |read: [Wire<_>; 2]| block(read, [d(x)], zero(&[x])).unwrap();

    let a = Atom::new(
        ctrl.clone(),
        vars(&[]),
        read.clone(),
        init(),
        update([*e, *x]),
        delay([*e, *x]),
    );
    assert!(a.is_ok());

    // in the order of creation
    let a = Atom::new(
        ctrl,
        vars(&[]),
        read,
        init(),
        update([*x, *e]),
        delay([*x, *e]),
    );
    assert!(a.is_err());
}
