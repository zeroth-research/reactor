use crate::Module;
use base::Wire;
use dataflow::diagram::{Block, Term};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

use crate::var::{AtomInterface, Var, Wire3};
use base::tangent::Tangent;
use std::fmt::Debug;

//============================================================
// Atom
//============================================================

/// This data structure corresponds to the atom of reactive modules.
#[derive(Debug, Clone)]
pub struct Atom<I, J, F, S = <I as Theory>::Sort>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// The controlled variables.
    ctrl: AtomInterface<S>,
    /// The awaited variables.
    wait: AtomInterface<S>,
    /// The read variables.
    read: AtomInterface<S>,
    /// Corresponds to the initial condition, with the boundary
    /// `X(wait) -> X(ctrl)`.
    init: Block<I>,
    /// Corresponds to the update action, with the boundary
    /// `read ++ X(wait) -> X(ctrl)`.
    update: Block<J>,
    /// Corresponds to the delay activity, with the boundary
    /// `read ++ d(wait) -> d(ctrl)`.
    delay: Block<F>,
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Returns a reference to the initial action.
    pub fn init(&self) -> &Block<I> {
        &self.init
    }
    /// Returns a reference to the update action.
    pub fn update(&self) -> &Block<J> {
        &self.update
    }

    /// Returns a reference to the delay activity.
    pub fn delay(&self) -> &Block<F> {
        &self.delay
    }

    /// Returns a reference to the controlled variables.
    pub fn ctrl(&self) -> &AtomInterface<S> {
        &self.ctrl
    }

    /// Returns a reference to the awaited variables.
    pub fn wait(&self) -> &AtomInterface<S> {
        &self.wait
    }

    /// Returns a reference to the read variables.
    pub fn read(&self) -> &AtomInterface<S> {
        &self.read
    }

    /// Constructs an atom with no variables and empty blocks.
    pub fn empty() -> Self {
        Self {
            ctrl: AtomInterface::empty(),
            wait: AtomInterface::empty(),
            read: AtomInterface::empty(),
            init: Block::empty(),
            update: Block::empty(),
            delay: Block::empty(),
        }
    }
}

impl<I, J, F, S> Default for Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    fn default() -> Self {
        Self::empty()
    }
}

//============================================================
// Private routines
//============================================================

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Returns true if this atom awaits the other atom.
    pub fn awaits(&self, other: &Atom<I, J, F, S>) -> bool {
        !self.wait.is_disjoint(&other.ctrl)
    }
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + PartialEq + Debug + Tangent,
{
    /// Constructs an atom **without performing any consistency checks**.
    ///
    /// This constructor provides **full control** to the caller and performs
    /// no inference or validation. Unlike the public constructors, which
    /// infer the interfaces from the blocks, it takes every component as
    /// given. It should be used only when all necessary
    /// checks and inference have already been handled externally.
    ///
    /// # Parameters
    /// - `ctrl`: The controlled variables, whose next and derivative wires the blocks write.
    /// - `wait`: The awaited variables, whose next or derivative wires the blocks read.
    /// - `read`: The read variables, whose latched wires the blocks read.
    /// - `init`, `update`, `delay`: The blocks of the atom. Their boundaries
    ///   list the wires of the variables in the order of the interfaces (of
    ///   the names): the init maps the next wires of `wait` to the next wires
    ///   of `ctrl`; the update maps the latched wires of `read` followed by the
    ///   next wires of `wait` to the next wires of `ctrl`; and the delay maps
    ///   the latched wires of `read` followed by the derivative wires of
    ///   `wait` to the derivative wires of `ctrl`.
    ///
    /// # Returns
    /// The constructed atom.
    ///
    /// # Correctness
    /// This function performs **no validation or inference** (beyond
    /// debug-only assertions). It is the caller’s responsibility to ensure
    /// that the interfaces and blocks are well-formed and consistent.
    ///
    /// # See Also
    /// - [`Atom::sequential`], [`Atom::combinatorial`] and the other public
    ///   constructors for checked, automated atom construction.
    fn new_unchecked(
        ctrl: AtomInterface<S>,
        wait: AtomInterface<S>,
        read: AtomInterface<S>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
    ) -> Self {
        debug_assert!(ctrl.is_disjoint(&wait));

        // the boundaries of the blocks carry the sorts of the interfaces, the
        // derivatives their tangents
        let sorts = |vars: &AtomInterface<S>| vars.iter().map(|v| v.dtype().clone()).collect();
        let tangents = |vars: &AtomInterface<S>| vars.iter().map(|v| v.dtype().tangent()).collect();
        let wires = |wires: &[Wire<S>]| wires.iter().map(|w| w.dtype().clone()).collect::<Vec<S>>();
        let concat = |a: Vec<S>, b: Vec<S>| [a, b].concat();
        debug_assert_eq!(wires(init.read()), sorts(&wait));
        debug_assert_eq!(wires(init.write()), sorts(&ctrl));
        debug_assert_eq!(wires(update.read()), concat(sorts(&read), sorts(&wait)));
        debug_assert_eq!(wires(update.write()), sorts(&ctrl));
        debug_assert_eq!(wires(delay.read()), concat(sorts(&read), tangents(&wait)));
        debug_assert_eq!(wires(delay.write()), tangents(&ctrl));

        Self {
            ctrl,
            wait,
            read,
            init,
            update,
            delay,
        }
    }
}

/// The variables in scope for an atom: the variables of their wires, and
/// their names.
struct WireMap<'a, S> {
    ltc: HashMap<&'a Wire<S>, &'a Wire3<S>>,
    nxt: HashMap<&'a Wire<S>, &'a Wire3<S>>,
    der: HashMap<&'a Wire<S>, &'a Wire3<S>>,
    names: HashMap<&'a Wire3<S>, Var<S>>,
}

impl<'a, S: PartialEq + Debug> WireMap<'a, S> {
    /// Unpacks the variables, each bound to its name.
    ///
    /// # Errors
    /// If the sort of a name is not the sort of the variable bound to it, or
    /// the binding is not one-to-one: a name is bound to two variables, or a
    /// variable to two names. Binding the same name to the same variable more
    /// than once is fine.
    fn unpack<V>(vars: V) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
    {
        let mut ltc: HashMap<&Wire<S>, &Wire3<S>> = HashMap::new();
        let mut nxt: HashMap<&Wire<S>, &Wire3<S>> = HashMap::new();
        let mut der: HashMap<&Wire<S>, &Wire3<S>> = HashMap::new();
        let mut names: HashMap<&Wire3<S>, Var<S>> = HashMap::new();
        let mut bound: HashMap<u32, &Wire3<S>> = HashMap::new();

        for (name, var) in vars.into_iter() {
            if name.dtype() != var.dtype() {
                return Err(format!(
                    "Variable {} of sort {:?} is bound to wires of sort {:?}",
                    name.name(),
                    name.dtype(),
                    var.dtype()
                ));
            }
            if bound.get(&name.name()).is_some_and(|&other| other != var) {
                return Err(format!(
                    "Variable {} is bound to two different wires",
                    name.name()
                ));
            }
            if let Some(other) = names.get(var).filter(|other| *other != &name) {
                return Err(format!(
                    "Wires {:?} are bound to two variables {} and {}",
                    var,
                    other.name(),
                    name.name()
                ));
            }

            bound.insert(name.name(), var);
            names.insert(var, name);
            ltc.insert(var.ltc(), var);
            nxt.insert(var.nxt(), var);
            der.insert(var.der(), var);
        }

        Ok(Self {
            ltc,
            nxt,
            der,
            names,
        })
    }
}

fn infer_ctrl<T, S: Clone>(
    block: &Block<T>,
    pool: &HashMap<&Wire<S>, &Wire3<S>>,
) -> BTreeSet<Wire3<S>>
where
    T: Theory<Sort = S>,
{
    // tree map on id is used to guarantee consistent order
    let mut ctrl: BTreeSet<Wire3<S>> = BTreeSet::new();

    for wt in block.write().iter() {
        // if the block writes to a wire in the pool of next or derived,
        // then the respective variable is controlled
        if let Some(&var) = pool.get(&wt) {
            ctrl.insert(var.clone());
        }
    }

    ctrl
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + PartialEq + Debug + Tangent,
{
    fn with_ctrl(
        wmap: WireMap<S>,
        ctrl: BTreeSet<Wire3<S>>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
    ) -> Result<Self, String> {
        let mut read: BTreeSet<Wire3<S>> = BTreeSet::new();
        let mut wait: BTreeSet<Wire3<S>> = BTreeSet::new();

        for rd in init.read().iter() {
            // init can only read from next wires
            if let Some(&var) = wmap.nxt.get(&rd) {
                wait.insert(var.clone());
                continue;
            }

            if wmap.ltc.contains_key(&rd) {
                return Err(format!("Init reads latched wire {:?}", rd));
            } else if wmap.der.contains_key(&rd) {
                return Err(format!("Init reads d wire {:?}", rd));
            } else {
                return Err(format!("Init reads dangling wire {:?}", rd));
            }
        }

        for rd in update.read().iter() {
            // if the update reads from a next wire, then this is awaited
            // otherwise, it must be read from outside the atom (dangling)
            if let Some(&var) = wmap.ltc.get(&rd) {
                read.insert(var.clone());
                continue;
            }

            if let Some(&var) = wmap.nxt.get(&rd) {
                wait.insert(var.clone());
                continue;
            }

            if wmap.der.contains_key(&rd) {
                return Err(format!("Update reads d wire {:?}", rd));
            } else {
                return Err(format!("Update reads dangling wire {:?}", rd));
            }
        }

        for rd in delay.read().iter() {
            // if the delay reads from a derived wire, then this is awaited
            // otherwise, it must be read from outside the atom (dangling)
            if let Some(&var) = wmap.ltc.get(&rd) {
                read.insert(var.clone());
                continue;
            }

            if let Some(&var) = wmap.der.get(&rd) {
                wait.insert(var.clone());
                continue;
            }

            if wmap.nxt.contains_key(&rd) {
                return Err(format!("Delay reads X wire {:?}", rd));
            } else {
                return Err(format!("Delay reads dangling wire {:?}", rd));
            }
        }

        for wt in init.write().iter().chain(update.write().iter()) {
            // if the init/update writes to a next wire, then this wire must be controlled
            // otherwise, it must be local
            if let Some(&var) = wmap.nxt.get(&wt) {
                if !ctrl.contains(var) {
                    return Err(format!("Init/update writes uncontrolled X wire {:?}", wt));
                }
                continue;
            }

            if wmap.ltc.contains_key(&wt) {
                return Err(format!("Init/update writes latched wire {:?}", wt));
            } else if wmap.der.contains_key(&wt) {
                return Err(format!("Init/update writes d wire {:?}", wt));
            }
        }

        for wt in delay.write().iter() {
            // if the delay writes to a derived wire, then this wire is controlled
            // otherwise, it must be local
            if let Some(&var) = wmap.der.get(&wt) {
                if !ctrl.contains(var) {
                    return Err(format!("Delay writes uncontrolled d wire {:?}", wt));
                }
                continue;
            }

            if wmap.ltc.contains_key(&wt) {
                return Err(format!("Delay writes latched wire {:?}", wt));
            } else if wmap.nxt.contains_key(&wt) {
                return Err(format!("Delay writes X wire {:?}", wt));
            }
        }

        for ctr in ctrl.iter() {
            if init.write().binary_search(ctr.nxt()).is_err() {
                return Err(format!(
                    "Controlled X wire {:?} is not written in init",
                    ctr.nxt()
                ));
            }
            if update.write().binary_search(ctr.nxt()).is_err() {
                return Err(format!(
                    "Controlled X wire {:?} is not written in update",
                    ctr.nxt()
                ));
            }
            if delay.write().binary_search(ctr.der()).is_err() {
                return Err(format!(
                    "Controlled d wire {:?} is not written in delay",
                    ctr.der()
                ));
            }
        }

        // the interfaces, and the boundaries, list the variables in the order
        // of their names
        let named = |vars: BTreeSet<Wire3<S>>| {
            let mut vars: Vec<(Var<S>, Wire3<S>)> = vars
                .into_iter()
                .map(|var| (wmap.names[&var].clone(), var))
                .collect();
            vars.sort_by_key(|(name, _)| name.name());
            vars
        };
        let (ctrl, wait, read) = (named(ctrl), named(wait), named(read));

        let ctrl_nxt = || ctrl.iter().map(|(_, var)| var.nxt());
        let ctrl_der = || ctrl.iter().map(|(_, var)| var.der());
        let read_ltc = || read.iter().map(|(_, var)| var.ltc());
        let wait_nxt = || wait.iter().map(|(_, var)| var.nxt());
        let wait_der = || wait.iter().map(|(_, var)| var.der());

        let init = with_boundary(init, wait_nxt(), ctrl_nxt())?;
        let update = with_boundary(update, read_ltc().chain(wait_nxt()), ctrl_nxt())?;
        let delay = with_boundary(delay, read_ltc().chain(wait_der()), ctrl_der())?;

        let interface = |vars: Vec<(Var<S>, Wire3<S>)>| {
            AtomInterface::from_sorted_unchecked(vars.into_iter().map(|(name, _)| name).collect())
        };
        Ok(Self::new_unchecked(
            interface(ctrl),
            interface(wait),
            interface(read),
            init,
            update,
            delay,
        ))
    }
}

/// Rebuilds the block with the boundary `read -> write`, consuming the inputs
/// its terms do not read with [`Theory::discard`] terms.
fn with_boundary<'a, T, S, R, W>(block: Block<T>, read: R, write: W) -> Result<Block<T>, String>
where
    T: Theory<Sort = S>,
    S: Clone + Debug + 'a,
    R: IntoIterator<Item = &'a Wire<S>>,
    W: IntoIterator<Item = &'a Wire<S>>,
{
    let read: Vec<Wire<S>> = read.into_iter().cloned().collect();
    let write: Vec<Wire<S>> = write.into_iter().cloned().collect();

    let used: HashSet<&Wire<S>> = block.iter().flat_map(Term::read).collect();
    let discard = |w: &Wire<S>| {
        let nothing: [Wire<S>; 0] = [];
        Term::function(T::discard(w.dtype()), nothing, [w.clone()])
    };
    let unused = read.iter().filter(|w| !used.contains(w));
    let discards: Vec<Term<T>> = unused.map(discard).collect::<Result<_, _>>()?;

    Block::new(read, write, block.into_iter().chain(discards))
}

//============================================================
// Public constructors
//============================================================

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + PartialEq + Debug + Tangent,
{
    /// Constructs a **sequential atom**: behaviour that evolves over discrete
    /// time steps.
    ///
    /// A sequential atom specifies an initialisation (`init`) and a discrete
    /// state update (`update`), relating the latched wires to the next wires
    /// across time steps. The delay is synthesised as `ZERO`, so the atom has
    /// no continuous dynamics.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::differential`], the continuous dual: only a delay, implicit skip update.
    /// - [`Atom::hybrid`], when both discrete and continuous dynamics are explicit.
    /// - [`Atom::combinatorial`], for time-independent, purely reactive behaviour.
    pub fn sequential<'b, V, W, U>(vars: V, init: W, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'b Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'b,
    {
        let wires = WireMap::unpack(vars)?;

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&init, &wires.nxt);
        let ctrl_der = ctrl.iter().map(Wire3::der).cloned();

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **differential atom**: behaviour that evolves continuously.
    ///
    /// A differential atom specifies an initialisation (`init`) and a
    /// continuous evolution (`delay`) driving the derivative wires. The
    /// discrete update is synthesised as one `SKIP` term per controlled
    /// variable, copying the latched value to the next wire, so all change
    /// comes from the continuous dynamics.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed differential atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], the discrete dual: only an update, zero delay.
    /// - [`Atom::hybrid`], when both discrete and continuous dynamics are explicit.
    pub fn differential<'b, V, W, Z>(vars: V, init: W, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'b Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'b,
    {
        let wires = WireMap::unpack(vars)?;

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&init, &wires.nxt);
        let ctrl_nxt = ctrl.iter().map(Wire3::nxt).cloned();
        let ctrl_ltc = ctrl.iter().map(Wire3::ltc).cloned();

        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **hybrid atom**, combining discrete and continuous dynamics.
    ///
    /// A hybrid atom specifies all three blocks explicitly: an initialisation
    /// (`init`), a discrete state update (`update`), and a continuous
    /// evolution (`delay`) driving the derivative wires. It is the most
    /// general constructor; [`Atom::sequential`] and [`Atom::differential`]
    /// are the special cases where the delay (resp. the update) is
    /// synthesised automatically.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed hybrid atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], for purely discrete behaviour (delay is zero).
    /// - [`Atom::differential`], for purely continuous behaviour (update is skip).
    pub fn hybrid<'a, V, W, U, Z>(vars: V, init: W, update: U, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars)?;

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        let ctrl = infer_ctrl(&init, &wires.nxt);

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **jump atom**: sequential behaviour whose
    /// initial state is left unconstrained.
    ///
    /// Only the update is given; the controlled variables are inferred from
    /// the next wires it writes. The initialisation is synthesised as one
    /// `HAVOC` term per controlled variable, so their initial values are
    /// arbitrary, and the delay is synthesised as `ZERO`, so the atom has no
    /// continuous dynamics.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `update`: The terms defining the discrete state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], when the initial state is constrained explicitly.
    /// - [`Atom::constant`], the dual: only the initial state, no update.
    pub fn jump<'a, V, U>(vars: V, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        U: IntoIterator<Item = Term<J>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars)?;

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&update, &wires.nxt);
        let ctrl_nxt = ctrl.iter().map(Wire3::nxt).cloned();
        let ctrl_der = ctrl.iter().map(Wire3::der).cloned();

        let init = Block::havoc(ctrl_nxt)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs an **uninitialized atom**: discrete and continuous
    /// dynamics with an unconstrained initial state.
    ///
    /// The discrete update and the continuous delay are given; the
    /// controlled variables are inferred from the next wires the update
    /// writes, and the initialisation is synthesised as one `HAVOC` term per
    /// controlled variable, so the initial values are arbitrary.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `update`: The terms defining the discrete state update at each time step.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::hybrid`], when the initial state is constrained explicitly.
    /// - [`Atom::jump`] and [`Atom::flow`], the purely discrete and purely
    ///   continuous special cases.
    pub fn uninitialized<'a, V, U, Z>(vars: V, update: U, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars)?;

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&update, &wires.nxt);
        let ctrl_nxt = ctrl.iter().map(Wire3::nxt).cloned();

        let init = Block::havoc(ctrl_nxt)?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **constant atom**: variables that are set once at
    /// initialisation and then hold their value forever.
    ///
    /// Only the initialisation is given; the controlled variables are
    /// inferred from the next wires it writes. The update is synthesised as
    /// one `SKIP` term per controlled variable, copying the latched value to
    /// the next wire at every step, and the delay is synthesised as `ZERO`,
    /// so the values never drift.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `init`: The terms defining the (only) assignment of the variables.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::jump`], the dual: only an update, arbitrary initial state.
    /// - [`Atom::sequential`], when the variables also evolve over time.
    pub fn constant<'a, V, W>(vars: V, init: W) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars)?;

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&init, &wires.nxt);
        let ctrl_ltc = ctrl.iter().map(Wire3::ltc).cloned();
        let ctrl_nxt = ctrl.iter().map(Wire3::nxt).cloned();
        let ctrl_der = ctrl.iter().map(Wire3::der).cloned();

        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **hold atom**: variables that hold an arbitrary but
    /// fixed value — an adversarial constant.
    ///
    /// The atom takes no blocks: it controls every variable in `vars`, with
    /// all three blocks synthesised per variable — `HAVOC` init, `SKIP`
    /// update, `ZERO` delay. Each value is chosen nondeterministically at
    /// initialisation and never changes: this models free parameters, i.e.,
    /// quantities a system depends on but does not compute.
    ///
    /// # Parameters
    /// - `vars`: The variables of the atom, all of which it controls, each bound
    ///   to its name.
    ///
    /// # Returns
    /// A `Result` containing the constructed hold atom if successful,
    /// or an error string if consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::constant`], when the held value is set explicitly at initialisation.
    /// - [`Atom::jump`], when the variables also evolve over time.
    /// - [`Module::hold`], for the module-level shorthand.
    pub fn hold<'a, V>(vars: V) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars)?;

        // every variable is controlled
        let ctrl: BTreeSet<Wire3<S>> = wires.ltc.values().cloned().cloned().collect();
        let ctrl_ltc = ctrl.iter().map(Wire3::ltc).cloned();
        let ctrl_nxt = ctrl.iter().map(Wire3::nxt).cloned();
        let ctrl_der = ctrl.iter().map(Wire3::der).cloned();

        let init = Block::havoc(ctrl_nxt.clone())?;
        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **flow atom**: purely continuous behaviour with an
    /// unconstrained initial state.
    ///
    /// Only the continuous evolution (`delay`) is given; the controlled
    /// variables are inferred from the derivative wires it writes. The
    /// initialisation is synthesised as one `HAVOC` term per controlled
    /// variable, so the initial values are arbitrary, and the update as
    /// `SKIP`, so all change comes from the continuous dynamics.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed flow atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::differential`], when the initial state is constrained explicitly.
    /// - [`Atom::hybrid`], when discrete dynamics are involved as well.
    pub fn flow<'a, V, Z>(vars: V, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars)?;

        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&delay, &wires.der);
        let ctrl_ltc = ctrl.iter().map(Wire3::ltc).cloned();
        let ctrl_nxt = ctrl.iter().map(Wire3::nxt).cloned();

        let init = Block::havoc(ctrl_nxt.clone())?;
        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **combinatorial atom**: purely reactive behaviour without
    /// temporal state.
    ///
    /// A combinatorial atom specifies a single block of assignments (`assign`)
    /// relating next wires *within the same time step*. The same block serves
    /// as both initialisation and update, and the delay is synthesised as
    /// `ZERO`, so the atom is time-independent.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `assign`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom, each bound to its name.
    /// - `assign`: The terms defining the combinatorial relationships between next wires.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], the stateful dual with explicit time evolution.
    /// - [`Module::combinatorial`], for combinatorial modules.
    pub fn combinatorial<'a, T, V, W>(vars: V, assign: W) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<T>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars)?;

        let assign: Block<T> = Block::try_from_iter(assign.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&assign, &wires.nxt);

        let init: Block<I> = Block::convert(assign.clone());
        let update: Block<J> = Block::convert(assign);
        let delay: Block<F> = Block::zero(ctrl.iter().map(Wire3::der).cloned())?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }
}

//============================================================
// Display routines
//============================================================

pub struct Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    atom: &'a Atom<I, J, F, S>,
    name: N,
    typed: bool,
}

/// The value of a variable a boundary wire of a block carries: latched, next,
/// or derivative.
#[derive(Clone, Copy)]
enum View {
    Ltc,
    Nxt,
    Der,
}

impl<'a, I, J, F, S, N> Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt_vars<V: Iterator<Item = &'a Var<S>>>(
        &self,
        f: &mut fmt::Formatter<'_>,
        iter: V,
    ) -> fmt::Result {
        for (i, var) in iter.enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if self.typed {
                write!(f, "{} : {}", (self.name)(var), var.dtype())?;
            } else {
                write!(f, "{}", (self.name)(var))?;
            }
        }
        Ok(())
    }

    /// Names the boundary wires of the block after the variables they carry:
    /// the inputs after `inputs`, and the outputs after `outputs`, in order.
    fn wirenames<T, V, W>(
        &self,
        block: &'a Block<T>,
        inputs: V,
        outputs: W,
    ) -> HashMap<&'a Wire<S>, Cow<'a, str>>
    where
        T: Theory<Sort = S>,
        V: Iterator<Item = (&'a Var<S>, View)>,
        W: Iterator<Item = (&'a Var<S>, View)>,
    {
        let name = |(var, view): (&'a Var<S>, View)| match view {
            // the plain name passes through as-is: borrowed stays borrowed
            View::Ltc => (self.name)(var),
            View::Nxt => Cow::Owned(format!("X({})", (self.name)(var))),
            View::Der => Cow::Owned(format!("d({})", (self.name)(var))),
        };
        let inputs = block.read().iter().zip(inputs.map(name));
        let outputs = block.write().iter().zip(outputs.map(name));
        inputs.chain(outputs).collect()
    }

    fn fmt_terms<T: 'a + Theory<Sort = S> + fmt::Display>(
        &self,
        f: &mut fmt::Formatter<'_>,
        block: &'a Block<T>,
        names: &HashMap<&'a Wire<S>, Cow<'a, str>>,
        pad: &str,
    ) -> fmt::Result {
        // the wires internal to the block are not named after variables; the
        // argument annotation makes the closure higher-ranked over the wire's
        // lifetime, as `with_wirenames`'s `Fn` bound requires
        let name = |w: &Wire<S>| match names.get(w) {
            Some(name) => name.clone(),
            None => Cow::Owned(format!("{}", w.untyped())),
        };

        for term in block.iter() {
            let display = term.with_wirenames(name);
            writeln!(f, "{pad}{}", display)?;
        }
        Ok(())
    }

    pub fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";

        let a = self.atom;
        write!(f, "{pad}{BOLD}atom{RESET}")?;
        write!(f, " {BOLD}controls{RESET} ")?;
        self.fmt_vars(f, a.ctrl.iter())?;
        if !a.read.is_empty() {
            write!(f, " {BOLD}reads{RESET} ")?;
            self.fmt_vars(f, a.read.iter())?;
        }
        if !a.wait.is_empty() {
            write!(f, " {BOLD}awaits{RESET} ")?;
            self.fmt_vars(f, a.wait.iter())?;
        }

        // the boundaries of the blocks, as in `Atom::new_unchecked`
        let view = |vars: &'a AtomInterface<S>, view: View| vars.iter().map(move |v| (v, view));
        let (ctrl, read, wait) = (&a.ctrl, &a.read, &a.wait);
        let init = self.wirenames(&a.init, view(wait, View::Nxt), view(ctrl, View::Nxt));
        let update = self.wirenames(
            &a.update,
            view(read, View::Ltc).chain(view(wait, View::Nxt)),
            view(ctrl, View::Nxt),
        );
        let delay = self.wirenames(
            &a.delay,
            view(read, View::Ltc).chain(view(wait, View::Der)),
            view(ctrl, View::Der),
        );

        let tpad = format!("{pad}{INDENT}");
        writeln!(f, "\n{pad}{BOLD}init{RESET}")?;
        self.fmt_terms(f, &a.init, &init, tpad.as_str())?;
        writeln!(f, "{pad}{BOLD}delay{RESET}")?;
        self.fmt_terms(f, &a.delay, &delay, tpad.as_str())?;
        writeln!(f, "{pad}{BOLD}update{RESET}")?;
        self.fmt_terms(f, &a.update, &update, tpad.as_str())?;

        Ok(())
    }
}

impl<'a, I, J, F, S, N> fmt::Display for Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}

impl<I, J, D, S> Atom<I, J, D, S>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    D: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
{
    /// Displays the atom with variables under the names the given function
    /// assigns, typed.
    pub fn with_varnames<'a, N, R>(
        &'a self,
        name: N,
    ) -> Display<'a, I, J, D, S, impl Fn(&Var<S>) -> Cow<'a, str>>
    where
        N: Fn(&Var<S>) -> R,
        R: Into<Cow<'a, str>>,
    {
        Display {
            atom: self,
            name: move |v: &Var<S>| name(v).into(),
            typed: true,
        }
    }

    /// Displays the atom with variables under the names the given function
    /// assigns, untyped.
    pub fn with_varnames_untyped<'a, N, R>(
        &'a self,
        name: N,
    ) -> Display<'a, I, J, D, S, impl Fn(&Var<S>) -> Cow<'a, str>>
    where
        N: Fn(&Var<S>) -> R,
        R: Into<Cow<'a, str>>,
    {
        Display {
            atom: self,
            name: move |v: &Var<S>| name(v).into(),
            typed: false,
        }
    }
}

//============================================================
// Atomic modules
//============================================================

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + PartialEq + Debug + Tangent,
{
    /// Constructs a **sequential module**: behaviour that evolves over
    /// discrete time steps.
    ///
    /// A sequential module specifies an initialisation and a discrete state
    /// update; it has no continuous dynamics. It is composed of a single
    /// [`Atom::sequential`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `update`: The set of terms defining the module's state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::differential`] and [`Module::hybrid`], when continuous
    ///   dynamics are involved.
    /// - [`Module::combinatorial`], for stateless, time-independent modules.
    /// - [`Atom::sequential`], for creating individual sequential atoms.
    pub fn sequential<'a, V, W, U>(vars: V, init: W, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'a,
    {
        let atom = Atom::sequential(vars, init, update)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **differential module**: behaviour that evolves
    /// continuously.
    ///
    /// A differential module specifies an initialisation and a continuous
    /// evolution of the derivatives; the discrete update is an implicit skip.
    /// It is composed of a single [`Atom::differential`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed differential module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], the discrete dual: only an update, no continuous dynamics.
    /// - [`Module::hybrid`], when both discrete and continuous dynamics are explicit.
    /// - [`Atom::differential`], for creating individual differential atoms.
    pub fn differential<'a, V, W, Z>(vars: V, init: W, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let atom = Atom::differential(vars, init, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **hybrid module**, combining discrete updates with
    /// continuous dynamics.
    ///
    /// A hybrid module specifies all three blocks explicitly — initialisation,
    /// discrete update, and continuous delay. It is composed of a single
    /// [`Atom::hybrid`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `update`: The set of terms defining the discrete state update.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed hybrid module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`] and [`Module::differential`], the purely
    ///   discrete and purely continuous special cases.
    /// - [`Atom::hybrid`], for creating individual hybrid atoms.
    pub fn hybrid<'a, V, W, U, Z>(vars: V, init: W, update: U, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let atom = Atom::hybrid(vars, init, update, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **jump module**: sequential behaviour whose
    /// initial state is left unconstrained.
    ///
    /// Only the update is given; the initial values of the controlled
    /// variables are havoced. It is composed of a single
    /// [`Atom::jump`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `update`: The set of terms defining the module's state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], when the initial state is constrained explicitly.
    /// - [`Module::constant`], the dual: only the initial state, no update.
    pub fn jump<'a, V, U>(vars: V, update: U) -> Result<Self, String>
    where
        U: IntoIterator<Item = Term<J>>,
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        S: 'a,
    {
        let atom = Atom::jump(vars, update)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs an **uninitialized module**: discrete and continuous
    /// dynamics with an unconstrained initial state.
    ///
    /// The discrete update and the continuous delay are given; the initial
    /// values of the controlled variables are havoced. It is composed of a
    /// single [`Atom::uninitialized`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `update`: The set of terms defining the discrete state update.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::hybrid`], when the initial state is constrained explicitly.
    /// - [`Module::jump`] and [`Module::flow`], the purely discrete and
    ///   purely continuous special cases.
    pub fn uninitialized<'a, V, U, Z>(vars: V, update: U, delay: Z) -> Result<Self, String>
    where
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        S: 'a,
    {
        let atom = Atom::uninitialized(vars, update, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **hold module**: variables that hold an arbitrary but
    /// fixed value — symbolic constants.
    ///
    /// The module takes no blocks: each value is chosen nondeterministically
    /// at initialisation and never changes, modelling free parameters of a
    /// composition. It is composed of a single [`Atom::hold`] atom, and is
    /// **fully observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    ///
    /// # Returns
    /// A `Result` containing the constructed hold module if successful,
    /// or an error string if consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::constant`], when the held value is set explicitly at initialisation.
    /// - [`Module::jump`], when the variables also evolve over time.
    /// - [`Atom::hold`], for creating individual hold atoms.
    pub fn hold<'a, V>(vars: V) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        S: 'a,
    {
        let atom = Atom::hold(vars)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **flow module**: purely continuous behaviour with an
    /// unconstrained initial state.
    ///
    /// Only the continuous evolution of the derivatives is given; the initial
    /// values are havoced and the discrete update is an implicit skip. It is
    /// composed of a single [`Atom::flow`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed flow module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::differential`], when the initial state is constrained explicitly.
    /// - [`Atom::flow`], for creating individual flow atoms.
    pub fn flow<'a, V, Z>(vars: V, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let atom = Atom::flow(vars, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **constant module**: variables that are set once at
    /// initialisation and then hold their value forever.
    ///
    /// Only the initialisation is given; the variables are held by an implicit
    /// skip update and have no continuous dynamics. It is composed of a single
    /// [`Atom::constant`] atom, and is **fully observable**: use
    /// [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `init`: The set of terms defining the (only) assignment of the variables.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::jump`], the dual: only an update, arbitrary initial state.
    /// - [`Atom::constant`], for creating individual constant atoms.
    pub fn constant<'a, V, W>(vars: V, init: W) -> Result<Self, String>
    where
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<I>>,
        S: 'a,
    {
        let atom = Atom::constant(vars, init)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **combinatorial module**: a stateless, time-independent
    /// relationship between observable wires.
    ///
    /// A combinatorial module specifies a single block of assignments
    /// computing the outputs from the inputs within the same time step. It is
    /// composed of a single [`Atom::combinatorial`] atom, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module, each bound to its name.
    /// - `assign`: The set of combinatorial assignment terms defining how the
    ///   output is computed from the input.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], the stateful dual with explicit time evolution.
    /// - [`Atom::combinatorial`], for creating individual combinatorial atoms.
    pub fn combinatorial<'a, T, V, W>(vars: V, assign: W) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        V: IntoIterator<Item = (Var<S>, &'a Wire3<S>)>,
        W: IntoIterator<Item = Term<T>>,
        S: 'a,
    {
        let atom = Atom::combinatorial(vars, assign)?;
        Self::observable(std::iter::once(atom))
    }
}
