use crate::atom::Atom;
use crate::topological_order;
use crate::var::{ModuleInterface, Var};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;
use theory::Theory;

//============================================================
// Module
//============================================================

/// This data structure corresponds to the module of reactive modules.
#[derive(Debug, Clone)]
pub struct Module<T, S = <T as Theory>::Sort>
where
    T: Theory<Sort = S>,
{
    /// The variables of the module divided by visibility
    /// ```text
    ///     *====================*
    ///     | extl | intf | prvt |
    ///     *--------------------*
    ///     |     obs     | prvt |
    ///     *--------------------*
    ///     | extl |    ctrl     |
    ///     *====================*
    /// ```
    extl: ModuleInterface<S>,
    intf: ModuleInterface<S>,
    prvt: ModuleInterface<S>,
    obs: ModuleInterface<S>,
    ctrl: ModuleInterface<S>,

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// according to their await relation.
    atoms: Vec<Atom<T, S>>,
}

impl<T, S> Module<T, S>
where
    T: Theory<Sort = S>,
{
    pub fn atoms(&self) -> &[Atom<T, S>] {
        &self.atoms
    }

    pub fn extl(&self) -> &ModuleInterface<S> {
        &self.extl
    }

    pub fn intf(&self) -> &ModuleInterface<S> {
        &self.intf
    }

    pub fn prvt(&self) -> &ModuleInterface<S> {
        &self.prvt
    }

    pub fn ctrl(&self) -> &ModuleInterface<S> {
        &self.ctrl
    }

    pub fn obs(&self) -> &ModuleInterface<S> {
        &self.obs
    }

    pub fn is_closed(&self) -> bool {
        self.extl.is_empty()
    }

    pub fn is_open(&self) -> bool {
        !self.extl.is_empty()
    }

    pub fn empty() -> Self {
        Module {
            extl: ModuleInterface::empty(),
            intf: ModuleInterface::empty(),
            prvt: ModuleInterface::empty(),
            obs: ModuleInterface::empty(),
            ctrl: ModuleInterface::empty(),
            atoms: Vec::new(),
        }
    }
}

//============================================================
// Private routines
//============================================================

impl<T, S> Module<T, S>
where
    T: Theory<Sort = S>,
    S: Debug,
{
    /// Constructs a module **without performing any consistency or visibility checks**.
    ///
    /// This constructor provides **full control** to the caller and performs no inference
    /// or validation. It should be used only when all necessary checks or automation
    /// have already been handled externally.
    ///
    /// Unlike the other constructors, which automatically infer variable visibility and module
    /// properties, `new_unchecked` allows manually specifying all variable sets and atoms.
    /// This is useful for advanced scenarios or for optimising performance when redundant
    /// automation would otherwise occur.
    ///
    /// # Variable layout and visibility
    ///
    /// Variable visibility and organization within a module can be visualized as follows:
    ///
    /// ```text
    ///     *====================*
    ///     | extl | intf | prvt |
    ///     *--------------------*
    ///     |     obs     | prvt |
    ///     *--------------------*
    ///     | extl |    ctrl     |
    ///     *====================*
    /// ```
    /// # Parameters
    /// - `extl` are external variables, exposed to the environment (module inputs).
    /// - `intf` are interface variables, forming the module’s public outputs.
    /// - `prvt` are private variables, hidden from external access.
    /// - `obs` are observable variables, visible through the module interface.
    /// - `ctrl` are controlled variables, used for state management and internal coordination.
    /// - `atoms`: The list of atoms defining the module’s internal behaviour.
    ///
    /// # Returns
    /// The constructed module.
    ///
    /// # Correctness
    /// This function performs **no validation or inference** (beyond
    /// debug-only assertions). It is the caller’s responsibility to ensure
    /// that all variables, atoms, and interfaces are well-formed and consistent.
    ///
    /// # See Also
    /// - [`Atom::new`] for creating individual atoms.
    /// - [`Module::partially_observable`], [`Module::observable`] for checked,
    ///   automated module construction
    fn new_unchecked(
        extl: ModuleInterface<S>,
        intf: ModuleInterface<S>,
        prvt: ModuleInterface<S>,
        obs: ModuleInterface<S>,
        ctrl: ModuleInterface<S>,
        atoms: Vec<Atom<T, S>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(obs.len(), extl.len() + intf.len());
            debug_assert_eq!(ctrl.len(), intf.len() + prvt.len());

            let mut module_vars: HashSet<&Var<S>> = HashSet::new();

            // extl & intf & prvt = empty
            debug_assert!(extl.iter().all(|v| module_vars.insert(v)));
            debug_assert!(intf.iter().all(|v| module_vars.insert(v)));
            debug_assert!(prvt.iter().all(|v| module_vars.insert(v)));

            // obs U ctrl <= extl U intf U prvt
            debug_assert!(obs.iter().all(|v| module_vars.contains(v)));
            debug_assert!(ctrl.iter().all(|v| module_vars.contains(v)));
            // obs <= extl U intf and ctrl <= intf U prvt
            debug_assert!(obs.iter().all(|v| extl.contains(v) || intf.contains(v)));
            debug_assert!(ctrl.iter().all(|v| intf.contains(v) || prvt.contains(v)));

            // check atoms consistency
            let mut written: HashSet<&Var<S>> = HashSet::new();
            written.extend(extl.iter());
            for atom in atoms.iter() {
                for var in atom.read().iter() {
                    debug_assert!(module_vars.contains(var));
                }
                for var in atom.wait().iter() {
                    debug_assert!(module_vars.contains(var));
                    debug_assert!(written.contains(var));
                }
                for var in atom.ctrl().iter() {
                    debug_assert!(module_vars.contains(var));
                    debug_assert!(written.insert(var));
                }
            }

            // check that all module control vars are written/controlled by an atom
            for var in ctrl.iter() {
                debug_assert!(written.contains(var));
            }

            // check vars consistency
            let mut atom_decl_vars = HashSet::new();
            atom_decl_vars.extend(atoms.iter().flat_map(Atom::read));
            atom_decl_vars.extend(atoms.iter().flat_map(Atom::ctrl));
            atom_decl_vars.extend(atoms.iter().flat_map(Atom::wait));
            debug_assert_eq!(atom_decl_vars, module_vars);
        }

        Module {
            extl,
            intf,
            prvt,
            obs,
            ctrl,
            atoms,
        }
    }
}

//============================================================
// Public constructors
//============================================================

impl<T, S> Module<T, S>
where
    T: Theory<Sort = S>,
    S: Clone + PartialEq + Debug,
{
    /// Constructs a **fully observable module** from a set of atoms.
    ///
    /// A fully observable module exposes all of its variables publicly, so
    /// that no internal state remains hidden. The variables and their roles
    /// (external or interface) are inferred entirely from the atoms: the
    /// controlled variables become the interface, the remaining read or
    /// awaited variables are external. Equivalent to
    /// [`Module::partially_observable`] with nothing hidden.
    ///
    /// The atoms must be presented in a linear order consistent with their
    /// **await relation**: an atom that awaits a variable (reads its next
    /// value within the round) must come after the atom controlling it. The
    /// given sequence is kept as the module's execution order — it is
    /// checked, not sorted — and an inconsistent order is rejected.
    ///
    /// # Parameters
    /// - `atoms`: An iterable collection of atoms defining the module's
    ///   behaviour, ordered consistently with their await relation.
    ///
    /// # Returns
    /// A `Result` containing the constructed fully observable module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::partially_observable`], for modules with private state.
    /// - [`Atom::new`] for creating individual atoms.
    pub fn observable<A>(atoms: A) -> Result<Self, String>
    where
        A: IntoIterator<Item = Atom<T, S>> + Sized,
    {
        Self::partially_observable(atoms, |_| false)
    }

    /// Constructs a **partially observable module** from a sequence of atoms,
    /// hiding the given variables.
    ///
    /// The variables and their roles are inferred from the atoms — the
    /// controlled variables form the interface, the remaining read or awaited
    /// variables are external — and the `hide` variables are then moved
    /// from the interface into the private state. Only controlled variables
    /// can be hidden: privates must be controlled by the module.
    ///
    /// The atoms must be presented in a linear order consistent with their
    /// **await relation**: an atom that awaits a variable (reads its next
    /// value within the round) must come after the atom controlling it. The
    /// given sequence is kept as the module's execution order — it is
    /// checked, not sorted — and an inconsistent order is rejected.
    ///
    /// # Parameters
    /// - `atoms`: An iterable collection of atoms defining the module's
    ///   behaviour, ordered consistently with their await relation.
    /// - `hide`: The variables to hide from the environment; hiding nothing
    ///   yields a fully observable module.
    ///
    /// # Returns
    /// A `Result` containing the constructed partially observable module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::observable`], the fully observable special case.
    /// - [`Atom::new`] for creating individual atoms.
    pub fn partially_observable<A, H>(atoms: A, hide: H) -> Result<Self, String>
    where
        A: IntoIterator<Item = Atom<T, S>> + Sized,
        H: Fn(&Var<S>) -> bool,
    {
        let mut sorts: HashMap<u32, S> = HashMap::new();
        let mut obs: BTreeSet<Var<S>> = BTreeSet::new();
        let mut ctrl: BTreeSet<Var<S>> = BTreeSet::new();
        let atoms = atoms.into_iter();
        let mut past_atoms: Vec<Atom<T, S>> = Vec::with_capacity(atoms.size_hint().0);

        for (n, atom) in atoms.enumerate() {
            //============================================================
            // Collect variables
            //============================================================
            let vars = atom.read().iter().chain(atom.wait()).chain(atom.ctrl());
            for var in vars {
                check_sort(&mut sorts, var)?;
            }
            obs.extend(atom.read().iter().cloned());
            obs.extend(atom.wait().iter().cloned());
            obs.extend(atom.ctrl().iter().cloned());

            //============================================================
            // Determine controlled
            //============================================================
            for var in atom.ctrl().iter() {
                if !ctrl.insert(var.clone()) {
                    return Err(format!("Variable {:?} is doubly controlled", var));
                }
            }

            //============================================================
            // Check await order
            //============================================================
            for past_atom in past_atoms.iter() {
                if past_atom.awaits(&atom) {
                    return Err(format!("{}-th atom is in inconsistent awaiting order", n));
                }
            }
            past_atoms.push(atom);
        }

        //============================================================
        // Determine private, interface, and external
        //============================================================
        let prvt: Vec<Var<S>> = obs.extract_if(.., &hide).collect();
        let intf = ctrl.iter().filter(|v| prvt.binary_search(v).is_err());
        let extl = obs.difference(&ctrl);

        for var in prvt.iter() {
            if !ctrl.contains(var) {
                return Err(format!("Hiding uncontrolled variable {:?}", var));
            }
        }

        let extl = ModuleInterface::from_sorted_unchecked(extl.cloned().collect());
        let intf = ModuleInterface::from_sorted_unchecked(intf.cloned().collect());
        let prvt = ModuleInterface::from_sorted_unchecked(prvt.into_iter().collect());
        let obs = ModuleInterface::from_sorted_unchecked(obs.into_iter().collect());
        let ctrl = ModuleInterface::from_sorted_unchecked(ctrl.into_iter().collect());

        Ok(Self::new_unchecked(extl, intf, prvt, obs, ctrl, past_atoms))
    }

    /// Constructs the **pure parallel composition** of several modules.
    ///
    /// This function takes an iterator of modules and returns a new module
    /// that represents all of them composed in parallel, coupling all shared
    /// observable variables. It is the special case of the *hiding
    /// composition* [`Module::compose_hiding`] that hides nothing: every observable
    /// of the components stays observable in the composite.
    ///
    /// # Semantics
    ///
    /// Observable variables shared across modules are *coupled* in the
    /// composed module: they represent the same value in the resulting
    /// system.
    ///
    /// # Error Conditions
    ///
    /// - A module attempts to couple a *private* variable with another module
    /// - Coupled variables (of the same name) have different sorts
    /// - A coupled variable is *controlled by more than one module*
    /// - Await dependency is cyclic
    ///
    /// # Returns
    ///
    /// - `Ok(Module)` containing the composed module.
    /// - `Err(Error)` describing the reason composition failed.
    ///
    /// # See Also
    /// - [`Module::compose_hiding`], the general form that also hides variables.
    pub fn compose<M>(modules: M) -> Result<Self, String>
    where
        M: IntoIterator<Item = Self>,
    {
        Self::compose_hiding(modules, |_| false)
    }

    /// The **hiding operator**: hides the given variables of this module.
    ///
    /// Consumes the module and returns one with the same behaviour in which
    /// the `hide` variables have left the observables and become private.
    /// It is the hiding composition [`Module::compose_hiding`] of the
    /// module alone, and composes with it: hiding after composing equals
    /// hiding at composition.
    ///
    /// Only interface variables can be hidden: hiding an external variable
    /// is rejected, since privates must be controlled.
    ///
    /// # Parameters
    /// - `hide`: The variables to hide; hiding nothing returns the module unchanged.
    ///
    /// # Returns
    ///
    /// - `Ok(Module)` containing the module with the variables hidden.
    /// - `Err(Error)` describing the reason hiding failed.
    ///
    /// # See Also
    /// - [`Module::compose_hiding`], hiding several modules at their composition.
    pub fn hide<H>(self, hide: H) -> Result<Self, String>
    where
        H: Fn(&Var<S>) -> bool,
    {
        Self::compose_hiding(std::iter::once(self), hide)
    }

    /// Constructs the **hiding composition** of several modules: parallel
    /// composition that simultaneously hides the given variables.
    ///
    /// The modules are composed as in [`Module::compose`], coupling all shared
    /// observable variables; the `hide` variables are then removed from the
    /// composite's observables and become private. The coupling still takes
    /// place — hiding restricts the visibility of the *composite*, not the
    /// communication between the components.
    ///
    /// Only variables controlled by one of the components can be hidden:
    /// hiding an external variable is rejected, since privates must be
    /// controlled.
    ///
    /// # Parameters
    /// - `modules`: The modules to compose.
    /// - `hide`: The variables to hide in the composite; hiding nothing is
    ///   exactly [`Module::compose`].
    ///
    /// # Error Conditions
    ///
    /// All error conditions of [`Module::compose`], and additionally:
    ///
    /// - A prvt variable is external to the composition
    ///
    /// # Returns
    ///
    /// - `Ok(Module)` containing the composed module.
    /// - `Err(Error)` describing the reason composition failed.
    pub fn compose_hiding<M, H>(modules: M, hide: H) -> Result<Self, String>
    where
        M: IntoIterator<Item = Self>,
        H: Fn(&Var<S>) -> bool,
    {
        let mut sorts: HashMap<u32, S> = HashMap::new();
        let mut prvt_pre_hiding: HashSet<u32> = HashSet::new();

        let mut extl: BTreeSet<Var<S>> = BTreeSet::new();
        let mut intf: BTreeSet<Var<S>> = BTreeSet::new();
        let mut prvt: BTreeSet<Var<S>> = BTreeSet::new();
        let mut obs: BTreeSet<Var<S>> = BTreeSet::new();
        let mut ctrl: BTreeSet<Var<S>> = BTreeSet::new();

        let mut atoms_pre_ordering: Vec<Atom<T, S>> = Vec::new();
        let mut await_graph: Vec<Vec<usize>> = Vec::new();

        for module in modules {
            //============================================================
            // Ensure decoupling and restrict visibility
            //============================================================

            // Check that coupled variables agree on their sorts
            for var in module.obs.iter().chain(module.prvt.iter()) {
                check_sort(&mut sorts, var)?;
            }

            // Check that observables are either uncoupled or coupled in right direction
            for var in module.obs {
                if prvt_pre_hiding.contains(&var.name()) {
                    return Err(format!("Coupling private variable {:?}", var));
                }

                // visit every observable no more than once, and skip otherwise
                if !obs.contains(&var) & !prvt.contains(&var) {
                    // hidden variables leave the observables and join the privates;
                    // we check at the end whether this has affected uncoupled externals
                    // hide() is called exactly once (no more no less) on each observable
                    if hide(&var) {
                        prvt.insert(var);
                    } else {
                        obs.insert(var);
                    }
                }
            }

            // Check that privates are uncoupled and restrict them
            for var in module.prvt {
                // visit every private no more than once, and raise otherwise
                if obs.contains(&var) | prvt.contains(&var) {
                    return Err(format!("Coupling private variable {:?}", var));
                }

                debug_assert!(!prvt_pre_hiding.contains(&var.name()));
                prvt_pre_hiding.insert(var.name());

                prvt.insert(var);
            }

            //============================================================
            // Couple external and interface variables
            //============================================================
            for var in module.extl {
                // external variables stay external only if they are not
                // deemed controlled by modules visited before
                if !ctrl.contains(&var) {
                    extl.insert(var);
                }
            }

            for var in module.intf {
                // interface variables necessarily leave the set of variables
                // deemed external by modules visited before
                extl.remove(&var);

                if ctrl.contains(&var) {
                    return Err(format!("Variable {:?} is doubly controlled", var));
                }

                if !prvt.contains(&var) {
                    intf.insert(var);
                }
            }

            ctrl.extend(module.ctrl);

            //============================================================
            // Populate await graph
            //============================================================
            for this_atom in module.atoms {
                let this_idx = atoms_pre_ordering.len();
                let mut this_adj = Vec::new();
                for (other_idx, other_atom) in atoms_pre_ordering.iter().enumerate() {
                    if this_atom.awaits(other_atom) {
                        await_graph[other_idx].push(this_idx);
                    }
                    if other_atom.awaits(&this_atom) {
                        this_adj.push(other_idx);
                    }
                }
                atoms_pre_ordering.push(this_atom);
                await_graph.push(this_adj);
            }
        }

        //============================================================
        // Check that no hidden variables are externals
        //============================================================

        for var in prvt.iter() {
            if extl.contains(var) {
                return Err(format!("Hiding uncontrolled var {:?}", var));
            }
        }

        //============================================================
        // Reorder atoms
        //============================================================

        let await_order = topological_order(&await_graph).ok_or("Invalid await dependency")?;
        debug_assert_eq!(await_order.len(), await_graph.len());

        let mut atoms: Vec<Atom<T, S>> = Vec::with_capacity(await_graph.len());
        for idx in await_order {
            atoms.push(std::mem::take(&mut atoms_pre_ordering[idx]));
        }

        //============================================================
        // Collect and construct
        //============================================================

        let extl = ModuleInterface::from_sorted_unchecked(extl.into_iter().collect());
        let intf = ModuleInterface::from_sorted_unchecked(intf.into_iter().collect());
        let prvt = ModuleInterface::from_sorted_unchecked(prvt.into_iter().collect());
        let obs = ModuleInterface::from_sorted_unchecked(obs.into_iter().collect());
        let ctrl = ModuleInterface::from_sorted_unchecked(ctrl.into_iter().collect());

        Ok(Module::new_unchecked(extl, intf, prvt, obs, ctrl, atoms))
    }
}

/// Checks that the variable has the sort of the variables of the same name
/// seen before, and records its sort.
fn check_sort<S>(sorts: &mut HashMap<u32, S>, var: &Var<S>) -> Result<(), String>
where
    S: Clone + PartialEq + Debug,
{
    match sorts.get(&var.name()) {
        Some(sort) if sort != var.dtype() => Err(format!(
            "Variable {} has conflicting sorts {:?} and {:?}",
            var.name(),
            sort,
            var.dtype()
        )),
        Some(_) => Ok(()),
        None => {
            sorts.insert(var.name(), var.dtype().clone());
            Ok(())
        }
    }
}

//============================================================
// Display routines
//============================================================

pub struct Display<'a, T, S, N>
where
    T: Theory<Sort = S>,
{
    module: &'a Module<T, S>,
    name: N,
}

impl<'a, T, S, N> Display<'a, T, S, N>
where
    T: Theory<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";
        const INDENT2: &str = "    ";

        let m = self.module;
        let name = &self.name;
        writeln!(f, "{pad}{BOLD}module{RESET}")?;
        if !m.extl.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}external{RESET}")?;
        }
        for var in m.extl.iter() {
            writeln!(f, "{pad}{INDENT2}{} : {}", name(var), var.dtype())?;
        }
        if !m.intf.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}interface{RESET}")?;
        }
        for var in m.intf.iter() {
            writeln!(f, "{pad}{INDENT2}{} : {}", name(var), var.dtype())?;
        }
        if !m.prvt.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}private{RESET}")?;
        }
        for var in m.prvt.iter() {
            writeln!(f, "{pad}{INDENT2}{} : {}", name(var), var.dtype())?;
        }
        for atom in &m.atoms {
            atom.with_varnames_untyped(name)
                .fmt_indent(f, &format!("{pad}{INDENT}"))?;
        }
        Ok(())
    }
}

impl<'a, T, S, N> fmt::Display for Display<'a, T, S, N>
where
    T: Theory<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}

impl<T, S> Module<T, S>
where
    T: Theory<Sort = S> + fmt::Display,
    S: fmt::Display,
{
    /// Displays the module with variables under the names the given
    /// function assigns; it is consulted for every module variable and may
    /// return anything that converts into a `Cow<'a, str>` -- `&'a str`,
    /// `String`, or a `Cow` itself.
    pub fn with_varnames<'a, N, R>(
        &'a self,
        name: N,
    ) -> Display<'a, T, S, impl Fn(&Var<S>) -> Cow<'a, str>>
    where
        N: Fn(&Var<S>) -> R,
        R: Into<Cow<'a, str>>,
    {
        Display {
            module: self,
            name: move |v: &Var<S>| name(v).into(),
        }
    }
}
