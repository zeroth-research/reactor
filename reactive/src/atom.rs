use base::Wire;
use dataflow::diagram::Block;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

use crate::var::{AtomInterface, Var};
use base::tangent::Tangent;
use std::fmt::Debug;

//============================================================
// Atom
//============================================================

/// This data structure corresponds to the atom of reactive modules.
///
/// An atom holds no wires of its own: it names its variables in its
/// interfaces and binds them to the boundaries of its blocks by position (see
/// [`Atom::new`]).
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
    /// Constructs an atom from its interfaces and its blocks.
    ///
    /// The atom binds the variables of its interfaces to the boundaries of
    /// its blocks by position, in the order of the interfaces (of the names):
    /// - `init` maps the next values of `wait` to the next values of `ctrl`,
    ///   `X(wait) -> X(ctrl)`;
    /// - `update` maps the latched values of `read`, followed by the next
    ///   values of `wait`, to the next values of `ctrl`,
    ///   `read ++ X(wait) -> X(ctrl)`;
    /// - `delay` maps the latched values of `read`, followed by the
    ///   derivatives of `wait`, to the derivatives of `ctrl`,
    ///   `read ++ d(wait) -> d(ctrl)`.
    ///
    /// The latched and next values of a variable have its sort, and its
    /// derivative has the tangent of the sort. The atom knows nothing of the
    /// wires of its blocks: a block that does not use an input still takes it
    /// (see [`Block::new`]).
    ///
    /// # Errors
    /// If a variable is both controlled and awaited, a name has different
    /// sorts in different interfaces, or the boundary of a block does not
    /// have the sorts above, position by position.
    pub fn new(
        ctrl: AtomInterface<S>,
        wait: AtomInterface<S>,
        read: AtomInterface<S>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
    ) -> Result<Self, String> {
        if let Some(var) = ctrl.iter().find(|var| wait.contains(var)) {
            return Err(format!(
                "Variable {} is both controlled and awaited",
                var.name()
            ));
        }
        // a name is one variable, of one sort, in all the interfaces
        let vars = ctrl.iter().chain(wait.iter()).chain(read.iter());
        AtomInterface::try_from_iter(vars.cloned())?;

        check_boundary(
            "Init",
            &init,
            sorts(&wait).collect(),
            sorts(&ctrl).collect(),
        )?;
        check_boundary(
            "Update",
            &update,
            sorts(&read).chain(sorts(&wait)).collect(),
            sorts(&ctrl).collect(),
        )?;
        check_boundary(
            "Delay",
            &delay,
            sorts(&read).chain(tangents(&wait)).collect(),
            tangents(&ctrl).collect(),
        )?;

        Ok(Self {
            ctrl,
            wait,
            read,
            init,
            update,
            delay,
        })
    }
}

/// The sorts of the variables, in order.
fn sorts<S: Clone>(vars: &AtomInterface<S>) -> impl Iterator<Item = S> + '_ {
    vars.iter().map(|var| var.dtype().clone())
}

/// The tangents of the sorts of the variables, in order: the sorts of their
/// derivatives.
fn tangents<S: Tangent>(vars: &AtomInterface<S>) -> impl Iterator<Item = S> + '_ {
    vars.iter().map(|var| var.dtype().tangent())
}

/// Checks that the boundary of the block has the sorts `dom -> cod`, position
/// by position.
fn check_boundary<T, S>(
    name: &str,
    block: &Block<T>,
    dom: Vec<S>,
    cod: Vec<S>,
) -> Result<(), String>
where
    T: Theory<Sort = S>,
    S: Clone + PartialEq + Debug,
{
    let sorts = |wires: &[Wire<S>]| wires.iter().map(|w| w.dtype().clone()).collect::<Vec<S>>();
    let (read, write) = (sorts(block.read()), sorts(block.write()));
    if read == dom && write == cod {
        return Ok(());
    }
    Err(format!(
        "{} has the boundary {:?} -> {:?}, but the interfaces expect {:?} -> {:?}",
        name, read, write, dom, cod
    ))
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
