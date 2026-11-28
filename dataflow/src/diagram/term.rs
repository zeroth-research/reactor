use super::Wire;
use base::boundary::{Boundary, Mono, Poly};
use std::borrow::Cow;
use std::fmt;

/// A single term corresponds to a single instruction
/// and has an input (`read`) and output (`write`).
///
/// Terms can be over different instruction sets (e.g., pytorch/linear guarded commands).
///
/// A list of terms represents a compute graph. A term is a node in the graph,
/// and it references the input/output edges (read/write wires).
/// `Interface`s are essentially single static assignments.
#[derive(Debug, Clone)]
pub struct Term<T, S> {
    /// The instruction to be executed by this node.
    itype: T,
    /// The outputs of this term.
    write: Vec<Wire<S>>,
    /// The inputs to this term.
    read: Vec<Wire<S>>,
}

impl<T, S> Term<T, S> {
    fn new_unchecked(itype: T, write: Vec<Wire<S>>, read: Vec<Wire<S>>) -> Self {
        Self { itype, write, read }
    }

    pub fn itype(&self) -> &T {
        &self.itype
    }

    pub fn write(&self) -> &[Wire<S>] {
        self.write.as_slice()
    }

    pub fn read(&self) -> &[Wire<S>] {
        self.read.as_slice()
    }
}

//============================================================
// Public constructors
//============================================================

impl<T, S> Term<T, S>
where
    T: Boundary<S>,
    S: Clone + Eq + fmt::Debug,
{
    /// Fails if the term reads a wire it also writes, or unless the sorts of
    /// the wires it reads are the domain of `itype`, and those of the wires
    /// it writes its codomain, in order.
    pub fn function<D, U, W, R>(itype: T, write: W, read: R) -> Result<Self, String>
    where
        D: Into<Wire<S>>,
        U: Into<Wire<S>>,
        W: IntoIterator<Item = D>,
        R: IntoIterator<Item = U>,
    {
        let read: Vec<Wire<S>> = read.into_iter().map(Into::into).collect();
        let write: Vec<Wire<S>> = write.into_iter().map(Into::into).collect();

        if read.iter().any(|i| write.iter().any(|j| i == j)) {
            return Err("Term reads and writes the same wire".into());
        }

        let dom: Poly<S> = Mono::from_iter(read.iter().map(|w| w.dtype().clone())).into();
        if dom != itype.dom() {
            return Err(format!(
                "Term reads wires of sorts {dom:?}, but its instruction has domain {:?}",
                itype.dom()
            ));
        }
        let cod: Poly<S> = Mono::from_iter(write.iter().map(|w| w.dtype().clone())).into();
        if cod != itype.cod() {
            return Err(format!(
                "Term writes wires of sorts {cod:?}, but its instruction has codomain {:?}",
                itype.cod()
            ));
        }

        Ok(Self::new_unchecked(itype, write, read))
    }

    /// Fails unless `itype` reads nothing, its domain the unit `I`, and the
    /// sorts of the wires it writes are its codomain, in order. A term that
    /// reads nothing cannot read a wire it writes.
    pub fn constant<D, W>(itype: T, write: W) -> Result<Self, String>
    where
        D: Into<Wire<S>>,
        W: IntoIterator<Item = D>,
    {
        let write: Vec<Wire<S>> = write.into_iter().map(Into::into).collect();

        if !itype.dom().is_unit() {
            return Err(format!(
                "Term reads no wires, but its instruction has domain {:?}",
                itype.dom()
            ));
        }
        let cod: Poly<S> = Mono::from_iter(write.iter().map(|w| w.dtype().clone())).into();
        if cod != itype.cod() {
            return Err(format!(
                "Term writes wires of sorts {cod:?}, but its instruction has codomain {:?}",
                itype.cod()
            ));
        }
        Ok(Self::new_unchecked(itype, write, Vec::new()))
    }
}

//============================================================
// Cast operators
//============================================================

impl<T, S> Term<T, S> {
    /// Re-types a term along an infallible theory embedding (`U: Into<T>`).
    ///
    /// Only the instruction is converted; the wires are untouched, so a term
    /// that was well-formed over `U` stays well-formed over `T` and no
    /// re-validation is needed, provided the embedding keeps the boundary of
    /// each instruction, as the embeddings of the logics into `theory::Op` do.
    pub(super) fn convert<U>(term: Term<U, S>) -> Self
    where
        U: Into<T>,
    {
        Self::new_unchecked(term.itype.into(), term.write, term.read)
    }
}

//============================================================
// Display routines
//============================================================

pub struct Display<'a, T, S, N>
where
    N: Fn(&Wire<S>) -> Cow<'a, str>,
{
    term: &'a Term<T, S>,
    name: N,
}

impl<'a, T, S, N> Display<'a, T, S, N>
where
    N: Fn(&Wire<S>) -> Cow<'a, str>,
{
    fn fmt_tuple<I>(&self, f: &mut fmt::Formatter<'_>, iter: I) -> fmt::Result
    where
        I: Iterator<Item = &'a Wire<S>>,
    {
        write!(f, "(")?;
        for (i, w) in iter.enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", (self.name)(w))?;
        }
        write!(f, ")")
    }

    fn fmt_intf<I>(&self, f: &mut fmt::Formatter<'_>, iter: I) -> fmt::Result
    where
        I: Iterator<Item = &'a Wire<S>> + ExactSizeIterator,
    {
        if iter.len() == 1 {
            let mut iter = iter;
            write!(f, "{}", (self.name)(iter.next().unwrap()))?;
        } else {
            self.fmt_tuple(f, iter)?;
        }
        Ok(())
    }
}

impl<'a, T, S, N> fmt::Display for Display<'a, T, S, N>
where
    T: fmt::Display,
    N: Fn(&Wire<S>) -> Cow<'a, str>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let term = self.term;
        self.fmt_intf(f, term.write.iter())?;
        write!(f, " := {} ", term.itype)?;
        if !term.read.is_empty() {
            self.fmt_intf(f, term.read.iter())?;
        }
        Ok(())
    }
}

impl<T, S> Term<T, S>
where
    T: fmt::Display,
{
    /// Displays the term with wires under the names the given function
    /// assigns. Covariance lets `'a` shrink to a caller-local buffer: the
    /// caller keeps ownership of the names and passes references on.
    /// Displays the term with wires under the names the given function
    /// assigns. The function may return anything that converts into a
    /// `Cow<'a, str>` -- `&'a str`, `String`, or a `Cow` itself -- and the
    /// conversion happens once, here.
    pub fn with_wirenames<'a, N, R>(
        &'a self,
        name: N,
    ) -> Display<'a, T, S, impl Fn(&Wire<S>) -> Cow<'a, str>>
    where
        N: Fn(&Wire<S>) -> R,
        R: Into<Cow<'a, str>>,
        T: 'a,
    {
        Display {
            term: self,
            name: move |w: &Wire<S>| name(w).into(),
        }
    }
}
