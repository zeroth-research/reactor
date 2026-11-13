use super::Wire;
use base::boundary::{Boundary, Mono, Poly};
use std::borrow::Cow;
use std::fmt;
use theory::any;

/// A single term corresponds to a single instruction
/// and has an input (`read`) and output (`write`).
///
/// Terms can be over different instruction sets (e.g., pytorch/linear guarded commands).
///
/// A list of terms represents a compute graph. A term is a node in the graph,
/// and it references the input/output edges (read/write wires).
/// `Interface`s are essentially single static assignments.
///
/// The wires carry values of the sorts `S`. A term is well-formed iff its boundary (the
/// sorts of the read and the written wires) is the boundary of its instruction; the
/// public constructors check it.
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

/// The boundary of a term is the sorts of its wires: the domain is the sorts of the read
/// wires and the codomain the sorts of the written wires, in order.
impl<T, S: Clone> Boundary<S> for Term<T, S> {
    fn dom(&self) -> Poly<S> {
        sorts(&self.read)
    }

    fn cod(&self) -> Poly<S> {
        sorts(&self.write)
    }
}

/// The sorts of `wires`, read or written all at once (i.e., a product)
fn sorts<S: Clone>(wires: &[Wire<S>]) -> Poly<S> {
    wires
        .iter()
        .map(|w| w.dtype().clone())
        .collect::<Mono<S>>()
        .into()
}

//============================================================
// Public constructors
//============================================================

impl<T, S> Term<T, S>
where
    T: Boundary<S> + fmt::Debug,
    S: Clone + PartialEq + fmt::Display,
{
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

        check(&itype, &write, &read)?;
        Ok(Self::new_unchecked(itype, write, read))
    }

    pub fn constant<D, W>(itype: T, write: W) -> Result<Self, String>
    where
        D: Into<Wire<S>>,
        W: IntoIterator<Item = D>,
    {
        let write: Vec<Wire<S>> = write.into_iter().map(Into::into).collect();

        check(&itype, &write, &[])?;
        Ok(Self::new_unchecked(itype, write, Vec::new()))
    }
}

/// Check that the instruction `itype` can write the wires `write` and read the wires
/// `read`: the term is well-formed iff its boundary (the sorts of the read and the
/// written wires) is the boundary of `itype`.
fn check<T, S>(itype: &T, write: &[Wire<S>], read: &[Wire<S>]) -> Result<(), String>
where
    T: Boundary<S> + fmt::Debug,
    S: Clone + PartialEq + fmt::Display,
{
    let (dom, cod) = (itype.dom(), itype.cod());
    let (read, write) = (sorts(read), sorts(write));
    if (&dom, &cod) != (&read, &write) {
        return Err(format!(
            "{itype:?}: expected `{dom} -> {cod}`, got `{read} -> {write}`"
        ));
    }
    Ok(())
}

//============================================================
// Cast operators
//============================================================

impl<T, S> Term<T, S> {
    /// Re-types a term along an infallible theory embedding (`U: Into<T>`).
    ///
    /// Only the instruction is converted; the wires are untouched, so a term
    /// that was well-formed over `U` stays well-formed over `T` and no
    /// re-validation is needed.
    pub(super) fn convert<U>(term: Term<U, S>) -> Self
    where
        U: Into<T>,
    {
        Self::new_unchecked(term.itype.into(), term.write, term.read)
    }

    /// Re-types a term along a fallible theory cast (`U: TryInto<T>`),
    /// failing exactly when the instruction does not belong to `T`.
    ///
    /// Only the instruction is converted; the wires are untouched, so a term
    /// that was well-formed over `U` stays well-formed over `T` and no
    /// re-validation is needed.
    fn try_convert<U, E>(term: Term<U, S>) -> Result<Self, E>
    where
        U: TryInto<T, Error = E>,
    {
        let itype = term.itype.try_into()?;
        Ok(Self::new_unchecked(itype, term.write, term.read))
    }
}

impl From<Term<any::Combinatorial, any::Sort>> for Term<any::Any, any::Sort> {
    fn from(term: Term<any::Combinatorial, any::Sort>) -> Self {
        Self::convert(term)
    }
}

impl From<Term<any::Sequential, any::Sort>> for Term<any::Any, any::Sort> {
    fn from(term: Term<any::Sequential, any::Sort>) -> Self {
        Self::convert(term)
    }
}

impl From<Term<any::Differential, any::Sort>> for Term<any::Any, any::Sort> {
    fn from(term: Term<any::Differential, any::Sort>) -> Self {
        Self::convert(term)
    }
}

impl TryFrom<Term<any::Any, any::Sort>> for Term<any::Combinatorial, any::Sort> {
    type Error = String;
    fn try_from(term: Term<any::Any, any::Sort>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

impl TryFrom<Term<any::Any, any::Sort>> for Term<any::Sequential, any::Sort> {
    type Error = String;
    fn try_from(term: Term<any::Any, any::Sort>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

impl TryFrom<Term<any::Any, any::Sort>> for Term<any::Differential, any::Sort> {
    type Error = String;
    fn try_from(term: Term<any::Any, any::Sort>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

//============================================================
// Display routines
//============================================================

pub struct Display<'a, T, S, N>
where
    T: 'a,
    S: 'a,
    N: Fn(&Wire<S>) -> Cow<'a, str>,
{
    term: &'a Term<T, S>,
    name: N,
}

impl<'a, T, S, N> Display<'a, T, S, N>
where
    T: 'a,
    S: 'a,
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
    T: 'a + fmt::Display,
    S: 'a,
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
        S: 'a,
    {
        Display {
            term: self,
            name: move |w: &Wire<S>| name(w).into(),
        }
    }
}
