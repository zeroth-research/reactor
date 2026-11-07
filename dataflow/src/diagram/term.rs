use super::Wire;
use std::borrow::Cow;
use std::fmt;
use theory::{Theory, any};

/// A single term corresponds to a single instruction
/// and has an input (`read`) and output (`write`).
///
/// Terms can be over different instruction sets (e.g., pytorch/linear guarded commands).
///
/// A list of terms represents a compute graph. A term is a node in the graph,
/// and it references the input/output edges (read/write wires).
/// `Interface`s are essentially single static assignments.
#[derive(Debug, Clone)]
pub struct Term<T: Theory> {
    /// The instruction to be executed by this node.
    itype: T,
    /// The outputs of this term.
    write: Vec<Wire<T::Sort>>,
    /// The inputs to this term.
    read: Vec<Wire<T::Sort>>,
}

impl<T: Theory> Term<T> {
    fn new_unchecked(itype: T, write: Vec<Wire<T::Sort>>, read: Vec<Wire<T::Sort>>) -> Self {
        Self { itype, write, read }
    }

    pub fn itype(&self) -> &T {
        &self.itype
    }

    pub fn write(&self) -> &[Wire<T::Sort>] {
        self.write.as_slice()
    }

    pub fn read(&self) -> &[Wire<T::Sort>] {
        self.read.as_slice()
    }
}

//============================================================
// Public constructors
//============================================================

impl<T> Term<T>
where
    T: Theory,
    T::Sort: Clone,
{
    pub fn function<D, U, W, R>(itype: T, write: W, read: R) -> Result<Self, String>
    where
        D: Into<Wire<T::Sort>>,
        U: Into<Wire<T::Sort>>,
        W: IntoIterator<Item = D>,
        R: IntoIterator<Item = U>,
    {
        let read: Vec<Wire<T::Sort>> = read.into_iter().map(Into::into).collect();
        let write: Vec<Wire<T::Sort>> = write.into_iter().map(Into::into).collect();

        if read.iter().any(|i| write.iter().any(|j| i == j)) {
            return Err("Term reads and writes the same wire".into());
        }

        // type-check the term. We do it only after contruction of the term, because type-checking
        // would consume the values of `write` and `read` otherwise
        let ok = Result::<_, String>::Ok;
        let r = read.iter().cloned().map(Into::into).map(ok);
        let w = write.iter().cloned().map(Into::into).map(ok);
        itype.check(r, w)?;

        Ok(Self::new_unchecked(itype, write, read))
    }

    pub fn constant<D, W>(itype: T, write: W) -> Result<Self, String>
    where
        D: Into<Wire<T::Sort>>,
        W: IntoIterator<Item = D>,
    {
        let write: Vec<Wire<T::Sort>> = write.into_iter().map(Into::into).collect();

        // type-check the term. We do it only after contruction of the term, because type-checking
        // would consume the values of `write` and `read` otherwise
        let ok = Result::<_, String>::Ok;
        let w = write.iter().cloned().map(Into::into).map(ok);
        itype.check(std::iter::empty(), w)?;

        Ok(Self::new_unchecked(itype, write, Vec::new()))
    }
}

//============================================================
// Cast operators
//============================================================

impl<T: Theory> Term<T> {
    /// Re-types a term along an infallible theory embedding (`U: Into<T>`).
    ///
    /// Only the instruction is converted; the wires are untouched, so a term
    /// that was well-formed over `U` stays well-formed over `T` and no
    /// re-validation is needed.
    pub(super) fn convert<U>(term: Term<U>) -> Self
    where
        U: Theory<Sort = T::Sort> + Into<T>,
    {
        Self::new_unchecked(term.itype.into(), term.write, term.read)
    }

    /// Re-types a term along a fallible theory cast (`U: TryInto<T>`),
    /// failing exactly when the instruction does not belong to `T`.
    ///
    /// Only the instruction is converted; the wires are untouched, so a term
    /// that was well-formed over `U` stays well-formed over `T` and no
    /// re-validation is needed.
    fn try_convert<U, E>(term: Term<U>) -> Result<Self, E>
    where
        U: Theory<Sort = T::Sort> + TryInto<T, Error = E>,
    {
        let itype = term.itype.try_into()?;
        Ok(Self::new_unchecked(itype, term.write, term.read))
    }
}

impl From<Term<any::Combinatorial>> for Term<any::Any> {
    fn from(term: Term<any::Combinatorial>) -> Self {
        Self::convert(term)
    }
}

impl From<Term<any::Sequential>> for Term<any::Any> {
    fn from(term: Term<any::Sequential>) -> Self {
        Self::convert(term)
    }
}

impl From<Term<any::Differential>> for Term<any::Any> {
    fn from(term: Term<any::Differential>) -> Self {
        Self::convert(term)
    }
}

impl TryFrom<Term<any::Any>> for Term<any::Combinatorial> {
    type Error = String;
    fn try_from(term: Term<any::Any>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

impl TryFrom<Term<any::Any>> for Term<any::Sequential> {
    type Error = String;
    fn try_from(term: Term<any::Any>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

impl TryFrom<Term<any::Any>> for Term<any::Differential> {
    type Error = String;
    fn try_from(term: Term<any::Any>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

//============================================================
// Display routines
//============================================================

pub struct Display<'a, T, N>
where
    T: Theory + 'a,
    N: Fn(&Wire<T::Sort>) -> Cow<'a, str>,
{
    term: &'a Term<T>,
    name: N,
}

impl<'a, T: Theory + 'a, N> Display<'a, T, N>
where
    T: Theory + 'a,
    N: Fn(&Wire<T::Sort>) -> Cow<'a, str>,
{
    fn fmt_tuple<I>(&self, f: &mut fmt::Formatter<'_>, iter: I) -> fmt::Result
    where
        I: Iterator<Item = &'a Wire<T::Sort>>,
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
        I: Iterator<Item = &'a Wire<T::Sort>> + ExactSizeIterator,
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

impl<'a, T, N> fmt::Display for Display<'a, T, N>
where
    T: Theory + 'a + fmt::Display,
    N: Fn(&Wire<T::Sort>) -> Cow<'a, str>,
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

impl<T> Term<T>
where
    T: Theory + fmt::Display,
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
    ) -> Display<'a, T, impl Fn(&Wire<T::Sort>) -> Cow<'a, str>>
    where
        N: Fn(&Wire<T::Sort>) -> R,
        R: Into<Cow<'a, str>>,
        T: 'a,
    {
        Display {
            term: self,
            name: move |w: &Wire<T::Sort>| name(w).into(),
        }
    }
}
