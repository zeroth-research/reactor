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
// Type-checking
//============================================================

/// Fails unless a term of the instruction `itype` may read the wires `read`
/// and write the wires `write`: unless they are of the sorts of its domain
/// and of its codomain, in order. [`Term::function`] and [`Term::constant`]
/// check the terms they build.
pub fn check<T, S>(itype: &T, read: &[Wire<S>], write: &[Wire<S>]) -> Result<(), String>
where
    T: Boundary<S> + fmt::Display,
    S: Clone + PartialEq + fmt::Display,
{
    let sorts = |wires: &[Wire<S>]| -> Poly<S> {
        wires
            .iter()
            .map(|w| w.dtype().clone())
            .collect::<Mono<S>>()
            .into()
    };
    let (dom, read) = (itype.dom(), sorts(read));
    if read != dom {
        return Err(format!("expected {itype} to read {dom}, got {read}"));
    }
    let (cod, write) = (itype.cod(), sorts(write));
    if write != cod {
        return Err(format!("expected {itype} to write {cod}, got {write}"));
    }
    Ok(())
}

//============================================================
// Public constructors
//============================================================

impl<T, S> Term<T, S>
where
    T: Boundary<S> + fmt::Display,
    S: Clone + PartialEq + fmt::Display,
{
    /// Fails if the term reads a wire it also writes, or unless its wires
    /// are of the sorts of `itype` (see [`check`]).
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
        check(&itype, &read, &write)?;

        Ok(Self::new_unchecked(itype, write, read))
    }

    /// Fails unless `itype` reads nothing and writes wires of the sorts of
    /// `write` (see [`check`]).
    pub fn constant<D, W>(itype: T, write: W) -> Result<Self, String>
    where
        D: Into<Wire<S>>,
        W: IntoIterator<Item = D>,
    {
        let write: Vec<Wire<S>> = write.into_iter().map(Into::into).collect();
        check(&itype, &[], &write)?;
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
    /// re-validation is needed.
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

#[cfg(test)]
mod tests {
    use super::*;
    use theory::tensor::{Havoc, real, sort};

    /// A tensor carrying its shape only, enough to compute boundaries.
    #[derive(Debug)]
    struct Mat([usize; 2]);

    impl theory::Tensor for Mat {
        fn shape(&self) -> [usize; 2] {
            self.0
        }
    }

    impl fmt::Display for Mat {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let [i, j] = self.0;
            write!(f, "Mat({i}, {j})")
        }
    }

    type Any = theory::tensor::Any<Mat>;

    fn wire(sort: impl Into<sort::Any>) -> Wire<sort::Any> {
        Wire::new(sort.into())
    }

    #[test]
    fn terms_read_and_write_the_sorts_of_their_operation() {
        let add = || Any::Real(real::Add::new([2, 3]).into());
        let (r, b) = (sort::Real::new([2, 3]), sort::Bool::new([2, 3]));
        assert!(Term::function(add(), [wire(r)], [wire(r), wire(r)]).is_ok());
        assert_eq!(
            Term::function(add(), [wire(r)], [wire(b), wire(b)]).unwrap_err(),
            "expected Add to read Real(2, 3) ⊗ Real(2, 3), got Bool(2, 3) ⊗ Bool(2, 3)"
        );
        assert!(Term::function(add(), [wire(r)], [wire(r)]).is_err());
        assert_eq!(
            Term::function(add(), [wire(sort::Int::new([2, 3]))], [wire(r), wire(r)]).unwrap_err(),
            "expected Add to write Real(2, 3), got Int(2, 3)"
        );
    }

    #[test]
    fn constants_read_nothing_and_write_their_sort() {
        let r = sort::Real::new([1, 2]);
        let havoc = || Any::Havoc(Havoc::new(r.into()));
        assert!(Term::constant(havoc(), [wire(r)]).is_ok());
        assert_eq!(
            Term::constant(havoc(), [wire(sort::Real::new([2, 1]))]).unwrap_err(),
            "expected Havoc to write Real(1, 2), got Real(2, 1)"
        );
        assert_eq!(
            Term::function(havoc(), [wire(r)], [wire(r)]).unwrap_err(),
            "expected Havoc to read I, got Real(1, 2)"
        );

        let one = || Any::Real(real::Const::new(Mat([1, 2])).into());
        assert!(Term::constant(one(), [wire(r)]).is_ok());
        assert_eq!(
            Term::constant(one(), [wire(r), wire(r)]).unwrap_err(),
            "expected Mat(1, 2) to write Real(1, 2), got Real(1, 2) ⊗ Real(1, 2)"
        );
    }
}
