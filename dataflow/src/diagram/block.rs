use super::Term;
use super::Wire;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::vec;
use theory::{Combinatorial, Differential, Sequential};

#[derive(Debug, Clone)]
pub struct Block<T, S> {
    terms: Vec<Term<T, S>>,
    read: Vec<Wire<S>>,
    write: Vec<Wire<S>>,
}

impl<T, S> Block<T, S> {
    pub fn iter(&self) -> impl Iterator<Item = &Term<T, S>> {
        self.terms.iter()
    }

    /// Returns a reference to the *read interface* of the block.
    ///
    /// The read interface lists all wires that must be provided externally
    /// for the block to operate, and are not written internally by the block.
    /// These wires are inputs required by the block as a whole.
    pub fn read(&self) -> &[Wire<S>] {
        self.read.as_slice()
    }

    /// Returns a reference to the *write interface* of the block.
    ///
    /// The write interface lists all wires that the block writes. These wires represent
    /// the outputs of the block as a whole; they can all be read outside the block.
    pub fn write(&self) -> &[Wire<S>] {
        self.write.as_slice()
    }

    /// Return a reference to the n-th term in the block
    pub fn get(&self, n: usize) -> Option<&Term<T, S>> {
        self.terms.get(n)
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn empty() -> Self {
        Self {
            terms: Vec::new(),
            read: Vec::new(),
            write: Vec::new(),
        }
    }
}

impl<'a, T, S> IntoIterator for &'a Block<T, S> {
    type Item = &'a Term<T, S>;
    type IntoIter = std::slice::Iter<'a, Term<T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.iter()
    }
}

impl<T, S> IntoIterator for Block<T, S> {
    type Item = Term<T, S>;
    type IntoIter = vec::IntoIter<Term<T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.into_iter()
    }
}

impl<T, S> Block<T, S>
where
    S: Clone + Debug,
{
    /// Builds a block from an iterator of *fallible* elements, mirroring the
    /// standard library's `FromIterator<Result<A, E>> for Result<V, E>`: the
    /// first `Err` element short-circuits and is returned as the block's
    /// error. Elements over a convertible theory (`U: Into<T>`) are embedded
    /// into `T`. Infallible sources wrap their terms with `.map(Ok)`.
    ///
    /// The error type is fixed to `String` — the crate's uniform error
    /// model — so that `Ok` needs no annotation at infallible call sites.
    pub fn try_from_iter<U, V>(iter: V) -> Result<Self, String>
    where
        U: Into<T>,
        V: IntoIterator<Item = Result<Term<U, S>, String>>,
    {
        let mut read: BTreeSet<Wire<S>> = BTreeSet::new();
        let mut write: BTreeSet<Wire<S>> = BTreeSet::new();

        let iter = iter.into_iter().map(|t| t.map(Term::convert));
        let terms: Vec<Term<T, S>> = iter.collect::<Result<Vec<_>, _>>()?;

        for term in terms.iter() {
            for rd in term.read().iter() {
                if !write.contains(rd) {
                    read.insert(rd.clone());
                }
            }

            for wt in term.write().iter() {
                if read.contains(wt) {
                    return Err(format!("Wire {:?} is read before being written", wt));
                }
                if !write.insert(wt.clone()) {
                    return Err(format!("Wire {:?} is written more than once", wt));
                }
            }
        }

        Ok(Block {
            terms,
            read: Vec::from_iter(read),
            write: Vec::from_iter(write),
        })
    }

    pub fn convert<U>(block: Block<U, S>) -> Self
    where
        U: Into<T>,
    {
        Block {
            terms: block.terms.into_iter().map(Term::convert).collect(),
            read: block.read,
            write: block.write,
        }
    }
}

impl<F, S> Block<F, S>
where
    F: Differential<S>,
    S: Clone + Debug,
{
    /// Builds a block of one `ZERO` term per write wire, so that theories can
    /// keep `ZERO` at a fixed arity and each term stays a single-wire node in
    /// the compute graph.
    pub fn zero<W: IntoIterator<Item = Wire<S>>>(write: W) -> Result<Self, String> {
        let write = write.into_iter();
        Block::try_from_iter(write.map(|w| Term::constant(F::zero(w.dtype()), [w])))
    }
}

impl<J, S> Block<J, S>
where
    J: Sequential<S>,
    S: Clone + Debug,
{
    /// Builds a block of one `SKIP` term per `(write, read)` wire pair, so
    /// that theories can keep `SKIP` at a fixed arity and each pair is an
    /// independent edge in the compute graph (no spurious dependency of one
    /// write on all the other reads).
    pub fn skip<W: IntoIterator<Item = Wire<S>>, R: IntoIterator<Item = Wire<S>>>(
        write: W,
        read: R,
    ) -> Result<Self, String> {
        let mut write = write.into_iter();
        let mut read = read.into_iter();
        Block::try_from_iter(std::iter::from_fn(move || {
            match (write.next(), read.next()) {
                (Some(w), Some(r)) => Some(Term::function(J::skip(w.dtype()), [w], [r])),
                (None, None) => None,
                _ => Some(Err("skip requires as many write as read wires".to_string())),
            }
        }))
    }
}

impl<I, S> Block<I, S>
where
    I: Combinatorial<S>,
    S: Clone + Debug,
{
    /// Builds a block of one `HAVOC` term per write wire
    pub fn havoc<W: IntoIterator<Item = Wire<S>>>(write: W) -> Result<Self, String> {
        let write = write.into_iter();
        Block::try_from_iter(write.map(|w| Term::constant(I::havoc(w.dtype()), [w])))
    }
}
