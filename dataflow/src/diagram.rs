use base::Wire;
use base::boundary::{Boundary, Poly};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};
use std::fmt::{self, Debug};
use std::vec;
use theory::{Combinatorial, Differential, Sequential, Theory, any};

//============================================================
// Term
//============================================================

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
        let r = read.iter().map(|w| w.dtype().clone()).map(ok);
        let w = write.iter().map(|w| w.dtype().clone()).map(ok);
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
        let w = write.iter().map(|w| w.dtype().clone()).map(ok);
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
    fn convert<U>(term: Term<U>) -> Self
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

//============================================================
// Block
//============================================================

#[derive(Debug, Clone)]
pub struct Block<T: Theory> {
    terms: Vec<Term<T>>,
    read: Vec<Wire<T::Sort>>,
    write: Vec<Wire<T::Sort>>,
}

impl<T: Theory> Block<T> {
    pub fn iter(&self) -> impl Iterator<Item = &Term<T>> {
        self.terms.iter()
    }

    /// Returns a reference to the *read interface* of the block: its inputs.
    ///
    /// The read interface lists the wires that must be provided externally
    /// for the block to operate, in the order of the block's boundary (see
    /// [`Block::new`]). A block built by [`Block::try_from_iter`] lists all
    /// wires it reads but does not write, sorted.
    pub fn read(&self) -> &[Wire<T::Sort>] {
        self.read.as_slice()
    }

    /// Returns a reference to the *write interface* of the block: its outputs.
    ///
    /// The write interface lists the wires the block provides to the outside,
    /// in the order of the block's boundary (see [`Block::new`]). A block
    /// built by [`Block::try_from_iter`] lists all wires it writes, sorted.
    pub fn write(&self) -> &[Wire<T::Sort>] {
        self.write.as_slice()
    }

    /// Return a reference to the n-th term in the block
    pub fn get(&self, n: usize) -> Option<&Term<T>> {
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

impl<'a, T: Theory> IntoIterator for &'a Block<T> {
    type Item = &'a Term<T>;
    type IntoIter = std::slice::Iter<'a, Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.iter()
    }
}

impl<T: Theory> IntoIterator for Block<T> {
    type Item = Term<T>;
    type IntoIter = vec::IntoIter<Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.into_iter()
    }
}

/// The boundary of a block: the product of the sorts of its inputs, and the
/// product of the sorts of its outputs, in order.
impl<T: Theory> Boundary<T::Sort> for Block<T>
where
    T::Sort: Clone,
{
    fn dom(&self) -> Poly<T::Sort> {
        self.read.iter().map(|w| w.dtype().clone()).collect()
    }

    fn cod(&self) -> Poly<T::Sort> {
        self.write.iter().map(|w| w.dtype().clone()).collect()
    }
}

impl<T: Theory> Block<T>
where
    T::Sort: Debug,
{
    /// Builds a block with the given boundary: the wires `read` it takes as
    /// inputs and the wires `write` it gives as outputs, both in order.
    ///
    /// The terms compute the outputs from the inputs in the order given:
    /// each term reads only inputs and wires written by earlier terms. Every
    /// input must be read by some term (an input the block has no other use
    /// for is consumed by a [`Theory::discard`] term), every output must be
    /// written by exactly one term, and no term may write an input. Wires that
    /// are written but are not outputs are internal to the block.
    ///
    /// # Errors
    /// If a wire is repeated in the boundary, or the terms violate the rules
    /// above.
    pub fn new<R, W, V>(read: R, write: W, terms: V) -> Result<Self, String>
    where
        R: IntoIterator<Item = Wire<T::Sort>>,
        W: IntoIterator<Item = Wire<T::Sort>>,
        V: IntoIterator<Item = Term<T>>,
    {
        let block = Block {
            terms: terms.into_iter().collect(),
            read: read.into_iter().collect(),
            write: write.into_iter().collect(),
        };
        block.check_boundary()?;
        Ok(block)
    }

    /// Checks that the terms compute the outputs from the inputs as required
    /// by [`Block::new`].
    fn check_boundary(&self) -> Result<(), String> {
        let mut inputs: HashSet<&Wire<T::Sort>> = HashSet::new();
        for rd in self.read.iter() {
            if !inputs.insert(rd) {
                return Err(format!("Input wire {:?} is repeated", rd));
            }
        }
        let mut outputs: HashSet<&Wire<T::Sort>> = HashSet::new();
        for wt in self.write.iter() {
            if inputs.contains(wt) {
                return Err(format!("Wire {:?} is both an input and an output", wt));
            }
            if !outputs.insert(wt) {
                return Err(format!("Output wire {:?} is repeated", wt));
            }
        }

        // the wires the next term may read, and the inputs no term has read yet
        let mut available: HashSet<&Wire<T::Sort>> = inputs.clone();
        let mut unread: HashSet<&Wire<T::Sort>> = inputs.clone();
        let mut written: HashSet<&Wire<T::Sort>> = HashSet::new();
        for term in self.terms.iter() {
            for rd in term.read().iter() {
                if !available.contains(rd) {
                    return Err(format!(
                        "Wire {:?} is read but is neither an input nor written before",
                        rd
                    ));
                }
                unread.remove(rd);
            }
            for wt in term.write().iter() {
                if inputs.contains(wt) {
                    return Err(format!("Input wire {:?} is written", wt));
                }
                if !written.insert(wt) {
                    return Err(format!("Wire {:?} is written more than once", wt));
                }
                available.insert(wt);
            }
        }

        if let Some(wt) = self.write.iter().find(|w| !written.contains(w)) {
            return Err(format!("Output wire {:?} is not written", wt));
        }
        if let Some(rd) = self.read.iter().find(|w| unread.contains(w)) {
            return Err(format!("Input wire {:?} is not read", rd));
        }
        Ok(())
    }
}

impl<T: Theory> Block<T>
where
    T::Sort: Clone + Debug,
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
        U: Theory<Sort = T::Sort> + Into<T>,
        V: IntoIterator<Item = Result<Term<U>, String>>,
    {
        let mut read: BTreeSet<Wire<T::Sort>> = BTreeSet::new();
        let mut write: BTreeSet<Wire<T::Sort>> = BTreeSet::new();

        let iter = iter.into_iter().map(|t| t.map(Term::convert));
        let terms: Vec<Term<T>> = iter.collect::<Result<Vec<_>, _>>()?;

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

    pub fn convert<U>(block: Block<U>) -> Self
    where
        U: Theory<Sort = T::Sort> + Into<T>,
    {
        Block {
            terms: block.terms.into_iter().map(Term::convert).collect(),
            read: block.read,
            write: block.write,
        }
    }
}

impl<F: Differential> Block<F>
where
    F::Sort: Clone + Debug,
{
    /// Builds a block of one `ZERO` term per write wire, so that theories can
    /// keep `ZERO` at a fixed arity and each term stays a single-wire node in
    /// the compute graph.
    pub fn zero<W: IntoIterator<Item = Wire<F::Sort>>>(write: W) -> Result<Self, String> {
        let write = write.into_iter();
        Block::try_from_iter(write.map(|w| Term::constant(F::zero(w.dtype()), [w])))
    }
}

impl<J: Sequential> Block<J>
where
    J::Sort: Clone + Debug,
{
    /// Builds a block of one `SKIP` term per `(write, read)` wire pair, so
    /// that theories can keep `SKIP` at a fixed arity and each pair is an
    /// independent edge in the compute graph (no spurious dependency of one
    /// write on all the other reads).
    pub fn skip<W: IntoIterator<Item = Wire<J::Sort>>, R: IntoIterator<Item = Wire<J::Sort>>>(
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

impl<I: Combinatorial> Block<I>
where
    I::Sort: Clone + Debug,
{
    /// Builds a block of one `HAVOC` term per write wire
    pub fn havoc<W: IntoIterator<Item = Wire<I::Sort>>>(write: W) -> Result<Self, String> {
        let write = write.into_iter();
        Block::try_from_iter(write.map(|w| Term::constant(I::havoc(w.dtype()), [w])))
    }
}
