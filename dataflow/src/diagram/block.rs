use super::Term;
use super::Wire;
use base::boundary::{Boundary, Mono, Poly};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::vec;
use theory::{Combinatorial, Differential, Sequential, Theory};

/// A string diagram: a list of terms, in a topological order, with an
/// interface of read and write wires.
///
/// The interface is ordered, and the sorts of its wires, in order, are the
/// type of the block (see [`read_type`](Self::read_type),
/// [`write_type`](Self::write_type) and the [`Boundary`] impl). Blocks built by
/// [`try_from_iter`](Self::try_from_iter) infer their interface and order it
/// by wire id; blocks built by [`new`](Self::new) take it from the caller, in
/// which case:
///
/// - a read wire may also be a write wire: the value passes through
///   unchanged, so identities need no terms;
/// - a read wire need not be read by any term: the value is discarded;
/// - a wire may occur several times among the write wires: the value is
///   copied.
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

    /// Returns a reference to the *read interface* of the block.
    ///
    /// The read interface lists all wires that must be provided externally
    /// for the block to operate, and are not written internally by the block.
    /// These wires are inputs required by the block as a whole.
    pub fn read(&self) -> &[Wire<T::Sort>] {
        self.read.as_slice()
    }

    /// Returns a reference to the *write interface* of the block.
    ///
    /// The write interface lists all wires that the block writes. These wires represent
    /// the outputs of the block as a whole; they can all be read outside the block.
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

impl<T: Theory> Block<T>
where
    T::Sort: Clone,
{
    /// The type of the inputs: the sorts of the read wires, in order.
    pub fn read_type(&self) -> Mono<T::Sort> {
        self.read.iter().map(|w| w.dtype().clone()).collect()
    }

    /// The type of the outputs: the sorts of the write wires, in order.
    pub fn write_type(&self) -> Mono<T::Sort> {
        self.write.iter().map(|w| w.dtype().clone()).collect()
    }
}

/// A block is typed by the sorts of its interface wires, in order, as a
/// single product type on each side. The degree of the wires is not part of
/// the type.
impl<T: Theory> Boundary<T::Sort> for Block<T>
where
    T::Sort: Clone,
{
    fn dom(&self) -> Poly<T::Sort> {
        self.read_type().into()
    }

    fn cod(&self) -> Poly<T::Sort> {
        self.write_type().into()
    }
}

impl<T: Theory> Block<T>
where
    T::Sort: Clone + Debug,
{
    /// Builds a block with the given ordered interface.
    ///
    /// The terms must be in a topological order: every wire a term reads is
    /// either a `read` wire or written by an earlier term. No wire is written
    /// twice, and no term writes a `read` wire. Every `write` wire is either
    /// a `read` wire (passed through) or written by a term; it may occur more
    /// than once (copied). The `read` wires are distinct; they need not be
    /// read by any term (discarded).
    pub fn new<R, W, I>(read: R, write: W, terms: I) -> Result<Self, String>
    where
        R: IntoIterator<Item = Wire<T::Sort>>,
        W: IntoIterator<Item = Wire<T::Sort>>,
        I: IntoIterator<Item = Term<T>>,
    {
        let read: Vec<_> = read.into_iter().collect();
        let write: Vec<_> = write.into_iter().collect();
        let terms: Vec<_> = terms.into_iter().collect();

        let mut defined: HashSet<&Wire<T::Sort>> = HashSet::new();
        for rd in read.iter() {
            if !defined.insert(rd) {
                return Err(format!("Wire {:?} is read more than once", rd));
            }
        }
        for term in terms.iter() {
            for rd in term.read() {
                if !defined.contains(rd) {
                    return Err(format!("Wire {:?} is read before being written", rd));
                }
            }
            for wt in term.write() {
                if !defined.insert(wt) {
                    return Err(format!("Wire {:?} is written more than once", wt));
                }
            }
        }
        for wt in write.iter() {
            if !defined.contains(wt) {
                return Err(format!("Wire {:?} is never written", wt));
            }
        }

        Ok(Block { terms, read, write })
    }

    /// Whether `wire` occurs in the block: as a read wire or as a wire
    /// written by a term. (Write wires are among these, by construction.)
    pub(crate) fn mentions(&self, wire: &Wire<T::Sort>) -> bool {
        self.read.contains(wire) || self.terms.iter().any(|t| t.write().contains(wire))
    }

    /// The identity on `sig`: a block of fresh `0`-form wires, one per
    /// factor, each passed through from the read to the write interface.
    pub fn id(sig: &Mono<T::Sort>) -> Self {
        let wires: Vec<_> = sig.iter().cloned().map(Wire::scalar).collect();
        Block {
            terms: Vec::new(),
            read: wires.clone(),
            write: wires,
        }
    }

    /// Sequential composition `f ; g`: the outputs of `f` feed the inputs of
    /// `g`, by position.
    ///
    /// The result reads what `f` reads and writes what `g` writes. The wires
    /// of `g` are renamed: its read wires to the matching write wires of
    /// `f`, all others to fresh wires, so the result is well-formed even
    /// when `f` and `g` share wires (e.g., when they are the same block).
    ///
    /// Fails if the type of the outputs of `f` is not the type of the inputs
    /// of `g`.
    pub fn compose(f: &Self, g: &Self) -> Result<Self, String>
    where
        T: Clone,
        T::Sort: PartialEq,
    {
        if f.write_type() != g.read_type() {
            return Err(format!(
                "Cannot compose a block writing {:?} with a block reading {:?}",
                f.write_type(),
                g.read_type()
            ));
        }
        let mut rename: HashMap<Wire<T::Sort>, Wire<T::Sort>> = g
            .read
            .iter()
            .cloned()
            .zip(f.write.iter().cloned())
            .collect();
        let mut rename = |w: &Wire<T::Sort>| {
            rename
                .entry(w.clone())
                .or_insert_with(|| Wire::new(w.dtype().clone(), w.degree()))
                .clone()
        };
        let g_terms: Vec<_> = g.terms.iter().map(|t| t.rename(&mut rename)).collect();
        let write = g.write.iter().map(&mut rename).collect();
        let mut terms = f.terms.clone();
        terms.extend(g_terms);
        Ok(Block {
            terms,
            read: f.read.clone(),
            write,
        })
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
