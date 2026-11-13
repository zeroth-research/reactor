//! Tapes: probabilistic and conditional choice between string diagrams.
//!
//! A tape `P → Q`, between polynomial types `P = U₁ ⊕ … ⊕ Uₙ` and
//! `Q = V₁ ⊕ … ⊕ Vₘ`, is an `m × n` matrix whose cell `(i, j)` is a formal sum
//! of [entries](Entry) `Uⱼ → Vᵢ`, following *Tapes as Stochastic Matrices of
//! String Diagrams* (Bonchi, Cioffo; arXiv:2601.01472). A value of the `j`-th
//! input summand `Uⱼ` goes, through the string diagram of an entry in column
//! `j`, to the output summand of the entry's row. An entry is a string
//! diagram with a probabilistic weight; the empty sum is `0·⋆`.
//!
//! # Conditions
//!
//! Conditional choice depends on values computed by the string diagrams
//! themselves. Its building block is the itype
//! [`assume(polarity)`](Conditional::assume) of the theory, which drops a
//! run unless its boolean read equals `polarity`: a block with `assume`
//! terms is partial. [`guard_split`](Tape::guard_split) sends a value to one
//! summand or another through two entries of weight `1` that assume
//! opposite polarities of the same boolean. (Conditionals between values of
//! the same type can instead stay inside a block, with
//! [`ite`](Conditional::ite).)
//!
//! # Stochasticity
//!
//! For every column and every input value, the weights of the entries of the
//! column whose `assume`s pass sum to at most `1`, as for the
//! subdistributions of the paper: the missing mass is the probability that
//! the run is dropped. The constructors give exactly `1`, except
//! [`lift`](Tape::lift) of a block with `assume` terms, and every operation
//! preserves the invariant, so it holds for every tape by construction and
//! is not checked at run time. The raw weights of a column can sum to more
//! than `1`: the two entries of a [`guard_split`](Tape::guard_split) have
//! weight `1` each, but never pass together. (Entries of weight `0` are
//! dropped.)

use crate::diagram::{Block, Term, Wire};
use base::boundary::{Boundary, Mono, Poly};
use base::semiring::{Probability, Semiring};
use std::fmt::Debug;
use theory::{Conditional, Theory};

/// A weighted string diagram: one summand of a cell of a [`Tape`].
///
/// In cell `(i, j)` of a tape `P → Q`, the block has type `Uⱼ → Vᵢ` for the
/// `j`-th summand `Uⱼ` of `P` and the `i`-th summand `Vᵢ` of `Q`.
pub struct Entry<T: Theory, P> {
    block: Block<T>,
    weight: P,
}

// Written by hand: `derive` would not require `Block<T>: Clone/Debug`, which
// needs the same of `T::Sort`.
impl<T: Theory, P: Clone> Clone for Entry<T, P>
where
    Block<T>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            block: self.block.clone(),
            weight: self.weight.clone(),
        }
    }
}

impl<T: Theory, P: Debug> Debug for Entry<T, P>
where
    Block<T>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("block", &self.block)
            .field("weight", &self.weight)
            .finish()
    }
}

impl<T: Theory, P> Entry<T, P> {
    /// The string diagram of the entry.
    pub fn block(&self) -> &Block<T> {
        &self.block
    }

    /// The probabilistic weight of the entry.
    pub fn weight(&self) -> &P {
        &self.weight
    }
}

impl<T: Theory, P: Semiring> Entry<T, P> {
    /// Builds an entry, or nothing if its weight is `0`.
    fn new(block: Block<T>, weight: P) -> Option<Self> {
        (weight != P::zero()).then_some(Self { block, weight })
    }
}

/// A tape `read → write`: a matrix with a row per summand of `write` and a
/// column per summand of `read`, whose cells are formal sums of [`Entry`]s
/// (see the [module documentation](crate::tape)).
///
/// Tapes are built from the constructors ([`id`](Self::id),
/// [`merge`](Self::merge), [`prob_split`](Self::prob_split),
/// [`guard_split`](Self::guard_split), [`branch`](Self::branch),
/// [`swap`](Self::swap), [`lift`](Self::lift)) and combined by [`sum`](Self::sum) and
/// [`compose`](Self::compose); these keep the tape (sub)stochastic.
#[derive(Debug, Clone)]
pub struct Tape<T: Theory, P> {
    read: Poly<T::Sort>,
    write: Poly<T::Sort>,
    // TODO: entries are owned by their cell; sharing them (e.g., through an
    // arena of entries indexed by an `EntryId`) is deferred, and so is merging
    // equal entries of a cell (e.g., `p·id + (1 − p)·id` into `1·id`).
    /// `cells[i][j]`: the entries from the `j`-th summand of `read` to the
    /// `i`-th summand of `write`.
    cells: Vec<Vec<Vec<Entry<T, P>>>>,
}

impl<T: Theory, P> Tape<T, P> {
    /// The input type: one summand per column.
    pub fn read(&self) -> &Poly<T::Sort> {
        &self.read
    }

    /// The output type: one summand per row.
    pub fn write(&self) -> &Poly<T::Sort> {
        &self.write
    }

    /// The number of rows: the number of summands of [`write`](Self::write).
    pub fn rows(&self) -> usize {
        self.cells.len()
    }

    /// The number of columns: the number of summands of
    /// [`read`](Self::read).
    pub fn cols(&self) -> usize {
        self.read.iter().count()
    }

    /// The entries of cell `(row, col)`: from the `col`-th summand of
    /// [`read`](Self::read) to the `row`-th summand of [`write`](Self::write).
    ///
    /// # Panics
    /// If `row` or `col` is out of bounds.
    pub fn cell(&self, row: usize, col: usize) -> &[Entry<T, P>] {
        &self.cells[row][col]
    }

    /// All entries, with their row and column.
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, &Entry<T, P>)> {
        self.cells.iter().enumerate().flat_map(|(i, row)| {
            row.iter()
                .enumerate()
                .flat_map(move |(j, cell)| cell.iter().map(move |e| (i, j, e)))
        })
    }
}

impl<T: Theory, P> Boundary<T::Sort> for Tape<T, P>
where
    T::Sort: Clone,
{
    fn dom(&self) -> Poly<T::Sort> {
        self.read.clone()
    }

    fn cod(&self) -> Poly<T::Sort> {
        self.write.clone()
    }
}

impl<T, P> Tape<T, P>
where
    T: Theory + Clone,
    T::Sort: Clone + Debug + PartialEq,
    P: Probability,
{
    /// A tape with no entries: every cell is `0·⋆`.
    fn empty(read: Poly<T::Sort>, write: Poly<T::Sort>) -> Self {
        let cols = read.iter().count();
        let rows = write.iter().count();
        let cells = (0..rows)
            .map(|_| (0..cols).map(|_| Vec::new()).collect())
            .collect();
        Self { read, write, cells }
    }

    /// Adds an entry in cell `(row, col)`, unless its weight is `0`.
    fn put(&mut self, row: usize, col: usize, block: Block<T>, weight: P) {
        self.cells[row][col].extend(Entry::new(block, weight));
    }

    /// Adds an identity entry on `m` in cell `(row, col)`, unless its weight
    /// is `0`.
    fn put_id(&mut self, row: usize, col: usize, m: &Mono<T::Sort>, weight: P) {
        self.put(row, col, Block::id(m), weight);
    }

    /// The identity `sig → sig`: the diagonal matrix of identity blocks,
    /// of weight `1`.
    pub fn id(sig: Poly<T::Sort>) -> Self {
        let mut tape = Self::empty(sig.clone(), sig.clone());
        for (k, m) in sig.iter().enumerate() {
            tape.put_id(k, k, m, P::one());
        }
        tape
    }

    /// The merge `sig ⊕ sig → sig` (the codiagonal): both copies of each
    /// summand go to that summand, through identity blocks of weight `1`.
    pub fn merge(sig: Poly<T::Sort>) -> Self {
        let n = sig.iter().count();
        let mut tape = Self::empty(sig.clone() + sig.clone(), sig.clone());
        for (k, m) in sig.iter().enumerate() {
            tape.put_id(k, k, m, P::one());
            tape.put_id(k, n + k, m, P::one());
        }
        tape
    }

    /// The probabilistic split `sig → sig ⊕ sig`: each summand goes to its
    /// first copy with probability `prob` and to its second with probability
    /// `1 − prob`, through identity blocks.
    ///
    /// For `prob = 1` (or `0`) the second (or first) copy is never reached,
    /// and the split behaves as the identity into the other copy.
    ///
    /// Fails if `prob` is not a probability (not in `[0, 1]`).
    pub fn prob_split(sig: Poly<T::Sort>, prob: P) -> Result<Self, String> {
        if !prob.is_probability() {
            return Err("prob_split: the weight is not in [0, 1]".to_string());
        }
        let n = sig.iter().count();
        let mut tape = Self::empty(sig.clone(), sig.clone() + sig.clone());
        let other = prob.complement();
        for (k, m) in sig.iter().enumerate() {
            tape.put_id(k, k, m, prob.clone());
            tape.put_id(n + k, k, m, other.clone());
        }
        Ok(tape)
    }

    /// The ⊕-symmetry `p ⊕ q → q ⊕ p`: each summand goes to its own copy on
    /// the other side, through identity blocks of weight `1`.
    pub fn swap(p: Poly<T::Sort>, q: Poly<T::Sort>) -> Self {
        let np = p.iter().count();
        let nq = q.iter().count();
        let mut tape = Self::empty(p.clone() + q.clone(), q.clone() + p.clone());
        for (k, m) in p.iter().enumerate() {
            tape.put_id(nq + k, k, m, P::one());
        }
        for (l, m) in q.iter().enumerate() {
            tape.put_id(l, np + l, m, P::one());
        }
        tape
    }

    /// The tape `u → v` of a single block `u → v`: one entry of weight `1`.
    ///
    /// The block may contain `assume` terms, and then the tape is partial
    /// (see [Stochasticity](crate::tape#stochasticity)).
    ///
    /// Fails if a wire of the interface of the block is not a `0`-form.
    pub fn lift(block: Block<T>) -> Result<Self, String> {
        if let Some(w) = block
            .read()
            .iter()
            .chain(block.write())
            .find(|w| w.degree() != 0)
        {
            return Err(format!("lift: interface wire {:?} is not a 0-form", w));
        }
        let read = Poly::from(block.read_type());
        let write = Poly::from(block.write_type());
        let mut tape = Self::empty(read, write);
        tape.put(0, 0, block, P::one());
        Ok(tape)
    }

    /// The direct sum `t ⊕ s`: the block-diagonal matrix, from
    /// `t.read ⊕ s.read` to `t.write ⊕ s.write`.
    pub fn sum(t: &Self, s: &Self) -> Self {
        let (tc, sc) = (t.cols(), s.cols());
        let mut cells: Vec<Vec<Vec<Entry<T, P>>>> = Vec::new();
        for row in t.cells.iter() {
            let mut row = row.clone();
            row.extend((0..sc).map(|_| Vec::new()));
            cells.push(row);
        }
        for row in s.cells.iter() {
            let mut r: Vec<Vec<Entry<T, P>>> = (0..tc).map(|_| Vec::new()).collect();
            r.extend(row.iter().cloned());
            cells.push(r);
        }
        Self {
            read: t.read.clone() + s.read.clone(),
            write: t.write.clone() + s.write.clone(),
            cells,
        }
    }

    /// Sequential composition `t ; s`, first `t` then `s`: the matrix product
    /// of `s` and `t`, from `t.read` to `s.write`.
    ///
    /// Cell `(u, i)` of the result has an entry for every entry `e` of
    /// `t[j][i]` and `d` of `s[u][j]`, for every summand `j` of
    /// `t.write = s.read`: its block is `e.block ; d.block` (which runs only
    /// if the `assume`s of both pass) and its weight `e.weight · d.weight`
    /// (unless it is `0`).
    ///
    /// Fails if `t.write` is not `s.read`.
    pub fn compose(t: &Self, s: &Self) -> Result<Self, String> {
        if t.write != s.read {
            return Err(format!(
                "compose: a tape writing {:?} cannot feed a tape reading {:?}",
                t.write, s.read
            ));
        }
        let mut tape = Self::empty(t.read.clone(), s.write.clone());
        for (u, s_row) in s.cells.iter().enumerate() {
            for i in 0..t.cols() {
                for (j, s_cell) in s_row.iter().enumerate() {
                    for e in t.cells[j][i].iter() {
                        for d in s_cell.iter() {
                            let block = Block::compose(&e.block, &d.block)?;
                            let weight = e.weight.clone() * d.weight.clone();
                            tape.put(u, i, block, weight);
                        }
                    }
                }
            }
        }
        Ok(tape)
    }
}

impl<T, P> Tape<T, P>
where
    T: Conditional + Clone,
    T::Sort: Clone + Debug + PartialEq,
    P: Probability,
{
    /// The guarded split `sig ⊗ B → sig ⊕ sig`, for the theory's
    /// [`boolean`](Conditional::boolean) sort `B`: for each summand `Uₖ` of
    /// `sig`, a value of `Uₖ ⊗ B` goes to the first copy of `Uₖ` when its
    /// boolean is true and to the second when it is false, and the boolean is
    /// consumed. Both entries have weight `1` and pass `Uₖ` through; one
    /// assumes the boolean true and the other false.
    pub fn guard_split(sig: Poly<T::Sort>) -> Self {
        let n = sig.iter().count();
        let read: Poly<T::Sort> = sig
            .iter()
            .map(|m| Mono::from_iter(m.iter().cloned().chain([T::boolean()])))
            .collect();
        let mut tape = Self::empty(read, sig.clone() + sig.clone());
        for (k, m) in sig.iter().enumerate() {
            for (row, polarity) in [(k, true), (n + k, false)] {
                let xs: Vec<_> = m.iter().cloned().map(Wire::scalar).collect();
                let b = Wire::scalar(T::boolean());
                let assume = Term::function(
                    T::assume(polarity),
                    Vec::<Wire<T::Sort>>::new(),
                    [b.clone()],
                )
                .expect("assume reads one boolean");
                let read = xs.iter().cloned().chain([b]);
                let block = Block::new(read, xs.clone(), [assume])
                    .expect("the interface of a guarded split is well formed");
                tape.put(row, k, block, P::one());
            }
        }
        tape
    }

    /// The conditional branch `u → u ⊕ u` on a condition `cond : u → B`, for
    /// the theory's [`boolean`](Conditional::boolean) sort `B`: a value goes
    /// to the first copy of `u` when `cond` holds on it and to the second
    /// when it does not.
    ///
    /// It is `lift(c) ; guard_split(u)`, for the block `c : u → u ⊗ B` that
    /// passes its input through and also runs `cond` on it.
    ///
    /// Fails if `cond` does not write exactly one value of sort `B`, or if
    /// [`lift`](Self::lift) fails on `c`.
    pub fn branch(cond: &Block<T>) -> Result<Self, String> {
        if cond.write_type() != Mono::from_iter([T::boolean()]) {
            return Err(format!(
                "branch: the condition writes {:?}, not one value of sort {:?}",
                cond.write_type(),
                T::boolean()
            ));
        }
        let write = cond.read().iter().chain(cond.write()).cloned();
        let c = Block::new(cond.read().to_vec(), write, cond.iter().cloned())?;
        let sig = Poly::from(cond.read_type());
        Self::compose(&Self::lift(c)?, &Self::guard_split(sig))
    }
}
