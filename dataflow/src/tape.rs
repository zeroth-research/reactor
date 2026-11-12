//! Tapes: probabilistic and guarded choice between string diagrams.
//!
//! A tape `P → Q`, between polynomial types `P = U₁ ⊕ … ⊕ Uₙ` and
//! `Q = V₁ ⊕ … ⊕ Vₘ`, is an `m × n` matrix whose cell `(i, j)` is a formal sum
//! of [entries](Entry) `Uⱼ → Vᵢ`, following *Tapes as Stochastic Matrices of
//! String Diagrams* (Bonchi, Cioffo; arXiv:2601.01472). A value of the `j`-th
//! input summand `Uⱼ` goes, through the string diagram of an entry in column
//! `j`, to the output summand of the entry's row. An entry carries a
//! probabilistic weight and a boolean guard; the empty sum is `0·⋆`.
//!
//! # Guards
//!
//! A guard is a conjunction of literals over *external* boolean wires: wires
//! that occur in no block of the tape, so that their value is fixed for the
//! whole tape. An entry is *enabled* under a valuation of the external wires
//! if its guard holds.
//!
//! # Stochasticity
//!
//! For every valuation of the external wires and every column, the weights
//! of the enabled entries of the column sum to `1`. Every constructor
//! establishes this invariant and every operation preserves it, so it holds
//! for every tape by construction and is not checked at run time. (Entries
//! of weight `0`, and entries whose guard contradicts itself, are never
//! enabled and are dropped.)

use crate::diagram::{Block, Wire};
use base::boundary::{Boundary, Mono, Poly};
use base::semiring::{Probability, Semiring};
use std::fmt::Debug;
use theory::{Conditional, Theory};

/// A literal of a guard: an external boolean wire and its polarity; the
/// literal `(w, true)` holds when `w` is true, `(w, false)` when it is false.
pub type Literal<S> = (Wire<S>, bool);

/// A weighted, guarded string diagram: one summand of a cell of a [`Tape`].
///
/// In cell `(i, j)` of a tape `P → Q`, the block has type `Uⱼ → Vᵢ` for the
/// `j`-th summand `Uⱼ` of `P` and the `i`-th summand `Vᵢ` of `Q`.
#[derive(Debug, Clone)]
pub struct Entry<T: Theory, P> {
    block: Block<T>,
    guard: Vec<Literal<T::Sort>>,
    weight: P,
}

impl<T: Theory, P> Entry<T, P> {
    /// The string diagram of the entry.
    pub fn block(&self) -> &Block<T> {
        &self.block
    }

    /// The guard of the entry: the conjunction of its literals, `true` if
    /// there are none. The literals are on distinct wires.
    pub fn guard(&self) -> &[Literal<T::Sort>] {
        &self.guard
    }

    /// The probabilistic weight of the entry.
    pub fn weight(&self) -> &P {
        &self.weight
    }

    /// Whether the entry is enabled under a valuation of the external wires.
    pub fn is_enabled<V>(&self, mut valuation: V) -> bool
    where
        V: FnMut(&Wire<T::Sort>) -> bool,
    {
        self.guard.iter().all(|(w, pol)| valuation(w) == *pol)
    }
}

impl<T: Theory, P: Semiring> Entry<T, P> {
    /// Builds an entry, or nothing if it can never be enabled: if its weight
    /// is `0` or its guard has a literal and its negation. Repeated literals
    /// are kept once.
    fn new<G>(block: Block<T>, guard: G, weight: P) -> Option<Self>
    where
        G: IntoIterator<Item = Literal<T::Sort>>,
    {
        if weight == P::zero() {
            return None;
        }
        let mut lits: Vec<Literal<T::Sort>> = Vec::new();
        for (w, pol) in guard {
            match lits.iter().find(|(v, _)| *v == w) {
                Some((_, p)) if *p == pol => {}
                Some(_) => return None,
                None => lits.push((w, pol)),
            }
        }
        Some(Self {
            block,
            guard: lits,
            weight,
        })
    }
}

/// A tape `read → write`: a matrix with a row per summand of `write` and a
/// column per summand of `read`, whose cells are formal sums of [`Entry`]s
/// (see the [module documentation](crate::tape)).
///
/// Tapes are built from the constructors ([`id`](Self::id),
/// [`merge`](Self::merge), [`prob_split`](Self::prob_split),
/// [`guard_split`](Self::guard_split), [`swap`](Self::swap),
/// [`lift`](Self::lift)) and combined by [`sum`](Self::sum) and
/// [`compose`](Self::compose); these keep the tape stochastic.
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

    /// The external wires the guards refer to, each once.
    fn guard_wires(&self) -> Vec<&Wire<T::Sort>> {
        let mut wires: Vec<&Wire<T::Sort>> = Vec::new();
        for (_, _, e) in self.entries() {
            for (w, _) in e.guard.iter() {
                if !wires.contains(&w) {
                    wires.push(w);
                }
            }
        }
        wires
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

    /// Adds an identity entry on `m` in cell `(row, col)`, unless it can
    /// never be enabled.
    fn put_id(
        &mut self,
        row: usize,
        col: usize,
        m: &Mono<T::Sort>,
        guard: Vec<Literal<T::Sort>>,
        weight: P,
    ) {
        if let Some(e) = Entry::new(Block::id(m), guard, weight) {
            self.cells[row][col].push(e);
        }
    }

    /// The identity `sig → sig`: the diagonal matrix of identity blocks,
    /// unguarded and of weight `1`.
    pub fn id(sig: Poly<T::Sort>) -> Self {
        let mut tape = Self::empty(sig.clone(), sig.clone());
        for (k, m) in sig.iter().enumerate() {
            tape.put_id(k, k, m, Vec::new(), P::one());
        }
        tape
    }

    /// The merge `sig ⊕ sig → sig` (the codiagonal): both copies of each
    /// summand go to that summand, through identity blocks, unguarded and of
    /// weight `1`.
    pub fn merge(sig: Poly<T::Sort>) -> Self {
        let n = sig.iter().count();
        let mut tape = Self::empty(sig.clone() + sig.clone(), sig.clone());
        for (k, m) in sig.iter().enumerate() {
            tape.put_id(k, k, m, Vec::new(), P::one());
            tape.put_id(k, n + k, m, Vec::new(), P::one());
        }
        tape
    }

    /// The probabilistic split `sig → sig ⊕ sig`: each summand goes to its
    /// first copy with probability `prob` and to its second with probability
    /// `1 − prob`, through identity blocks, unguarded.
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
            tape.put_id(k, k, m, Vec::new(), prob.clone());
            tape.put_id(n + k, k, m, Vec::new(), other.clone());
        }
        Ok(tape)
    }

    /// The ⊕-symmetry `p ⊕ q → q ⊕ p`: each summand goes to its own copy on
    /// the other side, through identity blocks, unguarded and of weight `1`.
    pub fn swap(p: Poly<T::Sort>, q: Poly<T::Sort>) -> Self {
        let np = p.iter().count();
        let nq = q.iter().count();
        let mut tape = Self::empty(p.clone() + q.clone(), q.clone() + p.clone());
        for (k, m) in p.iter().enumerate() {
            tape.put_id(nq + k, k, m, Vec::new(), P::one());
        }
        for (l, m) in q.iter().enumerate() {
            tape.put_id(l, np + l, m, Vec::new(), P::one());
        }
        tape
    }

    /// The tape `u → v` of a single block `u → v`: one entry, unguarded and
    /// of weight `1`.
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
        tape.cells[0][0].extend(Entry::new(block, Vec::new(), P::one()));
        Ok(tape)
    }

    /// Fails if a guard of one of the tapes refers to a wire that occurs in a
    /// block of one of the tapes: guard wires must stay external.
    fn check_external(tapes: &[&Self]) -> Result<(), String> {
        for t in tapes {
            for w in t.guard_wires() {
                for s in tapes {
                    if s.entries().any(|(_, _, e)| e.block.mentions(w)) {
                        return Err(format!("guard wire {:?} occurs in a block", w));
                    }
                }
            }
        }
        Ok(())
    }

    /// The direct sum `t ⊕ s`: the block-diagonal matrix, from
    /// `t.read ⊕ s.read` to `t.write ⊕ s.write`.
    ///
    /// Fails if a guard of one tape refers to a wire of a block of the other.
    pub fn sum(t: &Self, s: &Self) -> Result<Self, String> {
        Self::check_external(&[t, s])?;
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
        Ok(Self {
            read: t.read.clone() + s.read.clone(),
            write: t.write.clone() + s.write.clone(),
            cells,
        })
    }

    /// Sequential composition `t ; s`, first `t` then `s`: the matrix product
    /// of `s` and `t`, from `t.read` to `s.write`.
    ///
    /// Cell `(u, i)` of the result has an entry for every entry `e` of
    /// `t[j][i]` and `d` of `s[u][j]`, for every summand `j` of
    /// `t.write = s.read`: its block is `e.block ; d.block`, its guard
    /// `e.guard ∧ d.guard` and its weight `e.weight · d.weight` (unless it can
    /// never be enabled).
    ///
    /// Fails if `t.write` is not `s.read`, or if a guard of one tape refers
    /// to a wire of a block of the other.
    pub fn compose(t: &Self, s: &Self) -> Result<Self, String> {
        if t.write != s.read {
            return Err(format!(
                "compose: a tape writing {:?} cannot feed a tape reading {:?}",
                t.write, s.read
            ));
        }
        Self::check_external(&[t, s])?;
        let mut tape = Self::empty(t.read.clone(), s.write.clone());
        for (u, s_row) in s.cells.iter().enumerate() {
            for i in 0..t.cols() {
                for (j, s_cell) in s_row.iter().enumerate() {
                    for e in t.cells[j][i].iter() {
                        for d in s_cell.iter() {
                            let block = Block::compose(&e.block, &d.block)?;
                            let guard = e.guard.iter().chain(d.guard.iter()).cloned();
                            let weight = e.weight.clone() * d.weight.clone();
                            tape.cells[u][i].extend(Entry::new(block, guard, weight));
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
    /// The guarded split `sig → sig ⊕ sig`: each summand goes to its first
    /// copy when the external wire `guard` is true and to its second when it
    /// is false, through identity blocks of weight `1`.
    ///
    /// Fails if `guard` is not a `0`-form of the theory's
    /// [`boolean`](Conditional::boolean) sort.
    pub fn guard_split(sig: Poly<T::Sort>, guard: Wire<T::Sort>) -> Result<Self, String> {
        if *guard.dtype() != T::boolean() || guard.degree() != 0 {
            return Err(format!(
                "guard_split: the guard {:?} is not a 0-form of sort {:?}",
                guard,
                T::boolean()
            ));
        }
        let n = sig.iter().count();
        let mut tape = Self::empty(sig.clone(), sig.clone() + sig.clone());
        for (k, m) in sig.iter().enumerate() {
            tape.put_id(k, k, m, vec![(guard.clone(), true)], P::one());
            tape.put_id(n + k, k, m, vec![(guard.clone(), false)], P::one());
        }
        Ok(tape)
    }
}
