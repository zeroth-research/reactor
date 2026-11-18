/*!
# Sorts

The sorts of all theories, and the sorts of each theory.

[`Any`] holds the sorts of all theories, just as [`crate::tensor::Any`] holds the
operations of all theories. `subenum` defines the sorts of each theory as a
subset of them, and generates the casts between `Any` and each subset:
`From` up to `Any`, and `TryFrom` down from it.

## Naming the subsets

The subsets are named after the SMT-LIB2 *theories*, and not after the
*logics*, such as [`crate::tensor::LRA`] or [`crate::tensor::BV`], which name subsets of the
operations. A theory fixes the sorts, and each logic restricts its
operations: the `L` of `LRA` stands for linear, a restriction on the
operations, so `LRA` and a non-linear `NRA` would share the same sorts.

Each subset is the plural of the sort its theory adds, as in SMT-LIB2's
`Reals` and `Ints`, so that the singular names an individual sort and the
plural the sorts of a theory:

| Subset | Sorts | SMT-LIB2 theory |
|---|---|---|
| [`Prop`] | [`Bool`], [`Zero`] | `Core` |
| [`Reals`] | [`Real`], [`Bool`], [`Zero`] | `Reals` |
| [`Ints`] | [`Int`], [`Bool`], [`Zero`] | `Ints` |
| [`Nums`] | [`Real`], [`Int`], [`Bool`], [`Zero`] | `Reals_Ints` |
| [`Words`] | [`Word`], [`Bool`], [`Zero`] | `FixedSizeBitVectors` |

[`Nums`], the sorts of both kinds of numbers, shortens `Reals_Ints`, and
[`Words`] follows our sort [`Word`] rather than SMT-LIB2's longer name.
[`Prop`] is the exception to the plurals: the theory of booleans is
SMT-LIB2's `Core`, a name too generic for its sorts, so it takes the name
of propositional logic, whose operations are [`crate::tensor::Prop`].

Every subset includes [`Bool`], because every SMT-LIB2 theory includes the
Core theory, whose booleans are the sort that comparisons write; [`Prop`]
holds exactly the sorts that the others share.

Every subset also includes [`Zero`], which is not an SMT-LIB2 sort, but
the trivial tangent of the discrete sorts [`Bool`], [`Int`] and [`Word`],
and its own tangent. With it, each subset holds the tangents of its sorts,
and so is closed under taking tangents, as [`Any`] is.

The sorts are defined here, and the operations on them in the modules of
their theories: [`super::real`], [`super::int`], [`super::bool`] and
[`super::word`].
*/

use base::differentiable::{Differentiable, T};
use derive_more::From;
use std::fmt;
use subenum::subenum;

/// The sort of real matrices with the given `[rows, cols]` shape, graded by
/// their `degree`: values of degree `d` are `d`-th order rates of change,
/// and plain reals have degree 0.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Real {
    degree: u8,
    shape: [usize; 2],
}

impl Real {
    /// Plain real matrices of the given shape, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Real { degree: 0, shape }
    }

    pub fn degree(&self) -> u8 {
        self.degree
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        match self.degree {
            0 => write!(f, "Real({i}, {j})"),
            d => write!(f, "T^{d}(Real({i}, {j}))"),
        }
    }
}

/// The rate of change of a real matrix is a real matrix of the same shape,
/// one degree higher.
impl Differentiable for Real {
    type Tangent = Real;

    fn tangent(self) -> Real {
        Real {
            degree: self.degree + 1,
            shape: self.shape,
        }
    }
}

/// The sort of integer matrices with the given `[rows, cols]` shape.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Int {
    shape: [usize; 2],
}

impl Int {
    pub fn new(shape: [usize; 2]) -> Self {
        Int { shape }
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Int({i}, {j})")
    }
}

/// Integers are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Differentiable for Int {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sort of boolean matrices with the given `[rows, cols]` shape.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Bool {
    shape: [usize; 2],
}

impl Bool {
    pub fn new(shape: [usize; 2]) -> Self {
        Bool { shape }
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Bool({i}, {j})")
    }
}

/// Booleans are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Differentiable for Bool {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sort of matrices of `width`-bit words with the given `[rows, cols]`
/// shape: entrywise, the SMT-LIB2 sort `(_ BitVec width)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Word {
    width: usize,
    shape: [usize; 2],
}

impl Word {
    /// Matrices of `width`-bit words; as in SMT-LIB2, `width` must be
    /// positive, or this returns an error.
    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, String> {
        if width == 0 {
            return Err("words have at least one bit".to_string());
        }
        Ok(Word { width, shape })
    }

    /// Matrices of `width`-bit words, as [`Word::new`], but panics if
    /// `width` is zero.
    pub fn new_unchecked(width: usize, shape: [usize; 2]) -> Self {
        Self::new(width, shape).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Word<{}>({i}, {j})", self.width)
    }
}

/// Words are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Differentiable for Word {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The singleton sort, with exactly one value: the trivial tangent of
/// discrete sorts. Its own tangent is itself.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Zero;

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero")
    }
}

impl Differentiable for Zero {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sorts of all theories: a value of `Any` is a sort of one of them.
#[subenum(Prop, Reals, Ints, Nums, Words)]
#[derive(Clone, Copy, PartialEq, Debug, Eq, From)]
pub enum Any {
    #[subenum(Reals, Nums)]
    Real(Real),
    #[subenum(Ints, Nums)]
    Int(Int),
    #[subenum(Prop, Reals, Ints, Nums, Words)]
    Bool(Bool),
    #[subenum(Words)]
    Word(Word),
    #[subenum(Prop, Reals, Ints, Nums, Words)]
    Zero(Zero),
}

/// Each sort's tangent, taken within `Any`.
impl Differentiable for Any {
    type Tangent = Any;

    fn tangent(self) -> Any {
        match self {
            Any::Real(r) => T(r).into(),
            Any::Int(i) => T(i).into(),
            Any::Bool(b) => T(b).into(),
            Any::Word(w) => T(w).into(),
            Any::Zero(z) => T(z).into(),
        }
    }
}

impl fmt::Display for Words {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Words::Word(w) => w.fmt(f),
            Words::Bool(b) => b.fmt(f),
            Words::Zero(z) => z.fmt(f),
        }
    }
}

/// All the sorts of words are discrete, so their tangent is trivial: the singleton
/// sort [`Zero`].
impl Differentiable for Words {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_shape_and_display() {
        let r = Real::new([2, 3]);
        assert_eq!(r.shape(), [2, 3]);
        assert_eq!(r.to_string(), "Real(2, 3)");
    }

    #[test]
    fn real_tangent_raises_degree() {
        let r = Real::new([2, 3]);
        assert_eq!(T(r).degree(), 1);
        assert_eq!(T(r).shape(), [2, 3]);
        assert_eq!(T(T(r)).degree(), 2);
        assert_eq!(T(r).to_string(), "T^1(Real(2, 3))");
    }

    #[test]
    fn int_shape_and_display() {
        let r = Int::new([2, 3]);
        assert_eq!(r.shape(), [2, 3]);
        assert_eq!(r.to_string(), "Int(2, 3)");
    }

    #[test]
    fn int_tangent_is_trivial() {
        assert_eq!(T(Int::new([2, 3])), Zero);
        assert_eq!(T(T(Int::new([2, 3]))), Zero);
    }

    #[test]
    fn bool_shape_and_display() {
        let b = Bool::new([2, 3]);
        assert_eq!(b.shape(), [2, 3]);
        assert_eq!(b.to_string(), "Bool(2, 3)");
    }

    #[test]
    fn bool_tangent_is_trivial() {
        assert_eq!(T(Bool::new([2, 3])), Zero);
        assert_eq!(T(T(Bool::new([2, 3]))), Zero);
    }

    #[test]
    fn word_width_shape_and_display() {
        let x = Word::new(8, [2, 3]).unwrap();
        assert_eq!(x.width(), 8);
        assert_eq!(x.shape(), [2, 3]);
        assert_eq!(x.to_string(), "Word<8>(2, 3)");
    }

    #[test]
    fn word_has_positive_width() {
        assert_eq!(
            Word::new(0, [2, 3]),
            Err("words have at least one bit".to_string())
        );
        assert!(Word::new(1, [2, 3]).is_ok());
    }

    #[test]
    fn word_new_unchecked_builds_positive_widths() {
        assert_eq!(
            Word::new_unchecked(8, [2, 3]),
            Word::new(8, [2, 3]).unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "words have at least one bit")]
    fn word_new_unchecked_panics_on_zero_width() {
        Word::new_unchecked(0, [2, 3]);
    }

    #[test]
    fn words_display_their_sort() {
        let x = Word::new_unchecked(8, [2, 3]);
        assert_eq!(Words::from(x), Words::Word(x));
        assert_eq!(Words::from(x).to_string(), "Word<8>(2, 3)");
        assert_eq!(Words::from(Bool::new([2, 3])).to_string(), "Bool(2, 3)");
        assert_eq!(Words::from(Zero).to_string(), "Zero");
    }

    #[test]
    fn word_tangent_is_trivial() {
        assert_eq!(T(Word::new_unchecked(8, [2, 3])), Zero);
        assert_eq!(T(Words::from(Word::new_unchecked(8, [2, 3]))), Zero);
        assert_eq!(T(Words::from(Bool::new([2, 3]))), Zero);
    }

    #[test]
    fn zero_tangent_is_itself() {
        assert_eq!(T(Zero), Zero);
    }

    #[test]
    fn zero_display() {
        assert_eq!(Zero.to_string(), "Zero");
    }

    #[test]
    fn tangent_is_each_sorts_tangent() {
        let r = Real::new([2, 3]);
        assert_eq!(T(Any::from(r)), Any::Real(T(r)));
        assert_eq!(T(Any::from(Int::new([2, 3]))), Any::Zero(Zero));
        assert_eq!(T(Any::from(Bool::new([2, 3]))), Any::Zero(Zero));
        assert_eq!(
            T(Any::from(Word::new_unchecked(8, [2, 3]))),
            Any::Zero(Zero)
        );
        assert_eq!(T(Any::from(Zero)), Any::Zero(Zero));
    }

    #[test]
    fn subsets_hold_the_tangents_of_their_sorts() {
        let sorts = [
            Any::Real(Real::new([2, 3])),
            Any::Int(Int::new([2, 3])),
            Any::Bool(Bool::new([2, 3])),
            Any::Word(Word::new_unchecked(8, [2, 3])),
            Any::Zero(Zero),
        ];
        for s in sorts {
            if Prop::try_from(s).is_ok() {
                assert!(Prop::try_from(T(s)).is_ok());
            }
            if Reals::try_from(s).is_ok() {
                assert!(Reals::try_from(T(s)).is_ok());
            }
            if Ints::try_from(s).is_ok() {
                assert!(Ints::try_from(T(s)).is_ok());
            }
            if Nums::try_from(s).is_ok() {
                assert!(Nums::try_from(T(s)).is_ok());
            }
            if Words::try_from(s).is_ok() {
                assert!(Words::try_from(T(s)).is_ok());
            }
        }
    }

    #[test]
    fn subsets_cast_to_and_from_any() {
        let r = Real::new([2, 3]);
        let i = Int::new([2, 3]);
        let b = Bool::new([2, 3]);
        let w = Word::new_unchecked(8, [2, 3]);

        // each sort casts up from the subsets that hold it
        assert_eq!(Any::from(Reals::from(r)), Any::Real(r));
        assert_eq!(Any::from(Ints::from(i)), Any::Int(i));
        assert_eq!(Any::from(Nums::from(r)), Any::Real(r));
        assert_eq!(Any::from(Nums::from(i)), Any::Int(i));
        assert_eq!(Any::from(Words::from(w)), Any::Word(w));

        // and down to them only
        assert_eq!(Words::try_from(Any::Word(w)).ok(), Some(Words::Word(w)));
        assert!(Words::try_from(Any::Real(r)).is_err());
        assert!(Reals::try_from(Any::Int(i)).is_err());
        assert!(Ints::try_from(Any::Real(r)).is_err());
        assert!(Nums::try_from(Any::Word(w)).is_err());

        // every subset holds the booleans and `Zero`
        for s in [Any::Bool(b), Any::Zero(Zero)] {
            assert!(Prop::try_from(s).is_ok());
            assert!(Reals::try_from(s).is_ok());
            assert!(Ints::try_from(s).is_ok());
            assert!(Nums::try_from(s).is_ok());
            assert!(Words::try_from(s).is_ok());
        }

        // `Prop` holds only those
        assert!(Prop::try_from(Any::Real(r)).is_err());
        assert!(Prop::try_from(Any::Int(i)).is_err());
        assert!(Prop::try_from(Any::Word(w)).is_err());
    }
}
