//! Boundaries: the types of inputs and outputs (the domain and codomain)
//! of morphism-like objects.
//!
//! The boundary of a morphism is its domain and codomain. We represent the
//! domain and codomain as *polynomials over types*. "Polynomial" is meant
//! as in *algebraic data types*: the variables are base types, and `⊗` and `⊕`
//! build new types from old ones. With a base type `Real`, for example,
//! `Real ⊗ Real` is a pair of reals.
//!
//! More generally, the polynomials are built from base types `G` by two type
//! constructors and the type units. The constructors are
//!
//! - `⊗`, the product: a value of `x ⊗ y` is a value of `x` *and* a value of
//!   `y`, i.e., the tuple `(x, y)`;
//! - `⊕`, the sum: a value of `x ⊕ y` is a value of `x` *or* a value of `y`,
//!   i.e., the variant, like `enum { A(x), B(y) }`.
//!
//! The type units are
//!
//! - `I`, the unit type: the empty product, with exactly one value, like `()`;
//! - `0`, the empty type: the empty sum, with no values, like `enum {}`.
//!
//! For example, `I ⊕ I` has two values and carries no data, like `bool`, and
//! `I ⊕ Real` is either nothing or a real, like `Option<Real>`.
//!
//! # Normal form
//!
//! As for ordinary types, `⊗` distributes over `⊕`: a value of
//! `(x ⊕ y) ⊗ z` is either an `(x, z)` or a `(y, z)`, so the type is
//! `x ⊗ z ⊕ y ⊗ z`. Every polynomial therefore has a normal form as a sum of
//! products, and that is how it is stored: a [`Poly`] is a list of
//! alternatives, each alternative is a [`Mono`]. A `Mono` is a sequence of base
//! types.
//!
//! # Equality
//!
//! Types are compared structurally: two polynomials are equal iff they have
//! the same summands, in the same order, each the same word of base types.
//! Types that are isomorphic but not structurally equal are different: `x ⊕ y`
//! is not `y ⊕ x`, `x ⊗ y` is not `y ⊗ x`, and `I ⊕ I` is not `I`.

/// A product type: the tuple of a word of base types, `g₁ ⊗ … ⊗ gₙ`. A value
/// has one component of each base type, in order, all at once; the
/// components are positional, as in an anonymous tuple. The empty word is
/// the unit type `I`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mono<G>(Vec<G>);

/// A polynomial type: a sum of product types, `m₁ ⊕ … ⊕ mₖ`. A value is a
/// value of exactly one of the summands. This is like a
/// Rust enum whose variants are tuples, except that the variants are
/// anonymous and told apart by position, so their order is part of the type
/// (see *Equality* in the [module documentation](crate::boundary)). The
/// empty sum is the empty type `0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<G>(Vec<Mono<G>>);

impl<G> Mono<G> {
    /// The unit type `I`: the empty tuple. It has exactly one value, and it
    /// is the unit of `⊗`.
    pub fn unit() -> Self {
        Self(Vec::new())
    }

    /// The base types of the tuple, in order.
    pub fn iter(&self) -> std::slice::Iter<'_, G> {
        self.0.iter()
    }
}

impl<G> IntoIterator for Mono<G> {
    type Item = G;
    type IntoIter = std::vec::IntoIter<G>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, G> IntoIterator for &'a Mono<G> {
    type Item = &'a G;
    type IntoIter = std::slice::Iter<'a, G>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// A word of base types is the tuple type of them, in order.
impl<G> FromIterator<G> for Mono<G> {
    fn from_iter<I: IntoIterator<Item = G>>(word: I) -> Self {
        Self(word.into_iter().collect())
    }
}

/// A product type is a sum type with a single variant.
impl<G> From<Mono<G>> for Poly<G> {
    fn from(m: Mono<G>) -> Self {
        Self(vec![m])
    }
}

/// A word of base types is the sum type with a single variant: their tuple
/// type. An empty word therefore gives `I`, not `0`.
impl<G> FromIterator<G> for Poly<G> {
    fn from_iter<I: IntoIterator<Item = G>>(word: I) -> Self {
        Mono::from_iter(word).into()
    }
}

/// Product types, taken as alternatives in order, form their sum type. No
/// alternatives give the empty type `0`.
impl<G> FromIterator<Mono<G>> for Poly<G> {
    fn from_iter<I: IntoIterator<Item = Mono<G>>>(summands: I) -> Self {
        Self(summands.into_iter().collect())
    }
}

impl<G> std::ops::Mul for Mono<G> {
    type Output = Mono<G>;

    /// `⊗` of tuple types: the tuple type with the components of `self`
    /// followed by those of `other`.
    fn mul(mut self, other: Self) -> Mono<G> {
        self.0.extend(other.0);
        self
    }
}

impl<G> std::ops::Add for Mono<G> {
    type Output = Poly<G>;

    /// `⊕` of tuple types: the sum type with the variants `self` and
    /// `other`, in that order.
    fn add(self, other: Self) -> Poly<G> {
        Poly(vec![self, other])
    }
}

impl<G> Poly<G> {
    /// The empty type `0`: the sum with no variants, which has no values —
    /// the impossible boundary (a failed guard).
    pub fn zero() -> Self {
        Self(Vec::new())
    }

    /// The unit type `I`, as a sum: a single variant, the empty tuple.
    pub fn unit() -> Self {
        Self(vec![Mono::unit()])
    }
}

impl<G> Poly<G> {
    /// The variants of the sum, each a product type, in order.
    pub fn iter(&self) -> std::slice::Iter<'_, Mono<G>> {
        self.0.iter()
    }
}

impl<G> IntoIterator for Poly<G> {
    type Item = Mono<G>;
    type IntoIter = std::vec::IntoIter<Mono<G>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, G> IntoIterator for &'a Poly<G> {
    type Item = &'a Mono<G>;
    type IntoIter = std::slice::Iter<'a, Mono<G>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<G> std::ops::Add for Poly<G> {
    type Output = Poly<G>;

    /// `⊕` of sum types: the variants of `self` followed by those of
    /// `other`.
    fn add(mut self, other: Self) -> Poly<G> {
        self.0.extend(other.0);
        self
    }
}

/// Distributing copies base types into several variants, hence `G: Clone`.
impl<G: Clone> std::ops::Mul for Poly<G> {
    type Output = Poly<G>;

    /// `⊗` of sum types, distributed into a sum of products: one variant for
    /// every pair of a variant of `self` and a variant of `other`, in that
    /// order, so `(a ⊕ b) ⊗ (c ⊕ d) = a ⊗ c ⊕ a ⊗ d ⊕ b ⊗ c ⊕ b ⊗ d`.
    fn mul(self, other: Self) -> Poly<G> {
        self.iter()
            .flat_map(|m| other.iter().map(|n| m.clone() * n.clone()))
            .collect()
    }
}

/// Prints the base types joined by `⊗`, or `I` for the unit type.
impl<G: std::fmt::Display> std::fmt::Display for Mono<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return f.write_str("I");
        }
        for (i, g) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(" ⊗ ")?;
            }
            write!(f, "{g}")?;
        }
        Ok(())
    }
}

/// Prints the variants joined by `⊕`, or `0` for the empty type.
impl<G: std::fmt::Display> std::fmt::Display for Poly<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return f.write_str("0");
        }
        for (i, m) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(" ⊕ ")?;
            }
            write!(f, "{m}")?;
        }
        Ok(())
    }
}

/// The boundary of a morphism: the type of its inputs,
/// [`dom`](Self::dom), and the type of its outputs, [`cod`](Self::cod), both
/// polynomials over the base types `G` (for a theory's operations, its
/// sorts). Boundaries are total: an ill-formed morphism does not exist, by
/// construction. Type-checking is checking equality of types: an
/// application is well-formed iff its boundary equals the morphism's,
/// structurally (see *Equality* in the
/// [module documentation](crate::boundary)).
///
/// `dom` and `cod` return by value on purpose: the boundary is computed
/// from the morphism on demand, so implementors need not store it.
/// Caching should be implemented only if this becomes a critical path.
pub trait Boundary<G> {
    /// The domain: the type of the morphism's inputs.
    fn dom(&self) -> Poly<G>;

    /// The codomain: the type of the morphism's outputs.
    fn cod(&self) -> Poly<G>;
}
