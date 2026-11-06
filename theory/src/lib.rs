pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;
pub mod tensor;

use base::boundary::Poly;
use std::fmt;
pub use tensor::PyTensor;

pub trait Theory {
    type Sort;

    const NAME: &'static str;

    /// Check that the operation can read and write wires of the given sorts: the sorts of
    /// the read and written wires (the boundary of the application) must be the domain and
    /// the codomain of the operation (its [`base::boundary::Boundary`]).
    /// Derivatives are wires of the tangent sorts (see [`base::tangent::Tangent::tangent`]).
    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Self::Sort, E>>,
        W: IntoIterator<Item = Result<Self::Sort, E>>;

    /// Returns the unary itype that discards a value of the domain (`domain -> I`): it
    /// reads one wire and writes none, consuming an input a block does not otherwise use.
    fn discard(domain: &Self::Sort) -> Self;
}

pub trait Combinatorial: Theory {
    // Returns the unary itype that chooses an element within the range
    fn havoc(range: &Self::Sort) -> Self;
}

pub trait Sequential: Theory {
    // Returns the unary itype that copies a value within the range
    fn skip(range: &Self::Sort) -> Self;
}

pub trait Differential: Theory {
    // Returns the unary itype that indicates zero derivative, the range is the sort
    // of the derivative (i.e., a tangent sort)
    fn zero(range: &Self::Sort) -> Self;
}

// Helpers for type-checking procedures

/// The product `s₁ ⊗ … ⊗ sₙ` of the sorts of objects read or written all at once
fn prod<S, const N: usize>(sorts: [S; N]) -> Poly<S> {
    Poly::from_iter(sorts)
}

/// Check the application of an operation with the domain `dom` and the codomain `cod` to
/// wires: the sorts of the read wires are the domain of the application and the sorts of
/// the written wires its codomain. The application is well-formed iff its domain is `dom`
/// and its codomain is `cod`.
fn check_wires<S, R, W, E>(dom: Poly<S>, cod: Poly<S>, read: R, write: W) -> Result<(), String>
where
    S: PartialEq + fmt::Display,
    R: IntoIterator<Item = Result<S, E>>,
    W: IntoIterator<Item = Result<S, E>>,
    E: fmt::Display,
{
    let (read, write) = (wire_sorts(read)?, wire_sorts(write)?);
    if (&dom, &cod) != (&read, &write) {
        return Err(format!(
            "expected `{dom} -> {cod}`, got `{read} -> {write}`"
        ));
    }
    Ok(())
}

/// The sorts of `wires` (read or written all at once, i.e., a monomial), or the error of
/// the first wire that has none
fn wire_sorts<S, I, E: fmt::Display>(wires: I) -> Result<Poly<S>, String>
where
    I: IntoIterator<Item = Result<S, E>>,
{
    wires
        .into_iter()
        .map(|w| w.map_err(|e| e.to_string()))
        .collect()
}

/// The `[rows, cols]` shape of the 2D tensor `t`
fn shape_2d(t: &PyTensor) -> Result<[usize; 2], String> {
    match t.size().as_slice() {
        &[rows, cols] => Ok([rows as usize, cols as usize]),
        size => Err(format!("expected a 2D tensor, got {}D", size.len())),
    }
}

/// The number of inputs of the affine map `x ↦ A·x + B` with `A` of the shape
/// `[outputs, inputs]`
fn linear_inputs(a: &PyTensor) -> Result<usize, String> {
    let [_, inp] = shape_2d(a)?;
    Ok(inp)
}

/// The number of outputs of the affine map `x ↦ A·x + B` with `A` of the shape
/// `[outputs, inputs]`, or an error if `B` is neither empty nor a column vector `[outputs, 1]`
fn linear_outputs(a: &PyTensor, b: &PyTensor) -> Result<usize, String> {
    let [out, _] = shape_2d(a)?;
    if b.numel() != 0 && shape_2d(b)? != [out, 1] {
        return Err(format!("`B` must be a column vector [{out}x1]"));
    }
    Ok(out)
}
