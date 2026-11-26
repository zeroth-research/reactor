/*!
# Matrices

The [`Matrix`] trait, which the operations carrying data are generic over,
and the checks of the shapes of the matrices they carry.
*/

/// A matrix of values, as held by the operations that carry data: the
/// constants, and the coefficients of `Affine`. The operations are generic
/// over it, so that any representation of matrices, such as torch tensors, can
/// back them.
pub trait Matrix {
    /// The `[rows, cols]` shape of the matrix.
    fn shape(&self) -> [usize; 2];
}

/// Fails unless `shape` has at least one row and one column: a matrix of no
/// entries has exactly one value, the vector of no coordinates, so that its
/// sort would be another copy of [`Real0`](crate::zero::Real0).
pub(crate) fn check_shape(shape: [usize; 2]) -> Result<(), crate::Error> {
    if shape.contains(&0) {
        return Err(crate::Error(format!(
            "matrices have at least one row and one column, got the shape {shape:?}"
        )));
    }
    Ok(())
}

/// Fails unless `a` and `b` are the shapes `[m, n]` and `[m, k]` of the
/// matrices `A` and `B` of an affine map `x ↦ A·x + B`, none of them empty.
pub(crate) fn check_affine(a: [usize; 2], b: [usize; 2]) -> Result<(), crate::Error> {
    check_shape(a)?;
    check_shape(b)?;
    if a[0] != b[0] {
        return Err(crate::Error(format!(
            "expected A of shape [m, n] and B of shape [m, k], got {a:?} and {b:?}"
        )));
    }
    Ok(())
}
