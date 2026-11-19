//! A tensor type of the tests' own: the operations of `theory::tensor` are
//! generic over `theory::Tensor`, so any representation of matrices can
//! back them, here with no torch.

use std::fmt;

/// A real matrix, its entries stored row by row.
#[derive(Debug)]
pub struct Matrix {
    shape: [usize; 2],
    entries: Vec<f64>,
}

impl Matrix {
    /// The matrix of the given rows.
    pub fn new<const M: usize, const N: usize>(rows: [[f64; N]; M]) -> Self {
        Matrix {
            shape: [M, N],
            entries: rows.into_iter().flatten().collect(),
        }
    }

    /// The matrix of zeros of the given shape.
    pub fn zeros(shape @ [m, n]: [usize; 2]) -> Self {
        Matrix {
            shape,
            entries: vec![0.0; m * n],
        }
    }
}

impl theory::Tensor for Matrix {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Prints the rows, as `[[1, 0], [0, 1]]`. The terms require it: a term
/// that fails to type-check names its operation, a constant by its tensor.
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [_, cols] = self.shape;
        let rows: Vec<String> = self
            .entries
            .chunks(cols.max(1))
            .map(|row| {
                let row: Vec<String> = row.iter().map(f64::to_string).collect();
                format!("[{}]", row.join(", "))
            })
            .collect();
        write!(f, "[{}]", rows.join(", "))
    }
}
