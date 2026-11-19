//! The matrices held in Rust, as nested `Vec`s, in every build.

use super::{Error, Kind, Value, fmt_matrix};
use std::fmt;

/// A matrix held in Rust: its shape, and its rows of elements of one kind,
/// each row a `Vec`.
#[derive(Clone, Debug, PartialEq)]
pub struct VecTensor {
    shape: [usize; 2],
    pub(super) rows: Rows,
}

/// The rows of a [`VecTensor`], as many as its shape gives, each of as
/// many elements as it gives.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Rows {
    Bool(Vec<Vec<bool>>),
    Int(Vec<Vec<i64>>),
    Float(Vec<Vec<f64>>),
}

impl VecTensor {
    /// The matrix of `shape` of the rows of booleans `rows`, which fails
    /// unless they are the rows of a matrix of `shape`.
    pub fn bools(shape: [usize; 2], rows: Vec<Vec<bool>>) -> Result<Self, Error> {
        check(shape, &rows)?;
        Ok(VecTensor {
            shape,
            rows: Rows::Bool(rows),
        })
    }

    /// The matrix of `shape` of the rows of integers `rows`, which fails
    /// unless they are the rows of a matrix of `shape`.
    pub fn ints(shape: [usize; 2], rows: Vec<Vec<i64>>) -> Result<Self, Error> {
        check(shape, &rows)?;
        Ok(VecTensor {
            shape,
            rows: Rows::Int(rows),
        })
    }

    /// The matrix of `shape` of the rows of floating-point numbers `rows`,
    /// which fails unless they are the rows of a matrix of `shape`.
    pub fn floats(shape: [usize; 2], rows: Vec<Vec<f64>>) -> Result<Self, Error> {
        check(shape, &rows)?;
        Ok(VecTensor {
            shape,
            rows: Rows::Float(rows),
        })
    }

    /// The `[rows, cols]` shape of the matrix.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// The kind of the elements.
    pub fn kind(&self) -> Kind {
        match self.rows {
            Rows::Bool(_) => Kind::Bool,
            Rows::Int(_) => Kind::Int,
            Rows::Float(_) => Kind::Float,
        }
    }

    /// The element in row `i` and column `j`.
    fn value(&self, i: usize, j: usize) -> Value {
        match &self.rows {
            Rows::Bool(rows) => Value::Bool(rows[i][j]),
            Rows::Int(rows) => Value::Int(rows[i][j]),
            Rows::Float(rows) => Value::Float(rows[i][j]),
        }
    }
}

/// Fails unless `rows` are the rows of a matrix of `shape` `[m, n]`: `m`
/// rows of `n` elements.
fn check<T>(shape @ [m, n]: [usize; 2], rows: &[Vec<T>]) -> Result<(), Error> {
    if rows.len() == m && rows.iter().all(|row| row.len() == n) {
        Ok(())
    } else {
        Err(Error::Rows {
            shape,
            lengths: rows.iter().map(Vec::len).collect(),
        })
    }
}

impl fmt::Display for VecTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix(f, self.shape, |i, j| self.value(i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_hold_the_rows_of_their_shape() {
        let t = VecTensor::ints([3, 2], vec![vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        assert_eq!((t.shape(), t.kind()), ([3, 2], Kind::Int));
        assert_eq!(VecTensor::bools([0, 3], vec![]).unwrap().shape(), [0, 3]);
        assert_eq!(
            VecTensor::floats([1, 0], vec![vec![]]).unwrap().shape(),
            [1, 0]
        );
    }

    #[test]
    fn matrices_fail_on_rows_of_another_shape() {
        let rows = vec![vec![0.0; 3], vec![0.0; 3], vec![0.0; 2]];
        assert_eq!(
            VecTensor::floats([3, 3], rows),
            Err(Error::Rows {
                shape: [3, 3],
                lengths: vec![3, 3, 2]
            })
        );
        assert_eq!(
            VecTensor::ints([2, 1], vec![vec![1]]),
            Err(Error::Rows {
                shape: [2, 1],
                lengths: vec![1]
            })
        );
    }

    #[test]
    fn small_matrices_print_entirely_on_one_line() {
        let t = VecTensor::floats([1, 3], vec![vec![1.5, 2.0, 3.0]]).unwrap();
        assert_eq!(t.to_string(), "[[1.5, 2, 3]]");
        let b = VecTensor::bools([2, 1], vec![vec![true], vec![false]]).unwrap();
        assert_eq!(b.to_string(), "[[true], [false]]");
    }

    #[test]
    fn large_matrices_abbreviate_with_dots() {
        let t = VecTensor::ints([2, 12], vec![(0..12).collect(), (12..24).collect()]).unwrap();
        assert_eq!(
            t.to_string(),
            "[[0, 1, 2, ..., 9, 10, 11], [12, 13, 14, ..., 21, 22, 23]]"
        );
        let t = VecTensor::ints([20, 1], (0..20).map(|i| vec![i]).collect()).unwrap();
        assert_eq!(t.to_string(), "[[0], [1], [2], ..., [17], [18], [19]]");
    }
}
