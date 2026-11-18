//! The matrices held in Rust, in every build.

use super::{Kind, Value, fmt_matrix};
use std::fmt;

/// A matrix held in Rust: its elements, row after row, of one kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Builtin {
    shape: [usize; 2],
    elements: Elements,
}

#[derive(Clone, Debug, PartialEq)]
enum Elements {
    Bool(Vec<bool>),
    Int(Vec<i64>),
    Float(Vec<f64>),
}

impl Builtin {
    /// The matrix of the booleans `values`, row after row.
    ///
    /// # Panics
    ///
    /// Unless there are `m · n` values for `shape` `[m, n]`.
    pub fn bools(shape: [usize; 2], values: Vec<bool>) -> Self {
        Self::new(shape, values.len(), Elements::Bool(values))
    }

    /// The matrix of the integers `values`, row after row.
    ///
    /// # Panics
    ///
    /// Unless there are `m · n` values for `shape` `[m, n]`.
    pub fn ints(shape: [usize; 2], values: Vec<i64>) -> Self {
        Self::new(shape, values.len(), Elements::Int(values))
    }

    /// The matrix of the floating-point numbers `values`, row after row.
    ///
    /// # Panics
    ///
    /// Unless there are `m · n` values for `shape` `[m, n]`.
    pub fn floats(shape: [usize; 2], values: Vec<f64>) -> Self {
        Self::new(shape, values.len(), Elements::Float(values))
    }

    fn new([m, n]: [usize; 2], len: usize, elements: Elements) -> Self {
        assert_eq!(
            len,
            m * n,
            "expected {} values for a [{m}, {n}] matrix",
            m * n
        );
        Self {
            shape: [m, n],
            elements,
        }
    }

    /// The `[rows, cols]` shape of the matrix.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// The kind of the elements.
    pub fn kind(&self) -> Kind {
        match self.elements {
            Elements::Bool(_) => Kind::Bool,
            Elements::Int(_) => Kind::Int,
            Elements::Float(_) => Kind::Float,
        }
    }

    /// The element in row `i` and column `j`.
    fn value(&self, i: usize, j: usize) -> Value {
        let k = i * self.shape[1] + j;
        match &self.elements {
            Elements::Bool(v) => Value::Bool(v[k]),
            Elements::Int(v) => Value::Int(v[k]),
            Elements::Float(v) => Value::Float(v[k]),
        }
    }

    /// The rows of the matrix, each a list of its elements.
    #[cfg(feature = "pyo3")]
    pub(super) fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = Value> + '_> + '_ {
        let [m, n] = self.shape;
        (0..m).map(move |i| (0..n).map(move |j| self.value(i, j)))
    }
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_matrix(f, self.shape, |i, j| self.value(i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_matrices_print_entirely_on_one_line() {
        let t = Builtin::floats([1, 3], vec![1.5, 2.0, 3.0]);
        assert_eq!(t.to_string(), "[[1.5, 2, 3]]");
        let b = Builtin::bools([2, 1], vec![true, false]);
        assert_eq!(b.to_string(), "[[true], [false]]");
    }

    #[test]
    fn large_matrices_abbreviate_with_dots() {
        let t = Builtin::ints([2, 12], (0..24).collect());
        assert_eq!(
            t.to_string(),
            "[[0, 1, 2, ..., 9, 10, 11], [12, 13, 14, ..., 21, 22, 23]]"
        );
        let t = Builtin::ints([20, 1], (0..20).collect());
        assert_eq!(t.to_string(), "[[0], [1], [2], ..., [17], [18], [19]]");
    }

    #[test]
    #[should_panic(expected = "expected 6 values")]
    fn matrices_have_a_value_per_element() {
        Builtin::floats([2, 3], vec![0.0; 5]);
    }
}
