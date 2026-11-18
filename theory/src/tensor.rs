/*!
# Tensors

The matrices that the operations carrying data hold, the constants and the
coefficients of the affine maps: a [`Tensor`] is one of

- a [`Builtin`] matrix, held in Rust, in every build;
- a `Torch` tensor, with the `torch` feature.

The operations of a module may hold tensors of different backends. A
tensor gives its shape, the [`Kind`] of its elements, which the operations
check (a real constant holds floating-point elements), and its values, as
it displays them on a single line.

Only this module tells the backends apart. The variants of [`Tensor`]
depend on the features, which Cargo unifies across a build, so a `match` on
it elsewhere would break when another crate turns on a feature; code
elsewhere uses the methods of [`Tensor`] instead.

With the `pyo3` feature, a tensor converts from and into its Python object:
a `torch.Tensor`, or, for a [`Builtin`], a list of rows.
*/

mod builtin;
#[cfg(feature = "pyo3")]
mod py;
#[cfg(feature = "torch")]
mod torch;

pub use builtin::Builtin;
#[cfg(feature = "torch")]
pub use torch::Torch;

use std::fmt;

/// A matrix held by an operation, in one of the backends.
#[derive(Clone, Debug)]
pub enum Tensor {
    /// A matrix held in Rust.
    Builtin(Builtin),
    /// A torch tensor.
    #[cfg(feature = "torch")]
    Torch(Torch),
}

impl Tensor {
    /// The `[rows, cols]` shape of the matrix.
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Tensor::Builtin(t) => t.shape(),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.shape(),
        }
    }

    /// The kind of the elements.
    pub fn kind(&self) -> Kind {
        match self {
            Tensor::Builtin(t) => t.kind(),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.kind(),
        }
    }

    /// Fails unless the elements are of `kind`.
    pub fn expect_kind(&self, kind: Kind) -> Result<(), Error> {
        let got = self.kind();
        if got == kind {
            Ok(())
        } else {
            Err(Error::Kind {
                expected: kind,
                got,
            })
        }
    }
}

/// Fails unless `a` and `b` are the matrices of an affine map `x ↦ A·x + B`
/// with elements of `kind`: `A` of shape `[m, n]` and `B` of shape `[m, k]`.
pub fn expect_affine(a: &Tensor, b: &Tensor, kind: Kind) -> Result<(), Error> {
    a.expect_kind(kind.clone())?;
    b.expect_kind(kind)?;
    let (a, b) = (a.shape(), b.shape());
    if a[0] != b[0] {
        return Err(Error::Affine { a, b });
    }
    Ok(())
}

impl From<Builtin> for Tensor {
    fn from(t: Builtin) -> Self {
        Tensor::Builtin(t)
    }
}

#[cfg(feature = "torch")]
impl From<Torch> for Tensor {
    fn from(t: Torch) -> Self {
        Tensor::Torch(t)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tensor::Builtin(t) => t.fmt(f),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.fmt(f),
        }
    }
}

/// The kind of the elements of a tensor, as the operations tell them apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Bool,
    Int,
    Float,
    /// Elements of none of the other kinds, as complex numbers, under the
    /// name their backend gives them.
    Other(String),
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Bool => write!(f, "boolean"),
            Kind::Int => write!(f, "integer"),
            Kind::Float => write!(f, "floating-point"),
            Kind::Other(name) => write!(f, "{name}"),
        }
    }
}

/// Why a tensor cannot be the data of an operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The tensor, of the given shape, is not a matrix.
    NotAMatrix(Vec<usize>),
    /// The elements are of the kind `got`, where `expected` is required.
    Kind { expected: Kind, got: Kind },
    /// `A` and `B` are not of the shapes `[m, n]` and `[m, k]` of an affine
    /// map `x ↦ A·x + B`.
    Affine { a: [usize; 2], b: [usize; 2] },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotAMatrix(shape) => {
                write!(f, "expected a matrix, got a tensor of shape {shape:?}")
            }
            Error::Kind { expected, got } => {
                write!(
                    f,
                    "expected a tensor of {expected} elements, got {got} elements"
                )
            }
            Error::Affine { a, b } => write!(
                f,
                "expected A of shape [m, n] and B of shape [m, k], got {a:?} and {b:?}"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// An element of a tensor, as it displays.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(v) => write!(f, "{v}"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
        }
    }
}

/// Elements above which a matrix is no longer printed entirely.
const FMT_MAX_NUMEL: usize = 16;
/// Elements kept on each side of the `...` when truncating a dimension.
const FMT_EDGE: usize = 3;

/// Writes the matrix of shape `[m, n]` whose elements `value` gives, on a
/// single line: entirely when it is small, and abbreviated with `...` along
/// each oversized dimension when it is not.
fn fmt_matrix(
    f: &mut fmt::Formatter<'_>,
    [m, n]: [usize; 2],
    value: impl Fn(usize, usize) -> Value,
) -> fmt::Result {
    let truncate = m * n > FMT_MAX_NUMEL;
    // the indices printed along a dimension of `len`, `None` for the `...`
    let indices = |len: usize| -> Vec<Option<usize>> {
        if truncate && len > 2 * FMT_EDGE {
            let head = (0..FMT_EDGE).map(Some);
            let tail = (len - FMT_EDGE..len).map(Some);
            head.chain([None]).chain(tail).collect()
        } else {
            (0..len).map(Some).collect()
        }
    };
    let list = |f: &mut fmt::Formatter<'_>,
                len: usize,
                item: &dyn Fn(&mut fmt::Formatter<'_>, usize) -> fmt::Result| {
        write!(f, "[")?;
        for (k, index) in indices(len).into_iter().enumerate() {
            if k > 0 {
                write!(f, ", ")?;
            }
            match index {
                Some(index) => item(f, index)?,
                None => write!(f, "...")?,
            }
        }
        write!(f, "]")
    };
    list(f, m, &|f, i| {
        list(f, n, &|f, j| write!(f, "{}", value(i, j)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensors_check_the_kind_of_their_elements() {
        let floats: Tensor = Builtin::floats([2, 3], vec![0.0; 6]).into();
        assert_eq!(floats.kind(), Kind::Float);
        assert!(floats.expect_kind(Kind::Float).is_ok());
        assert_eq!(
            floats.expect_kind(Kind::Int),
            Err(Error::Kind {
                expected: Kind::Int,
                got: Kind::Float
            })
        );
    }

    #[test]
    fn affine_maps_take_a_and_b_of_the_same_rows() {
        let a: Tensor = Builtin::ints([4, 2], vec![0; 8]).into();
        let b: Tensor = Builtin::ints([4, 3], vec![0; 12]).into();
        assert!(expect_affine(&a, &b, Kind::Int).is_ok());
        assert!(matches!(
            expect_affine(&a, &b, Kind::Float),
            Err(Error::Kind { .. })
        ));
        let c: Tensor = Builtin::ints([3, 3], vec![0; 9]).into();
        assert_eq!(
            expect_affine(&a, &c, Kind::Int),
            Err(Error::Affine {
                a: [4, 2],
                b: [3, 3]
            })
        );
        assert_eq!(
            Error::Affine {
                a: [4, 2],
                b: [3, 3]
            }
            .to_string(),
            "expected A of shape [m, n] and B of shape [m, k], got [4, 2] and [3, 3]"
        );
    }
}
