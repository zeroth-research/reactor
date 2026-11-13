/*!
# Casts between the theories' sorts

Every other op reads and writes the sorts of one theory. A [`Cast`] crosses
from one theory's sort to another's, elementwise and keeping the shape:

- [`Cast::IntToReal`] — `Int(m,n) → Real(m,n)`, the same number: SMT-LIB's
  `to_real`.
- [`Cast::RealToInt`] — `Real(m,n) → Int(m,n)`, the floor: SMT-LIB's
  `to_int`.
- [`Cast::BVToInt`] — `BV<w>(m,n) → Int(m,n)`, the unsigned value:
  SMT-LIB's `ubv_to_int`.
- [`Cast::BitToBool`] — `BV<1>(m,n) → Bool(m,n)`, whether the bit is 1.
- [`Cast::BoolToBit`] — `Bool(m,n) → BV<1>(m,n)`, 1 for true and 0 for
  false.

A BV comparison writes a 1-bit vector where LIA's writes a Bool, so a
formula saying both needs the last two to join them.

## Examples

```
use theory::Theory;
use theory::any::Sort;
use theory::cast::Cast;

let deg0 = |s| Ok::<_, String>((s, 0u8));

assert!(Cast::IntToReal()
    .check([Sort::Int([2, 1])].map(deg0), [Sort::Real([2, 1])].map(deg0))
    .is_ok());
// The shape is kept, and a 1-bit vector is what a Bool comes from.
assert!(Cast::IntToReal()
    .check([Sort::Int([2, 1])].map(deg0), [Sort::Real([1, 2])].map(deg0))
    .is_err());
assert!(Cast::BitToBool()
    .check([Sort::BitVec(8, [1, 1])].map(deg0), [Sort::Bool([1, 1])].map(deg0))
    .is_err());
```
*/

use crate::any::Sort;
use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum Cast {
    /// `Int → Real`, the same number
    IntToReal(),
    /// `Real → Int`, the floor
    RealToInt(),
    /// `BV<w> → Int`, the unsigned value
    BVToInt(),
    /// `BV<1> → Bool`, whether the bit is 1
    BitToBool(),
    /// `Bool → BV<1>`, 1 for true
    BoolToBit(),
}

impl Cast {
    /// The sort this cast writes from `read`, or `None` for a sort it does
    /// not read.
    fn types(&self, read: Sort) -> Option<Sort> {
        match (self, read) {
            (Cast::IntToReal(), Sort::Int(s)) => Some(Sort::Real(s)),
            (Cast::RealToInt(), Sort::Real(s)) => Some(Sort::Int(s)),
            (Cast::BVToInt(), Sort::BitVec(_, s)) => Some(Sort::Int(s)),
            (Cast::BitToBool(), Sort::BitVec(1, s)) => Some(Sort::Bool(s)),
            (Cast::BoolToBit(), Sort::Bool(s)) => Some(Sort::BitVec(1, s)),
            _ => None,
        }
    }

    fn reads(&self) -> &'static str {
        match self {
            Cast::IntToReal() => "an Int matrix",
            Cast::RealToInt() => "a Real matrix",
            Cast::BVToInt() => "a bit-vector matrix",
            Cast::BitToBool() => "a 1-bit vector matrix",
            Cast::BoolToBit() => "a Bool matrix",
        }
    }
}

impl Theory for Cast {
    type Sort = Sort;
    const NAME: &'static str = "Cast";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<(Sort, u8), E>>,
        W: IntoIterator<Item = Result<(Sort, u8), E>>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        let (r, None) = (next_expect_degree(&mut read, 0, 0)?, read.next()) else {
            return Err(format!("{self}: must read exactly one value"));
        };
        let (w, None) = (next_expect_degree(&mut write, 0, 0)?, write.next()) else {
            return Err(format!("{self}: must write exactly one value"));
        };
        match self.types(r) {
            None => Err(format!("{self}: reads {}, got {r}", self.reads())),
            Some(want) if want != w => Err(format!("{self}: output must be {want}, got {w}")),
            Some(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg0(s: Sort) -> Result<(Sort, u8), String> {
        Ok((s, 0))
    }

    fn ok(cast: Cast, from: Sort, to: Sort) -> bool {
        cast.check([from].map(deg0), [to].map(deg0)).is_ok()
    }

    #[test]
    fn each_cast_crosses_once_and_keeps_the_shape() {
        let s = [2, 3];
        assert!(ok(Cast::IntToReal(), Sort::Int(s), Sort::Real(s)));
        assert!(ok(Cast::RealToInt(), Sort::Real(s), Sort::Int(s)));
        assert!(ok(Cast::BVToInt(), Sort::BitVec(8, s), Sort::Int(s)));
        assert!(ok(Cast::BitToBool(), Sort::BitVec(1, s), Sort::Bool(s)));
        assert!(ok(Cast::BoolToBit(), Sort::Bool(s), Sort::BitVec(1, s)));
    }

    #[test]
    fn a_cast_refuses_another_sort_or_shape() {
        assert!(!ok(
            Cast::IntToReal(),
            Sort::Real([1, 1]),
            Sort::Real([1, 1])
        ));
        assert!(!ok(
            Cast::IntToReal(),
            Sort::Int([1, 2]),
            Sort::Real([2, 1])
        ));
        assert!(!ok(
            Cast::BVToInt(),
            Sort::BitVec(8, [1, 1]),
            Sort::Real([1, 1])
        ));
        assert!(!ok(
            Cast::BitToBool(),
            Sort::BitVec(2, [1, 1]),
            Sort::Bool([1, 1])
        ));
        assert!(!ok(
            Cast::BoolToBit(),
            Sort::Bool([1, 1]),
            Sort::BitVec(2, [1, 1])
        ));
    }

    #[test]
    fn a_cast_reads_and_writes_one_value() {
        let i = Sort::Int([1, 1]);
        let r = Sort::Real([1, 1]);
        assert!(
            Cast::IntToReal()
                .check([i, i].map(deg0), [r].map(deg0))
                .is_err()
        );
        assert!(
            Cast::IntToReal()
                .check([i].map(deg0), [r, r].map(deg0))
                .is_err()
        );
    }
}
