/*!
# Enumerations

Defines one type per operation on the sort [`Enum`] of enumerations of
symbolic constants, as in nuXmv, whose symbolic constants admit only
equality and choice. Every operation carries the enumerations it reads or
writes, and so enough information to compute its [`Boundary`]:

- [`Val`] — a symbolic constant of an enumeration.
- [`Into`] — a value of an enumeration, as one of another that holds all its
  values.
- [`Eq`](struct@Eq), [`Ne`] — equality and disequality of two values of an
  enumeration, writing a [`Bool`].
- [`Ite`] — the choice `if c then x else y` between two values of an
  enumeration, by a boolean `c`.

SMT-LIB2 has no anonymous enumerations: it declares an enumeration as a
datatype of nullary constructors, `(declare-datatypes ((E 0)) (((idle)
(busy))))`, whose constructors are its constants.

As in SMT-LIB2, the comparisons and the choice read values of a single
enumeration. Values of different enumerations meet through [`Into`], which
keeps the name of a value, as SMT-LIB2's `to_real` keeps the value of an
integer: a value of `{a, b}` and one of `{b, c}` compare as values of `{a,
b, c}`, equal when both are `b`, as in nuXmv. In SMT-LIB2, `Into` is the
function defined by cases from the datatype of its domain to that of its
codomain, or the identity when all the symbolic constants of a model are of
one datatype, each enumeration a constraint of membership.
*/

pub use super::sort::Any as Sort;
use super::sort::{Bool, Enum};
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use std::fmt;
// `Into`, the operation, shadows the trait of the prelude, whose `into` this
// module still calls.
use std::convert::Into as _;

/// A symbolic constant of an enumeration: one of its values.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct Val {
    value: String,
    sort: Enum,
}

impl Val {
    /// The constant `value`, writing the enumeration of `cod`; fails unless
    /// `value` is one of its values.
    pub fn with_cod(value: &str, cod: [Enum; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        if !sort.contains(value) {
            return Err(crate::matrix::unexpected_cod(
                &format!("1 enumeration of the value {value:?}"),
                &[sort],
            ));
        }
        Ok(Val {
            value: value.to_string(),
            sort,
        })
    }
}

/// A value of the enumeration `dom`, as the value of the same name of the
/// enumeration `cod`, which holds all the values of `dom`: total, and the
/// identity on names.
///
/// SMT-LIB2: the function from the datatype of `dom` to that of `cod` mapping
/// each constructor to the one of its name.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct Into {
    dom: Enum,
    cod: Enum,
}

impl Into {
    /// The conversion reading the enumeration `dom` and writing `cod`; fails
    /// unless `cod` holds all the values of `dom`.
    pub fn new(dom: Enum, cod: Enum) -> Result<Self, crate::Error> {
        if !dom.is_subset(&cod) {
            return Err(crate::Error(format!(
                "expected a codomain holding all the values of {dom}, got {cod}"
            )));
        }
        Ok(Into { dom, cod })
    }
}

/// Equality `x = y` of two values of an enumeration.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct Eq {
    sort: Enum,
}

/// Disequality `x ≠ y` of two values of an enumeration.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct Ne {
    sort: Enum,
}

/// Choice `if c then x else y` between two values `x` and `y` of an
/// enumeration, by the boolean `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct Ite {
    sort: Enum,
}

/// The two enumerations of `dom`, which must be the same.
fn same_enum(dom: [Enum; 2]) -> Result<Enum, crate::Error> {
    let [x, y] = dom;
    if x != y {
        return Err(crate::matrix::unexpected_dom(
            "2 equal enumerations",
            &[x, y],
        ));
    }
    Ok(x)
}

impl Eq {
    /// The comparison reading the enumerations of `dom`; fails unless they are
    /// the same. Its codomain, a [`Bool`], does not determine it.
    pub fn with_dom(dom: [Enum; 2]) -> Result<Self, crate::Error> {
        Ok(Eq {
            sort: same_enum(dom)?,
        })
    }
}

impl Ne {
    /// The comparison reading the enumerations of `dom`; fails unless they are
    /// the same. Its codomain, a [`Bool`], does not determine it.
    pub fn with_dom(dom: [Enum; 2]) -> Result<Self, crate::Error> {
        Ok(Ne {
            sort: same_enum(dom)?,
        })
    }
}

impl Ite {
    /// The choice reading the boolean condition and the branches of `dom`;
    /// fails unless the branches are of the same enumeration.
    pub fn with_dom(dom: [Sort; 3]) -> Result<Self, crate::Error> {
        match dom {
            [Sort::Bool(_), Sort::Enum(x), Sort::Enum(y)] if x == y => Ok(Ite { sort: x }),
            _ => Err(crate::matrix::unexpected_dom(
                "a condition Bool and two branches of the same enumeration",
                &dom,
            )),
        }
    }

    /// The choice writing the enumeration of `cod`.
    pub fn with_cod(cod: [Enum; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Ite { sort })
    }
}

/// The operations on enumerations, with their boundary in [`Sort`].
#[derive(Clone, PartialEq, Debug, Eq, Hash, From)]
pub enum Any {
    Val(Val),
    Into(Into),
    Eq(Eq),
    Ne(Ne),
    Ite(Ite),
}

impl Boundary<Sort> for Val {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.clone().into()].into()
    }
}

impl Boundary<Sort> for Into {
    fn dom(&self) -> Poly<Sort> {
        mono![self.dom.clone().into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.cod.clone().into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.clone().into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool.into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.clone().into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool.into()].into()
    }
}

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool.into(),
            self.sort.clone().into(),
            self.sort.clone().into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.clone().into()].into()
    }
}

impl Boundary<Sort> for Any {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Any::Val(op) => op.dom(),
            Any::Into(op) => op.dom(),
            Any::Eq(op) => op.dom(),
            Any::Ne(op) => op.dom(),
            Any::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Any::Val(op) => op.cod(),
            Any::Into(op) => op.cod(),
            Any::Eq(op) => op.cod(),
            Any::Ne(op) => op.cod(),
            Any::Ite(op) => op.cod(),
        }
    }
}

/// Writes the symbolic constant itself.
impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Display for Into {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Into")
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

impl fmt::Display for Ne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ne")
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Val(op) => op.fmt(f),
            Any::Into(op) => op.fmt(f),
            Any::Eq(op) => op.fmt(f),
            Any::Ne(op) => op.fmt(f),
            Any::Ite(op) => op.fmt(f),
        }
    }
}
