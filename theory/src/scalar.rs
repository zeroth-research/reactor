/*!
# Scalars

The scalar operations of each theory, one type per operation, in a module per
theory: `enum` for the enumerations of symbolic constants, and [`event`] for
the events, booleans whose change signals them. Each operation reads and
writes the sorts of [`sort`], which have no shape, and the booleans of the
matrix theories of the shape `[1, 1]`, which their comparisons write and their
choices read, and computes its own boundary.

Unlike the operations of [`crate::matrix`], the scalar operations carry no
matrices, so they are not generic over a [`Matrix`](crate::Matrix) type.

[`Any`] holds the operations of all of them, one variant per theory, with
their boundary in the sorts of all theories, [`crate::sort::Any`], which hold
those booleans. The structural
operations, which belong to no theory, are those of all theories,
[`crate::Havoc`], [`crate::Skip`] and [`crate::Zero`].
*/

pub mod r#enum;
pub mod event;
pub mod sort;

use base::boundary::{Boundary, Poly};
use derive_more::From;
use std::fmt;

/// The operations of all the scalar theories: a value of `Any` is an
/// operation of one of them.
#[derive(Clone, PartialEq, Debug, Eq, Hash, From)]
pub enum Any {
    Enum(r#enum::Any),
    Event(event::Any),
}

/// The boundary of each operation, in the sorts of all theories.
impl Boundary<crate::sort::Any> for Any {
    fn dom(&self) -> Poly<crate::sort::Any> {
        match self {
            Any::Enum(op) => op.dom(),
            Any::Event(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<crate::sort::Any> {
        match self {
            Any::Enum(op) => op.cod(),
            Any::Event(op) => op.cod(),
        }
    }
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Enum(op) => op.fmt(f),
            Any::Event(op) => op.fmt(f),
        }
    }
}
