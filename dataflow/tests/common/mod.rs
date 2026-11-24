//! Fixtures shared by the tape tests: a small opaque theory, helpers to
//! write tape arrays by hand, and two tapes in normal form.
#![allow(dead_code)]

use base::boundary::{Boundary, Mono, Poly};
use dataflow::tape::{
    Arm, Node, NodeId, NodeKind, Operation, Route, RouteId, Span, TapeParts, Value,
};
use std::fmt;

/// The sorts of the tests: reals, booleans, and distributions over booleans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sort {
    R,
    B,
    D,
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

pub use Sort::{B, D, R};

/// The opaque operations of the tests.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// `R → B`: `x > 0`.
    Gt0,
    /// `R ⊗ R → B`: `x < y`.
    Lt,
    /// `R ⊗ R → B`: `x == y`.
    Eq,
    /// `R ⊗ R → R`: `x + y`.
    Add,
    /// `I → R`: a literal.
    Lit(&'static str),
    /// `B ⊗ B → B`: disjunction, as opaque to the tape as any other operation.
    Or,
    /// `I → B`: the constant `false`, also opaque.
    False,
    /// `D → B`: draws a boolean. The only non-deterministic operation.
    Sample,
    /// `B → I ⊕ I`: not an operation a tape accepts, since its codomain is a
    /// sum. Only used to test that it is rejected.
    Choice,
}

impl Op {
    fn signature(&self) -> (Vec<Sort>, Vec<Sort>) {
        match self {
            Op::Gt0 => (vec![R], vec![B]),
            Op::Lt | Op::Eq => (vec![R, R], vec![B]),
            Op::Add => (vec![R, R], vec![R]),
            Op::Lit(_) => (vec![], vec![R]),
            Op::Or => (vec![B, B], vec![B]),
            Op::False => (vec![], vec![B]),
            Op::Sample => (vec![D], vec![B]),
            Op::Choice => (vec![B], vec![]),
        }
    }
}

impl Boundary<Sort> for Op {
    fn dom(&self) -> Poly<Sort> {
        Poly::from(Mono::from_iter(self.signature().0))
    }

    fn cod(&self) -> Poly<Sort> {
        if *self == Op::Choice {
            return Mono::unit() + Mono::unit();
        }
        Poly::from(Mono::from_iter(self.signature().1))
    }
}

impl Operation<Sort> for Op {
    fn is_deterministic(&self) -> bool {
        !matches!(self, Op::Sample)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Gt0 => f.write_str("gt0"),
            Op::Lt => f.write_str("lt"),
            Op::Eq => f.write_str("eq"),
            Op::Add => f.write_str("add"),
            Op::Lit(l) => f.write_str(l),
            Op::Or => f.write_str("or"),
            Op::False => f.write_str("false"),
            Op::Sample => f.write_str("sample"),
            Op::Choice => f.write_str("choice"),
        }
    }
}

pub fn poly(summands: &[&[Sort]]) -> Poly<Sort> {
    summands
        .iter()
        .map(|m| Mono::from_iter(m.iter().copied()))
        .collect()
}

pub fn mono(sorts: &[Sort]) -> Poly<Sort> {
    poly(&[sorts])
}

pub fn node(kind: NodeKind<Op>, args: (u32, u32), n_outs: u32) -> Node<Op> {
    Node {
        kind,
        args: Span::new(args.0, args.1),
        n_outs,
    }
}

pub fn split(binds: (u32, u32), cond: u32, hi: u32, lo: u32) -> Route {
    Route::Split {
        binds: Span::new(binds.0, binds.1),
        cond: Value(cond),
        hi: RouteId(hi),
        lo: RouteId(lo),
    }
}

pub fn end(output: u32, outs: (u32, u32)) -> Route {
    Route::End {
        binds: Span::EMPTY,
        output,
        outs: Span::new(outs.0, outs.1),
    }
}

pub fn values(vs: &[u32]) -> Vec<Value> {
    vs.iter().copied().map(Value).collect()
}

/// The example of the module documentation, `(R ⊗ D) ⊕ I → R ⊕ I`:
///
/// ```text
/// match input {
///     inl (x, d) => if gt0(x) then (let b = sample(d);
///                                   if b then inl x else inr ())
///                   else inr ()
///     inr ()     => inl zero()
/// }
/// ```
pub fn sampled_guard() -> TapeParts<Sort, Op> {
    TapeParts {
        boolean: B,
        dom: poly(&[&[R, D], &[]]),
        cod: poly(&[&[R], &[]]),
        arms: vec![
            Arm {
                nodes: Span::new(0, 2),
                routes: Span::new(0, 4),
            },
            Arm {
                nodes: Span::new(2, 1),
                routes: Span::new(4, 1),
            },
        ],
        nodes: vec![
            node(NodeKind::Det(Op::Gt0), (0, 1), 1),
            node(
                NodeKind::NonDet {
                    op: Op::Sample,
                    level: 0,
                },
                (1, 1),
                1,
            ),
            node(NodeKind::Det(Op::Lit("zero")), (0, 0), 1),
        ],
        args: values(&[0, 1]),
        routes: vec![
            end(0, (0, 1)),
            end(1, (0, 0)),
            split((0, 1), 3, 0, 1),
            split((0, 0), 2, 2, 1),
            end(0, (1, 1)),
        ],
        bound: vec![NodeId(1)],
        outs: values(&[0, 0]),
    }
}

/// The translation of
///
/// ```text
/// if x < y or x < z:
///     new_x = x + 1.0
/// else:
///     new_x = 0.0
/// at_target = new_x == y or new_x == z
/// return new_x, (1.0 if at_target else 0.0), at_target, False
/// ```
///
/// as a tape `R ⊗ R ⊗ R → R ⊗ R ⊗ B ⊗ B`.
///
/// Booleans are opaque, so each `or` is a single split on the value the
/// theory's `or` computes, and `at_target` is output as computed, not as a
/// constant, although each end's path has already tested it. The code after
/// the `if` is copied into both branches, so `at_target` is computed twice,
/// from `x + 1.0` (`v10`) and from `0.0` (`v15`).
pub fn program() -> TapeParts<Sort, Op> {
    let (x, y, z) = (0, 1, 2);
    TapeParts {
        boolean: B,
        dom: mono(&[R, R, R]),
        cod: mono(&[R, R, B, B]),
        arms: vec![Arm {
            nodes: Span::new(0, 13),
            routes: Span::new(0, 7),
        }],
        nodes: vec![
            node(NodeKind::Det(Op::Lt), (0, 2), 1),         // v3 = x < y
            node(NodeKind::Det(Op::Lt), (2, 2), 1),         // v4 = x < z
            node(NodeKind::Det(Op::Or), (4, 2), 1),         // v5 = v3 or v4
            node(NodeKind::Det(Op::Lit("1.0")), (0, 0), 1), // v6
            node(NodeKind::Det(Op::Add), (6, 2), 1),        // v7 = x + 1.0
            node(NodeKind::Det(Op::Eq), (8, 2), 1),         // v8 = v7 == y
            node(NodeKind::Det(Op::Eq), (10, 2), 1),        // v9 = v7 == z
            node(NodeKind::Det(Op::Or), (12, 2), 1),        // v10 = at_target
            node(NodeKind::Det(Op::False), (0, 0), 1),      // v11
            node(NodeKind::Det(Op::Lit("0.0")), (0, 0), 1), // v12
            node(NodeKind::Det(Op::Eq), (14, 2), 1),        // v13 = 0.0 == y
            node(NodeKind::Det(Op::Eq), (16, 2), 1),        // v14 = 0.0 == z
            node(NodeKind::Det(Op::Or), (18, 2), 1),        // v15 = at_target
        ],
        args: values(&[
            x, y, x, z, 3, 4, x, 6, 7, y, 7, z, 8, 9, 12, y, 12, z, 13, 14,
        ]),
        routes: vec![
            end(0, (0, 4)),          // r0: (x+1, 1.0, at_target, false)
            end(0, (4, 4)),          // r1: (x+1, 0.0, at_target, false)
            split((0, 0), 10, 0, 1), // r2: at_target, then-branch
            end(0, (8, 4)),          // r3: (0.0, 1.0, at_target, false)
            end(0, (12, 4)),         // r4: (0.0, 0.0, at_target, false)
            split((0, 0), 15, 3, 4), // r5: at_target, else-branch
            split((0, 0), 5, 2, 5),  // r6: x < y or x < z, the root
        ],
        bound: vec![],
        outs: values(&[7, 6, 10, 11, 7, 12, 10, 11, 12, 6, 15, 11, 12, 12, 15, 11]),
    }
}
