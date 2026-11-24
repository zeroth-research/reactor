//! Tests for the tape data structures: two tapes built by hand in normal
//! form, their boundaries, and the well-formedness check
//! rejecting broken variants of them.

use base::boundary::{Boundary, Mono, Poly};
use dataflow::tape::{
    Arm, Node, NodeId, NodeKind, Operation, Route, RouteId, Span, Tape, TapeParts, Value,
};

/// The sorts of the tests: reals, booleans, and distributions over booleans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sort {
    R,
    B,
    D,
}

use Sort::{B, D, R};

/// The opaque operations of the tests.
#[derive(Clone, Debug, PartialEq)]
enum Op {
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
        }
    }
}

impl Boundary<Sort> for Op {
    fn dom(&self) -> Poly<Sort> {
        Poly::from(Mono::from_iter(self.signature().0))
    }

    fn cod(&self) -> Poly<Sort> {
        Poly::from(Mono::from_iter(self.signature().1))
    }
}

impl Operation<Sort> for Op {
    fn is_deterministic(&self) -> bool {
        !matches!(self, Op::Sample)
    }
}

fn poly(summands: &[&[Sort]]) -> Poly<Sort> {
    summands
        .iter()
        .map(|m| Mono::from_iter(m.iter().copied()))
        .collect()
}

fn mono(sorts: &[Sort]) -> Poly<Sort> {
    poly(&[sorts])
}

fn node(kind: NodeKind<Op>, args: (u32, u32), n_outs: u32) -> Node<Op> {
    Node {
        kind,
        args: Span::new(args.0, args.1),
        n_outs,
    }
}

fn split(binds: (u32, u32), cond: u32, hi: u32, lo: u32) -> Route {
    Route::Split {
        binds: Span::new(binds.0, binds.1),
        cond: Value(cond),
        hi: RouteId(hi),
        lo: RouteId(lo),
    }
}

fn end(output: u32, outs: (u32, u32)) -> Route {
    Route::End {
        binds: Span::EMPTY,
        output,
        outs: Span::new(outs.0, outs.1),
    }
}

fn values(vs: &[u32]) -> Vec<Value> {
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
fn sampled_guard() -> TapeParts<Sort, Op> {
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
            node(NodeKind::Det(Op::Lit("zero")), (2, 0), 1),
        ],
        args: values(&[0, 1]),
        routes: vec![
            end(1, (0, 0)),
            end(0, (0, 1)),
            split((0, 1), 3, 1, 0),
            split((0, 0), 2, 2, 0),
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
fn program() -> TapeParts<Sort, Op> {
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
            node(NodeKind::Det(Op::Lt), (0, 2), 1),          // v3 = x < y
            node(NodeKind::Det(Op::Lt), (2, 2), 1),          // v4 = x < z
            node(NodeKind::Det(Op::Or), (4, 2), 1),          // v5 = v3 or v4
            node(NodeKind::Det(Op::Lit("1.0")), (6, 0), 1),  // v6
            node(NodeKind::Det(Op::Add), (6, 2), 1),         // v7 = x + 1.0
            node(NodeKind::Det(Op::Eq), (8, 2), 1),          // v8 = v7 == y
            node(NodeKind::Det(Op::Eq), (10, 2), 1),         // v9 = v7 == z
            node(NodeKind::Det(Op::Or), (12, 2), 1),         // v10 = at_target
            node(NodeKind::Det(Op::False), (14, 0), 1),      // v11
            node(NodeKind::Det(Op::Lit("0.0")), (14, 0), 1), // v12
            node(NodeKind::Det(Op::Eq), (14, 2), 1),         // v13 = 0.0 == y
            node(NodeKind::Det(Op::Eq), (16, 2), 1),         // v14 = 0.0 == z
            node(NodeKind::Det(Op::Or), (18, 2), 1),         // v15 = at_target
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

/// The error of `Tape::new` on `parts`, which must fail.
fn error(parts: TapeParts<Sort, Op>) -> String {
    Tape::new(parts).expect_err("the tape should be ill formed")
}

#[test]
fn sampled_guard_is_well_formed() {
    let tape = Tape::new(sampled_guard()).unwrap();
    assert_eq!(tape.arms().len(), 2);
    assert_eq!(tape.outputs().len(), 2);
    assert_eq!(Boundary::dom(&tape), poly(&[&[R, D], &[]]));
    assert_eq!(Boundary::cod(&tape), poly(&[&[R], &[]]));

    // the raw parts declare the same boundary
    let parts = tape.into_parts();
    assert_eq!(parts.dom(), poly(&[&[R, D], &[]]));
    assert_eq!(parts.cod(), poly(&[&[R], &[]]));
}

#[test]
fn boundaries_of_the_parts() {
    let tape = Tape::new(sampled_guard()).unwrap();

    // an arm reads its input summand and may end in any output summand
    let arm = tape.arm(0);
    assert_eq!(arm.dom(), mono(&[R, D]));
    assert_eq!(arm.cod(), poly(&[&[R], &[]]));

    // a node has the boundary of its operation, and knows its outputs
    let nodes: Vec<_> = arm.nodes().collect();
    assert_eq!(nodes[1].dom(), mono(&[D]));
    assert_eq!(nodes[1].cod(), mono(&[B]));
    assert_eq!(nodes[1].outputs().collect::<Vec<_>>(), vec![Value(3)]);

    // a node needs no context: the raw `Node` has the same boundary
    let raw = &tape.nodes()[1];
    assert_eq!(raw.dom(), mono(&[D]));
    assert_eq!(raw.cod(), mono(&[B]));
    assert_eq!(raw.kind.dom(), mono(&[D]));

    // a split may still end in any output summand; an end node in its own
    let root = arm.root();
    assert_eq!(root.id(), RouteId(3));
    assert_eq!(root.cod(), poly(&[&[R], &[]]));
    let (hi, _) = root.children().unwrap();
    assert_eq!(hi.binds(), &[NodeId(1)]);
    let (to_r, to_i) = hi.children().unwrap();
    assert_eq!(to_r.dom(), mono(&[R, D]));
    assert_eq!(to_r.cod(), mono(&[R]));
    assert_eq!(to_r.outs(), Some(&[Value(0)][..]));
    assert_eq!(to_i.cod(), mono(&[]));

    // the second arm has no inputs
    assert_eq!(tape.arm(1).dom(), mono(&[]));
}

#[test]
fn program_is_well_formed() {
    let tape = Tape::new(program()).unwrap();
    assert_eq!((tape.arms().len(), tape.outputs().len()), (1, 1));
    assert_eq!(Boundary::cod(&tape), mono(&[R, R, B, B]));
    let ends = tape.arm(0).routes().filter(|d| d.outs().is_some()).count();
    assert_eq!(ends, 4);
}

#[test]
fn rejects_wrong_number_of_arms() {
    let mut parts = sampled_guard();
    parts.dom = poly(&[&[R, D]]);
    assert!(error(parts).contains("arms"));
}

#[test]
fn rejects_forward_argument() {
    let mut parts = sampled_guard();
    parts.args[0] = Value(3); // gt0 reads its own arm's later value
    assert!(error(parts).contains("not defined before"));
}

#[test]
fn rejects_wrong_argument_sort() {
    let mut parts = sampled_guard();
    parts.args[0] = Value(1); // gt0 reads the distribution `d`
    assert!(error(parts).contains("sorts"));
}

#[test]
fn rejects_wrong_determinism() {
    let mut parts = sampled_guard();
    parts.nodes[0].kind = NodeKind::NonDet {
        op: Op::Gt0,
        level: 0,
    };
    assert!(error(parts).contains("deterministic"));
}

#[test]
fn rejects_child_in_another_arm() {
    let mut parts = sampled_guard();
    parts.routes[3] = split((0, 0), 2, 4, 0); // r4 belongs to arm 1
    assert!(error(parts).contains("earlier route of arm 0"));
}

#[test]
fn rejects_leaf_of_wrong_sort() {
    let mut parts = sampled_guard();
    parts.outs[0] = Value(2); // a boolean into `R`
    assert!(error(parts).contains("output 0"));
}

#[test]
fn rejects_unbound_draw() {
    let mut parts = sampled_guard();
    parts.routes[2] = split((0, 0), 3, 1, 0);
    assert!(error(parts).contains("never bound"));
}

#[test]
fn rejects_wrong_level() {
    let mut parts = sampled_guard();
    parts.nodes[1].kind = NodeKind::NonDet {
        op: Op::Sample,
        level: 1,
    };
    assert!(error(parts).contains("level"));
}

#[test]
fn rejects_use_before_draw() {
    // bind the sample at the end `r1` instead of at the split `r2` that reads
    // it (with the level it would have there)
    let mut parts = sampled_guard();
    parts.routes[2] = split((0, 0), 3, 1, 0);
    parts.routes[1] = Route::End {
        binds: Span::new(0, 1),
        output: 0,
        outs: Span::new(0, 1),
    };
    parts.nodes[1].kind = NodeKind::NonDet {
        op: Op::Sample,
        level: 0,
    };
    assert!(error(parts).contains("not made on every path"));
}

#[test]
fn rejects_unreachable_route() {
    let mut parts = sampled_guard();
    // neither split leads to r0 any more
    parts.routes[2] = split((0, 1), 3, 1, 1);
    parts.routes[3] = split((0, 0), 2, 2, 1);
    assert!(error(parts).contains("r0 is not reachable"));
}

#[test]
fn rejects_shared_binder_at_different_depths() {
    // `D ⊗ B → B ⊕ I`: the end r0 draws a second sample, but is reached
    // both after the first draw (from r2) and before any draw (from r3), so
    // its draw has no well-defined level.
    let sample = |level| NodeKind::NonDet {
        op: Op::Sample,
        level,
    };
    let parts = TapeParts {
        boolean: B,
        dom: mono(&[D, B]),
        cod: poly(&[&[B], &[]]),
        arms: vec![Arm {
            nodes: Span::new(0, 2),
            routes: Span::new(0, 4),
        }],
        nodes: vec![node(sample(0), (0, 1), 1), node(sample(1), (0, 1), 1)],
        args: values(&[0]),
        routes: vec![
            Route::End {
                binds: Span::new(1, 1),
                output: 0,
                outs: Span::new(0, 1),
            },
            end(1, (0, 0)),
            split((0, 1), 2, 0, 1),
            split((0, 0), 1, 2, 0),
        ],
        bound: vec![NodeId(0), NodeId(1)],
        outs: values(&[3]),
    };
    assert!(error(parts).contains("reached after 0 draws on one path and after 1"));
}

/// `B ⊗ B → I ⊕ I`, branching on `a ∨ b`, either with one split on the value
/// of the theory's `or`, or with nested splits on `a` and on `b`.
fn disjunction(nested: bool) -> TapeParts<Sort, Op> {
    let (nodes, routes) = if nested {
        let routes = vec![
            end(0, (0, 0)),
            end(1, (0, 0)),
            split((0, 0), 1, 0, 1), // b
            split((0, 0), 0, 0, 2), // a
        ];
        (vec![], routes)
    } else {
        let nodes = vec![node(NodeKind::Det(Op::Or), (0, 2), 1)];
        let routes = vec![end(0, (0, 0)), end(1, (0, 0)), split((0, 0), 2, 0, 1)];
        (nodes, routes)
    };
    TapeParts {
        boolean: B,
        dom: mono(&[B, B]),
        cod: poly(&[&[], &[]]),
        arms: vec![Arm {
            nodes: Span::new(0, nodes.len() as u32),
            routes: Span::new(0, routes.len() as u32),
        }],
        nodes,
        args: values(&[0, 1]),
        routes,
        bound: vec![],
        outs: vec![],
    }
}

#[test]
fn booleans_are_opaque() {
    // Both are well formed, but they are different tapes for the same
    // function: the tape does not know that `or` is a disjunction.
    let opaque = Tape::new(disjunction(false)).unwrap();
    let nested = Tape::new(disjunction(true)).unwrap();
    assert_ne!(opaque, nested);
}

#[test]
fn rejects_non_boolean_condition() {
    let mut parts = sampled_guard();
    parts.routes[3] = split((0, 0), 0, 2, 0); // splits on `x`, a real
    assert!(error(parts).contains("not the boolean sort"));
}
