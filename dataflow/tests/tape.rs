//! Tests for the tape data structures: two tapes built by hand in normal
//! form, their boundaries and listings, and the well-formedness check
//! rejecting broken variants of them.

mod common;

use base::boundary::Boundary;
use common::*;
use dataflow::tape::{Arm, NodeId, NodeKind, Route, RouteId, Span, Tape, TapeParts, Value};

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
fn sampled_guard_listing() {
    let tape = Tape::new(sampled_guard()).unwrap();
    let expected = "\
(R ⊗ D) ⊕ I → R ⊕ I
arm 0: R ⊗ D
  in v0: R, v1: D
  n0  v2 = gt0(v0)
  n1  v3 = sample@0(v1)
  r0  end 0 (v0)
  r1  end 1 ()
  r2  bind n1; split v3 ? r0 : r1
  r3  split v2 ? r2 : r1
arm 1: I
  in (none)
  n2  v0 = zero()
  r4  end 0 (v0)
";
    assert_eq!(tape.to_string(), expected);
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
fn program_listing() {
    let tape = Tape::new(program()).unwrap();
    let expected = "\
R ⊗ R ⊗ R → R ⊗ R ⊗ B ⊗ B
arm 0: R ⊗ R ⊗ R
  in v0: R, v1: R, v2: R
  n0  v3 = lt(v0, v1)
  n1  v4 = lt(v0, v2)
  n2  v5 = or(v3, v4)
  n3  v6 = 1.0()
  n4  v7 = add(v0, v6)
  n5  v8 = eq(v7, v1)
  n6  v9 = eq(v7, v2)
  n7  v10 = or(v8, v9)
  n8  v11 = false()
  n9  v12 = 0.0()
  n10 v13 = eq(v12, v1)
  n11 v14 = eq(v12, v2)
  n12 v15 = or(v13, v14)
  r0  end 0 (v7, v6, v10, v11)
  r1  end 0 (v7, v12, v10, v11)
  r2  split v10 ? r0 : r1
  r3  end 0 (v12, v6, v15, v11)
  r4  end 0 (v12, v12, v15, v11)
  r5  split v15 ? r3 : r4
  r6  split v5 ? r2 : r5
";
    assert_eq!(tape.to_string(), expected);
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
    parts.routes[2] = split((0, 0), 3, 0, 1);
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
    // bind the sample at the end `r0` instead of at the split `r2` that reads
    // it (with the level it would have there)
    let mut parts = sampled_guard();
    parts.routes[2] = split((0, 0), 3, 0, 1);
    parts.routes[0] = Route::End {
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
