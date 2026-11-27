//! Tests for the scalar theories, `enum` and `event`: their sorts, their
//! operations, one module per theory, and those operations in `Op`.

/// Tests for the sorts of the scalar theories.
mod sort {
    use base::differentiable::T;
    use theory::Any;
    use theory::zero::Real0;
    use theory::{r#enum::Enum, event::Event};

    #[test]
    fn an_enumeration_is_the_set_of_its_values() {
        let e = Enum::new(["idle", "busy"]).unwrap();
        // the order of its values does not matter, and none is repeated
        assert_eq!(e, Enum::new(["busy", "idle"]).unwrap());
        assert!(Enum::new(["busy", "idle", "busy"]).is_err());
        assert_ne!(e, Enum::new(["idle", "busy", "done"]).unwrap());
        assert_eq!(e.len(), 2);
        assert!(e.contains("idle") && e.contains("busy"));
        assert!(!e.contains("done"));
        assert_eq!(e.as_slice(), ["busy", "idle"]);
        // it iterates over its values, sorted
        let values: Vec<&str> = e.iter().map(String::as_str).collect();
        assert_eq!(values, ["busy", "idle"]);
        let values: Vec<&String> = (&e).into_iter().collect();
        assert_eq!(values, ["busy", "idle"]);
    }

    #[test]
    fn the_union_of_enumerations_holds_the_values_of_both() {
        let a = Enum::new(["idle", "busy"]).unwrap();
        let b = Enum::new(["busy", "done"]).unwrap();
        let union = Enum::new(["idle", "busy", "done"]).unwrap();
        assert_eq!(a.union(&b), union);
        assert_eq!(b.union(&a), union);
        assert_eq!(a.union(&a), a);
        assert!(a.is_subset(&union) && b.is_subset(&union) && a.is_subset(&a));
        assert!(!union.is_subset(&a) && !a.is_subset(&b));
        // the union of disjoint enumerations, interleaved, keeps them sorted
        let c = Enum::new(["a", "c", "e"]).unwrap();
        let d = Enum::new(["b", "d", "f", "g"]).unwrap();
        let union = c.union(&d);
        assert_eq!(union.as_slice(), ["a", "b", "c", "d", "e", "f", "g"]);
        // inclusion needs every value, wherever it falls
        assert!(!Enum::new(["a", "z"]).unwrap().is_subset(&c));
        assert!(!Enum::new(["0"]).unwrap().is_subset(&c));
        assert!(Enum::new(["c"]).unwrap().is_subset(&c));
    }

    #[test]
    fn an_enumeration_has_a_value() {
        assert!(Enum::new(Vec::<String>::new()).is_err());
        // one value is enough: a sort with a single value
        assert_eq!(Enum::new(["only"]).unwrap().len(), 1);
    }

    #[test]
    fn display() {
        let e = Enum::new(["idle", "busy"]).unwrap();
        assert_eq!(e.to_string(), r#"Enum(["busy", "idle"])"#);
        // an event by its identity, as a wire
        let ev = Event::new();
        let name = format!("Event#{}", ev.id());
        assert_eq!(ev.to_string(), name);
        assert_eq!(Any::from(ev).to_string(), name);
        assert_eq!(Any::from(e).to_string(), r#"Enum(["busy", "idle"])"#);
    }

    #[test]
    fn the_scalar_sorts_are_discrete() {
        let e = Enum::new(["idle", "busy"]).unwrap();
        assert_eq!(T(&e), Real0);
        assert_eq!(T(e.clone()), Real0);
        assert_eq!(T(Event::new()), Real0);
    }

    #[test]
    fn each_event_is_its_own_sort() {
        let (e, f) = (Event::new(), Event::new());
        assert_ne!(e, f);
        assert_ne!(e.id(), f.id());
        // a copy is the same event
        let g = e;
        assert_eq!(g, e);
        assert_ne!(Any::from(e), Any::from(f));
    }
}

/// Tests for `enum`.
mod r#enum {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::r#enum::*;
    // `Into`, the operation, shadows the trait of the prelude
    use std::convert::Into as _;
    use theory::bool::Bool;
    use theory::r#enum::Enum;

    /// The boolean of the shape `[1, 1]` that the comparisons write.
    fn b() -> Bool {
        Bool::new_unchecked([1, 1])
    }

    fn state() -> Enum {
        Enum::new(["idle", "busy", "done"]).unwrap()
    }

    fn other() -> Enum {
        Enum::new(["on", "off"]).unwrap()
    }

    #[test]
    fn a_constant_is_a_value_of_its_enumeration() {
        let c = Sym::with_cod("busy", [state()]).unwrap();
        assert_eq!(c.dom(), Poly::unit());
        assert_eq!(c.cod(), mono![state().into()].into());
        assert_eq!(c.to_string(), "busy");
        assert!(Sym::with_cod("on", [state()]).is_err());
    }

    #[test]
    fn comparisons_read_two_values_of_an_enumeration_and_write_a_bool() {
        for op in [
            Op::from(Eq::with_dom([state(), state()]).unwrap()),
            Op::from(Ne::with_dom([state(), state()]).unwrap()),
        ] {
            assert_eq!(op.dom(), mono![state().into(); 2].into());
            assert_eq!(op.cod(), mono![b().into()].into());
        }
        assert_eq!(Eq::with_dom([state(), state()]).unwrap().to_string(), "Eq");
        assert_eq!(Ne::with_dom([state(), state()]).unwrap().to_string(), "Ne");
    }

    #[test]
    fn comparisons_read_the_same_enumeration() {
        assert!(Eq::with_dom([state(), other()]).is_err());
        assert!(Ne::with_dom([state(), other()]).is_err());
        // the same values in another order make the same enumeration
        let reordered = Enum::new(["done", "busy", "idle"]).unwrap();
        assert!(Eq::with_dom([state(), reordered]).is_ok());
    }

    #[test]
    fn ite_chooses_by_a_bool_between_values_of_an_enumeration() {
        let op = Ite::with_dom([b().into(), state().into(), state().into()]).unwrap();
        assert_eq!(
            op.dom(),
            mono![b().into(), state().into(), state().into()].into()
        );
        assert_eq!(op.cod(), mono![state().into()].into());
        assert_eq!(op, Ite::with_cod([state()]).unwrap());
        assert_eq!(op.to_string(), "Ite");
    }

    #[test]
    fn ite_reads_a_bool_and_branches_of_the_same_enumeration() {
        // branches of different enumerations
        assert!(Ite::with_dom([b().into(), state().into(), other().into()]).is_err());
        // a condition that is not a boolean
        assert!(Ite::with_dom([state().into(), state().into(), state().into()]).is_err());
        // branches that are not enumerations
        assert!(Ite::with_dom([b().into(), b().into(), b().into()]).is_err());
        // a condition of another shape
        assert!(
            Ite::with_dom([
                Bool::new_unchecked([2, 2]).into(),
                state().into(),
                state().into()
            ])
            .is_err()
        );
    }

    #[test]
    fn into_converts_a_value_to_an_enumeration_holding_all_the_values() {
        let small = Enum::new(["idle", "busy"]).unwrap();
        let op = Into::new(small.clone(), state()).unwrap();
        assert_eq!(op.dom(), mono![small.clone().into()].into());
        assert_eq!(op.cod(), mono![state().into()].into());
        assert_eq!(op.to_string(), "Into");
        // into itself: the identity
        assert!(Into::new(state(), state()).is_ok());
        // a codomain missing values of the domain
        assert!(Into::new(state(), small).is_err());
        assert!(Into::new(state(), other()).is_err());
    }

    #[test]
    fn values_of_different_enumerations_compare_through_into() {
        // x of {idle, busy}, y of {busy, off}: both into their union, then compared
        let x = Enum::new(["idle", "busy"]).unwrap();
        let y = Enum::new(["busy", "off"]).unwrap();
        let union = x.union(&y);
        let into_x = Into::new(x, union.clone()).unwrap();
        let into_y = Into::new(y, union.clone()).unwrap();
        let eq = Eq::with_dom([union.clone(), union.clone()]).unwrap();
        // the conversions write the sort that the comparison reads
        assert_eq!(into_x.cod(), mono![union.clone().into()].into());
        assert_eq!(into_y.cod(), mono![union.clone().into()].into());
        assert_eq!(eq.dom(), mono![union.into(); 2].into());
    }

    #[test]
    fn the_operations_display_and_compute_their_boundary_through_op() {
        let c: Op = Sym::with_cod("idle", [state()]).unwrap().into();
        assert_eq!(c.to_string(), "idle");
        assert_eq!(c.cod(), mono![state().into()].into());
        let ite: Op = Ite::with_cod([state()]).unwrap().into();
        assert_eq!(ite.to_string(), "Ite");
        assert_eq!(ite.cod(), mono![state().into()].into());
    }
}

/// Tests for `event`.
mod event {
    use base::boundary::Boundary;
    use base::mono;
    use theory::Events;
    use theory::bool::Bool;
    use theory::event::Event;
    use theory::event::*;

    #[test]
    fn the_difference_of_two_values_of_an_event_is_a_bool() {
        let e = Event::new();
        let c: Events = Bool::new_unchecked([1, 1]).into();
        let op = Xor::with_dom([e.into(), e.into()]).unwrap();
        assert_eq!(op.dom(), mono![e.into(), e.into()].into());
        assert_eq!(op.cod(), mono![c].into());
    }

    #[test]
    fn the_shift_of_an_event_by_a_bool_is_a_value_of_it() {
        let e = Event::new();
        let c: Events = Bool::new_unchecked([1, 1]).into();
        let shift = Xor::with_dom([e.into(), c]).unwrap();
        assert_eq!(shift.dom(), mono![e.into(), c].into());
        assert_eq!(shift.cod(), mono![e.into()].into());
        // in either order
        let op = Xor::with_dom([c, e.into()]).unwrap();
        assert_eq!(op.dom(), mono![c, e.into()].into());
        assert_eq!(op.cod(), mono![e.into()].into());
        // each typing is its own operation
        let diff = Xor::with_dom([e.into(), e.into()]).unwrap();
        assert_ne!(op, shift);
        assert_ne!(shift, diff);
    }

    #[test]
    fn the_values_of_two_events_never_meet() {
        let (e, f) = (Event::new(), Event::new());
        let c: Events = Bool::new_unchecked([1, 1]).into();
        assert!(Xor::with_dom([e.into(), f.into()]).is_err());
        assert_ne!(
            Xor::with_dom([e.into(), e.into()]).unwrap(),
            Xor::with_dom([f.into(), f.into()]).unwrap()
        );
        assert_ne!(
            Xor::with_dom([e.into(), c]).unwrap(),
            Xor::with_dom([f.into(), c]).unwrap()
        );
        // no event is made of booleans, nor shifted by a wider boolean
        assert!(Xor::with_dom([c, c]).is_err());
        let wide: Events = Bool::new_unchecked([2, 2]).into();
        assert!(Xor::with_dom([e.into(), wide]).is_err());
        assert!(Xor::with_dom([wide, e.into()]).is_err());
    }

    #[test]
    fn display() {
        let e = Event::new();
        let c: Events = Bool::new_unchecked([1, 1]).into();
        for dom in [[e.into(), e.into()], [e.into(), c], [c, e.into()]] {
            assert_eq!(Xor::with_dom(dom).unwrap().to_string(), "Xor");
        }
    }
}

/// Tests for the operations of the scalar theories in `Op`, the operations of
/// all theories.
mod op {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::bool::Bool;
    use theory::{Any, Matrix};
    use theory::{r#enum, r#enum::Enum, event, event::Event};

    /// No matrix: the scalar operations carry none.
    #[derive(Debug)]
    struct Mat;

    impl Matrix for Mat {
        fn shape(&self) -> [usize; 2] {
            [1, 1]
        }
    }

    impl std::fmt::Display for Mat {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Mat")
        }
    }

    type Op = theory::Op<Mat>;

    #[test]
    fn the_operations_of_each_theory_keep_their_boundary_and_display() {
        let e = Enum::new(["idle", "busy"]).unwrap();
        let b = Bool::new_unchecked([1, 1]);
        let sym = r#enum::Sym::with_cod("idle", [e.clone()]).unwrap();
        let ev = Event::new();
        let cases: [(Op, Poly<Any>, Poly<Any>, &str); 3] = [
            (
                r#enum::Op::from(sym).into(),
                Poly::unit(),
                mono![Any::from(e.clone())].into(),
                "idle",
            ),
            (
                r#enum::Op::from(r#enum::Eq::with_dom([e.clone(), e.clone()]).unwrap()).into(),
                mono![Any::from(e.clone()), Any::from(e)].into(),
                mono![Any::from(b)].into(),
                "Eq",
            ),
            (
                event::Xor::with_dom([ev.into(), b.into()]).unwrap().into(),
                mono![Any::from(ev), Any::from(b)].into(),
                mono![Any::from(ev)].into(),
                "Xor",
            ),
        ];
        for (op, dom, cod, name) in cases {
            assert_eq!(op.dom(), dom);
            assert_eq!(op.cod(), cod);
            assert_eq!(op.to_string(), name);
        }
    }

    #[test]
    fn the_differences_write_the_booleans_of_the_matrix_theories() {
        let ev = Event::new();
        let op: Op = event::Xor::with_dom([ev.into(), ev.into()]).unwrap().into();
        assert_eq!(
            op.cod(),
            mono![Any::Bool(Bool::new_unchecked([1, 1]))].into()
        );
    }
}
