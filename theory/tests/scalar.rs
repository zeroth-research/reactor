//! Tests for the operations of `scalar`, one module per module of `scalar`.

/// Tests for `scalar::sort`.
mod sort {
    use base::differentiable::T;
    use theory::scalar::sort::*;
    use theory::sort::Zero;

    #[test]
    fn an_enumeration_is_the_set_of_its_values() {
        let e = Enum::new(["idle", "busy"]).unwrap();
        // the order and the repetitions of its values do not matter
        assert_eq!(e, Enum::new(["busy", "idle", "busy"]).unwrap());
        assert_ne!(e, Enum::new(["idle", "busy", "done"]).unwrap());
        assert_eq!(e.len(), 2);
        assert!(e.contains("idle") && e.contains("busy"));
        assert!(!e.contains("done"));
        let values: Vec<&str> = e.values().iter().map(|v| &**v).collect();
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
        let values: Vec<&str> = union.values().iter().map(String::as_str).collect();
        assert_eq!(values, ["a", "b", "c", "d", "e", "f", "g"]);
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
        assert_eq!(Bool.to_string(), "Bool()");
        assert_eq!(Event.to_string(), "Event()");
        assert_eq!(Any::from(Event).to_string(), "Event()");
        assert_eq!(Any::from(e).to_string(), r#"Enum(["busy", "idle"])"#);
    }

    #[test]
    fn the_scalar_sorts_are_discrete() {
        let e = Enum::new(["idle", "busy"]).unwrap();
        assert_eq!(T(&e), Zero);
        assert_eq!(T(e.clone()), Zero);
        assert_eq!(T(Bool), Zero);
        assert_eq!(T(T(Bool)), Zero);
        assert_eq!(T(Event), Zero);
    }
}

/// Tests for `scalar::enum`.
mod r#enum {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::scalar::r#enum::*;
    // `Into`, the operation, shadows the trait of the prelude
    use std::convert::Into as _;
    use theory::scalar::sort::{Bool, Enum};

    fn state() -> Enum {
        Enum::new(["idle", "busy", "done"]).unwrap()
    }

    fn other() -> Enum {
        Enum::new(["on", "off"]).unwrap()
    }

    #[test]
    fn a_constant_is_a_value_of_its_enumeration() {
        let c = Val::with_cod("busy", [state()]).unwrap();
        assert_eq!(c.dom(), Poly::unit());
        assert_eq!(c.cod(), mono![state().into()].into());
        assert_eq!(c.to_string(), "busy");
        assert!(Val::with_cod("on", [state()]).is_err());
    }

    #[test]
    fn comparisons_read_two_values_of_an_enumeration_and_write_a_bool() {
        for op in [
            Any::from(Eq::with_dom([state(), state()]).unwrap()),
            Any::from(Ne::with_dom([state(), state()]).unwrap()),
        ] {
            assert_eq!(op.dom(), mono![state().into(); 2].into());
            assert_eq!(op.cod(), mono![Bool.into()].into());
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
        let op = Ite::with_dom([Bool.into(), state().into(), state().into()]).unwrap();
        assert_eq!(
            op.dom(),
            mono![Bool.into(), state().into(), state().into()].into()
        );
        assert_eq!(op.cod(), mono![state().into()].into());
        assert_eq!(op, Ite::with_cod([state()]).unwrap());
        assert_eq!(op.to_string(), "Ite");
    }

    #[test]
    fn ite_reads_a_bool_and_branches_of_the_same_enumeration() {
        // branches of different enumerations
        assert!(Ite::with_dom([Bool.into(), state().into(), other().into()]).is_err());
        // a condition that is not a boolean
        assert!(Ite::with_dom([state().into(), state().into(), state().into()]).is_err());
        // branches that are not enumerations
        assert!(Ite::with_dom([Bool.into(), Bool.into(), Bool.into()]).is_err());
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
        let c: Any = Val::with_cod("idle", [state()]).unwrap().into();
        assert_eq!(c.to_string(), "idle");
        assert_eq!(c.cod(), mono![state().into()].into());
        let ite: Any = Ite::with_cod([state()]).unwrap().into();
        assert_eq!(ite.to_string(), "Ite");
        assert_eq!(ite.cod(), mono![state().into()].into());
    }
}

/// Tests for `scalar::bool`.
mod bool {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::scalar::bool::*;
    use theory::scalar::sort::Bool;

    #[test]
    fn the_constants_read_nothing_and_write_a_bool() {
        for op in [Any::from(True), Any::from(False)] {
            assert_eq!(op.dom(), Poly::unit());
            assert_eq!(op.cod(), mono![Bool].into());
        }
        assert_ne!(Any::from(True), Any::from(False));
    }

    #[test]
    fn the_operations_read_and_write_bools() {
        let ops: [(Any, usize); 7] = [
            (And.into(), 2),
            (Or.into(), 2),
            (Xor.into(), 2),
            (Not.into(), 1),
            (Eq.into(), 2),
            (Ne.into(), 2),
            (Ite.into(), 3),
        ];
        for (op, arity) in ops {
            assert_eq!(
                op.dom(),
                std::iter::repeat_n(Bool, arity)
                    .collect::<base::boundary::Mono<_>>()
                    .into()
            );
            assert_eq!(op.cod(), mono![Bool].into());
        }
    }

    #[test]
    fn display() {
        let names = [
            (Any::from(And), "And"),
            (Any::from(Or), "Or"),
            (Any::from(Xor), "Xor"),
            (Any::from(Not), "Not"),
            (Any::from(Eq), "Eq"),
            (Any::from(Ne), "Ne"),
            (Any::from(Ite), "Ite"),
            (Any::from(True), "True"),
            (Any::from(False), "False"),
        ];
        for (op, name) in names {
            assert_eq!(op.to_string(), name);
        }
    }
}

/// Tests for `scalar::event`.
mod event {
    use base::boundary::Boundary;
    use base::mono;
    use theory::scalar::event::*;
    use theory::scalar::sort::{Bool, Event};

    #[test]
    fn comparisons_read_two_events_and_write_a_bool() {
        for op in [Any::from(Eq), Any::from(Ne)] {
            assert_eq!(op.dom(), mono![Event.into(), Event.into()].into());
            assert_eq!(op.cod(), mono![Bool.into()].into());
        }
    }

    #[test]
    fn toggle_and_hold_read_an_event_and_write_one() {
        for op in [Any::from(Toggle), Any::from(Hold)] {
            assert_eq!(op.dom(), mono![Event.into()].into());
            assert_eq!(op.cod(), mono![Event.into()].into());
        }
        assert_ne!(Any::from(Toggle), Any::from(Hold));
    }

    #[test]
    fn ite_chooses_by_a_bool_between_events() {
        let op = Any::from(Ite);
        assert_eq!(
            op.dom(),
            mono![Bool.into(), Event.into(), Event.into()].into()
        );
        assert_eq!(op.cod(), mono![Event.into()].into());
    }

    #[test]
    fn display() {
        for (op, name) in [
            (Any::from(Eq), "Eq"),
            (Any::from(Ne), "Ne"),
            (Any::from(Toggle), "Toggle"),
            (Any::from(Hold), "Hold"),
            (Any::from(Ite), "Ite"),
        ] {
            assert_eq!(op.to_string(), name);
        }
    }
}

/// Tests for `scalar::Any`, the operations of all the scalar theories.
mod any {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::scalar;
    use theory::scalar::sort::{Any, Bool, Enum, Event};
    use theory::scalar::{bool, r#enum, event};

    #[test]
    fn the_operations_of_each_theory_keep_their_boundary_and_display() {
        let e = Enum::new(["idle", "busy"]).unwrap();
        let val = r#enum::Val::with_cod("idle", [e.clone()]).unwrap();
        let cases: [(scalar::Any, Poly<Any>, Poly<Any>, &str); 4] = [
            (
                r#enum::Any::from(val).into(),
                Poly::unit(),
                mono![Any::from(e.clone())].into(),
                "idle",
            ),
            (
                r#enum::Any::from(r#enum::Eq::with_dom([e.clone(), e.clone()]).unwrap()).into(),
                mono![Any::from(e.clone()), Any::from(e)].into(),
                mono![Any::from(Bool)].into(),
                "Eq",
            ),
            (
                bool::Any::from(bool::And).into(),
                mono![Any::from(Bool), Any::from(Bool)].into(),
                mono![Any::from(Bool)].into(),
                "And",
            ),
            (
                event::Any::from(event::Toggle).into(),
                mono![Any::from(Event)].into(),
                mono![Any::from(Event)].into(),
                "Toggle",
            ),
        ];
        for (op, dom, cod, name) in cases {
            assert_eq!(op.dom(), dom);
            assert_eq!(op.cod(), cod);
            assert_eq!(op.to_string(), name);
        }
    }

    #[test]
    fn the_constants_of_the_booleans_read_nothing() {
        let op: scalar::Any = bool::Any::from(bool::True).into();
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), mono![Any::from(Bool)].into());
    }
}
