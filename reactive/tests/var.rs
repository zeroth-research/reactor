use base::Wire;
use base::tangent::Tangent;
use dataflow::diagram::Term;
use reactive::var::{X, X_ref, d, d_ref};
use reactive::{ModuleInterface, Var, Wire3};
use theory::lra::{LRA, Sort};

#[test]
fn variables_are_globally_ordered_by_creation() {
    let x = Wire3::new(Sort::real([1, 1]));
    let y = Wire3::new(Sort::real([1, 1]));
    let z = Wire3::new(Sort::real([1, 1]));

    // the latched wire's id increases with creation
    assert!(x < y && y < z);
    assert!(x <= x && x == x);

    // sorting recovers the creation order
    let mut vars = [z, x, y];
    vars.sort();
    assert_eq!(vars, [x, y, z]);
}

fn borrow_copy_and_compare(x: &Wire3<Sort>, nxt: Wire<Sort>, der: Wire<Sort>) {
    assert_eq!(d(x), nxt);
    assert_eq!(X(x), der);
}

#[test]
fn variable_casts_to_its_wires() {
    // variables are Copy: owned bindings, ampersand-free use everywhere
    let real = Sort::real([1, 1]);
    let x = Wire3::new(real);

    let a: Wire<_> = *x; // the latched wire
    let b: Wire<_> = d(x); //     its derivative
    let c: Wire<_> = X(x); //   the next wire

    // d/nxt copy, never clone and never consume: the variable stays
    // usable and the views are stable
    assert_eq!(d(x), b);
    assert_eq!(X(x), c);

    // three distinct wires; the latched and next carry values, the
    // derivative carries the tangent
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
    assert_eq!(*a.dtype(), real);
    assert_eq!(*c.dtype(), real);
    assert_eq!(*b.dtype(), real.tangent());
    assert_eq!(
        *b.dtype(),
        Sort::Real {
            shape: [1, 1],
            rank: 1
        }
    );

    // the variable dereferences to its latched wire: `Wire`'s methods
    // apply to it directly, and `&x` coerces to `&Wire<_>`
    assert_eq!(*x.dtype(), real);
    assert_eq!(x.id(), a.id());
    let borrowed: &Wire<_> = &x;
    assert_eq!(*borrowed, a);

    // X and d take any borrow of the variable: by reference as well
    borrow_copy_and_compare(&x, b, c);

    // the _ref alternatives borrow all the way through: reference in,
    // reference out, no clone
    let da: &Wire<_> = &x;
    let db: &Wire<_> = d_ref(&x);
    let cb: &Wire<_> = X_ref(&x);
    assert_eq!(da, &a);
    assert_eq!(db, &b);
    assert_eq!(cb, &c);
}

#[test]
fn variables_pass_into_term_constructors() {
    use crate::Term;
    use theory::lra::{LRA, Sort};

    let x = Wire3::new(Sort::real([1, 1]));

    // a variable stands for its latched wire in a term position
    let t = Term::constant(LRA::AnyReal(1, 1), [x]).unwrap();
    assert_eq!(t.write().iter().next().map(Wire::id), Some(x.id()));

    // and its views pass as well: x' = x
    let t = Term::function(LRA::IdReal(1, 1), [X(x)], [x]).unwrap();
    assert_eq!(t.write().iter().next().map(Wire::id), Some(X(x).id()));
    assert_eq!(t.read().iter().next().map(Wire::id), Some(x.id()));
}

#[test]
fn variable_fail_into_term_constructors() {
    let x = Wire3::new(Sort::real([1, 1]));
    // dx = x: the derivative has the tangent sort, not the sort of `x`
    assert!(Term::function(LRA::IdReal(1, 1), [d(x)], [x]).is_err());
    assert!(Term::function(LRA::TId(1, 1, 1), [d(x)], [x]).is_err());
}

#[test]
fn named_variables_are_equal_by_name() {
    let real = Sort::real([1, 1]);
    let x = Var::new(3, real);
    assert_eq!(x.name(), 3);
    assert_eq!(*x.dtype(), real);
    assert_ne!(x, Var::new(4, real));
    assert!(Var::new(2, real) < x && x < Var::new(4, real));

    // the sort does not take part in identity
    assert_eq!(x, Var::new(3, Sort::real([1, 2])));
}

fn interface(names: &[u32]) -> ModuleInterface<Sort> {
    let real = Sort::real([1, 1]);
    let vars = names.iter().map(|&n| Var::new(n, real));
    ModuleInterface::try_from_iter(vars).unwrap()
}

#[test]
fn module_interface_is_a_sorted_set() {
    let intf = interface(&[5, 1, 3, 1, 5]);

    // sorted by name, duplicates kept once
    assert_eq!(intf.len(), 3);
    let names: Vec<u32> = intf.iter().map(Var::name).collect();
    assert_eq!(names, [1, 3, 5]);
    assert_eq!(intf.nth(1).map(Var::name), Some(3));
    assert_eq!(intf.nth(3), None);

    assert!(intf.contains(&Var::new(5, Sort::real([1, 1]))));
    assert!(!intf.contains(&Var::new(2, Sort::real([1, 1]))));

    // the same names in any order make the same interface
    assert_eq!(intf, interface(&[3, 5, 1]));
}

#[test]
fn module_interface_rejects_conflicting_sorts() {
    let x = Var::new(1, Sort::real([1, 1]));
    let y = Var::new(2, Sort::real([1, 1]));
    let x_vec = Var::new(1, Sort::real([1, 2]));

    let res = ModuleInterface::try_from_iter([x, y, x_vec]);
    assert!(
        res.is_err(),
        "the same name with two sorts must be rejected"
    );

    // a repeated listing with the same sort is fine
    let res = ModuleInterface::try_from_iter([x, y, x]);
    assert_eq!(res.map(|i| i.len()), Ok(2));
}

#[test]
fn module_interface_disjointness() {
    let a = interface(&[1, 3]);
    let b = interface(&[2, 4]);
    let c = interface(&[3, 4]);
    let empty = ModuleInterface::empty();

    assert!(a.is_disjoint(&b) && b.is_disjoint(&a));
    assert!(!a.is_disjoint(&c) && !c.is_disjoint(&a));
    assert!(a.is_disjoint(&empty) && empty.is_disjoint(&a));
    assert!(empty.is_empty());
}
