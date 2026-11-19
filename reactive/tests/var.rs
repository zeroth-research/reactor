mod common;

use common::{REAL, Sort, mk_op};
use dataflow::diagram::Term;
use dataflow::diagram::Wire;
use reactive::Var;
use reactive::var::{X, X_ref, d, d_ref};

#[test]
fn variables_are_globally_ordered_by_creation() {
    let x = Var::new(Sort("t"));
    let y = Var::new(Sort("t"));
    let z = Var::new(Sort("t"));

    // the latched wire's id increases with creation
    assert!(x < y && y < z);
    assert!(x <= x && x == x);

    // sorting recovers the creation order
    let mut vars = [z, x, y];
    vars.sort();
    assert_eq!(vars, [x, y, z]);
}

fn borrow_copy_and_compare(x: &Var<Sort>, nxt: Wire<Sort>, der: Wire<Sort>) {
    assert_eq!(d(x), nxt);
    assert_eq!(X(x), der);
}

#[test]
fn variable_casts_to_its_wires() {
    // variables are Copy: owned bindings, ampersand-free use everywhere
    let x = Var::new(Sort("type"));

    let a: Wire<_> = *x; // the latched wire
    let b: Wire<_> = d(x); //     its derivative
    let c: Wire<_> = X(x); //   the next wire

    // d/nxt copy, never clone and never consume: the variable stays
    // usable and the views are stable
    assert_eq!(d(x), b);
    assert_eq!(X(x), c);

    // three distinct wires
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);

    // the variable dereferences to its latched wire: `Wire`'s methods
    // apply to it directly, and `&x` coerces to `&Wire<_>`
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

    let x = Var::new(REAL);

    // a variable stands for its latched wire in a term position
    let t = Term::constant(mk_op("HAVOC", [REAL], []), [x]).unwrap();
    assert_eq!(t.write().iter().next().map(Wire::id), Some(x.id()));

    // and its views pass as well: x' = x
    let t = Term::function(mk_op("ID", [REAL], [REAL]), [X(x)], [x]).unwrap();
    assert_eq!(t.write().iter().next().map(Wire::id), Some(X(x).id()));
    assert_eq!(t.read().iter().next().map(Wire::id), Some(x.id()));
}

/// The latched and next wires carry values of the variable's sort, and the
/// derivative wire their rates of change, of its tangent sort: one degree
/// higher for reals, and the singleton `Zero` for discrete sorts.
#[test]
fn derivative_wire_has_the_tangent_sort() {
    use theory::tensor::sort::{Any, Bool, Int, Real, Word, Zero};

    let r: Any = Real::new([2, 3]).into();
    let x = Var::new(r);
    assert_eq!((*x.dtype(), *X(x).dtype()), (r, r));
    let Any::Real(dx) = *d(x).dtype() else {
        panic!("the rate of change of a real is a real");
    };
    assert_eq!((dx.shape(), dx.degree()), ([2, 3], 1));

    let discrete: [Any; 3] = [
        Bool::new([2, 3]).into(),
        Int::new([2, 3]).into(),
        Word::new_unchecked(8, [2, 3]).into(),
    ];
    for s in discrete {
        let x = Var::new(s);
        assert_eq!((*x.dtype(), *X(x).dtype()), (s, s));
        assert_eq!(*d(x).dtype(), Any::Zero(Zero));
    }
}
