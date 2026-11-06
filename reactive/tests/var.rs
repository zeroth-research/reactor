use reactive::{ModuleInterface, Var};
use theory::lra::Sort;

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
