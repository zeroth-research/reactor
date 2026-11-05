//! Tests for boundaries: polynomials of `⊗`-monomials under `⊕` and `⊗`.

use base::boundary::{Mono, Poly};

/// A monomial over single-character objects: `mono("xy")` is `x ⊗ y`.
fn mono(word: &str) -> Mono<char> {
    word.chars().collect()
}

/// A polynomial from its summands, in order: `poly(&["xy", "z"])` is
/// `xy ⊕ z`. The empty word `""` is the unit monomial `I`.
fn poly(words: &[&str]) -> Poly<char> {
    words.iter().map(|w| mono(w)).collect()
}

#[test]
fn boundaries() {
    // ⊗ of monomials is concatenation, with the unit I
    let xy = Mono::from_iter(['x']) * Mono::from_iter(['y']);
    assert_eq!(
        Poly::from(xy.clone()),
        Poly::from_iter([Mono::from_iter(['x', 'y'])])
    );
    assert_eq!(xy.clone() * Mono::unit(), xy);

    // operator syntax: x * y = xy
    assert_eq!(
        Mono::from_iter(['x']) * Mono::from_iter(['y']),
        Mono::from_iter(['x', 'y'])
    );
}

#[test]
fn concatenation_of_monomials() {
    // ⊗ of two monomials is a monomial: its word is the first word followed
    // by the second
    let (m, n) = (mono("xy"), mono("zwv"));
    let mn = m.clone() * n.clone();
    assert_eq!(mn, mono("xyzwv"));
    assert_eq!(
        mn.iter().collect::<Vec<_>>(),
        m.iter().chain(n.iter()).collect::<Vec<_>>()
    );
}

#[test]
fn addition_of_polynomials() {
    // ⊕ of two polynomials is a polynomial: its summands are those of the
    // first followed by those of the second, unit monomials included
    let (p, q) = (poly(&["xy", ""]), poly(&["z", "wv", "u"]));
    let pq = p.clone() + q.clone();
    assert_eq!(pq, poly(&["xy", "", "z", "wv", "u"]));
    assert_eq!(
        pq.iter().collect::<Vec<_>>(),
        p.iter().chain(q.iter()).collect::<Vec<_>>()
    );

    // likewise for monomials: two summands, in order
    assert_eq!(mono("xy") + mono("z"), poly(&["xy", "z"]));
}

#[test]
fn unit_on_both_sides() {
    // I is the unit of ⊗ on both sides
    let m = mono("xy");
    assert_eq!(Mono::unit() * m.clone(), m);
    assert_eq!(m.clone() * Mono::unit(), m);

    // the empty word is I, and I ⊗ I = I
    assert_eq!(mono(""), Mono::unit());
    assert_eq!(Mono::unit() * mono(""), Mono::unit());

    // the sum of the empty word alone is I
    assert_eq!(poly(&[""]), Poly::unit());
}

#[test]
fn zero_on_both_sides() {
    // the empty sum is 0, and 0 ⊕ 0 = 0
    assert_eq!(poly(&[]), Poly::zero());
    assert_eq!(Poly::zero() + poly(&[]), Poly::zero());

    let p = poly(&["xy", "z"]);

    // 0 is the unit of ⊕ on both sides
    assert_eq!(Poly::zero() + p.clone(), p);
    assert_eq!(p.clone() + Poly::zero(), p);
}

#[test]
fn associativity() {
    let (p, q, r) = (poly(&["a", "b"]), poly(&["c", "d"]), poly(&["e", "f"]));
    assert_eq!((p.clone() + q.clone()) + r.clone(), p + (q + r));

    let (x, y, z) = (mono("ab"), mono("c"), mono("de"));
    assert_eq!((x.clone() * y.clone()) * z.clone(), x * (y * z));
}

#[test]
fn equality_is_structural() {
    // sums are lists, not sets: Bool = I ⊕ I is not I
    assert_ne!(Poly::<char>::unit() + Poly::unit(), Poly::unit());

    // ⊕ does not commute: the order of the alternatives is tracked
    assert_ne!(poly(&["x"]) + poly(&["y"]), poly(&["y"]) + poly(&["x"]));

    // nor does ⊗: words are ordered
    assert_ne!(mono("x") * mono("y"), mono("y") * mono("x"));

    // the impossible boundary 0 is not the empty boundary I
    assert_ne!(Poly::<char>::zero(), Poly::unit());
}

#[test]
fn collecting_nothing() {
    // an empty sum collects into 0: no summands at all
    assert_eq!(
        Poly::<char>::from_iter(std::iter::empty::<Mono<char>>()),
        Poly::zero()
    );

    // the unit monomial, injected, is the unit polynomial
    assert_eq!(Poly::from(Mono::<char>::unit()), Poly::unit());
}

#[test]
fn iteration_preserves_order() {
    // a monomial iterates over its word in order, by reference or by value
    let m = mono("xyz");
    assert_eq!(m.iter().collect::<String>(), "xyz");
    let mut word = String::new();
    for g in &m {
        word.push(*g);
    }
    assert_eq!(word, "xyz");
    assert_eq!(m.clone().into_iter().collect::<Mono<char>>(), m);

    // a polynomial iterates over its summands in order, unit monomials
    // included
    let p = poly(&["xy", "", "z"]);
    let words: Vec<String> = p.iter().map(|m| m.iter().collect()).collect();
    assert_eq!(words, ["xy", "", "z"]);
    let mut words = Vec::new();
    for m in &p {
        words.push(m.iter().collect::<String>());
    }
    assert_eq!(words, ["xy", "", "z"]);
    assert_eq!(p.clone().into_iter().collect::<Poly<char>>(), p);

    // 0 has no summands, I has one
    assert_eq!(Poly::<char>::zero().iter().count(), 0);
    assert_eq!(Poly::<char>::unit().iter().count(), 1);
}

#[test]
fn products_fold_with_tensor() {
    // a product of base types is their tuple, in order
    assert_eq!("xyz".chars().product::<Mono<char>>(), mono("xyz"));

    // a product of monomials is their ⊗, in order
    let (x, y, z) = (mono("x"), mono("yw"), mono("z"));
    assert_eq!(
        [x.clone(), y.clone(), z.clone()]
            .into_iter()
            .product::<Mono<char>>(),
        x.clone() * y.clone() * z.clone()
    );
    assert_ne!(
        [y.clone(), x.clone()].into_iter().product::<Mono<char>>(),
        [x, y].into_iter().product::<Mono<char>>()
    );

    // the empty product is I, from base types and from monomials alike
    assert_eq!(
        std::iter::empty::<char>().product::<Mono<char>>(),
        Mono::unit()
    );
    assert_eq!(
        std::iter::empty::<Mono<char>>().product::<Mono<char>>(),
        Mono::unit()
    );
}

#[test]
fn sums_fold_with_plus() {
    // a sum of base types has one single-component summand for each, in
    // order
    assert_eq!("xyz".chars().sum::<Poly<char>>(), poly(&["x", "y", "z"]));

    // a sum of monomials is their ⊕, in order
    let (x, y, z) = (mono("x"), mono("yw"), mono(""));
    assert_eq!(
        [x.clone(), y.clone(), z.clone()]
            .into_iter()
            .sum::<Poly<char>>(),
        x.clone() + y.clone() + Poly::from(z)
    );
    assert_ne!(
        [y.clone(), x.clone()].into_iter().sum::<Poly<char>>(),
        [x, y].into_iter().sum::<Poly<char>>()
    );

    // a sum of polynomials concatenates their summands, in order
    let (p, q) = (poly(&["xy", ""]), poly(&["z"]));
    assert_eq!(
        [p.clone(), Poly::zero(), q.clone()]
            .into_iter()
            .sum::<Poly<char>>(),
        p + q
    );

    // the empty sum is 0, from base types, monomials and polynomials alike
    assert_eq!(std::iter::empty::<char>().sum::<Poly<char>>(), Poly::zero());
    assert_eq!(
        std::iter::empty::<Mono<char>>().sum::<Poly<char>>(),
        Poly::zero()
    );
    assert_eq!(
        std::iter::empty::<Poly<char>>().sum::<Poly<char>>(),
        Poly::zero()
    );
}

#[test]
fn display() {
    assert_eq!(Poly::<char>::zero().to_string(), "0");
    assert_eq!(Poly::<char>::unit().to_string(), "I");
    assert_eq!(Mono::<char>::unit().to_string(), "I");
    assert_eq!(Poly::from(mono("xy")).to_string(), "x ⊗ y");
    assert_eq!(poly(&["xz", "yz"]).to_string(), "(x ⊗ z) ⊕ (y ⊗ z)");
    // the unit monomial is printed also as a summand
    assert_eq!(poly(&["", "x"]).to_string(), "I ⊕ x");
}

#[test]
fn display_of_products() {
    // a single base type is printed alone, without ⊗
    assert_eq!(mono("x").to_string(), "x");

    // a monomial prints its whole word, in order, repeats included, and
    // without parentheses
    assert_eq!(mono("xy").to_string(), "x ⊗ y");
    assert_eq!(mono("xyz").to_string(), "x ⊗ y ⊗ z");
    assert_eq!(mono("zyx").to_string(), "z ⊗ y ⊗ x");
    assert_eq!(mono("xxyx").to_string(), "x ⊗ x ⊗ y ⊗ x");

    // a monomial and its injection into polynomials print the same: the only
    // summand is not put in parentheses
    let m = mono("uvw");
    assert_eq!(Poly::from(m.clone()).to_string(), m.to_string());
    assert_eq!(poly(&["uvw"]).to_string(), "u ⊗ v ⊗ w");
    assert_eq!(poly(&["u"]).to_string(), "u");

    // ⊗ prints as joining the words, with no parentheses
    assert_eq!((mono("xy") * mono("zw")).to_string(), "x ⊗ y ⊗ z ⊗ w");
}

#[test]
fn display_of_sums_of_products() {
    // among two or more summands, every summand of two or more base types
    // prints its whole word, in order, in parentheses, and the summands are
    // joined by ⊕ in order
    let p = poly(&["xyz", "uv", "wvut"]);
    assert_eq!(p.to_string(), "(x ⊗ y ⊗ z) ⊕ (u ⊗ v) ⊕ (w ⊗ v ⊗ u ⊗ t)");
    assert_eq!(
        p.to_string(),
        p.iter()
            .map(|m| format!("({m})"))
            .collect::<Vec<_>>()
            .join(" ⊕ ")
    );

    // summands of a single base type, or of none, are not in parentheses
    assert_eq!(poly(&["x", "yz", "w"]).to_string(), "x ⊕ (y ⊗ z) ⊕ w");
    assert_eq!(poly(&["x", "y", "z"]).to_string(), "x ⊕ y ⊕ z");
    assert_eq!(poly(&["xy", ""]).to_string(), "(x ⊗ y) ⊕ I");

    // summands are neither reordered nor deduplicated, and neither are the
    // base types within them
    assert_eq!(
        poly(&["yx", "xy", "yx"]).to_string(),
        "(y ⊗ x) ⊕ (x ⊗ y) ⊕ (y ⊗ x)"
    );
    assert_eq!(poly(&["xx", "xx"]).to_string(), "(x ⊗ x) ⊕ (x ⊗ x)");

    // unit summands print as I among the products, at either end and in
    // between
    assert_eq!(
        poly(&["", "xy", "", "zwv", ""]).to_string(),
        "I ⊕ (x ⊗ y) ⊕ I ⊕ (z ⊗ w ⊗ v) ⊕ I"
    );

    // ⊕ of polynomials of two or more summands each prints as joining their
    // prints with ⊕
    let (p, q) = (poly(&["xy", "z"]), poly(&["uvw", "st"]));
    assert_eq!((p.clone() + q.clone()).to_string(), format!("{p} ⊕ {q}"));

    // but a product alone in its polynomial gains parentheses once it is
    // summed with another
    let (r, s) = (poly(&["xy"]), poly(&["zw"]));
    assert_eq!(r.to_string(), "x ⊗ y");
    assert_eq!(s.to_string(), "z ⊗ w");
    assert_eq!((r + s).to_string(), "(x ⊗ y) ⊕ (z ⊗ w)");
    assert_eq!((mono("xy") + mono("zw")).to_string(), "(x ⊗ y) ⊕ (z ⊗ w)");

    // 0 summands vanish from a sum of polynomials, and from its print
    assert_eq!(
        [p.clone(), Poly::zero(), q.clone()]
            .into_iter()
            .sum::<Poly<char>>()
            .to_string(),
        "(x ⊗ y) ⊕ z ⊕ (u ⊗ v ⊗ w) ⊕ (s ⊗ t)"
    );

    // and a product left alone with 0 loses its parentheses
    assert_eq!((poly(&["xy"]) + Poly::zero()).to_string(), "x ⊗ y");
}

#[test]
fn display_uses_display_of_base_types() {
    // base types print with their own Display, here multi-letter names
    let pair: Mono<&str> = Mono::from_iter(["Real", "Real"]);
    assert_eq!(pair.to_string(), "Real ⊗ Real");
    assert_eq!(Poly::from(pair.clone()).to_string(), "Real ⊗ Real");

    let p: Poly<&str> = [
        pair,
        Mono::from_iter(["Int", "Bool", "Real"]),
        Mono::unit(),
        Mono::from_iter(["String"]),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        p.to_string(),
        "(Real ⊗ Real) ⊕ (Int ⊗ Bool ⊗ Real) ⊕ I ⊕ String"
    );
}
