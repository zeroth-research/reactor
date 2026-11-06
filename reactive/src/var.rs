use base::Wire;
use base::tangent::Tangent;
use base::wire::PREFIX;
use std::borrow::{Borrow, Cow};
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

#[derive(Clone, Copy)]
pub struct Wire3<S> {
    ltc: Wire<S>,
    nxt: Wire<S>,
    der: Wire<S>,
}

impl<S: Tangent + Copy> Wire3<S> {
    /// A fresh variable of the sort `dtype`: its latched and next wires carry
    /// values of `dtype`, its derivative wire carries the tangent `dtype.tangent()`.
    pub fn new(dtype: S) -> Self {
        Self {
            ltc: Wire::new(dtype),
            nxt: Wire::new(dtype),
            der: Wire::new(dtype.tangent()),
        }
    }
}

impl<S> Wire3<S> {
    pub(crate) fn ltc(&self) -> &Wire<S> {
        &self.ltc
    }

    pub(crate) fn nxt(&self) -> &Wire<S> {
        &self.nxt
    }

    pub(crate) fn der(&self) -> &Wire<S> {
        &self.der
    }
}

/// Equality goes through the latched wire — the variable's bearing element,
/// consistent with its [`Deref`] view: two variables are the same variable
/// iff they share the same latched wire.
impl<S> PartialEq for Wire3<S> {
    fn eq(&self, other: &Self) -> bool {
        self.ltc == other.ltc
    }
}

impl<S> Eq for Wire3<S> {}

/// Hashing matches equality: only the latched wire is hashed.
impl<S> Hash for Wire3<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ltc.hash(state);
    }
}

/// Variables are totally ordered by their latched wire — the
/// bearing element of equality — which globally orders them by creation.
impl<S> Ord for Wire3<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ltc.cmp(&other.ltc)
    }
}

impl<S> PartialOrd for Wire3<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A variable casts into its latched wire: `let a: Wire<_> = *x;`, and
/// implicitly at any call boundary bounded on `Into<Wire<S>>` (e.g. the term
/// constructors).
impl<S> From<Wire3<S>> for Wire<S> {
    fn from(var: Wire3<S>) -> Self {
        var.ltc
    }
}

/// A variable borrows as its latched wire, consistently with `Eq`, `Ord`,
/// and `Hash`, which all delegate to it: a `Wire3` stands for its latched wire
/// wherever a `Borrow<Wire<S>>` is accepted (e.g. keyed lookups, printing).
impl<S> Borrow<Wire<S>> for Wire3<S> {
    fn borrow(&self) -> &Wire<S> {
        self
    }
}

/// A variable dereferences to its latched wire: `&x` coerces to `&Wire<S>`
/// wherever one is expected, and `Wire`'s methods (`x.id()`, `x.dtype()`)
/// apply to the variable directly.
///
/// `Deref` for a non-pointer type is deliberate here: the latched wire is the
/// variable's primary view, the way `String` derefs to `str`.
impl<S> Deref for Wire3<S> {
    type Target = Wire<S>;

    fn deref(&self) -> &Self::Target {
        &self.ltc
    }
}

/// The variable's derived (derivative) wire: `let w: Wire<_> = d(x);`.
///
/// Bounded on `Borrow`, so it accepts the variable by value (`d(x)`, natural
/// for `Copy` sorts) or by reference (`d(&x)`); either way only the wire is
/// cloned and the variable stays usable.
pub fn d<S: Clone>(var: impl Borrow<Wire3<S>>) -> Wire<S> {
    var.borrow().der.clone()
}

/// The borrowing alternative to [`d`]: a reference to the variable's derived
/// (derivative) wire, `let w: &Wire<_> = d_ref(&x);`.
///
/// Reference in, reference out — no clone
pub fn d_ref<S>(var: &Wire3<S>) -> &Wire<S> {
    &var.der
}

/// The variable's next wire: `let w: Wire<_> = X(x);`.
///
/// Bounded on `Borrow`, so it accepts the variable by value (`X(x)`, natural
/// for `Copy` sorts) or by reference (`X(&x)`); either way only the wire is
/// cloned and the variable stays usable.
#[allow(non_snake_case)]
pub fn X<S: Clone>(var: impl Borrow<Wire3<S>>) -> Wire<S> {
    var.borrow().nxt.clone()
}

/// The borrowing alternative to [`X`]: a reference to the variable's next
/// wire, `let w: &Wire<_> = X_ref(&x);`.
///
/// Reference in, reference out — no clone
#[allow(non_snake_case)]
pub fn X_ref<S>(var: &Wire3<S>) -> &Wire<S> {
    &var.nxt
}

pub struct Display<'a, S> {
    var: &'a Wire3<S>,
    name: Cow<'a, str>,
}

impl<S: fmt::Display> fmt::Display for Display<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} : {}", self.name, self.var.dtype())
    }
}

impl<S: Debug> Debug for Display<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} : {:?}", self.name, self.var.dtype())
    }
}

impl<'a, S> Wire3<S> {
    pub fn with_name(&'a self, name: Cow<'a, str>) -> Display<'a, S> {
        Display { var: self, name }
    }
}

impl<S: Debug> Debug for Wire3<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Wire3 {{ ltc: {PREFIX}{:?}, nxt: {PREFIX}{:?}, der: {PREFIX}{:?}, dtype: {:?} }} ",
            self.ltc.id(),
            self.nxt.id(),
            self.der.id(),
            self.dtype()
        )
    }
}

/// A variable identified by its name.
///
/// Unlike [`Wire3`], a `Var` carries no wires: it is a name and the sort of
/// the values it holds. Two variables are the same variable exactly when they
/// have the same name, so a `Var` mentioned by an atom and by a module refers
/// to one variable. Equality, hashing, and ordering all go through the name;
/// as for a [`Wire`], the sort does not take part in identity.
#[derive(Debug, Clone, Copy)]
pub struct Var<S> {
    name: u32,
    dtype: S,
}

impl<S> Var<S> {
    /// The variable named `name`, holding values of the sort `dtype`.
    pub const fn new(name: u32, dtype: S) -> Self {
        Self { name, dtype }
    }

    /// The variable's name; variables are equal, hash alike, and order
    /// exactly by it.
    pub const fn name(&self) -> u32 {
        self.name
    }

    /// The sort of the values the variable holds.
    pub const fn dtype(&self) -> &S {
        &self.dtype
    }
}

impl<S> PartialEq for Var<S> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<S> Eq for Var<S> {}

impl<S> Hash for Var<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl<S> Ord for Var<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name.cmp(&other.name)
    }
}

impl<S> PartialOrd for Var<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The interface of a module: a set of named variables.
///
/// The variables are stored sorted by name and without duplicates, so
/// membership and disjointness checks use binary search.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModuleInterface<S> {
    vars: Vec<Var<S>>,
}

impl<S> ModuleInterface<S> {
    pub fn empty() -> Self {
        Self { vars: Vec::new() }
    }

    /// Returns true if no variable of `self` is a variable of `other`.
    /// Runs in place, in O(self.len() * log(other.len())) time.
    pub fn is_disjoint(&self, other: &ModuleInterface<S>) -> bool {
        self.vars
            .iter()
            .all(|a| other.vars.binary_search(a).is_err())
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// The variables in ascending order of their names.
    pub fn iter(&self) -> impl Iterator<Item = &Var<S>> {
        self.vars.iter()
    }

    /// The `index`-th variable in ascending order of names.
    pub fn nth(&self, index: usize) -> Option<&Var<S>> {
        self.vars.get(index)
    }

    /// Whether a variable of the same name is in the interface; the sort is
    /// not compared.
    pub fn contains(&self, var: &Var<S>) -> bool {
        self.vars.binary_search(var).is_ok()
    }
}

impl<S: PartialEq + Debug> ModuleInterface<S> {
    /// Collects the variables into a set, sorted by name.
    ///
    /// A variable listed more than once is kept once, provided every listing
    /// agrees on its sort.
    ///
    /// # Errors
    /// If two listings of the same name disagree on the sort.
    pub fn try_from_iter<I>(iter: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = Var<S>>,
    {
        let mut vars: Vec<Var<S>> = iter.into_iter().collect();
        vars.sort_by_key(Var::name);
        for pair in vars.windows(2) {
            if pair[0] == pair[1] && pair[0].dtype != pair[1].dtype {
                return Err(format!(
                    "Variable {} has conflicting sorts {:?} and {:?}",
                    pair[0].name, pair[0].dtype, pair[1].dtype
                ));
            }
        }
        vars.dedup();
        Ok(Self { vars })
    }
}

impl<S: Debug> ModuleInterface<S> {
    /// The interface of the variables `vars`, which must be sorted by name
    /// without duplicates; not checked beyond debug assertions.
    pub(crate) fn from_sorted_unchecked(vars: Vec<Var<S>>) -> Self {
        debug_assert!(
            vars.is_sorted_by(|a, b| a < b),
            "duplicate or unsorted {:?}",
            vars
        );
        Self { vars }
    }
}

/// The interface of an atom: the set of the named variables it controls,
/// awaits, or reads. It is the same set as that of a module.
pub type AtomInterface<S> = ModuleInterface<S>;

impl<S> Default for ModuleInterface<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S> IntoIterator for ModuleInterface<S> {
    type Item = Var<S>;
    type IntoIter = std::vec::IntoIter<Var<S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vars.into_iter()
    }
}

impl<'a, S> IntoIterator for &'a ModuleInterface<S> {
    type Item = &'a Var<S>;
    type IntoIter = std::slice::Iter<'a, Var<S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vars.iter()
    }
}
