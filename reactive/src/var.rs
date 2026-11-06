use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// The prefix that marks a variable name when displayed, as in `#3`.
pub const PREFIX: &str = "#";

/// A variable identified by its name.
///
/// A `Var` is a name and the sort of the values it holds. Two variables are
/// the same variable exactly when they have the same name, so a `Var`
/// mentioned by an atom and by a module refers to one variable. Equality,
/// hashing, and ordering all go through the name; the sort does not take part
/// in identity.
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
