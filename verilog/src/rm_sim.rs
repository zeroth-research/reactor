/*!
# Running a module round by round

[`Sim`] executes a [`Module`] over [`SvOp`](crate::SvOp): the first round
runs the init block of every atom, every later round the update block. The
atoms run in the module's order, which is consistent with the await
relation, and the terms of a block in their order. Delay blocks are not
run: the modules here are discrete.

In each round the environment provides the next value of every external
variable. The latched value of every variable is its next value from the
previous round.
*/

use crate::value::{BitVec, eval};
use crate::{Module, Var, Wire};
use base::var::X;
use std::collections::{BTreeMap, HashMap};

/// A value for each of some variables.
pub type Valuation = BTreeMap<Var, BitVec>;

pub struct Sim<'m> {
    module: &'m Module,
    /// The next values of all variables in the previous round, or `None`
    /// before the init round.
    last: Option<Valuation>,
}

impl<'m> Sim<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self { module, last: None }
    }

    /// Runs the next round, with `extl` giving the next value of every
    /// external variable (and of nothing else). Returns the next values of
    /// all variables of the module.
    pub fn step(&mut self, extl: &Valuation) -> Result<Valuation, String> {
        self.run(extl, &Valuation::new())
    }

    /// Runs the init round like [`step`](Self::step), but with each
    /// controlled variable in `forced` set to the value given there instead
    /// of the one its atom computes, before the atoms after it read it: a
    /// first round in a state the init blocks need not reach, e.g. with
    /// registers away from their initial values.
    pub fn step_forced(
        &mut self,
        extl: &Valuation,
        forced: &Valuation,
    ) -> Result<Valuation, String> {
        if self.last.is_some() {
            return Err("only the init round can be forced".into());
        }
        if let Some(var) = forced.keys().find(|v| !self.module.ctrl().contains(*v)) {
            return Err(format!("{var:?} is not a controlled variable"));
        }
        self.run(extl, forced)
    }

    fn run(&mut self, extl: &Valuation, forced: &Valuation) -> Result<Valuation, String> {
        let module = self.module;
        let mut env: HashMap<Wire, BitVec> = HashMap::new();

        if let Some(last) = &self.last {
            for (var, value) in last {
                let ltc: &Wire = var;
                env.insert(*ltc, value.clone());
            }
        }

        for (var, value) in extl {
            if !module.extl().contains(var) {
                return Err(format!("{var:?} is not an external variable"));
            }
            if value.sort() != *var.dtype() {
                return Err(format!("{var:?} given a value of sort {}", value.sort()));
            }
            env.insert(X(var), value.clone());
        }
        if let Some(var) = module.extl().iter().find(|v| !extl.contains_key(*v)) {
            return Err(format!("no value for external variable {var:?}"));
        }

        for atom in module.atoms() {
            let block = if self.last.is_none() {
                atom.init().iter()
            } else {
                atom.update().iter()
            };
            for term in block {
                let args = term
                    .read()
                    .iter()
                    .map(|w| {
                        env.get(w)
                            .cloned()
                            .ok_or(format!("{w:?} read before written"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let value = eval(term.itype(), &args)?;
                env.insert(term.write()[0], value);
            }
            for var in atom.ctrl() {
                if let Some(value) = forced.get(var) {
                    env.insert(X(var), value.clone());
                }
            }
        }

        let vars = module.extl().iter().chain(module.ctrl().iter());
        let next = vars
            .map(|var| match env.get(&X(var)) {
                Some(value) => Ok((*var, value.clone())),
                None => Err(format!("no next value for {var:?}")),
            })
            .collect::<Result<Valuation, String>>()?;
        self.last = Some(next.clone());
        Ok(next)
    }
}
