//! `reactor.verilog`: SystemVerilog designs as modules, translated by the
//! `verilog` crate (synthesizable subset, 2-state, one round per time step).
//!
//! [`translate`] parses, elaborates and translates a design, rewrites it into
//! the ops of `theory::bv` and returns the module with the names and roles of
//! its variables. Words are evaluated by `reactor.eval` as `int64` tensors
//! holding the unsigned value, which is right only below 64 bits, so a design
//! with a wider word is refused.

use crate::module::Module;
use crate::var::Var;
use ::verilog::ast::{Error, ErrorKind, Override};
use ::verilog::lower::Translation;
use ::verilog::rm_sim::{Sim, Valuation};
use ::verilog::{BitVec, any, constant, elab, frontend, lower, qfbv};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::path::PathBuf;
use theory::bv::BV;

create_exception!(
    verilog,
    VerilogError,
    PyException,
    "A design cannot be translated."
);
create_exception!(
    verilog,
    UserError,
    VerilogError,
    "The design breaks a rule of SystemVerilog or of the translation."
);
create_exception!(
    verilog,
    UnsupportedError,
    VerilogError,
    "The design uses a construct the translation does not support (yet), or a word of 64 bits or more."
);
create_exception!(
    verilog,
    InternalError,
    VerilogError,
    "A bug of the translation."
);
create_exception!(
    verilog,
    VerilogWarning,
    PyUserWarning,
    "Something about the design the translation assumes or leaves out."
);

/// Words of this many bits or more are refused: `reactor.eval` stores a word
/// as the unsigned value in an `int64`, which holds it only below 64 bits.
const LIMIT: usize = 64;

/// The stack of the thread translating: the parser recurses once per level of
/// nesting, and deep expressions need more than a default stack. Only
/// reserved, not used, unless a design needs it.
const STACK: usize = 1 << 30;

pub(crate) fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "verilog")?;
    m.add_function(wrap_pyfunction!(translate, &m)?)?;
    m.add_function(wrap_pyfunction!(simulate, &m)?)?;
    m.add("VerilogError", py.get_type::<VerilogError>())?;
    m.add("UserError", py.get_type::<UserError>())?;
    m.add("UnsupportedError", py.get_type::<UnsupportedError>())?;
    m.add("InternalError", py.get_type::<InternalError>())?;
    m.add("VerilogWarning", py.get_type::<VerilogWarning>())?;
    Ok(m)
}

/// A translated design.
struct Loaded {
    t: Translation,
    d: any::Desugared,
    /// The variable of the desugared module of each named variable.
    names: BTreeMap<String, any::Var>,
    /// Warnings, located, in the order v2rm prints them.
    warnings: Vec<String>,
}

/// Parses, elaborates and translates the design in `paths` with top module
/// `top` (the one no module instantiates if none) and the parameter values
/// `params` (as `-G NAME=VALUE`), then bridges and desugars it. Errors are
/// located in their files.
fn load(
    paths: &[PathBuf],
    top: Option<String>,
    params: &[(String, String)],
) -> Result<Loaded, Error> {
    let first = paths.first().cloned().unwrap_or_default();
    let mut design = None;
    let result = (|| {
        let d = design.insert(frontend::parse_design_files(paths)?);
        let top = match top {
            Some(t) => t,
            None => d.top()?,
        };
        let mut overrides = vec![];
        for (name, value) in params {
            let value = constant::parse(value).map_err(|e| {
                Error::unlocated(ErrorKind::User, format!("parameter {name} = {value}: {e}"))
            })?;
            overrides.push(Override {
                name: name.clone(),
                value,
                span: None,
            });
        }
        let inst = elab::elaborate(d, &top, &overrides)?;
        lower::lower_inst(&inst)
    })();
    let located = |e: Error| {
        let e = match &design {
            Some(d) => d.located(e),
            None => e,
        };
        if e.file.is_some() {
            e
        } else {
            e.in_file(&first)
        }
    };
    let t = result.map_err(located)?;
    let mut warnings = vec![];
    for name in &t.unread_inputs {
        warnings.push(format!(
            "{}: input `{name}` is never read and is left out",
            first.display()
        ));
    }
    let unwritten = t.unwritten.iter().map(|u| u.warning());
    for w in t
        .blackboxes
        .iter()
        .chain(&t.out_of_range)
        .cloned()
        .chain(unwritten)
    {
        let w = match &design {
            Some(d) => d.located_warning(w),
            None => w,
        };
        let file = w.file.unwrap_or_else(|| first.clone());
        warnings.push(format!("{}:{}: warning: {}", file.display(), w.span, w.msg));
    }
    let b = qfbv::bridge(&t).map_err(located)?;
    let d = any::desugar(&b).map_err(located)?;
    let mut names = BTreeMap::new();
    for (name, var) in &t.vars {
        if let Some(v) = b.vars.get(var).and_then(|v| d.vars.get(v)) {
            names.insert(name.clone(), *v);
        }
    }
    narrow(&d)?;
    Ok(Loaded {
        t,
        d,
        names,
        warnings,
    })
}

/// Fails, as not supported, if a wire of `d` has [`LIMIT`] bits or more,
/// naming the variables and local signals that do.
fn narrow(d: &any::Desugared) -> Result<(), Error> {
    let m = &d.module;
    let mut wide = BTreeSet::new();
    for v in m
        .extl()
        .iter()
        .chain(m.intf().iter())
        .chain(m.prvt().iter())
    {
        if v.dtype().width() >= LIMIT {
            wide.insert(format!("`{}` ({} bits)", d.name(v), v.dtype().width()));
        }
    }
    let mut locals = 0;
    for a in m.atoms() {
        for t in a.init().iter().chain(a.update().iter()) {
            for w in t.write() {
                if w.dtype().width() >= LIMIT {
                    match d.wire_names.get(w) {
                        Some(n) => _ = wide.insert(format!("`{n}` ({} bits)", w.dtype().width())),
                        None => locals += 1,
                    }
                }
            }
        }
    }
    if wide.is_empty() && locals == 0 {
        return Ok(());
    }
    let mut msg = format!(
        "words of {LIMIT} bits or more are not supported yet (reactor.eval stores a word in an int64)"
    );
    if !wide.is_empty() {
        msg += &format!(": {}", wide.into_iter().collect::<Vec<_>>().join(", "));
    }
    if locals > 0 {
        msg += &format!(" ({locals} intermediate values)");
    }
    Err(Error::unlocated(ErrorKind::Unsupported, msg))
}

/// The files and the parameter values (name, value as text) of a design.
type Arguments = (Vec<PathBuf>, Vec<(String, String)>);

/// The arguments of [`translate`] in Rust form.
fn arguments(files: &Bound<'_, PyAny>, params: Option<&Bound<'_, PyDict>>) -> PyResult<Arguments> {
    let paths = match files.extract::<PathBuf>() {
        Ok(p) => vec![p],
        Err(_) => files.extract::<Vec<PathBuf>>()?,
    };
    let mut ps = vec![];
    if let Some(params) = params {
        for (k, v) in params {
            ps.push((k.extract::<String>()?, v.str()?.to_string()));
        }
    }
    Ok((paths, ps))
}

/// Runs [`load`] on a thread with a big stack, without the GIL.
fn load_big(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    top: Option<String>,
    params: Vec<(String, String)>,
) -> PyResult<Loaded> {
    let result = py.allow_threads(move || {
        let thread = std::thread::Builder::new()
            .name("verilog".to_string())
            .stack_size(STACK);
        match thread.spawn(move || load(&paths, top, &params)) {
            Ok(t) => t.join().map_err(|_| "the translation panicked".to_string()),
            Err(e) => Err(format!("cannot start the translation thread: {e}")),
        }
    });
    match result {
        Ok(Ok(l)) => Ok(l),
        Ok(Err(e)) => Err(exception(e)),
        Err(msg) => Err(InternalError::new_err(msg)),
    }
}

/// The exception of the kind of `e`, with the located message.
fn exception(e: Error) -> PyErr {
    let msg = e.to_string();
    match e.kind {
        ErrorKind::User => UserError::new_err(msg),
        ErrorKind::Unsupported => UnsupportedError::new_err(msg),
        ErrorKind::Internal => InternalError::new_err(msg),
    }
}

/// The constant op of the word `value` of `width` bits: a 1×1 `int64`
/// tensor (the value fits, since words are narrower than [`LIMIT`]).
fn constant(value: u64, width: usize) -> BV {
    debug_assert!(width < LIMIT);
    let tensor = tch::Tensor::from_slice(&[value as i64]).view([1, 1]);
    BV::Const(theory::PyTensor { tensor })
}

/// Translates the design in `files` (a path or a list of paths) with top
/// module `top` (by default the one no module instantiates) and parameter
/// values `params` (name to int or text, as v2rm's `-G NAME=VALUE`).
///
/// Returns `(module, names, roles)`: the module, every variable by name
/// (generated names contain `@`: `q@init` is the power-on value of register
/// `q`), and the role of each name: `"input"`, `"clock"` (an input clock),
/// `"init"` (`q@init`), `"blackbox"` (a blackbox output, free in every round),
/// `"output"` or `"internal"`. Inputs, clocks, inits and blackbox outputs are
/// the external variables: the environment gives their next value in every
/// round. Warnings are issued as `VerilogWarning`; errors raise a subclass of
/// `VerilogError`.
#[pyfunction]
#[pyo3(signature = (files, top = None, params = None))]
fn translate<'py>(
    py: Python<'py>,
    files: &Bound<'py, PyAny>,
    top: Option<String>,
    params: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Module, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let (paths, params) = arguments(files, params)?;
    let l = load_big(py, paths, top, params)?;
    let e = any::export(&l.d, constant).map_err(exception)?;
    let names = PyDict::new(py);
    let roles = PyDict::new(py);
    let m = &l.d.module;
    for (name, v) in &l.names {
        names.set_item(name, Var::from(e.vars[v]))?;
        let role = if m.extl().contains(v) {
            if l.t.clocks.contains(name) {
                "clock"
            } else if l.t.blackbox_outputs.contains(name) {
                "blackbox"
            } else if name.ends_with("@init") {
                "init"
            } else {
                "input"
            }
        } else if m.intf().contains(v) {
            "output"
        } else {
            "internal"
        };
        roles.set_item(name, role)?;
    }
    let category = py.get_type::<VerilogWarning>();
    for w in &l.warnings {
        let msg = CString::new(w.as_str()).unwrap_or_default();
        PyErr::warn(py, &category, &msg, 1)?;
    }
    Ok((Module::from(e.module), names, roles))
}

/// For tests: runs the translation of the design (as [`translate`]) in the
/// crate's own simulator, before the rewriting into `theory::bv`. `rounds`
/// gives, per round, the next value of every external variable by name;
/// returns, per round, the next value of every variable by name.
#[pyfunction]
#[pyo3(name = "_simulate", signature = (files, rounds, top = None, params = None))]
fn simulate<'py>(
    py: Python<'py>,
    files: &Bound<'py, PyAny>,
    rounds: Vec<BTreeMap<String, u64>>,
    top: Option<String>,
    params: Option<&Bound<'py, PyDict>>,
) -> PyResult<Vec<BTreeMap<String, u64>>> {
    let (paths, params) = arguments(files, params)?;
    let l = load_big(py, paths, top, params)?;
    let t = &l.t;
    let mut sim = Sim::new(&t.module);
    let mut out = vec![];
    for round in rounds {
        let mut extl = Valuation::new();
        for (name, value) in round {
            let var = t
                .vars
                .get(&name)
                .ok_or_else(|| PyException::new_err(format!("`{name}` is not a variable")))?;
            let value = BitVec::new(var.dtype().width(), value)
                .ok_or_else(|| PyException::new_err(format!("{value} does not fit `{name}`")))?;
            extl.insert(*var, value);
        }
        let next = sim.step(&extl).map_err(PyException::new_err)?;
        let mut values = BTreeMap::new();
        for (var, value) in &next {
            if let Some(name) = t.vars.name(var) {
                let digits = value.unsigned().to_u64_digits();
                if digits.len() > 1 {
                    return Err(PyException::new_err(format!(
                        "`{name}` is wider than 64 bits"
                    )));
                }
                values.insert(name.to_string(), digits.first().copied().unwrap_or(0));
            }
        }
        out.push(values);
    }
    Ok(out)
}
