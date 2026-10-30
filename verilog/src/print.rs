/*!
# Printing modules with names

Like `base`'s display, but plain text (for golden tests) and with readable
names for the wires local to an atom: `%1`, `%2`, … in the order they are
written. Delay blocks are left out: in the discrete modules here they only
set every derivative to `ZERO`. When the init and update blocks of an atom
are the same (a combinatorial atom), they are printed once as
`init = update`.
*/

use crate::{Atom, Module, Sort, Var, Wire};
use base::Interface;
use base::var::{X, d};
use std::collections::HashMap;
use std::fmt::Write;

/// The module `title`, with `name` giving the names of its variables.
pub fn module(m: &Module, title: &str, name: impl Fn(&Var) -> String) -> String {
    let mut out = format!("module {title}\n");
    let mut vars: HashMap<Wire, String> = HashMap::new();
    for (section, intf) in [
        ("external", m.extl()),
        ("interface", m.intf()),
        ("private", m.prvt()),
    ] {
        if intf.is_empty() {
            continue;
        }
        writeln!(out, "  {section}").unwrap();
        for v in intf.iter() {
            let n = name(v);
            writeln!(out, "    {n} : {}", v.dtype()).unwrap();
            vars.insert(X(v), format!("X({n})"));
            vars.insert(d(v), format!("d({n})"));
            vars.insert(**v, n);
        }
    }
    for atom in m.atoms() {
        print_atom(&mut out, atom, &vars);
    }
    out
}

fn print_atom(out: &mut String, a: &Atom, vars: &HashMap<Wire, String>) {
    let list = |intf: &Interface<Sort>| {
        intf.iter()
            .map(|v| vars[&**v].as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    write!(out, "  atom controls {}", list(a.ctrl())).unwrap();
    if !a.read().is_empty() {
        write!(out, " reads {}", list(a.read())).unwrap();
    }
    if !a.wait().is_empty() {
        write!(out, " awaits {}", list(a.wait())).unwrap();
    }
    out.push('\n');

    // local wires, numbered in the order they are first written
    let mut local: HashMap<Wire, String> = HashMap::new();
    for t in a.init().iter().chain(a.update().iter()) {
        for w in t.write() {
            if !vars.contains_key(w) && !local.contains_key(w) {
                local.insert(*w, format!("%{}", local.len() + 1));
            }
        }
    }
    let name = |w: &Wire| {
        vars.get(w)
            .or_else(|| local.get(w))
            .cloned()
            .unwrap_or_else(|| w.untyped().to_string())
    };
    let block = |terms: &mut dyn Iterator<Item = String>| {
        terms
            .map(|t| format!("      {}\n", t.trim_end()))
            .collect::<String>()
    };
    let init = block(&mut a.init().iter().map(|t| t.with_wirenames(name).to_string()));
    let update = block(
        &mut a
            .update()
            .iter()
            .map(|t| t.with_wirenames(name).to_string()),
    );
    if init == update {
        write!(out, "    init = update\n{init}").unwrap();
    } else {
        write!(out, "    init\n{init}    update\n{update}").unwrap();
    }
}
