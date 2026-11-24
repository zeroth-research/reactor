//! A plain-text listing of a tape, for reading and for tests.

use super::{NodeKind, Route, Tape, Value};
use std::fmt::{self, Display, Formatter, Write};

/// Writes `items` separated by `sep`.
fn join<T: Display>(f: &mut Formatter<'_>, items: &[T], sep: &str) -> fmt::Result {
    for (k, item) in items.iter().enumerate() {
        if k > 0 {
            f.write_str(sep)?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

/// Lists the tape arm by arm. The first line is the boundary. Each
/// arm then lists its input values, its nodes and its routes, one per
/// line, in storage order (so the root is the last route):
///
/// ```text
/// (R ⊗ D) ⊕ I → R ⊕ I
/// arm 0: R ⊗ D
///   in v0: R, v1: D
///   n0  v2 = gt0(v0)
///   n1  v3 = sample@0(v1)
///   r0  end 0 (v0)
///   r1  end 1 ()
///   r2  bind n1; split v3 ? r0 : r1
///   r3  split v2 ? r2 : r1
/// arm 1: I
///   in (none)
///   n2  v0 = zero()
///   r4  end 0 (v0)
/// ```
///
/// A non-deterministic node shows its level after `@`. A split
/// `split c ? hi : lo` continues with `hi` when `c` is true.
impl<S: Display, O: Display> Display for Tape<S, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let parts = self.parts();
        writeln!(f, "{} → {}", parts.dom, parts.cod)?;
        for arm in self.arms() {
            writeln!(f, "arm {}: {}", arm.index(), arm.inputs())?;
            f.write_str("  in ")?;
            if arm.inputs().iter().count() == 0 {
                f.write_str("(none)")?;
            }
            for (k, sort) in arm.inputs().iter().enumerate() {
                let sep = if k > 0 { ", " } else { "" };
                write!(f, "{sep}{}: {sort}", Value(k as u32))?;
            }
            f.write_char('\n')?;

            for node in arm.nodes() {
                let outputs: Vec<Value> = node.outputs().collect();
                write!(f, "  {:<3} ", node.id().to_string())?;
                if outputs.is_empty() {
                    f.write_str("()")?;
                } else {
                    join(f, &outputs, ", ")?;
                }
                f.write_str(" = ")?;
                match node.kind() {
                    NodeKind::Det(op) => write!(f, "{op}(")?,
                    NodeKind::NonDet { op, level } => write!(f, "{op}@{level}(")?,
                }
                join(f, node.args(), ", ")?;
                f.write_str(")\n")?;
            }

            for route in arm.routes() {
                write!(f, "  {:<3} ", route.id().to_string())?;
                let binds = route.binds();
                if !binds.is_empty() {
                    f.write_str("bind ")?;
                    join(f, binds, ", ")?;
                    f.write_str("; ")?;
                }
                match *route.route() {
                    Route::Split { cond, hi, lo, .. } => {
                        write!(f, "split {cond} ? {hi} : {lo}")?;
                    }
                    Route::End { output, .. } => {
                        write!(f, "end {output} (")?;
                        join(f, route.outs().unwrap_or_default(), ", ")?;
                        f.write_char(')')?;
                    }
                }
                f.write_char('\n')?;
            }
        }
        Ok(())
    }
}
