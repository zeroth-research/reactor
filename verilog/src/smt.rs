/*!
# SvOp terms as SMT-LIB

Prints sorts, values and op applications of [`SvOp`] as SMT-LIB 2.6 text
in the logic `QF_BV`. Each function returns one SMT-LIB expression; a
caller combines them into a script (declarations, definitions, asserts).

## Sorts and values

- The sort of width `N` is `(_ BitVec N)`. There is no `Bool` sort for
  1-bit values: conditions and the results of comparisons and reductions
  are `(_ BitVec 1)`, with `#b1` for true. Where SMT-LIB needs a `Bool`
  (inside `ite`) it is `(= c #b1)`; where an SMT-LIB predicate gives one
  (`=`, `bvult`, …) it is turned back with `(ite p #b1 #b0)`.
- A value of width `N` is a binary literal with exactly `N` digits, most
  significant first: `8'd5` is `#b00000101`.
- A name is a quoted symbol `|name|`. Generated names contain `@`, `%`, `.`
  or `*`, which quoting allows; `|` and `\` cannot occur in a name.

## Ops

Every op application has the sort `(_ BitVec w)` of its [result
sort](SvOp::result_sort) and the value [`value::eval`](crate::value::eval)
gives it. For operands `a`, `b`, `c` of width `w` (and `k` for a shift
amount):

| op | SMT-LIB |
|---|---|
| `Lit(v)` | the literal of `v` |
| `Id` | `a` |
| `Not And Or Xor` | `bvnot bvand bvor bvxor` |
| `Neg Add Sub Mul` | `bvneg bvadd bvsub bvmul` |
| `UDiv URem` | `bvudiv bvurem` |
| `SDiv SRem` | `bvsdiv bvsrem` (`bvsrem` has the sign of the dividend, unlike `bvsmod`) |
| `Eq Ne` | `(ite (= a b) #b1 #b0)`, `(ite (= a b) #b0 #b1)` |
| `ULt ULe SLt SLe` | `(ite (bvult a b) #b1 #b0)`, with `bvule bvslt bvsle` |
| `RedAnd` | `(ite (= a #b1…1) #b1 #b0)` |
| `RedOr` | `(ite (= a #b0…0) #b0 #b1)` |
| `RedXor` | `(bvxor ((_ extract 0 0) a) … ((_ extract w-1 w-1) a))`; `a` if `w = 1` |
| `Ite` | `(ite (= a #b1) b c)` |
| `Shl LShr AShr` | `(bvshl a b')`, with `bvlshr bvashr`, and `b'` below |
| `Extract{hi,lo}` | `((_ extract hi lo) a)` |
| `Concat` | `(concat a b)` (the first operand gives the high bits) |
| `ZExt(to)` `SExt(to)` | `((_ zero_extend k) a)`, `((_ sign_extend k) a)` with `k = to - w`; `a` if `k = 0` |

SMT-LIB shifts need an amount of the value's width `w`, while `SvOp` takes
an amount of any width `k`. The amount `b'` is `b` if `k = w`,
`((_ zero_extend w-k) b)` if `k < w`, and otherwise `b` saturated at `w`:
`(ite (bvult b W) ((_ extract w-1 0) b) Wʹ)`, where `W` and `Wʹ` are the
literals of the number `w` in `k` and `w` bits (`w < 2^w`, so it fits). A
shift by `w` or more gives 0 (`AShr`: copies of the top bit) in both
languages; plain truncation of `b` would not.

Division by zero cannot occur in a translated design (the divisor is a
nonzero literal), and [`value::eval`](crate::value::eval) leaves it
undefined. SMT-LIB defines it: `bvudiv` by 0 gives all ones, `bvurem` the
dividend, `bvsdiv` all ones for a non-negative dividend and 1 otherwise,
`bvsrem` the dividend.

`Havoc` (any value) has no closed expression: a caller declares a fresh
constant for it. `Zero` writes a derivative and only occurs in delay
blocks, which have no counterpart here. [`apply`] panics on both.

Operands are repeated in the output (`RedXor`, saturated shift amounts),
so they should be short: names or literals.
*/

use crate::theory::{Sort, SvOp};
use crate::value::BitVec;

/// `(_ BitVec N)`.
pub fn sort(s: Sort) -> String {
    format!("(_ BitVec {})", s.width())
}

/// The binary literal of `v`, with one digit per bit.
pub fn lit(v: &BitVec) -> String {
    let digits: String = (0..v.width())
        .rev()
        .map(|i| if v.unsigned().bit(i as u64) { '1' } else { '0' })
        .collect();
    format!("#b{digits}")
}

/// The quoted symbol `|name|`. Panics if the name contains `|` or `\`,
/// which a quoted symbol cannot.
pub fn symbol(name: &str) -> String {
    assert!(
        !name.contains(['|', '\\']),
        "name {name:?} cannot be an SMT-LIB symbol"
    );
    format!("|{name}|")
}

/// The application of `op` to the operands, each given as an expression
/// and its sort. Panics if the op does not apply to the sorts, and on
/// `Havoc` and `Zero`.
pub fn apply(op: &SvOp, args: &[(String, Sort)]) -> String {
    use SvOp::*;
    let sorts: Vec<Sort> = args.iter().map(|(_, s)| *s).collect();
    let w = match op.result_sort(&sorts) {
        Ok(s) => s.width(),
        Err(e) => panic!("{e}"),
    };
    let a = |i: usize| args[i].0.as_str();
    let width = |i: usize| args[i].1.width();
    let call = |f: &str| {
        let operands: Vec<&str> = (0..args.len()).map(a).collect();
        format!("({f} {})", operands.join(" "))
    };
    // a `Bool` as a 1-bit vector
    let bit = |p: String| format!("(ite {p} #b1 #b0)");

    match op {
        Lit(v) => lit(v),
        Havoc(_) | Zero(_) => panic!("{op} has no SMT-LIB expression"),
        Id => a(0).to_string(),

        Not => call("bvnot"),
        And => call("bvand"),
        Or => call("bvor"),
        Xor => call("bvxor"),

        Neg => call("bvneg"),
        Add => call("bvadd"),
        Sub => call("bvsub"),
        Mul => call("bvmul"),
        UDiv => call("bvudiv"),
        URem => call("bvurem"),
        SDiv => call("bvsdiv"),
        SRem => call("bvsrem"),

        Eq => bit(call("=")),
        Ne => format!("(ite {} #b0 #b1)", call("=")),
        ULt => bit(call("bvult")),
        ULe => bit(call("bvule")),
        SLt => bit(call("bvslt")),
        SLe => bit(call("bvsle")),

        RedAnd => bit(format!("(= {} {})", a(0), lit(&ones(width(0))))),
        RedOr => format!(
            "(ite (= {} {}) #b0 #b1)",
            a(0),
            lit(&BitVec::zero(width(0)))
        ),
        RedXor if width(0) == 1 => a(0).to_string(),
        RedXor => {
            let bits: Vec<String> = (0..width(0))
                .map(|i| format!("((_ extract {i} {i}) {})", a(0)))
                .collect();
            format!("(bvxor {})", bits.join(" "))
        }

        Ite => format!("(ite (= {} #b1) {} {})", a(0), a(1), a(2)),

        Shl | LShr | AShr => {
            let f = match op {
                Shl => "bvshl",
                LShr => "bvlshr",
                _ => "bvashr",
            };
            format!("({f} {} {})", a(0), amount(a(1), width(1), w))
        }

        Extract { hi, lo } => format!("((_ extract {hi} {lo}) {})", a(0)),
        Concat => call("concat"),
        ZExt(to) | SExt(to) if *to == width(0) => a(0).to_string(),
        ZExt(to) => format!("((_ zero_extend {}) {})", to - width(0), a(0)),
        SExt(to) => format!("((_ sign_extend {}) {})", to - width(0), a(0)),
    }
}

/// The shift amount `b` of width `k` as an amount of width `w`, saturated
/// at `w` if it is wider.
fn amount(b: &str, k: usize, w: usize) -> String {
    use std::cmp::Ordering::*;
    match k.cmp(&w) {
        Equal => b.to_string(),
        Less => format!("((_ zero_extend {}) {b})", w - k),
        Greater => format!(
            "(ite (bvult {b} {}) ((_ extract {} 0) {b}) {})",
            lit(&BitVec::wrap(k, w)),
            w - 1,
            lit(&BitVec::wrap(w, w)),
        ),
    }
}

/// The value with all `w` bits set.
fn ones(w: usize) -> BitVec {
    BitVec::from_signed(w, -1)
}
