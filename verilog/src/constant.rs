/*!
# Constant expressions

Parameter values, and the integers in ranges, replication counts, select
bounds and size casts, are constant expressions. The frontend builds them
like any other expression ([`Expr`]), with the values of the parameters
declared before them as constants, and evaluates them when a module is
elaborated, once for each instance. Every operator of value expressions
can be used, with the width and signedness rules of IEEE 1800 (see
[`sizing`](crate::sizing)): the expression is sized at its own type, or,
for a typed parameter, as assigned to the parameter.

*Integer constant expressions* are the exception where an integer is
needed and in the values of untyped parameters: built from integer
constants (unsized decimal literals, `'sd5`, parameters of type `int`, and
`$clog2`) with `+ - * / % **` and unary `-` `+`, they are evaluated
exactly ([`exact`]), and every intermediate result must fit in an `int`
(32 bits, signed); an overflow is an error, as is a division by zero. So
whenever the evaluation succeeds, IEEE 1800's 32-bit arithmetic gives the
same value, and so do simulators that compute such expressions without
loss (iverilog). The least `int`, `-2147483648`, is the literal
2147483648, which is no `int` by itself, negated at once. In a value
expression, and as the value of a typed parameter, an integer constant is
not exact: it is evaluated at the type of its context like any other
operand (`65536 * 65536` assigned to 64 bits is 2^32, to an `int` 0).
Other operators, and operands of other types, follow the width rules
everywhere: `[W-1:0]` with `parameter [4:0] W` subtracts at 32 unsigned
bits.
*/

use crate::ast::{BinOp, Const, Error, Expr, ExprKind, Name, Ty, UnOp};
use crate::sizing::Sizer;
use crate::typing::Env;
use crate::value::BitVec;
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, ToPrimitive};

/// Why a constant expression is not evaluated.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Failure {
    /// The expression is no constant expression: it reads a signal; the
    /// reason is a message.
    NotConstant(String),
    /// The expression is a constant expression whose evaluation fails
    /// (overflow, division by zero, …).
    Error(Error),
}

/// A literal: sized (`8'hff`), an unsized decimal (`255`), which is an
/// integer, or unsized based (`'hff`, `'sd5`), which has 32 bits and is
/// unsigned unless marked `s` (then it is an integer too).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Literal {
    Sized(Ty, BitVec),
    Int(i64),
    Unsized(Ty, BitVec),
}

/// Why a literal with x or z digits is rejected, and where x is accepted.
pub const X_HINT: &str = "values are 2-state; x is accepted only as the whole value of an \
    assignment in a block that every path overwrites, e.g. `y = 'x;` before a `case`";

/// Parses a literal without whitespace and `_`: `N'[s]<b|o|d|h>digits` or
/// decimal digits, or an unsized based literal `'[s]<b|o|d|h>digits`. x/z
/// digits are rejected (values are 2-state), and so are values that do
/// not fit in the width (32 bits for unsized literals).
pub fn literal(text: &str) -> Result<Literal, String> {
    let decimal = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    let Some((size, rest)) = text.split_once('\'') else {
        if !decimal(text) {
            return Err(format!("malformed literal `{text}`"));
        }
        let value: i64 = text
            .parse()
            .ok()
            .filter(|v| *v <= i32::MAX as i64)
            .ok_or_else(|| format!("integer `{text}` too large"))?;
        return Ok(Literal::Int(value));
    };
    let width: usize = if size.is_empty() {
        32
    } else if !decimal(size) {
        return Err(format!("malformed literal `{text}`"));
    } else {
        size.parse()
            .map_err(|_| format!("literal size `{size}` too large"))?
    };
    if width == 0 {
        return Err(format!("literal `{text}` of width 0"));
    }
    let (signed, rest) = match rest.strip_prefix(['s', 'S']) {
        Some(rest) => (true, rest),
        None => (false, rest),
    };
    let mut chars = rest.chars();
    let radix = match chars.next().map(|c| c.to_ascii_lowercase()) {
        Some('b') => 2,
        Some('o') => 8,
        Some('d') => 10,
        Some('h') => 16,
        _ => return Err(format!("malformed literal `{text}`")),
    };
    let digits = chars.as_str();
    if digits.contains(['x', 'X', 'z', 'Z', '?']) {
        return Err(format!(
            "literal `{text}` with x/z digits not supported ({X_HINT})"
        ));
    }
    let value = BigUint::parse_bytes(digits.as_bytes(), radix)
        .ok_or_else(|| format!("malformed literal `{text}`"))?;
    let value = BitVec::new(width, value)
        .ok_or_else(|| format!("literal `{text}` does not fit in {width} bits"))?;
    let ty = Ty { width, signed };
    Ok(if size.is_empty() {
        Literal::Unsized(ty, value)
    } else {
        Literal::Sized(ty, value)
    })
}

/// The integer constant `v`.
pub fn int_const(v: i64) -> Const {
    Const {
        ty: Ty::INT,
        value: BitVec::from_signed(32, v),
    }
}

/// The value of an integer constant, `None` for a vector constant.
pub fn as_int(c: &Const) -> Option<i64> {
    (c.ty == Ty::INT).then(|| c.value.signed().to_i64().expect("32 bits"))
}

/// `a op b` on integers, for `op` one of `+ - * / % **`: exact, and an
/// error unless the result fits in 32 bits. Division rounds toward zero and
/// the remainder has the sign of the dividend, as in IEEE 1800; a negative
/// exponent gives 0 except for the bases 1 and -1, and is an error for the
/// base 0 (the result is x).
pub fn int_op(op: BinOp, a: i64, b: i64) -> Result<i64, String> {
    let r = match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div | BinOp::Mod if b == 0 => {
            return Err(format!("division by zero in `{a} {op} {b}`"));
        }
        BinOp::Div => a / b,
        BinOp::Mod => a % b,
        BinOp::Pow if b < 0 => match a {
            0 => {
                return Err(format!(
                    "`{a} ** {b}`: 0 to a negative power has no value (it is x)"
                ));
            }
            1 => 1,
            -1 if b % 2 == 0 => 1,
            -1 => -1,
            _ => 0,
        },
        BinOp::Pow => u32::try_from(b)
            .ok()
            .and_then(|b| a.checked_pow(b))
            .unwrap_or(i64::MAX),
        op => unreachable!("integer operator {op}"),
    };
    fits(r).ok_or_else(|| format!("`{a} {op} {b}` overflows 32 bits"))
}

/// `-a` on integers.
pub fn int_neg(a: i64) -> Result<i64, String> {
    fits(-a).ok_or_else(|| format!("`-({a})` overflows 32 bits"))
}

fn fits(v: i64) -> Option<i64> {
    (i32::MIN as i64..=i32::MAX as i64)
        .contains(&v)
        .then_some(v)
}

/// `$clog2(c)`: the number of bits needed for the values `0 .. n-1`, the
/// least `k` with `2^k ≥ n` (0 for `n` ≤ 1), for `n` the value of `c`; a
/// negative `int` is an error.
pub fn clog2(c: &Const) -> Result<i64, String> {
    if c.ty.signed && c.value.signed().is_negative() {
        return Err(format!(
            "`$clog2` of the negative value {}",
            c.value.signed()
        ));
    }
    let n = c.value.unsigned();
    let k = if *n <= BigUint::from(1u8) {
        0
    } else {
        (n - 1u8).bits()
    };
    Ok(k as i64)
}

/// The first signal (or array) that `e` reads, if any: then `e` is no
/// constant expression.
pub fn signal(e: &Expr) -> Option<&Name> {
    let mut reads = Vec::new();
    e.reads(&mut reads);
    reads.first().map(|(x, _)| *x)
}

/// The exact value of `e` if it is an integer constant expression (see the
/// module documentation): `None` if it is not one, an error message if an
/// intermediate result does not fit in an `int`, or for a division by
/// zero.
pub fn exact(e: &Expr) -> Option<Result<i64, String>> {
    match &e.kind {
        ExprKind::Const { ty, value, .. } if *ty == Ty::INT => {
            Some(Ok(value.signed().to_i64().expect("32 bits")))
        }
        ExprKind::Unary(UnOp::Plus, a) => exact(a),
        ExprKind::Unary(UnOp::Neg, a) => Some(exact(a)?.and_then(int_neg)),
        ExprKind::Binary(
            op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow),
            a,
            b,
        ) => {
            let (a, b) = (exact(a)?, exact(b)?);
            Some(a.and_then(|a| b.and_then(|b| int_op(*op, a, b))))
        }
        _ => None,
    }
}

/// The value of the constant expression `e` at its own type: exact for an
/// integer constant expression, else as IEEE 1800 evaluates it.
pub fn value(e: &Expr) -> Result<Const, Failure> {
    constant(e)?;
    match exact(e) {
        Some(Ok(v)) => Ok(int_const(v)),
        Some(Err(msg)) => Err(Failure::Error(Error::new(e.span, msg))),
        None => sized(e, None),
    }
}

/// The value of the constant expression `e` as assigned to a value of type
/// `ty` (a typed parameter): evaluated at the larger of both widths, then
/// truncated, as IEEE 1800 assigns.
pub fn assigned(e: &Expr, ty: Ty) -> Result<Const, Failure> {
    constant(e)?;
    let c = sized(e, Some(ty.width))?;
    Ok(Const { ty, value: c.value })
}

/// The value of the constant expression `e` as an integer (see [`value`]),
/// read as signed or unsigned as its type says.
pub fn int(e: &Expr) -> Result<i64, Failure> {
    let c = value(e)?;
    let n: BigInt = if c.ty.signed {
        c.value.signed()
    } else {
        c.value.unsigned().clone().into()
    };
    n.to_i64()
        .ok_or_else(|| Failure::Error(Error::new(e.span, format!("`{e}` is too large ({n})"))))
}

/// A value for a parameter given from outside (`v2rm -G N=V`): a literal,
/// or an unsized decimal with a sign (`-3`).
pub fn parse(text: &str) -> Result<Expr, String> {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(d) if d.contains('\'') => {
            return Err(format!("`{text}`: only an unsized decimal can have a sign"));
        }
        Some(d) => (true, d),
        None => (false, text),
    };
    let digits: String = digits.chars().filter(|c| *c != '_').collect();
    let (ty, value) = match literal(&digits) {
        Ok(Literal::Int(v)) if negative => (Ty::INT, BitVec::from_signed(32, -v)),
        Err(_) if negative && digits == "2147483648" => {
            (Ty::INT, BitVec::from_signed(32, i32::MIN))
        }
        Ok(Literal::Int(v)) => (Ty::INT, BitVec::from_signed(32, v)),
        Ok(Literal::Sized(ty, value) | Literal::Unsized(ty, value)) if !negative => (ty, value),
        Ok(_) => return Err(format!("`{text}`: only an unsized decimal can have a sign")),
        Err(m) => return Err(m),
    };
    Ok(Expr {
        kind: ExprKind::Const {
            text: text.to_string(),
            ty,
            value,
            is_unsized: false,
        },
        span: Default::default(),
    })
}

/// Fails unless `e` is a constant expression.
fn constant(e: &Expr) -> Result<(), Failure> {
    match signal(e) {
        Some(x) => Err(Failure::NotConstant(format!("`{x}` is not a parameter"))),
        None => Ok(()),
    }
}

/// `e` sized as IEEE 1800 says, at its own type or as assigned to `width`
/// bits, and folded into one constant.
fn sized(e: &Expr, width: Option<usize>) -> Result<Const, Failure> {
    let env = Env::new();
    let s = Sizer::constants(&env);
    let sized = match width {
        None => s.synth(e),
        Some(width) => s.assign(width, e),
    }
    .map_err(Failure::Error)?;
    match sized.kind {
        ExprKind::Const { ty, value, .. } | ExprKind::Lit { ty, value } => Ok(Const { ty, value }),
        _ => Err(Failure::Error(Error::new(
            e.span,
            format!("internal error: the constant expression `{e}` is not folded"),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers() {
        assert_eq!(int_op(BinOp::Sub, 8, 1), Ok(7));
        assert_eq!(int_op(BinOp::Pow, 2, 3), Ok(8));
        assert_eq!(int_op(BinOp::Div, -7, 2), Ok(-3));
        assert_eq!(int_op(BinOp::Mod, -7, 2), Ok(-1));
        assert_eq!(int_op(BinOp::Pow, 2, -1), Ok(0));
        assert_eq!(int_op(BinOp::Pow, -1, -3), Ok(-1));
        assert_eq!(int_op(BinOp::Pow, -1, -2), Ok(1));
        assert_eq!(int_op(BinOp::Pow, 1, -2), Ok(1));
        assert!(int_op(BinOp::Pow, 0, -1).is_err());
        assert!(int_op(BinOp::Div, 1, 0).is_err());
        assert!(int_op(BinOp::Mul, 65536, 65536).is_err());
        assert!(int_op(BinOp::Pow, 2, 31).is_err());
        assert_eq!(int_op(BinOp::Pow, -2, 31), Ok(i32::MIN as i64));
        assert_eq!(int_neg(i32::MAX as i64), Ok(-(i32::MAX as i64)));
        let int = |v: i64| int_const(v);
        assert_eq!(clog2(&int(8)), Ok(3));
        assert_eq!(clog2(&int(9)), Ok(4));
        assert_eq!(clog2(&int(1)), Ok(0));
        assert_eq!(clog2(&int(0)), Ok(0));
        assert!(clog2(&int(-1)).is_err());
        let wide = Const {
            ty: Ty::unsigned(40),
            value: BitVec::wrap(40, 1u64 << 35),
        };
        assert_eq!(clog2(&wide), Ok(35));
    }

    #[test]
    fn literals() {
        assert_eq!(literal("1000"), Ok(Literal::Int(1000)));
        assert!(literal("2147483648").is_err());
        assert!(literal("'hx").is_err());
        for text in ["", "abc", "1a", "4x'h3", "'", "8'"] {
            assert_eq!(literal(text), Err(format!("malformed literal `{text}`")));
        }
        let Ok(Literal::Sized(ty, v)) = literal("4'sb1010") else {
            panic!()
        };
        assert_eq!(
            (ty, v),
            (
                Ty {
                    width: 4,
                    signed: true
                },
                BitVec::wrap(4, 10u8)
            )
        );
        let Ok(Literal::Unsized(ty, v)) = literal("'hff") else {
            panic!()
        };
        assert_eq!((ty, v), (Ty::unsigned(32), BitVec::wrap(32, 255u8)));
    }

    #[test]
    fn outside_values() {
        let value = |t: &str| match parse(t).map(|e| e.kind) {
            Ok(ExprKind::Const { ty, value, .. }) => Ok((ty, value)),
            Ok(k) => panic!("{k:?}"),
            Err(m) => Err(m),
        };
        assert_eq!(value("6"), Ok((Ty::INT, BitVec::from_signed(32, 6))));
        assert_eq!(value("-3"), Ok((Ty::INT, BitVec::from_signed(32, -3))));
        assert_eq!(
            value("-2147483648"),
            Ok((Ty::INT, BitVec::from_signed(32, i32::MIN)))
        );
        assert_eq!(
            value("8'hff"),
            Ok((Ty::unsigned(8), BitVec::wrap(8, 255u8)))
        );
        assert!(value("-8'hff").is_err());
        assert!(value("N + 1").is_err());
    }
}
