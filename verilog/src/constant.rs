/*!
# Constant expressions

Parameter values, and the integers in ranges, replication counts, select
bounds and size casts, are constant expressions. They are evaluated when a
module is elaborated, once for each instance, from their source text and
the values of the parameters declared before them.

An *integer constant* has type `int` (`logic signed [31:0]`; with 2-state
values the two are the same). It is built from unsized decimal literals,
parameters of that type, `+ - * / % **`, unary `-` and `+`, parentheses
and `$clog2`. The arithmetic here is exact, and a result outside the
32-bit range is an error, as are a division by zero and a negative
exponent. So whenever the evaluation succeeds, the 32-bit arithmetic of
IEEE 1800 gives the same value, and so does any wider arithmetic a context
may impose.

A *vector constant* is a sized literal, a parameter of another type, or a
size cast `N'(e)`; there is no arithmetic on vector constants.
*/

use crate::ast::{Const, Name, Ty};
use crate::value::BitVec;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::collections::BTreeMap;

/// Why a text is not evaluated.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Failure {
    /// The text is no constant expression, e.g. it names a signal; the
    /// reason is a message.
    NotConstant(String),
    /// The text is a constant expression whose evaluation fails (overflow,
    /// division by zero, …).
    Error(String),
}

/// A literal: sized (`8'hff`), or an unsized decimal (`255`), which is an
/// integer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Literal {
    Sized(Ty, BitVec),
    Int(i64),
}

/// Parses a literal without whitespace and `_`: `N'[s]<b|o|d|h>digits` or
/// decimal digits. x/z digits are rejected (values are 2-state), and so
/// are unsized based literals (`'hff`): their width would come from the
/// context.
pub fn literal(text: &str) -> Result<Literal, String> {
    let Some((size, rest)) = text.split_once('\'') else {
        let value: i64 = text
            .parse()
            .ok()
            .filter(|v| *v <= i32::MAX as i64)
            .ok_or_else(|| format!("integer `{text}` too large"))?;
        return Ok(Literal::Int(value));
    };
    if size.is_empty() {
        return Err(format!(
            "unsized literal `{text}` not supported, write the width (e.g. `8{text}`)"
        ));
    }
    let width: usize = size
        .parse()
        .map_err(|_| format!("literal size `{size}` too large"))?;
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
            "literal `{text}` with x/z digits not supported (2-state)"
        ));
    }
    let value = BigUint::parse_bytes(digits.as_bytes(), radix)
        .ok_or_else(|| format!("malformed literal `{text}`"))?;
    let value = BitVec::new(width, value)
        .ok_or_else(|| format!("literal `{text}` does not fit in {width} bits"))?;
    Ok(Literal::Sized(Ty { width, signed }, value))
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
/// the remainder has the sign of the dividend, as in IEEE 1800.
pub fn int_op(op: &str, a: i64, b: i64) -> Result<i64, String> {
    let r = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" | "%" if b == 0 => return Err(format!("division by zero in `{a} {op} {b}`")),
        "/" => a / b,
        "%" => a % b,
        "**" if b < 0 => return Err(format!("negative exponent in `{a} ** {b}`")),
        "**" => u32::try_from(b)
            .ok()
            .and_then(|b| a.checked_pow(b))
            .unwrap_or(i64::MAX),
        _ => unreachable!("integer operator {op}"),
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

/// `$clog2(n)`: the number of bits needed for the values `0 .. n-1`, the
/// least `k` with `2^k ≥ n` (0 for `n` ≤ 1).
pub fn clog2(n: i64) -> Result<i64, String> {
    if n < 0 {
        return Err(format!("`$clog2` of the negative value {n}"));
    }
    let mut k = 0;
    while (1i64 << k) < n {
        k += 1;
    }
    Ok(k)
}

/// `N'(c)`: truncates, or extends as the signedness of `c` says; the
/// result keeps it.
pub fn resize(width: usize, c: &Const) -> Const {
    let value = if c.ty.signed {
        BitVec::from_signed(width, c.value.signed())
    } else {
        BitVec::wrap(width, c.value.unsigned().clone())
    };
    Const {
        ty: Ty { width, ..c.ty },
        value,
    }
}

/// Evaluates the constant expression `text`, with `params` the values of
/// the parameters in scope.
pub fn eval(text: &str, params: &BTreeMap<Name, Const>) -> Result<Const, Failure> {
    let tokens = lex(text)?;
    let mut p = Parser {
        tokens,
        at: 0,
        params,
    };
    let c = p.add()?;
    if p.at < p.tokens.len() {
        return Err(Failure::NotConstant(format!(
            "`{text}` is not a constant expression"
        )));
    }
    Ok(c)
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Token {
    /// A literal, sized or not.
    Num(String),
    Ident(String),
    /// `$clog2`
    Clog2,
    /// `+ - * / % **`
    Op(&'static str),
    Open,
    Close,
    /// `'` of a size cast `N'(e)`
    Tick,
}

fn lex(text: &str) -> Result<Vec<Token>, Failure> {
    let not_constant = || Failure::NotConstant(format!("`{text}` is not a constant expression"));
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    let take = |i: &mut usize, ok: &dyn Fn(char) -> bool| {
        let start = *i;
        while *i < chars.len() && (ok(chars[*i]) || chars[*i] == '_') {
            *i += 1;
        }
        chars[start..*i]
            .iter()
            .filter(|c| **c != '_')
            .collect::<String>()
    };
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let mut num = take(&mut i, &|c| c.is_ascii_digit());
            // a sized literal `N'[s]<base>digits`, else the size of a cast
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let base = |k: usize| {
                chars
                    .get(k)
                    .is_some_and(|c| "bodhBODH".contains(*c) && c.is_ascii_alphabetic())
            };
            if chars.get(j) == Some(&'\'') {
                let s = matches!(chars.get(j + 1), Some('s' | 'S'));
                let b = j + 1 + s as usize;
                if base(b) {
                    num.push('\'');
                    if s {
                        num.push('s');
                    }
                    num.push(chars[b]);
                    i = b + 1;
                    while i < chars.len() && chars[i].is_whitespace() {
                        i += 1;
                    }
                    num += &take(&mut i, &|c| c.is_ascii_alphanumeric() || c == '?');
                }
            }
            tokens.push(Token::Num(num));
        } else if c.is_ascii_alphabetic() || c == '_' {
            tokens.push(Token::Ident(take(&mut i, &|c| {
                c.is_ascii_alphanumeric() || c == '$'
            })));
        } else if c == '$' {
            i += 1;
            let name = take(&mut i, &|c| c.is_ascii_alphanumeric());
            if name != "clog2" {
                return Err(Failure::NotConstant(format!(
                    "`${name}` not supported in constant expressions"
                )));
            }
            tokens.push(Token::Clog2);
        } else if c == '\'' {
            // an unsized based literal, or the tick of a cast
            if chars.get(i + 1) == Some(&'(') {
                tokens.push(Token::Tick);
                i += 1;
            } else {
                let lit: String = chars[i..]
                    .iter()
                    .take_while(|c| c.is_ascii_alphanumeric() || **c == '\'' || **c == '_')
                    .collect();
                return Err(Failure::Error(
                    literal(&lit).expect_err("an unsized literal"),
                ));
            }
        } else {
            let op = match (c, chars.get(i + 1)) {
                ('*', Some('*')) => "**",
                ('+', _) => "+",
                ('-', _) => "-",
                ('*', _) => "*",
                ('/', _) => "/",
                ('%', _) => "%",
                ('(', _) => {
                    tokens.push(Token::Open);
                    i += 1;
                    continue;
                }
                (')', _) => {
                    tokens.push(Token::Close);
                    i += 1;
                    continue;
                }
                _ => return Err(not_constant()),
            };
            i += op.len();
            tokens.push(Token::Op(op));
        }
    }
    Ok(tokens)
}

struct Parser<'p> {
    tokens: Vec<Token>,
    at: usize,
    params: &'p BTreeMap<Name, Const>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn eat(&mut self, t: &Token) -> bool {
        if self.peek() == Some(t) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Token, what: &str) -> Result<(), Failure> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(Failure::NotConstant(format!("expected {what}")))
        }
    }

    /// Left-associative binary operators `ops` over `next`.
    fn chain(
        &mut self,
        ops: &[&str],
        next: fn(&mut Self) -> Result<Const, Failure>,
    ) -> Result<Const, Failure> {
        let mut acc = next(self)?;
        while let Some(Token::Op(op)) = self.peek().cloned() {
            if !ops.contains(&op) {
                break;
            }
            self.at += 1;
            let b = next(self)?;
            let (Some(x), Some(y)) = (as_int(&acc), as_int(&b)) else {
                let t = if as_int(&acc).is_some() { b.ty } else { acc.ty };
                return Err(Failure::Error(format!(
                    "`{op}` on the sized constant of type `{t}` not supported in a \
                     constant expression (use `int` parameters and unsized literals)"
                )));
            };
            acc = int_const(int_op(op, x, y).map_err(Failure::Error)?);
        }
        Ok(acc)
    }

    fn add(&mut self) -> Result<Const, Failure> {
        self.chain(&["+", "-"], Self::mul)
    }

    fn mul(&mut self) -> Result<Const, Failure> {
        self.chain(&["*", "/", "%"], Self::pow)
    }

    fn pow(&mut self) -> Result<Const, Failure> {
        self.chain(&["**"], Self::unary)
    }

    fn unary(&mut self) -> Result<Const, Failure> {
        if self.eat(&Token::Op("+")) {
            return self.int_operand("+");
        }
        if self.eat(&Token::Op("-")) {
            let v = self.int_operand("-")?;
            return Ok(int_const(
                int_neg(as_int(&v).unwrap()).map_err(Failure::Error)?,
            ));
        }
        let c = self.atom()?;
        if self.eat(&Token::Tick) {
            self.expect(&Token::Open, "`(` after `'`")?;
            let e = self.add()?;
            self.expect(&Token::Close, "`)`")?;
            let width = match as_int(&c) {
                Some(n) if n > 0 => n as usize,
                _ => {
                    return Err(Failure::Error(
                        "cast size must be a positive integer".into(),
                    ));
                }
            };
            return Ok(resize(width, &e));
        }
        Ok(c)
    }

    /// The integer operand of the unary operator `op`.
    fn int_operand(&mut self, op: &str) -> Result<Const, Failure> {
        let c = self.unary()?;
        if as_int(&c).is_none() {
            return Err(Failure::Error(format!(
                "unary `{op}` on the sized constant of type `{}` not supported in a \
                 constant expression",
                c.ty
            )));
        }
        Ok(c)
    }

    fn atom(&mut self) -> Result<Const, Failure> {
        let Some(t) = self.peek().cloned() else {
            return Err(Failure::NotConstant("expected an operand".into()));
        };
        self.at += 1;
        match t {
            Token::Num(text) => match literal(&text).map_err(Failure::Error)? {
                Literal::Int(v) => Ok(int_const(v)),
                Literal::Sized(ty, value) => Ok(Const { ty, value }),
            },
            Token::Ident(x) => self
                .params
                .get(&x)
                .cloned()
                .ok_or_else(|| Failure::NotConstant(format!("`{x}` is not a parameter"))),
            Token::Clog2 => {
                self.expect(&Token::Open, "`(` after `$clog2`")?;
                let c = self.add()?;
                self.expect(&Token::Close, "`)`")?;
                let n = as_int(&c).ok_or_else(|| {
                    Failure::Error(format!("`$clog2` of the sized constant of type `{}`", c.ty))
                })?;
                Ok(int_const(clog2(n).map_err(Failure::Error)?))
            }
            Token::Open => {
                let c = self.add()?;
                self.expect(&Token::Close, "`)`")?;
                Ok(c)
            }
            _ => Err(Failure::NotConstant("expected an operand".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(text: &str) -> Result<i64, Failure> {
        let params = BTreeMap::from([
            ("N".to_string(), int_const(8)),
            (
                "K".to_string(),
                Const {
                    ty: Ty::unsigned(4),
                    value: BitVec::wrap(4, 5u8),
                },
            ),
        ]);
        eval(text, &params).map(|c| as_int(&c).expect("an integer"))
    }

    #[test]
    fn integers() {
        assert_eq!(int("N - 1"), Ok(7));
        assert_eq!(int("2 + 3 * 4"), Ok(14));
        assert_eq!(int("(2 + 3) * 4"), Ok(20));
        assert_eq!(int("2 ** 3 ** 2"), Ok(64)); // left-associative
        assert_eq!(int("-7 / 2"), Ok(-3));
        assert_eq!(int("-7 % 2"), Ok(-1));
        assert_eq!(int("$clog2(N)"), Ok(3));
        assert_eq!(int("$clog2(N + 1)"), Ok(4));
        assert_eq!(int("$clog2(1)"), Ok(0));
        assert_eq!(int("1_000"), Ok(1000));
        assert!(matches!(int("1 / (N - 8)"), Err(Failure::Error(_))));
        assert!(matches!(int("65536 * 65536"), Err(Failure::Error(_))));
        assert!(matches!(int("2 ** -1"), Err(Failure::Error(_))));
        assert!(matches!(int("K + 1"), Err(Failure::Error(_))));
        assert!(matches!(int("x + 1"), Err(Failure::NotConstant(_))));
        assert!(matches!(int("N == 1"), Err(Failure::NotConstant(_))));
        assert!(matches!(int("'hff"), Err(Failure::Error(_))));
    }

    #[test]
    fn vectors() {
        let params = BTreeMap::from([("N".to_string(), int_const(8))]);
        let c = eval("4'sb1010", &params).unwrap();
        assert_eq!(
            c.ty,
            Ty {
                width: 4,
                signed: true
            }
        );
        assert_eq!(c.value, BitVec::wrap(4, 10u8));
        let c = eval("N'(N - 9)", &params).unwrap();
        assert_eq!(
            c.ty,
            Ty {
                width: 8,
                signed: true
            }
        );
        assert_eq!(c.value, BitVec::wrap(8, 255u8));
        let c = eval("2'(7)", &params).unwrap();
        assert_eq!(c.value, BitVec::wrap(2, 3u8));
        let c = eval("8 'h 1F", &params).unwrap();
        assert_eq!(c.value, BitVec::wrap(8, 31u8));
    }
}
