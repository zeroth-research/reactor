//! The mutation tool of the planted-bug suite (`tests/planted.rs`): the
//! tokens of a source that a mutant changes, and the mutants of a source,
//! each with one change. Included as a module by the suite, which also
//! runs the direct tests at the end of this file.

/// A planted bug: one change to the source of an example.
pub struct Mutant {
    /// Where and what, e.g. ``8:24 `+` -> `-` ``.
    pub what: String,
    pub source: String,
}

/// The operators a mutant replaces, with their replacements; `""` deletes.
const OPERATORS: &[(&str, &str)] = &[
    ("+", "-"),
    ("-", "+"),
    ("*", "+"),
    ("/", "*"),
    ("%", "/"),
    ("&", "|"),
    ("|", "&"),
    ("^", "&"),
    ("~^", "^"),
    ("&&", "||"),
    ("||", "&&"),
    ("==", "!="),
    ("!=", "=="),
    ("<", "<="),
    ("<=", "<"),
    (">", ">="),
    (">=", ">"),
    ("<<", ">>"),
    (">>", "<<"),
    (">>>", ">>"),
    ("+:", "-:"),
    ("-:", "+:"),
    ("~", ""),
    ("!", ""),
];

/// The tokens of `source` that mutants change, with their byte ranges:
/// operators (longest match), sized literals `8'd3`, unsized literals
/// (decimal `3`, based `'d3`, `'0`, `'1`), the keywords `if`, `posedge` and
/// `negedge`, and `$signed`, `$unsigned`. Comments are skipped.
pub fn tokens(source: &str) -> Vec<(usize, usize, &str)> {
    let b = source.as_bytes();
    let mut ops: Vec<&str> = OPERATORS.iter().map(|(o, _)| *o).collect();
    ops.extend([
        "=", "(", ")", "[", "]", "{", "}", ":", "?", ";", ",", "'", "#", ".", "@",
    ]);
    ops.sort_by_key(|o| std::cmp::Reverse(o.len()));
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'$';
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        let start = i;
        if source[i..].starts_with("//") {
            i += source[i..].find('\n').unwrap_or(b.len() - i);
        } else if source[i..].starts_with("/*") {
            i += source[i..].find("*/").unwrap() + 2;
        } else if b[i].is_ascii_digit() {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
                i += 1;
            }
            let rest = &source[i..];
            let base = rest
                .strip_prefix('\'')
                .map(|r| r.strip_prefix('s').unwrap_or(r));
            if let Some(r) = base.filter(|r| r.starts_with(['b', 'o', 'd', 'h'])) {
                i = b.len() - r.len() + 1;
                while i < b.len() && (b[i].is_ascii_hexdigit() || b[i] == b'_') {
                    i += 1;
                }
            }
            out.push((start, i, &source[start..i]));
        } else if word(b[i]) {
            while i < b.len() && word(b[i]) {
                i += 1;
            }
            let w = &source[start..i];
            if ["if", "posedge", "negedge", "$signed", "$unsigned"].contains(&w) {
                out.push((start, i, w));
            }
        } else if b[i] == b'\''
            && (source[i + 1..].starts_with(['0', '1'])
                || source[i + 1..]
                    .trim_start_matches(['s', 'S'])
                    .starts_with(['b', 'o', 'd', 'h']))
        {
            // an unsized literal: `'0`, `'1`, `'d3`, `'shff`
            i += 1;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push((start, i, &source[start..i]));
        } else if let Some(op) = ops.iter().find(|o| source[i..].starts_with(**o)) {
            i += op.len();
            out.push((start, i, op));
        } else {
            i += 1;
        }
    }
    out
}

/// Every mutant of `source` with one change: an operator replaced (see
/// [`OPERATORS`]), a sized or unsized based literal one larger (wrapping)
/// or, if signed, unsigned (`8'sd3` becomes `8'd3`), an unsized decimal
/// number one smaller (`0` becomes `1`) or unsigned (`'d3` instead of the
/// `int` 3), `'0` and `'1` swapped, `$signed` or
/// `$unsigned` dropped (its operand stays), a name as select index
/// inverted (`x[i]` becomes `x[~i]`), the condition of an `if`
/// negated, or the edge of an `always_ff` block swapped (`posedge` and
/// `negedge`); for a block with an asynchronous reset also the polarity of
/// the reset flipped (`posedge rst` with `if (rst)` becomes `negedge rst`
/// with `if (!rst)`, and back), and the reset made synchronous (its event
/// removed).
pub fn mutants(source: &str) -> Vec<Mutant> {
    let at = |offset: usize| {
        let line = source[..offset].matches('\n').count() + 1;
        let col = offset - source[..offset].rfind('\n').map_or(0, |n| n + 1) + 1;
        format!("{line}:{col}")
    };
    let mutant = |start: usize, end: usize, to: &str| Mutant {
        what: format!("{} `{}` -> `{to}`", at(start), &source[start..end]),
        source: format!("{}{to}{}", &source[..start], &source[end..]),
    };
    let tokens = tokens(source);
    // the end of the parenthesis opened by token `k`
    let close = |k: usize| {
        let mut depth = 0;
        tokens[k..]
            .iter()
            .find(|(_, _, t)| {
                depth += match *t {
                    "(" => 1,
                    ")" => -1,
                    _ => 0,
                };
                depth == 0
            })
            .unwrap()
            .1
    };
    let mut out = vec![];
    for (k, &(start, end, text)) in tokens.iter().enumerate() {
        if text == "@" {
            out.extend(reset_mutants(source, &tokens, k, &close, &at));
        }
        if let Some((_, to)) = OPERATORS.iter().find(|(o, _)| *o == text) {
            out.push(mutant(start, end, to));
        } else if text == "posedge" {
            out.push(mutant(start, end, "negedge"));
        } else if text == "negedge" {
            out.push(mutant(start, end, "posedge"));
        } else if text == "$signed" || text == "$unsigned" {
            out.push(mutant(start, end, ""));
        } else if text == "'0" || text == "'1" {
            out.push(mutant(start, end, if text == "'0" { "'1" } else { "'0" }));
        } else if text == "[" {
            // a select whose index is a name: every index value changes
            let rest = &source[end..];
            let len = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            let after = rest[len..].trim_start();
            if rest.starts_with(|c: char| c.is_ascii_alphabetic())
                && (after.starts_with(']') || after.starts_with("+:") || after.starts_with("-:"))
            {
                let name = &rest[..len];
                out.push(Mutant {
                    what: format!("{} `[{name}` -> `[~{name}`", at(start)),
                    source: format!("{}~{}", &source[..end], &source[end..]),
                });
            }
        } else if text == "if" {
            // the condition: from `(` to its `)`
            let (open, _, _) = tokens[k + 1];
            let close = close(k + 1);
            let cond = &source[open..close];
            out.push(Mutant {
                what: format!("{} `if {cond}` -> `if (!{cond})`", at(start)),
                source: format!("{}(!{cond}){}", &source[..open], &source[close..]),
            });
        } else if let Some((size, rest)) = text.split_once('\'').filter(|(_, r)| !r.is_empty()) {
            // an unsized based literal has 32 bits
            let width: u32 = if size.is_empty() {
                32
            } else {
                size.parse().unwrap()
            };
            let (signed, rest) = match rest.strip_prefix('s') {
                Some(r) => ("s", r),
                None => ("", rest),
            };
            let radix = match &rest[..1] {
                "b" => 2,
                "o" => 8,
                "d" => 10,
                _ => 16,
            };
            let digits = rest[1..].replace('_', "");
            // x digits (only in assignments of x, which have no effect; the
            // tokens split `1'bx` after the base): no mutant
            let Ok(v) = u128::from_str_radix(&digits, radix) else {
                assert!(
                    digits.is_empty() || digits.to_ascii_lowercase().ends_with('x'),
                    "{text}"
                );
                continue;
            };
            if !signed.is_empty() {
                out.push(mutant(start, end, &format!("{size}'{rest}")));
            }
            let v = (v + 1) & ((1u128 << width) - 1);
            out.push(mutant(start, end, &format!("{size}'{signed}d{v}")));
        } else if text.starts_with(|c: char| c.is_ascii_digit()) {
            let n: u64 = text.replace('_', "").parse().unwrap();
            let to = if n == 0 { 1 } else { n - 1 };
            out.push(mutant(start, end, &to.to_string()));
            // unsigned instead of an `int`, where the sign can matter
            if !integer_position(&tokens, k) {
                out.push(mutant(start, end, &format!("'d{n}")));
            }
        }
    }
    out
}

/// Whether the number at token `k` is a whole integer position, where only
/// its value counts, not its type: a bound of a range or part select, the
/// width of an indexed part select, a cast size or a replication count.
fn integer_position(tokens: &[(usize, usize, &str)], k: usize) -> bool {
    let text = |i: usize| tokens.get(i).map(|t| t.2);
    if text(k + 1) == Some("'") && text(k + 2) == Some("(") {
        return true;
    }
    if k > 0 && text(k - 1) == Some("{") && text(k + 1) == Some("{") {
        return true;
    }
    // the innermost bracket around `k`
    let mut depth = 0;
    let Some(open) = (0..k).rev().find(|&i| {
        depth += match text(i) {
            Some("]") => 1,
            Some("[") => -1,
            _ => 0,
        };
        depth < 0
    }) else {
        return false;
    };
    let (mut depth, mut conditionals) = (0, 0);
    let mut colon = None;
    for i in open + 1.. {
        match text(i) {
            Some("[" | "(" | "{") => depth += 1,
            Some(")" | "}") => depth -= 1,
            Some("]") if depth == 0 => break,
            Some("]") => depth -= 1,
            // the `:` of a `?:` is no range
            Some("?") if depth == 0 => conditionals += 1,
            Some(":") if depth == 0 && conditionals > 0 => conditionals -= 1,
            Some(":" | "+:" | "-:") if depth == 0 => colon = Some(i),
            None => return false,
            _ => {}
        }
    }
    match colon {
        // `[b +: w]`: only the width
        Some(c) if text(c) != Some(":") => k > c,
        // `[hi:lo]`
        Some(_) => true,
        None => false,
    }
}

/// The mutants of the asynchronous reset of the block whose event control
/// starts at token `k` (`@`), if it has one: the event list `(e1 or e2)`
/// and the first `if` after it, whose condition `(rst)`, `(!rst)` or
/// `(~rst)` names the reset.
fn reset_mutants(
    source: &str,
    tokens: &[(usize, usize, &str)],
    k: usize,
    close: &dyn Fn(usize) -> usize,
    at: &dyn Fn(usize) -> String,
) -> Vec<Mutant> {
    if tokens[k + 1].2 != "(" {
        return vec![]; // `@*`
    }
    let (open, end) = (tokens[k + 1].0, close(k + 1));
    let list = &source[open + 1..end - 1];
    let events: Vec<&str> = list.split([',']).flat_map(|e| e.split(" or ")).collect();
    // two edge events (a level list `@(a or b)` has no reset)
    let [e1, e2] = events[..] else {
        return vec![];
    };
    if ![e1, e2].iter().all(|e| e.contains("edge ")) {
        return vec![];
    }
    let i = k + tokens[k..].iter().position(|t| t.2 == "if").unwrap();
    let (c0, c1) = (tokens[i + 1].0, close(i + 1));
    let cond = source[c0 + 1..c1 - 1].trim();
    let name = cond.trim_start_matches(['!', '~']);
    let edge_name = |e: &str| {
        let (edge, name) = e.trim().split_once(' ').unwrap();
        (edge.to_string(), name.trim().to_string())
    };
    let (r, c) = if edge_name(e1).1 == name {
        (e1, e2)
    } else {
        (e2, e1)
    };
    let (edge, rst) = edge_name(r);
    let (flipped, test) = if edge == "posedge" {
        ("negedge", format!("!{rst}"))
    } else {
        ("posedge", rst.clone())
    };
    let between = &source[end..c0];
    vec![
        Mutant {
            what: format!(
                "{} reset `{edge} {rst}`, `if ({cond})` -> `{flipped} {rst}`, `if ({test})`",
                at(open)
            ),
            source: format!(
                "{}({}){between}({test}){}",
                &source[..open],
                list.replacen(&format!("{edge} {rst}"), &format!("{flipped} {rst}"), 1),
                &source[c1..]
            ),
        },
        Mutant {
            what: format!("{} `({list})` -> `({})`", at(open), c.trim()),
            source: format!("{}({}){}", &source[..open], c.trim(), &source[end..]),
        },
    ]
}

#[test]
fn mutants_of_text() {
    let source = "// a + b\nassign y = a+b; always_ff @(posedge clk) if (en && (x)) q <= 4'hf;\n\
                  assign z = w[3 +: 2] >>> 8'sd1; always_ff @(negedge c) r <= 1'b0;";
    let whats: Vec<String> = mutants(source).into_iter().map(|m| m.what).collect();
    assert_eq!(
        whats,
        [
            "2:13 `+` -> `-`",
            "2:29 `posedge` -> `negedge`",
            "2:42 `if (en && (x))` -> `if (!(en && (x)))`",
            "2:49 `&&` -> `||`",
            "2:59 `<=` -> `<`",
            "2:62 `4'hf` -> `4'd0`",
            "3:14 `3` -> `2`",
            "3:14 `3` -> `'d3`",
            "3:16 `+:` -> `-:`",
            "3:19 `2` -> `1`",
            "3:22 `>>>` -> `>>`",
            "3:26 `8'sd1` -> `8'd1`",
            "3:26 `8'sd1` -> `8'sd2`",
            "3:45 `negedge` -> `posedge`",
            "3:58 `<=` -> `<`",
            "3:61 `1'b0` -> `1'd1`",
        ]
    );
    let m = mutants(source);
    assert!(m[1].source.contains("@(negedge clk) if"));
    assert!(m[2].source.contains("if (!(en && (x))) q"));
    assert!(m[5].source.contains("q <= 4'd0;"));
    assert!(m[13].source.contains("@(posedge c) r"));
    // names as select indices, not in declarations or nested selects
    let source =
        "logic [W-1:0] v; assign y = a[i] ^ b[j +: 2] ^ c[k[1:0]]; always_comb z[n] = 1'b0;";
    let m = mutants(source);
    let whats: Vec<&str> = m.iter().map(|m| m.what.as_str()).collect();
    assert_eq!(
        whats,
        [
            // no `'d` mutants of range bounds and widths: only the value counts
            "1:9 `-` -> `+`",
            "1:10 `1` -> `0`",
            "1:12 `0` -> `1`",
            "1:30 `[i` -> `[~i`",
            "1:34 `^` -> `&`",
            "1:37 `[j` -> `[~j`",
            "1:40 `+:` -> `-:`",
            "1:43 `2` -> `1`",
            "1:46 `^` -> `&`",
            "1:52 `1` -> `0`",
            "1:54 `0` -> `1`",
            "1:72 `[n` -> `[~n`",
            "1:78 `1'b0` -> `1'd1`",
        ]
    );
    assert!(m[3].source.contains("a[~i] ^"));
    // unsized literals and sign casts
    let source = "assign y = $signed(a) + 'd3 + 'sh7 + '1 + $unsigned('0);";
    let whats: Vec<String> = mutants(source).into_iter().map(|m| m.what).collect();
    assert_eq!(
        whats,
        [
            "1:12 `$signed` -> ``",
            "1:23 `+` -> `-`",
            "1:25 `'d3` -> `'d4`",
            "1:29 `+` -> `-`",
            "1:31 `'sh7` -> `'h7`",
            "1:31 `'sh7` -> `'sd8`",
            "1:36 `+` -> `-`",
            "1:38 `'1` -> `'0`",
            "1:41 `+` -> `-`",
            "1:43 `$unsigned` -> ``",
            "1:53 `'0` -> `'1`",
        ]
    );
    let source = "always_ff @(posedge clk, negedge rst_n)\n  if (~rst_n) q <= 1'b0; else q <= d;";
    let m = mutants(source);
    let whats: Vec<&str> = m.iter().map(|m| m.what.as_str()).collect();
    assert_eq!(
        whats[..2],
        [
            "1:12 reset `negedge rst_n`, `if (~rst_n)` -> `posedge rst_n`, `if (rst_n)`",
            "1:12 `(posedge clk, negedge rst_n)` -> `(posedge clk)`",
        ]
    );
    assert_eq!(
        m[0].source,
        "always_ff @(posedge clk, posedge rst_n)\n  if (rst_n) q <= 1'b0; else q <= d;"
    );
    assert_eq!(
        m[1].source,
        "always_ff @(posedge clk)\n  if (~rst_n) q <= 1'b0; else q <= d;"
    );
}

/// The tokens: operators by longest match, literals of every form with
/// their bases and digits, the keywords that mutants change (other words
/// are skipped), nothing inside comments.
#[test]
fn tokens_of_text() {
    let texts = |source: &str| -> Vec<String> {
        tokens(source)
            .into_iter()
            .map(|(start, end, t)| {
                assert_eq!(&source[start..end], t);
                t.to_string()
            })
            .collect()
    };
    assert_eq!(
        texts("y = a >>> 2 <= b ~^ c; // a + b\n/* if */ z = !x;"),
        ["=", ">>>", "2", "<=", "~^", ";", "=", "!", ";"]
    );
    assert_eq!(
        texts("8'sd3 + 4'hF_f - 1_000 + 'd7 + 'sh7 + '0 + '1 + 16'b1010"),
        [
            "8'sd3", "+", "4'hF_f", "-", "1_000", "+", "'d7", "+", "'sh7", "+", "'0", "+", "'1",
            "+", "16'b1010"
        ]
    );
    assert_eq!(
        texts("always_ff @(posedge c, negedge r) if (r) q <= $signed(a) >> $unsigned(b);"),
        [
            "@",
            "(",
            "posedge",
            ",",
            "negedge",
            ")",
            "if",
            "(",
            ")",
            "<=",
            "$signed",
            "(",
            ")",
            ">>",
            "$unsigned",
            "(",
            ")",
            ";"
        ]
    );
    // a part select and a cast
    assert_eq!(
        texts("w[3 +: 2] = 4'(x[1:0]);"),
        [
            "[", "3", "+:", "2", "]", "=", "4", "'", "(", "[", "1", ":", "0", "]", ")", ";"
        ]
    );
}

/// Integer positions: range bounds, the width of an indexed part select
/// (not its base), a cast size and a replication count; not an operand,
/// a constant index or the arms of `?:` inside a select.
#[test]
fn integer_positions() {
    let at = |source: &str, number: &str| {
        let tokens = tokens(source);
        let k = tokens.iter().position(|t| t.2 == number).unwrap();
        integer_position(&tokens, k)
    };
    assert!(at("logic [7:0] v;", "7"));
    assert!(at("logic [7:0] v;", "0"));
    assert!(at("assign y = v[5:2];", "2"));
    assert!(at("assign y = v[i +: 3];", "3"));
    assert!(!at("assign y = v[5 +: 3];", "5"));
    assert!(at("assign y = 4'(v);", "4"));
    assert!(at("assign y = {2{v}};", "2"));
    assert!(!at("assign y = v + 3;", "3"));
    assert!(!at("assign y = v[3];", "3"));
    assert!(!at("assign y = v[s ? 3 : 1];", "3"));
    assert!(!at("assign y = {v, 2};", "2"));
}
