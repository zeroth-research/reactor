//! Arrays: element selects, `$readmemb`/`$readmemh`, and resolving element accesses once all
//! declarations are known.

use super::{Cx, Result, system_call};
use crate::ast::{
    Array, Elem, Elems, Error, Expr, ExprKind, Index, Init, Item, Lvalue, LvalueKind, Module,
    Name, Rom, Select, SignalId, Span, Stmt, StmtKind,
};
use crate::value::BitVec;
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use sv_parser as sv;

/// An array loaded by `$readmemb` or `$readmemh`: the contents of a ROM, or
/// the initial values of the elements of a written array.
pub(super) struct Load {
    array: Name,
    binary: bool,
    /// The file name as written, and the file read.
    file: String,
    path: PathBuf,
    /// The value of each element, if the file gives one.
    values: Vec<Option<BitVec>>,
    span: Span,
}

/// The words of a file read by `$readmemb` (`binary`) or `$readmemh` for
/// the array `m`: the value of each element the file gives one. Words are
/// separated by white space and `//` and `/* */` comments, `_` in words is
/// ignored, and `@a` (hexadecimal) sets the address of the next word. Every
/// word must fit the element; x and z digits are not supported (2-state).
pub(super) fn readmem(
    text: &str,
    binary: bool,
    m: &Array,
) -> std::result::Result<Vec<Option<BitVec>>, String> {
    let mut values: Vec<Option<BitVec>> = vec![None; m.size];
    let mut address = 0usize;
    let mut rest = text;
    let mut line = 1;
    loop {
        // skip white space and comments
        let trimmed = rest.trim_start();
        line += rest[..rest.len() - trimmed.len()].matches('\n').count();
        rest = trimmed;
        if let Some(r) = rest.strip_prefix("//") {
            rest = r.find('\n').map_or("", |k| &r[k..]);
            continue;
        }
        if let Some(r) = rest.strip_prefix("/*") {
            let k = r
                .find("*/")
                .ok_or(format!("line {line}: comment not closed"))?;
            line += r[..k].matches('\n').count();
            rest = &r[k + 2..];
            continue;
        }
        if rest.is_empty() {
            break;
        }
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let word = &rest[..end];
        rest = &rest[end..];
        let digits: String = word.chars().filter(|c| *c != '_').collect();
        if let Some(a) = digits.strip_prefix('@') {
            address = usize::from_str_radix(a, 16)
                .map_err(|_| format!("line {line}: bad address `{word}`"))?;
            continue;
        }
        if digits.chars().any(|c| "xXzZ?".contains(c)) {
            return Err(format!(
                "line {line}: `{word}` has x/z digits (not supported: values are 2-state)"
            ));
        }
        let value =
            BigUint::parse_bytes(digits.as_bytes(), if binary { 2 } else { 16 }).ok_or(format!(
                "line {line}: `{word}` is no {} number",
                if binary { "binary" } else { "hexadecimal" }
            ))?;
        if value.bits() as usize > m.ty.width {
            return Err(format!(
                "line {line}: `{word}` does not fit the {} bits of an element of `{}`",
                m.ty.width, m.name
            ));
        }
        let slot = values.get_mut(address).ok_or(format!(
            "line {line}: `{word}` would go to address {address}, but `{}` has addresses 0 to {}",
            m.name,
            m.size - 1
        ))?;
        *slot = Some(BitVec::wrap(m.ty.width, value));
        address += 1;
    }
    Ok(values)
}

/// The arrays assigned anywhere in `m` (before resolving, every access to
/// an array is an element).
fn written_arrays(m: &Module) -> BTreeSet<Name> {
    let mut written = BTreeSet::new();
    for item in &m.items {
        let mut targets = Vec::new();
        match item {
            Item::Assign { lhs, .. } => targets.push(lhs),
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => {
                body.lvalues(&mut targets)
            }
            Item::Instance(_) => {}
        }
        while let Some(l) = targets.pop() {
            match &l.kind {
                LvalueKind::Elem(e) => {
                    written.insert(e.array.clone());
                }
                LvalueKind::Concat(ls) => targets.extend(ls),
                LvalueKind::Var(_) | LvalueKind::Select(..) => {}
            }
        }
    }
    written
}

/// Applies the file `load` to its array of `m`: initial values of the
/// elements if the array is among the `written` ones, else its contents
/// as a ROM.
fn apply_load(m: &mut Module, load: Load, written: &BTreeSet<Name>) -> Result<()> {
    let a = m
        .arrays
        .iter_mut()
        .find(|a| a.name == load.array)
        .expect("a declared array");
    if written.contains(&a.name) {
        for (k, v) in load.values.into_iter().enumerate() {
            let Some(value) = v else { continue };
            let target = SignalId::element(a.name.clone(), k);
            if let Some(i) = m.inits.iter().find(|i| i.id == target) {
                return Err(Error::new(
                    i.span,
                    format!("`{target}` already has an initial value at {}", load.span),
                ));
            }
            let value = Expr {
                kind: ExprKind::Lit { ty: a.ty, value },
                span: load.span,
            };
            m.inits.push(Init::new(target, value, load.span));
        }
        return Ok(());
    }
    if let Some(k) = load.values.iter().position(Option::is_none) {
        return Err(Error::new(
            load.span,
            format!(
                "{}: no value for `{}` (every element of an array that is never written \
                 needs one: it is a ROM)",
                load.path.display(),
                SignalId::element(a.name.clone(), k)
            ),
        ));
    }
    a.rom = Some(Rom {
        binary: load.binary,
        file: load.file,
        values: load.values.into_iter().flatten().collect(),
        span: load.span,
    });
    Ok(())
}

/// Checks the initial values of elements of arrays in `m`: inside the
/// array, and not of a ROM.
fn element_inits(m: &Module) -> Result<()> {
    for i in &m.inits {
        let SignalId {
            base: name,
            index: Some(k),
        } = &i.id
        else {
            continue;
        };
        let k = *k;
        let a = m.arrays.iter().find(|a| a.name == *name).expect("an array");
        if k >= a.size {
            return Err(Error::new(
                i.span,
                format!(
                    "`{}` out of range: `{name}` has addresses 0 to {}",
                    i.target,
                    a.size - 1
                ),
            ));
        }
        if let Some(r) = &a.rom {
            return Err(Error::new(
                i.span,
                format!(
                    "`{name}` is loaded at {}, and nothing writes it: its elements are \
                     constants, without initial values of their own",
                    r.span
                ),
            ));
        }
    }
    Ok(())
}

/// Resolves the elements of arrays in `m` once all of its items are known
/// (an array is a ROM if `initial` loads it and nothing writes it, possibly
/// after its uses): a ROM's elements are its contents, every one given by
/// the file; the element `m[3]` of any other array is the signal `m[3]`,
/// and `m[i]` with a variable index any of its element signals. A file
/// loaded into a written array gives initial values to its elements.
pub(super) fn resolve(m: &mut Module, loads: Vec<Load>) -> Result<()> {
    let written = written_arrays(m);
    for load in loads {
        apply_load(m, load, &written)?;
    }
    element_inits(m)?;
    let arrays: BTreeMap<Name, Array> = m
        .arrays
        .iter()
        .map(|a| (a.name.clone(), a.clone()))
        .collect();
    m.decls
        .retain(|d| match arrays.get(&d.id.base) {
            Some(a) => a.rom.is_none(),
            None => true,
        });
    let r = Resolver { arrays: &arrays };
    for item in &mut m.items {
        match item {
            Item::Assign { lhs, rhs, .. } => {
                r.lvalue(lhs)?;
                r.expr(rhs)?;
            }
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => r.stmt(body)?,
            Item::Instance(i) => {
                for c in &mut i.conns {
                    if let Some(e) = &mut c.actual {
                        r.expr(e)?;
                    }
                }
            }
        }
    }
    Ok(())
}

struct Resolver<'a> {
    arrays: &'a BTreeMap<Name, Array>,
}

impl Resolver<'_> {
    fn stmt(&self, s: &mut Stmt) -> Result<()> {
        match &mut s.kind {
            StmtKind::Block(stmts) => stmts.iter_mut().try_for_each(|s| self.stmt(s)),
            StmtKind::If(c, a, b) => {
                self.expr(c)?;
                self.stmt(a)?;
                self.stmt(b)
            }
            StmtKind::Case {
                sel,
                items,
                default,
                ..
            } => {
                self.expr(sel)?;
                for item in items {
                    item.labels.iter_mut().try_for_each(|l| self.expr(l))?;
                    self.stmt(&mut item.body)?;
                }
                match default {
                    Some(d) => self.stmt(d),
                    None => Ok(()),
                }
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                self.lvalue(lhs)?;
                self.expr(rhs)
            }
            StmtKind::Unknown { lhs, .. } => self.lvalue(lhs),
        }
    }

    fn select(&self, sel: &mut Select) -> Result<()> {
        match sel {
            Select::Bit(Index::Expr(e))
            | Select::Up(Index::Expr(e), _)
            | Select::Down(Index::Expr(e), _) => self.expr(e),
            _ => Ok(()),
        }
    }

    /// Resolves the element `e` at `span`: the signal of a constant
    /// index of an array that is no ROM, else `None` with the elements
    /// filled in.
    fn elem(&self, e: &mut Elem, span: Span, written: bool) -> Result<Option<Name>> {
        let a = &self.arrays[&e.array];
        if let Index::Expr(i) = &mut e.index {
            self.expr(i)?;
        }
        if let Some(sel) = &mut e.sel {
            self.select(sel)?;
        }
        if let Index::Const(k) = e.index
            && k >= a.size
        {
            return Err(Error::new(
                span,
                format!(
                    "`{e}` out of range: `{}` has addresses 0 to {}",
                    a.name,
                    a.size - 1
                ),
            ));
        }
        match (&a.rom, &e.index) {
            (Some(_), _) if written => unreachable!("a ROM is not written"),
            (Some(r), _) => {
                e.elems = Elems::Consts(a.ty, r.values.clone());
                Ok(None)
            }
            (None, Index::Const(k)) => Ok(Some(SignalId::element(a.name.clone(), *k).name())),
            (None, Index::Expr(_)) => {
                let elements = (0..a.size).map(|k| SignalId::element(a.name.clone(), k).name());
                e.elems = Elems::Signals(elements.collect());
                Ok(None)
            }
        }
    }

    fn lvalue(&self, l: &mut Lvalue) -> Result<()> {
        match &mut l.kind {
            LvalueKind::Var(_) => Ok(()),
            LvalueKind::Select(_, sel) => self.select(sel),
            LvalueKind::Concat(ls) => ls.iter_mut().try_for_each(|l| self.lvalue(l)),
            LvalueKind::Elem(e) => {
                if let Some(y) = self.elem(e, l.span, true)? {
                    l.kind = match e.sel.take() {
                        None => LvalueKind::Var(y),
                        Some(sel) => LvalueKind::Select(y, sel),
                    };
                }
                Ok(())
            }
        }
    }

    fn expr(&self, e: &mut Expr) -> Result<()> {
        match &mut e.kind {
            ExprKind::Lit { .. }
            | ExprKind::Const { .. }
            | ExprKind::Var(_)
            | ExprKind::Fill(_) => Ok(()),
            ExprKind::Unary(_, a) | ExprKind::Cast(_, a) | ExprKind::Resize(_, a) => self.expr(a),
            ExprKind::Binary(_, a, b) => {
                self.expr(a)?;
                self.expr(b)
            }
            ExprKind::Cond(c, a, b) => {
                self.expr(c)?;
                self.expr(a)?;
                self.expr(b)
            }
            ExprKind::Concat(es) | ExprKind::Repl(_, es) => {
                es.iter_mut().try_for_each(|e| self.expr(e))
            }
            ExprKind::Select(_, sel) => self.select(sel),
            ExprKind::Elem(el) => {
                if let Some(y) = self.elem(el, e.span, false)? {
                    e.kind = match el.sel.take() {
                        None => ExprKind::Var(y),
                        Some(sel) => ExprKind::Select(y, sel),
                    };
                }
                Ok(())
            }
        }
    }
}

impl Cx<'_> {
    /// Whether `x` is an array declared so far.
    pub(super) fn is_array(&self, x: &str) -> bool {
        self.arrays.borrow().iter().any(|a| a.name == x)
    }

    /// The message for an array `m` used other than by one element.
    pub(super) fn whole_array(&self, m: &str) -> String {
        format!(
            "`{m}` is an array: only its elements can be used (`{m}[i]`, `{m}[i][3:0]`); \
             slices and whole arrays are not supported"
        )
    }

    /// The select `[i]`, `[i][j]` or `[i][…]` after the name of an array
    /// `m`: an element (resolved after the module, see [`resolve`]).
    pub(super) fn element(&self, m: Name, select: &sv::Select, span: Span) -> Result<Elem> {
        let (member, bits, part) = &select.nodes;
        self.absent(member, "member select")?;
        let (i, sel) = match (bits.nodes.0.as_slice(), part) {
            ([b], None) => (self.index(&b.nodes.1)?, None),
            ([b, c], None) => (
                self.index(&b.nodes.1)?,
                Some(Select::Bit(self.index(&c.nodes.1)?)),
            ),
            ([b], Some(p)) => (self.index(&b.nodes.1)?, Some(self.part(&p.nodes.1)?)),
            _ => return Err(Error::unsupported(span, self.whole_array(&m))),
        };
        Ok(Elem {
            array: m,
            index: i,
            sel,
            elems: Elems::Signals(Vec::new()),
        })
    }

    /// The number of elements of a declaration with unpacked dimensions
    /// `dims` (`None` if there are none): one dimension `[0:N-1]`,
    /// `[N-1:0]` or `[N]`.
    pub(super) fn array_size(
        &self,
        id: &sv::Identifier,
        dims: &[&sv::UnpackedDimension],
    ) -> Result<Option<usize>> {
        let d = match dims {
            [] => return Ok(None),
            [d] => *d,
            [_, d, ..] => {
                return self
                    .not_supported(*d, "arrays with several unpacked dimensions not supported");
            }
        };
        let size = match d {
            sv::UnpackedDimension::Range(r) => {
                let (a, _, b) = &r.nodes.0.nodes.1.nodes;
                let (a, b) = (self.const_int(a)?, self.const_int(b)?);
                if a.min(b) != 0 {
                    return self.not_supported(
                        d,
                        format!(
                            "array range `[{a}:{b}]` not supported: addresses must start at 0 \
                             (`[0:N-1]`, `[N-1:0]` or `[N]`)"
                        ),
                    );
                }
                a.max(b) + 1
            }
            sv::UnpackedDimension::Expression(e) => self.const_int(&e.nodes.0.nodes.1)?,
        };
        if size == 0 {
            return self.err(id, "an array needs at least one element");
        }
        Ok(Some(size))
    }
}

impl Cx<'_> {
    /// A call `$readmemb("file", m);` or `$readmemh(…)` in an `initial`
    /// (anything else is rejected with `only`): reads the file and records
    /// the words for the array `m`.
    pub(super) fn load(&self, c: &sv::SubroutineCallStatement, only: &str) -> Result<()> {
        let sv::SubroutineCallStatement::SubroutineCall(c) = c else {
            return self.not_supported(c, only);
        };
        let sv::SubroutineCall::SystemTfCall(call) = &c.0 else {
            return self.not_supported(&c.0, only);
        };
        let Some((name, args)) = system_call(call) else {
            return self.not_supported(&c.0, only);
        };
        let task = self.text(name);
        let binary = match task.as_str() {
            "$readmemb" => true,
            "$readmemh" => false,
            _ => return self.not_supported(&c.0, only),
        };
        let span = self.span(&c.0);
        let (file, m) = match args.as_slice() {
            [Some(f), Some(m)] => (f, m),
            [_, _, _, ..] => {
                return self.not_supported(
                    &c.0,
                    format!("`{task}` with start and end addresses not supported"),
                );
            }
            _ => return self.err(&c.0, format!("`{task}` takes a file name and an array")),
        };
        let file_text = self.text(file);
        let file_name = match file {
            sv::Expression::Primary(p)
                if matches!(
                    p.as_ref(),
                    sv::Primary::PrimaryLiteral(l)
                        if matches!(l.as_ref(), sv::PrimaryLiteral::StringLiteral(_))
                ) && !file_text.contains('\\') =>
            {
                file_text.trim_matches('"').to_string()
            }
            f => {
                return self
                    .not_supported(f, format!("the file of `{task}` must be a string literal"));
            }
        };
        let m_text = self.text(m);
        let arrays = self.arrays.borrow();
        let Some(array) = arrays.iter().find(|a| a.name == m_text) else {
            return self.err(m, format!("`{m_text}` is not an array declared before"));
        };
        if let Some(l) = self.loads.borrow().iter().find(|l| l.array == m_text) {
            return self.not_supported(&c.0, format!("`{m_text}` is already loaded at {}", l.span));
        }
        let path = self.dir.join(&file_name);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            Error::new(
                self.span(file),
                format!("cannot read `{}`: {e}", path.display()),
            )
        })?;
        let values = readmem(&text, binary, array)
            .map_err(|msg| Error::new(span, format!("{}: {msg}", path.display())))?;
        self.loads.borrow_mut().push(Load {
            array: m_text,
            binary,
            file: file_name,
            path,
            values,
            span,
        });
        Ok(())
    }
}
