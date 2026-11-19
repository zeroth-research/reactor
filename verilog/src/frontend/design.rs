//! Source files, the parsed design and positions in it.

use super::{Cx, ModuleBuilder, Result};
use crate::ast::{Error, ErrorKind, Module, Name, Override, Span, Warning};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use sv_parser as sv;
use sv_parser::{Iter, RefNode, SyntaxTree};

/// Parses `source` and elaborates its top module with the default
/// parameter values; `path` is used for messages of the preprocessor and
/// to find the files of `$readmemb`/`$readmemh` (relative to its
/// directory).
pub fn parse(source: &str, path: impl AsRef<Path>) -> Result<Module> {
    let d = parse_design(source, path)?;
    d.module(&d.top()?, &[])
}

/// Parses a file.
pub(crate) fn parse_design_file(path: impl AsRef<Path>) -> Result<Design> {
    parse_design_files(&[path])
}

/// Parses several files as one design, as if they were one file in the
/// order given (a macro defined in one is defined in the following ones,
/// as for Yosys and iverilog). Errors are located with [`Design::locate`];
/// those of this function already carry their file.
pub fn parse_design_files<P: AsRef<Path>>(paths: &[P]) -> Result<Design> {
    parse_sources(read_sources(paths)?)
}

/// The contents of the files, each with its path.
fn read_sources<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<(String, PathBuf)>> {
    let mut sources = Vec::new();
    for path in paths {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path)
            .map_err(|e| Error::unlocated(ErrorKind::User, e.to_string()).in_file(path))?;
        sources.push((source, path.to_path_buf()));
    }
    Ok(sources)
}

/// Whether the text the files preprocess to (as one design, see
/// [`parse_design_files`]) changes when the macro `name` is predefined (as
/// `1`) besides `SYNTHESIS`: whether the design tests it, or uses it.
/// A mention in a comment, or in a branch of a conditional that is not read
/// either way, changes nothing. Yosys defines `YOSYS` in every file it
/// reads, so it reads another text than we do exactly if this holds for
/// `YOSYS`. A design that fails to preprocess with `name` defined (but not
/// without it) depends on it too.
pub fn depends_on_macro<P: AsRef<Path>>(paths: &[P], name: &str) -> Result<bool> {
    let sources = read_sources(paths)?;
    let first = sources
        .first()
        .ok_or_else(|| Error::unlocated(ErrorKind::User, "no source file"))?
        .1
        .clone();
    let (text, _) = join_sources(sources);
    let includes: [PathBuf; 0] = [];
    let preprocess = |defines: &HashMap<String, Option<sv::Define>>| {
        sv::preprocess_str(&text, &first, defines, &includes, false, false, 0, 0)
            .map(|(t, _)| t.text().to_string())
    };
    let mut defines = predefined();
    let ours =
        preprocess(&defines).map_err(|e| Error::unlocated(ErrorKind::User, e.to_string()))?;
    defines.insert(name.to_string(), Some(define(name, "1")));
    Ok(preprocess(&defines).map_or(true, |theirs| theirs != ours))
}

/// A macro `name` without arguments, defined as `value`.
pub(super) fn define(name: &str, value: &str) -> sv::Define {
    sv::Define::new(
        name.to_string(),
        vec![],
        Some(sv::DefineText::new(value.to_string(), None)),
    )
}

/// The predefined macros: `SYNTHESIS` (as `1`), as by Yosys, since the
/// translation follows what synthesis reads.
pub(super) fn predefined() -> HashMap<String, Option<sv::Define>> {
    HashMap::from([("SYNTHESIS".to_string(), Some(define("SYNTHESIS", "1")))])
}

/// The sources as one text, each followed by a line break if it does not
/// end with one, and the files with the line at which each starts.
fn join_sources(sources: Vec<(String, PathBuf)>) -> (String, Vec<SourceFile>) {
    let mut text = String::new();
    let mut files = Vec::new();
    for (source, path) in sources {
        let line = text.matches('\n').count() + 1;
        let dir = path.parent().unwrap_or(Path::new("")).to_path_buf();
        files.push(SourceFile { path, dir, line });
        text.push_str(&source);
        if !source.ends_with('\n') {
            text.push('\n');
        }
    }
    (text, files)
}

/// Parses `source`; `path` is used for messages of the preprocessor and to
/// find the files of `$readmemb`/`$readmemh` (relative to its directory).
pub fn parse_design(source: &str, path: impl AsRef<Path>) -> Result<Design> {
    parse_sources(vec![(source.to_string(), path.as_ref().to_path_buf())])
}

/// Parses the sources, each with its path, as one text.
fn parse_sources(sources: Vec<(String, PathBuf)>) -> Result<Design> {
    let Some((_, first)) = sources.first() else {
        return Err(Error::unlocated(ErrorKind::User, "no source file"));
    };
    let first = first.clone();
    let defines = predefined();
    let includes: [PathBuf; 0] = [];
    let (text, files) = join_sources(sources);
    let mut lines = vec![0];
    lines.extend(text.match_indices('\n').map(|(i, _)| i + 1));
    let several = files.len() > 1;
    let located = |e: Error| -> Error { if several { locate_in(&files, e) } else { e } };
    let (tree, _) = sv::parse_sv_str(&text, &first, &defines, &includes, false, false).map_err(
        |e| match e {
            sv::Error::Parse(Some((_, offset))) => {
                located(Error::new(at(&lines, offset), "syntax error"))
            }
            e => Error::unlocated(ErrorKind::User, e.to_string()),
        },
    )?;
    let mut d = Design {
        tree,
        lines,
        files,
        modules: Vec::new(),
        by_name: HashMap::new(),
    };
    let mut seen: BTreeMap<Name, Span> = BTreeMap::new();
    let cx = d.cx(&[], 0);
    let mut modules = Vec::new();
    for (k, m) in d.decls().map_err(|e| d.located(e))? {
        let (attrs, _, _, id, ..) = &m.nodes.0.nodes;
        let blackbox = cx.module_attributes(attrs).map_err(|e| d.located(e))?;
        let name = cx.identifier(&id.nodes.0).map_err(|e| d.located(e))?;
        if let Some(prev) = seen.insert(name.clone(), cx.span(id)) {
            let e = Error::new(
                cx.span(id),
                format!("module `{name}` already declared at {prev}"),
            );
            return Err(d.located(e));
        }
        // the name as `find` compares it
        modules.push((cx.text(id), k, blackbox));
    }
    d.by_name = (modules.iter().enumerate())
        .map(|(i, (name, ..))| (name.clone(), i))
        .collect();
    d.modules = modules;
    Ok(d)
}

/// `e` with its position in its own file, and the file (see
/// [`Design::locate`]). The positions its message mentions (`at L:C`, see
/// [`mentions_in`]) are given in their files too.
fn locate_in(files: &[SourceFile], e: Error) -> Error {
    let Some(span) = e.span else {
        return e;
    };
    let (f, span) = file_of(files, span);
    let msg = mentions_in(files, &f.path, &e.msg);
    Error {
        span: Some(span),
        msg,
        ..e
    }
    .in_file(&f.path)
}

/// The file of a position of the design, with the position in that file.
fn file_of(files: &[SourceFile], span: Span) -> (&SourceFile, Span) {
    let k = files.partition_point(|f| f.line <= span.line).max(1) - 1;
    let f = &files[k];
    let span = Span {
        line: span.line - f.line + 1,
        col: span.col,
    };
    (f, span)
}

/// `msg` with every position of the design it mentions as `at L:C` (the
/// form messages use to point to a second place, such as an earlier
/// driver) given in its own file: as `L:C` in that file if it is `file`
/// (the file of the error), else as `path:L:C`. A mention is the word `at`
/// followed by one space and `L:C` (identifiers in messages cannot
/// contain spaces or colons).
fn mentions_in(files: &[SourceFile], file: &Path, msg: &str) -> String {
    let digits = |s: &str| s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let mut out = String::new();
    let mut rest = msg;
    while let Some(k) = rest.find("at ") {
        out.push_str(&rest[..k]);
        // `at` is a word: no letter, digit or `_` right before it
        let word = !out.ends_with(|c: char| c.is_alphanumeric() || c == '_');
        out.push_str("at ");
        rest = &rest[k + 3..];
        let n = digits(rest);
        let m = rest.get(n + 1..).map_or(0, digits);
        if !word || n == 0 || m == 0 || !rest[n..].starts_with(':') {
            continue;
        }
        let (Ok(line), Ok(col)) = (rest[..n].parse(), rest[n + 1..n + 1 + m].parse()) else {
            continue;
        };
        let (f, span) = file_of(files, Span { line, col });
        if f.path != file {
            out.push_str(&format!("{}:", f.path.display()));
        }
        out.push_str(&span.to_string());
        rest = &rest[n + 1 + m..];
    }
    out.push_str(rest);
    out
}

/// The position of a byte offset, with `lines` the offsets at which lines
/// start.
pub(super) fn at(lines: &[usize], offset: usize) -> Span {
    let line = lines.partition_point(|&start| start <= offset);
    Span {
        line,
        col: offset - lines[line - 1] + 1,
    }
}

/// Parsed source files. Their modules are elaborated on demand: once for
/// each set of parameter values. Positions ([`Span`]) count the lines of
/// all files in order, as if they were one text; [`Design::locate`] gives
/// the file and the position in it.
pub struct Design {
    tree: SyntaxTree,
    lines: Vec<usize>,
    files: Vec<SourceFile>,
    /// The modules in source order: name, index among the descriptions of
    /// the source text, and whether it is marked `(* blackbox *)`. The
    /// declarations are checked when the design is parsed, so finding one
    /// later cannot fail.
    modules: Vec<(Name, usize, bool)>,
    /// The index of each module in `modules`, by name.
    by_name: HashMap<Name, usize>,
}

/// A source file of a [`Design`].
struct SourceFile {
    path: PathBuf,
    /// The directory of the file.
    dir: PathBuf,
    /// The line at which the file starts in the design.
    line: usize,
}

impl Design {
    /// The file of a position of the design, with the position in that
    /// file.
    pub fn locate(&self, span: Span) -> (PathBuf, Span) {
        let e = locate_in(&self.files, Error::new(span, ""));
        let path = e.file.unwrap_or_else(|| self.files[0].path.clone());
        (path, e.span.unwrap_or(span))
    }

    /// The error with its position in its file, and the file, if the
    /// design has several files (an error with a file is left alone).
    pub fn located(&self, e: Error) -> Error {
        if self.files.len() > 1 && e.file.is_none() {
            locate_in(&self.files, e)
        } else {
            e
        }
    }

    /// [`Design::located`] for a warning.
    pub fn located_warning(&self, w: Warning) -> Warning {
        let e = Error {
            kind: ErrorKind::User,
            span: Some(w.span),
            msg: w.msg,
            file: w.file,
        };
        let e = self.located(e);
        Warning {
            span: e.span.unwrap_or(w.span),
            msg: e.msg,
            file: e.file,
        }
    }

    /// The paths of the files, in order.
    pub fn paths(&self) -> Vec<&Path> {
        self.files.iter().map(|f| f.path.as_path()).collect()
    }

    /// A context for elaborating a module at `line`, whose `$readmem`
    /// files are found relative to the directory of its file.
    fn cx<'t>(&'t self, overrides: &'t [Override], line: usize) -> Cx<'t> {
        let k = self.files.partition_point(|f| f.line <= line).max(1) - 1;
        Cx {
            tree: &self.tree,
            lines: &self.lines,
            overrides,
            dir: &self.files[k].dir,
            state: ModuleBuilder::default(),
        }
    }

    /// The descriptions of the source text.
    fn descriptions(&self) -> Result<&[sv::Description]> {
        let st = (&self.tree)
            .into_iter()
            .find_map(|n| match n {
                RefNode::SourceText(x) => Some(x),
                _ => None,
            })
            .ok_or_else(|| Error::unlocated(ErrorKind::User, "empty source"))?;
        let cx = self.cx(&[], 0);
        let (_, timeunits, descriptions) = &st.nodes;
        cx.absent(timeunits, "timeunits declaration")?;
        Ok(descriptions)
    }

    /// The module declarations, in source order, each with its index among
    /// the descriptions.
    fn decls(&self) -> Result<Vec<(usize, &sv::ModuleDeclarationAnsi)>> {
        let cx = self.cx(&[], 0);
        let mut modules = Vec::new();
        for (k, d) in self.descriptions()?.iter().enumerate() {
            match d {
                sv::Description::ModuleDeclaration(m) => match m.as_ref() {
                    sv::ModuleDeclaration::Ansi(m) => modules.push((k, m.as_ref())),
                    sv::ModuleDeclaration::Nonansi(_) => {
                        return cx.not_supported(m.as_ref(), "non-ANSI port list not supported");
                    }
                    m => return cx.unsupported(m),
                },
                d => return cx.unsupported(d),
            }
        }
        if modules.is_empty() {
            return Err(Error::unlocated(ErrorKind::User, "no module"));
        }
        Ok(modules)
    }

    /// The declaration of the module `name`, if there is one.
    fn find(&self, name: &str) -> Option<&sv::ModuleDeclarationAnsi> {
        let &(_, k, _) = &self.modules[*self.by_name.get(name)?];
        let descriptions = self.descriptions().expect("checked when parsed");
        match &descriptions[k] {
            sv::Description::ModuleDeclaration(m) => match m.as_ref() {
                sv::ModuleDeclaration::Ansi(m) => Some(m.as_ref()),
                _ => unreachable!("checked when parsed"),
            },
            _ => unreachable!("checked when parsed"),
        }
    }

    /// The names of the modules, in source order.
    pub fn names(&self) -> Vec<Name> {
        self.modules.iter().map(|(name, ..)| name.clone()).collect()
    }

    /// Whether the file declares the module `name`.
    pub fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Whether the module `name` is marked `(* blackbox *)`.
    pub fn is_blackbox(&self, name: &str) -> bool {
        (self.by_name.get(name)).is_some_and(|&i| self.modules[i].2)
    }

    /// The top module: the one no module instantiates (blackboxes aside).
    pub fn top(&self) -> Result<Name> {
        let cx = self.cx(&[], 0);
        let mut instantiated = BTreeSet::new();
        for (_, m) in self.decls()? {
            for n in Iter::new(m.into()) {
                if let RefNode::ModuleInstantiation(i) = n {
                    instantiated.insert(cx.text(&i.nodes.0));
                }
            }
        }
        let tops: Vec<Name> = self
            .names()
            .into_iter()
            .filter(|n| !instantiated.contains(n) && !self.is_blackbox(n))
            .collect();
        match tops.as_slice() {
            [top] => Ok(top.clone()),
            [] => Err(Error::unlocated(
                ErrorKind::Unsupported,
                "no top module: every module is instantiated (recursive instantiation \
                 is not supported)",
            )),
            _ => Err(Error::unlocated(
                ErrorKind::User,
                format!(
                    "several top modules ({}): choose one",
                    tops.iter()
                        .map(|t| format!("`{t}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )),
        }
    }

    /// Elaborates the module `name` with the parameter values `overrides`
    /// (the others take their defaults). The instances in it are not
    /// elaborated (see [`elab`](crate::elab)).
    pub fn module(&self, name: &str, overrides: &[Override]) -> Result<Module> {
        let m = self
            .find(name)
            .ok_or_else(|| Error::unlocated(ErrorKind::User, format!("no module `{name}`")))?;
        let line = self.cx(&[], 0).span(m).line;
        self.cx(overrides, line).module(m)
    }
}
