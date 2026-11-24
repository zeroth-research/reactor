//! The builder: combinators that put tapes together, in a mutable scratch
//! form, and [`Builder::freeze`], which turns a draft into a [`Tape`] in
//! normal form.
//!
//! # Scratch form
//!
//! The builder keeps every draft in a scratch form, close to a tape but
//! cheaper to extend:
//!
//! - Like a tape, a draft has one arm per summand of its domain, each with a
//!   circuit and a routing DAG. Values are numbered per arm in the same way.
//! - Deterministic nodes are hash-consed per arm, and so are routes: a split
//!   whose two children are the same route is that route.
//! - Draws (non-deterministic nodes) are never shared: each draw made by a
//!   combinator is a new node. Draws are not bound at any route, and have no
//!   level. [`Builder::freeze`] places the binders and computes the levels
//!   (see the `canon` module).
//!
//! Drafts are immutable: every combinator creates a new draft. Drafts share
//! what they can. A draft lists its arms by index into the builder's arena,
//! with an offset for the outputs of their ends, so `sum` copies nothing.
//! Composition creates new arms, which inline the arms of the second draft
//! at the ends of the first.

use super::canon::{self, LocalArm};
use super::{NodeKind, Operation, Route, Span, Tape, Value};
use base::boundary::{Boundary, Mono, Poly};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};

/// A handle to a draft: a tape under construction in a [`Builder`].
///
/// A draft is only meaningful for the builder that created it, which it
/// records: using it with another builder panics. Its boundary is available
/// through [`Builder::draft`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Draft {
    builder: u64,
    index: u32,
}

/// The id of the next builder, so that each builder has its own.
static NEXT_BUILDER: AtomicU64 = AtomicU64::new(0);

/// What a scratch node computes.
#[derive(Clone, Debug)]
pub(super) enum SKind<O> {
    /// A deterministic operation: hash-consed.
    Det(O),
    /// A non-deterministic operation: a new draw, never shared.
    Draw(O),
}

/// A node of a scratch arm. Its outputs follow the values defined before it,
/// as in a tape.
#[derive(Clone, Debug)]
pub(super) struct SNode<O> {
    pub(super) kind: SKind<O>,
    /// A span of [`Arena::args`].
    pub(super) args: Span,
    pub(super) n_outs: u32,
}

/// A route of a scratch arm: like a [`Route`], without binders. `hi` and
/// `lo` are indices into [`Arena::routes`], of earlier routes of the same
/// arm.
#[derive(Clone, Copy, Debug)]
pub(super) enum SRoute {
    Split { cond: Value, hi: u32, lo: u32 },
    End { output: u32, outs: Span },
}

/// A scratch arm: contiguous runs of the arena's nodes and routes, and its
/// root route.
#[derive(Clone, Copy, Debug)]
pub(super) struct SArm {
    pub(super) n_inputs: u32,
    pub(super) nodes: Span,
    pub(super) routes: Span,
    /// An index into [`Arena::routes`], within `routes`.
    pub(super) root: u32,
}

/// The arrays shared by all scratch arms of a builder.
#[derive(Clone, Debug)]
pub(super) struct Arena<O> {
    pub(super) nodes: Vec<SNode<O>>,
    pub(super) args: Vec<Value>,
    pub(super) routes: Vec<SRoute>,
    pub(super) outs: Vec<Value>,
    pub(super) arms: Vec<SArm>,
}

/// One arm of a draft: a scratch arm, whose ends' outputs are shifted by
/// `shift` in the draft's codomain.
#[derive(Clone, Copy, Debug)]
struct DraftArm {
    arm: u32,
    shift: u32,
}

#[derive(Clone, Debug)]
struct DraftData<S> {
    dom: Poly<S>,
    cod: Poly<S>,
    /// A span of [`Builder::draft_arms`].
    arms: Span,
}

/// Puts tapes together.
///
/// A builder owns drafts, tapes under construction, and offers the
/// combinators that create them: the generators ([`lift`](Self::lift),
/// [`case`](Self::case)), the structure of strings
/// ([`copy`](Self::copy), [`discard`](Self::discard), [`cross`](Self::cross),
/// and [`wiring`](Self::wiring) for any mix of them), the structure of sums
/// ([`id`](Self::id), [`swap`](Self::swap), [`merge`](Self::merge),
/// [`initial`](Self::initial)), and the operations on drafts
/// ([`sum`](Self::sum), [`whisker`](Self::whisker),
/// [`compose`](Self::compose)). Any draft can be frozen at any time into a
/// [`Tape`] in normal form with [`freeze`](Self::freeze), and a tape can be
/// brought back as a draft with [`import`](Self::import).
///
/// Combinators check the boundaries of their arguments and fail with a
/// message if they do not fit.
#[derive(Debug)]
pub struct Builder<S, O> {
    /// This builder's id, recorded in its drafts.
    id: u64,
    boolean: S,
    arena: Arena<O>,
    drafts: Vec<DraftData<S>>,
    draft_arms: Vec<DraftArm>,
}

/// The boundary of a draft, borrowed from its builder.
pub struct DraftRef<'a, S> {
    dom: &'a Poly<S>,
    cod: &'a Poly<S>,
}

impl<S: Clone> Boundary<S> for DraftRef<'_, S> {
    fn dom(&self) -> Poly<S> {
        self.dom.clone()
    }

    fn cod(&self) -> Poly<S> {
        self.cod.clone()
    }
}

/// Appends `items` to `array`, and returns their span. An empty list is
/// always [`Span::EMPTY`].
pub(super) fn push_span<T>(array: &mut Vec<T>, items: Vec<T>) -> Span {
    if items.is_empty() {
        return Span::EMPTY;
    }
    let span = Span::new(array.len() as u32, items.len() as u32);
    array.extend(items);
    span
}

fn len<S>(m: &Mono<S>) -> u32 {
    m.iter().count() as u32
}

fn summands<S>(p: &Poly<S>) -> u32 {
    p.iter().count() as u32
}

/// The key under which a scratch route is hash-consed.
#[derive(PartialEq, Eq, Hash)]
enum RouteKey {
    Split(Value, u32, u32),
    End(u32, Vec<Value>),
}

/// Writes one new scratch arm into the arena, hash-consing its
/// deterministic nodes and its routes.
struct ArmBuilder<O> {
    n_inputs: u32,
    next_value: u32,
    nodes_start: u32,
    routes_start: u32,
    det: HashMap<(O, Vec<Value>), Value>,
    routes: HashMap<RouteKey, u32>,
}

impl<O: Clone + Eq + Hash> ArmBuilder<O> {
    fn new(arena: &Arena<O>, n_inputs: u32) -> Self {
        Self {
            n_inputs,
            next_value: n_inputs,
            nodes_start: arena.nodes.len() as u32,
            routes_start: arena.routes.len() as u32,
            det: HashMap::new(),
            routes: HashMap::new(),
        }
    }

    /// The input values of the arm.
    fn inputs(&self) -> Vec<Value> {
        (0..self.n_inputs).map(Value).collect()
    }

    /// Adds a node, unless it is deterministic and already there, and
    /// returns its first output.
    fn node(
        &mut self,
        arena: &mut Arena<O>,
        kind: SKind<O>,
        args: Vec<Value>,
        n_outs: u32,
    ) -> Value {
        if let SKind::Det(op) = &kind {
            let key = (op.clone(), args.clone());
            if let Some(&first) = self.det.get(&key) {
                return first;
            }
            self.det.insert(key, Value(self.next_value));
        }
        let first = Value(self.next_value);
        let args = push_span(&mut arena.args, args);
        arena.nodes.push(SNode { kind, args, n_outs });
        self.next_value += n_outs;
        first
    }

    fn route(
        &mut self,
        arena: &mut Arena<O>,
        key: RouteKey,
        route: impl FnOnce(&mut Arena<O>) -> SRoute,
    ) -> u32 {
        if let Some(&r) = self.routes.get(&key) {
            return r;
        }
        let route = route(arena);
        let r = arena.routes.len() as u32;
        arena.routes.push(route);
        self.routes.insert(key, r);
        r
    }

    /// The end of a path in `output`, with the values `outs`.
    fn end(&mut self, arena: &mut Arena<O>, output: u32, outs: Vec<Value>) -> u32 {
        let key = RouteKey::End(output, outs.clone());
        self.route(arena, key, |arena| SRoute::End {
            output,
            outs: push_span(&mut arena.outs, outs),
        })
    }

    /// A split on `cond`, or `hi` itself if both children are the same.
    fn split(&mut self, arena: &mut Arena<O>, cond: Value, hi: u32, lo: u32) -> u32 {
        if hi == lo {
            return hi;
        }
        self.route(arena, RouteKey::Split(cond, hi, lo), |_| SRoute::Split {
            cond,
            hi,
            lo,
        })
    }

    /// Stores the arm, with root `root`, and returns its index.
    fn finish(self, arena: &mut Arena<O>, root: u32) -> u32 {
        let nodes = Span::new(
            self.nodes_start,
            arena.nodes.len() as u32 - self.nodes_start,
        );
        let routes = Span::new(
            self.routes_start,
            arena.routes.len() as u32 - self.routes_start,
        );
        arena.arms.push(SArm {
            n_inputs: self.n_inputs,
            nodes,
            routes,
            root,
        });
        arena.arms.len() as u32 - 1
    }
}

/// What to do at an end of a copied arm: given its output and its (copied)
/// values, return the route that replaces it.
type OnEnd<'a, O> = dyn FnMut(&mut Arena<O>, &mut ArmBuilder<O>, u32, Vec<Value>) -> u32 + 'a;

/// Copies the scratch arm `src` into `ab`, with its inputs replaced by
/// `inputs`, and every end replaced by what `on_end` returns. Returns the
/// copy of the root. Deterministic nodes are hash-consed with those already
/// in `ab`; draws are new.
fn copy_arm<O: Clone + Eq + Hash>(
    arena: &mut Arena<O>,
    ab: &mut ArmBuilder<O>,
    src: u32,
    inputs: &[Value],
    on_end: &mut OnEnd<'_, O>,
) -> u32 {
    let arm = arena.arms[src as usize];
    debug_assert_eq!(inputs.len() as u32, arm.n_inputs);
    // `map[v]`: the copy of value `v` of `src`.
    let mut map: Vec<Value> = inputs.to_vec();
    for n in arm.nodes.range() {
        let node = arena.nodes[n].clone();
        let args = arena.args[node.args.range()]
            .iter()
            .map(|v| map[v.index()])
            .collect();
        let first = ab.node(arena, node.kind, args, node.n_outs);
        map.extend((0..node.n_outs).map(|k| Value(first.0 + k)));
    }
    // `copies[r - start]`: the copy of route `r` of `src`. Children come
    // before their parents, so one pass in order suffices.
    let start = arm.routes.start;
    let mut copies: Vec<u32> = Vec::with_capacity(arm.routes.len as usize);
    for r in arm.routes.range() {
        let copy = match arena.routes[r] {
            SRoute::Split { cond, hi, lo } => {
                let (hi, lo) = (copies[(hi - start) as usize], copies[(lo - start) as usize]);
                ab.split(arena, map[cond.index()], hi, lo)
            }
            SRoute::End { output, outs } => {
                let outs = arena.outs[outs.range()]
                    .iter()
                    .map(|v| map[v.index()])
                    .collect();
                on_end(arena, ab, output, outs)
            }
        };
        copies.push(copy);
    }
    copies[(arm.root - start) as usize]
}

impl<S, O> Builder<S, O>
where
    S: Clone + Eq + Hash + Debug,
    O: Operation<S> + Clone + Eq + Hash + Debug,
{
    /// An empty builder, for tapes whose tests read the sort `boolean`.
    pub fn new(boolean: S) -> Self {
        Self {
            id: NEXT_BUILDER.fetch_add(1, Ordering::Relaxed),
            boolean,
            arena: Arena {
                nodes: Vec::new(),
                args: Vec::new(),
                routes: Vec::new(),
                outs: Vec::new(),
                arms: Vec::new(),
            },
            drafts: Vec::new(),
            draft_arms: Vec::new(),
        }
    }

    /// The boolean sort.
    pub fn boolean(&self) -> &S {
        &self.boolean
    }

    /// The boundary of `draft`.
    ///
    /// # Panics
    /// If `draft` was not created by this builder.
    pub fn draft(&self, draft: Draft) -> DraftRef<'_, S> {
        let data = self.data(draft);
        DraftRef {
            dom: &data.dom,
            cod: &data.cod,
        }
    }

    fn data(&self, draft: Draft) -> &DraftData<S> {
        assert_eq!(
            draft.builder, self.id,
            "the draft was created by another builder"
        );
        &self.drafts[draft.index as usize]
    }

    fn arms_of(&self, draft: Draft) -> Vec<DraftArm> {
        self.draft_arms[self.data(draft).arms.range()].to_vec()
    }

    fn add(&mut self, dom: Poly<S>, cod: Poly<S>, arms: Vec<DraftArm>) -> Draft {
        let arms = push_span(&mut self.draft_arms, arms);
        self.drafts.push(DraftData { dom, cod, arms });
        Draft {
            builder: self.id,
            index: self.drafts.len() as u32 - 1,
        }
    }

    /// A draft `inputs → cod` with a single arm, whose root is
    /// `routes(arena, arm builder)`.
    fn single(
        &mut self,
        inputs: Mono<S>,
        cod: Poly<S>,
        routes: impl FnOnce(&mut Arena<O>, &mut ArmBuilder<O>) -> u32,
    ) -> Draft {
        let mut ab = ArmBuilder::new(&self.arena, len(&inputs));
        let root = routes(&mut self.arena, &mut ab);
        let arm = ab.finish(&mut self.arena, root);
        self.add(Poly::from(inputs), cod, vec![DraftArm { arm, shift: 0 }])
    }

    /// One arm per summand of `dom`: the `k`-th sends its inputs unchanged
    /// to output `target(k)`.
    fn routing(&mut self, dom: Poly<S>, cod: Poly<S>, target: impl Fn(u32) -> u32) -> Draft {
        let mut arms = Vec::new();
        for (k, m) in dom.iter().enumerate() {
            let mut ab = ArmBuilder::new(&self.arena, len(m));
            let root = ab.end(&mut self.arena, target(k as u32), ab.inputs());
            arms.push(DraftArm {
                arm: ab.finish(&mut self.arena, root),
                shift: 0,
            });
        }
        self.add(dom, cod, arms)
    }

    /// The draft `M → N` of one operation `op : M → N`: a single node, as
    /// a draw if `op` is not deterministic.
    ///
    /// Fails if the domain or the codomain of `op` is not a single product.
    pub fn lift(&mut self, op: O) -> Result<Draft, String> {
        let single = |p: Poly<S>, what: &str| {
            let mut summands = p.into_iter();
            match (summands.next(), summands.next()) {
                (Some(m), None) => Ok(m),
                _ => Err(format!(
                    "lift: the {what} of {op:?} is not a single product"
                )),
            }
        };
        let dom = single(op.dom(), "domain")?;
        let cod = single(op.cod(), "codomain")?;
        let n_outs = len(&cod);
        let kind = if op.is_deterministic() {
            SKind::Det(op)
        } else {
            SKind::Draw(op)
        };
        Ok(self.single(dom, Poly::from(cod), |arena, ab| {
            let first = ab.node(arena, kind, ab.inputs(), n_outs);
            let outs = (0..n_outs).map(|k| Value(first.0 + k)).collect();
            ab.end(arena, 0, outs)
        }))
    }

    /// The draft that only moves strings, the general form of
    /// [`copy`](Self::copy), [`discard`](Self::discard) and
    /// [`cross`](Self::cross).
    ///
    /// `outputs` lists, for each output string in order, the position of the
    /// input string it carries. An input listed several times is copied, and
    /// one never listed is discarded. The codomain is the word of the listed
    /// sorts. For instance, with inputs `x : R` and `d : D`:
    ///
    /// | `outputs` | draft | |
    /// |---|---|---|
    /// | `[0, 1]` | `R ⊗ D → R ⊗ D` | the identity |
    /// | `[1, 0]` | `R ⊗ D → D ⊗ R` | `cross(R, D)` |
    /// | `[0, 1, 0]` | `R ⊗ D → R ⊗ D ⊗ R` | a copy of `x` after both |
    /// | `[1]` | `R ⊗ D → D` | `x` is discarded |
    /// | `[]` | `R ⊗ D → I` | `discard(R ⊗ D)` |
    ///
    /// Fails if a listed position is not an input position.
    pub fn wiring(&mut self, inputs: Mono<S>, outputs: &[usize]) -> Result<Draft, String> {
        let sorts: Vec<&S> = inputs.iter().collect();
        let mut cod = Vec::with_capacity(outputs.len());
        for &k in outputs {
            let sort = sorts.get(k).ok_or_else(|| {
                format!("wiring: there is no input {k} in {} inputs", sorts.len())
            })?;
            cod.push((*sort).clone());
        }
        let cod = Poly::from(Mono::from_iter(cod));
        Ok(self.single(inputs, cod, |arena, ab| {
            let outs = outputs.iter().map(|&k| Value(k as u32)).collect();
            ab.end(arena, 0, outs)
        }))
    }

    /// The copy `m → m ⊗ m`.
    pub fn copy(&mut self, m: Mono<S>) -> Draft {
        let n = m.iter().count();
        let outputs: Vec<usize> = (0..n).chain(0..n).collect();
        self.wiring(m, &outputs).expect("copy picks only inputs")
    }

    /// The discard `m → I`.
    pub fn discard(&mut self, m: Mono<S>) -> Draft {
        self.wiring(m, &[]).expect("discard picks no inputs")
    }

    /// The symmetry of strings `a ⊗ b → b ⊗ a`, the counterpart for `⊗` of
    /// [`swap`](Self::swap).
    pub fn cross(&mut self, a: Mono<S>, b: Mono<S>) -> Draft {
        let (na, nb) = (a.iter().count(), b.iter().count());
        let inputs = Mono::from_iter(a.iter().chain(b.iter()).cloned());
        let outputs: Vec<usize> = (na..na + nb).chain(0..na).collect();
        self.wiring(inputs, &outputs)
            .expect("cross picks only inputs")
    }

    /// The case split `M ⊗ B → M ⊕ M`, for the boolean sort `B`: the
    /// values of `M` go to the first output if the boolean is true, to the
    /// second if it is false. The boolean is consumed.
    pub fn case(&mut self, m: Mono<S>) -> Draft {
        let n = len(&m);
        let dom = m.clone() * Mono::from_iter([self.boolean.clone()]);
        let cod = m.clone() + m;
        self.single(dom, cod, |arena, ab| {
            let values: Vec<Value> = (0..n).map(Value).collect();
            let hi = ab.end(arena, 0, values.clone());
            let lo = ab.end(arena, 1, values);
            ab.split(arena, Value(n), hi, lo)
        })
    }

    /// The identity `p → p`.
    pub fn id(&mut self, p: Poly<S>) -> Draft {
        self.routing(p.clone(), p, |k| k)
    }

    /// The symmetry `p ⊕ q → q ⊕ p`.
    pub fn swap(&mut self, p: Poly<S>, q: Poly<S>) -> Draft {
        let (np, nq) = (summands(&p), summands(&q));
        let dom = p.clone() + q.clone();
        self.routing(dom, q + p, move |k| if k < np { nq + k } else { k - np })
    }

    /// The merge `p ⊕ p → p`: both copies of each summand go to that
    /// summand. This is how tapes join.
    pub fn merge(&mut self, p: Poly<S>) -> Draft {
        let n = summands(&p);
        self.routing(p.clone() + p.clone(), p, move |k| k % n)
    }

    /// The draft `0 → p` from the empty sum: it has no arms. With
    /// [`sum`](Self::sum) it gives the injections, for instance
    /// `inl = id(A) ⊕ initial(B) : A → A ⊕ B`.
    pub fn initial(&mut self, p: Poly<S>) -> Draft {
        self.add(Poly::zero(), p, Vec::new())
    }

    /// The direct sum `t ⊕ s`: the arms of `t`, then those of `s`, whose
    /// outputs follow those of `t`. Nothing is copied.
    pub fn sum(&mut self, t: Draft, s: Draft) -> Draft {
        let shift = summands(&self.data(t).cod);
        let mut arms = self.arms_of(t);
        arms.extend(self.arms_of(s).into_iter().map(|a| DraftArm {
            arm: a.arm,
            shift: a.shift + shift,
        }));
        let dom = self.data(t).dom.clone() + self.data(s).dom.clone();
        let cod = self.data(t).cod.clone() + self.data(s).cod.clone();
        self.add(dom, cod, arms)
    }

    /// The whiskering `l ⊗ t ⊗ r`: the strings `l` and `r` run alongside
    /// `t`, on every tape. Every summand `U` of the domain of `t` becomes
    /// `l ⊗ U ⊗ r`, and so does every summand of its codomain.
    pub fn whisker(&mut self, l: Mono<S>, t: Draft, r: Mono<S>) -> Draft {
        let (nl, nr) = (len(&l), len(&r));
        let wrap = |p: &Poly<S>| -> Poly<S> {
            p.iter()
                .map(|m| l.clone() * m.clone() * r.clone())
                .collect()
        };
        let dom = wrap(&self.data(t).dom);
        let cod = wrap(&self.data(t).cod);
        let mut arms = Vec::new();
        for a in self.arms_of(t) {
            let n = self.arena.arms[a.arm as usize].n_inputs;
            let mut ab = ArmBuilder::new(&self.arena, nl + n + nr);
            let left: Vec<Value> = (0..nl).map(Value).collect();
            let right: Vec<Value> = (nl + n..nl + n + nr).map(Value).collect();
            let inputs: Vec<Value> = (nl..nl + n).map(Value).collect();
            let root = copy_arm(
                &mut self.arena,
                &mut ab,
                a.arm,
                &inputs,
                &mut |arena, ab, output, outs| {
                    let outs = left
                        .iter()
                        .copied()
                        .chain(outs)
                        .chain(right.iter().copied())
                        .collect();
                    ab.end(arena, output, outs)
                },
            );
            arms.push(DraftArm {
                arm: ab.finish(&mut self.arena, root),
                shift: a.shift,
            });
        }
        self.add(dom, cod, arms)
    }

    /// Sequential composition `t ; s`, first `t` then `s`.
    ///
    /// Every arm of `t` is copied, and at each of its ends, to output `i`
    /// with values `outs`, the arm `i` of `s` is inlined with its inputs
    /// replaced by `outs`. The draws of the inlined arm are new at each end.
    ///
    /// Fails if the codomain of `t` is not the domain of `s`.
    pub fn compose(&mut self, t: Draft, s: Draft) -> Result<Draft, String> {
        if self.data(t).cod != self.data(s).dom {
            return Err(format!(
                "compose: a draft writing {:?} cannot feed a draft reading {:?}",
                self.data(t).cod,
                self.data(s).dom
            ));
        }
        let s_arms = self.arms_of(s);
        let mut arms = Vec::new();
        for a in self.arms_of(t) {
            let n = self.arena.arms[a.arm as usize].n_inputs;
            let mut ab = ArmBuilder::new(&self.arena, n);
            let inputs = ab.inputs();
            let root = copy_arm(
                &mut self.arena,
                &mut ab,
                a.arm,
                &inputs,
                &mut |arena, ab, output, outs| {
                    let b = s_arms[(output + a.shift) as usize];
                    copy_arm(arena, ab, b.arm, &outs, &mut |arena, ab, output, outs| {
                        ab.end(arena, output + b.shift, outs)
                    })
                },
            );
            arms.push(DraftArm {
                arm: ab.finish(&mut self.arena, root),
                shift: 0,
            });
        }
        let dom = self.data(t).dom.clone();
        let cod = self.data(s).cod.clone();
        Ok(self.add(dom, cod, arms))
    }

    /// Brings a tape back as a draft, to build on it. Its binders and levels
    /// are dropped: every non-deterministic node becomes a draw.
    ///
    /// Fails if the tape's boolean sort is not the builder's.
    pub fn import(&mut self, tape: &Tape<S, O>) -> Result<Draft, String> {
        if *tape.boolean() != self.boolean {
            return Err(format!(
                "import: the tape's boolean sort is {:?}, not {:?}",
                tape.boolean(),
                self.boolean
            ));
        }
        let mut arms = Vec::new();
        for arm in tape.arms() {
            let mut ab = ArmBuilder::new(&self.arena, len(arm.inputs()));
            let mut map: Vec<Value> = ab.inputs();
            for node in arm.nodes() {
                let kind = match node.kind() {
                    NodeKind::Det(op) => SKind::Det(op.clone()),
                    NodeKind::NonDet { op, .. } => SKind::Draw(op.clone()),
                };
                let args = node.args().iter().map(|v| map[v.index()]).collect();
                let first = ab.node(&mut self.arena, kind, args, node.node().n_outs);
                map.extend((0..node.node().n_outs).map(|k| Value(first.0 + k)));
            }
            let start = arm.arm().routes.start;
            let mut copies: Vec<u32> = Vec::new();
            for route in arm.routes() {
                let copy = match *route.route() {
                    Route::Split { cond, hi, lo, .. } => {
                        let hi = copies[(hi.0 - start) as usize];
                        let lo = copies[(lo.0 - start) as usize];
                        ab.split(&mut self.arena, map[cond.index()], hi, lo)
                    }
                    Route::End { output, .. } => {
                        let outs = route.outs().unwrap_or_default();
                        let outs = outs.iter().map(|v| map[v.index()]).collect();
                        ab.end(&mut self.arena, output, outs)
                    }
                };
                copies.push(copy);
            }
            let root = *copies.last().expect("an arm has a root");
            arms.push(DraftArm {
                arm: ab.finish(&mut self.arena, root),
                shift: 0,
            });
        }
        let parts = tape.parts();
        Ok(self.add(parts.dom.clone(), parts.cod.clone(), arms))
    }

    /// The tape of `draft`, in normal form. The builder is left unchanged,
    /// and `draft` can still be built on.
    ///
    /// # Panics
    /// If `draft` was not created by this builder, or, which would be a bug,
    /// if the normal form is not a well-formed tape.
    pub fn freeze(&self, draft: Draft) -> Tape<S, O> {
        let data = self.data(draft);
        let arms = self
            .arms_of(draft)
            .into_iter()
            .map(|a| LocalArm::from_scratch(&self.arena, a.arm, a.shift))
            .collect();
        canon::freeze(
            self.boolean.clone(),
            data.dom.clone(),
            data.cod.clone(),
            arms,
        )
    }
}
