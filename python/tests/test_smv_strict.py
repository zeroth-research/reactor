"""`parse_smv(strict=True)`: the module means what NuSMV means, or the parse
is refused; `read_smv`: that module, and the properties it states."""

from pathlib import Path

import pytest
import torch

from reactor import Bool, X
from reactor import tensor
from reactor.eval import eval_itype, execute_init, execute_update
from reactor.smv import SMVRefused, parse_smv, read_smv

FIXTURES = Path(__file__).parent / "smv_fixtures"


def _run(text, steps, *, strict=True):
    """Values of every variable after init and after each step in `steps`,
    a list of {input: (value read this round, value latched)}."""
    module, names = parse_smv(text, strict=strict)

    def value(name, v):
        bool_ = isinstance(names[name].dtype, Bool)
        return torch.tensor([[v]], dtype=torch.bool if bool_ else torch.int64)

    state = {}
    execute_init(state, module.atoms)
    trace = []

    def snapshot():
        trace.append({n: int(state[X(w)].item()) for n, w in names.items()
                      if X(w) in state})
        for v in module.ctrl:
            state[v] = state[X(v)].clone()

    snapshot()
    for inputs in steps:
        for name, (now, latched) in inputs.items():
            state[X(names[name])] = value(name, now)
            state[names[name]] = value(name, latched)
        execute_update(state, module.atoms)
        snapshot()
    return trace


INVAR_C_D = """MODULE main
IVAR i : boolean;
VAR {first} : unsigned word[4];
    {second} : unsigned word[4];
{define}
ASSIGN
  init(c) := 0ud4_0;
  next(c) := i ? d : c;
"""


@pytest.mark.parametrize("define", ["INVAR d = c + 0ud4_1;",
                                    "ASSIGN d := c + 0ud4_1;"])
@pytest.mark.parametrize("first,second", [("c", "d"), ("d", "c")])
def test_a_value_given_in_every_state_holds_in_every_state(define, first, second):
    text = INVAR_C_D.format(first=first, second=second, define=define)
    trace = _run(text, [{"i": (1, 0)}, {"i": (1, 1)}])
    assert [(s["c"], s["d"]) for s in trace] == [(0, 1), (1, 2), (2, 3)]


def test_an_input_is_read_in_the_round_it_labels():
    text = """MODULE main
IVAR i : boolean;
VAR c : boolean;
ASSIGN init(c) := FALSE; next(c) := i;
"""
    trace = _run(text, [{"i": (1, 0)}, {"i": (0, 1)}])
    assert [s["c"] for s in trace] == [0, 1, 0]


def test_next_of_an_expression_reads_next_values():
    text = """MODULE main
VAR a : unsigned word[4];
    b : unsigned word[4];
ASSIGN
  init(a) := 0ud4_0; next(a) := a + 0ud4_1;
  init(b) := 0ud4_0; next(b) := next(a + 0ud4_1);
"""
    trace = _run(text, [{}])
    assert trace[1] == {"a": 1, "b": 2}


def test_trans_assignments_are_read():
    text = """MODULE main
VAR x : boolean;
INIT x = FALSE
TRANS next(x) = (!x)
"""
    for strict in (True, False):
        assert [s["x"] for s in _run(text, [{}, {}], strict=strict)] == [0, 1, 0]


@pytest.mark.parametrize("strict", [True, False])
def test_mod_is_read(strict):
    text = """MODULE main
VAR x : unsigned word[4];
ASSIGN init(x) := 0ud4_7; next(x) := x mod 0ud4_3 mod 0ud4_2;
"""
    assert [s["x"] for s in _run(text, [{}], strict=strict)] == [7, 1]


@pytest.mark.parametrize("strict", [True, False])
def test_signed_division_mod_and_extend_are_read(strict):
    """As in NuSMV: `/` on signed words rounds toward zero, `mod` takes the sign of
    the dividend, and `extend` of a signed word extends its sign."""
    text = """MODULE main
VAR q : signed word[4];
    r : signed word[4];
    e : signed word[6];
    u : unsigned word[6];
ASSIGN
  init(q) := -0sd4_3 / 0sd4_2;       next(q) := q;
  init(r) := -0sd4_3 mod 0sd4_2;     next(r) := r;
  init(e) := extend(-0sd4_1, 2);     next(e) := e;
  init(u) := extend(0ud4_15, 2);     next(u) := u;
"""
    (first,) = _run(text, [], strict=strict)
    assert first == {"q": 0b1111, "r": 0b1111, "e": 0b111111, "u": 15}


@pytest.mark.parametrize("expr", [
    "(-x) < 0sd4_0",
    "(!x) < 0sd4_0",
    "(x -> n) < 0sd4_0",
    "(x = 0sd4_0 ? x : n) < 0sd4_0",
    "case x = 0sd4_0 : x; TRUE : n; esac < 0sd4_0",
])
def test_a_signed_word_stays_signed_through_an_operator(expr):
    """`-x` of a signed word is a signed word, and `<` compares it signed:
    with `x = 1`, `-x < 0` holds. Read unsigned, `-x` is 15 and it does not."""
    text = f"""MODULE main
VAR x : signed word[4];
    n : signed word[4];
    b : boolean;
ASSIGN
  init(x) := 0sd4_1;  next(x) := x;
  init(n) := -0sd4_1; next(n) := n;
  init(b) := FALSE;   next(b) := {expr};
"""
    assert [s["b"] for s in _run(text, [{}])] == [0, 1]


def _model(body, decls="VAR x : boolean;"):
    return f"MODULE main\n{decls}\n{body}\n"


REFUSED = {
    "no next": (_model("ASSIGN init(x) := TRUE;"), "`x` has no next value"),
    "no init": (_model("ASSIGN next(x) := x;"), "`x` has no initial value"),
    "integer with words": (_model("", "VAR x : integer; w : unsigned word[4];"),
                           "`x` has an integer type in a model with words"),
    "range with words": (_model("", "VAR x : 0..9; w : unsigned word[4];"),
                         "`x` has an integer type in a model with words"),
    "numeric enum with words": (_model("", "VAR x : {0, 2}; w : unsigned word[4];"),
                                "`x` has an integer type in a model with words"),
    "mixed enum": (_model("", "VAR x : {idle, 2};"), "mixes names and numbers"),
    "enum clash": (_model("ASSIGN init(x) := TRUE; next(x) := x;",
                          "VAR x : boolean; s : {x, y};"),
                   "`x` is both an enum value"),
    "input in init": (_model("ASSIGN init(x) := i; next(x) := x;",
                             "IVAR i : boolean; VAR x : boolean;"),
                      r"input variable `i` is read .* \(in init\(x\)\)"),
    "input in INVAR": (_model("INVAR x = i",
                              "IVAR i : boolean; VAR x : boolean;"),
                       "input variable `i` is read"),
    "next of input": (_model("ASSIGN init(x) := TRUE; next(x) := next(i);",
                             "IVAR i : boolean; VAR x : boolean;"),
                      "input variable `i` is read"),
    "next in init": (_model("ASSIGN init(x) := next(x); next(x) := x;"),
                     r"`next\(\)` is not allowed here"),
    "undeclared": (_model("ASSIGN init(x) := TRUE; next(x) := y;"),
                   r"`y` is not declared \(in next\(x\)\)"),
    "case without TRUE": (
        _model("ASSIGN init(x) := TRUE;\n"
               "next(x) := case x : FALSE; !x : TRUE; esac;"),
        "the last branch of a `case` must have condition TRUE"),
    "INIT constraint": (_model("INIT x | !x\nASSIGN next(x) := x;"),
                        "line 3: INIT is not `x = e`"),
    "TRANS constraint": (_model("ASSIGN init(x) := TRUE;\nTRANS next(x) | x"),
                         r"TRANS is not `next\(x\) = e`"),
    "INVAR constraint": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                                "INVAR x | !x"),
                         "INVAR is not `x = e`"),
    "FAIRNESS": (_model("ASSIGN init(x) := TRUE; next(x) := x;\nFAIRNESS x;"),
                 "FAIRNESS constraints are not read"),
    "JUSTICE": (_model("ASSIGN init(x) := TRUE; next(x) := x;\nJUSTICE x;"),
                "JUSTICE constraints are not read"),
    "COMPASSION": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                          "COMPASSION (x, x);"),
                   "COMPASSION constraints are not read"),
    "assigned twice": (_model("ASSIGN init(x) := TRUE; init(x) := FALSE;\n"
                              "next(x) := x;"),
                       "`x` is assigned twice"),
    "input assigned": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                              "next(i) := x;",
                              "IVAR i : boolean; VAR x : boolean;"),
                       "input variable `i` is assigned"),
    "undeclared assigned": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                                   "next(y) := x;"),
                            "`y` is assigned but not declared"),
    "frozen next": (_model("ASSIGN init(x) := TRUE; next(x) := x;",
                           "FROZENVAR x : boolean;"),
                    "FROZENVAR `x` is assigned a next value"),
    "every state and init": (_model("ASSIGN x := TRUE; init(x) := FALSE;"),
                             "in every state and also"),
    "cycle": (_model("ASSIGN init(x) := TRUE; init(y) := TRUE;\n"
                     "next(x) := next(y); next(y) := next(x);",
                     "VAR x : boolean; y : boolean;"),
              "the next state depends on itself"),
    "nonlinear": (_model("ASSIGN init(x) := 1; next(x) := x * x;",
                         "VAR x : integer;"),
                  "`\\*` of two non-constants"),
    "division": (_model("ASSIGN init(x) := 1; next(x) := x / 2;",
                        "VAR x : integer;"),
                 "`/` is not linear integer arithmetic"),
    "mod": (_model("ASSIGN init(x) := 1; next(x) := x mod 2;",
                   "VAR x : integer;"),
            "`mod` is not linear integer arithmetic"),
    "boolean as integer": (_model("ASSIGN init(x) := 1; next(x) := x + b;\n"
                                  "init(b) := TRUE; next(b) := b;",
                                  "VAR x : integer; b : boolean;"),
                           "a boolean where an integer is needed"),
    "integer to boolean": (_model("ASSIGN init(b) := 0; next(b) := b;",
                                  "VAR b : boolean;"),
                           "an integer where a boolean is needed"),
    "bare number": (_model("ASSIGN init(w) := 0; next(w) := w;",
                           "VAR w : unsigned word[4];"),
                    "the number 0 is not a word constant"),
    "word as boolean": (_model("ASSIGN init(b) := w; next(b) := b;\n"
                               "init(w) := 0ud1_0; next(w) := w;",
                               "VAR b : boolean; w : unsigned word[1];"),
                        "a word\\[1\\] where a boolean is needed"),
    "words of two widths": (_model("ASSIGN init(w) := 0ud4_0;\n"
                                   "next(w) := w + 0ud8_1;",
                                   "VAR w : unsigned word[4];"),
                            "a word\\[8\\] where a word\\[4\\] is needed"),
    "word condition": (_model("ASSIGN init(w) := 0ud1_0;\n"
                              "next(w) := w ? 0ud1_0 : 0ud1_1;",
                              "VAR w : unsigned word[1];"),
                       "a word\\[1\\] where a boolean is needed"),
    "bool of a wide word": (_model("ASSIGN init(b) := bool(w); next(b) := b;\n"
                                   "init(w) := 0ud4_0; next(w) := w;",
                                   "VAR b : boolean; w : unsigned word[4];"),
                            r"no `bool\(\)` on a word\[4\]"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_refused(case):
    text, match = REFUSED[case]
    with pytest.raises(SMVRefused, match=match):
        parse_smv(text, strict=True)


def test_hrm_is_refused_for_its_free_variables():
    text = (FIXTURES / "hrm_word_level_no_prop.smv").read_text()
    with pytest.raises(SMVRefused) as e:
        parse_smv(text, strict=True)
    assert "`hrm_enhanced_ebmc_dut_MAX_BPM` has no initial value" in str(e.value)
    assert "TRANS" not in str(e.value)


def test_a_model_without_words_is_integers():
    from reactor import Int

    text = (FIXTURES / "mod10.smv").read_text()
    module, names = parse_smv(text, strict=True)
    assert names["x"].dtype == Int([1, 1])
    assert names["inc"].dtype == Bool([1, 1])
    assert [str(v) for v in module.ctrl] == [str(names["x"])]
    assert [str(v) for v in module.extl] == [str(names["inc"])]
    trace = _run(text, [{"inc": (1, 0)}] * 10 + [{"inc": (0, 1)}])
    assert [s["x"] for s in trace] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0]


def test_integer_operators():
    text = """MODULE main
VAR x : integer;
    n : integer;
    b : boolean;
    s : {idle, busy};
ASSIGN
  init(x) := -3;  next(x) := 2 * x + abs(n);
  init(n) := -1;  next(n) := n - 1;
  init(b) := FALSE; next(b) := (x < 0 -> b) <-> (n != -1);
  init(s) := idle; next(s) := s = idle ? busy : idle;
"""
    trace = _run(text, [{}, {}])
    assert [(t["x"], t["n"], t["b"], t["s"]) for t in trace] == [
        (-3, -1, 0, 0), (-5, -2, 1, 1), (-8, -3, 1, 0)]


def test_an_integer_model_is_linear():
    """A product with a constant scales, and the constant is no term of its own."""
    module, _ = parse_smv("""MODULE main
VAR x : integer;
ASSIGN init(x) := 1; next(x) := x * -2;
""", strict=True)
    update = module.atoms[0].update
    assert [type(t.itype) for t in update] == [tensor.int.Scale]
    assert int(update[0].itype.tensor.item()) == -2


@pytest.mark.parametrize("strict", [True, False])
def test_implies_and_iff_are_read_on_words(strict):
    text = """MODULE main
VAR a : boolean;
    b : boolean;
    w : unsigned word[1];
ASSIGN
  init(a) := FALSE; next(a) := !a;
  init(b) := TRUE;  next(b) := (a -> b) <-> a;
  init(w) := 0ud1_0; next(w) := w;
"""
    trace = _run(text, [{}, {}], strict=strict)
    # a: 0 1 0; b: 1, ((0->1)<->0)=0, ((1->0)<->1)=0
    assert [(t["a"], t["b"]) for t in trace] == [(0, 1), (1, 0), (0, 0)]


def test_a_wire_does_not_take_the_sign_of_a_dead_one():
    """The lowerer records which wires hold signed words. A temporary's
    Python object is freed once its `Term` holds a copy, and the next wire
    is allocated at the same address: keyed by `id()`, it was signed too."""
    from lark import Token, Tree

    from reactor import Word, Wire
    from reactor.smv.parser import _Lowerer

    low = _Lowerer({}, {}, {}, {}, strict=True)
    sort = Word(4, [1, 1])
    low.lower(Tree("word_lit", [Token("WORD_LIT", "0sd4_1")]))
    fresh = Wire(sort)              # first allocation after the literal's
    assert not low._is_signed(fresh)


# ── the properties a model states ───────────────────────────────────────────

MOD10 = (FIXTURES / "mod10.smv").read_text()


@pytest.mark.parametrize("spec, kind", [
    ("INVARSPEC x <= 9;", "safety"),
    ("LTLSPEC G x <= 9", "safety"),
    ("LTLSPEC G F x = 0;", "buchi"),
    ("LTLSPEC G (F (x = 0))", "buchi"),
    ("LTLSPEC NAME back := G F x = 0", "buchi"),
])
def test_a_spec_reads_as_a_claim(spec, kind):
    _, _, specs = read_smv(MOD10 + spec + "\n")
    assert [(s.kind, s.where) for s in specs] == [(kind, f"line 14: {spec.split()[0]}")]


@pytest.mark.parametrize("spec, match", [
    ("LTLSPEC F x = 0", "only `INVARSPEC p`"),
    ("LTLSPEC x = 0", "only `INVARSPEC p`"),
    ("LTLSPEC G X x = 0", "only `INVARSPEC p`"),
    ("INVARSPEC inc", r"input variable `inc` is read .* \(in line 14: INVARSPEC\)"),
    ("INVARSPEC x", "the property is not a boolean"),
    ("INVARSPEC next(x) = 0", r"`next\(\)` is not allowed here"),
])
def test_any_other_spec_is_refused(spec, match):
    with pytest.raises(SMVRefused, match=match):
        read_smv(MOD10 + spec + "\n")


def test_a_spec_reads_the_state_after_a_round():
    module, names, (spec,) = read_smv(MOD10 + "INVARSPEC 0 <= x & x <= 9;\n")
    assert module.ctrl == [names["x"]]
    for x, holds in ((0, True), (9, True), (10, False), (-1, False)):
        state = {X(names["x"]): torch.tensor([[x]])}
        for t in spec.terms:
            state.update(zip(t.write, eval_itype(t.itype, [state[w] for w in t.read])))
        assert bool(state[spec.wire].item()) is holds


def test_specs_leave_the_module_alone():
    import re

    plain, _ = parse_smv(MOD10, strict=True)
    stated, _ = parse_smv(MOD10 + "INVARSPEC x <= 9;\n", strict=True)
    ids = re.compile(r"#\d+")
    assert ids.sub("#N", str(plain)) == ids.sub("#N", str(stated))
