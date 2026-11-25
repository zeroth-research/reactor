"""Tests for the pure-Python SMV parser."""

from pathlib import Path
import pytest

from reactor import Wire, Module, parse_smv, Var, X
from reactor.matrix.sort import Word, Bool
from reactor import matrix
from reactor.matrix import Skip
from reactor.eval import eval_itype

FIXTURES = Path(__file__).parent / "smv_fixtures"


def test_counter_basic():
    """Parse counter.smv — 5 variables (3 VAR + 2 IVAR), check obs pairs."""
    text = (FIXTURES / "counter.smv").read_text()
    module, name_map = parse_smv(text)

    assert isinstance(module, Module)
    assert len(name_map) == 5  # x, y, z, y0, z0

    # All 5 variables appear as obs pairs
    assert len(module.obs) == 5

    # VAR names
    for v in ("x", "y", "z"):
        assert v in name_map
        var = name_map[v]
        assert isinstance(var, Var)
        assert isinstance(X(var), Wire)
        assert var != X(var)

    # IVAR names
    for v in ("y0", "z0"):
        assert v in name_map


def test_counter_init_terms():
    """counter.smv: init(x) := 0 should produce a Const(0) term."""
    text = (FIXTURES / "counter.smv").read_text()
    module, name_map = parse_smv(text)

    # The module has atoms; first atom covers init/update
    atom = module.atoms[0]
    init_terms = atom.init
    assert len(init_terms) > 0

    # At least one init term should be a zero-valued word constant for x.
    def _is_const_zero(itype):
        match itype:
            case matrix.word.Const(data):
                return int(data.item()) == 0
        return False

    const_zero_found = any(_is_const_zero(t.itype) for t in init_terms)
    assert const_zero_found, "Expected a word.Const(0) init term for x"


def test_test_itypes():
    """Parse test_itypes.smv — 6 variables (3 VAR + 3 IVAR)."""
    text = (FIXTURES / "test_itypes.smv").read_text()
    module, name_map = parse_smv(text)

    assert isinstance(module, Module)
    assert len(name_map) == 6  # i, j, a, i0, j0, a0
    assert len(module.obs) == 5  # j0 is unused


def test_hrm_word_level():
    """Smoke test for the large word-level HRM file with INVAR/TRANS."""
    text = (FIXTURES / "hrm_word_level_no_prop.smv").read_text()
    module, name_map = parse_smv(text)

    assert isinstance(module, Module)
    # Should have many variables
    assert len(name_map) > 50
    assert len(module.obs) == len(name_map)


def test_wire_overrides():
    """Overrides should reuse provided wires instead of creating new ones."""
    text = (FIXTURES / "counter.smv").read_text()

    # Create override wires for 'x'
    # `integer` lowers to 32-bit words; overrides must match.
    dtype = Word(32, [1, 1])
    ov = Var(dtype)

    module, name_map = parse_smv(text, overrides={"x": ov})

    assert name_map["x"] == ov
    # Other variables should get fresh wires
    assert name_map["y"] != ov


def test_parse_error():
    """Invalid input should raise an error."""
    with pytest.raises(Exception):
        parse_smv("THIS IS NOT VALID SMV")


def test_define_expansion():
    """DEFINE macros should be expanded inline."""
    text = """
    MODULE main
    VAR x : integer;
    DEFINE doubled := x + x;
    ASSIGN
        init(x) := 0;
        next(x) := doubled;
    """
    module, name_map = parse_smv(text)
    assert isinstance(module, Module)
    assert "x" in name_map

    # The update terms should contain Add (from doubled := x + x)
    atom = module.atoms[0]
    update_itypes = [str(t.itype) for t in atom.update]
    assert any("Add" in it for it in update_itypes), f"Expected Add in update terms, got {update_itypes}"


def test_enum_type():
    """Enum types should map to integer, and enum values should be resolvable."""
    text = """
    MODULE main
    VAR state : {idle, running, stopped};
    ASSIGN
        init(state) := idle;
        next(state) := case
            state = idle    : running;
            state = running : stopped;
            TRUE            : state;
        esac;
    """
    module, name_map = parse_smv(text)
    assert isinstance(module, Module)
    assert "state" in name_map
    # state should have TensorInt dtype (enum mapped to int)
    state = name_map["state"]
    # SMV parser maps `integer` / enum types to Word32([1, 1]).
    assert isinstance(state.dtype, Word)


def test_frozen_var():
    """FROZENVAR should get identity update (next := current)."""
    text = """
    MODULE main
    FROZENVAR c : integer;
    VAR x : integer;
    ASSIGN
        init(x) := c;
        next(x) := x + c;
    """
    module, name_map = parse_smv(text)
    assert isinstance(module, Module)
    assert "c" in name_map
    assert "x" in name_map

    # Should have a Skip term (the identity) in update for the frozen var
    atom = module.atoms[0]
    assert any(isinstance(t.itype, Skip) for t in atom.update)


# ── booleans and words ───────────────────────────────────────────────────────

def _update(text):
    module, name_map = parse_smv(text)
    return module.atoms[0].update, name_map


def test_boolean_vars_are_bool():
    """`boolean` lowers to Bool, and TRUE/FALSE to boolean constants."""
    module, name_map = parse_smv("""
    MODULE main
    VAR b : boolean;
    ASSIGN init(b) := TRUE; next(b) := !b;
    """)
    assert name_map["b"].dtype == Bool([1, 1])
    init = module.atoms[0].init
    assert any(isinstance(t.itype, matrix.bool.Const) for t in init)
    assert any(isinstance(t.itype, matrix.bool.Not) for t in module.atoms[0].update)


def test_comparisons_write_bool_and_respect_sign():
    update, _ = _update("""
    MODULE main
    VAR x : signed word[8]; y : unsigned word[8]; b : boolean; c : boolean;
    ASSIGN next(b) := x < 0sd8_1; next(c) := y < 0ud8_1;
    """)
    kinds = {type(t.itype) for t in update}
    assert matrix.word.signed.Lt in kinds and matrix.word.unsigned.Lt in kinds
    cmps = [t for t in update if isinstance(t.itype, (matrix.word.signed.Lt, matrix.word.unsigned.Lt))]
    assert all(t.write[0].dtype == Bool([1, 1]) for t in cmps)


def test_literal_takes_the_width_of_its_operand():
    """A bare number combined with an 8-bit word is an 8-bit constant."""
    update, _ = _update("""
    MODULE main
    VAR x : unsigned word[8];
    ASSIGN next(x) := x + 1;
    """)
    consts = [t for t in update if isinstance(t.itype, matrix.word.Const)]
    assert consts and all(t.write[0].dtype == Word(8, [1, 1]) for t in consts)
    assert any(isinstance(t.itype, matrix.word.Add) and t.itype.width == 8 for t in update)


def test_word_constants_add_modulo_their_width():
    """`0ud8_200 + 0ud8_100` adds two 8-bit words modulo 2^8, as in NuSMV: 44."""
    update, _ = _update("""
    MODULE main
    VAR x : unsigned word[8];
    ASSIGN next(x) := 0ud8_200 + 0ud8_100;
    """)
    terms = list(update)
    consts = [t for t in terms if isinstance(t.itype, matrix.word.Const)]
    assert len(consts) == 2 and all(t.write[0].dtype == Word(8, [1, 1]) for t in consts)
    assert any(isinstance(t.itype, matrix.word.Add) and t.itype.width == 8 for t in terms)
    state = {}
    for t in terms:
        state.update(zip(t.write, eval_itype(t.itype, [state[w] for w in t.read], t.write[0].dtype)))
    assert state[terms[-1].write[0]].item() == (200 + 100) % 256


def test_bool_of_word1_uses_extract():
    """bool(x) of a 1-bit word is its bit 0 compared to 1."""
    update, _ = _update("""
    MODULE main
    VAR w : unsigned word[1]; b : boolean;
    ASSIGN next(b) := bool(w);
    """)
    kinds = [type(t.itype) for t in update]
    assert matrix.word.Extract in kinds and matrix.word.Eq in kinds


def test_case_lowers_to_ite_with_bool_condition():
    update, name_map = _update("""
    MODULE main
    VAR x : unsigned word[4]; b : boolean;
    ASSIGN next(x) := case b : x + 1; TRUE : x; esac;
    """)
    ites = [t for t in update if isinstance(t.itype, matrix.word.Ite)]
    assert ites and all(t.read[0].dtype == Bool([1, 1]) for t in ites)
