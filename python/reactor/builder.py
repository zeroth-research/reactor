"""Stateless Term factories, one per theory.

Temporary scaffolding kept to move fast during the theory_pyo3 migration; the
end-state is to call the theory ops directly, so this should not grow.

A builder picks the sorts of the literals of its theory (`Real` for LRA, `Int` for
LIA, `Word(32)` for BV, and `Bool` for all), and the operations by the sorts of their
operands, through `reactor.overload`; shape, width and const payloads are obtained
via `match`/unpacking (see the module helpers below).
"""

import torch
from . import op, overload as ov
from .reactor import Wire, Term, Var
from .op import LRA, LIA, BV
from .overload import TheoryError
from .sort import Sort, Bool, Real, Int, Word, shape as _shape, with_shape as _with_shape


def _wire(t) -> Wire:
    if isinstance(t, Term):
        return t.write[0]
    elif isinstance(t, list):
        return t[0]
    elif isinstance(t, Var):
        return t
    elif isinstance(t, Wire):
        return t
    else:
        raise TypeError(f'_wire() unexpected type(t)={type(t)}')


def _dtype(t) -> Sort:
    return _wire(t).dtype


def _normalize_shape(shape: list) -> list:
    if len(shape) == 0:
        return [1, 1]
    if len(shape) == 1:
        return [1, shape[0]]
    return shape


# --- Sort / op access via `match` --------------------------------------------


def _bv_width(sort: Sort) -> int:
    match sort:
        case Word(bw, _):
            return bw
    raise TypeError(f"not a Word sort: {sort}")


def _bool_sort(sort: Sort) -> Sort:
    return Bool(_shape(sort))


def _const_tensor(itype):
    """The constant tensor carried by a const op, or None."""
    match itype:
        case op.real.Const(t) | op.int.Const(t) | op.bool.Const(t) | op.word.Const(t, _):
            return t
    return None


class NonLinearError(TheoryError):
    """A non-linear op (e.g. variable * variable) in a linear theory (LIA/LRA),
    which only support multiplication by a constant."""

    def __init__(self, theory: str, detail: str = "multiplication of two non-constant operands"):
        self.theory = theory
        super().__init__(f"{theory} does not support {detail}")


class TermBuilder:
    """Abstract base. Subclasses set `_logic = LRA / LIA / BV`."""

    _logic = None

    # --- low-level helpers ---

    def _binary_op(self, itype, output_sort: Sort, a, b) -> Term:
        return Term(itype, [Wire(output_sort)], [_wire(a), _wire(b)])

    def _unary_op(self, overloaded, a) -> Term:
        return Term(overloaded(_dtype(a)), [Wire(_dtype(a))], [_wire(a)])

    def _cmp(self, overloaded, a, b) -> Term:
        return self._binary_op(overloaded(_dtype(a)), _bool_sort(_dtype(a)), a, b)

    # --- theory-independent ops ---

    def id_(self, src, output_wire=None) -> Term:
        w = output_wire or Wire(_dtype(src))
        return Term(ov.id_(_dtype(src)), [w], [_wire(src)])

    def ite(self, cond, a, b, output_wire=None) -> Term:
        w = output_wire or Wire(_dtype(a))
        return Term(ov.ite(_dtype(a)), [w], [_wire(cond), _wire(a), _wire(b)])

    # Logic on booleans, and bitwise on words: the result has the operands' sort.
    def not_(self, a) -> Term:
        return self._unary_op(ov.not_, a)

    def and_(self, a, b) -> Term:
        return self._binary_op(ov.and_(_dtype(a)), _dtype(a), a, b)

    def or_(self, a, b) -> Term:
        return self._binary_op(ov.or_(_dtype(a)), _dtype(a), a, b)

    def xor_(self, a, b) -> Term:
        return self._binary_op(ov.xor_(_dtype(a)), _dtype(a), a, b)

    def const_bool(self, value: bool, output_wire=None) -> Term:
        return self.const(
            torch.tensor([[bool(value)]], dtype=torch.bool), output_wire=output_wire
        )

    def transpose(self, a) -> Term:
        w = _wire(a)
        s = _shape(w.dtype)
        return Term(ov.transpose(w.dtype), [Wire(_with_shape(w.dtype, [s[1], s[0]]))], [w])

    def relu(self, a) -> Term:
        return self._unary_op(ov.relu, a)

    def argmax(self, a) -> Term:
        w = _wire(a)
        # Output keeps the input's sort family, reshaped to [1, 1].
        return Term(ov.argmax(w.dtype), [Wire(_with_shape(w.dtype, [1, 1]))], [w])

    def python_type_to_dtype(self, python_type: type, shape: list):
        raise NotImplementedError

    def const_for_value(self, value, output_wire=None) -> "Term":
        raise NotImplementedError

    def _try_mul_as_linear(self, a, b, const_cls, scalar_dtype):
        for const_t, var_t in [(a, b), (b, a)]:
            if not isinstance(const_t, Term):
                continue
            if not isinstance(const_t.itype, const_cls):
                continue
            data = _const_tensor(const_t.itype)
            if data is None or data.numel() != 1 or data.dtype == torch.bool:
                continue
            var_wire = _wire(var_t)
            s = _shape(var_wire.dtype)
            if len(s) < 2 or s[-1] != 1:
                continue
            n = s[0]
            A = torch.eye(n, dtype=scalar_dtype) * data.item()
            b_bias = torch.zeros(n, 1, dtype=scalar_dtype)
            itype = ov.linear(A, b_bias, var_wire.dtype)
            return Term(itype, [Wire(var_wire.dtype)], [var_wire])
        return None

    def uninterpreted(self, name: str, sort: Sort) -> Term:
        return Term.constant(ov.uninterpreted(name, sort), [Wire(sort)])

    def _numeric_wire(self, shape: list) -> Wire:
        """Wire of this theory's numeric sort with the given shape."""
        raise NotImplementedError

    def linear(self, x, weight, bias) -> list:
        """Emit [Transpose-in, Linear, Transpose-out]. Last write[0] is the output wire."""
        x_wire = _wire(x)
        shape = _shape(x_wire.dtype)  # [1, in_features] row-major
        in_features = shape[-1]
        out_features = weight.shape[0]

        input_col_wire = Wire(_with_shape(x_wire.dtype, [in_features, 1]))
        t_in = Term(ov.transpose(x_wire.dtype), [input_col_wire], [x_wire])

        linear_out_wire = self._numeric_wire([out_features, 1])
        t_lin = Term(
            ov.linear(weight, bias, input_col_wire.dtype), [linear_out_wire], [input_col_wire]
        )

        out_wire = self._numeric_wire([1, out_features])
        t_out = Term(ov.transpose(linear_out_wire.dtype), [out_wire], [linear_out_wire])

        return [t_in, t_lin, t_out]

    def linear_symbolic(self, x, out_features: int):
        # Linear carries its weight/bias as *constant* tensors in the op, so a Linear
        # with symbolic (wire) weights is not expressible.
        raise NotImplementedError(
            "symbolic Linear (wire weights) is not expressible: theory Linear takes constant A, B"
        )

    # --- arithmetic and comparisons, on the operands' sort ---

    def add(self, a, b) -> Term:
        return self._binary_op(ov.add(_dtype(a)), _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary_op(ov.sub(_dtype(a)), _dtype(a), a, b)

    def lt(self, a, b) -> Term:
        return self._cmp(ov.lt, a, b)

    def le(self, a, b) -> Term:
        return self._cmp(ov.le, a, b)

    def gt(self, a, b) -> Term:
        return self._cmp(ov.gt, a, b)

    def ge(self, a, b) -> Term:
        return self._cmp(ov.ge, a, b)

    def eq(self, a, b) -> Term:
        return self._cmp(ov.eq, a, b)

    def ne(self, a, b) -> Term:
        return self._cmp(ov.ne, a, b)

    # --- theory-specific ops (abstract) ---

    def mul(self, a, b) -> Term:
        raise NotImplementedError

    def const(self, tensor, output_wire=None) -> Term:
        raise NotImplementedError

    def div(self, a, b) -> Term:
        raise ValueError(f"{type(self).__name__} does not support div")

    def matmul(self, a, b) -> Term:
        raise NotImplementedError

    # --- deferred ops (no current theory provides them) ----------------------
    # These stay routed through the builder so the analyzer (and the tutorials)
    # remain theory-agnostic. When a suitable theory is added, only the relevant
    # TermBuilder subclass (and the theory crate) override these -- no analyzer
    # or notebook changes are needed.

    def _theory(self) -> str:
        return self._logic.__name__ if self._logic is not None else "the selected theory"

    def sin(self, a) -> Term:
        raise NonLinearError(self._theory(), "sin (non-linear; needs a non-linear theory)")

    def cos(self, a) -> Term:
        raise NonLinearError(self._theory(), "cos (non-linear; needs a non-linear theory)")

    def tanh(self, a) -> Term:
        raise NonLinearError(self._theory(), "tanh (non-linear; needs a non-linear theory)")

    def pow(self, base, exponent) -> Term:
        raise NonLinearError(self._theory(), "** / pow (non-linear; needs a non-linear theory)")

    def stack(self, elements) -> Term:
        raise TheoryError(
            f"{self._theory()} has no Stack op "
            "(list-valued state needs a theory that provides Stack)"
        )

    def tensor_get(self, base, index) -> Term:
        raise TheoryError(
            f"{self._theory()} has no TensorGet op "
            "(indexing needs a theory that provides TensorGet)"
        )

    # --- constants, of the theory's sorts ---

    def _const(self, tensor, numeric, output_wire=None) -> Term:
        """A constant: a boolean tensor is a `Bool`, and any other one of the sort
        `numeric(shape)`, unless `output_wire` gives the sort."""
        shape = _normalize_shape(list(tensor.size()))
        tensor = tensor.reshape(shape)  # theory const ops require a 2-D initializer
        if output_wire is not None:
            w = output_wire
        elif tensor.dtype == torch.bool:
            w = Wire(Bool(shape))
        else:
            w = Wire(numeric(shape))
        return Term.constant(ov.const(tensor, w.dtype), [w])


class LRATermBuilder(TermBuilder):
    _logic = LRA

    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(Real(shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Bool(shape)
        return Real(shape)

    # LRA is linear real arithmetic: no transcendentals / generic division.
    # (sin/cos/tanh/pow/stack/tensor_get inherit the base's deferred-op errors.)
    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b, op.real.Const, torch.float32)
        if result is not None:
            return result
        raise NonLinearError("LRA")

    def matmul(self, a, b) -> Term:
        a_wire, b_wire = _wire(a), _wire(b)
        out_shape = [_shape(a_wire.dtype)[0], _shape(b_wire.dtype)[1]]
        if isinstance(a, Term) and isinstance(a.itype, op.real.Const):
            no_bias = torch.empty(0, 0)
            return Term(
                ov.linear(_const_tensor(a.itype), no_bias, b_wire.dtype),
                [Wire(Real(out_shape))],
                [b_wire],
            )
        raise RuntimeError(
            "LRA matmul requires a constant left operand; use builder.linear() instead"
        )

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            return self.const_bool(value, output_wire=output_wire)
        return self.const(
            torch.tensor([float(value)], dtype=torch.float32), output_wire=output_wire
        )

    def const(self, tensor, output_wire=None) -> Term:
        return self._const(tensor, Real, output_wire)


class LIATermBuilder(TermBuilder):
    _logic = LIA

    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(Int(shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Bool(shape)
        return Int(shape)

    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b, op.int.Const, torch.int64)
        if result is not None:
            return result
        raise NonLinearError("LIA")

    def matmul(self, a, b) -> Term:
        a_wire, b_wire = _wire(a), _wire(b)
        out_shape = [_shape(a_wire.dtype)[0], _shape(b_wire.dtype)[1]]
        if isinstance(a, Term) and isinstance(a.itype, op.int.Const):
            no_bias = torch.empty(0, 0)
            return Term(
                ov.linear(_const_tensor(a.itype), no_bias, b_wire.dtype),
                [Wire(Int(out_shape))],
                [b_wire],
            )
        raise RuntimeError(
            "LIA matmul requires a constant left operand; use builder.linear() instead"
        )

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            return self.const_bool(value, output_wire=output_wire)
        tensor = torch.tensor([int(value)], dtype=torch.int64)
        return self.const(tensor, output_wire=output_wire)

    def const(self, tensor, output_wire=None) -> Term:
        return self._const(tensor, Int, output_wire)


class BVTermBuilder(TermBuilder):
    _logic = BV

    # FIXME: make 32 a parameter
    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(Word(32, shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Bool(shape)
        return Word(32, shape)

    def mul(self, a, b) -> Term:
        return self._binary_op(ov.mul(_dtype(a)), _dtype(a), a, b)

    # Comparisons default to unsigned (Sort carries no signedness; the smv parser
    # tracks signedness and picks U*/S* directly), and write booleans, as in QF_BV.

    def neg(self, a) -> Term:
        return self._unary_op(ov.neg, a)

    def matmul(self, a, b) -> Term:
        raise TheoryError("BV has no MatMul")

    def bit_select(self, a, high: int, low: int) -> Term:
        a_w = _wire(a)
        out = Wire(Word(high - low + 1, _shape(a_w.dtype)))
        return Term(ov.extract(a_w.dtype, high, low), [out], [a_w])

    def extend(self, a, extra: int) -> Term:
        a_w = _wire(a)
        out = Wire(Word(_bv_width(a_w.dtype) + extra, _shape(a_w.dtype)))
        return Term(ov.extend(a_w.dtype, extra), [out], [a_w])

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            tensor = torch.tensor([value], dtype=torch.bool)
        else:
            tensor = torch.tensor([int(value)], dtype=torch.int64)
        return self.const(tensor, output_wire=output_wire)

    def const(self, tensor, output_wire=None) -> Term:
        return self._const(tensor, lambda shape: Word(32, shape), output_wire)


def builder_for(theory=None) -> "TermBuilder":
    """Create the appropriate TermBuilder for the given theory.

    theory: LRA, LIA, BV, or None (defaults to LRA).
    """
    if theory is None or theory is LRA:
        return LRATermBuilder()
    if theory is LIA:
        return LIATermBuilder()
    if theory is BV:
        return BVTermBuilder()
    raise ValueError(f"Unknown theory: {theory!r}. Use LRA, LIA, or BV.")
