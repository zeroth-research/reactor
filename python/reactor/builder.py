"""A stateless Term factory.

``TermBuilder`` makes terms from wires, with the operations of the namespace of their
operands' sort (``reactor.tensor.real``, ``.int``, ``.bool`` or ``.word``), sized by their
shapes. Constants are ``Bool`` for booleans, and otherwise of the builder's numeric sort,
``number`` (``Real`` by default, or ``Int``, or ``Word`` of width 32), unless an output wire
gives their sort.
"""

from functools import partial

import torch

from .reactor import Wire, Term, Var
from .reactor import tensor
from .reactor.tensor import Skip
from .sort import Real, Int, Bool, Word, Zero, namespace
from .sort import shape as _shape, with_shape as _with_shape, width as _width


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


def _dtype(t) -> Real | Int | Bool | Word | Zero:
    return _wire(t).dtype


def _normalize_shape(shape: list) -> list:
    if len(shape) == 0:
        return [1, 1]
    if len(shape) == 1:
        return [1, shape[0]]
    return shape


def _sized(name, sort):
    """The operation `name` of the namespace of `sort`, sized by it."""
    match sort:
        case Word(w, s):
            return getattr(tensor.word, name)(w, s)
    return getattr(namespace(sort), name)(_shape(sort))


# The element kind of the constants of each sort.
_TORCH_DTYPE = {Bool: torch.bool, Real: torch.float32, Int: torch.int64, Word: torch.int64}


class TheoryError(Exception):
    """An operation that no theory provides."""


def number_sort(number):
    """The numeric sort family `number` as a function from shapes to sorts: `Real` and
    `Int` as they are, `Word` as words of width 32 (or any such function itself)."""
    return partial(Word, 32) if number is Word else number


class TermBuilder:
    def __init__(self, number=Real):
        self._number = number_sort(number)

    # --- low-level helpers ---

    def _binary(self, op, output_sort: Real | Int | Bool | Word | Zero, a, b) -> Term:
        return Term(op, [Wire(output_sort)], [_wire(a), _wire(b)])

    def _unary(self, op, output_sort: Real | Int | Bool | Word | Zero, a) -> Term:
        return Term(op, [Wire(output_sort)], [_wire(a)])

    # --- structural and boolean ops ---

    def id_(self, src, output_wire=None) -> Term:
        w = output_wire or Wire(_dtype(src))
        return Term(Skip(_dtype(src)), [w], [_wire(src)])

    def ite(self, cond, a, b, output_wire=None) -> Term:
        w = output_wire or Wire(_dtype(a))
        return Term(_sized("Ite", _dtype(a)), [w], [_wire(cond), _wire(a), _wire(b)])

    def not_(self, a) -> Term:
        return self._unary(_sized("Not", _dtype(a)), _dtype(a), a)

    def and_(self, a, b) -> Term:
        return self._binary(_sized("And", _dtype(a)), _dtype(a), a, b)

    def or_(self, a, b) -> Term:
        return self._binary(_sized("Or", _dtype(a)), _dtype(a), a, b)

    def xor_(self, a, b) -> Term:
        return self._binary(_sized("Xor", _dtype(a)), _dtype(a), a, b)

    # --- arithmetic ---

    def add(self, a, b) -> Term:
        return self._binary(_sized("Add", _dtype(a)), _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary(_sized("Sub", _dtype(a)), _dtype(a), a, b)

    def mul(self, a, b) -> Term:
        return self._binary(_sized("Mul", _dtype(a)), _dtype(a), a, b)

    def div(self, a, b) -> Term:
        s = _dtype(a)
        op = tensor.word.unsigned.Div(_width(s), _shape(s)) if isinstance(s, Word) else _sized("Div", s)
        return self._binary(op, s, a, b)

    def neg(self, a) -> Term:
        return self._unary(_sized("Neg", _dtype(a)), _dtype(a), a)

    def abs_(self, a) -> Term:
        s = _dtype(a)
        return self._unary(tensor.word.signed.Abs(_width(s), _shape(s)), s, a)

    def relu(self, a) -> Term:
        return self._unary(_sized("ReLU", _dtype(a)), _dtype(a), a)

    def argmax(self, a) -> Term:
        # the index is written as a [1, 1] matrix of the operand's sort
        return self._unary(_sized("Argmax", _dtype(a)), _with_shape(_dtype(a), [1, 1]), a)

    def transpose(self, a) -> Term:
        s = _shape(_dtype(a))
        return self._unary(_sized("Transpose", _dtype(a)), _with_shape(_dtype(a), [s[1], s[0]]), a)

    def matmul(self, a, b) -> Term:
        sa, sb = _dtype(a), _dtype(b)
        out = _with_shape(sa, [_shape(sa)[0], _shape(sb)[1]])
        return self._binary(namespace(sa).MatMul(_shape(sa), _shape(sb)), out, a, b)

    def linear(self, x, weight, bias) -> list:
        """Emit [Transpose-in, Linear, Transpose-out]. Last write[0] is the output wire."""
        x_wire = _wire(x)
        in_features = _shape(x_wire.dtype)[-1]  # [1, in_features] row-major
        out_features = weight.shape[0]

        input_col_wire = Wire(_with_shape(x_wire.dtype, [in_features, 1]))
        t_in = self.transpose(x_wire)
        t_in = Term(t_in.itype, [input_col_wire], [x_wire])

        linear_out_wire = Wire(_with_shape(x_wire.dtype, [out_features, 1]))
        if bias is None or bias.numel() == 0:
            bias = torch.zeros(out_features, 1, dtype=weight.dtype)
        linear = namespace(x_wire.dtype).Linear(weight, bias.reshape(out_features, 1))
        t_lin = Term(linear, [linear_out_wire], [input_col_wire])

        t_out = self.transpose(linear_out_wire)
        return [t_in, t_lin, t_out]

    def linear_symbolic(self, x, out_features: int):
        # Linear carries its weight/bias as *constant* tensors in the op, so a Linear
        # with symbolic (wire) weights is not expressible.
        raise NotImplementedError(
            "symbolic Linear (wire weights) is not expressible: theory Linear takes constant A, B"
        )

    # --- comparisons: write booleans of the operands' shape ---

    def _compare(self, name, a, b) -> Term:
        s = _dtype(a)
        if isinstance(s, Word):
            op = getattr(tensor.word.unsigned, name)(_width(s), _shape(s))
        else:
            op = _sized(name, s)
        return self._binary(op, Bool(_shape(s)), a, b)

    def lt(self, a, b) -> Term:
        return self._compare("Lt", a, b)

    def le(self, a, b) -> Term:
        return self._compare("Le", a, b)

    def gt(self, a, b) -> Term:
        return self._compare("Gt", a, b)

    def ge(self, a, b) -> Term:
        return self._compare("Ge", a, b)

    def eq(self, a, b) -> Term:
        return self._binary(_sized("Eq", _dtype(a)), Bool(_shape(_dtype(a))), a, b)

    def ne(self, a, b) -> Term:
        return self._binary(_sized("Ne", _dtype(a)), Bool(_shape(_dtype(a))), a, b)

    # --- words ---

    def bit_select(self, a, high: int, low: int) -> Term:
        s = _dtype(a)
        return self._unary(tensor.word.Extract(high, low, _width(s), _shape(s)), Word(high - low + 1, _shape(s)), a)

    def extend(self, a, extra: int) -> Term:
        s = _dtype(a)
        op = tensor.word.unsigned.Extend(extra, _width(s), _shape(s))
        return self._unary(op, Word(_width(s) + extra, _shape(s)), a)

    # --- constants ---

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Bool(shape)
        return self._number(shape)

    def const_bool(self, value: bool, output_wire=None) -> Term:
        return self.const(torch.tensor([[bool(value)]], dtype=torch.bool), output_wire=output_wire)

    def const_for_value(self, value, output_wire=None) -> Term:
        return self.const(torch.tensor([value]), output_wire=output_wire)

    def const(self, tensor, output_wire=None) -> Term:
        shape = _normalize_shape(list(tensor.size()))
        tensor = tensor.reshape(shape)  # theory const ops require a 2-D initializer
        if output_wire is not None:
            sort = output_wire.dtype
        elif tensor.dtype == torch.bool:
            sort = Bool(shape)
        else:
            sort = self._number(shape)
        tensor = tensor.to(_TORCH_DTYPE[type(sort)])
        ns = namespace(sort)
        op = ns.Const(tensor, _width(sort)) if isinstance(sort, Word) else ns.Const(tensor)
        return Term.constant(op, [output_wire or Wire(sort)])

    # --- ops that no theory provides --------------------------------------------

    def uninterpreted(self, name: str, sort: Real | Int | Bool | Word | Zero) -> Term:
        raise TheoryError(f"uninterpreted functions are not supported (`{name}`)")

    def sin(self, a) -> Term:
        raise TheoryError("sin is not supported by any theory")

    def cos(self, a) -> Term:
        raise TheoryError("cos is not supported by any theory")

    def tanh(self, a) -> Term:
        raise TheoryError("tanh is not supported by any theory")

    def pow(self, base, exponent) -> Term:
        raise TheoryError("** / pow is not supported by any theory")

    def stack(self, elements) -> Term:
        raise TheoryError("no theory provides Stack (list-valued state)")

    def tensor_get(self, base, index) -> Term:
        raise TheoryError("no theory provides TensorGet (indexing)")
