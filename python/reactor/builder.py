"""Stateless Term factories, one per numeric sort family.

``TermBuilder`` is the abstract base: it builds the terms shared by every numeric sort, the
structural ones and those on ``Bool`` values (``reactor.tensor.bool``), and raises
``TheoryError`` for the operations its numeric sort does not provide. Its subclasses,
``RealTermBuilder``, ``IntTermBuilder`` and ``WordTermBuilder``, provide the operations of
``reactor.tensor.real``, ``.int`` and ``.word``, sized by their operands' sort. Constants
are ``Bool`` for booleans, and otherwise of the builder's numeric sort, unless an output wire
gives their sort. ``builder_for`` makes the builder of a numeric sort family.
"""

from functools import partial

import torch

from .reactor import Wire, Term, Var
from .reactor.tensor import Skip, real, int as int_, bool as bool_, word
from .reactor.tensor.word import signed, unsigned
from .sort import Real, Int, Bool, Word, Zero
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


def _is_bool(t) -> bool:
    return isinstance(_dtype(t), Bool)


def _normalize_shape(shape: list) -> list:
    if len(shape) == 0:
        return [1, 1]
    if len(shape) == 1:
        return [1, shape[0]]
    return shape


class TheoryError(Exception):
    """An operation that no theory provides."""


def number_sort(number):
    """The numeric sort family `number` as a function from shapes to sorts: `Real` and
    `Int` as they are, `Word` as words of width 32 (or any such function itself)."""
    return partial(Word, 32) if number is Word else number


class TermBuilder:
    """Abstract base. Subclasses give the numeric sort, `number`, and its operations."""

    def number(self, shape: list) -> Real | Int | Word:
        """The numeric sort of the builder, of the given shape."""
        raise NotImplementedError

    # --- low-level helpers ---

    def _apply(self, op, *args, output_wire=None) -> Term:
        """The term of the operation `op` reading `args`, writing `output_wire`, or a wire
        of its codomain."""
        return Term(op, [output_wire or Wire(op.cod)], [_wire(a) for a in args])

    def _unary(self, op, a) -> Term:
        """The term of the operation class `op`, built from the sort of `a`, its domain."""
        return self._apply(op(_dtype(a)), a)

    def _binary(self, op, a, b) -> Term:
        """The term of the operation class `op`, built from the sorts of `a` and `b`, its
        domain."""
        return self._apply(op([_dtype(a), _dtype(b)]), a, b)

    def _compare(self, op, a, b) -> Term:
        """The term of the comparison `op` of `a` and `b`, writing booleans of their shape."""
        return self._binary(op, a, b)

    def _ite(self, op, cond, a, b, output_wire) -> Term:
        op = op([_dtype(cond), _dtype(a), _dtype(b)])
        return self._apply(op, cond, a, b, output_wire=output_wire)

    def _unsupported(self, method):
        raise TheoryError(f"{method.__name__} is not supported by {type(self).__name__}")

    # --- structural ops ---

    def id_(self, src, output_wire=None) -> Term:
        w = output_wire or Wire(_dtype(src))
        return Term(Skip(_dtype(src)), [w], [_wire(src)])

    def linear(self, x, weight, bias = None) -> list:
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
        linear = self._linear(weight, bias.reshape(out_features, 1), input_col_wire.dtype)
        t_lin = Term(linear, [linear_out_wire], [input_col_wire])

        t_out = self.transpose(linear_out_wire)
        return [t_in, t_lin, t_out]

    def _linear(self, weight, bias, dom):
        """The operation `weight @ x + bias`, for constant `weight` and `bias`, reading
        `x` of the sort `dom`."""
        self._unsupported(self.linear)

    def linear_symbolic(self, x, out_features: int):
        # Linear carries its weight/bias as *constant* tensors in the op, so a Linear
        # with symbolic (wire) weights is not expressible.
        raise NotImplementedError(
            "symbolic Linear (wire weights) is not expressible: theory Linear takes constant A, B"
        )

    # --- ops on booleans, shared by every numeric sort ---

    def ite(self, cond, a, b, output_wire=None) -> Term:
        if _is_bool(a):
            return self._ite(bool_.Ite, cond, a, b, output_wire)
        self._unsupported(self.ite)

    def not_(self, a) -> Term:
        return self._unary(bool_.Not, a)

    def and_(self, a, b) -> Term:
        return self._binary(bool_.And, a, b)

    def or_(self, a, b) -> Term:
        return self._binary(bool_.Or, a, b)

    def xor_(self, a, b) -> Term:
        return self._binary(bool_.Xor, a, b)

    def eq(self, a, b) -> Term:
        if _is_bool(a):
            return self._compare(bool_.Eq, a, b)
        self._unsupported(self.eq)

    def ne(self, a, b) -> Term:
        if _is_bool(a):
            return self._compare(bool_.Ne, a, b)
        self._unsupported(self.ne)

    # --- numeric ops (abstract) ---

    def add(self, a, b) -> Term:
        self._unsupported(self.add)

    def sub(self, a, b) -> Term:
        self._unsupported(self.sub)

    def mul(self, a, b) -> Term:
        self._unsupported(self.mul)

    def div(self, a, b) -> Term:
        self._unsupported(self.div)

    def neg(self, a) -> Term:
        self._unsupported(self.neg)

    def abs_(self, a) -> Term:
        self._unsupported(self.abs_)

    def relu(self, a) -> Term:
        self._unsupported(self.relu)

    def argmax(self, a) -> Term:
        self._unsupported(self.argmax)

    def transpose(self, a) -> Term:
        self._unsupported(self.transpose)

    def matmul(self, a, b) -> Term:
        self._unsupported(self.matmul)

    def lt(self, a, b) -> Term:
        self._unsupported(self.lt)

    def le(self, a, b) -> Term:
        self._unsupported(self.le)

    def gt(self, a, b) -> Term:
        self._unsupported(self.gt)

    def ge(self, a, b) -> Term:
        self._unsupported(self.ge)

    def bit_select(self, a, high: int, low: int) -> Term:
        self._unsupported(self.bit_select)

    def extend(self, a, extra: int) -> Term:
        self._unsupported(self.extend)

    # --- constants ---

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Bool(shape)
        return self.number(shape)

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
            sort = self.number(shape)
        if isinstance(sort, Bool):
            op = bool_.Const(tensor.to(torch.bool), sort)
        else:
            op = self._const(tensor, sort)
        return Term.constant(op, [output_wire or Wire(sort)])

    def _const(self, tensor, sort):
        """The constant `tensor`, of the numeric sort `sort`."""
        self._unsupported(self.const)

    # --- ops that no theory provides ---

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


class RealTermBuilder(TermBuilder):
    """Real numbers, with the operations of `reactor.tensor.real`."""

    def number(self, shape: list) -> Real:
        return Real(shape)

    def ite(self, cond, a, b, output_wire=None) -> Term:
        if _is_bool(a):
            return super().ite(cond, a, b, output_wire)
        return self._ite(real.Ite, cond, a, b, output_wire)

    def _degree(self, a, b) -> int:
        """The degree of both `a` and `b`, which their sum, difference, extremum or choice
        keeps, and which their comparison requires: reals of different degrees, rates of
        change of different orders, are not comparable."""
        p, q = _dtype(a).degree, _dtype(b).degree
        if p != q:
            raise TheoryError(f"cannot combine reals of degrees {p} and {q}")
        return p

    def _compare(self, op, a, b) -> Term:
        if isinstance(_dtype(a), Real):
            self._degree(a, b)
        return super()._compare(op, a, b)

    def _ite(self, op, cond, a, b, output_wire) -> Term:
        if isinstance(_dtype(a), Real):
            self._degree(a, b)
        return super()._ite(op, cond, a, b, output_wire)

    def add(self, a, b) -> Term:
        self._degree(a, b)
        return self._binary(real.Add, a, b)

    def sub(self, a, b) -> Term:
        self._degree(a, b)
        return self._binary(real.Sub, a, b)

    def mul(self, a, b) -> Term:
        # the degrees of the factors add up
        return self._binary(real.Mul, a, b)

    def div(self, a, b) -> Term:
        # the degree of the divisor is taken off that of the dividend, which it must not
        # exceed: a negative degree is not defined
        sa, sb = _dtype(a), _dtype(b)
        p, q = sa.degree, sb.degree
        if q > p:
            raise TheoryError(f"cannot divide a real of degree {p} by one of degree {q}")
        return self._binary(real.Div, a, b)

    def neg(self, a) -> Term:
        # the negation keeps the degree
        return self._unary(real.Neg, a)

    def abs_(self, a) -> Term:
        return self._unary(real.Abs, a)

    def relu(self, a) -> Term:
        return self._unary(real.ReLU, a)

    def argmax(self, a) -> Term:
        # the index is written as a plain [1, 1] real
        return self._unary(real.Argmax, a)

    def transpose(self, a) -> Term:
        return self._unary(real.Transpose, a)

    def matmul(self, a, b) -> Term:
        # the degrees of the factors add up, as for the pointwise product
        return self._binary(real.MatMul, a, b)

    def _linear(self, weight, bias, dom):
        return real.Linear(weight, bias, dom)

    def lt(self, a, b) -> Term:
        return self._compare(real.Lt, a, b)

    def le(self, a, b) -> Term:
        return self._compare(real.Le, a, b)

    def gt(self, a, b) -> Term:
        return self._compare(real.Gt, a, b)

    def ge(self, a, b) -> Term:
        return self._compare(real.Ge, a, b)

    def eq(self, a, b) -> Term:
        if _is_bool(a):
            return super().eq(a, b)
        return self._compare(real.Eq, a, b)

    def ne(self, a, b) -> Term:
        if _is_bool(a):
            return super().ne(a, b)
        return self._compare(real.Ne, a, b)

    def _const(self, tensor, sort):
        return real.Const(tensor.to(torch.float32), sort)


class IntTermBuilder(TermBuilder):
    """Integers, with the operations of `reactor.tensor.int`."""

    def number(self, shape: list) -> Int:
        return Int(shape)

    def ite(self, cond, a, b, output_wire=None) -> Term:
        if _is_bool(a):
            return super().ite(cond, a, b, output_wire)
        return self._ite(int_.Ite, cond, a, b, output_wire)

    def add(self, a, b) -> Term:
        return self._binary(int_.Add, a, b)

    def sub(self, a, b) -> Term:
        return self._binary(int_.Sub, a, b)

    def mul(self, a, b) -> Term:
        return self._binary(int_.Mul, a, b)

    def div(self, a, b) -> Term:
        return self._binary(int_.Div, a, b)

    def neg(self, a) -> Term:
        return self._unary(int_.Neg, a)

    def abs_(self, a) -> Term:
        return self._unary(int_.Abs, a)

    def relu(self, a) -> Term:
        return self._unary(int_.ReLU, a)

    def argmax(self, a) -> Term:
        # the index is written as a [1, 1] matrix of the operand's sort
        return self._unary(int_.Argmax, a)

    def transpose(self, a) -> Term:
        return self._unary(int_.Transpose, a)

    def matmul(self, a, b) -> Term:
        return self._binary(int_.MatMul, a, b)

    def _linear(self, weight, bias, dom):
        return int_.Linear(weight, bias, dom)

    def lt(self, a, b) -> Term:
        return self._compare(int_.Lt, a, b)

    def le(self, a, b) -> Term:
        return self._compare(int_.Le, a, b)

    def gt(self, a, b) -> Term:
        return self._compare(int_.Gt, a, b)

    def ge(self, a, b) -> Term:
        return self._compare(int_.Ge, a, b)

    def eq(self, a, b) -> Term:
        if _is_bool(a):
            return super().eq(a, b)
        return self._compare(int_.Eq, a, b)

    def ne(self, a, b) -> Term:
        if _is_bool(a):
            return super().ne(a, b)
        return self._compare(int_.Ne, a, b)

    def _const(self, tensor, sort):
        return int_.Const(tensor.to(torch.int64), sort)


class WordTermBuilder(TermBuilder):
    """Words of `width` bits (32 by default), with the operations of `reactor.tensor.word`,
    sized by the width of their operands too. Division and comparisons are unsigned,
    absolute values signed. On words, the logical operations are bitwise."""

    def __init__(self, width: int = 32):
        self.width = width

    def number(self, shape: list) -> Word:
        return Word(self.width, shape)

    def ite(self, cond, a, b, output_wire=None) -> Term:
        if _is_bool(a):
            return super().ite(cond, a, b, output_wire)
        return self._ite(word.Ite, cond, a, b, output_wire)

    def not_(self, a) -> Term:
        return super().not_(a) if _is_bool(a) else self._unary(word.Not, a)

    def and_(self, a, b) -> Term:
        return super().and_(a, b) if _is_bool(a) else self._binary(word.And, a, b)

    def or_(self, a, b) -> Term:
        return super().or_(a, b) if _is_bool(a) else self._binary(word.Or, a, b)

    def xor_(self, a, b) -> Term:
        return super().xor_(a, b) if _is_bool(a) else self._binary(word.Xor, a, b)

    def add(self, a, b) -> Term:
        return self._binary(word.Add, a, b)

    def sub(self, a, b) -> Term:
        return self._binary(word.Sub, a, b)

    def mul(self, a, b) -> Term:
        return self._binary(word.Mul, a, b)

    def div(self, a, b) -> Term:
        return self._binary(unsigned.Div, a, b)

    def neg(self, a) -> Term:
        return self._unary(word.Neg, a)

    def abs_(self, a) -> Term:
        return self._unary(signed.Abs, a)

    def lt(self, a, b) -> Term:
        return self._compare(unsigned.Lt, a, b)

    def le(self, a, b) -> Term:
        return self._compare(unsigned.Le, a, b)

    def gt(self, a, b) -> Term:
        return self._compare(unsigned.Gt, a, b)

    def ge(self, a, b) -> Term:
        return self._compare(unsigned.Ge, a, b)

    def eq(self, a, b) -> Term:
        if _is_bool(a):
            return super().eq(a, b)
        return self._compare(word.Eq, a, b)

    def ne(self, a, b) -> Term:
        if _is_bool(a):
            return super().ne(a, b)
        return self._compare(word.Ne, a, b)

    def bit_select(self, a, high: int, low: int) -> Term:
        return self._apply(word.Extract(high, low, _dtype(a)), a)

    def extend(self, a, extra: int) -> Term:
        return self._apply(unsigned.Extend(extra, _dtype(a)), a)

    def _const(self, tensor, sort):
        return word.Const(tensor.to(torch.int64), sort)


def builder_for(number=Real) -> TermBuilder:
    """The term builder of the numeric sort family `number`: `Real` (the default), `Int`,
    or `Word` (of width 32)."""
    if number is Real:
        return RealTermBuilder()
    if number is Int:
        return IntTermBuilder()
    if number is Word:
        return WordTermBuilder()
    raise ValueError(f"unknown numeric sort: {number!r}; use Real, Int or Word")
