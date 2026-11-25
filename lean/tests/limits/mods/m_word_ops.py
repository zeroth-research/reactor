"""Every word op that has a Lean form only since the circuit encoding went:
the inverted bitwise ops, unsigned and signed division and comparison,
shifts, rotations, and the ops that change a word's width, on 1×1 words
and entry by entry on 2×1 ones.

Each op reads the inputs, drawn afresh every round, so the divisions meet
a zero divisor as often as a draw is zero. The results are folded into the
state by `Xor`: `x` the 1×1 ones, `v` the 2×1 ones, `n` the 8-bit ones;
`lo` and `ls` are the unsigned and signed comparisons of the vectors.
`tests/agree.py` holds each op alone to the interpreter, at 1×1.
"""
import torch
from reactor import Bool, Module, Term, Var, Wire, Word, X, tensor

W, S, V = 4, [1, 1], [2, 1]
word, unsigned, signed = tensor.word, tensor.word.unsigned, tensor.word.signed

# The ops of two words of width `W` and a shape, to one of the same.
BINARY = (word.Nand, word.Nor, word.Xnor, unsigned.Div, signed.Div,
          signed.Rem, word.Shl, unsigned.Shr, signed.Shr)


def _fold(sort, ops, out):
    """`ops` -- (op, reads) pairs -- each written to a wire of `sort`, and
    their `Xor` written to `out`."""
    width, shape = sort
    terms, acc = [], None
    for op, reads in ops:
        r = Wire(Word(width, shape))
        terms.append(Term(op, [r], reads))
        if acc is not None:
            nxt = Wire(Word(width, shape))
            terms.append(Term(word.Xor(width, shape), [nxt], [acc, r]))
            r = nxt
        acc = r
    terms.append(Term(tensor.Skip(Word(width, shape)), [out], [acc]))
    return terms


def module() -> Module:
    x, v, n = Var(Word(W, S)), Var(Word(W, V)), Var(Word(2 * W, S))
    lo, ls = Var(Bool(V)), Var(Bool(V))
    d, f = Var(Word(W, S)), Var(Word(W, S))
    e, g = Var(Word(W, V)), Var(Word(W, V))

    init = [
        Term(word.Const(torch.tensor([[5]]), W), [X(x)]),
        Term(word.Const(torch.tensor([[3], [12]]), W), [X(v)]),
        Term(word.Const(torch.tensor([[7]]), 2 * W), [X(n)]),
        Term(tensor.bool.Const(torch.tensor([[True], [False]])), [X(lo)]),
        Term(tensor.bool.Const(torch.tensor([[False], [True]])), [X(ls)]),
    ]

    low_d, low_x = Wire(Word(2, S)), Wire(Word(2 * W - 2, S))
    update = [
        *_fold((W, S), [(op(W, S), [X(d), X(f)]) for op in BINARY]
               + [(word.RotateLeft(1, W, S), [X(d)]),
                  (word.RotateRight(3, W, S), [X(f)])], X(x)),
        *_fold((W, V), [(op(W, V), [X(e), X(g)]) for op in BINARY]
               + [(word.RotateLeft(2, W, V), [X(e)])], X(v)),
        Term(unsigned.Lt(W, V), [X(lo)], [X(e), X(g)]),
        Term(signed.Ge(W, V), [X(ls)], [X(e), X(g)]),
        Term(word.Extract(2, 1, W, S), [low_d], [X(d)]),
        Term(word.Extract(5, 0, 2 * W, S), [low_x], [n]),
        *_fold((2 * W, S), [
            (unsigned.Extend(W, W, S), [X(d)]),
            (signed.Extend(W, W, S), [X(f)]),
            (word.Repeat(2, W, S), [X(d)]),
            (word.Concat([2, 2 * W - 2], S), [low_d, low_x]),
        ], X(n)),
    ]
    return Module.sequential([x, v, n, lo, ls, d, f, e, g], init, update)
