"""Reading a predicate's text into reactor's terms, the IR a module is
written in (`native`), which `proof.from_ir` prints. It loads no solver.

It reads two dialects: the Python predicate syntax first (`dsl`), and
SMT-LIB for text that is no Python (`sexpr` and `smtlib`).
"""

