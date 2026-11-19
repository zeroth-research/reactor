"""Reading a predicate's text: the parsers a run's predicates go through.

There are two, and a run picks one (`PARSERS`, `reactor --parser`):

* `cvc5` reads into cvc5's terms (`smt.predicates`, `smt.queries`), which
  `proof.from_smt` prints;
* `native` reads into reactor's terms, the IR a module is written in
  (`native`), which `proof.from_ir` prints. It loads no solver.

Both read the same two dialects, the same way: the Python predicate syntax
first (`dsl`, walked over whichever term algebra the parser hands it), and
SMT-LIB for text that is no Python (cvc5's own parser, or `sexpr` and
`smtlib`). Nothing here imports cvc5.
"""

# The parsers a run can read its predicates with; the first is the default.
PARSERS = ("cvc5", "native")
