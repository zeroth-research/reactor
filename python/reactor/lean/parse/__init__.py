"""Reading a predicate's text.

`dsl` reads the Python predicate syntax -- `s0 > 0 and s1 == s4` -- over
whatever term algebra its namespace holds: `smt.predicates` hands it
cvc5's terms, and reads SMT-LIB with cvc5's own parser. `smtlib` reads
SMT-LIB with no solver, from the s-expressions `sexpr` reads, into
reactor's own terms, the IR a module is written in (`formula`), and
`native` is the parser that reads a predicate so: the Python syntax
first, over reactor's terms, as `smt.predicates` reads it over cvc5's.
Nothing here imports cvc5.
"""
