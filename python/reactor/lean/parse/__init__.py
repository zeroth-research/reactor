"""Reading a predicate's text.

`dsl` reads the Python predicate syntax -- `s0 > 0 and s1 == s4` -- over
whatever term algebra its namespace holds: `smt.predicates` hands it
cvc5's terms, and reads SMT-LIB with cvc5's own parser. Nothing here
imports cvc5.
"""
