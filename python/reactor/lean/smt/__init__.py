"""The module in cvc5, as the certificate's predicates are read against
it: the encoder (`encode`, `module`), the run's one encoding of the module
(`encoding`), the predicate parser (`predicates`), the claim and the
certificate read on it (`queries`), and walks over cvc5 terms (`terms`).

`encode`, `module` and `predicates` import cvc5; nothing else here does at
import time, and `encoding` imports them only when a run builds one. A
`reactor` run with no claim never loads the solver.
"""
