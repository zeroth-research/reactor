"""The module in cvc5, and what everything that asks cvc5 about it
shares: the encoder (`encode`), the run's one encoding of the module
(`encoding`), the obligations stated once, walks over cvc5 terms, the
SMT-LIB printer, solver time limits and the artifact recorder.

`encode` and `module` import cvc5; nothing else here does at import time,
and `encoding` imports those two only when a run builds one. `cli` imports
`limits`, `main` imports `encoding`, and a bare `reactor` run never loads the
solver.
"""
