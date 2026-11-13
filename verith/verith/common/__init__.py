"""What more than one route uses and no route owns.

`encoding`: the run's encoding of the module, with the round a question is
stated over and the solvers it is asked on. `limits`: how long a question
may take. `obligations`: what `rule_buchi` and `rule_globally` ask, over
any frame and over the one a search asks them in (`Obligations`,
`Candidate`, `Question`), and the ones the routes state beyond
`rule_buchi`'s. `llm`: the model both LLM routes ask. `synth`: what a
searching route reads a module as -- its columns and readings, and the
constants and branch conditions a template may use. `options`: the options
two rows spell the same way, declared once. `claim`: a claim read against
the run's encoding for a search (`SynthContext`). `solvers`: cvc5 and
Vampire answering a `Question`. `vampire`: the Vampire driver. `smtlib`:
the SMT-LIB text a route writes, and the scripts. `terms`: reading cvc5
terms apart. `record`: what a route leaves in `artifacts/`.
"""
