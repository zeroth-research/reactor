"""A reactive module as Lean text.

`context` is what every encoding of one module shares, computed once;
`literals` prints its constants. `terms` translates a block's term list
into the `let` bindings of a Lean function -- over matrices or over flat
scalars. `encodings` prints the module's encodings (`ModuleToLean4`),
each tied to the functional one by equivalence theorems.
"""
