"""The certificate as Lean.

`render` prints the certificate's files (`Certificate/Data.lean`,
`Certificate/Proofs.lean`, and the fixed `Certificate/Module.lean` and
`Certificate/Certificate.lean`), its predicates through `from_smt`, which
prints a cvc5 term as a Lean expression. The obligations are stated in
`Proofs.lean` and proved by `sorry`, or by the scripts a prover hands over
(`render.ProofScripts`).
"""
