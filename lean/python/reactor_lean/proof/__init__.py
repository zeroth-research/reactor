"""The certificate as Lean.

`render` prints the certificate's files (`Certificate/Data.lean`,
`Certificate/Proofs.lean`, and the fixed `Certificate/Module.lean` and
`Certificate/Certificate.lean`), its predicates through `from_ir`, which
prints reactor's terms as a Lean expression. The obligations are stated in
`Proofs.lean` and proved by `sorry`, for a prover to replace.
"""
