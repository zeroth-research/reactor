- Try using facts from Vampire to generate tactics; dump to artifacts/
- Prune selectively -- if Vampire proves inv but not ranking, try preserving that inv
- [later] Cooperative infer through artifacts: smt + vampire in parallel for 4 seconds; then one after another (with known artifacts) and nuterm; then ai-cegis; then ai-fix
- [later] Try vampire instead of lean-smt in hammer
- fbk-proveit: `_REVERT` / `_SMT_CLOSER` (`python/zrth/lean/infer/fbk_proveit/engine.py`) regex-patch the Lean proveit.py generates. A run warns when a patch finds nothing to change while the tactic it hooks onto is still there (2026-10-19). Revisit: fix it upstream in lean-ltl-certifying's template
- In Sygus, Houdini, ... (all that uses templates), allow generating the templates from the system -- simple way is to look at the definition of the system and guess the right template from that (linear, quadratic, using max/min, ...). More complex would be to unroll the transition relation (as SMT formula) several times and see what expressions we get and classify then what kind of template we might need.
- Nothing in CI builds a `verith -o` project, so a project that fails `lake build` goes unnoticed (fractional Real constants, 1ff16cb3..8a7816e5). Add a slow test: a few `limits` cases and one hybrid module through `verith -o`, then `lake build` (recipe: `python/tests/limits/README.md`).

-  The unsupported note is still never read. 6d396840 says an LLM resuming from artifacts/ would be told the shape fails. But ruled_out reads only no_solution, and the ai-cegis prompt presents everything it gets as "cvc5 proved this space empty". Telling the model would need a prompt section of its own, so that's your decision.
