# Mechanical, meaning-preserving rewrites applied for the "rewritten" column.
# Each one only changes spelling that our subset rejects but that means the same
# on synthesizable code (our typing still checks the rest, e.g. that an always_ff
# uses only <= and an always_comb only =).
# 1. `reg` is the same 2-state vector as `logic` in our subset.
s/\breg\b/logic/g
# 2. edge-triggered plain `always` = `always_ff` (incl. `or`-ed async reset events).
s/always[ \t]*@[ \t]*\([ \t]*(posedge|negedge)/always_ff @(\1/
# 3. `always @(*)` / `always @*` = `always_comb`.
s/always[ \t]*@[ \t]*\([ \t]*\*[ \t]*\)/always_comb/
s/always[ \t]*@[ \t]*\*/always_comb/
