// Selects with a variable index: one bit of `d` (also written `-:`, which
// with a variable index is only accepted with width 1, since a wider
// select would reach below bit 0 for index 0), and a 4-bit window of `w`
// starting at bit `k`. Every value of the index must stay inside the
// signal: `k` has 3 bits, so the window reaches at most bit 7 + 3 < 16.
// The checks go by the index width only, so a nibble is picked with a
// shift instead (`w[{sel, 2'b00} +: 4]` would be rejected: a 4-bit index
// can be 15).
module mux4 (
  input  logic [1:0]  sel,
  input  logic [2:0]  k,
  input  logic [3:0]  d,
  input  logic [15:0] w,
  output logic        y,
  output logic        y1,
  output logic [3:0]  win,
  output logic [3:0]  nib,
  output logic [3:0]  top
);
  assign y = d[sel];
  assign y1 = d[sel -: 1];
  assign win = w[k +: 4];
  assign nib = 4'(w >> {sel, 2'b00});
  assign top = w[15 -: 4];
endmodule
