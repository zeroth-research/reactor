// Reductions, logical and bitwise operators, and an internal signal.
module flags (
  input  logic [7:0] x, y,
  output logic       zero, parity, all_ones, any_both, none, same,
  output logic [7:0] blend
);
  logic [7:0] both;
  assign both = x & y;
  assign any_both = |both;
  assign zero = ~|x;
  assign parity = ^x;
  assign all_ones = &x;
  assign none = !x && !y;
  assign same = &(x ~^ y);
  // `^` binds tighter than `|`
  assign blend = (x & ~y) | (~x & y) ^ 8'h0f;
endmodule
