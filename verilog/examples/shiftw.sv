// Shifts. The shifted operand takes the width of the context, so
// `a << 2` at 16 bits loses no bits; `1 << n` shifts a 32-bit `int` and
// is truncated to 8 bits. The shift amount is self-determined: in
// `16'd1 << (n + 3'd7)` the amount wraps at 3 bits.
module shiftw (
  input  logic [7:0]  a,
  input  logic [2:0]  n,
  output logic [15:0] sl, sn, amt,
  output logic [7:0]  one
);
  assign sl = a << 2;
  assign sn = a << n;
  assign one = 1 << n;
  assign amt = 16'd1 << (n + 3'd7);
endmodule
