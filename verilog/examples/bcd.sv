// Decimal digits of an 8-bit number, and signed division. Divisors are
// nonzero literals (a negative one written in two's complement, since
// `-8'sd3` is an expression), so no division by zero can happen. Signed division
// rounds toward zero, and the remainder has the sign of the dividend.
module bcd (
  input  logic        [7:0] a,
  input  logic signed [7:0] s,
  output logic        [3:0] hundreds, tens, ones,
  output logic signed [7:0] q, r
);
  logic [7:0] h, t, o;

  assign h = a / 8'd100;
  assign t = a / 8'd10 % 8'd10;
  assign o = a % 8'd10;
  assign {hundreds, tens, ones} = {h[3:0], t[3:0], o[3:0]};
  assign q = s / 8'shfd;  // -3
  assign r = s % 8'sd3;
endmodule
