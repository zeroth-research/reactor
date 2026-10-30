// Precedence of arithmetic, comparison, logical and conditional
// operators, signed arithmetic, and an unread input.
module arith (
  input  logic [7:0]        a, b, c,
  input  logic signed [7:0] s, t,
  input  logic              clk,
  output logic [7:0]        r,
  output logic signed [7:0] q,
  output logic              big
);
  assign r = a * b + c - a;
  assign q = -s * t + +s;
  assign big = a + b > c || s < t && a != 8'd0 ? 1'b1 : 1'b0;
endmodule
