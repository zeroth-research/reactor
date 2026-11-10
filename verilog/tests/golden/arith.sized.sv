module arith (
  input  logic [7:0] a,
  input  logic [7:0] b,
  input  logic [7:0] c,
  input  logic signed [7:0] s,
  input  logic signed [7:0] t,
  input  logic clk,
  output logic [7:0] r,
  output logic signed [7:0] q,
  output logic big
);
  assign r = (((a * b) + c) - a);
  assign q = (((-s) * t) + (+s));
  assign big = ((((a + b) > c) || ((s < t) && (a != 8'd0))) ? 1'd1 : 1'd0);
endmodule

