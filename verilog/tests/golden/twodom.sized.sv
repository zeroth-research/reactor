module tally (
  input  logic clk,
  input  logic [3:0] step,
  output logic [3:0] q
);
  always_ff @(posedge clk) q <= (q + step);
endmodule

module twodom (
  input  logic clk_a,
  input  logic clk_b,
  input  logic [3:0] step,
  output logic [3:0] a,
  output logic [3:0] b,
  output logic same
);
  tally ua (.clk(clk_a), .step(step), .q(a));
  tally ub (.clk(clk_b), .step((~step)), .q(b));
  assign same = (a == b);
endmodule

