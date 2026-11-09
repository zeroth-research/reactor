// Two instances of one accumulator, each on its own clock: the clock
// ports are connected to different inputs of the top module.
module twodom (
  input  logic       clk_a, clk_b,
  input  logic [3:0] step,
  output logic [3:0] a, b,
  output logic       same
);
  tally ua (.clk(clk_a), .step, .q(a));
  tally ub (.clk(clk_b), .step(~step), .q(b));
  assign same = a == b;
endmodule

module tally (
  input  logic       clk,
  input  logic [3:0] step,
  output logic [3:0] q
);
  always_ff @(posedge clk) q <= q + step;
endmodule
