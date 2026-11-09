// A divider on the falling edges of `clk`, a register on the falling
// edges of the divided clock, and one on the rising edges of `clk`.
module negdiv (
  input  logic       clk,
  input  logic [3:0] d,
  output logic       slow,
  output logic [3:0] p, n
);
  always_ff @(negedge clk) slow <= !slow;
  always_ff @(negedge slow) n <= d;
  always_ff @(posedge clk) p <= p ^ d;
endmodule
