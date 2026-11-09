// Two divider stages, `s1` on the rising edges of `clk` and `s2` on the
// falling edges of `s1`, and an accumulator on the rising edges of `s2`.
// The blocks are written against the order in which their registers
// change in a time step: the translation must order the atoms by the
// clocks they wait for.
module divchain (
  input  logic       clk,
  input  logic [3:0] a,
  output logic [3:0] y
);
  logic s1, s2;
  always_ff @(posedge s2) y <= y + a;
  always_ff @(negedge s1) s2 <= !s2;
  always_ff @(posedge clk) s1 <= !s1;
endmodule
