// Counters on both edges of one clock: `p` counts rising edges, `n`
// falling edges, and `h` takes `p` at every falling edge, half a cycle
// after `p` changed.
module halves (
  input  logic       clk, en,
  output logic [3:0] p, n, h,
  output logic [4:0] total
);
  always_ff @(posedge clk)
    if (en) p <= p + 4'd1;
  always_ff @(negedge clk)
    if (en) n <= n + 4'd1;
  always_ff @(negedge clk)
    h <= p;
  assign total = {1'b0, p} + {1'b0, n};
endmodule
