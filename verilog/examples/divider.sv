// A clock divider: `slow` toggles at every rising edge of `clk`, so it
// rises every second cycle of `clk`, in the same time step as an edge of
// `clk`. At that edge `r` samples the counter `q`, which changes in the
// same step: `r` gets `q`'s value from before the edge of `clk` (hold
// times met), where a simulator gives the new one. `slow` is also an
// output: a clock read as data.
module divider (
  input  logic       clk,
  input  logic       en,
  output logic       slow,
  output logic [7:0] q, r
);
  always_ff @(posedge clk) begin
    slow <= !slow;
    if (en) q <= q + 8'd1;
  end
  always_ff @(posedge slow) r <= q;
endmodule
