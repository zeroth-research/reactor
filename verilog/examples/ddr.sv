// Double data rate: `d` is sampled at both edges of the clock, and `y`
// shows the value taken at the last edge, selected by the clock level.
module ddr (
  input  logic       clk,
  input  logic [3:0] d,
  output logic [3:0] y
);
  logic [3:0] p, n;
  always_ff @(posedge clk) p <= d;
  always_ff @(negedge clk) n <= d;
  assign y = clk ? p : n;
endmodule
