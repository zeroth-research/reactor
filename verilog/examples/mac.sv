// A signed multiply-accumulate unit. The 8-bit operands are widened with
// `16'(…)` before the multiplication, which keeps their sign; widening
// `16'(a * b)` instead would be rejected, since the standard evaluates the
// product at 16 bits there.
module mac (
  input  logic               clk,
  input  logic               clear,
  input  logic signed [7:0]  a, b,
  output logic signed [15:0] acc,
  output logic        [7:0]  hi
);
  always_ff @(posedge clk) begin
    if (clear) acc <= 16'sd0;
    else acc <= acc + 16'(a) * 16'(b);
  end
  assign hi = acc[15:8];
endmodule
