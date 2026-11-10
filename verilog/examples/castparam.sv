// Size casts and typed parameters. `16'(u + u)` assigns the sum to a
// 16-bit value, so the sum is evaluated at 16 bits and keeps its carry;
// `8'(a + b)` truncates, and the cast's result is then extended like an
// operand (`h` sees no carry). A typed parameter takes its value as in an
// assignment: `S0 = 0` is a 4-bit zero, `M = -1` eight ones, `K = 4'd3`
// the `int` 3.
module castparam #(
  parameter [3:0] S0 = 0,
  localparam logic [7:0] M = -1,
  localparam int K = 4'd3
) (
  input  logic [7:0]  u, a, b,
  output logic [15:0] wide,
  output logic [8:0]  h,
  output logic [7:0]  m,
  output logic [3:0]  low, s,
  output logic [31:0] k
);
  assign wide = 16'(u + u);
  assign h = 8'(a + b) >> 1;
  assign low = 4'(a + b);
  assign m = M;
  assign s = S0 + 4'd5;
  assign k = K;
endmodule
