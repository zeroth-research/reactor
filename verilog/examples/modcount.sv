// A counter modulo N with a parameterized width: it counts 0, 1, …, N-1,
// 0, … while `en` is set and raises `wrap` in the cycle it is at N-1.
// The width is derived from N with `$clog2`. The integer constant N-1
// is an `int` (signed, 32 bits); the typed local parameter LAST holds it
// at the width and signedness of the counter.
module modcount #(
  parameter int N = 10,
  localparam int W = $clog2(N)
) (
  input  logic         clk, en,
  output logic [W-1:0] q,
  output logic         wrap
);
  localparam logic [W-1:0] LAST = W'(N - 1), ZERO = W'(0), ONE = W'(1);
  assign wrap = q == LAST;
  always_ff @(posedge clk)
    if (en) q <= wrap ? ZERO : q + ONE;
endmodule
