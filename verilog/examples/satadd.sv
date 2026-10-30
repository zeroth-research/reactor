// Saturating addition. `sum` is assigned twice in one `always_comb`, and
// the reads in between see the first value; `y` and `sum` end up equal.
module satadd (
  input  logic [7:0] a, b,
  output logic [7:0] y,
  output logic       sat
);
  logic [7:0] sum;

  always_comb begin
    sum = a + b;
    sat = sum < a;
    if (sat) sum = 8'd255;
    y = sum;
  end
endmodule
