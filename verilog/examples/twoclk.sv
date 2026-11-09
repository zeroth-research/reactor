// Two counters on unrelated clocks. Neither register reads the other, so
// no value crosses between the clock domains; only the sum combines them.
module twoclk (
  input  logic       clk_a, clk_b, en,
  output logic [7:0] qa, qb, sum
);
  always_ff @(posedge clk_a)
    if (en) qa <= qa + 8'd1;
  always_ff @(posedge clk_b)
    qb <= qb + 8'd3;
  assign sum = qa + qb;
endmodule
