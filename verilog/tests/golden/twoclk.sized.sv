module twoclk (
  input  logic clk_a,
  input  logic clk_b,
  input  logic en,
  output logic [7:0] qa,
  output logic [7:0] qb,
  output logic [7:0] sum
);
  always_ff @(posedge clk_a) if (en) qa <= (qa + 8'd1); else begin end
  always_ff @(posedge clk_b) qb <= (qb + 8'd3);
  assign sum = (qa + qb);
endmodule

