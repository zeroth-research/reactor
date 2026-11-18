module div (
  input  logic clk,
  output logic s,
  output logic [1:0] c
);
  always_ff @(posedge clk) s <= (!s);
  always_ff @(posedge s) c <= (c + 2'd1);
endmodule

module clkout (
  input  logic clk,
  input  logic [3:0] a,
  output logic s,
  output logic [1:0] c,
  output logic [3:0] q
);
  div d (.clk(clk), .s(s), .c(c));
  always_ff @(negedge s) q <= (q + a);
endmodule

