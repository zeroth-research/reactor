module ddr (
  input  logic clk,
  input  logic [3:0] d,
  output logic [3:0] y
);
  logic [3:0] p;
  logic [3:0] n;
  always_ff @(posedge clk) p <= d;
  always_ff @(negedge clk) n <= d;
  assign y = (clk ? p : n);
endmodule

