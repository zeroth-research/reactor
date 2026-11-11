module lane (
  input  logic clk,
  input  logic [2:0] k,
  input  logic d,
  output logic [7:0] q
);
  always_ff @(negedge clk) q[k] <= d;
endmodule

module lane_2 (
  input  logic clk,
  input  logic [1:0] k,
  input  logic d,
  output logic [3:0] q
);
  always_ff @(negedge clk) q[k] <= d;
endmodule

module wrinst (
  input  logic clk,
  input  logic [2:0] k,
  input  logic d,
  output logic [7:0] q8,
  output logic [3:0] q4
);
  lane a (.clk(clk), .k(k), .d(d), .q(q8));
  lane_2 b (.clk(clk), .k(k[1:0]), .d((!d)), .q(q4));
endmodule

