module memoor (
  input  logic clk,
  input  logic [2:0] i,
  input  logic [1:0] j,
  input  logic [3:0] d,
  output logic [3:0] x,
  output logic [3:0] y,
  output logic [3:0] z
);
  logic [3:0] m [0:5];
  always_ff @(posedge clk) m[i] <= d;
  assign x = m[j];
  assign y = m[5];
  assign z = (m[4] | m[{1'd0, j[0]}]);
endmodule

