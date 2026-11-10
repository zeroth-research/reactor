module rcount (
  input  logic clk,
  input  logic rst,
  output logic [3:0] q
);
  always_ff @(posedge clk or posedge rst) if (rst) q <= 4'd0; else q <= (q + 4'd1);
endmodule

module rcount_2 (
  input  logic clk,
  input  logic rst,
  output logic [5:0] q
);
  always_ff @(posedge clk or posedge rst) if (rst) q <= 6'd9; else q <= (q + 6'd1);
endmodule

module arinst (
  input  logic clk,
  input  logic rst,
  input  logic go,
  output logic [3:0] p,
  output logic [5:0] q
);
  logic go_n;
  logic vr;
  always_ff @(posedge clk) go_n <= (!go);
  assign vr = go_n;
  rcount u (.clk(clk), .rst(rst), .q(p));
  rcount_2 v (.clk(clk), .rst(vr), .q(q));
endmodule

