module leaf (
  input  logic clk,
  input  logic [3:0] d,
  output logic [3:0] q,
  output logic [3:0] sum
);
  always_ff @(posedge clk) q <= d;
  assign sum = (q + 4'd2);
endmodule

module mid (
  input  logic clk,
  input  logic [3:0] i,
  output logic [3:0] o
);
  logic [3:0] q;
  logic [3:0] s;
  leaf l (.clk(clk), .d((i ^ s)), .q(q), .sum(s));
  assign o = (s + q);
endmodule

module leaf_2 (
  input  logic clk,
  input  logic [3:0] d,
  output logic [3:0] q,
  output logic [3:0] sum
);
  always_ff @(posedge clk) q <= d;
  assign sum = (q + 4'd7);
endmodule

module mid_2 (
  input  logic clk,
  input  logic [3:0] i,
  output logic [3:0] o
);
  logic [3:0] q;
  logic [3:0] s;
  leaf_2 l (.clk(clk), .d((i ^ s)), .q(q), .sum(s));
  assign o = (s + q);
endmodule

module nest (
  input  logic clk,
  input  logic [3:0] a,
  output logic [3:0] y,
  output logic [3:0] z
);
  mid m1 (.clk(clk), .i(a), .o(y));
  mid_2 m2 (.clk(clk), .i(y), .o(z));
endmodule

