(* blackbox *)
module sensor (
  input  logic clk,
  output logic [7:0] sample,
  output logic valid
);
endmodule

module filter (
  input  logic clk,
  input  logic rst,
  output logic [9:0] acc
);
  logic [7:0] s;
  logic v;
  sensor u (.clk(clk), .sample(s), .valid(v));
  always_ff @(posedge clk) if (rst) acc <= 10'd0; else if (v) acc <= (acc + 10'(10'(unsigned'(s)))); else begin end
endmodule

module filter_2 (
  input  logic clk,
  input  logic rst,
  output logic [7:0] acc
);
  logic [7:0] s;
  logic v;
  sensor u (.clk(clk), .sample(s), .valid(v));
  always_ff @(posedge clk) if (rst) acc <= 8'd0; else if (v) acc <= (acc + 8'(s)); else begin end
endmodule

(* blackbox *)
module osc (
  output logic clk
);
endmodule

module ticker (
  output logic [2:0] n
);
  logic c;
  logic [2:0] q;
  osc o (.clk(c));
  always_ff @(negedge c) q <= (q + 3'd1);
  assign n = q;
endmodule

module bbchild (
  input  logic clk,
  input  logic rst,
  output logic [9:0] sum,
  output logic [2:0] ticks
);
  logic [9:0] a;
  logic [7:0] b;
  logic [2:0] t1;
  logic [2:0] t2;
  filter f1 (.clk(clk), .rst(rst), .acc(a));
  filter_2 f2 (.clk(clk), .rst(rst), .acc(b));
  ticker k1 (.n(t1));
  ticker k2 (.n(t2));
  assign sum = (a + 10'(10'(unsigned'(b))));
  assign ticks = (t1 ^ t2);
endmodule

