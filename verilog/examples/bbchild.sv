// Blackboxes inside instantiated modules: each instance of `filter` has
// its own sensor, whose outputs are free in every time step; each `ticker`
// has its own oscillator, an input clock of its own.

(* blackbox *)
module sensor #(parameter int RATE = 1) (
  input  logic       clk,
  output logic [7:0] sample,
  output logic       valid
);
endmodule

(* blackbox *)
module osc (output logic clk);
endmodule

module filter #(parameter int W = 8) (
  input  logic         clk,
  input  logic         rst,
  output logic [W-1:0] acc
);
  logic [7:0] s;
  logic v;
  sensor #(.RATE(4)) u (.clk(clk), .sample(s), .valid(v));
  always_ff @(posedge clk)
    if (rst) acc <= '0;
    else if (v) acc <= acc + W'(s);
endmodule

module ticker (output logic [2:0] n);
  logic c;
  logic [2:0] q;
  osc o (.clk(c));
  always_ff @(negedge c) q <= q + 3'd1;
  assign n = q;
endmodule

module bbchild (
  input  logic       clk,
  input  logic       rst,
  output logic [9:0] sum,
  output logic [2:0] ticks
);
  logic [9:0] a;
  logic [7:0] b;
  logic [2:0] t1, t2;
  filter #(.W(10)) f1 (.clk(clk), .rst(rst), .acc(a));
  filter f2 (.clk(clk), .rst(rst), .acc(b));
  ticker k1 (.n(t1));
  ticker k2 (.n(t2));
  assign sum = a + 10'(b);
  assign ticks = t1 ^ t2;
endmodule
