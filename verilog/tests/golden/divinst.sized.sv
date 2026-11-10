module half (
  input  logic clk,
  output logic s
);
  always_ff @(posedge clk) s <= (!s);
endmodule

module acc (
  input  logic clk,
  input  logic [3:0] a,
  output logic [3:0] q
);
  always_ff @(posedge clk) q <= (q + a);
endmodule

module divinst (
  input  logic clk,
  input  logic [3:0] a,
  output logic [3:0] q,
  output logic [3:0] y
);
  logic s;
  half h (.clk(clk), .s(s));
  acc u (.clk(s), .a(a), .q(q));
  acc v (.clk(clk), .a(q), .q(y));
endmodule

