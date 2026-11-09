// A divider in an instance: its output `s` clocks the accumulator `u`;
// `v` on the input clock accumulates `u`'s register (a read between
// related clocks).
module divinst (
  input  logic       clk,
  input  logic [3:0] a,
  output logic [3:0] q, y
);
  logic s;
  half h (.clk, .s);
  acc u (.clk(s), .a, .q);
  acc v (.clk, .a(q), .q(y));
endmodule

module half (
  input  logic clk,
  output logic s
);
  always_ff @(posedge clk) s <= !s;
endmodule

module acc (
  input  logic       clk,
  input  logic [3:0] a,
  output logic [3:0] q
);
  always_ff @(posedge clk) q <= q + a;
endmodule
