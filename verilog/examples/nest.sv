// Three levels of hierarchy: `nest` instantiates two `mid`s, each of which
// instantiates a `leaf` register and adds a combinational path from its
// input through the leaf's output back out. The parameter K is passed down
// from `nest` to `mid` to `leaf`. Values flow combinationally through
// instance boundaries in both directions within one cycle.
module leaf #(parameter logic [3:0] K = 4'd1) (
  input  logic       clk,
  input  logic [3:0] d,
  output logic [3:0] q, sum
);
  always_ff @(posedge clk) q <= d;
  assign sum = q + K;
endmodule

module mid #(parameter logic [3:0] K = 4'd2) (
  input  logic       clk,
  input  logic [3:0] i,
  output logic [3:0] o
);
  logic [3:0] q, s;
  leaf #(.K(K)) l (.clk, .d(i ^ s), .q, .sum(s));
  assign o = s + q;
endmodule

module nest (
  input  logic       clk,
  input  logic [3:0] a,
  output logic [3:0] y, z
);
  mid m1 (.clk, .i(a), .o(y));
  mid #(.K(4'd7)) m2 (.clk, .i(y), .o(z));
endmodule
