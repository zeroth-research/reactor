// A parameterized memory in a child, with the address width from its
// depth, instantiated twice with different sizes.
module mem #(
    parameter DEPTH = 4,
    parameter W = 4
) (
    input  logic                     clk,
    input  logic                     we,
    input  logic [$clog2(DEPTH)-1:0] wa,
    input  logic [$clog2(DEPTH)-1:0] ra,
    input  logic [W-1:0]             d,
    output logic [W-1:0]             q
);
  logic [W-1:0] m[DEPTH];
  always_ff @(posedge clk) if (we) m[wa] <= d;
  assign q = m[ra];
endmodule

module meminst (
    input  logic       clk,
    input  logic       we,
    input  logic [2:0] wa,
    input  logic [2:0] ra,
    input  logic [5:0] d,
    output logic [5:0] q8,
    output logic [3:0] q2
);
  mem #(.DEPTH(8), .W(6)) deep (
      .clk(clk),
      .we (we),
      .wa (wa),
      .ra (ra),
      .d  (d),
      .q  (q8)
  );
  mem #(.DEPTH(2)) shallow (
      .clk(clk),
      .we (we),
      .wa (wa[0]),
      .ra (ra[0]),
      .d  (d[3:0]),
      .q  (q2)
  );
endmodule
