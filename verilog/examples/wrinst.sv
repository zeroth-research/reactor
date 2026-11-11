// A variable-index write in a parameterized child, on the falling edge.
module lane #(
    parameter W = 4
) (
    input  logic                 clk,
    input  logic [$clog2(W)-1:0] k,
    input  logic                 d,
    output logic [        W-1:0] q
);
  always_ff @(negedge clk) q[k] <= d;
endmodule

module wrinst (
    input  logic       clk,
    input  logic [2:0] k,
    input  logic       d,
    output logic [7:0] q8,
    output logic [3:0] q4
);
  lane #(.W(8)) a (
      .clk(clk),
      .k  (k),
      .d  (d),
      .q  (q8)
  );
  lane b (
      .clk(clk),
      .k  (k[1:0]),
      .d  (!d),
      .q  (q4)
  );
endmodule
