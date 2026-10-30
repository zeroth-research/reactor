// A 4-stage shift register in one block. With `<=` every stage reads the
// value of its predecessor before the edge, so the statement order does
// not matter.
module shiftreg (
  input  logic       clk,
  input  logic [7:0] d,
  output logic [7:0] q
);
  logic [7:0] s1, s2, s3;
  always_ff @(posedge clk) begin
    s1 <= d;
    s2 <= s1;
    s3 <= s2;
    q  <= s3;
  end
endmodule
