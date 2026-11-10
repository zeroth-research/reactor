module shiftreg (
  input  logic clk,
  input  logic [7:0] d,
  output logic [7:0] q
);
  logic [7:0] s1;
  logic [7:0] s2;
  logic [7:0] s3;
  always_ff @(posedge clk) begin s1 <= d; s2 <= s1; s3 <= s2; q <= s3; end
endmodule

