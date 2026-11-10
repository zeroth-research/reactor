module carry (
  input  logic clk,
  input  logic cin,
  input  logic [7:0] a,
  input  logic [7:0] b,
  output logic [7:0] sum,
  output logic cout,
  output logic [3:0] cnt
);
  assign {cout, sum} = ((9'(unsigned'(a)) + 9'(unsigned'(b))) + 9'(unsigned'(cin)));
  always_ff @(posedge clk) if ((32'(unsigned'(cnt)) == 32'd9)) cnt <= 4'sd0; else cnt <= 4'((32'(unsigned'(cnt)) + 32'd1));
endmodule

