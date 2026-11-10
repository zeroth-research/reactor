module accum (
  input  logic clk,
  input  logic clear,
  input  logic [7:0] x,
  output logic [7:0] sum,
  output logic overflow
);
  logic [7:0] acc;
  logic [7:0] next;
  logic carry;
  assign next = (acc + x);
  assign carry = (next < acc);
  always_ff @(posedge clk) if (clear) acc <= 8'd0; else acc <= next;
  always_ff @(posedge clk) overflow <= (clear ? 1'd0 : (carry || overflow));
  assign sum = acc;
endmodule

