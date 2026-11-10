module alu (
  input  logic [1:0] op,
  input  logic [7:0] a,
  input  logic [7:0] b,
  output logic [7:0] y,
  output logic zero
);
  always_comb begin case (op) 2'd0: y = (a + b); 2'd1: y = (a - b); 2'd2: y = (a & b); default: y = (a ^ b); endcase zero = (y == 8'd0); end
endmodule

