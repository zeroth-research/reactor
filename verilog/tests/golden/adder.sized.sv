module adder (
  input  logic [7:0] a,
  input  logic [7:0] b,
  output logic [7:0] sum,
  output logic [7:0] diff,
  output logic carry,
  output logic borrow
);
  assign carry = (sum < a);
  assign sum = (a + b);
  assign diff = (a - b);
  assign borrow = (a < b);
endmodule

