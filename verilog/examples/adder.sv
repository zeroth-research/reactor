// An 8-bit adder and subtractor with carry and borrow. Without
// concatenation, the carry is detected by wrap-around: the sum is smaller
// than an operand exactly when it overflowed.
module adder (
  input  logic [7:0] a, b,
  output logic [7:0] sum, diff,
  output logic       carry, borrow
);
  // before `sum` in the source: lowering puts the atoms in await order
  assign carry = sum < a;
  assign sum = a + b;
  assign diff = a - b;
  assign borrow = a < b;
endmodule
