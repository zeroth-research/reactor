// Addition with carry in and carry out. The operands are widened to 9 bits
// explicitly, and the 9-bit sum is split by the concatenation on the left.
module addc (
  input  logic [7:0] a, b,
  input  logic       cin,
  output logic [7:0] s,
  output logic       cout
);
  assign {cout, s} = 9'(a) + 9'(b) + 9'(cin);
endmodule
