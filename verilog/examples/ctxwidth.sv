// Context-determined and self-determined widths. An operator like `+` is
// evaluated at the width of its context, the largest width among its
// operands and the target of the assignment: `sum16` keeps the carry of
// the 8-bit addition, `sum4` keeps only the low 4 bits. `>>` passes the
// context down to its left operand, so `half16` shifts the carry in and
// `half8` does not. Reductions, `&&` and the items of `{…}` are
// self-determined: `red`, `land` and `cat` see the 8-bit sum. A
// comparison sizes both operands to the larger one: `carry` compares a
// 9-bit sum. `~a` at 16 bits inverts the extended operand.
module ctxwidth (
  input  logic [7:0]  a, b,
  input  logic        c,
  output logic [15:0] sum16, half16, cat, inv,
  output logic [7:0]  half8,
  output logic [3:0]  sum4,
  output logic        red, land, carry
);
  assign sum16 = a + b;
  assign sum4 = a + b;
  assign half16 = (a + b) >> 1;
  assign half8 = (a + b) >> 1;
  assign red = &(a + b);
  assign land = (a + b) && c;
  assign cat = {a + b};
  assign carry = a + b > 9'd255;
  assign inv = ~a;
endmodule
