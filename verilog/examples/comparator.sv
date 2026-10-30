// All comparisons, on unsigned and on signed operands.
module comparator (
  input  logic [7:0]        a, b,
  input  logic signed [7:0] sa, sb,
  output logic eq, ne, lt, le, gt, ge,
  output logic slt, sle, sgt, sge
);
  assign eq = a == b;
  assign ne = a != b;
  assign lt = a < b;
  assign le = a <= b;
  assign gt = a > b;
  assign ge = a >= b;
  assign slt = sa < sb;
  assign sle = sa <= sb;
  assign sgt = sa > sb;
  assign sge = sa >= sb;
endmodule
