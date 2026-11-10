module comparator (
  input  logic [7:0] a,
  input  logic [7:0] b,
  input  logic signed [7:0] sa,
  input  logic signed [7:0] sb,
  output logic eq,
  output logic ne,
  output logic lt,
  output logic le,
  output logic gt,
  output logic ge,
  output logic slt,
  output logic sle,
  output logic sgt,
  output logic sge
);
  assign eq = (a == b);
  assign ne = (a != b);
  assign lt = (a < b);
  assign le = (a <= b);
  assign gt = (a > b);
  assign ge = (a >= b);
  assign slt = (sa < sb);
  assign sle = (sa <= sb);
  assign sgt = (sa > sb);
  assign sge = (sa >= sb);
endmodule

