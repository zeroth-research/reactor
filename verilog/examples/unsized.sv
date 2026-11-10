// Unsized literals. A decimal `1` is an `int` (signed, 32 bits), so `q + 1`
// is evaluated at 32 bits and `(q + 1) == 0` never holds for a 4-bit `q`,
// while `(q + 4'd1) == 4'd0` holds when `q` is 15. An unsized based literal
// (`'d3`) is unsigned, 32 bits. `'0` and `'1` fill the width of their
// context with zeros or ones; `x == '1` tests for all ones. In a context of
// 40 bits, `'h7fffffff + 'h7fffffff + 2` keeps its carry.
module unsized (
  input  logic [3:0]  q,
  input  logic [7:0]  x,
  input  logic        c,
  output logic [3:0]  inc,
  output logic        wrapped, wrapped4, full,
  output logic [7:0]  ones, zeros, pick, based,
  output logic [39:0] big
);
  assign inc = q + 1;
  assign wrapped = q + 1 == 0;
  assign wrapped4 = q + 4'd1 == 4'd0;
  assign ones = '1;
  assign zeros = '0;
  assign pick = c ? x : '1;
  assign full = x == '1;
  assign based = x + 'd3;
  assign big = 'h7fffffff + 'h7fffffff + 2;
endmodule
