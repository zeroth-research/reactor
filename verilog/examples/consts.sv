// Constant expressions use the whole expression syntax (comments
// included) and IEEE 1800's rules. A range or an untyped parameter made
// of integers only is exact and must fit an `int`: `P` is the least one.
// Other operators follow the width rules: `W - 1` with the typed `W` is a
// 32-bit unsigned subtraction, `H` is multiplied at the 8 bits of its
// type. In a value expression a constant is evaluated at the width of its
// context: `65536 * 65536` and `2 ** 31` at 64 bits, `(-3) ** 3` at the
// 40 bits of `b`, so it adds 2^40 - 27.
module consts #(
  parameter N = 9 /* bits */ - 1,
  parameter [4:0] W = 16,
  localparam S = 3,
  localparam L = 1 << S,
  localparam K = S > 2 ? 1 : 0,
  localparam C = $clog2(W),
  localparam P = -2147483648,
  localparam [7:0] H = 4'd15 * 4'd15,
  localparam [7:0] J = {4'd9, 4'd2}
) (
  input  logic [N /* width */ - 1:0] a,
  input  logic [39:0] b,
  output logic [W-1:0] w,
  output logic [L-1:0] l,
  output logic [K:0] k,
  output logic [C:0] c,
  output logic [7:0] h,
  output logic [63:0] big, pow, least,
  output logic [39:0] negpow
);
  assign w = a + J;
  assign l = ~a;
  assign k = a[K:0];
  assign c = a[C:0];
  assign h = H + a;
  assign big = 65536 * 65536;
  assign pow = 2 ** 31;
  assign least = P;
  assign negpow = b + (-3) ** 3;
endmodule
