// A barrel shifter: shift left, logical right, or arithmetic right by a
// variable amount. The amount is self-determined and unsigned; shifting by
// 8 or more gives 0 (or all copies of the sign bit for `>>>`). `>>>` is
// arithmetic only on a signed operand, so `s` is declared signed.
module barrel (
  input  logic        [1:0] op,
  input  logic        [3:0] n,
  input  logic        [7:0] a,
  input  logic signed [7:0] s,
  output logic        [7:0] y,
  output logic signed [7:0] z
);
  always_comb begin
    case (op)
      2'd0: y = a << n;
      2'd1: y = a >> n;
      2'd2: y = a <<< n;
      default: y = a >>> n;
    endcase
  end
  assign z = s >>> n;
endmodule
