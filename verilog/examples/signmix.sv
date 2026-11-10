// Signed and unsigned operands. An operator is signed only if all its
// context-determined operands are; then every operand is extended by the
// sign of the operator, not by its own: in `s + u` the signed `s` is
// zero-extended. A decimal literal is signed (`s + 0` sign-extends `s`),
// a based one unsigned (`s + 'd0` does not). `$signed(u + u)` adds at
// 8 bits (its operand is self-determined) and sign-extends the sum. A
// comparison is signed only if both operands are: `s < u` compares
// unsigned, `s < 0` signed. `>>>` shifts in the sign only on a signed
// value; a concatenation is unsigned, and so is `(s >>> 1) + u`. In
// `c ? s : u` the arms are unsigned, in `c ? t : s` signed; the condition
// `u` is self-determined. A `case` sizes its selector and labels
// together: the signed 4-bit `t` is zero-extended to the unsigned label
// `8'd255`, so `t` = -1 does not match it.
module signmix (
  input  logic signed [7:0] s,
  input  logic signed [3:0] t,
  input  logic        [7:0] u,
  input  logic              c,
  output logic       [15:0] su, s0, sd0, sg, ash, cash, mash, div, cs, cw,
  output logic        [7:0] cc,
  output logic              lt, neg, m
);
  assign su = s + u;
  assign s0 = s + 0;
  assign sd0 = s + 'd0;
  assign sg = $signed(u + u);
  assign lt = s < u;
  assign neg = s < 0;
  assign ash = s >>> 1;
  assign cash = {s} >>> 1;
  assign mash = (s >>> 1) + u;
  assign div = s / 2;
  assign cs = c ? s : u;
  assign cw = c ? t : s;
  assign cc = u ? 8'd1 : 8'd2;
  always_comb
    case (t)
      8'd255: m = 1'b1;
      default: m = 1'b0;
    endcase
endmodule
