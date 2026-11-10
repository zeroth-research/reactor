module flags (
  input  logic [7:0] x,
  input  logic [7:0] y,
  output logic zero,
  output logic parity,
  output logic all_ones,
  output logic any_both,
  output logic none,
  output logic same,
  output logic [7:0] blend
);
  logic [7:0] both;
  assign both = (x & y);
  assign any_both = (|both);
  assign zero = (~|x);
  assign parity = (^x);
  assign all_ones = (&x);
  assign none = ((!x) && (!y));
  assign same = (&(x ~^ y));
  assign blend = ((x & (~y)) | (((~x) & y) ^ 8'd15));
endmodule

