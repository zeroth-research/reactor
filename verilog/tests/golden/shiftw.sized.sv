module shiftw (
  input  logic [7:0] a,
  input  logic [2:0] n,
  output logic [15:0] sl,
  output logic [15:0] sn,
  output logic [15:0] amt,
  output logic [7:0] one
);
  assign sl = (16'(unsigned'(a)) << 32'sd2);
  assign sn = (16'(unsigned'(a)) << n);
  assign one = 8'((32'sd1 << n));
  assign amt = (16'd1 << (n + 3'd7));
endmodule

