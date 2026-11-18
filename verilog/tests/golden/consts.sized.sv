module consts (
  input  logic [7:0] a,
  input  logic [39:0] b,
  output logic [15:0] w,
  output logic [7:0] l,
  output logic [1:0] k,
  output logic [4:0] c,
  output logic [7:0] h,
  output logic [63:0] big,
  output logic [63:0] pow,
  output logic [63:0] least,
  output logic [39:0] negpow
);
  assign w = (16'(unsigned'(a)) + 16'd146);
  assign l = (~a);
  assign k = a[1:0];
  assign c = a[4:0];
  assign h = (8'd225 + a);
  assign big = 64'sd4294967296;
  assign pow = 64'sd2147483648;
  assign least = 64'sd18446744071562067968;
  assign negpow = (b + 40'd1099511627749);
endmodule

