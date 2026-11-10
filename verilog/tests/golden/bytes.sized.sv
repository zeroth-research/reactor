module bytes (
  input  logic [15:0] w,
  input  logic [7:0] b,
  input  logic signed [7:0] s,
  output logic [15:0] swapped,
  output logic [15:0] ext_b,
  output logic signed [15:0] ext_s,
  output logic same
);
  assign swapped = {w[7:0], w[15:8]};
  assign ext_b = {{8{b[7]}}, b};
  assign ext_s = 16'(16'(signed'(s)));
  assign same = (signed'(ext_b) == 16'(16'(signed'(signed'(b)))));
endmodule

