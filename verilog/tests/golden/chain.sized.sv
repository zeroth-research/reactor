module chain (
  input  logic [7:0] a,
  output logic [7:0] y
);
  logic [7:0] x;
  logic [7:0] z;
  always_comb y = (z ^ 8'd15);
  always_comb z = (x + x);
  always_comb x = (a - 8'd3);
endmodule

