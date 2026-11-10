module satadd (
  input  logic [7:0] a,
  input  logic [7:0] b,
  output logic [7:0] y,
  output logic sat
);
  logic [7:0] sum;
  always_comb begin sum = (a + b); sat = (sum < a); if (sat) sum = 8'd255; else begin end y = sum; end
endmodule

