module onehot (
  input  logic [2:0] sel,
  input  logic [3:0] n,
  input  logic [15:0] a,
  output logic [7:0] y,
  output logic [15:0] z
);
  always_comb begin y = 8'd0; y[sel] = 1'd1; end
  always_comb begin z = a; if (sel[0]) z[({1'd0, n} + 5'd3) -: 4] = 4'd10; else begin end z[z[3:0]] = 1'd0; end
endmodule

