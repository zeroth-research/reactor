// Writes with a variable index in `always_comb`: each follows an
// assignment of the whole signal (otherwise the other bits would keep
// their old values, a latch).
module onehot (
    input  logic [ 2:0] sel,
    input  logic [ 3:0] n,
    input  logic [15:0] a,
    output logic [ 7:0] y,
    output logic [15:0] z
);
  always_comb begin
    y = '0;
    y[sel] = 1'b1;
  end

  // bits above bit 15 (index 16 or more) are not written
  always_comb begin
    z = a;
    if (sel[0]) z[{1'b0, n} + 5'd3 -: 4] = 4'b1010;
    z[z[3:0]] = 1'b0;
  end
endmodule
