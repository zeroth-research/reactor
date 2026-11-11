module oorwr (
  input  logic clk,
  input  logic [3:0] i,
  input  logic [2:0] e,
  input  logic d,
  output logic [7:0] p,
  output logic [7:0] q,
  output logic [7:0] r,
  output logic f
);
  always_ff @(posedge clk) begin q[i] <= d; p[i +: 3] <= e; r[({1'd0, i} + 5'd2) -: 3] <= e; f <= d; q[(32'(unsigned'(i)) - 32'd1)] <= (~d); if (e[0]) q[i[2:0]] <= e[1]; else begin end end
endmodule

