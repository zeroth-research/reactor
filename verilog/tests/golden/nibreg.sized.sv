module nibreg (
  input  logic clk,
  input  logic [1:0] we,
  input  logic [3:0] d,
  output logic [7:0] r,
  output logic [3:0] flags
);
  always_ff @(posedge clk) begin if (we[0]) r[3:0] <= d; else begin end if (we[1]) r[7:4] <= d; else begin end end
  always_comb begin flags[0] = (r == 8'd0); flags[1] = r[7]; flags[2] = (^r); flags[3] = (|flags[2:0]); end
endmodule

