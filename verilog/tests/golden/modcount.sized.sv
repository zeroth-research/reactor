module modcount (
  input  logic clk,
  input  logic en,
  output logic [3:0] q,
  output logic wrap
);
  assign wrap = (q == 4'd9);
  always_ff @(posedge clk) if (en) q <= (wrap ? 4'd0 : (q + 4'd1)); else begin end
endmodule

