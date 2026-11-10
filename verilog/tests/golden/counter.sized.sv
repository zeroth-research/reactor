module counter (
  input  logic clk,
  input  logic en,
  output logic [7:0] q,
  output logic full
);
  always_ff @(posedge clk) if (en) q <= (q + 8'd1); else begin end
  assign full = (&q);
endmodule

