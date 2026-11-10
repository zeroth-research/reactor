module divider (
  input  logic clk,
  input  logic en,
  output logic slow,
  output logic [7:0] q,
  output logic [7:0] r
);
  always_ff @(posedge clk) begin slow <= (!slow); if (en) q <= (q + 8'd1); else begin end end
  always_ff @(posedge slow) r <= q;
endmodule

