// An 8-bit counter with enable, and a flag computed from the register.
module counter (
  input  logic       clk, en,
  output logic [7:0] q,
  output logic       full
);
  always_ff @(posedge clk)
    if (en) q <= q + 8'd1;
  assign full = &q;
endmodule
