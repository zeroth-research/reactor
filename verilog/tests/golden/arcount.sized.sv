module arcount (
  input  logic clk,
  input  logic rst,
  input  logic en,
  output logic [7:0] q,
  output logic [7:0] last
);
  always_ff @(posedge clk or posedge rst) if (rst) q <= 8'd5; else if (en) q <= (q + 8'd1); else begin end
  always_ff @(posedge clk) last <= q;
endmodule

