module halves (
  input  logic clk,
  input  logic en,
  output logic [3:0] p,
  output logic [3:0] n,
  output logic [3:0] h,
  output logic [4:0] total
);
  always_ff @(posedge clk) if (en) p <= (p + 4'd1); else begin end
  always_ff @(negedge clk) if (en) n <= (n + 4'd1); else begin end
  always_ff @(negedge clk) h <= p;
  assign total = ({1'd0, p} + {1'd0, n});
endmodule

