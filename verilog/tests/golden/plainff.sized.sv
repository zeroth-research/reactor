module plainff (
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  input  logic [3:0] d,
  output logic [3:0] cnt,
  output logic [3:0] low,
  output logic seen
);
  logic [3:0] hist [0:1];
  initial seen = 1'd0;
  always_ff @(posedge clk or negedge rst_n) if ((!rst_n)) cnt <= 4'd0; else if (en) cnt <= (cnt + 4'd1); else begin end
  always_ff @(negedge clk) begin low = d; hist[d[0]] <= cnt; end
  always_ff @(posedge clk) seen <= (seen | (hist[0] == hist[1]));
endmodule

