module ffblock (
  input  logic clk,
  input  logic en,
  input  logic [1:0] s,
  input  logic [7:0] d,
  output logic [7:0] cnt,
  output logic hit,
  output logic [7:0] w,
  output logic [3:0] m
);
  always_ff @(posedge clk) begin if ((cnt == 8'd4)) hit <= 1'd1; else hit <= 1'd0; if (en) cnt = (cnt + 8'd1); else begin end end
  always_ff @(negedge clk) begin w[3:0] = d[3:0]; case (s) 2'd0: w[7:4] = (d[7:4] ^ w[7:4]); 2'd1: w[7:4] = (w[7:4] + 4'd1); default: begin end endcase m[s] = d[0]; end
endmodule

