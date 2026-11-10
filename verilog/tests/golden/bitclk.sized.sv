module bitclk (
  input  logic clk,
  input  logic [7:0] a,
  output logic [2:0] cnt,
  output logic [7:0] w
);
  logic c;
  always_ff @(posedge clk) cnt <= (cnt + 3'd1);
  assign c = cnt[2];
  always_ff @(posedge c) w <= (w + a);
endmodule

