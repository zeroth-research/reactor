// Active-low asynchronous resets, written `!rst_n` and `~rst_n`, on both
// edges of the clock. The reset branch of the first block leaves `b`
// alone: `b` is not reset, it keeps its value while the reset is active.
module arlow (
  input  logic       clk, rst_n,
  input  logic [3:0] d,
  output logic [3:0] a, b,
  output logic [1:0] s,
  output logic [3:0] n
);
  always_ff @(posedge clk, negedge rst_n)
    if (!rst_n) begin
      a <= 4'd0;
      s <= {1'b0, 1'b1};
    end else begin
      a <= d;
      b <= a;
      s <= {s[0], s[1]};
    end
  always_ff @(negedge rst_n or negedge clk)
    if (~rst_n) n <= 4'hf;
    else n <= n - 4'd1;
endmodule
