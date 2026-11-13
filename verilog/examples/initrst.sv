// Initial values with an asynchronous reset (active in the first round,
// the reset value wins), with a synchronous reset, on a falling edge, and
// on a derived clock: `slow` starts at 0, so its first rising edge comes
// with the second rising edge of `clk`.
module initrst (
  input  logic       clk,
  input  logic       rst,
  input  logic [7:0] d,
  output logic [7:0] q, n, s,
  output logic       slow
);
  logic [7:0] r = 8'd17;
  initial slow = 1'b0;
  initial n = 8'h80;
  initial s = 8'd4;
  always_ff @(posedge clk or posedge rst)
    if (rst) q <= 8'd1;
    else q <= q + r;
  always_ff @(posedge clk) begin
    if (rst) r <= 8'd0;
    else r <= r + d;
    slow <= ~slow;
  end
  always_ff @(negedge clk) n <= {n[0], n[7:1]};
  always_ff @(posedge slow) s <= s + 8'd2;
endmodule
