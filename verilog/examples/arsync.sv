// A reset synchronizer: the input reset `rst` is asserted asynchronously
// and released through two registers, so the reset `r2` of the counter is
// a register that is released at a clock edge.
module arsync (
  input  logic       clk, rst,
  output logic [3:0] q
);
  logic r1, r2;
  always_ff @(posedge clk or posedge rst)
    if (rst) begin
      r1 <= 1'b1;
      r2 <= 1'b1;
    end else begin
      r1 <= 1'b0;
      r2 <= r1;
    end
  always_ff @(posedge clk or posedge r2)
    if (r2) q <= 4'd0;
    else q <= q + 4'd1;
endmodule
