// A counter with an active-high asynchronous reset to 5, next to a block
// without reset that samples it: `last` is a plain register.
module arcount (
  input  logic       clk, rst, en,
  output logic [7:0] q, last
);
  always_ff @(posedge clk or posedge rst)
    if (rst) q <= 8'd5;
    else if (en) q <= q + 8'd1;
  always_ff @(posedge clk) last <= q;
endmodule
