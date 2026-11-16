// Verilog-style registers: `reg` and plain `always` blocks with edge
// events, which mean the same as `logic` and `always_ff`. A counter with an
// active-low asynchronous reset, a block on the falling edge that writes a
// register with a blocking `=` (accepted where it means `<=`: nothing reads
// it later in the block) and a memory, and a flag with an initial value.
module plainff (
  input            clk, rst_n, en,
  input      [3:0] d,
  output reg [3:0] cnt,
  output reg [3:0] low,
  output reg       seen = 1'b0
);
  reg [3:0] hist [0:1];
  always @(posedge clk or negedge rst_n)
    if (!rst_n) cnt <= 4'd0;
    else if (en) cnt <= cnt + 4'd1;
  always @ (negedge clk) begin
    low = d;
    hist[d[0]] <= cnt;
  end
  always @(posedge clk) seen <= seen | (hist[0] == hist[1]);
endmodule
