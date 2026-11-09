// Resettable counters in instances: `u` is reset by the input `rst`
// through its port, `v` by a register `go_n` of the parent passed down
// through a copy `vr`, with its own reset value.
module arinst (
  input  logic       clk, rst, go,
  output logic [3:0] p,
  output logic [5:0] q
);
  logic go_n, vr;
  always_ff @(posedge clk) go_n <= !go;
  assign vr = go_n;
  rcount #(.W(4)) u (.clk, .rst, .q(p));
  rcount #(.W(6), .INIT(6'd9)) v (.clk, .rst(vr), .q);
endmodule

module rcount #(
  parameter int W = 4,
  parameter logic [W-1:0] INIT = W'(0)
) (
  input  logic         clk, rst,
  output logic [W-1:0] q
);
  localparam logic [W-1:0] ONE = W'(1);
  always_ff @(posedge clk or posedge rst)
    if (rst) q <= INIT;
    else q <= q + ONE;
endmodule
