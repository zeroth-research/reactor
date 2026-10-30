// An accumulator: the register `acc` feeds combinational logic that feeds
// it back. The loop passes through a register, so it is no combinational
// loop. `overflow` is a sticky flag in a second block.
module accum (
  input  logic       clk, clear,
  input  logic [7:0] x,
  output logic [7:0] sum,
  output logic       overflow
);
  logic [7:0] acc, next;
  logic       carry;
  assign next = acc + x;
  assign carry = next < acc;
  always_ff @(posedge clk)
    if (clear) acc <= 8'd0;
    else acc <= next;
  always_ff @(posedge clk)
    overflow <= clear ? 1'b0 : carry || overflow;
  assign sum = acc;
endmodule
