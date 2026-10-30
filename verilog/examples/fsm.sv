// A Moore machine detecting the bit sequence 1, 0, 1 on `din` (overlapping:
// 1, 0, 1, 0, 1 is found twice). States: 0 = nothing, 1 = seen 1,
// 2 = seen 1 0, 3 = seen 1 0 1. The state register has a synchronous reset;
// the next state is computed by an `always_comb`.
module fsm (
  input  logic clk, rst, din,
  output logic found
);
  logic [1:0] state, nxt;

  always_ff @(posedge clk)
    if (rst) state <= 2'd0;
    else state <= nxt;

  always_comb
    case (state)
      2'd0: nxt = din ? 2'd1 : 2'd0;
      2'd1: nxt = din ? 2'd1 : 2'd2;
      2'd2: nxt = din ? 2'd3 : 2'd0;
      default: nxt = din ? 2'd1 : 2'd2;
    endcase

  assign found = state == 2'd3;
endmodule
