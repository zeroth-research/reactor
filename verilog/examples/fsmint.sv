// A state machine whose states are untyped local parameters with `int`
// values. The `case` selector and its labels are sized together, so the
// 2-bit state is compared at 32 bits; assigning a state truncates it to
// 2 bits. In the second `case`, the 3-bit `k` is extended to the 4 bits of
// the label `4'd9`, which therefore never matches.
module fsmint (
  input  logic       clk, go,
  input  logic [2:0] k,
  output logic [1:0] st,
  output logic       busy, hit
);
  localparam IDLE = 0, RUN = 1, DONE = 2;
  always_ff @(posedge clk)
    case (st)
      IDLE: if (go) st <= RUN;
      RUN: st <= DONE;
      DONE, 3: st <= IDLE;
    endcase
  assign busy = st == RUN;
  always_comb
    case (k)
      4'd9: hit = 1;
      1, 2: hit = 1'b1;
      default: hit = 0;
    endcase
endmodule
