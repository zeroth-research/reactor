// A traffic light with named states (local parameters): green for 4
// cycles, yellow for 1, red for 3, then green again; `walk` is set while
// it is red. A timer counts the cycles spent in the current state.
module traffic (
  input  logic       clk, rst,
  output logic [1:0] light,
  output logic       walk
);
  localparam GREEN = 2'd0, YELLOW = 2'd1, RED = 2'd2;
  localparam logic [2:0] TGREEN = 3'd3, TYELLOW = 3'd0, TRED = 3'd2;
  logic [1:0] state;
  logic [2:0] timer;
  logic done;

  always_comb
    case (state)
      GREEN:   done = timer == TGREEN;
      YELLOW:  done = timer == TYELLOW;
      default: done = timer == TRED;
    endcase

  always_ff @(posedge clk)
    if (rst) begin
      state <= GREEN;
      timer <= 3'd0;
    end else if (done) begin
      timer <= 3'd0;
      case (state)
        GREEN:   state <= YELLOW;
        YELLOW:  state <= RED;
        default: state <= GREEN;
      endcase
    end else
      timer <= timer + 3'd1;

  assign light = state;
  assign walk = state == RED;
endmodule
