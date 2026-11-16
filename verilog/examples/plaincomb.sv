// Verilog-style combinational blocks: `always @*`, `always @(*)` and event
// lists that name every signal the block reads (`@(state or go)`,
// `@(a, b)`), which mean the same as `always_comb`. An FSM with its next
// state computed in a level-list block, a mux, and a block that reads its
// own target after assigning it (the target need not be in the list).
module plaincomb (
  input            clk, go,
  input      [7:0] a, b,
  input      [1:0] sel,
  output reg [7:0] y,
  output reg [7:0] m,
  output reg [1:0] state
);
  localparam IDLE = 2'd0;
  localparam RUN = 2'd1;
  localparam STOP = 2'd2;
  reg [1:0] next;
  reg [7:0] t;
  always @(posedge clk) state <= next;
  always @(state or go)
    case (state)
      IDLE: next = go ? RUN : IDLE;
      RUN: next = go ? RUN : STOP;
      default: next = IDLE;
    endcase
  always @(*)
    case (sel)
      2'd0: m = a;
      2'd1: m = b;
      default: m = a & b;
    endcase
  always @(a, b) begin
    t = a + b;
    t = t ^ {t[3:0], t[7:4]};
  end
  always @* y = (state == RUN) ? t : m;
endmodule
