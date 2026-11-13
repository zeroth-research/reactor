// Assignments of x that every path overwrites (dead stores): the LED mode
// mux of the lighthouse top level, a next-state block, a register.
module xstore (
    input  logic       clk,
    input  logic [1:0] mode,
    input  logic       slow, fast,
    input  logic [3:0] d,
    output logic       led,
    output logic [3:0] q,
    output logic [4:0] wide
);
  logic blink;
  always_comb begin
    blink = 1'bx;
    case (mode)
      2'b00: blink = 1'b0;
      2'b01: blink = slow;
      2'b10: blink = fast;
      default: blink = 1'b1;
    endcase
  end
  assign led = !blink;

  logic [1:0] state, next;
  always_comb begin
    next = 'x;
    {wide[4], wide[3:0]} = 5'hx;
    if (state == 2'd3) next = 2'd0;
    else next = state + 2'd1;
    wide = {next == 2'd0, d};
  end

  always_ff @(posedge clk) begin
    state <= next;
    q <= 4'bxxxx;
    if (d[0]) q <= d;
    else q <= q + 4'd1;
  end
endmodule
