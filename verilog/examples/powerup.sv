// Registers with initial values (their power-on state on an FPGA) next to
// registers without: a declaration initializer, an `initial` block, an
// initial value of an output, an x initial value (the same as none), and a
// constant (initialized, never written) read with a variable bit index.
// `wide` starts at the signed 4-bit -3 extended by its sign to 12 bits (12'hffd).
module powerup (
  input  logic       clk,
  input  logic       en,
  input  logic [3:0] i,
  input  logic [7:0] d,
  output logic [3:0] st = 4'd9,
  output logic [7:0] y,
  output logic       b,
  output logic [3:0] w
);
  logic [7:0] cnt = 8'd250;
  logic [7:0] last;
  logic [7:0] seen;
  logic       armed;
  logic [11:0] wide = 4'sb1101;
  logic [15:0] pattern = {15'h392b, 1'b0};
  wire  [7:0] next = cnt + 8'd3;
  initial begin
    seen = 'x;
    armed = 1;
  end
  always_ff @(posedge clk) begin
    if (en) cnt <= next;
    last <= d;
    seen <= seen | d;
    if (armed && cnt == 8'd255) armed <= 1'b0;
    st <= st - 4'd1;
    wide <= wide + d;
  end
  assign y = cnt ^ last ^ seen;
  assign b = pattern[i] & armed;
  assign w = wide[11:8];
endmodule
