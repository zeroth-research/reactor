module xstore (
  input  logic clk,
  input  logic [1:0] mode,
  input  logic slow,
  input  logic fast,
  input  logic [3:0] d,
  output logic led,
  output logic [3:0] q,
  output logic [4:0] wide
);
  logic blink;
  logic [1:0] state;
  logic [1:0] next;
  always_comb begin blink = 1'bx; case (mode) 2'd0: blink = 1'd0; 2'd1: blink = slow; 2'd2: blink = fast; default: blink = 1'd1; endcase end
  assign led = (!blink);
  always_comb begin next = 'x; {wide[4], wide[3:0]} = 5'bx; if ((state == 2'd3)) next = 2'd0; else next = (state + 2'd1); wide = {(next == 2'd0), d}; end
  always_ff @(posedge clk) begin state <= next; q <= 4'bx; if (d[0]) q <= d; else q <= (q + 4'd1); end
endmodule

