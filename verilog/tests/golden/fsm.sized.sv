module fsm (
  input  logic clk,
  input  logic rst,
  input  logic din,
  output logic found
);
  logic [1:0] state;
  logic [1:0] nxt;
  always_ff @(posedge clk) if (rst) state <= 2'd0; else state <= nxt;
  always_comb case (state) 2'd0: nxt = (din ? 2'd1 : 2'd0); 2'd1: nxt = (din ? 2'd1 : 2'd2); 2'd2: nxt = (din ? 2'd3 : 2'd0); default: nxt = (din ? 2'd1 : 2'd2); endcase
  assign found = (state == 2'd3);
endmodule

