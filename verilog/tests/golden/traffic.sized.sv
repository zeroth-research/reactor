module traffic (
  input  logic clk,
  input  logic rst,
  output logic [1:0] light,
  output logic walk
);
  logic [1:0] state;
  logic [2:0] timer;
  logic done;
  always_comb case (state) 2'd0: done = (timer == 3'd3); 2'd1: done = (timer == 3'd0); default: done = (timer == 3'd2); endcase
  always_ff @(posedge clk) if (rst) begin state <= 2'd0; timer <= 3'd0; end else if (done) begin timer <= 3'd0; case (state) 2'd0: state <= 2'd1; 2'd1: state <= 2'd2; default: state <= 2'd0; endcase end else timer <= (timer + 3'd1);
  assign light = state;
  assign walk = (state == 2'd2);
endmodule

