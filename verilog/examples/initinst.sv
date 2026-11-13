// Initial values from parameters: two instances of a counter that starts
// at `INIT`, and a constant output `id` (initialized, never written) of
// the child, read by the parent.
module stepper #(parameter int W = 4, parameter logic [W-1:0] INIT = '0, parameter int ID = 0) (
  input  logic         clk,
  input  logic         en,
  output logic [W-1:0] q = INIT,
  output logic [3:0]   id = ID
);
  always_ff @(posedge clk) if (en) q <= q + 1'b1;
endmodule

module initinst (
  input  logic       clk,
  input  logic       en,
  output logic [5:0] a,
  output logic [3:0] b,
  output logic [3:0] ids
);
  logic [3:0] ia, ib;
  stepper #(.W(6), .INIT(6'd60), .ID(5)) u (.clk(clk), .en(en), .q(a), .id(ia));
  stepper #(.INIT(4'd3), .ID(9)) v (.clk(clk), .en(!en), .q(b), .id(ib));
  assign ids = ia ^ ib;
endmodule
