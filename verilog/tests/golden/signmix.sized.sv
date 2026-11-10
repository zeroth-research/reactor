module signmix (
  input  logic signed [7:0] s,
  input  logic signed [3:0] t,
  input  logic [7:0] u,
  input  logic c,
  output logic [15:0] su,
  output logic [15:0] s0,
  output logic [15:0] sd0,
  output logic [15:0] sg,
  output logic [15:0] ash,
  output logic [15:0] cash,
  output logic [15:0] mash,
  output logic [15:0] div,
  output logic [15:0] cs,
  output logic [15:0] cw,
  output logic [7:0] cc,
  output logic lt,
  output logic neg,
  output logic m
);
  assign su = (16'(unsigned'(s)) + 16'(unsigned'(u)));
  assign s0 = 16'((32'(signed'(s)) + 32'sd0));
  assign sd0 = 16'((32'(unsigned'(s)) + 32'd0));
  assign sg = 16'(signed'(signed'((u + u))));
  assign lt = (unsigned'(s) < u);
  assign neg = (32'(signed'(s)) < 32'sd0);
  assign ash = (16'(signed'(s)) >>> 32'sd1);
  assign cash = (16'(unsigned'({s})) >>> 32'sd1);
  assign mash = ((16'(unsigned'(s)) >>> 32'sd1) + 16'(unsigned'(u)));
  assign div = 16'((32'(signed'(s)) / 32'sd2));
  assign cs = (c ? 16'(unsigned'(s)) : 16'(unsigned'(u)));
  assign cw = (c ? 16'(signed'(t)) : 16'(signed'(s)));
  assign cc = (u ? 8'd1 : 8'd2);
  always_comb case (8'(unsigned'(t))) 8'd255: m = 1'd1; default: m = 1'd0; endcase
endmodule

