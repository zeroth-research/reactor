module fsmint (
  input  logic clk,
  input  logic go,
  input  logic [2:0] k,
  output logic [1:0] st,
  output logic busy,
  output logic hit
);
  always_ff @(posedge clk) case (32'(unsigned'(st))) 32'd0: if (go) st <= 2'sd1; else begin end 32'd1: st <= 2'sd2; 32'd2, 32'd3: st <= 2'sd0; endcase
  assign busy = (32'(unsigned'(st)) == 32'd1);
  always_comb case (32'(unsigned'(k))) 32'd9: hit = 1'sd1; 32'd1, 32'd2: hit = 1'd1; default: hit = 1'sd0; endcase
endmodule

