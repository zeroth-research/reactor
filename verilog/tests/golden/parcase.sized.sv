module parcase (
  input  logic clk,
  input  logic go,
  input  logic [7:0] d,
  output logic [1:0] mode,
  output logic [7:0] y
);
  logic [2:0] st;
  logic [7:0] acc;
  initial st = 3'd0;
  always_ff @(posedge clk) begin (* parallel_case *) case (32'(unsigned'(st))) 32'd0: if (go) st <= 3'sd1; else begin end 32'd1: begin acc <= d; st <= 3'sd2; end 32'd2: begin acc <= (acc + d); if ((!go)) st <= 3'sd3; else begin end end 32'd3, 32'd4: st <= 3'((go ? 32'sd2 : 32'sd4)); default: st <= 3'sd0; endcase end
  always_comb begin (* parallel_case *) case (32'(unsigned'(st))) 32'd0: begin mode = 2'd0; y = 8'd0; end 32'd1, 32'd2: begin mode = 2'd1; y = acc; end 32'd3: begin mode = 2'd2; y = acc; end default: begin mode = 2'd3; y = (~acc); end endcase end
endmodule

