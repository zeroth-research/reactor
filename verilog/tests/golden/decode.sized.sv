module decode (
  input  logic clk,
  input  logic [2:0] cmd,
  input  logic [7:0] d,
  output logic [7:0] q
);
  always_ff @(posedge clk) case (cmd) 3'd0: q <= 8'd0; 3'd1, 3'd2: q <= d; 3'd3: q <= (q + 8'd1); 3'd4, 3'd0: q <= (q - 8'd1); endcase
endmodule

