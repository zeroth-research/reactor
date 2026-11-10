module enreg (
  input  logic clk,
  input  logic rst,
  input  logic load,
  input  logic inc,
  input  logic [7:0] d,
  output logic [7:0] q,
  output logic changed
);
  always_ff @(posedge clk) begin changed <= 1'd0; if (rst) begin q <= 8'd0; changed <= (q != 8'd0); end else if (load) begin q <= d; changed <= (q != d); end else if (inc) begin q <= (q + 8'd1); changed <= 1'd1; end else begin end end
endmodule

