module arsync (
  input  logic clk,
  input  logic rst,
  output logic [3:0] q
);
  logic r1;
  logic r2;
  always_ff @(posedge clk or posedge rst) if (rst) begin r1 <= 1'd1; r2 <= 1'd1; end else begin r1 <= 1'd0; r2 <= r1; end
  always_ff @(posedge clk or posedge r2) if (r2) q <= 4'd0; else q <= (q + 4'd1);
endmodule

