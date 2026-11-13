module initrst (
  input  logic clk,
  input  logic rst,
  input  logic [7:0] d,
  output logic [7:0] q,
  output logic [7:0] n,
  output logic [7:0] s,
  output logic slow
);
  logic [7:0] r;
  initial r = 8'd17;
  initial slow = 1'd0;
  initial n = 8'd128;
  initial s = 8'd4;
  always_ff @(posedge clk or posedge rst) if (rst) q <= 8'd1; else q <= (q + r);
  always_ff @(posedge clk) begin if (rst) r <= 8'd0; else r <= (r + d); slow <= (~slow); end
  always_ff @(negedge clk) n <= {n[0], n[7:1]};
  always_ff @(posedge slow) s <= (s + 8'd2);
endmodule

