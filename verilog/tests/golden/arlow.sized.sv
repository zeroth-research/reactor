module arlow (
  input  logic clk,
  input  logic rst_n,
  input  logic [3:0] d,
  output logic [3:0] a,
  output logic [3:0] b,
  output logic [1:0] s,
  output logic [3:0] n
);
  always_ff @(posedge clk or negedge rst_n) if ((!rst_n)) begin a <= 4'd0; s <= {1'd0, 1'd1}; end else begin a <= d; b <= a; s <= {s[0], s[1]}; end
  always_ff @(negedge clk or negedge rst_n) if ((~rst_n)) n <= 4'd15; else n <= (n - 4'd1);
endmodule

