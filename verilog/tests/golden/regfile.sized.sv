module regfile (
  input  logic clk,
  input  logic we,
  input  logic [2:0] wa,
  input  logic [1:0] ra,
  input  logic [1:0] rb,
  input  logic signed [7:0] d,
  output logic signed [7:0] a,
  output logic signed [8:0] sum,
  output logic [3:0] lo,
  output logic b
);
  logic signed [7:0] r [0:7];
  logic [7:0] s [0:3];
  always_ff @(posedge clk) begin if (we) r[wa] <= d; else r[rb] <= (-d); r[0] <= 8'sd0; end
  always_ff @(negedge clk) begin s[ra] <= d; if (we) s[rb][7:4] <= 4'd10; else begin end end
  assign a = r[{1'd0, ra}];
  assign sum = (9'(signed'(r[wa])) + 9'(signed'(r[3])));
  assign lo = s[ra][3:0];
  assign b = s[rb][ra];
endmodule

