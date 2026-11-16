module oorread (
  input  logic clk,
  input  logic [5:0] c,
  input  logic [3:0] i,
  input  logic [15:0] v,
  output logic b,
  output logic [3:0] up,
  output logic [3:0] dn,
  output logic [2:0] e,
  output logic q
);
  logic [15:0] k;
  logic [5:0] mem [0:3];
  assign b = v[(32'd15 - 32'(unsigned'(c)))];
  assign up = v[({1'd0, i} + 5'd10) +: 4];
  assign dn = v[i -: 4];
  assign e = mem[c[1:0]][c[5:2] -: 3];
  always_ff @(posedge clk) begin mem[i[1:0]] <= v[5:0]; q <= k[(32'd15 - 32'(unsigned'(c)))]; end
  assign k = {15'd14635, 1'd0};
endmodule

