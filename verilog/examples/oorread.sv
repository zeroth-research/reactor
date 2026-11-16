// Reads with a variable index that may select bits outside the signal:
// those bits read 0 (the standard's value for 2-state signals; Yosys
// builds the same). Beyond the top, below bit 0 with `-:`, in an element
// of an array, and the lighthouse configurator's shape `k[15 - c]` (a
// 32-bit index, since `15` is an `int`).
module oorread(input logic clk, input logic [5:0] c, input logic [3:0] i,
               input logic [15:0] v, output logic b, output logic [3:0] up,
               output logic [3:0] dn, output logic [2:0] e, output logic q);
  logic [15:0] k = {15'h392b, 1'b0};
  logic [5:0] mem [0:3];

  assign b = v[15 - c];
  assign up = v[{1'b0, i} + 5'd10 +: 4];
  assign dn = v[i -: 4];
  assign e = mem[c[1:0]][c[5:2] -: 3];

  always_ff @(posedge clk) begin
    mem[i[1:0]] <= v[5:0];
    q <= k[15 - c];
  end
endmodule
