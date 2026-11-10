module mac (
  input  logic clk,
  input  logic clear,
  input  logic signed [7:0] a,
  input  logic signed [7:0] b,
  output logic signed [15:0] acc,
  output logic [7:0] hi
);
  always_ff @(posedge clk) begin if (clear) acc <= 16'sd0; else acc <= (acc + (16'(16'(signed'(a))) * 16'(16'(signed'(b))))); end
  assign hi = acc[15:8];
endmodule

