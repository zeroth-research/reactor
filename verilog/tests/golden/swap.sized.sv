module swap (
  input  logic clk,
  input  logic load,
  input  logic sw,
  input  logic [3:0] a,
  input  logic [3:0] b,
  output logic [3:0] x,
  output logic [3:0] y
);
  always_ff @(posedge clk) if (load) begin x <= a; y <= b; end else if (sw) begin x <= y; y <= x; end else begin end
endmodule

