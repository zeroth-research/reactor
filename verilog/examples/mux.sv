// A 4-to-1 multiplexer from nested conditionals, and a 2-to-1 one.
module mux (
  input  logic [1:0] sel,
  input  logic [7:0] a, b, c, d,
  input  logic       s,
  output logic [7:0] y4,
  output logic [7:0] y2
);
  assign y4 = sel == 2'd0 ? a : sel == 2'd1 ? b : sel == 2'd2 ? c : d;
  assign y2 = s ? a : b;
endmodule
