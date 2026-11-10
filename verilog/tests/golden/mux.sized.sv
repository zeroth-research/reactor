module mux (
  input  logic [1:0] sel,
  input  logic [7:0] a,
  input  logic [7:0] b,
  input  logic [7:0] c,
  input  logic [7:0] d,
  input  logic s,
  output logic [7:0] y4,
  output logic [7:0] y2
);
  assign y4 = ((sel == 2'd0) ? a : ((sel == 2'd1) ? b : ((sel == 2'd2) ? c : d)));
  assign y2 = (s ? a : b);
endmodule

