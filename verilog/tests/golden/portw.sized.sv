module sink (
  input  logic [7:0] x,
  output logic [7:0] y
);
  assign y = x;
endmodule

module portw (
  input  logic [3:0] n4,
  input  logic signed [3:0] s4,
  input  logic [15:0] w16,
  input  logic [7:0] a,
  input  logic [7:0] b,
  output logic [7:0] y1,
  output logic [7:0] y2,
  output logic [7:0] y3,
  output logic [7:0] y4,
  output logic [7:0] y5
);
  sink u1 (.x(8'(unsigned'(n4))), .y(y1));
  sink u2 (.x(8'(signed'(s4))), .y(y2));
  sink u3 (.x(8'(w16)), .y(y3));
  sink u4 (.x((a + b)), .y(y4));
  sink u5 (.x(8'sd5), .y(y5));
endmodule

