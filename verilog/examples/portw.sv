// Input ports of other widths. A connection to an input acts as an
// assignment to the port: a narrower actual is extended by its own sign
// (`n4` with zeros, `s4` with its sign bit), a wider one truncated, an
// operator expression evaluated at the width of the port, and the `int`
// constant 5 truncated to 8 bits.
module sink (
  input  logic [7:0] x,
  output logic [7:0] y
);
  assign y = x;
endmodule

module portw (
  input  logic        [3:0]  n4,
  input  logic signed [3:0]  s4,
  input  logic        [15:0] w16,
  input  logic        [7:0]  a, b,
  output logic        [7:0]  y1, y2, y3, y4, y5
);
  sink u1 (.x(n4), .y(y1));
  sink u2 (.x(s4), .y(y2));
  sink u3 (.x(w16), .y(y3));
  sink u4 (.x(a + b), .y(y4));
  sink u5 (.x(5), .y(y5));
endmodule
