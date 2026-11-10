module addc (
  input  logic [7:0] a,
  input  logic [7:0] b,
  input  logic cin,
  output logic [7:0] s,
  output logic cout
);
  assign {cout, s} = ((9'(9'(unsigned'(a))) + 9'(9'(unsigned'(b)))) + 9'(9'(unsigned'(cin))));
endmodule

