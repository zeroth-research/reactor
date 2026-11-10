module ctxwidth (
  input  logic [7:0] a,
  input  logic [7:0] b,
  input  logic c,
  output logic [15:0] sum16,
  output logic [15:0] half16,
  output logic [15:0] cat,
  output logic [15:0] inv,
  output logic [7:0] half8,
  output logic [3:0] sum4,
  output logic red,
  output logic land,
  output logic carry
);
  assign sum16 = (16'(unsigned'(a)) + 16'(unsigned'(b)));
  assign sum4 = 4'((a + b));
  assign half16 = ((16'(unsigned'(a)) + 16'(unsigned'(b))) >> 32'sd1);
  assign half8 = ((a + b) >> 32'sd1);
  assign red = (&(a + b));
  assign land = ((a + b) && c);
  assign cat = 16'(unsigned'({(a + b)}));
  assign carry = ((9'(unsigned'(a)) + 9'(unsigned'(b))) > 9'd255);
  assign inv = (~16'(unsigned'(a)));
endmodule

