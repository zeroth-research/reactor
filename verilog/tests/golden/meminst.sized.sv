module mem (
  input  logic clk,
  input  logic we,
  input  logic [2:0] wa,
  input  logic [2:0] ra,
  input  logic [5:0] d,
  output logic [5:0] q
);
  logic [5:0] m [0:7];
  always_ff @(posedge clk) if (we) m[wa] <= d; else begin end
  assign q = m[ra];
endmodule

module mem_2 (
  input  logic clk,
  input  logic we,
  input  logic wa,
  input  logic ra,
  input  logic [3:0] d,
  output logic [3:0] q
);
  logic [3:0] m [0:1];
  always_ff @(posedge clk) if (we) m[wa] <= d; else begin end
  assign q = m[ra];
endmodule

module meminst (
  input  logic clk,
  input  logic we,
  input  logic [2:0] wa,
  input  logic [2:0] ra,
  input  logic [5:0] d,
  output logic [5:0] q8,
  output logic [3:0] q2
);
  mem deep (.clk(clk), .we(we), .wa(wa), .ra(ra), .d(d), .q(q8));
  mem_2 shallow (.clk(clk), .we(we), .wa(wa[0]), .ra(ra[0]), .d(d[3:0]), .q(q2));
endmodule

