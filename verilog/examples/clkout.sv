// A divider whose output `s` also clocks a counter inside it; the parent
// uses `s` as a clock too (a derived clock leaving an instance).
module clkout (
  input  logic       clk,
  input  logic [3:0] a,
  output logic       s,
  output logic [1:0] c,
  output logic [3:0] q
);
  div d (.clk, .s, .c);
  always_ff @(negedge s) q <= q + a;
endmodule

module div (
  input  logic       clk,
  output logic       s,
  output logic [1:0] c
);
  always_ff @(posedge clk) s <= !s;
  always_ff @(posedge s) c <= c + 2'd1;
endmodule
