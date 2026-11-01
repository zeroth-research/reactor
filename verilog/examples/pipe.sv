// A two-stage pipeline of parameterized registers with enable. Stage 1
// is 8 bits wide; stage 2 keeps the low nibble of its sum with the input.
// The clock is passed down to both instances. The second instance's
// `changed` output is left unconnected.
module stage #(parameter int W = 8) (
  input  logic         clk, en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q,
  output logic         changed
);
  always_ff @(posedge clk)
    if (en) q <= d;
  assign changed = q != d;
endmodule

module pipe (
  input  logic       clk, en,
  input  logic [7:0] x,
  output logic [7:0] y1,
  output logic [3:0] y2,
  output logic       c1
);
  stage s1 (.clk, .en, .d(x), .q(y1), .changed(c1));
  stage #(.W(4)) s2 (.clk, .en(en & c1), .d(y1[3:0] + x[3:0]), .q(y2), .changed());
endmodule
