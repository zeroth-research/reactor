module stage (
  input  logic clk,
  input  logic en,
  input  logic [7:0] d,
  output logic [7:0] q,
  output logic changed
);
  always_ff @(posedge clk) if (en) q <= d; else begin end
  assign changed = (q != d);
endmodule

module stage_2 (
  input  logic clk,
  input  logic en,
  input  logic [3:0] d,
  output logic [3:0] q,
  output logic changed
);
  always_ff @(posedge clk) if (en) q <= d; else begin end
  assign changed = (q != d);
endmodule

module pipe (
  input  logic clk,
  input  logic en,
  input  logic [7:0] x,
  output logic [7:0] y1,
  output logic [3:0] y2,
  output logic c1
);
  stage s1 (.clk(clk), .en(en), .d(x), .q(y1), .changed(c1));
  stage_2 s2 (.clk(clk), .en((en & c1)), .d((y1[3:0] + x[3:0])), .q(y2), .changed());
endmodule

