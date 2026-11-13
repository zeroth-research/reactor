module stepper (
  input  logic clk,
  input  logic en,
  output logic [5:0] q,
  output logic [3:0] id
);
  initial q = 6'd60;
  always_ff @(posedge clk) if (en) q <= (q + 6'd1); else begin end
  assign id = 4'sd5;
endmodule

module stepper_2 (
  input  logic clk,
  input  logic en,
  output logic [3:0] q,
  output logic [3:0] id
);
  initial q = 4'd3;
  always_ff @(posedge clk) if (en) q <= (q + 4'd1); else begin end
  assign id = 4'sd9;
endmodule

module initinst (
  input  logic clk,
  input  logic en,
  output logic [5:0] a,
  output logic [3:0] b,
  output logic [3:0] ids
);
  logic [3:0] ia;
  logic [3:0] ib;
  stepper u (.clk(clk), .en(en), .q(a), .id(ia));
  stepper_2 v (.clk(clk), .en((!en)), .q(b), .id(ib));
  assign ids = (ia ^ ib);
endmodule

