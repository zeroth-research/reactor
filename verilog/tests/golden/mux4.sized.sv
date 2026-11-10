module mux4 (
  input  logic [1:0] sel,
  input  logic [2:0] k,
  input  logic [3:0] d,
  input  logic [15:0] w,
  output logic y,
  output logic y1,
  output logic [3:0] win,
  output logic [3:0] nib,
  output logic [3:0] top
);
  assign y = d[sel];
  assign y1 = d[sel -: 1];
  assign win = w[k +: 4];
  assign nib = 4'((w >> {sel, 2'd0}));
  assign top = w[15 -: 4];
endmodule

