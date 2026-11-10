module negdiv (
  input  logic clk,
  input  logic [3:0] d,
  output logic slow,
  output logic [3:0] p,
  output logic [3:0] n
);
  always_ff @(negedge clk) slow <= (!slow);
  always_ff @(negedge slow) n <= d;
  always_ff @(posedge clk) p <= (p ^ d);
endmodule

